"""Experimental RGB depth, food masks, plate calibration, and density pipeline."""

import json
import math
import threading
import time
from dataclasses import replace

import numpy as np

from .grounded_vlm_analyzer import GroundedVLMAnalyzer

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"
SAM_MODEL = "facebook/sam-vit-base"
SAM_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
DENSITY_SOURCE = "https://nutrola.app/fr/blog/how-ai-estimates-portion-sizes-from-photos-technical-deep-dive"
DENSITY_G_PER_ML = {"white rice": 0.74, "raw spinach": 0.13, "olive oil": 0.92,
                    "milk": 1.03, "water": 1.0, "peanut butter": 1.09}


def pixel_box(box, width, height):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1000):
        raise ValueError("Boxes must contain four finite coordinates in [0, 1000]")
    x0, y0, x1, y1 = values * np.array([width, height, width, height]) / 1000
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Boxes must have positive width and height")
    return [x0, y0, x1, y1]


def estimate_volumes(depth, masks, plate_box, plate_diameter_cm=26.0, horizontal_fov_degrees=60.0,
                     plate_shape="ellipse"):
    """Integrate height above an exposed plate plane with perspective pixel areas."""
    depth = np.asarray(depth, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    if depth.ndim != 2 or min(depth.shape) < 3 or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Depth must be a positive finite H x W map")
    if masks.ndim != 3 or masks.shape[1:] != depth.shape or not len(masks):
        raise ValueError("Food masks must have shape N x H x W")
    if not math.isfinite(plate_diameter_cm) or plate_diameter_cm <= 0:
        raise ValueError("Plate diameter must be finite and positive")
    if not 10 < horizontal_fov_degrees < 150:
        raise ValueError("Invalid horizontal field of view")
    height, width = depth.shape
    x0, y0, x1, y1 = pixel_box(plate_box, width, height)
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    radius_x, radius_y = (x1 - x0) / 2, (y1 - y0) / 2
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    ellipse_radius = ((xx + 0.5 - center_x) / radius_x) ** 2 + ((yy + 0.5 - center_y) / radius_y) ** 2
    if plate_shape == "ellipse":
        plate = ellipse_radius <= 1
        interior = ellipse_radius < 0.85 ** 2
    elif plate_shape == "rectangle":
        plate = (np.abs(xx + 0.5 - center_x) <= radius_x) & (np.abs(yy + 0.5 - center_y) <= radius_y)
        interior = (np.abs(xx + 0.5 - center_x) < radius_x * 0.85) & (np.abs(yy + 0.5 - center_y) < radius_y * 0.85)
    else:
        raise ValueError("Unsupported plate shape")
    union = masks.any(axis=0)
    exposed = interior & ~union
    if exposed.sum() < max(50, int(plate.sum() * 0.02)):
        raise ValueError("Not enough exposed flat plate to estimate a support plane")
    focal = width / (2 * math.tan(math.radians(horizontal_fov_degrees) / 2))
    rays = np.stack(((xx + 0.5 - width / 2) / focal,
                     (yy + 0.5 - height / 2) / focal, np.ones_like(xx)), axis=-1)
    points = rays * depth[..., None]
    support = points[exposed]
    if len(support) > 4096:
        support = support[np.random.default_rng(42).choice(len(support), 4096, replace=False)]
    for _ in range(4):
        center = support.mean(axis=0)
        _, singular, vectors = np.linalg.svd(support - center, full_matrices=False)
        if singular[1] < singular[0] * 0.02:
            raise ValueError("Exposed plate points do not constrain a plane")
        normal = vectors[-1]
        if normal[2] < 0:
            normal = -normal
        residual = (support - center) @ normal
        median = np.median(residual)
        threshold = max(3 * 1.4826 * np.median(np.abs(residual - median)), 1e-6)
        keep = np.abs(residual - median) <= threshold
        if keep.all() or keep.sum() < 50:
            break
        support = support[keep]
    center = support.mean(axis=0)
    normal = np.linalg.svd(support - center, full_matrices=False)[2][-1]
    if normal[2] < 0:
        normal = -normal
    offset = -float(center @ normal)
    denominators = rays @ normal
    if offset >= 0 or np.any(denominators[plate] <= 0):
        raise ValueError("Estimated support plane is not visible to the camera")
    plane_points = rays * (-offset / np.maximum(denominators, 1e-6))[..., None]
    area = np.linalg.norm(np.cross(np.gradient(plane_points, axis=0),
                                   np.gradient(plane_points, axis=1)), axis=-1)
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    boundary = np.stack(((center_x + radius_x * np.cos(angles) - width / 2) / focal,
                         (center_y + radius_y * np.sin(angles) - height / 2) / focal,
                         np.ones_like(angles)), axis=-1)
    boundary *= (-offset / (boundary @ normal))[:, None]
    if plate_shape == "rectangle":
        diameter = np.linalg.norm(boundary[0] - boundary[64])
    else:
        diameter = np.linalg.norm(boundary[:, None] - boundary[None, :], axis=-1).max()
    scale = plate_diameter_cm / 100 / diameter
    plane_rmse_cm = float(np.sqrt(np.mean(((support - center) @ normal) ** 2)) * scale * 100)
    if plane_rmse_cm > plate_diameter_cm * 0.02:
        raise ValueError(f"Support-plane depth residual is too large ({plane_rmse_cm:.2f}cm)")
    elevations = np.maximum(-(points @ normal + offset), 0)
    column_volume_ml = elevations * area * scale ** 3 * 1e6
    claimed = np.zeros_like(plate)
    volumes = np.zeros(len(masks), dtype=float)
    overlap_pixels = int(np.sum(masks.sum(axis=0) > 1))
    # Small toppings own overlapping pixels; every projected column is counted once.
    for index in np.argsort(masks.sum(axis=(1, 2)), kind="stable"):
        visible = masks[index] & plate & ~claimed
        volumes[index] = float(column_volume_ml[visible].sum())
        claimed |= visible
    return volumes, {"scale_factor": float(scale), "plate_diameter_cm": plate_diameter_cm,
                     "reference_shape": plate_shape,
                     "horizontal_fov_degrees": horizontal_fov_degrees,
                     "exposed_plate_pixels": int(exposed.sum()), "overlap_pixels": overlap_pixels,
                     "support_plane_rmse_cm": plane_rmse_cm}


class RGBGeometryEstimator:
    def __init__(self, device=None):
        self.device = device
        self.lock = threading.Lock()
        self.depth_model = None
        self.sam_model = None

    def _load_models(self):
        if self.depth_model is not None and self.sam_model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation, SamProcessor, SamModel

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL, revision=DEPTH_REVISION)
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, revision=DEPTH_REVISION).to(self.device).eval()
        self.sam_processor = SamProcessor.from_pretrained(SAM_MODEL, revision=SAM_REVISION)
        self.sam_model = SamModel.from_pretrained(SAM_MODEL, revision=SAM_REVISION).to(self.device).eval()

    def volumes(self, image_path, boxes, plate_box, plate_diameter_cm, horizontal_fov_degrees, plate_shape="ellipse"):
        import torch
        from PIL import Image

        with self.lock, torch.inference_mode():
            self._load_models()
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            width, height = image.size
            depth_inputs = self.depth_processor(images=image, return_tensors="pt").to(self.device)
            depth = self.depth_model(**depth_inputs).predicted_depth
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(height, width),
                                                    mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
            sam_inputs = self.sam_processor(image, input_boxes=[[pixel_box(box, width, height) for box in boxes]],
                                            return_tensors="pt").to(self.device)
            outputs = self.sam_model(**sam_inputs, multimask_output=False)
            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu())[0][:, 0].numpy()
        volumes, diagnostics = estimate_volumes(depth, masks, plate_box, plate_diameter_cm, horizontal_fov_degrees,
                                                plate_shape)
        diagnostics.update(depth_model=DEPTH_MODEL, depth_revision=DEPTH_REVISION,
                           segmentation_model=SAM_MODEL, segmentation_revision=SAM_REVISION)
        return volumes, diagnostics


class GeometryVLMAnalyzer(GroundedVLMAnalyzer):
    method_name = "geometry_vlm_db"
    INVENTORY_PROMPT = GroundedVLMAnalyzer.INVENTORY_PROMPT + """
Also include plate_reference: {"surface": "flat_plate", "shape": "ellipse",
"bbox": [x0,y0,x1,y1]} in the outer JSON object. Use shape="rectangle" for
rectangular plates. Use surface="bowl" or "none" if a flat plate is not visible.
The plate bbox must enclose its outer outline, not just the food. Its left and
right edges must both be visible, since plate width sets the scale.
Add bbox and density_g_per_ml to EACH food item. All boxes use coordinates 0..1000
relative to image width/height, in x0,y0,x1,y1 order. Boxes must tightly enclose
each edible component. Density is an estimated BULK density including air gaps,
for the stated preparation, not the density of solid tissue. It is an assumption.
Local SAM and metric-depth models will estimate volumes from these boxes.
Your gram estimates are retained only if geometric estimation is unavailable."""

    def __init__(self, *args, plate_diameter_cm=None, horizontal_fov_degrees=60.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.plate_diameter_cm = plate_diameter_cm
        self.horizontal_fov_degrees = horizontal_fov_degrees
        self.geometry_estimator = RGBGeometryEstimator()

    def analyze(self, image_path):
        started = time.perf_counter()
        result = super().analyze(image_path)
        if result.error:
            return result
        trace = json.loads(result.raw_response)
        try:
            inventory = self._parse_json(trace["inventory_response"], dict)
            predictions = self._parse_json(trace["matching_response"], dict)["items"]
            reference = inventory.get("plate_reference", {})
            if reference.get("surface") != "flat_plate":
                raise ValueError("No visible flat plate; bowls and missing references need separate geometry")
            ordered = [inventory["items"][prediction["item_id"]] for prediction in predictions]
            volumes, diagnostics = self.geometry_estimator.volumes(
                image_path, [item["bbox"] for item in ordered], reference["bbox"],
                self.plate_diameter_cm if self.plate_diameter_cm is not None else 26.0,
                self.horizontal_fov_degrees, reference.get("shape", "ellipse"))
            dimension = "width" if reference.get("shape") == "rectangle" else "diameter"
            diagnostics["scale_source"] = f"user-supplied {dimension}" if self.plate_diameter_cm is not None else f"assumed 26cm plate {dimension}"
            diagnostics["camera_source"] = "assumed pinhole intrinsics and field of view"
            diagnostics["items"] = []
            corrected = []
            for food, item, prediction, volume in zip(result.food_items, ordered, predictions, volumes):
                food = replace(food)
                density_name = prediction.get("db_name")
                density = DENSITY_G_PER_ML.get(density_name)
                if density is None:
                    density = float(item["density_g_per_ml"])
                    density_source = "VLM-estimated bulk density assumption"
                else:
                    density_source = f"reference density: {density_name}; {DENSITY_SOURCE}"
                if not math.isfinite(density) or density <= 0:
                    raise ValueError("Bulk density must be finite and positive")
                grams = float(volume) * density
                item_trace = {"name": food.name, "volume_ml": float(volume), "density_g_per_ml": density,
                              "density_source": density_source}
                if not math.isfinite(grams) or grams <= 0:
                    food.portion_description += "; geometry fallback: no positive visible volume"
                    item_trace["used_geometry"] = False
                else:
                    ratio = grams / food.quantity_grams
                    for field in ("calories", "protein_g", "carbs_g", "fat_g"):
                        setattr(food, field, getattr(food, field) * ratio)
                    food.quantity_grams = grams
                    food.portion_description += f"; RGB geometry: {volume:.1f}ml x {density:.3f}g/ml; {density_source}; {diagnostics['scale_source']}"
                    item_trace["used_geometry"] = True
                diagnostics["items"].append(item_trace)
                corrected.append(food)
            trace["geometry"] = diagnostics
            result.food_items = corrected
            result.compute_totals()
        except Exception as exc:
            trace["geometry"] = {"error": str(exc), "used_geometry": False}
            for food in result.food_items:
                food.portion_description += f"; geometry fallback: {exc}"
        result.raw_response = json.dumps(trace, ensure_ascii=False)
        result.elapsed_seconds = time.perf_counter() - started
        return result
