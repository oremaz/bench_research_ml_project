"""
Method 1: RF-DETR Fine-tuning Pipeline for Food Detection.

Uses Roboflow's RF-DETR (ICLR 2026) — a real-time transformer detector
with a DINOv2 backbone — fine-tuned on food detection datasets.

Pipeline:
  1. RF-DETR detects food items in the image with bounding boxes + labels
  2. Portion estimation from bounding box area relative to plate
  3. Nutrition DB lookup for calorie/macro estimates

Requires:
  pip install rfdetr supervision

Fine-tuning (run once):
  See RFDETRFoodTrainer.train() or the included Colab-ready script.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict

from .base import FoodAnalyzer, FoodAnalysisResult, FoodItem
from .nutrition_db import NutritionDB

logger = logging.getLogger(__name__)


# Default portion estimates based on relative bounding box area
# (fraction of image area → approximate grams)
PORTION_AREA_MAP = [
    (0.40, 350),  # >40% of image → large portion ~350g
    (0.25, 250),  # 25-40% → medium-large ~250g
    (0.15, 180),  # 15-25% → medium ~180g
    (0.08, 120),  # 8-15% → small-medium ~120g
    (0.03, 70),   # 3-8% → small ~70g
    (0.00, 30),   # <3% → garnish/condiment ~30g
]


def estimate_grams_from_area_fraction(area_fraction: float) -> float:
    """Estimate portion weight from bbox area as fraction of image."""
    for threshold, grams in PORTION_AREA_MAP:
        if area_fraction >= threshold:
            return grams
    return 30.0


class RFDETRAnalyzer(FoodAnalyzer):
    """
    Food analyzer using RF-DETR object detection.

    Supports both:
    - Pre-trained COCO model (detects generic food items)
    - Fine-tuned model on food datasets (much better accuracy)
    """

    method_name = "rf_detr"

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        model_size: str = "base",
    ):
        """
        Args:
            model_path: Path to fine-tuned weights. None = use COCO pretrained.
            confidence_threshold: Minimum detection confidence.
            model_size: 'base' or 'large' for RF-DETR variant.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        self.nutrition_db = NutritionDB()
        self._model = None

    def _load_model(self):
        """Lazy-load the RF-DETR model."""
        if self._model is not None:
            return

        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except ImportError:
            raise ImportError(
                "rfdetr is required for RFDETRAnalyzer. "
                "Install with: pip install rfdetr"
            )

        ModelClass = RFDETRLarge if self.model_size == "large" else RFDETRBase

        if self.model_path and Path(self.model_path).exists():
            self._model = ModelClass(pretrain_weights=self.model_path)
            logger.info("Loaded fine-tuned RF-DETR from %s", self.model_path)
        else:
            self._model = ModelClass()
            logger.info("Loaded RF-DETR with COCO pretrained weights (%s)", self.model_size)

    def analyze(self, image_path: str) -> FoodAnalysisResult:
        """Detect food items and estimate nutrition from a plate image."""
        start = time.time()
        result = FoodAnalysisResult(method=self.method_name)

        try:
            self._load_model()
            from PIL import Image

            img = Image.open(image_path)
            img_w, img_h = img.size
            img_area = img_w * img_h

            # Run detection
            detections = self._model.predict(image_path, threshold=self.confidence_threshold)

            # Parse detections (supervision Detections format)
            food_items = self._parse_detections(detections, img_area)

            result.food_items = food_items
            result.compute_totals()

        except ImportError as e:
            result.error = str(e)
        except Exception as e:
            logger.error("RF-DETR analysis failed: %s", e)
            result.error = f"Detection failed: {str(e)}"

        result.elapsed_seconds = time.time() - start
        return result

    def _parse_detections(self, detections, img_area: int) -> List[FoodItem]:
        """Convert RF-DETR detections to FoodItem list."""
        food_items = []

        # COCO food-related class IDs and names
        coco_food_classes = {
            46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
            50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
            54: "donut", 55: "cake",
        }

        # If fine-tuned, class names come from the model's config
        has_custom_names = hasattr(detections, 'data') and 'class_name' in getattr(detections, 'data', {})

        xyxy = detections.xyxy if hasattr(detections, 'xyxy') else []
        class_ids = detections.class_id if hasattr(detections, 'class_id') else []
        confidences = detections.confidence if hasattr(detections, 'confidence') else []

        for i in range(len(xyxy)):
            bbox = xyxy[i]
            class_id = int(class_ids[i]) if i < len(class_ids) else -1
            conf = float(confidences[i]) if i < len(confidences) else 0.0

            # Get food name
            if has_custom_names:
                name = detections.data['class_name'][i]
            elif class_id in coco_food_classes:
                name = coco_food_classes[class_id]
            else:
                continue  # Skip non-food detections

            # Estimate portion from bbox area
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_fraction = bbox_area / img_area if img_area > 0 else 0.1
            grams = estimate_grams_from_area_fraction(area_fraction)

            # Lookup nutrition
            nutrients = self.nutrition_db.enrich_food_item(name, grams)

            food_items.append(FoodItem(
                name=name,
                quantity_grams=grams,
                confidence=conf,
                calories=nutrients["calories"],
                protein_g=nutrients["protein_g"],
                carbs_g=nutrients["carbs_g"],
                fat_g=nutrients["fat_g"],
                portion_description=f"~{grams:.0f}g (bbox {area_fraction:.1%} of image)",
            ))

        return food_items


class RFDETRFoodTrainer:
    """
    Fine-tuning helper for RF-DETR on food detection datasets.

    Supports:
    - FoodSeg103 (103 ingredient classes, segmentation masks)
    - UEC-FoodPix Complete (100 food categories)
    - Custom Roboflow datasets (COCO format)

    Usage:
        trainer = RFDETRFoodTrainer()
        trainer.train(
            dataset_dir="./food_coco/",
            epochs=50,
            output_dir="./rf_detr_food_weights/",
        )
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size

    def train(
        self,
        dataset_dir: str,
        epochs: int = 50,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        lr: float = 1e-4,
        output_dir: str = "./rf_detr_food_weights",
        resume: Optional[str] = None,
    ) -> str:
        """
        Fine-tune RF-DETR on a food detection dataset in COCO format.

        Expected dataset structure:
            dataset_dir/
              train/
                images/
                _annotations.coco.json
              valid/
                images/
                _annotations.coco.json
              test/  (optional)
                images/
                _annotations.coco.json

        Returns:
            Path to the best checkpoint.
        """
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except ImportError:
            raise ImportError("rfdetr is required. Install with: pip install rfdetr")

        ModelClass = RFDETRLarge if self.model_size == "large" else RFDETRBase
        model = ModelClass()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting RF-DETR fine-tuning: %s, %d epochs, bs=%d, lr=%s",
            self.model_size, epochs, batch_size, lr,
        )

        model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            output_dir=str(output_path),
            **({"resume": resume} if resume else {}),
        )

        # Find best checkpoint
        checkpoints = sorted(output_path.glob("*.pth"))
        best = str(checkpoints[-1]) if checkpoints else str(output_path / "model_final.pth")
        logger.info("Training complete. Best checkpoint: %s", best)
        return best

    @staticmethod
    def download_food_dataset(
        dataset_name: str = "food-detection",
        workspace: str = "roboflow-universe",
        version: int = 1,
        output_dir: str = "./food_coco",
        api_key: Optional[str] = None,
    ) -> str:
        """
        Download a food detection dataset from Roboflow Universe in COCO format.

        Popular food datasets on Roboflow:
        - "food-detection" (general food items)
        - "food-items-detection" (packaged foods)
        - "fruits-and-vegetables" (produce)

        Returns:
            Path to the downloaded dataset directory.
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError(
                "roboflow is required. Install with: pip install roboflow"
            )

        rf = Roboflow(api_key=api_key or "")
        project = rf.workspace(workspace).project(dataset_name)
        dataset = project.version(version).download("coco", location=output_dir)
        logger.info("Dataset downloaded to %s", output_dir)
        return output_dir
