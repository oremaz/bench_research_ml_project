"""Train a frozen DINOv2 RGB encoder with five supervised ridge outputs."""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nutricoach.food_vision.evaluate_nutrition5k import BASE_URL, download, digest, parse_metadata, write_json
from nutricoach.food_vision.supervised_analyzers import TARGETS, RGB_CHECKPOINT


def split_training_dishes(rgb_train, depth_train, rgb_test, metadata, seed=42, count=None):
    if rgb_train & rgb_test:
        raise ValueError("Official train/test overlap")
    dishes = sorted(rgb_train & depth_train & metadata.keys())
    if count is not None:
        if count < 10 or count > len(dishes):
            raise ValueError("Training count must be between 10 and the available training dishes")
        dishes = sorted(np.random.default_rng(seed).choice(dishes, count, replace=False).tolist())
    # Incremental scans of a plate stay on the same acquisition day.
    days = {dish: datetime.fromtimestamp(int(dish.removeprefix("dish_")), timezone.utc).date().isoformat()
            for dish in dishes}
    unique_days = sorted(set(days.values()))
    if len(unique_days) < 2:
        raise ValueError("Need at least two acquisition days for grouped validation")
    validation_days = set(np.random.default_rng(seed).choice(
        unique_days, max(1, round(len(unique_days) * 0.2)), replace=False).tolist())
    train = [dish for dish in dishes if days[dish] not in validation_days]
    validation = [dish for dish in dishes if days[dish] in validation_days]
    return train, validation


def fit_head(features, labels, train_indices, validation_indices, alphas=(0.1, 1, 10, 100, 1000)):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if labels.shape != (len(features), len(TARGETS)) or not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("Invalid features or five-target labels")
    if set(train_indices) & set(validation_indices) or not len(train_indices) or not len(validation_indices):
        raise ValueError("Training and validation indices must be nonempty and disjoint")
    feature_scaler = StandardScaler().fit(features[train_indices])
    target_scaler = StandardScaler().fit(labels[train_indices])
    x = feature_scaler.transform(features)
    y = target_scaler.transform(labels[train_indices])
    normalization = np.maximum(labels[train_indices].mean(axis=0), 1e-6)
    best = None
    scores = []
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(x[train_indices], y)
        predicted = np.maximum(target_scaler.inverse_transform(model.predict(x[validation_indices])), 0)
        mae = np.abs(predicted - labels[validation_indices]).mean(axis=0)
        score = float((mae / normalization).mean())
        scores.append({"alpha": alpha, "mean_normalized_mae": score, "mae": dict(zip(TARGETS, mae.tolist()))})
        if best is None or score < best[0]:
            best = score, model, alpha
    return feature_scaler, target_scaler, best[1], best[2], scores


def train(args):
    import torch
    import timm
    import sklearn
    from PIL import Image
    from timm.data import create_transform, resolve_model_data_config
    from ml_pipeline.pipelines_torch.vision_models import get_model

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    source_names = [f"dish_ids/splits/{name}.txt" for name in
                    ("rgb_train_ids", "depth_train_ids", "rgb_test_ids")]
    source_names += [f"metadata/dish_metadata_cafe{cafe}.csv" for cafe in (1, 2)]
    paths = [download(name, args.data_dir / name) for name in source_names]
    metadata = {}
    for path in paths[3:]:
        part = parse_metadata(path.read_text())
        if metadata.keys() & part.keys():
            raise ValueError("Duplicate metadata across cafes")
        metadata.update(part)
    train_ids, validation_ids = split_training_dishes(
        *(set(path.read_text().split()) for path in paths[:3]), metadata, args.seed, args.count)
    dishes = train_ids + validation_ids
    unavailable_path = args.data_dir / "unavailable_images.json"
    known_missing = set()
    if unavailable_path.exists():
        unavailable = json.loads(unavailable_path.read_text())
        if unavailable["source_url"] != BASE_URL:
            raise ValueError("Unavailable-image cache refers to a different dataset source")
        known_missing = set(unavailable["http_404_dishes"])
    def fetch(dish):
        if dish in known_missing:
            return None
        try:
            return download(f"imagery/realsense_overhead/{dish}/rgb.png", args.data_dir / "images" / f"{dish}.png")
        except HTTPError as exc:
            if exc.code != 404:
                raise
            return None
    print(f"Preparing {len(train_ids)} training and {len(validation_ids)} validation RGB images", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        images = []
        for index, path in enumerate(pool.map(fetch, dishes), 1):
            images.append(path)
            if index % 100 == 0:
                print(f"Downloaded or cached {index}/{len(dishes)}", flush=True)
    missing = [dish for dish, path in zip(dishes, images) if path is None]
    write_json(unavailable_path, {"source_url": BASE_URL, "http_404_dishes": sorted(known_missing | set(missing))})
    available = {dish for dish, path in zip(dishes, images) if path is not None}
    train_ids = [dish for dish in train_ids if dish in available]
    validation_ids = [dish for dish in validation_ids if dish in available]
    dishes = [dish for dish in dishes if dish in available]
    images = [path for path in images if path is not None]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_model("timm_dinov2_vit_small", num_classes=0, pretrained=True, img_size=224).to(device).eval()
    if args.prepare_only:
        print(f"Prepared {len(images)} training/validation images and DINOv2 weights; {len(missing)} HTTP 404 exclusions", flush=True)
        return
    preprocessing = resolve_model_data_config(encoder.model)
    preprocessing["input_size"] = (3, 224, 224)
    transform = create_transform(**preprocessing, is_training=False)
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), args.batch_size):
            batch = []
            for path in images[start:start + args.batch_size]:
                with Image.open(path) as source:
                    batch.append(transform(source.convert("RGB")))
            features.append(encoder(torch.stack(batch).to(device)).cpu().numpy())
            if start % (args.batch_size * 20) == 0:
                print(f"Encoded {min(start + args.batch_size, len(images))}/{len(images)}", flush=True)
    labels = np.array([[metadata[dish][target] for target in TARGETS] for dish in dishes])
    scalers = fit_head(np.concatenate(features), labels, np.arange(len(train_ids)),
                       np.arange(len(train_ids), len(dishes)))
    feature_scaler, target_scaler, head, alpha, scores = scalers
    provenance = {"train_ids": train_ids, "validation_ids": validation_ids, "seed": args.seed,
                  "validation_grouping": "UTC acquisition day, 20 percent of days from official training split",
                  "source_sha256": {name: digest(path) for name, path in zip(source_names, paths)},
                  "image_sha256": {dish: digest(path) for dish, path in zip(dishes, images)},
                  "source_code_sha256": {name: digest(Path(__file__).parent / name) for name in
                                         ("train_rgb_regression.py", "supervised_analyzers.py")},
                  "validation_scores": scores, "selected_alpha": alpha,
                  "excluded_http_404_dishes": missing,
                  "training_count_limit": args.count, "device": device,
                  "versions": {"torch": str(torch.__version__), "timm": timm.__version__,
                               "sklearn": sklearn.__version__, "numpy": np.__version__},
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
    state = {"format_version": 1, "targets": list(TARGETS), "backbone": "timm_dinov2_vit_small",
             "image_size": 224, "preprocessing": preprocessing,
             "encoder_state": {key: value.cpu() for key, value in encoder.state_dict().items()},
             "feature_mean": torch.from_numpy(feature_scaler.mean_),
             "feature_scale": torch.from_numpy(feature_scaler.scale_),
             "target_mean": torch.from_numpy(target_scaler.mean_),
             "target_scale": torch.from_numpy(target_scaler.scale_),
             "coefficient": torch.from_numpy(head.coef_), "intercept": torch.from_numpy(head.intercept_),
             "provenance": provenance}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(args.checkpoint)
    write_json(args.checkpoint.with_suffix(".json"), provenance)
    print(json.dumps({"checkpoint": str(args.checkpoint), "selected_alpha": alpha,
                      "validation_scores": scores}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data/nutrition5k_training")
    parser.add_argument("--checkpoint", type=Path, default=RGB_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, help="Optional training-only subset; default uses all available overhead training images")
    parser.add_argument("--device")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prepare-only", action="store_true", help="Download training images and encoder weights without fitting")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch-size and threads must be positive")
    train(args)


if __name__ == "__main__":
    main()
