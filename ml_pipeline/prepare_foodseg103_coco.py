"""
Convert FoodSeg103 (semantic segmentation) into a COCO detection dataset for
RF-DETR fine-tuning.

Bounding boxes are derived per class from connected components of the semantic
mask (components below MIN_AREA_FRAC of the image are dropped as noise).
Output layout matches what rfdetr expects:

    ml_pipeline/data/foodseg103_coco/
      train/  *.jpg + _annotations.coco.json
      valid/  *.jpg + _annotations.coco.json

Run:
    PYTHONPATH=. uv run python ml_pipeline/prepare_foodseg103_coco.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

OUT_DIR = Path(__file__).parent / "data" / "foodseg103_coco"
DATASET = "EduardoPacheco/FoodSeg103"
MIN_AREA_FRAC = 0.002
VALID_MAX_IMAGES = 500
SEED = 42


def masks_to_boxes(label_arr: np.ndarray):
    """Yield (class_id, [x, y, w, h], area) from a semantic mask."""
    for cls in np.unique(label_arr):
        if cls == 0:
            continue
        binary = label_arr == cls
        components, n = ndimage.label(binary)
        for comp_idx in range(1, n + 1):
            ys, xs = np.nonzero(components == comp_idx)
            area = len(ys)
            if area < MIN_AREA_FRAC * label_arr.size:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            yield int(cls), [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)], float(area)


def convert_split(ds, split_dir: Path, id2label: dict, max_images=None):
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    ann_id = 1

    n = len(ds) if max_images is None else min(max_images, len(ds))
    for i in range(n):
        ex = ds[i]
        img = ex["image"].convert("RGB")
        label_arr = np.array(ex["label"])
        w, h = img.size

        file_name = f"{ex['id']:06d}.jpg"
        img.save(split_dir / file_name, "JPEG", quality=92)
        image_id = int(ex["id"])
        images.append({"id": image_id, "file_name": file_name, "width": w, "height": h})

        for cls, bbox, area in masks_to_boxes(label_arr):
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

        if (i + 1) % 500 == 0:
            print(f"  {split_dir.name}: {i + 1}/{n} images")

    categories = [
        {"id": int(k), "name": v, "supercategory": "food"}
        for k, v in sorted(id2label.items(), key=lambda kv: int(kv[0]))
        if int(k) != 0
    ]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    with open(split_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)
    print(f"{split_dir.name}: {len(images)} images, {len(annotations)} boxes")


def main():
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    id2label_path = hf_hub_download(DATASET, "id2label.json", repo_type="dataset")
    with open(id2label_path) as f:
        id2label = json.load(f)

    train_marker = OUT_DIR / "train" / "_annotations.coco.json"
    valid_marker = OUT_DIR / "valid" / "_annotations.coco.json"
    if train_marker.exists() and valid_marker.exists():
        print("Dataset already converted, skipping.")
        return

    print("Loading FoodSeg103...")
    ds = load_dataset(DATASET)

    convert_split(ds["train"], OUT_DIR / "train", id2label)
    valid = ds["validation"].shuffle(seed=SEED)
    convert_split(valid, OUT_DIR / "valid", id2label, max_images=VALID_MAX_IMAGES)
    print("Conversion complete.")


if __name__ == "__main__":
    main()
