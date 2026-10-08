"""Cache test data and published research models before an offline GPU job."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nutricoach.food_vision.evaluate_nutrition5k import prepare_dataset, FOOD_VISION_DIR
from nutricoach.food_vision.supervised_analyzers import FOOD_R1_MODEL, FOOD_R1_REVISION


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=FOOD_VISION_DIR / "data/nutrition5k_50")
    parser.add_argument("--food-r1", action="store_true", help="Download the roughly 17.5 GB full Food-R1 checkpoint")
    parser.add_argument("--geometry", action="store_true")
    args = parser.parse_args()
    manifest = prepare_dataset(args.data_dir, 50, 42)
    print(f"Prepared {len(manifest['dish_ids'])} test RGB images", flush=True)
    if args.food_r1:
        path = snapshot_download(FOOD_R1_MODEL, revision=FOOD_R1_REVISION,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        print(f"Food-R1 cached at {path}", flush=True)
    if args.geometry:
        from nutricoach.food_vision.geometry_analyzer import DEPTH_MODEL, DEPTH_REVISION, SAM_MODEL, SAM_REVISION
        for model, revision in ((DEPTH_MODEL, DEPTH_REVISION), (SAM_MODEL, SAM_REVISION)):
            snapshot_download(model, revision=revision, allow_patterns=["*.json", "*.safetensors"])
            print(f"Cached {model}@{revision}", flush=True)


if __name__ == "__main__":
    main()
