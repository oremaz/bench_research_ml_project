"""Cache test data and published research models before an offline GPU job."""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nutricoach.food_vision.evaluate_nutrition5k import prepare_dataset, FOOD_VISION_DIR
from nutricoach.food_vision.supervised_analyzers import FOOD_R1_MODEL, FOOD_R1_REVISION


def main():
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=FOOD_VISION_DIR / "data/nutrition5k_500")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--food-r1", action="store_true", help="Download the roughly 17.5 GB full Food-R1 checkpoint")
    parser.add_argument("--papers", action="store_true", help="Cache pinned FoodLMM and third-party DPF releases")
    parser.add_argument("--local-qwen", action="store_true", help="Cache the pinned Qwen3.8-27B model for local inference")
    args = parser.parse_args()
    manifest = prepare_dataset(args.data_dir, args.count, args.seed)
    print(f"Prepared {len(manifest['dish_ids'])} test RGB images", flush=True)
    if args.local_qwen:
        from shared.llm_provider import LOCAL_MODEL_ID, LOCAL_MODEL_REVISION
        path = snapshot_download(LOCAL_MODEL_ID, revision=LOCAL_MODEL_REVISION,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        print(f"Local Qwen cached at {path}", flush=True)
    if args.food_r1:
        path = snapshot_download(FOOD_R1_MODEL, revision=FOOD_R1_REVISION,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
        print(f"Food-R1 cached at {path}", flush=True)
    if args.papers:
        from huggingface_hub import hf_hub_download
        from nutricoach.food_vision.evaluate_paper_system import RELEASE_COMMITS, FOODLMM_MODEL, FOODLMM_REVISION
        repositories = {"FoodLMM": "https://github.com/YuehaoYin/FoodLMM.git",
                        "DPF-Nutrition": "https://github.com/T0MYYY/Nutrition5k.git"}
        for name, commit in RELEASE_COMMITS.values():
            destination = FOOD_VISION_DIR / "data/paper_releases" / name
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "clone", "--no-checkout", repositories[name], str(destination)], check=True)
                subprocess.run(["git", "-C", str(destination), "checkout", "--detach", commit], check=True)
            actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
            if actual != commit:
                raise ValueError(f"Unexpected release commit in {destination}")
        snapshot_download(FOODLMM_MODEL, revision=FOODLMM_REVISION)
        snapshot_download("openai/clip-vit-large-patch14", revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
                          allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
        snapshot_download("depth-anything/Depth-Anything-V2-Small-hf", revision="5426e4f0f36572d16453bbda7a8389317b1bef99",
                          allow_patterns=["*.json", "*.safetensors"])
        hf_hub_download("T0MYYY/dpf-nutrition", "dpf_nutrition_food2k.pt",
                        revision="05850dd44c18c12e2af4bfd7ff17e570164ea098")
        print("Cached pinned FoodLMM and DPF reproduction assets", flush=True)


if __name__ == "__main__":
    main()
