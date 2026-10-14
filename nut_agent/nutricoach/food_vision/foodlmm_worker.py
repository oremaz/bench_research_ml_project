"""Persistent JSON-line inference worker for the isolated FoodLMM environment."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(output=None):
    output = output if output is not None else sys.stdout
    sys.stdout = sys.stderr
    model = None
    for line in sys.stdin:
        try:
            from PIL import Image, ImageOps
            from nutricoach.food_vision.evaluate_paper_system import FoodLMM, prepare_release, select_device

            request = json.loads(line)
            if model is None:
                release = prepare_release("foodlmm")
                sys.path.insert(0, str(release))
                from huggingface_hub import snapshot_download
                snapshot_download("openai/clip-vit-large-patch14",
                    revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
                    allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
                configured = os.getenv("FOODLMM_QUANTIZED_CHECKPOINT")
                checkpoint = Path(configured) if configured else Path(__file__).parent / "data/foodlmm_4bit"
                packed = all((checkpoint / name).is_file() for name in ("configuration.json", "model.safetensors"))
                if configured and not packed:
                    raise ValueError(f"FoodLMM packed checkpoint missing or incomplete: {checkpoint}")
                print(f"FoodLMM: {'loading packed checkpoint' if packed else 'loading original weights and quantizing'}", file=sys.stderr, flush=True)
                model = FoodLMM(release, select_device(os.getenv("FOODLMM_DEVICE", "auto")),
                               quantization="4bit", quantized_checkpoint=checkpoint if packed else None)
                if not packed:
                    print(f"FoodLMM: saving packed checkpoint to {checkpoint}", file=sys.stderr, flush=True)
                    model.export_quantized(checkpoint)
                print("FoodLMM: model ready", file=sys.stderr, flush=True)
            with Image.open(request["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((1024, 1024))
            values, raw_response = model.predict(image)
            response = {"values": values, "raw_response": raw_response}
        except Exception as exc:
            response = {"error": f"FoodLMM 4-bit: {exc}"}
        output.write(json.dumps(response) + "\n")
        output.flush()


if __name__ == "__main__":
    with os.fdopen(os.dup(1), "w", buffering=1) as output:
        os.dup2(2, 1)
        main(output)
