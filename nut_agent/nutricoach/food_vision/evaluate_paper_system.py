"""Run pinned public paper releases on the existing frozen RGB test manifest."""

import argparse
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nutricoach.food_vision.evaluate_nutrition5k import (
    FOOD_VISION_DIR, write_json, summarize, nutrient_comparisons)

FOODLMM_MODEL = "Yueha0/FoodLMM-Chat"
FOODLMM_REVISION = "a349a7fb973a6064bb096406ec95bb1a2584759a"
RELEASE_COMMITS = {
    "foodlmm": ("FoodLMM", "b1c52d8ea94ef19e4801dbb5a13673732bafce27"),
    "dpf_reproduction": ("DPF-Nutrition", "ed37fbefd5582e7963c08c2815761ab4566cd838"),
}
QUANTIZED_MODULES = "model.layers.*"


def prepare_release(method):
    name, commit = RELEASE_COMMITS[method]
    release = FOOD_VISION_DIR / "data/paper_releases" / name
    repositories = {"foodlmm": "https://github.com/YuehaoYin/FoodLMM.git",
                    "dpf_reproduction": "https://github.com/T0MYYY/Nutrition5k.git"}
    if not release.exists():
        release.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", repositories[method], str(release)], check=True)
        subprocess.run(["git", "-C", str(release), "checkout", "--detach", commit], check=True)
    actual = subprocess.check_output(["git", "-C", str(release), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit or subprocess.check_output(["git", "-C", str(release), "status", "--porcelain"], text=True).strip():
        raise ValueError("Paper release must match the clean pinned commit")
    return release


def select_device(device):
    import torch
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    selected = torch.device(device)
    if selected.type not in ("cuda", "mps", "cpu"):
        raise ValueError("Paper inference supports cuda, mps or cpu")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; run this backend on an Apple Silicon Mac")
    return selected


def synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_memory(device):
    import torch
    if device.type == "cuda":
        return {"gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3}
    if device.type == "mps":
        return {"mps_current_allocated_gib": torch.mps.current_allocated_memory() / 1024**3,
                "mps_driver_allocated_gib": torch.mps.driver_allocated_memory() / 1024**3}
    return {}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


@contextmanager
def quanto_unpack_context(device):
    if device.type != "mps":
        yield
        return
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    class MPSUnpackMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.quanto.unpack.default:
                packed, bits = args
                if bits not in (2, 4):
                    raise ValueError("Quanto unpack supports only 2 or 4 bits")
                mask = (1 << bits) - 1
                return torch.cat([(packed // (1 << shift)) & mask
                                  for shift in range(0, 8, bits)], dim=0).to(torch.uint8)
            return func(*args, **(kwargs or {}))

    # Quanto's native MPS unpack recompiles Metal shaders and synchronizes per call.
    with MPSUnpackMode():
        yield


def nutrition_stopping_criteria(token_ids, prompt_length):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class NutritionTokensComplete(StoppingCriteria):
        def __init__(self):
            self.remaining = set(token_ids)
            self.seen_length = prompt_length

        def __call__(self, input_ids, scores, **kwargs):
            self.remaining.difference_update(input_ids[0, self.seen_length:].tolist())
            self.seen_length = input_ids.shape[-1]
            return not self.remaining

    return StoppingCriteriaList([NutritionTokensComplete()])


class FoodLMM:
    def __init__(self, release, device, quantization="none", quantized_checkpoint=None):
        import torch
        from transformers import AutoTokenizer, CLIPImageProcessor, AutoConfig
        from huggingface_hub import snapshot_download
        from model.LISA import LISAForCausalLM
        from utils.conversation import conv_templates
        from model.llava.mm_utils import tokenizer_image_token

        self.device = torch.device(device)
        self.dtype = {"cuda": torch.bfloat16, "mps": torch.float16, "cpu": torch.float32}[self.device.type]
        self.quantization = quantization
        if quantization not in ("none", "4bit"):
            raise ValueError("FoodLMM quantization must be none or 4bit")
        if quantized_checkpoint and quantization != "4bit":
            raise ValueError("Packed checkpoint requires --quantization 4bit")
        self.tokenizer = AutoTokenizer.from_pretrained(
            FOODLMM_MODEL, revision=FOODLMM_REVISION, use_fast=False, padding_side="right")
        self.tokenizer.pad_token = self.tokenizer.unk_token
        prefixes = {"seg": "SEG", "mass": "MASS", "calorie": "CAL", "fat": "FAT",
                    "carbohydrate": "CARB", "protein": "PRO"}
        indices = {}
        for name, token in prefixes.items():
            for number in range(21):
                text = f"[{token}{number}]" if number else f"[{token}_TOTAL]" if name != "seg" else "[SEG]"
                encoded = self.tokenizer(text, add_special_tokens=False).input_ids
                if len(encoded) != 1:
                    raise ValueError(f"Missing nutritional vocabulary token: {text}")
                indices[f"{name}_token_idx" + (f"_{number}" if number else "")] = encoded[0]
        vision_path = snapshot_download("openai/clip-vit-large-patch14",
            revision="32bd64288804d66eefd0ccbe215aa642df71cc41", local_files_only=True)
        self.total_token_ids = [indices[f"{name}_token_idx"]
                                for name in ("mass", "calorie", "fat", "carbohydrate", "protein")]
        config = AutoConfig.from_pretrained(FOODLMM_MODEL, revision=FOODLMM_REVISION)
        config.vision_tower = vision_path
        config.mm_vision_tower = vision_path
        if quantized_checkpoint:
            from accelerate import init_empty_weights
            from safetensors.torch import load_file
            metadata = json.loads((quantized_checkpoint / "configuration.json").read_text())
            if (metadata["revision"] != FOODLMM_REVISION or metadata["quantized_modules"] != QUANTIZED_MODULES
                    or metadata["weights_sha256"] != digest(quantized_checkpoint / "model.safetensors")):
                raise ValueError("Packed checkpoint provenance differs from FoodLMM-Chat")
            with init_empty_weights():
                self.model = LISAForCausalLM(config, **indices)
            self.model.model.visual_model = None
            self.model.get_model().initialize_vision_modules(config)
            self.model.to(dtype=self.dtype)
            from optimum.quanto import quantize, qint4
            quantize(self.model, weights=qint4, include=QUANTIZED_MODULES)
            load_device = str(self.device) if self.device.type == "cuda" else "cpu"
            state = load_file(str(quantized_checkpoint / "model.safetensors"), device=load_device)
            for name, tensor in state.items():
                if tensor.is_floating_point() and tensor.dtype != self.dtype:
                    state[name] = tensor.to(dtype=self.dtype)
            self.model.load_state_dict(state, strict=True, assign=True)
            del state
        else:
            self.model, information = LISAForCausalLM.from_pretrained(
                FOODLMM_MODEL, revision=FOODLMM_REVISION, torch_dtype=self.dtype,
                config=config, low_cpu_mem_usage=True, output_loading_info=True, **indices)
            if information["missing_keys"] or information["mismatched_keys"]:
                raise ValueError(f"Incomplete author checkpoint: {information}")
            self.model.model.visual_model = None
            self.model.get_model().initialize_vision_modules(config)
            if quantization == "4bit":
                from optimum.quanto import quantize, freeze, qint4
                self.model.to(device=self.device, dtype=self.dtype)
                quantize(self.model, weights=qint4, include=QUANTIZED_MODULES)
                freeze(self.model)
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.processor = CLIPImageProcessor.from_pretrained(vision_path)
        self.conversation = conv_templates["conv_food_assistant"]
        self.tokenize_image = tokenizer_image_token

    def export_quantized(self, directory):
        from safetensors.torch import save_file
        if self.quantization != "4bit":
            raise ValueError("Only 4-bit FoodLMM can be exported as a packed checkpoint")
        directory.mkdir(parents=True, exist_ok=True)
        save_file({name: tensor.cpu().contiguous() for name, tensor in self.model.state_dict().items()},
                  str(directory / "model.safetensors"))
        write_json(directory / "configuration.json", {"model": FOODLMM_MODEL, "revision": FOODLMM_REVISION,
            "quantized_modules": QUANTIZED_MODULES, "quanto": version("optimum-quanto"),
            "weights_sha256": digest(directory / "model.safetensors")})

    def predict(self, image):
        import torch

        clip = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device, dtype=self.dtype)
        conversation = self.conversation.copy()
        conversation.append_message(conversation.roles[0],
            "<im_start><image><im_end>\nCan you provide the total caloric, fat, carbohydrate, and protein content of this dish directly?")
        conversation.append_message(conversation.roles[1], None)
        ids = self.tokenize_image(conversation.get_prompt(), self.tokenizer, return_tensors="pt").unsqueeze(0).to(self.device)
        # Quanto tensor subclasses require version counters during inference.
        with torch.no_grad(), quanto_unpack_context(self.device):
            synchronize(self.device)
            started = time.perf_counter()
            print(f"FoodLMM: generating nutrition tokens on {self.device}", file=sys.stderr, flush=True)
            outputs = self.model.generate(images=clip, input_ids=ids, max_new_tokens=256,
                attention_mask=torch.ones_like(ids), num_beams=1, do_sample=False,
                use_cache=True, output_hidden_states=False, return_dict_in_generate=True,
                stopping_criteria=nutrition_stopping_criteria(self.total_token_ids, ids.shape[-1]))
            output_ids = outputs.sequences
            synchronize(self.device)
            print(f"FoodLMM: generated {output_ids.shape[-1] - ids.shape[-1]} tokens in {time.perf_counter() - started:.2f}s; evaluating nutrition heads", file=sys.stderr, flush=True)
            started = time.perf_counter()
            # Preserve the author's full-prefix states for the trained nutrition heads.
            final = self.model(images=clip, input_ids=output_ids[:, :-1], past_key_values=None,
                use_cache=False, output_hidden_states=False, return_dict=True).hidden_states
            hidden = self.model.model.text_hidden_fcs[0](final)
            masks = self.model.get_token_mask(output_ids, self.device, is_train=False)
            keys = {"calories": ("calorie", 2), "mass_g": ("mass", 1), "fat_g": ("fat", 3),
                    "carbs_g": ("carb", 4), "protein_g": ("protein", 5)}
            prediction = {}
            for target, (key, index) in keys.items():
                values = getattr(self.model, f"total_{key}_head")(hidden[masks[index]]).reshape(-1)
                if values.numel() != 1:
                    raise ValueError(f"Expected one whole-dish {target} token, received {values.numel()}")
                prediction[target] = float(values.item())
            synchronize(self.device)
            print(f"FoodLMM: nutrition heads completed in {time.perf_counter() - started:.2f}s", file=sys.stderr, flush=True)
        return prediction, self.tokenizer.decode(output_ids[0][output_ids[0] != -200], skip_special_tokens=False)


class DPFReproduction:
    def __init__(self, release, device):
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import pipeline
        from nutrition5k_pkg.models.dpf_nutrition import DPFNutritionNet
        from nutrition5k_pkg.data.transforms import get_dpf_rgb_transform, get_dpf_depth_transform

        checkpoint = hf_hub_download("T0MYYY/dpf-nutrition", "dpf_nutrition_food2k.pt",
            revision="05850dd44c18c12e2af4bfd7ff17e570164ea098")
        class CheckpointDPFNutritionNet(DPFNutritionNet):
            def _load_food2k_or_log_fallback(self, path):
                # The complete released checkpoint replaces initialization weights.
                return

        self.model = CheckpointDPFNutritionNet(["calories", "mass", "fat", "carb", "protein"], pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        print("DPF: complete released checkpoint loaded, including RGB and depth backbones", file=sys.stderr, flush=True)
        self.model.to(device).eval()
        self.depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
            revision="5426e4f0f36572d16453bbda7a8389317b1bef99", device=device)
        self.rgb_transform = get_dpf_rgb_transform()
        self.depth_transform = get_dpf_depth_transform()
        self.device = device

    def predict(self, image):
        import numpy as np
        import torch
        from PIL import Image

        depth = self.depth(image)["predicted_depth"].detach().float().cpu().numpy().squeeze()
        low, high = float(depth.min()), float(depth.max())
        normalized = (depth - low) / (high - low) if high > low else np.zeros_like(depth)
        depth_image = Image.fromarray(np.clip(normalized * 65535, 0, 65535).astype(np.uint16))
        with torch.inference_mode():
            outputs = self.model(self.rgb_transform(image).unsqueeze(0).to(self.device),
                                 self.depth_transform(depth_image).unsqueeze(0).to(self.device))
        keys = {"calories": "calories", "mass_g": "mass", "fat_g": "fat", "carbs_g": "carb", "protein_g": "protein"}
        return {target: float(outputs[key].item()) for target, key in keys.items()}, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=RELEASE_COMMITS, required=True)
    parser.add_argument("--data-dir", type=Path, default=FOOD_VISION_DIR / "data/nutrition5k_500")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cuda[:index], mps or cpu")
    parser.add_argument("--quantization", choices=("none", "4bit"), default="none")
    parser.add_argument("--quantized-checkpoint", type=Path)
    parser.add_argument("--export-quantized", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.method != "foodlmm" and (args.quantization != "none" or args.quantized_checkpoint or args.export_quantized):
        parser.error("Quantization options apply only to FoodLMM")
    if args.export_quantized and args.quantization != "4bit":
        parser.error("--export-quantized requires --quantization 4bit")
    device = select_device(args.device)
    method = "foodlmm_4bit" if args.quantization == "4bit" else args.method
    _, commit = RELEASE_COMMITS[args.method]
    release = prepare_release(args.method)
    sys.path.insert(0, str(release))
    from PIL import Image
    import torch
    import transformers
    torch.set_num_threads(4)
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    configuration = {"manifest_sha256": digest(manifest_path), "release_commit": commit,
        "method": method, "device": str(device), "torch": torch.__version__,
        "transformers": transformers.__version__, "code_sha256": digest(Path(__file__)),
        "foodlmm_revision": FOODLMM_REVISION if args.method == "foodlmm" else None,
        "vision_revision": "32bd64288804d66eefd0ccbe215aa642df71cc41" if args.method == "foodlmm" else None,
        "dpf_revision": "05850dd44c18c12e2af4bfd7ff17e570164ea098" if args.method == "dpf_reproduction" else None,
        "depth_revision": "5426e4f0f36572d16453bbda7a8389317b1bef99" if args.method == "dpf_reproduction" else None,
        "max_new_tokens": 256 if args.method == "foodlmm" else None, "do_sample": False,
        "quantization": args.quantization, "quantized_modules": QUANTIZED_MODULES if args.quantization == "4bit" else None,
        "quanto": version("optimum-quanto") if args.quantization == "4bit" else None,
        "packed_weights_sha256": digest(args.quantized_checkpoint / "model.safetensors") if args.quantized_checkpoint else None,
        "nutrition_only": args.method == "foodlmm",
        "generation_use_cache": True if args.method == "foodlmm" else None,
        "generation_stop": "all five whole-dish nutrition tokens, EOS, or token limit" if args.method == "foodlmm" else None,
        "nutrition_hidden_states": "full-prefix forward pass" if args.method == "foodlmm" else None,
        "training_provenance": "Published external weights; fitted IDs cannot be independently audited",
        "implementation": "Author FoodLMM-Chat release" if args.method == "foodlmm" else
                          "Third-party DPF reproduction with Depth Anything V2 and modified fusion"}
    configuration_path = args.output_dir / "configuration.json"
    if configuration_path.exists() and json.loads(configuration_path.read_text()) != configuration:
        raise ValueError("Output configuration differs; use another output directory")
    write_json(configuration_path, configuration)
    path = args.output_dir / f"{method}.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    pending = [dish for dish in manifest["dish_ids"]
               if dish not in records or (args.retry_errors and records[dish].get("error"))]
    if not pending and not args.prepare_only and not args.export_quantized:
        print(f"{method}: all dishes already attempted; preserving recorded resources")
        return
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    analyzer = (FoodLMM(release, device, args.quantization, args.quantized_checkpoint)
                if args.method == "foodlmm" else DPFReproduction(release, device))
    synchronize(device)
    resources = {"device": str(device), "profiled_dishes": len(pending),
                 "model_loading_seconds": time.perf_counter() - started,
                 "loading_memory": device_memory(device)}
    write_json(args.output_dir / "resources.json", resources)
    if args.export_quantized:
        analyzer.export_quantized(args.export_quantized)
    if args.prepare_only:
        return
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    for index, dish in enumerate(pending, 1):
        synchronize(device)
        started = time.perf_counter()
        record = {"dish_id": dish, "http_requests": 0, "used_fallback": False}
        try:
            image_path = args.data_dir / "images" / f"{dish}.png"
            if digest(image_path) != manifest["image_sha256"][dish]:
                raise ValueError("Test image hash differs from the frozen manifest")
            with Image.open(image_path) as opened:
                prediction, raw = analyzer.predict(opened.convert("RGB"))
            if any(not math.isfinite(value) or value < 0 for value in prediction.values()):
                raise ValueError(f"Invalid nutritional prediction: {prediction}")
            record.update(prediction=prediction, raw_response=raw)
        except Exception as exc:
            record["error"] = str(exc)
        synchronize(device)
        record["inference_seconds"] = time.perf_counter() - started
        records[dish] = record
        write_json(path, records)
        print(f"{method}: {index}/{len(pending)} {dish} {record.get('error', 'OK')}", flush=True)
    resources.update(inference_memory=device_memory(device),
        peak_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2))
    write_json(args.output_dir / "resources.json", resources)
    summary = summarize({method: records}, manifest, [method], manifest["seed"])
    summary["nutrient_comparisons"] = nutrient_comparisons({method: records}, manifest, manifest["seed"])
    summary["resources"] = resources
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
