"""
Zero-shot food image classification benchmark for configurable CLIP-compatible
and Jina v5 omni models on a Food101 validation subset.

Each model ranks the 101 Food101 class prompts ("a photo of {label}") against
each image embedding; reports top-1/top-5 accuracy and per-image latency.
Feeds the backend choice for food_vision Method 3 (clip_analyzer.py).

Run:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python \
        nut_agent/nutricoach/food_vision/bench_food_image_zeroshot.py
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python \
        nut_agent/nutricoach/food_vision/bench_food_image_zeroshot.py \
        --models openai/clip-vit-large-patch14 jinaai/jina-embeddings-v5-omni-small
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

FOOD_VISION_DIR = Path(__file__).parent

SEED = 42
N_IMAGES = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = FOOD_VISION_DIR / "results" / "bench_food_image_zeroshot.json"
DEFAULT_MODELS = [
    "openai/clip-vit-base-patch32",
    "jinaai/jina-embeddings-v5-omni-small",
]


def model_key(model_name):
    return model_name.lower().replace("/", "__").replace("-", "_").replace(".", "_")


def parse_model_spec(spec):
    if "=" in spec:
        backend, model_name = spec.split("=", 1)
        if backend not in {"clip", "jina"} or not model_name:
            raise ValueError(f"Invalid model spec {spec!r}; expected clip=MODEL_ID or jina=MODEL_ID")
        return backend, model_name
    backend = "jina" if "jina-embeddings-v5" in spec.lower() else "clip"
    return backend, spec


def load_eval_set():
    from datasets import load_dataset

    ds = load_dataset("ethz/food101", split="validation")
    label_names = [n.replace("_", " ") for n in ds.features["label"].names]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ds), size=N_IMAGES, replace=False)
    subset = ds.select(idx.tolist())
    images = [img.convert("RGB") for img in subset["image"]]
    labels = np.array(subset["label"])
    return images, labels, label_names


def topk_metrics(sims, labels, ks=(1, 5)):
    order = np.argsort(-sims, axis=1)
    out = {}
    for k in ks:
        hits = (order[:, :k] == labels[:, None]).any(axis=1)
        out[f"top{k}_accuracy"] = round(float(hits.mean()), 4)
    return out


def bench_clip(model_name, images, labels, prompts):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()

    # Same code path as clip_analyzer.py: full forward, logits_per_image
    tin = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    tin = {k: v.to(DEVICE) for k, v in tin.items()}

    sims_all = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(images), 32):
            iin = processor(images=images[i:i + 32], return_tensors="pt")
            iin = {k: v.to(DEVICE) for k, v in iin.items()}
            out = model(**tin, **iin)
            sims_all.append(out.logits_per_image.cpu())
    img_seconds = time.time() - t0

    sims = torch.cat(sims_all).numpy()
    metrics = topk_metrics(sims, labels)
    metrics["ms_per_image"] = round(1000 * img_seconds / len(images), 1)

    del model
    torch.cuda.empty_cache()
    return metrics


def bench_jina(model_name, images, labels, prompts):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        model_kwargs={"default_task": "retrieval"},
        device=DEVICE,
    )

    t_emb = np.asarray(model.encode_query(prompts, normalize_embeddings=True))

    i_embs = []
    t0 = time.time()
    for i in range(0, len(images), 16):
        emb = model.encode_document(images[i:i + 16], normalize_embeddings=True)
        i_embs.append(np.asarray(emb))
    img_seconds = time.time() - t0

    sims = np.concatenate(i_embs) @ t_emb.T
    metrics = topk_metrics(sims, labels)
    metrics["ms_per_image"] = round(1000 * img_seconds / len(images), 1)

    del model
    torch.cuda.empty_cache()
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=("Hugging Face model IDs to compare. Use clip=MODEL_ID or jina=MODEL_ID "
              "to select an implementation explicitly; otherwise Jina v5 is detected "
              "from its ID and other IDs are treated as CLIP-compatible."),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Device: {DEVICE}")
    images, labels, label_names = load_eval_set()
    prompts = [f"a photo of {name}" for name in label_names]
    print(f"Eval set: {len(images)} Food101 validation images, {len(prompts)} classes")

    results = {"meta": {
        "n_images": len(images), "seed": SEED,
        "dataset": "ethz/food101 validation", "models": args.models,
    }}

    seen = set()
    for spec in args.models:
        backend, model_name = parse_model_spec(spec)
        key = model_key(model_name)
        if key in seen:
            raise ValueError(f"Duplicate model ID: {model_name}")
        seen.add(key)
        if backend == "jina":
            metrics = bench_jina(model_name, images, labels, prompts)
        else:
            metrics = bench_clip(model_name, images, labels, prompts)
        results[key] = {"model": model_name, "backend": backend, **metrics}
        print(f"[{model_name}]", metrics)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
