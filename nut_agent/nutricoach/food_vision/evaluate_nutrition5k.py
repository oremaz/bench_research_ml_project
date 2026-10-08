"""Evaluate food vision pipelines on a fixed Nutrition5k official-test RGB subset."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.config import OPENROUTER_MODEL_ID, OPENROUTER_REASONING_EFFORT, OPENROUTER_MAX_OUTPUT_TOKENS, OPENROUTER_REASONING_EFFORTS, openrouter_reasoning
from nutricoach.food_vision.compare import get_available_methods
from nutricoach.food_vision.rf_detr_analyzer import DEFAULT_MODEL_PATH
from nutricoach.food_vision.supervised_analyzers import TARGETS, RGB_CHECKPOINT, FOOD_R1_MODEL, FOOD_R1_REVISION, validate_regression_split

BASE_URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset/"
FOOD_VISION_DIR = Path(__file__).parent
METHODS = ("vlm_single", "geometry_vlm_db", "rgb_regression", "food_r1")
LEGACY_METHODS = ("rf_detr", "clip_ensemble", "rag_vlm", "vlm_db_grounded")
API_METHODS = {"vlm_single", "geometry_vlm_db", "clip_ensemble", "rag_vlm", "vlm_db_grounded"}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def download(relative_path, destination):
    destination = Path(destination)
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(BASE_URL + relative_path, timeout=90) as response:
        content = response.read()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(destination)
    return destination


def parse_metadata(text):
    dishes = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].startswith("dish_"):
            continue
        values = dict(zip(TARGETS, map(float, row[1:6])))
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f"Invalid reference nutrition for {row[0]}")
        if row[0] in dishes:
            raise ValueError(f"Duplicate dish metadata: {row[0]}")
        dishes[row[0]] = values
    return dishes


def select_dishes(rgb_test, depth_test, rgb_train, metadata, count, seed):
    if rgb_test & rgb_train:
        raise ValueError("Official RGB train and test splits overlap")
    eligible = sorted(rgb_test & depth_test & metadata.keys())
    if count <= 0 or count > len(eligible):
        raise ValueError(f"Requested {count} dishes from {len(eligible)} eligible test dishes")
    selected = np.random.default_rng(seed).choice(eligible, size=count, replace=False).tolist()
    return selected, len(eligible)


def prepare_dataset(data_dir, count, seed):
    data_dir = Path(data_dir)
    sources = [f"dish_ids/splits/{name}.txt" for name in
               ("rgb_test_ids", "depth_test_ids", "rgb_train_ids")]
    sources += [f"metadata/dish_metadata_cafe{cafe}.csv" for cafe in (1, 2)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        paths = list(pool.map(lambda name: download(name, data_dir / name), sources))
    split_sets = [set(path.read_text().split()) for path in paths[:3]]
    metadata = {}
    for path in paths[3:]:
        parsed = parse_metadata(path.read_text())
        if metadata.keys() & parsed.keys():
            raise ValueError("Duplicate dish IDs across cafes")
        metadata.update(parsed)
    selected, eligible_count = select_dishes(*split_sets, metadata, count, seed)
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["dish_ids"] != selected:
            raise ValueError("Existing manifest differs; use another data directory")
    with ThreadPoolExecutor(max_workers=6) as pool:
        images = list(pool.map(
            lambda dish: download(f"imagery/realsense_overhead/{dish}/rgb.png",
                                  data_dir / "images" / f"{dish}.png"), selected))
    from PIL import Image
    for image in images:
        with Image.open(image) as opened:
            opened.verify()
    manifest = {
        "dataset": "Nutrition5k", "seed": seed, "dish_ids": selected,
        "selection": "seeded sample from official rgb_test_ids intersect depth_test_ids",
        "eligible_dishes": eligible_count, "image_view": "one overhead RGB image per dish; no depth input",
        "source_url": BASE_URL,
        "source_sha256": {name: digest(path) for name, path in zip(sources, paths)},
        "image_sha256": {dish: digest(path) for dish, path in zip(selected, images)},
        "references": {dish: metadata[dish] for dish in selected},
    }
    write_json(manifest_path, manifest)
    return manifest


class RequestLimiter:
    def __init__(self, interval):
        self.interval = interval
        self.last_request = 0.0
        self.lock = threading.Lock()
        self.fatal_error = None

    def wait(self):
        with self.lock:
            if self.fatal_error:
                raise RuntimeError(self.fatal_error)
            delay = self.interval - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            self.last_request = time.monotonic()


class APITracker:
    def __init__(self, interval, max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS, reasoning_effort=OPENROUTER_REASONING_EFFORT,
                 limiter=None):
        self.interval = interval
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.limiter = limiter or RequestLimiter(interval)
        self.calls = []
        self.http_requests = 0
        self.fatal_error = None

    def before_request(self, request):
        self.limiter.wait()
        self.http_requests += 1

    def client(self, key):
        import httpx
        from openai import OpenAI
        client = OpenAI(
            api_key=key, base_url="https://openrouter.ai/api/v1", timeout=90,
            max_retries=2,
            http_client=httpx.Client(event_hooks={"request": [self.before_request]}),
        )

        def create(**kwargs):
            kwargs["max_tokens"] = self.max_tokens
            if self.reasoning_effort:
                kwargs["extra_body"] = {**kwargs.get("extra_body", {}),
                                        **openrouter_reasoning(self.reasoning_effort)}
            try:
                completion = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if getattr(exc, "status_code", None) in (401, 403, 404) or (
                        getattr(exc, "status_code", None) == 429 and
                        any(word in str(exc).lower() for word in ("daily", "day", "quota", "credits"))):
                    self.fatal_error = str(exc)
                    self.limiter.fatal_error = self.fatal_error
                raise
            self.calls.append({"model": completion.model,
                               "finish_reason": completion.choices[0].finish_reason,
                               "usage": completion.usage.model_dump() if completion.usage else None})
            return completion
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def prediction_values(result):
    if result.error:
        raise ValueError(result.error)
    if not result.food_items:
        raise ValueError("No food predictions")
    for item in result.food_items:
        values = (item.quantity_grams, item.calories, item.fat_g, item.carbs_g, item.protein_g)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
               for value in values):
            raise ValueError("Prediction contains invalid numeric values")
    result.compute_totals()
    return dict(zip(TARGETS, (result.total_calories, sum(item.quantity_grams for item in result.food_items),
                             result.total_fat_g, result.total_carbs_g, result.total_protein_g)))


def summarize(records, manifest, methods, seed):
    summary = {}
    for method in methods:
        valid = [record for record in records[method].values() if not record.get("error")]
        statistics = {}
        for target in TARGETS:
            truth = np.array([manifest["references"][r["dish_id"]][target] for r in valid])
            predicted = np.array([r["prediction"][target] for r in valid])
            if len(valid):
                errors = predicted - truth
                mae = float(np.abs(errors).mean())
                statistics[target] = {"mae": mae, "mean_signed_error": float(errors.mean()),
                                      "normalized_mae_percent": 100 * mae / float(truth.mean())
                                      if truth.mean() else None}
        summary[method] = {
            "attempted": len(records[method]), "successful": len(valid),
            "failures": sum(bool(r.get("error")) for r in records[method].values()),
            "fallback_predictions": sum(r.get("used_fallback", False) for r in valid),
            "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in valid])) if valid else None,
            "http_requests": sum(r.get("http_requests", 0) for r in records[method].values()),
            "errors": statistics,
        }
    common = [dish for dish in manifest["dish_ids"]
              if all(dish in records[method] and not records[method][dish].get("error") for method in methods)]
    paired = {}
    if common:
        indices = np.random.default_rng(seed).integers(len(common), size=(2000, len(common)))
        for first, second in combinations(methods, 2):
            differences = np.array([
                abs(records[first][dish]["prediction"]["calories"] - manifest["references"][dish]["calories"])
                - abs(records[second][dish]["prediction"]["calories"] - manifest["references"][dish]["calories"])
                for dish in common])
            paired[f"{first} minus {second}"] = {
                "calorie_mae_difference": float(differences.mean()),
                "bootstrap_95_percent_interval": np.quantile(differences[indices].mean(axis=1), [0.025, 0.975]).tolist(),
            }
    return {"methods": summary, "common_successful_dishes": len(common), "paired_calorie_comparisons": paired}


def nutrient_comparisons(records, manifest, seed):
    rng = np.random.default_rng(seed)
    result = {"methods": {}, "paired": {}}
    for method, predictions in records.items():
        valid = [dish for dish in manifest["dish_ids"] if dish in predictions and not predictions[dish].get("error")]
        statistics = {}
        if valid:
            indices = rng.integers(len(valid), size=(2000, len(valid)))
            true_mass = np.array([manifest["references"][dish]["mass_g"] for dish in valid])
            predicted_mass = np.array([predictions[dish]["prediction"]["mass_g"] for dish in valid])
            for target in TARGETS:
                truth = np.array([manifest["references"][dish][target] for dish in valid])
                predicted = np.array([predictions[dish]["prediction"][target] for dish in valid])
                errors = predicted - truth
                absolute = np.abs(errors)
                denominator = float(np.sum((truth - truth.mean()) ** 2))
                statistics[target] = {
                    "n": len(valid), "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "r2": 1 - float(np.sum(errors ** 2)) / denominator if denominator else None,
                    "mae_bootstrap_95_percent_interval": np.quantile(absolute[indices].mean(axis=1), [0.025, 0.975]).tolist()}
                if target != "mass_g":
                    eligible = (predicted_mass > 0) & (true_mass > 0)
                    statistics[target]["oracle_mass_diagnostic_n"] = int(eligible.sum())
                    statistics[target]["oracle_mass_diagnostic_mae"] = float(np.abs(
                        predicted[eligible] * true_mass[eligible] / predicted_mass[eligible] - truth[eligible]).mean()) if eligible.any() else None
        result["methods"][method] = {"complete": len(predictions) == len(manifest["dish_ids"]),
                                    "missing_dishes": len(manifest["dish_ids"]) - len(predictions),
                                    "nutrients": statistics}
    for first, second in combinations(records, 2):
        common = [dish for dish in manifest["dish_ids"] if all(
            dish in records[method] and not records[method][dish].get("error") for method in (first, second))]
        pair = {"n": len(common), "nutrients": {}}
        if common:
            indices = rng.integers(len(common), size=(2000, len(common)))
            for target in TARGETS:
                differences = np.array([
                    abs(records[first][dish]["prediction"][target] - manifest["references"][dish][target])
                    - abs(records[second][dish]["prediction"][target] - manifest["references"][dish][target])
                    for dish in common])
                pair["nutrients"][target] = {"mae_difference": float(differences.mean()),
                    "bootstrap_95_percent_interval": np.quantile(differences[indices].mean(axis=1), [0.025, 0.975]).tolist()}
        result["paired"][f"{first} minus {second}"] = pair
    return result


def run_evaluation(args, manifest):
    workers = getattr(args, "workers", 1)
    if workers < 1:
        raise ValueError("workers must be positive")
    if set(args.methods) & API_METHODS and not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY required for API-based methods")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    signature = {"manifest_sha256": digest(args.data_dir / "manifest.json"), "model": args.model,
                 "max_output_tokens": args.max_output_tokens, "reasoning_effort": args.reasoning_effort,
                 "workers": workers, "request_interval": args.request_interval,
                 "shared_config_sha256": digest(FOOD_VISION_DIR.parents[1] / "shared/config.py"),
                 "rf_detr_checkpoint_sha256": digest(DEFAULT_MODEL_PATH) if "rf_detr" in args.methods else None,
                 "rf_detr_labels_sha256": digest(DEFAULT_MODEL_PATH.parent / "food_classes.json") if "rf_detr" in args.methods else None,
                 "rgb_checkpoint_sha256": digest(getattr(args, "rgb_checkpoint", RGB_CHECKPOINT)) if "rgb_regression" in args.methods else None,
                 "food_r1": {"model": FOOD_R1_MODEL, "revision": FOOD_R1_REVISION,
                             "max_new_tokens": 512, "do_sample": False, "max_image_side": 1024} if "food_r1" in args.methods else None,
                 "device": getattr(args, "device", None),
                 "code_sha256": {name: digest(FOOD_VISION_DIR / name) for name in
                                  ("rf_detr_analyzer.py", "vlm_analyzer.py", "clip_analyzer.py", "rag_vlm_analyzer.py", "grounded_vlm_analyzer.py", "geometry_analyzer.py", "supervised_analyzers.py", "nutrition_db.py", "base.py", "evaluate_nutrition5k.py")}}
    signature_path = args.output_dir / "configuration.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError("Output configuration differs; use another output directory")
    write_json(signature_path, signature)
    records = {}
    initialization_path = args.output_dir / "initialization.json"
    initialization = json.loads(initialization_path.read_text()) if initialization_path.exists() else {}
    limiter = RequestLimiter(args.request_interval)
    classes = get_available_methods()
    if "rgb_regression" in args.methods:
        import torch
        checkpoint = torch.load(getattr(args, "rgb_checkpoint", RGB_CHECKPOINT), map_location="cpu", weights_only=True)
        provenance = checkpoint["provenance"]
        official_train = set((args.data_dir / "dish_ids/splits/rgb_train_ids.txt").read_text().split())
        validate_regression_split(provenance, official_train, manifest["dish_ids"])
    for method in args.methods:
        path = args.output_dir / f"{method}.json"
        records[method] = json.loads(path.read_text()) if path.exists() else {}
        if all(dish in records[method] and not (args.retry_errors and records[method][dish].get("error"))
               for dish in manifest["dish_ids"]):
            continue
        local = threading.local()
        shared_clip = None
        shared_geometry = None
        classification_lock = threading.Lock()
        if method == "clip_ensemble":
            started = time.perf_counter()
            shared_clip = classes[method](llm_model=args.model, reasoning_effort=args.reasoning_effort)
            shared_clip._load_clip()
            initialization[method] = initialization.get(method, 0.0) + time.perf_counter() - started
        if method == "geometry_vlm_db":
            from nutricoach.food_vision.geometry_analyzer import RGBGeometryEstimator
            shared_geometry = RGBGeometryEstimator()

        def classify(image_path):
            with classification_lock:
                return shared_clip._classify_food(image_path)

        def predict(dish):
            initialization_seconds = 0.0
            if not hasattr(local, "analyzer"):
                started = time.perf_counter()
                local.tracker = APITracker(args.request_interval, args.max_output_tokens, args.reasoning_effort,
                                           limiter=limiter)
                local.analyzer = classes[method](**({"checkpoint": getattr(args, "rgb_checkpoint", RGB_CHECKPOINT),
                                                     "device": getattr(args, "device", None)} if method == "rgb_regression" else
                                                    {"device": getattr(args, "device", None)} if method == "food_r1" else
                                                    {} if method == "rf_detr" else
                                                    {"llm_model": args.model, "reasoning_effort": args.reasoning_effort}
                                                    if method == "clip_ensemble" else
                                                    {"model": args.model, "reasoning_effort": args.reasoning_effort}))
                if method in ("rf_detr", "rgb_regression", "food_r1"):
                    local.analyzer._load_model()
                elif method == "clip_ensemble":
                    local.analyzer._classify_food = classify
                    local.analyzer._refinement_client = local.tracker.client(os.environ["OPENROUTER_API_KEY"])
                else:
                    local.analyzer._client = local.tracker.client(os.environ["OPENROUTER_API_KEY"])
                    if shared_geometry is not None:
                        local.analyzer.geometry_estimator = shared_geometry
                initialization_seconds = time.perf_counter() - started
            tracker, analyzer = local.tracker, local.analyzer
            before, calls_before = tracker.http_requests, len(tracker.calls)
            started = time.perf_counter()
            record = {"dish_id": dish}
            try:
                result = analyzer.analyze(str(args.data_dir / "images" / f"{dish}.png"))
                record.update(prediction=prediction_values(result), result=result.to_dict(),
                              raw_response=result.raw_response,
                              used_fallback=any("fallback" in item.portion_description for item in result.food_items))
            except Exception as exc:
                record["error"] = str(exc)
            record.update(inference_seconds=time.perf_counter() - started,
                          http_requests=tracker.http_requests - before, api_calls=tracker.calls[calls_before:])
            return dish, record, initialization_seconds, tracker.fatal_error

        pending = [dish for dish in manifest["dish_ids"]
                   if dish not in records[method] or (args.retry_errors and records[method][dish].get("error"))]
        initialization.setdefault(method, 0.0)
        with ThreadPoolExecutor(max_workers=workers if method in API_METHODS else 1) as pool:
            for index, (dish, record, elapsed, fatal_error) in enumerate(pool.map(predict, pending), 1):
                initialization[method] += elapsed
                write_json(initialization_path, initialization)
                records[method][dish] = record
                write_json(path, records[method])
                print(f"{method}: {index}/{len(pending)} {dish} "
                      f"{'ERROR: ' + record['error'][:120] if record.get('error') else 'OK'}", flush=True)
                if fatal_error:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise RuntimeError(f"API configuration prevents this benchmark: {fatal_error}")
    summary = summarize(records, manifest, args.methods, args.seed)
    summary["nutrient_comparisons"] = nutrient_comparisons(records, manifest, args.seed)
    summary.update(configuration=signature, initialization_seconds=initialization,
                   finished_utc=datetime.now(timezone.utc).isoformat(),
                   caveats=["50-dish pilot; overhead RGB subset only", "Serving priors are not visual mass estimates",
                            "Success-only errors must be read alongside failures", "No ingredient recognition scoring"])
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["methods"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=os.getenv("OPENROUTER_MODEL_ID", OPENROUTER_MODEL_ID))
    parser.add_argument("--methods", nargs="+", choices=METHODS + LEGACY_METHODS, default=list(METHODS))
    parser.add_argument("--rgb-checkpoint", type=Path, default=RGB_CHECKPOINT)
    parser.add_argument("--device", help="Local inference device, e.g. cpu or cuda:0")
    parser.add_argument("--request-interval", type=float, default=3.2)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel API predictions with a global request interval; local inference is serialized")
    parser.add_argument("--max-output-tokens", type=int, default=OPENROUTER_MAX_OUTPUT_TOKENS)
    parser.add_argument("--reasoning-effort", choices=OPENROUTER_REASONING_EFFORTS, default=OPENROUTER_REASONING_EFFORT)
    parser.add_argument("--data-dir", type=Path, default=FOOD_VISION_DIR / "data" / "nutrition5k_50")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = FOOD_VISION_DIR / "results" / f"nutrition5k_{args.count}_{args.reasoning_effort}_reasoning"
    manifest = prepare_dataset(args.data_dir, args.count, args.seed)
    print(f"Prepared {len(manifest['dish_ids'])} official-test dishes", flush=True)
    if not args.prepare_only:
        run_evaluation(args, manifest)


if __name__ == "__main__":
    main()
