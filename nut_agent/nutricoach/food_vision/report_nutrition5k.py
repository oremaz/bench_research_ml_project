"""Combine independently executed methods on one frozen Nutrition5k manifest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nutricoach.food_vision.evaluate_nutrition5k import (
    FOOD_VISION_DIR, METHODS, API_METHODS, digest, write_json, summarize, nutrient_comparisons)
from shared.config import OPENROUTER_MODEL_ID


def build_report(manifest_path, run_dirs, model=OPENROUTER_MODEL_ID):
    manifest = json.loads(manifest_path.read_text())
    records = {method: {} for method in METHODS}
    configurations = {}
    api_settings = None
    for directory in run_dirs:
        configuration = json.loads((directory / "configuration.json").read_text())
        if configuration["manifest_sha256"] != digest(manifest_path):
            raise ValueError(f"Different frozen test manifest: {directory}")
        for method in METHODS:
            path = directory / f"{method}.json"
            if not path.exists():
                continue
            if method in API_METHODS and configuration["model"] != model:
                raise ValueError(f"Different API model for {method}: {directory}")
            if method in API_METHODS:
                settings = {key: configuration.get(key) for key in ("model", "max_output_tokens", "reasoning_effort")}
                if api_settings is not None and settings != api_settings:
                    raise ValueError(f"Different API reasoning or token budget: {directory}")
                api_settings = settings
            if method in configurations:
                raise ValueError(f"Multiple runs supplied for {method}; select one explicitly")
            predictions = json.loads(path.read_text())
            if set(predictions) - set(manifest["dish_ids"]):
                raise ValueError(f"Unexpected test dish in {directory}")
            for dish, prediction in predictions.items():
                if prediction["dish_id"] != dish:
                    raise ValueError(f"Mismatched prediction dish ID in {directory}")
            records[method] = predictions
            configurations[method] = configuration
    summary = summarize(records, manifest, METHODS, manifest["seed"])
    summary["nutrient_comparisons"] = nutrient_comparisons(records, manifest, manifest["seed"])
    compact = {method: {dish: {key: value for key, value in record.items() if key in
                              ("dish_id", "prediction", "error", "inference_seconds", "used_fallback", "http_requests")}
                        for dish, record in predictions.items()} for method, predictions in records.items()}
    return {"protocol": "RGB-only official-test pilot; one fixed image per dish; five targets",
            "manifest": manifest, "configurations": configurations, "predictions": compact, "summary": summary}


def markdown_report(report):
    lines = ["# Nutrition5k: current RGB-only benchmark", "",
             "All methods use the same frozen 50-dish official-test RGB subset. Pending methods have no measured score.", "",
             "| Method | Successes / 50 | Calories MAE (kcal) | Mass MAE (g) | Fat MAE (g) | Carbs MAE (g) | Protein MAE (g) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        statistics = report["summary"]["methods"][method]
        errors = statistics["errors"]
        row = [f"{errors[target]['mae']:.1f}" if target in errors else "pending" for target in
               ("calories", "mass_g", "fat_g", "carbs_g", "protein_g")]
        lines.append(f"| {method} | {statistics['successful']} / 50 | " + " | ".join(row) + " |")
    lines.extend(["", "MAE is computed on successful predictions; consult JSON for failures, missing dishes, signed errors,",
                  "normalized MAE, RMSE, R2, dish-bootstrap confidence intervals and paired comparisons for all five targets.",
                  "Oracle-mass errors are post-hoc diagnostics using reference masses, never deployable RGB-only scores.", "",
                  "Configurations and original model IDs are preserved in the JSON. Published paper scores are not local measurements."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=FOOD_VISION_DIR / "data/nutrition5k_50/manifest.json")
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--model", default=OPENROUTER_MODEL_ID)
    parser.add_argument("--output", type=Path, default=FOOD_VISION_DIR / "nutrition5k_dots_results.json")
    args = parser.parse_args()
    report = build_report(args.manifest, args.run_dir, args.model)
    write_json(args.output, report)
    args.output.with_suffix(".md").write_text(markdown_report(report))
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
