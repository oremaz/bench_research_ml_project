"""Combine independently executed methods on one frozen Nutrition5k manifest."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nutricoach.food_vision.evaluate_nutrition5k import (
    FOOD_VISION_DIR, METHODS, API_METHODS, digest, write_json, summarize, nutrient_comparisons)
from shared.config import OPENROUTER_MODEL_ID

REPORT_METHODS = (*METHODS, "foodlmm", "foodlmm_cached", "foodlmm_4bit", "dpf_reproduction")


def build_report(manifest_path, run_dirs, model=OPENROUTER_MODEL_ID):
    manifest = json.loads(manifest_path.read_text())
    records = {method: {} for method in REPORT_METHODS}
    configurations = {}
    execution_status = {}
    resources = {}
    api_settings = None
    for directory in run_dirs:
        configuration = json.loads((directory / "configuration.json").read_text())
        if configuration["manifest_sha256"] != digest(manifest_path):
            raise ValueError(f"Different frozen test manifest: {directory}")
        for method in REPORT_METHODS:
            path = directory / f"{method}.json"
            if not path.exists():
                continue
            if method == "foodlmm" and configuration.get("generation_use_cache"):
                method = "foodlmm_cached"
            if method in API_METHODS and configuration["model"] != model:
                raise ValueError(f"Different API model for {method}: {directory}")
            if method in API_METHODS:
                settings = {key: configuration.get(key) for key in ("model", "max_output_tokens", "reasoning_effort", "local_model_revision", "local_quantization")}
                settings["provider"] = configuration.get("provider", "openrouter")
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
            resource_path = directory / "resources.json"
            if resource_path.exists():
                resources[method] = json.loads(resource_path.read_text())
            status_path = directory / "execution_status.json"
            execution_status[method] = json.loads(status_path.read_text()) if status_path.exists() else {
                "state": "all dishes attempted" if len(predictions) == len(manifest["dish_ids"]) else "incomplete"}
            if len(predictions) == len(manifest["dish_ids"]):
                execution_status[method]["state"] = "all dishes attempted"
    summary = summarize(records, manifest, REPORT_METHODS, manifest["seed"])
    summary["nutrient_comparisons"] = nutrient_comparisons(records, manifest, manifest["seed"])
    compact = {method: {dish: {key: value for key, value in record.items() if key in
                              ("dish_id", "prediction", "error", "inference_seconds", "used_fallback", "http_requests")}
                        for dish, record in predictions.items()} for method, predictions in records.items()}
    return {"protocol": "RGB-only official-test evaluation; one fixed image per dish; five targets",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "execution_status": execution_status,
            "manifest": manifest, "configurations": configurations, "predictions": compact,
            "resources": resources, "summary": summary}


def markdown_report(report):
    count = len(report["manifest"]["dish_ids"])
    lines = ["# Nutrition5k: current RGB-only benchmark", "",
             f"All methods use the same frozen {count}-dish official-test RGB subset. Pending methods have no measured score.", "",
             "| Method | Calories MAE (kcal) | Mass MAE (g) | Fat MAE (g) | Carbs MAE (g) | Protein MAE (g) |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in report["summary"]["methods"]:
        statistics = report["summary"]["methods"][method]
        errors = statistics["errors"]
        row = [f"{errors[target]['mae']:.1f}" if target in errors else "pending" for target in
               ("calories", "mass_g", "fat_g", "carbs_g", "protein_g")]
        label = {"foodlmm": "FoodLMM (16-bit)", "foodlmm_cached": "FoodLMM (16-bit, cached)",
                 "foodlmm_4bit": "FoodLMM (4-bit decoder)"}.get(method, method)
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines.extend(["", "MAE is computed on successful predictions; consult JSON for failures, missing dishes, signed errors,",
                  "normalized MAE, RMSE, R2, dish-bootstrap confidence intervals and paired comparisons for all five targets.",
                  "Oracle-mass errors are post-hoc diagnostics using reference masses, never deployable RGB-only scores.", "",
                  "Configurations and original model IDs are preserved in the JSON. Published paper scores are not local measurements."])
    lines.extend(["", "The FoodLMM 4-bit variant quantizes decoder linear weights with Quanto qint4;",
                  "vision, embeddings, language output and nutrition heads remain at 16-bit precision on GPU.",
                  "The original 16-bit result is retained. The cached 16-bit control and 4-bit variant",
                  "use the same cached generation and full-prefix nutrition-head pass."])
    if report.get("resources"):
        lines.extend(["", "## Measured compute resources", "",
            "GPU peaks include model weights. Loading and inference peaks are recorded separately;",
            "host RSS is the process lifetime peak. Latency uses device synchronization and excludes model loading.", "",
            "| Method | Device | Mean inference (s) | Inference peak allocated (GiB) | Inference peak reserved (GiB) | Loading peak allocated (GiB) | Peak host RSS (GiB) |",
            "|---|---|---:|---:|---:|---:|---:|"])
        for method, values in report["resources"].items():
            inference = values.get("inference_memory", {})
            loading = values.get("loading_memory", {})
            statistics = report["summary"]["methods"][method]
            numbers = [statistics["mean_inference_seconds"], inference.get("gpu_peak_allocated_gib"),
                       inference.get("gpu_peak_reserved_gib"), loading.get("gpu_peak_allocated_gib"),
                       values.get("peak_host_rss_gib")]
            cells = [f"{number:.2f}" if number is not None else "not measured" for number in numbers]
            lines.append(f"| {method} | {values['device']} | " + " | ".join(cells) + " |")
    vision = report.get("configurations", {}).get("vlm_single")
    if vision:
        lines.extend(["", f"VLM provider: `{vision.get('provider', 'openrouter')}`; model: `{vision['model']}`; "
                      f"reasoning: `{vision.get('reasoning_effort', 'unspecified')}`."])
        if vision.get("provider") == "local":
            lines.append(f"Local revision: `{vision['local_model_revision']}`; quantization: `{vision['local_quantization']}`.")
    lines.extend(["", "## Run coverage", ""])
    for method, statistics in report["summary"]["methods"].items():
        state = report.get("execution_status", {}).get(method, {}).get("state", "not started")
        lines.append(f"- `{method}`: {statistics['attempted']} attempted, {statistics['failures']} failed, "
                     f"{count - statistics['attempted']} not attempted, {statistics['fallback_predictions']} fallback predictions; {state}.")
    lines.extend(["", "Partial runs must be compared on common successful dishes. An incomplete one-shot baseline",
                  f"cannot establish a winner for the full {count}-dish evaluation.", "",
                  "FoodLMM uses the author's Chat checkpoint; DPF uses a third-party reproduction with",
                  "Depth Anything V2 and modified fusion. See [methods and limitations](README.md#methods-in-practice)."])
    pairs = report["summary"]["nutrient_comparisons"]["paired"]
    rows = []
    for method in report["summary"]["methods"]:
        pair = pairs.get(f"vlm_single minus {method}")
        if pair and pair["n"]:
            statistics = pair["nutrients"]["calories"]
            low, high = statistics["bootstrap_95_percent_interval"]
            rows.append(f"| {method} | {pair['n']} | {-statistics['mae_difference']:.1f} | [{-high:.1f}, {-low:.1f}] |")
    if rows:
        lines.extend(["", "## Paired calorie comparisons against one-shot", "",
            "Differences are method MAE minus one-shot MAE on the same successful dishes; negative is better.",
            "Comparisons with missing or failed predictions apply only to the common successful subset.", "",
            "| Method | Common dishes | MAE difference (kcal) | Dish-bootstrap 95% interval (kcal) |",
            "|---|---:|---:|---:|", *rows])
    precision_base = "foodlmm_cached" if "foodlmm_cached" in report["configurations"] else "foodlmm"
    precision_pair = pairs.get(f"{precision_base} minus foodlmm_4bit")
    if precision_pair and precision_pair["n"]:
        lines.extend(["", "## FoodLMM precision comparison", "",
            f"Differences are 4-bit MAE minus `{precision_base}` MAE on common successful dishes; positive is worse.", "",
            "| Target | Common dishes | MAE difference | Dish-bootstrap 95% interval |",
            "|---|---:|---:|---:|"])
        for target, statistics in precision_pair["nutrients"].items():
            low, high = statistics["bootstrap_95_percent_interval"]
            lines.append(f"| {target} | {precision_pair['n']} | {-statistics['mae_difference']:.2f} | [{-high:.2f}, {-low:.2f}] |")
        calorie_change = -precision_pair["nutrients"]["calories"]["mae_difference"]
        failures = report["summary"]["methods"]["foodlmm_4bit"]["failures"]
        control_failures = report["summary"]["methods"][precision_base]["failures"]
        lines.extend(["", f"The paired calorie MAE change is {calorie_change:+.2f} kcal. "
            f"The 4-bit run has {failures}/{count} failures; its 16-bit control has {control_failures}/{count}.",
            "Method rows use different successful subsets; the precision comparison uses matched dishes."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=FOOD_VISION_DIR / "data/nutrition5k_500/manifest.json")
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--model", default=OPENROUTER_MODEL_ID)
    parser.add_argument("--output", type=Path, default=FOOD_VISION_DIR / "nutrition5k_500_qwen_results.json")
    args = parser.parse_args()
    report = build_report(args.manifest, args.run_dir, args.model)
    write_json(args.output, report)
    args.output.with_suffix(".md").write_text(markdown_report(report))
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
