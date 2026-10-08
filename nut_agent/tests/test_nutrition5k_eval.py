"""Evaluation split and metric correctness without downloads or inference."""

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.food_vision.evaluate_nutrition5k import APITracker, RequestLimiter, run_evaluation, parse_metadata, select_dishes, summarize, prediction_values
from nutricoach.food_vision.base import FoodAnalysisResult, FoodItem
from nutricoach.food_vision.rag_vlm_analyzer import RAGVLMAnalyzer


def test_released_metadata_column_order():
    metadata = parse_metadata("dish_1,250,200,10,30,15,ingr_1,rice,100,130,1,28,3\n")
    assert metadata["dish_1"] == {"calories": 250, "mass_g": 200, "fat_g": 10, "carbs_g": 30, "protein_g": 15}


def test_selection_is_reproducible_and_excludes_train_and_missing_overhead():
    rgb = {f"dish_{i}" for i in range(100)}
    depth = {f"dish_{i}" for i in range(70)}
    metadata = {dish: {} for dish in rgb}
    selection, eligible = select_dishes(rgb, depth, {"dish_train"}, metadata, 50, 42)
    assert eligible == 70
    assert len(set(selection)) == 50
    assert set(selection) <= depth
    assert selection == select_dishes(rgb, depth, {"dish_train"}, metadata, 50, 42)[0]
    with pytest.raises(ValueError, match="overlap"):
        select_dishes(rgb, depth, {"dish_1"}, metadata, 50, 42)


def test_failures_are_not_silently_scored_as_zero():
    manifest = {"dish_ids": ["a", "b"], "references": {
        "a": {"calories": 100, "mass_g": 100, "fat_g": 10, "carbs_g": 20, "protein_g": 10},
        "b": {"calories": 200, "mass_g": 200, "fat_g": 20, "carbs_g": 40, "protein_g": 20},
    }}
    record = {"dish_id": "a", "prediction": manifest["references"]["a"] | {"calories": 120}, "inference_seconds": 1}
    records = {"first": {"a": record, "b": {"error": "API failed"}}, "second": {"a": record}}
    summary = summarize(records, manifest, list(records), 42)
    assert summary["methods"]["first"]["failures"] == 1
    assert summary["methods"]["first"]["errors"]["calories"]["mae"] == 20
    assert summary["common_successful_dishes"] == 1


def test_prediction_mass_is_sum_of_food_weights():
    result = FoodAnalysisResult(method="test", food_items=[FoodItem("rice", 100, calories=130), FoodItem("egg", 50, calories=77.5)])
    assert prediction_values(result)["mass_g"] == 150
    result.food_items[0].calories = float("nan")
    with pytest.raises(ValueError, match="invalid numeric"):
        prediction_values(result)


def test_normalized_error_uses_mean_reference_and_pairing_excludes_failures():
    references = {dish: dict.fromkeys(("calories", "mass_g", "fat_g", "carbs_g", "protein_g"), value)
                  for dish, value in (("a", 100), ("b", 300))}
    manifest = {"dish_ids": ["a", "b"], "references": references}
    records = {"first": {}, "second": {}}
    for dish in references:
        records["first"][dish] = {"dish_id": dish, "prediction": references[dish] | {"calories": 200},
                                  "inference_seconds": 1}
        records["second"][dish] = {"dish_id": dish, "prediction": references[dish], "inference_seconds": 1}
    summary = summarize(records, manifest, list(records), 42)
    assert summary["methods"]["first"]["errors"]["calories"] == {
        "mae": 100, "mean_signed_error": 0, "normalized_mae_percent": 50}
    paired = summary["paired_calorie_comparisons"]["first minus second"]
    assert paired["calorie_mae_difference"] == 100
    assert paired["bootstrap_95_percent_interval"] == [100, 100]
    records["second"]["b"] = {"dish_id": "b", "error": "failed"}
    assert summarize(records, manifest, list(records), 42)["common_successful_dishes"] == 1


def test_portion_adjustment_scales_nutrition_consistently():
    item = {"name": "rice", "quantity_grams": 10, "calories": 13, "protein_g": 0.27, "carbs_g": 2.8, "fat_g": 0.03}
    adjusted = RAGVLMAnalyzer()._cross_validate_portions([item])[0]
    assert adjusted["quantity_grams"] == 130
    assert adjusted["calories"] == pytest.approx(169)
    assert adjusted["protein_g"] == pytest.approx(3.51)


@pytest.mark.parametrize("effort", ["low", "high"])
def test_api_settings_are_uniform_and_recorded(monkeypatch, effort):
    import openai

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model="resolved-model", usage=None,
                               choices=[SimpleNamespace(finish_reason="stop")])

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    tracker = APITracker(0, reasoning_effort=effort)
    tracker.client("test-key").chat.completions.create(model="test-model", max_tokens=500)
    assert captured["max_tokens"] == 8192
    assert captured["extra_body"] == {"reasoning": {"effort": effort}}
    assert tracker.calls == [{"model": "resolved-model", "finish_reason": "stop", "usage": None}]


def test_parallel_evaluation_keeps_usage_with_its_image(tmp_path, monkeypatch):
    import time

    dishes = [f"dish_{index}" for index in range(6)]
    manifest = {"dish_ids": dishes, "references": {
        dish: dict.fromkeys(("calories", "mass_g", "protein_g", "carbs_g", "fat_g"), 100)
        for dish in dishes}}
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps(manifest))

    def client(tracker, key):
        def create(image):
            tracker.before_request(None)
            tracker.calls.append({"image": image})
            time.sleep(0.01)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    class Analyzer:
        def __init__(self, **kwargs):
            pass

        def analyze(self, image):
            self._client.chat.completions.create(image=image)
            return FoodAnalysisResult(method="vlm_single", food_items=[FoodItem("rice", 100, calories=130)])

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setattr(APITracker, "client", client)
    monkeypatch.setattr("nutricoach.food_vision.evaluate_nutrition5k.get_available_methods",
                        lambda: {"vlm_single": Analyzer})
    monkeypatch.setattr("nutricoach.food_vision.evaluate_nutrition5k.DEFAULT_MODEL_PATH", tmp_path / "missing.pth")
    args = SimpleNamespace(workers=3, methods=["vlm_single"], data_dir=data_dir, output_dir=tmp_path / "results",
                           model="test", max_output_tokens=8192, reasoning_effort="high", request_interval=0,
                           retry_errors=False, seed=42)
    run_evaluation(args, manifest)
    records = json.loads((args.output_dir / "vlm_single.json").read_text())
    assert set(records) == set(dishes)
    for dish, record in records.items():
        assert not record.get("error")
        assert record["http_requests"] == 1
        assert record["api_calls"] == [{"image": str(data_dir / "images" / f"{dish}.png")}]
    run_evaluation(args, manifest)
    assert json.loads((args.output_dir / "vlm_single.json").read_text()) == records


def test_request_limiter_is_shared_by_trackers(monkeypatch):
    limiter = RequestLimiter(0)
    calls = []
    monkeypatch.setattr(limiter, "wait", lambda: calls.append(1))
    trackers = [APITracker(0, limiter=limiter) for _ in range(3)]
    for tracker in trackers:
        tracker.before_request(None)
    assert len(calls) == 3
    assert [tracker.http_requests for tracker in trackers] == [1, 1, 1]


@pytest.mark.parametrize("filename", ["nutrition5k_test50_results.json", "nutrition5k_test50_high_results.json"])
def test_published_snapshot_metrics_match_predictions(filename):
    path = Path(__file__).parent.parent / "nutricoach/food_vision" / filename
    snapshot = json.loads(path.read_text())
    manifest = snapshot["manifest"]
    assert len(set(manifest["dish_ids"])) == 50
    records = {method: {dish: {"dish_id": dish, **record} for dish, record in predictions.items()}
               for method, predictions in snapshot["predictions"].items()}
    for predictions in records.values():
        assert set(predictions) == set(manifest["dish_ids"])
    computed = summarize(records, manifest, list(records), manifest["seed"])
    for key, value in computed.items():
        assert value == snapshot["summary"][key]


def test_high_snapshot_preserves_low_and_paired_comparisons():
    import hashlib
    import numpy as np

    directory = Path(__file__).parent.parent / "nutricoach/food_vision"
    low_path = directory / "nutrition5k_test50_results.json"
    low = json.loads(low_path.read_text())
    high = json.loads((directory / "nutrition5k_test50_high_results.json").read_text())
    assert hashlib.sha256(low_path.read_bytes()).hexdigest() == high["low_snapshot_sha256"]
    assert low["manifest"] == high["manifest"]
    for key in ("model", "max_output_tokens", "rf_detr_checkpoint_sha256"):
        assert low["summary"]["configuration"][key] == high["summary"]["configuration"][key]
    assert low["summary"]["configuration"]["reasoning_effort"] == "low"
    assert high["summary"]["configuration"]["reasoning_effort"] == "high"
    for method, comparison in high["reasoning_comparison"].items():
        common = [dish for dish in high["manifest"]["dish_ids"]
                  if not low["predictions"][method][dish].get("error")
                  and not high["predictions"][method][dish].get("error")]
        errors = {}
        for name, snapshot in (("low", low), ("high", high)):
            errors[name] = np.array([abs(snapshot["predictions"][method][dish]["prediction"]["calories"]
                                        - high["manifest"]["references"][dish]["calories"]) for dish in common])
            assert errors[name].mean() == comparison[f"{name}_calorie_mae"]
        differences = errors["high"] - errors["low"]
        indices = np.random.default_rng(42).integers(len(common), size=(2000, len(common)))
        assert len(common) == comparison["common_successful_dishes"]
        assert differences.mean() == comparison["high_minus_low_calorie_mae"]
        assert np.quantile(differences[indices].mean(axis=1), [0.025, 0.975]).tolist() == comparison["bootstrap_95_percent_interval"]
