"""Split isolation, train-only scaling, specialized output parsing, and reporting."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.food_vision.supervised_analyzers import TARGETS, parse_food_r1, total_result, validate_regression_split
from nutricoach.food_vision.train_rgb_regression import split_training_dishes, fit_head
from nutricoach.food_vision.evaluate_nutrition5k import nutrient_comparisons, APITracker, RequestLimiter
from nutricoach.food_vision.report_nutrition5k import build_report


def test_validation_groups_incremental_scans_by_acquisition_day():
    start = int(datetime(2019, 5, 1, tzinfo=timezone.utc).timestamp())
    dishes = {f"dish_{start + day * 86400 + second}" for day in range(10) for second in (10, 40, 80)}
    train, validation = split_training_dishes(dishes, dishes, {"dish_1"}, dict.fromkeys(dishes), seed=42)
    days = lambda ids: {int(dish[5:]) // 86400 for dish in ids}
    assert not days(train) & days(validation)
    assert set(train) | set(validation) == dishes
    assert (train, validation) == split_training_dishes(dishes, dishes, {"dish_1"}, dict.fromkeys(dishes), seed=42)
    with pytest.raises(ValueError, match="overlap"):
        split_training_dishes(dishes, dishes, dishes, dict.fromkeys(dishes))


def test_scalers_and_head_never_fit_validation_labels():
    features = np.arange(24).reshape(8, 3)
    labels = np.arange(40).reshape(8, 5)
    train, validation = np.arange(6), np.arange(6, 8)
    first = fit_head(features, labels, train, validation, alphas=(1,))
    changed = labels.copy()
    changed[validation] += 10000
    second = fit_head(features, changed, train, validation, alphas=(1,))
    assert first[0].mean_ == pytest.approx(features[train].mean(axis=0))
    assert first[1].mean_ == pytest.approx(labels[train].mean(axis=0))
    assert second[1].mean_ == pytest.approx(first[1].mean_)
    assert second[2].coef_ == pytest.approx(first[2].coef_)
    assert second[4] != first[4]


def test_checkpoint_must_have_disjoint_official_training_subsets():
    valid = {"train_ids": ["a"], "validation_ids": ["b"]}
    validate_regression_split(valid, {"a", "b"}, {"test"})
    for provenance in ({"train_ids": ["a"], "validation_ids": ["a"]},
                       {"train_ids": ["a"], "validation_ids": ["test"]},
                       {"train_ids": ["outside"], "validation_ids": ["b"]},
                       {"train_ids": [], "validation_ids": ["b"]}):
        with pytest.raises(ValueError, match="test separation"):
            validate_regression_split(provenance, {"a", "b"}, {"test"})


def test_food_r1_extracts_only_final_total_nutrition():
    text = "<think>99 kcal</think><answer>The dish weighs 210 g in total and provides about 350 kcal, including 12 g of fat, 40 g of carbohydrate, and 18 g of protein overall.</answer>"
    values = parse_food_r1(text)
    assert values == dict(zip(TARGETS, (350, 210, 12, 40, 18)))
    assert total_result("food_r1", values, 0).total_protein_g == 18
    for invalid in (text.replace("18 g of protein", "missing"), text.replace("350 kcal", "-1 kcal"), text + text,
                    text.replace("</answer>", "")):
        with pytest.raises(ValueError):
            parse_food_r1(invalid)


def test_all_nutrient_pairing_and_oracle_mass_diagnostic():
    truth = dict(zip(TARGETS, (100, 100, 10, 20, 15)))
    manifest = {"dish_ids": ["a", "b"], "references": {"a": truth, "b": truth}}
    records = {"first": {"a": {"prediction": {target: value * 2 for target, value in truth.items()}}},
               "second": {"a": {"prediction": truth}, "b": {"error": "failed"}}}
    summary = nutrient_comparisons(records, manifest, 42)
    assert summary["methods"]["first"]["missing_dishes"] == 1
    assert summary["methods"]["first"]["nutrients"]["calories"]["oracle_mass_diagnostic_mae"] == 0
    for target in TARGETS:
        assert summary["paired"]["first minus second"]["nutrients"][target]["mae_difference"] == truth[target]


def test_daily_api_quota_stops_shared_workers(monkeypatch):
    import openai

    class QuotaError(Exception):
        status_code = 429

    def create(**kwargs):
        raise QuotaError("Daily free model quota exhausted")

    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    limiter = RequestLimiter(0)
    tracker = APITracker(0, limiter=limiter)
    with pytest.raises(QuotaError):
        tracker.client("test").chat.completions.create()
    with pytest.raises(RuntimeError, match="quota exhausted"):
        APITracker(0, limiter=limiter).before_request(None)


def test_report_rejects_model_and_manifest_mismatch(tmp_path):
    from nutricoach.food_vision.evaluate_nutrition5k import digest

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"dish_ids": [], "references": {}, "seed": 42}))
    run = tmp_path / "run"
    run.mkdir()
    (run / "vlm_single.json").write_text("{}")
    configuration = {"manifest_sha256": digest(manifest_path), "model": "old-model"}
    (run / "configuration.json").write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="Different API model"):
        build_report(manifest_path, [run], "dots")
    configuration["manifest_sha256"] = "other"
    (run / "configuration.json").write_text(json.dumps(configuration))
    with pytest.raises(ValueError, match="Different frozen"):
        build_report(manifest_path, [run], "dots")


def test_dots_snapshot_metrics_are_reproducible():
    path = Path(__file__).parent.parent / "nutricoach/food_vision/nutrition5k_dots_results.json"
    snapshot = json.loads(path.read_text())
    from nutricoach.food_vision.evaluate_nutrition5k import summarize
    computed = summarize(snapshot["predictions"], snapshot["manifest"], list(snapshot["predictions"]), 42)
    for key, value in computed.items():
        assert snapshot["summary"][key] == value
    assert snapshot["summary"]["methods"]["vlm_single"]["successful"] == 50
    assert snapshot["configurations"]["vlm_single"]["model"] == "dots-studio/dots-3-note-preview:free"
