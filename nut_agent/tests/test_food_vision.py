"""Food vision behavior without model downloads or live APIs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.food_vision.rf_detr_analyzer import RFDETRAnalyzer
from nutricoach.food_vision.vlm_analyzer import VLMAnalyzerSingleShot
from nutricoach.food_vision.compare import get_available_methods, run_comparison
from nutricoach.tools import analyze_food_image


@pytest.mark.parametrize("effort", ["high", "low"])
def test_single_shot_makes_one_image_call(tmp_path, effort):
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = VLMAnalyzerSingleShot(api_key="test", model="test/vision", reasoning_effort=effort)
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [SimpleNamespace(
        message=SimpleNamespace(content=json.dumps({"items": [{
            "name": "rice", "quantity_grams": 200, "calories": 260,
            "protein_g": 5.4, "carbs_g": 56.4, "fat_g": 0.6,
        }]}))
    )]
    analyzer._client = client
    result = analyzer.analyze(str(image))
    assert result.error is None
    assert result.total_calories == 260
    client.chat.completions.create.assert_called_once()
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {"reasoning": {"effort": effort}}
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 8192
    contents = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert contents[1]["type"] == "image_url"
    assert result.method == "vlm_single"
    assert "vlm_chain" not in get_available_methods()


def test_single_shot_agent_dispatch(tmp_path, monkeypatch):
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = MagicMock()
    monkeypatch.setattr("nutricoach.food_vision.vlm_analyzer.VLMAnalyzerSingleShot", analyzer)
    analyze_food_image.invoke(
        {"image_path": str(image)},
        config={"configurable": {"openrouter_api_key": "test", "vision_model_id": "test/vision"}},
    )
    analyzer.assert_called_once_with(api_key="test", model="test/vision", reasoning_effort="high")
    assert set(analyze_food_image.tool_call_schema.model_fields) == {"image_path", "log_meal"}


@pytest.mark.parametrize("effort", ["high", "low"])
def test_rag_reasoning_applies_to_both_calls(tmp_path, effort):
    from nutricoach.food_vision.rag_vlm_analyzer import RAGVLMAnalyzer

    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = RAGVLMAnalyzer(api_key="test", reasoning_effort=effort)
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
        for payload in (["rice"], [{"name": "rice", "quantity_grams": 200, "calories": 260}])
    ]
    analyzer._client = client
    assert analyzer.analyze(str(image)).error is None
    assert client.chat.completions.create.call_count == 2
    for call in client.chat.completions.create.call_args_list:
        assert call.kwargs["extra_body"] == {"reasoning": {"effort": effort}}
        assert call.kwargs["max_tokens"] == 8192


@pytest.mark.parametrize("effort", ["high", "low"])
def test_clip_refinement_reasoning(tmp_path, effort):
    from nutricoach.food_vision.clip_analyzer import CLIPFoodAnalyzer

    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = CLIPFoodAnalyzer(openrouter_api_key="test", reasoning_effort=effort)
    client = MagicMock()
    items = [{"name": "rice", "quantity_grams": 200}]
    client.chat.completions.create.return_value.choices = [
        SimpleNamespace(message=SimpleNamespace(content=json.dumps(items)))]
    analyzer._refinement_client = client
    assert analyzer._llm_refine_portions([("rice", 0.8)], str(image)) == items
    assert client.chat.completions.create.call_args.kwargs["extra_body"] == {"reasoning": {"effort": effort}}
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 8192


def test_rf_detr_missing_checkpoint_never_loads_coco(tmp_path, monkeypatch):
    constructor = MagicMock()
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRBase=constructor))
    result = RFDETRAnalyzer(model_path=str(tmp_path / "missing.pth")).analyze("meal.jpg")
    assert "Generic COCO fallback is disabled" in result.error
    constructor.assert_not_called()


def test_rf_detr_requires_food_labels(tmp_path):
    checkpoint = tmp_path / "food.pth"
    checkpoint.write_bytes(b"checkpoint")
    result = RFDETRAnalyzer(model_path=str(checkpoint)).analyze("meal.jpg")
    assert "Food class mapping required" in result.error


def test_rf_detr_uses_checkpoint_class_ids(tmp_path, monkeypatch):
    checkpoint = tmp_path / "food.pth"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "food_classes.json").write_text(json.dumps({"1": "rice", "2": "egg"}))
    constructor = MagicMock()
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(
        RFDETRBase=constructor, RFDETRLarge=constructor,
    ))
    analyzer = RFDETRAnalyzer(model_path=str(checkpoint))
    analyzer._load_model()
    constructor.assert_called_once_with(pretrain_weights=str(checkpoint), num_classes=2)
    assert analyzer.class_names == {1: "rice", 2: "egg"}


def _detections():
    return SimpleNamespace(
        class_id=np.array([1, 1]), confidence=np.array([0.8, 0.9]),
        xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
    )


def test_custom_coco_categories_use_zero_based_model_indices(tmp_path, monkeypatch):
    checkpoint = tmp_path / "food.pth"
    checkpoint.write_bytes(b"checkpoint")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"categories": [
        {"id": 66, "name": "rice"}, {"id": 24, "name": "egg"},
    ]}))
    constructor = MagicMock()
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(
        RFDETRBase=constructor, RFDETRLarge=constructor,
    ))
    analyzer = RFDETRAnalyzer(model_path=str(checkpoint), labels_path=str(annotations))
    analyzer._load_model()
    assert analyzer.class_names == {0: "egg", 1: "rice"}


def test_measured_weights_are_not_multiplied_by_detections_or_box_area():
    analyzer = RFDETRAnalyzer(portion_weights={"rice": 200}, nutrition_names={"rice": "white rice"})
    analyzer.class_names = {1: "rice"}
    detections = _detections()
    first = analyzer._parse_detections(detections)
    detections.xyxy *= 10
    second = analyzer._parse_detections(detections)
    assert len(first) == 1
    assert first == second
    assert first[0].quantity_grams == 200
    assert first[0].calories == 260
    assert "supplied total weight" in first[0].portion_description


def test_rf_detr_predicts_automatically_without_measurements():
    analyzer = RFDETRAnalyzer()
    analyzer.class_names = {1: "rice"}
    analyzer._model = MagicMock()
    analyzer._model.predict.return_value = _detections()
    result = analyzer.analyze("meal.jpg")
    assert result.error is None
    assert len(result.food_items) == 1
    assert result.food_items[0].quantity_grams == 190
    assert result.total_calories == 247
    assert "assumed serving" in result.food_items[0].portion_description
    assert "white rice" in result.food_items[0].portion_description


def test_unknown_nutrition_fallback_is_explicit():
    analyzer = RFDETRAnalyzer()
    analyzer.class_names = {1: "unlisted ingredient"}
    item = analyzer._parse_detections(_detections())[0]
    assert item.quantity_grams == 150
    assert "generic average nutrition fallback" in item.portion_description


@pytest.mark.parametrize("grams", [0, -1, float("nan"), float("inf")])
def test_rf_detr_rejects_invalid_measured_weights(grams):
    with pytest.raises(ValueError, match="finite and positive"):
        RFDETRAnalyzer(portion_weights={"rice": grams})


def test_comparison_passes_offline_inputs(monkeypatch):
    constructor = MagicMock()
    monkeypatch.setattr("nutricoach.food_vision.compare.get_available_methods",
                        lambda: {"rf_detr": constructor})
    run_comparison("meal.jpg", methods=["rf_detr"], rf_detr_weights="food.pth",
                   rf_detr_labels="labels.json", rf_detr_portion_weights={"rice": 200})
    constructor.assert_called_once_with(model_path="food.pth", labels_path="labels.json",
                                        portion_weights={"rice": 200}, nutrition_names=None)
