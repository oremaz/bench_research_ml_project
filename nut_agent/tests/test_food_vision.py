"""Food vision behavior without model downloads or live APIs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.food_vision.vlm_analyzer import VLMAnalyzerSingleShot
from nutricoach.food_vision.compare import get_available_methods, run_comparison
from nutricoach.tools import analyze_food_image


@pytest.mark.parametrize("method", ["rf_detr", "clip_vlm", "rag_vlm", "grounded_vlm", "geometry_vlm_db"])
def test_removed_methods_are_rejected(method):
    from nutricoach.food_vision.report_nutrition5k import REPORT_METHODS
    assert method not in REPORT_METHODS
    with pytest.raises(ValueError, match="Unknown food vision methods"):
        run_comparison("unused.png", methods=[method])


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
    analyzer.assert_called_once_with(api_key="test", model="test/vision", reasoning_effort="high",
                                     provider="openrouter", device=None, quantization="none")
    assert set(analyze_food_image.tool_call_schema.model_fields) == {"image_path", "log_meal"}
