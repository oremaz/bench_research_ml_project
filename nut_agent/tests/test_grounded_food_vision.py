"""Database matching and nutrient arithmetic without live API calls."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.food_vision.grounded_vlm_analyzer import GroundedVLMAnalyzer
from nutricoach.food_vision.nutrition_db import NutritionDB, NutrientInfo


def mocked_analyzer(predictions, effort="high"):
    analyzer = GroundedVLMAnalyzer(api_key="test", model="test/vision", reasoning_effort=effort)
    analyzer._client = MagicMock()
    inventory = {"items": [{"name": "white rice", "search_names": ["cooked rice"]}]}
    analyzer._client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
        for payload in (inventory, {"items": predictions})
    ]
    return analyzer


@pytest.mark.parametrize("effort", ["low", "high"])
def test_database_values_override_model_nutrients_without_serving_clamp(tmp_path, effort):
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    analyzer = mocked_analyzer([{"item_id": 0, "db_name": "white rice", "quantity_grams": 10,
                                "calories": 9999}], effort)
    result = analyzer.analyze(str(image))
    assert result.error is None
    assert result.food_items[0].quantity_grams == 10
    assert result.total_calories == 13
    assert result.total_carbs_g == pytest.approx(2.82)
    assert "database: white rice" in result.food_items[0].portion_description
    calls = analyzer._client.chat.completions.create.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["extra_body"] == {"reasoning": {"effort": effort}}
        assert call.kwargs["max_tokens"] == 8192
        assert call.kwargs["messages"][0]["content"][1]["type"] == "image_url"
    assert json.loads(result.raw_response)["candidates"][0]["candidates"]


@pytest.mark.parametrize("predictions", [
    [{"item_id": 0, "db_name": "hallucinated food", "quantity_grams": 100}],
    [{"item_id": 0, "db_name": "white rice", "quantity_grams": -1}],
    [{"item_id": 0, "db_name": "white rice", "quantity_grams": None}],
    [{"item_id": 0, "db_name": "white rice", "quantity_grams": float("nan")}],
    [{"item_id": 0}, {"item_id": 0}],
    [{"item_id": 2}],
    [],
])
def test_invalid_matches_fail_without_partial_predictions(tmp_path, predictions):
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    result = mocked_analyzer(predictions).analyze(str(image))
    assert result.error
    assert not result.food_items


def test_unmatched_food_has_explicit_scaled_fallback(tmp_path):
    image = tmp_path / "meal.jpg"
    image.write_bytes(b"image")
    result = mocked_analyzer([{"item_id": 0, "db_name": None, "quantity_grams": 200,
                               "fallback_per_100g": {"calories": 100, "protein_g": 2,
                                                     "carbs_g": 20, "fat_g": 1}}]).analyze(str(image))
    assert result.error is None
    assert result.total_calories == 200
    assert "model fallback" in result.food_items[0].portion_description


def test_candidates_preserve_preparation_alternatives():
    names = [candidate["name"] for candidate in NutritionDB().search_candidates(["egg"])]
    assert names[0] == "egg"
    assert "fried egg" in names
    assert "boiled egg" in names
    assert NutritionDB().search_candidates(["xyzzyq"]) == []


@pytest.mark.parametrize("grams", [0, -1, float("nan"), float("inf")])
def test_invalid_quantity_is_rejected(grams):
    with pytest.raises(ValueError, match="finite and positive"):
        NutrientInfo(130, 2.7, 28.2, 0.3).scaled(grams)
