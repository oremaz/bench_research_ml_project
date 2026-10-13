"""Database candidate lookup and nutrient quantity validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nutricoach.food_vision.nutrition_db import NutritionDB, NutrientInfo


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
