"""Tests for recipe_lab.predictor module (pure functions only, no API/model calls).
We mock all heavy dependencies (torch, google, food_preds) at sys.modules level
so the module can be imported in test environments without those packages.
"""

import sys
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Pre-mock heavy modules that recipe_lab.predictor imports
_MOCK_MODULES = {
    "torch": MagicMock(),
    "google": MagicMock(),
    "google.genai": MagicMock(),
    "google.genai.types": MagicMock(),
    "google.api_core": MagicMock(),
    "google.api_core.retry": MagicMock(),
    "food_preds": MagicMock(),
    "food_preds.pipelines_torch": MagicMock(),
    "food_preds.pipelines_torch.base": MagicMock(),
    "food_preds.utils": MagicMock(),
    "food_preds.utils.utils": MagicMock(),
    "food_preds.pipelines_torch.models": MagicMock(),
}

# Apply mocks and import
_saved = {k: sys.modules.get(k) for k in _MOCK_MODULES}
sys.modules.update(_MOCK_MODULES)

try:
    from recipe_lab.predictor import FoodModelPredictor
finally:
    # Restore original modules
    for k, v in _saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _make_predictor_stub():
    """Create a FoodModelPredictor stub without loading models or calling APIs."""
    predictor = FoodModelPredictor.__new__(FoodModelPredictor)
    predictor.client = None
    predictor.api_key = None
    predictor.models_path = Path("/fake")
    predictor.difficulty_pipeline = None
    predictor.meal_type_pipeline = None
    predictor.time_class_pipeline = None
    predictor.difficulty_labels = ["Easy", "More effort", "A challenge"]
    predictor.meal_type_labels = ["breakfast", "lunch", "dinner", "snack", "dessert"]
    predictor.time_class_labels = ["<15 min", "15-30 min", "30-60 min", ">60 min"]
    return predictor


class TestFormatRecipeText:
    def test_format_with_strings(self):
        p = _make_predictor_stub()
        result = p.format_recipe_text({
            "name": "Grilled Chicken",
            "ingredients": "chicken, salt, pepper",
            "steps": "Grill until done"
        })
        assert "name: Grilled Chicken" in result
        assert "ingredients: chicken, salt, pepper" in result
        assert "steps: Grill until done" in result

    def test_format_with_lists(self):
        p = _make_predictor_stub()
        result = p.format_recipe_text({
            "name": "Pasta",
            "ingredients": ["pasta", "tomato sauce", "cheese"],
            "steps": ["Boil pasta", "Add sauce", "Top with cheese"]
        })
        assert "ingredients: pasta, tomato sauce, cheese" in result
        assert "steps: Boil pasta. Add sauce. Top with cheese" in result

    def test_format_with_empty_data(self):
        p = _make_predictor_stub()
        result = p.format_recipe_text({})
        assert result == "name:  ingredients:  steps: "

    def test_format_cleans_whitespace(self):
        p = _make_predictor_stub()
        result = p.format_recipe_text({
            "name": "  Messy\n  Name  ",
            "ingredients": "a,  b,\nc",
            "steps": "step\n  one"
        })
        assert "\n" not in result
        assert "  " not in result


class TestFallbackBehavior:
    def test_enhance_recipe_fallback_without_client(self):
        p = _make_predictor_stub()
        result = p.enhance_recipe_description("Grilled salmon with lemon")
        assert result["name"] == "Grilled salmon with lemon"

    def test_get_embedding_fallback_without_client(self):
        p = _make_predictor_stub()
        result = p.get_text_embedding("some text")
        assert len(result) == 3072
        assert all(v == 0.0 for v in result)

    def test_predict_difficulty_without_model(self):
        p = _make_predictor_stub()
        result = p.predict_difficulty_from_embedding([0.0] * 3072)
        assert result["prediction"] == "Unknown"
        assert "error" in result

    def test_predict_meal_type_without_model(self):
        p = _make_predictor_stub()
        result = p.predict_meal_type_from_embedding([0.0] * 3072)
        assert result["prediction"] == "Unknown"

    def test_predict_time_class_without_model(self):
        p = _make_predictor_stub()
        result = p.predict_time_class_from_embedding([0.0] * 3072)
        assert result["prediction"] == "Unknown"


class TestPredictWithMockModel:
    def test_predict_difficulty_from_embedding(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])
        p.difficulty_pipeline = mock_pipeline

        result = p.predict_difficulty_from_embedding([0.5] * 3072)
        assert result["prediction"] == "More effort"
        assert abs(result["confidence"] - 0.7) < 0.01
        assert "all_probabilities" in result

    def test_predict_meal_type_breakfast(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict_proba.return_value = np.array([[0.8, 0.05, 0.05, 0.05, 0.05]])
        p.meal_type_pipeline = mock_pipeline

        result = p.predict_meal_type_from_embedding([0.5] * 3072)
        assert result["prediction"] == "breakfast"

    def test_predict_meal_type_lunch_dinner(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict_proba.return_value = np.array([[0.1, 0.6, 0.15, 0.1, 0.05]])
        p.meal_type_pipeline = mock_pipeline

        result = p.predict_meal_type_from_embedding([0.5] * 3072)
        assert result["prediction"] == "lunch/dinner"

    def test_predict_time_class(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict_proba.return_value = np.array([[0.05, 0.15, 0.6, 0.2]])
        p.time_class_pipeline = mock_pipeline

        result = p.predict_time_class_from_embedding([0.5] * 3072)
        assert result["prediction"] == "30-60 min"
        assert "all_probabilities" in result


class TestAnalyzeRecipe:
    def test_analyze_without_client(self):
        p = _make_predictor_stub()
        result = p.analyze_recipe("Spaghetti carbonara")
        assert "original_description" in result
        assert result["original_description"] == "Spaghetti carbonara"
        assert result["difficulty"]["prediction"] == "Unknown"
