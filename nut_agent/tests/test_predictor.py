"""Tests for recipe_lab.predictor module (pure functions only, no API/model calls).
The predictor imports real ml_pipeline modules; tests avoid loading checkpoints
or the sentence-transformers encoder by building stubs via __new__.
"""

import sys
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recipe_lab.predictor import FoodModelPredictor, DEFAULT_TASKS, EMBEDDING_DIM


def _make_predictor_stub():
    """Create a FoodModelPredictor stub without loading models or calling APIs."""
    predictor = FoodModelPredictor.__new__(FoodModelPredictor)
    predictor.client = None
    predictor.google_api_key = None
    predictor.openrouter_api_key = None
    predictor.models_path = Path("/fake")
    predictor.meta = {}
    predictor.tasks = DEFAULT_TASKS
    predictor.embedding_dim = EMBEDDING_DIM
    failing_embedder = MagicMock()
    failing_embedder.embed.side_effect = RuntimeError("no encoder in tests")
    predictor.embedder = failing_embedder
    predictor.difficulty_pipeline = None
    predictor.meal_type_pipeline = None
    predictor.time_class_pipeline = None
    predictor.nutrients_pipeline = None
    predictor.nutrients_hf = None
    predictor.difficulty_labels = DEFAULT_TASKS["difficulty"]["labels"]
    predictor.meal_type_labels = DEFAULT_TASKS["meal_type"]["labels"]
    predictor.time_class_labels = DEFAULT_TASKS["time_class"]["labels"]
    predictor.nutrient_targets = DEFAULT_TASKS["nutrients"]["targets"]
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

    def test_get_embedding_fallback_without_encoder(self):
        p = _make_predictor_stub()
        result = p.get_text_embedding("some text")
        assert len(result) == EMBEDDING_DIM
        assert all(v == 0.0 for v in result)

    def test_predict_difficulty_without_model(self):
        p = _make_predictor_stub()
        result = p.predict_difficulty_from_embedding([0.0] * EMBEDDING_DIM)
        assert result["prediction"] == "Unknown"
        assert "error" in result

    def test_predict_meal_type_without_model(self):
        p = _make_predictor_stub()
        result = p.predict_meal_type_from_embedding([0.0] * EMBEDDING_DIM)
        assert result["prediction"] == "Unknown"

    def test_predict_time_class_without_model(self):
        p = _make_predictor_stub()
        result = p.predict_time_class_from_embedding([0.0] * EMBEDDING_DIM)
        assert result["prediction"] == "Unknown"


class TestPredictWithMockModel:
    def test_predict_difficulty_from_embedding(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        p.difficulty_pipeline = mock_pipeline

        result = p.predict_difficulty_from_embedding([0.5] * EMBEDDING_DIM)
        assert result["prediction"] == "More effort"
        assert abs(result["confidence"] - 0.7) < 0.01
        assert "all_probabilities" in result

    def test_predict_meal_type_breakfast(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict_proba.return_value = np.array([[0.8, 0.2]])
        p.meal_type_pipeline = mock_pipeline

        result = p.predict_meal_type_from_embedding([0.5] * EMBEDDING_DIM)
        assert result["prediction"] == "Breakfast"

    def test_predict_meal_type_lunch_dinner(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict_proba.return_value = np.array([[0.3, 0.7]])
        p.meal_type_pipeline = mock_pipeline

        result = p.predict_meal_type_from_embedding([0.5] * EMBEDDING_DIM)
        assert result["prediction"] == "Lunch/Dinner"
        assert "all_probabilities" in result

    def test_predict_time_class(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict_proba.return_value = np.array([[0.05, 0.15, 0.6, 0.2]])
        p.time_class_pipeline = mock_pipeline

        result = p.predict_time_class_from_embedding([0.5] * EMBEDDING_DIM)
        assert result["prediction"] == "30-60 min"
        assert "all_probabilities" in result

    def test_predict_nutrients(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict.return_value = np.array(
            [[420.0, 12.0, 4.0, 55.0, 8.0, 6.0, 22.0, -0.2]]
        )
        p.nutrients_pipeline = mock_pipeline

        result = p.predict_nutrients_from_embedding([0.5] * EMBEDDING_DIM)
        per_serving = result["per_serving"]
        assert per_serving["kcal"] == 420.0
        assert per_serving["salt"] == 0.0
        assert list(per_serving) == p.nutrient_targets

    def test_predict_nutrients_without_model(self):
        p = _make_predictor_stub()
        result = p.predict_nutrients_from_embedding([0.0] * EMBEDDING_DIM)
        assert result["error"] == "Model not loaded"

    def test_predict_nutrients_from_text_prefers_finetune(self):
        p = _make_predictor_stub()
        hf = MagicMock()
        hf.predict.return_value = np.array([500.0, 20.0, 8.0, 60.0, 10.0, 5.0, 25.0, 1.2])
        hf.targets = DEFAULT_TASKS["nutrients"]["targets"]
        p.nutrients_hf = hf

        result = p.predict_nutrients_from_text("name: pizza ingredients: dough steps: bake")
        assert result["method"] == "bge_finetune"
        assert result["per_serving"]["kcal"] == 500.0

    def test_predict_nutrients_from_text_falls_back_to_registry(self):
        p = _make_predictor_stub()
        mock_pipeline = MagicMock()
        mock_pipeline.model.predict.return_value = np.array(
            [[300.0, 10.0, 3.0, 40.0, 6.0, 4.0, 15.0, 0.5]]
        )
        p.nutrients_pipeline = mock_pipeline

        result = p.predict_nutrients_from_text("text", embedding=[0.5] * EMBEDDING_DIM)
        assert result["per_serving"]["kcal"] == 300.0
        assert result["method"] == "lightgbm"


class TestAnalyzeRecipe:
    def test_analyze_without_client(self):
        p = _make_predictor_stub()
        result = p.analyze_recipe("Spaghetti carbonara")
        assert "original_description" in result
        assert result["original_description"] == "Spaghetti carbonara"
        assert result["difficulty"]["prediction"] == "Unknown"


class TestModelFamilyRegistry:
    """The predictor must be able to serve every deployable model family."""

    def test_new_families_registered(self):
        from pipelines_torch.models import MODEL_REGISTRY

        for key in ("catboost_classifier", "catboost_regressor",
                    "stacking_classifier", "stacking_regressor"):
            assert key in MODEL_REGISTRY

    def test_registry_class_resolution(self):
        from pipelines_torch.models import MODEL_REGISTRY

        for name in ("lightgbm", "xgboost", "catboost", "stacking"):
            cls = FoodModelPredictor._registry_class(name, "classification")
            assert cls is MODEL_REGISTRY[f"{name}_classifier"]
        cls = FoodModelPredictor._registry_class("catboost", "regression")
        assert cls is MODEL_REGISTRY["catboost_regressor"]
        # unknown family falls back to lightgbm
        cls = FoodModelPredictor._registry_class("nonexistent", "classification")
        assert cls is MODEL_REGISTRY["lightgbm_classifier"]

    def test_catboost_wrappers_fit_predict(self):
        from pipelines_torch.models import MODEL_REGISTRY

        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 8)).astype(np.float32)
        y_cls = (X[:, 0] > 0).astype(int)
        y_reg = np.stack([X[:, 0] * 2, X[:, 1] - 1], axis=1)

        clf = MODEL_REGISTRY["catboost_classifier"](iterations=20)
        clf.fit(X, y_cls)
        probs = clf.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

        reg = MODEL_REGISTRY["catboost_regressor"](iterations=20)
        reg.fit(X, y_reg)
        preds = reg.predict(X[:5])
        assert tuple(preds.shape) == (5, 2)

    def test_stacking_wrappers_fit_predict(self):
        from pipelines_torch.models import MODEL_REGISTRY

        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 8)).astype(np.float32)
        y_cls = (X[:, 0] + X[:, 1] > 0).astype(int)
        y_reg = np.stack([X[:, 0], X[:, 1] * 3], axis=1)

        clf = MODEL_REGISTRY["stacking_classifier"](cv=3, n_estimators=25)
        clf.fit(X, y_cls)
        probs = clf.predict_proba(X[:5])
        assert probs.shape == (5, 2)

        reg = MODEL_REGISTRY["stacking_regressor"](cv=3, n_estimators=25)
        reg.fit(X, y_reg)
        preds = reg.predict(X[:5])
        assert tuple(preds.shape) == (5, 2)
