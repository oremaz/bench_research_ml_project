"""Tests for recipe_lab.predictor module (pure functions only, no API/model calls).
The predictor imports real ml_pipeline modules; tests avoid loading checkpoints
or the sentence-transformers encoder by building stubs via __new__.
"""

import sys
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from recipe_lab.predictor import FoodModelPredictor, DEFAULT_TASKS, EMBEDDING_DIM
from recipe_lab.local_models import _cache_key, _total_minutes, load_or_train_models


def _make_predictor_stub():
    """Create a FoodModelPredictor stub without loading models or calling APIs."""
    predictor = FoodModelPredictor.__new__(FoodModelPredictor)
    predictor.client = None
    predictor.api_key = None
    predictor.model_id = "dots-studio/dots-3-note-preview:free"
    predictor.reasoning_effort = "high"
    predictor.local_encoder = None
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
    predictor.difficulty_labels = DEFAULT_TASKS["difficulty"]["labels"]
    predictor.meal_type_labels = DEFAULT_TASKS["meal_type"]["labels"]
    predictor.time_class_labels = DEFAULT_TASKS["time_class"]["labels"]
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

    def test_zero_shot_nutrients(self):
        p = _make_predictor_stub()
        p.client = MagicMock()
        p._generate_text = MagicMock(return_value=(
            '{"kcal":300,"fat":10,"saturates":3,"carbs":40,'
            '"sugars":6,"fibre":4,"protein":15,"salt":0.5}'
        ))
        result = p.estimate_nutrients_zero_shot("text")
        assert result["per_serving"]["kcal"] == 300.0
        assert result["method"] == "openrouter_zero_shot"

    def test_zero_shot_nutrients_requires_openrouter(self):
        p = _make_predictor_stub()
        result = p.estimate_nutrients_zero_shot("text")
        assert "API key required" in result["error"]


class TestAnalyzeRecipe:
    def test_analyze_without_client(self):
        p = _make_predictor_stub()
        result = p.analyze_recipe("Spaghetti carbonara")
        assert "original_description" in result
        assert result["original_description"] == "Spaghetti carbonara"
        assert result["difficulty"]["prediction"] == "Unknown"


class TestModelFamilyRegistry:
    """The predictor must be able to serve every deployable model family."""

    @pytest.fixture(autouse=True)
    def requires_model_registry(self):
        try:
            from pipelines_torch.models import MODEL_REGISTRY
        except Exception as exc:
            pytest.skip(f"model registry unavailable in this environment: {exc}")
        return MODEL_REGISTRY

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


def test_training_embeddings_are_cached(tmp_path, monkeypatch):
    data_path = tmp_path / "recipes.csv"
    pd.DataFrame({
        "recipe_text": ["oats", "pasta", "soup"],
        "difficult": ["Easy", "More effort", "Easy"],
        "subcategory": ["Breakfast recipes", "Dinner recipes", "Lunch recipes"],
        "times": ["{'Preparation': '10 mins'}", "{'Cooking': '20 mins'}", "{'Cooking': '40 mins'}"],
    }).to_csv(data_path, index=False)
    encoder = MagicMock()
    encoder.encode.return_value = np.ones((3, 4), dtype=np.float32)
    model_loads = MagicMock(return_value=encoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=model_loads))
    with patch("recipe_lab.local_models._fit", return_value="trained"):
        load_or_train_models(data_path, tmp_path)
        assert encoder.encode.call_count == 1
        assert encoder.encode.call_args.kwargs["prompt_name"] == "document"
        assert model_loads.call_args.kwargs["trust_remote_code"] is True
        (tmp_path / f"lightgbm_{_cache_key(data_path)}.joblib").unlink()
        load_or_train_models(data_path, tmp_path)
        assert encoder.encode.call_count == 1


def test_time_parser():
    assert _total_minutes("{'Preparation': '1 hr 15 mins', 'Cooking': '20 mins'}") == 95
    assert _total_minutes("No Time") == 0
