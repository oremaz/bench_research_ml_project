"""GPU integration tests for the trained recipe models.

Run on the GPU box with:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=nut_agent uv run python -m pytest nut_agent/tests/test_gpu_pipeline.py -v

Tests are skipped when no CUDA device or no trained checkpoints are available,
so the suite stays green on CPU-only checkouts.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = REPO_ROOT / "ml_pipeline" / "results"
META_PATH = RESULTS_DIR / "recipe_models_meta.json"

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
requires_checkpoints = pytest.mark.skipif(not META_PATH.exists(), reason="trained checkpoints not available")


@pytest.fixture(scope="module")
def predictor():
    from recipe_lab.predictor import FoodModelPredictor
    return FoodModelPredictor()


@requires_cuda
class TestLocalEmbedderGPU:
    def test_embeds_on_cuda_with_correct_dim(self):
        from recipe_lab.predictor import LocalEmbedder, EMBEDDING_DIM

        emb = LocalEmbedder(device="cuda")
        v = emb.embed("name: pancakes ingredients: flour, eggs, milk steps: mix and fry")
        assert emb.device == "cuda"
        assert v.shape == (EMBEDDING_DIM,)
        assert np.isfinite(v).all()
        # normalize_embeddings=True must hold at train and inference time
        assert abs(np.linalg.norm(v) - 1.0) < 1e-3

    def test_batch_matches_single(self):
        from recipe_lab.predictor import LocalEmbedder

        emb = LocalEmbedder(device="cuda")
        single = emb.embed("tomato soup")
        batch = emb.embed(["tomato soup", "chocolate cake"])
        assert batch.shape[0] == 2
        assert np.allclose(single, batch[0], atol=1e-5)


@requires_cuda
@requires_checkpoints
class TestTrainedRecipeModels:
    def test_all_three_models_load(self, predictor):
        assert predictor.difficulty_pipeline is not None
        assert predictor.meal_type_pipeline is not None
        assert predictor.time_class_pipeline is not None

    def test_probability_widths_match_labels(self, predictor):
        emb = predictor.get_text_embedding("name: toast ingredients: bread steps: toast it")
        d = predictor.predict_difficulty_from_embedding(emb)
        m = predictor.predict_meal_type_from_embedding(emb)
        t = predictor.predict_time_class_from_embedding(emb)
        assert len(d["all_probabilities"]) == len(predictor.difficulty_labels)
        assert len(m["all_probabilities"]) == len(predictor.meal_type_labels)
        assert len(t["all_probabilities"]) == len(predictor.time_class_labels)
        for r in (d, m, t):
            assert "error" not in r
            probs = list(r["all_probabilities"].values())
            assert abs(sum(probs) - 1.0) < 1e-6

    def test_breakfast_recipe_predicted_sensibly(self, predictor):
        emb = predictor.get_text_embedding(predictor.format_recipe_text({
            "name": "porridge with banana",
            "ingredients": ["oats", "milk", "banana"],
            "steps": ["microwave oats and milk for 3 minutes", "top with sliced banana"],
        }))
        meal = predictor.predict_meal_type_from_embedding(emb)
        time_c = predictor.predict_time_class_from_embedding(emb)
        assert meal["prediction"] == "Breakfast"
        assert time_c["prediction"] in ("<15 min", "15-30 min")

    def test_long_recipe_predicted_slow(self, predictor):
        emb = predictor.get_text_embedding(predictor.format_recipe_text({
            "name": "beef wellington",
            "ingredients": ["beef fillet", "puff pastry", "mushrooms", "prosciutto"],
            "steps": ["sear the beef", "prepare mushroom duxelles", "wrap in prosciutto and pastry",
                       "chill for 30 minutes", "bake for 40 minutes", "rest before serving"],
        }))
        time_c = predictor.predict_time_class_from_embedding(emb)
        assert time_c["prediction"] in ("30-60 min", ">60 min")

    def test_meta_file_consistent_with_predictor(self, predictor):
        with open(META_PATH) as f:
            meta = json.load(f)
        assert meta["embedding"]["model"] == predictor.embedder.model_name
        assert meta["embedding"]["dim"] == predictor.embedding_dim
        for task, key in [("difficulty", "difficulty_labels"),
                          ("meal_type", "meal_type_labels"),
                          ("time_class", "time_class_labels")]:
            assert meta["tasks"][task]["labels"] == getattr(predictor, key)
