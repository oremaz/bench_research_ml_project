"""GPU integration tests for the trained recipe models and food vision stack.

Run on the GPU box with:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=nut_agent uv run python -m pytest nut_agent/tests/test_gpu_pipeline.py -v

Tests are skipped when no CUDA device or no trained checkpoints are available,
so the suite stays green on CPU-only checkouts.
Set RUN_SLOW_GPU_TESTS=1 to include RF-DETR inference (downloads/loads ~500MB).
"""

import os
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
slow = pytest.mark.skipif(os.environ.get("RUN_SLOW_GPU_TESTS") != "1", reason="set RUN_SLOW_GPU_TESTS=1")


@pytest.fixture(scope="module")
def predictor():
    from recipe_lab.predictor import FoodModelPredictor
    return FoodModelPredictor()


@pytest.fixture(scope="module")
def synthetic_food_image(tmp_path_factory):
    """A simple synthetic plate image; enough to exercise vision pipelines."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), (240, 235, 225))
    draw = ImageDraw.Draw(img)
    draw.ellipse([56, 56, 456, 456], fill=(250, 250, 250), outline=(200, 200, 200), width=4)
    draw.ellipse([120, 140, 280, 260], fill=(180, 120, 60))
    draw.ellipse([260, 240, 400, 360], fill=(90, 140, 60))
    path = tmp_path_factory.mktemp("imgs") / "plate.jpg"
    img.save(path, "JPEG")
    return str(path)


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

    def test_nutrients_model_loads_and_predicts(self, predictor):
        assert predictor.nutrients_pipeline is not None
        emb = predictor.get_text_embedding(predictor.format_recipe_text({
            "name": "spaghetti bolognese",
            "ingredients": ["spaghetti", "minced beef", "tomato sauce", "onion"],
            "steps": ["cook pasta", "brown the beef", "simmer with sauce"],
        }))
        out = predictor.predict_nutrients_from_embedding(emb)
        per_serving = out["per_serving"]
        assert list(per_serving) == predictor.nutrient_targets
        assert all(v >= 0 for v in per_serving.values())
        assert 50 <= per_serving["kcal"] <= 2000


@requires_cuda
class TestCLIPAnalyzerGPU:
    def test_clip_classification_runs_on_cuda(self, synthetic_food_image):
        from nutricoach.food_vision.clip_analyzer import CLIPFoodAnalyzer

        analyzer = CLIPFoodAnalyzer(openrouter_api_key="", device="cuda")
        results = analyzer._classify_food(synthetic_food_image)
        assert analyzer.device == "cuda"
        assert isinstance(results, list)
        for label, score in results:
            assert isinstance(label, str)
            assert 0.0 <= score <= 1.0

    def test_full_pipeline_without_llm(self, synthetic_food_image):
        from nutricoach.food_vision.clip_analyzer import CLIPFoodAnalyzer

        analyzer = CLIPFoodAnalyzer(openrouter_api_key="", device="cuda")
        result = analyzer.analyze(synthetic_food_image)
        assert result.error is None
        assert result.method == "clip_ensemble"
        for item in result.food_items:
            assert item.quantity_grams > 0
            assert item.calories >= 0
        assert result.total_calories == pytest.approx(
            sum(i.calories for i in result.food_items), abs=1e-6)

    @slow
    def test_jina_backend_classification(self, synthetic_food_image):
        from nutricoach.food_vision.clip_analyzer import CLIPFoodAnalyzer

        analyzer = CLIPFoodAnalyzer(openrouter_api_key="", device="cuda", backend="jina")
        results = analyzer._classify_food(synthetic_food_image)
        assert isinstance(results, list) and results
        for label, score in results:
            assert isinstance(label, str)
            assert 0.0 <= score <= 1.0


@requires_cuda
class TestRFDETRAnalyzer:
    @slow
    def test_coco_inference_filters_non_food(self, synthetic_food_image):
        from nutricoach.food_vision.rf_detr_analyzer import RFDETRAnalyzer

        analyzer = RFDETRAnalyzer()
        result = analyzer.analyze(synthetic_food_image)
        assert result.error is None
        coco_food = {"banana", "apple", "sandwich", "orange", "broccoli",
                     "carrot", "hot dog", "pizza", "donut", "cake"}
        for item in result.food_items:
            assert item.name in coco_food

    @slow
    def test_finetuned_weights_load_if_present(self, synthetic_food_image):
        from nutricoach.food_vision.rf_detr_analyzer import RFDETRAnalyzer

        weights = sorted((RESULTS_DIR / "rf_detr_food").glob("*.pth"))
        if not weights:
            pytest.skip("no fine-tuned RF-DETR checkpoint yet")
        analyzer = RFDETRAnalyzer(model_path=str(weights[-1]))
        result = analyzer.analyze(synthetic_food_image)
        assert result.error is None


class TestPortionEstimation:
    def test_area_fraction_map_monotonic(self):
        from nutricoach.food_vision.rf_detr_analyzer import estimate_grams_from_area_fraction

        fractions = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
        grams = [estimate_grams_from_area_fraction(f) for f in fractions]
        assert grams == sorted(grams, reverse=True)
        assert grams[0] == 350
        assert grams[-1] == 30
