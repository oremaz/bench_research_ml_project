"""
ML-powered recipe analysis using local classifiers and OpenRouter.

Embeddings are computed locally with a sentence-transformers model (GPU when
available) so that training and inference share the exact same text encoder.
The trained checkpoints and their embedding/label metadata live in
ml_pipeline/results/ (see ml_pipeline/train_recipe_models.py).
"""

import os
import sys
import json
import logging
import numpy as np
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
from openai import OpenAI

from recipe_lab.local_models import (
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    _adapt_lfm_shortconv,
    load_or_train_models,
)
from shared.config import OPENROUTER_MODEL_ID, OPENROUTER_REASONING_EFFORT, OPENROUTER_MAX_OUTPUT_TOKENS, openrouter_reasoning

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "ml_pipeline"))

from pipelines_torch.base import GeneralPipelineSklearn
from utils.utils import load_model_by_name

logger = logging.getLogger(__name__)

# Must match ml_pipeline/train_recipe_models.py; recipe_models_meta.json
# written at training time is the source of truth.
LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

JINA_MODEL_TAG = "jina-embeddings-v5"
JINA_DOC_PREFIX = "Document: "

DEFAULT_TASKS = {
    "difficulty": {"path_start": "difficulty_train", "labels": ["Easy", "More effort"]},
    "meal_type": {"path_start": "meal_train", "labels": ["Breakfast", "Lunch/Dinner"]},
    "time_class": {"path_start": "total_time_train", "labels": ["<15 min", "15-30 min", "30-60 min", ">60 min"]},
}

NUTRIENT_TARGETS = ("kcal", "fat", "saturates", "carbs", "sugars", "fibre", "protein", "salt")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LocalEmbedder:
    """Sentence-transformers text encoder shared by training and inference."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def is_jina(self) -> bool:
        return JINA_MODEL_TAG in self.model_name

    @property
    def is_lfm(self) -> bool:
        return self.model_name == EMBEDDING_MODEL_ID

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            kwargs = {}
            if self.is_jina:
                # Jina v5 ships custom code; the classification task adapter
                # expects a "Document: " prefix on inputs.
                kwargs = {
                    "trust_remote_code": True,
                    "model_kwargs": {"default_task": "classification"},
                }
            elif self.is_lfm:
                kwargs = {
                    "revision": EMBEDDING_MODEL_REVISION,
                    "trust_remote_code": True,
                }
            self._model = SentenceTransformer(self.model_name, device=self.device, **kwargs)
            if self.is_lfm:
                _adapt_lfm_shortconv(self._model)
        return self._model

    def embed(self, texts: Union[str, List[str]], batch_size: int = 64) -> np.ndarray:
        model = self._load()
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        if self.is_jina:
            inputs = [JINA_DOC_PREFIX + t for t in inputs]
            batch_size = min(batch_size, 16)
        encode_kwargs = {}
        if self.is_lfm:
            encode_kwargs["prompt_name"] = "document"
            batch_size = min(batch_size, 16)
        emb = model.encode(
            inputs, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=False, **encode_kwargs,
        )
        emb = np.asarray(emb, dtype=np.float32)
        return emb[0] if single else emb


class FoodModelPredictor:
    """
    Wrapper class to load and use the trained food prediction models with text embeddings.
    Supports local classification and zero-shot per-serving nutrition estimation.
    """

    def __init__(self, models_path: str = None, api_key: str = None, device: Optional[str] = None,
                 model_id: str = OPENROUTER_MODEL_ID, reasoning_effort: str = OPENROUTER_REASONING_EFFORT):
        if models_path is None:
            models_path = REPO_ROOT / "ml_pipeline" / "results"
        self.models_path = Path(models_path)

        self.meta = self._load_meta()
        self.tasks = self.meta.get("tasks", DEFAULT_TASKS)
        emb_meta = self.meta.get("embedding", {})
        self.legacy_compatible = emb_meta.get("model") in (
            LOCAL_EMBEDDING_MODEL, "jinaai/jina-embeddings-v5-omni-small",
        )
        self.embedder = LocalEmbedder(
            model_name=emb_meta.get("model", LOCAL_EMBEDDING_MODEL),
            device=device,
        )
        self.embedding_dim = emb_meta.get("dim", EMBEDDING_DIM)

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_id = model_id
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=self.api_key) if self.api_key else None
        self.local_encoder = None
        self.local_models = {}

        self.difficulty_pipeline = None
        self.meal_type_pipeline = None
        self.time_class_pipeline = None
        self.difficulty_labels = self.tasks["difficulty"]["labels"]
        self.meal_type_labels = self.tasks["meal_type"]["labels"]
        self.time_class_labels = self.tasks["time_class"]["labels"]

        if self.legacy_compatible:
            self._load_models()
        try:
            self.local_encoder, self.local_models = load_or_train_models()
            for task in ("difficulty", "meal_type", "time_class"):
                if task in self.local_models:
                    setattr(self, f"{task}_pipeline", GeneralPipelineSklearn(
                        model=self.local_models[task], task_type="classification"))
                    setattr(self, f"{task}_labels", list(self.local_models[task].classes_))
            self.embedding_dim = 1024
        except Exception:
            if any(getattr(self, attr) is None for attr in
                   ("difficulty_pipeline", "meal_type_pipeline", "time_class_pipeline")):
                raise
            logger.warning("LFM2.5 unavailable; using existing compatible recipe checkpoints")

    def _load_meta(self) -> Dict[str, Any]:
        meta_path = self.models_path / "recipe_models_meta.json"
        try:
            if meta_path.exists():
                with open(meta_path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Could not read %s: %s", meta_path, e)
        return {}

    @staticmethod
    def _registry_class(model_name: str, task_type: str):
        """Resolve the wrapper class for a checkpoint's model family."""
        from pipelines_torch.models import MODEL_REGISTRY
        suffix = "classifier" if task_type == "classification" else "regressor"
        key = f"{model_name}_{suffix}"
        return MODEL_REGISTRY.get(key, MODEL_REGISTRY[f"lightgbm_{suffix}"])

    def _task_model_name(self, task: str) -> str:
        """Per-task model family from meta, falling back to the global name."""
        return self.tasks[task].get("model_name", self.meta.get("model_name", "lightgbm"))

    def _load_models(self):
        """Load the trained classification models."""
        for attr, task in [
            ("difficulty_pipeline", "difficulty"),
            ("meal_type_pipeline", "meal_type"),
            ("time_class_pipeline", "time_class"),
        ]:
            try:
                model_name = self._task_model_name(task)
                path_start = str(self.models_path / self.tasks[task]["path_start"])
                model = load_model_by_name(
                    self._registry_class(model_name, "classification"),
                    model_name,
                    {},
                    path_start=path_start,
                    task_type="classification",
                )
                setattr(self, attr, GeneralPipelineSklearn(model=model, task_type="classification"))
            except Exception as e:
                logging.error(f"Error loading {task} model: {e}")

    # --- LLM helpers ---

    def _generate_text(self, prompt: str) -> Optional[str]:
        """Generate optional recipe text with OpenRouter."""
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=OPENROUTER_MAX_OUTPUT_TOKENS,
                    extra_body=openrouter_reasoning(self.reasoning_effort),
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning("OpenRouter generation failed: %s", e)

        return None

    def enhance_recipe_description(self, user_description: str) -> Dict[str, Union[str, List[str]]]:
        """Use LLM to enhance user's recipe description and extract structured information."""
        fallback_data = {
            'name': user_description.split(',')[0].strip(),
            'ingredients': 'Not specified',
            'steps': 'Not specified'
        }

        prompt = f"""
        Given this recipe description: "{user_description}"

        Please extract or infer the following information and format it as a JSON object.

        JSON format:
        {{
            "name": "Recipe name (infer if not explicitly given)",
            "ingredients": "A list of strings for ingredients (infer typical ingredients if not specified)",
            "steps": "A list of strings for cooking steps (infer basic steps if not specified)"
        }}

        Make reasonable inferences based on the recipe type. For example, if the description is "grilled chicken",
        infer ingredients like chicken breast, salt, pepper, oil, and basic grilling steps.

        IMPORTANT: Your entire response must be ONLY the raw JSON object, without any markdown formatting (like ```json), explanations, or other text.
        """

        raw_text = self._generate_text(prompt)
        if not raw_text:
            return fallback_data

        try:
            json_start_index = raw_text.find('{')
            json_end_index = raw_text.rfind('}') + 1

            if json_start_index == -1 or json_end_index == 0:
                return fallback_data

            json_string = raw_text[json_start_index:json_end_index]
            return json.loads(json_string)
        except Exception:
            return fallback_data

    def format_recipe_text(self, recipe_data: Dict[str, str]) -> str:
        """
        Format recipe data matching training format: "name: [name] ingredients: [ingredients] steps: [steps]"
        """
        name = recipe_data.get('name', '').strip() if isinstance(recipe_data.get('name', ''), str) else str(recipe_data.get('name', ''))

        ingredients = recipe_data.get('ingredients', '')
        if isinstance(ingredients, list):
            ingredients = ', '.join(ingredients)
        elif isinstance(ingredients, str):
            ingredients = ingredients.strip()
        else:
            ingredients = str(ingredients)

        steps = recipe_data.get('steps', '')
        if isinstance(steps, list):
            steps = '. '.join(steps)
        elif isinstance(steps, str):
            steps = steps.strip()
        else:
            steps = str(steps)

        def clean_text(text):
            return ' '.join(text.replace('\n', ' ').split())

        return f"name: {clean_text(name)} ingredients: {clean_text(ingredients)} steps: {clean_text(steps)}"

    def get_text_embedding(self, text: str, task_type: str = "classification") -> List[float]:
        """Embed text with the same local encoder used at training time."""
        try:
            if self.local_encoder is not None:
                return self.local_encoder.encode(
                    [text], prompt_name="document", normalize_embeddings=True,
                )[0].tolist()
            return self.embedder.embed(text).tolist()
        except Exception as e:
            logger.error("Embedding failed: %s", e)
            return [0.0] * self.embedding_dim

    def predict_difficulty_from_embedding(self, embedding: List[float]) -> Dict[str, Any]:
        """Predict cooking difficulty from text embedding."""
        if self.difficulty_pipeline is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        try:
            embedding_array = np.array(embedding).reshape(1, -1)
            probabilities = self.difficulty_pipeline.model.predict_proba(embedding_array)
            predicted_class = np.argmax(probabilities[0])
            confidence = float(probabilities[0][predicted_class])
            all_probs = {label: float(prob) for label, prob in zip(self.difficulty_labels, probabilities[0])}
            return {
                "prediction": self.difficulty_labels[predicted_class],
                "confidence": confidence,
                "class_index": int(predicted_class),
                "all_probabilities": all_probs
            }
        except Exception as e:
            return {"prediction": "Unknown", "confidence": 0.0, "error": str(e)}

    def predict_meal_type_from_embedding(self, embedding: List[float]) -> Dict[str, Any]:
        """Predict meal type (Breakfast / Dinner / Lunch) from text embedding."""
        if self.meal_type_pipeline is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        try:
            embedding_array = np.array(embedding).reshape(1, -1)
            probabilities = self.meal_type_pipeline.model.predict_proba(embedding_array)
            predicted_class = np.argmax(probabilities[0])
            confidence = float(probabilities[0][predicted_class])
            all_probs = {label: float(prob) for label, prob in zip(self.meal_type_labels, probabilities[0])}
            return {
                "prediction": self.meal_type_labels[predicted_class],
                "confidence": confidence,
                "class_index": int(predicted_class),
                "all_probabilities": all_probs
            }
        except Exception as e:
            return {"prediction": "Unknown", "confidence": 0.0, "error": str(e)}

    def predict_time_class_from_embedding(self, embedding: List[float]) -> Dict[str, Any]:
        """Predict total time class from text embedding."""
        if self.time_class_pipeline is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        try:
            embedding_array = np.array(embedding).reshape(1, -1)
            probabilities = self.time_class_pipeline.model.predict_proba(embedding_array)
            predicted_class = int(np.argmax(probabilities[0]))
            confidence = float(probabilities[0][predicted_class])
            if predicted_class < len(self.time_class_labels):
                label = self.time_class_labels[predicted_class]
            else:
                label = str(predicted_class)
            all_probs = {
                self.time_class_labels[i] if i < len(self.time_class_labels) else str(i): float(prob)
                for i, prob in enumerate(probabilities[0])
            }
            return {
                "prediction": label,
                "confidence": confidence,
                "class_index": predicted_class,
                "all_probabilities": all_probs
            }
        except Exception as e:
            return {"prediction": "Unknown", "confidence": 0.0, "error": str(e)}

    def estimate_nutrients_zero_shot(self, text: str) -> Dict[str, Any]:
        """Estimate per-serving nutrition directly with the configured OpenRouter LLM."""
        if not self.client:
            return {"error": "OpenRouter API key required for nutrition estimation"}
        prompt = f"""
Estimate the nutrition per serving for the recipe below. Infer realistic ingredient
quantities and serving count when they are missing. Return only one raw JSON object
with exactly these numeric, non-negative fields: kcal, fat, saturates, carbs,
sugars, fibre, protein, salt. kcal is in kcal and every other value is in grams.
Do not include units, ranges, markdown, commentary, or extra fields.

Recipe:
{text}
"""
        raw_text = self._generate_text(prompt)
        if not raw_text:
            return {"error": "OpenRouter nutrition estimation failed"}
        try:
            payload = json.loads(raw_text[raw_text.find("{"):raw_text.rfind("}") + 1])
            values = {
                target: round(max(0.0, float(payload[target])), 1)
                for target in NUTRIENT_TARGETS
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Invalid OpenRouter nutrition response: %s", exc)
            return {"error": "OpenRouter returned an invalid nutrition estimate"}
        return {"per_serving": values, "method": "openrouter_zero_shot"}

    def analyze_recipe(self, recipe_description: str) -> Dict[str, Any]:
        """Analyze a recipe with local classifiers and zero-shot LLM nutrition."""
        try:
            enhanced_recipe = self.enhance_recipe_description(recipe_description)
            formatted_text = self.format_recipe_text(enhanced_recipe)
            class_embedding = self.get_text_embedding(formatted_text, "classification")
            return {
                "original_description": recipe_description,
                "enhanced_recipe": enhanced_recipe,
                "difficulty": self.predict_difficulty_from_embedding(class_embedding),
                "meal_type": self.predict_meal_type_from_embedding(class_embedding),
                "time_class": self.predict_time_class_from_embedding(class_embedding),
                "nutrients": self.estimate_nutrients_zero_shot(formatted_text),
            }
        except Exception as e:
            return {
                "error": f"Error analyzing recipe: {str(e)}",
                "original_description": recipe_description
            }

    def generate_llm_interpretation(self, analysis_results: Dict[str, Any]) -> str:
        """Use LLM to interpret and explain the model results."""
        difficulty = analysis_results.get('difficulty', {})
        meal_type = analysis_results.get('meal_type', {})
        enhanced_recipe = analysis_results.get('enhanced_recipe', {})
        prompt = f"""
        Please provide a comprehensive analysis of this recipe based on ML model predictions:

        **Recipe Information:**
        - Name: {enhanced_recipe.get('name', 'N/A')}
        - Ingredients: {enhanced_recipe.get('ingredients', 'N/A')}
        - Steps: {enhanced_recipe.get('steps', 'N/A')}

        **ML Model Predictions:**
        - Difficulty: {difficulty.get('prediction', 'Unknown')} (confidence: {difficulty.get('confidence', 0):.1%})
        - Meal Type: {meal_type.get('prediction', 'Unknown')} (confidence: {meal_type.get('confidence', 0):.1%})

        Please provide:
        1. A summary of the recipe's characteristics
        2. Explanation of why it's classified as this difficulty level
        3. Why it fits this meal type category
        4. Any cooking tips or variations to consider

        Keep the analysis informative but concise.
        """
        text = self._generate_text(prompt)
        if text is None:
            return "LLM interpretation not available (no API key provided)."
        return text
