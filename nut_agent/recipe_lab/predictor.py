"""
ML-powered recipe analysis using text embeddings and trained LightGBM models.

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

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "ml_pipeline"))

from pipelines_torch.base import GeneralPipelineSklearn
from pipelines_torch.models import MODEL_REGISTRY
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
    "nutrients": {
        "path_start": "nutrients_train",
        "targets": ["kcal", "fat", "saturates", "carbs", "sugars", "fibre", "protein", "salt"],
    },
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TEXT_MODEL = os.environ.get("OPENROUTER_TEXT_MODEL", "google/gemma-4-31b-it:free")


class LocalEmbedder:
    """Sentence-transformers text encoder shared by training and inference."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def is_jina(self) -> bool:
        return JINA_MODEL_TAG in self.model_name

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
            self._model = SentenceTransformer(self.model_name, device=self.device, **kwargs)
        return self._model

    def embed(self, texts: Union[str, List[str]], batch_size: int = 64) -> np.ndarray:
        model = self._load()
        single = isinstance(texts, str)
        inputs = [texts] if single else list(texts)
        if self.is_jina:
            inputs = [JINA_DOC_PREFIX + t for t in inputs]
            batch_size = min(batch_size, 16)
        emb = model.encode(
            inputs,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        return emb[0] if single else emb


class HFNutrientRegressor:
    """Fine-tuned bge-base regression head predicting per-serving nutrients
    from raw recipe text (see ml_pipeline/train_recipe_models.py)."""

    def __init__(self, model_dir: Union[str, Path], device: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.device = device
        self._model = None
        self._tok = None
        with open(self.model_dir / "regressor_meta.json") as f:
            meta = json.load(f)
        self.targets = meta["targets"]
        self.mu = np.array(meta["mu"], dtype=np.float64)
        self.sd = np.array(meta["sd"], dtype=np.float64)
        self.max_len = meta.get("max_len", 512)

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tok = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir).to(self.device).eval()

    def predict(self, text: str) -> np.ndarray:
        import torch

        self._load()
        enc = self._tok(text, truncation=True, max_length=self.max_len,
                        padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**enc).logits.cpu().numpy()[0]
        return out * self.sd + self.mu


class FoodModelPredictor:
    """
    Wrapper class to load and use the trained food prediction models with text embeddings.
    Supports: difficulty, meal type, time class, and per-serving nutrient prediction.
    """

    def __init__(self, models_path: str = None, api_key: str = None, device: Optional[str] = None):
        if models_path is None:
            models_path = REPO_ROOT / "ml_pipeline" / "results"
        self.models_path = Path(models_path)

        self.meta = self._load_meta()
        self.tasks = self.meta.get("tasks", DEFAULT_TASKS)
        emb_meta = self.meta.get("embedding", {})
        self.embedder = LocalEmbedder(
            model_name=emb_meta.get("model", LOCAL_EMBEDDING_MODEL),
            device=device,
        )
        self.embedding_dim = emb_meta.get("dim", EMBEDDING_DIM)

        # Optional LLM backends for description enhancement / interpretation
        self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = None
        if self.google_api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.google_api_key)
            except Exception as e:
                logger.warning("Could not init Google genai client: %s", e)

        self.difficulty_pipeline = None
        self.meal_type_pipeline = None
        self.time_class_pipeline = None
        self.nutrients_pipeline = None
        self.nutrients_hf = None
        self._device = device

        self.difficulty_labels = self.tasks["difficulty"]["labels"]
        self.meal_type_labels = self.tasks["meal_type"]["labels"]
        self.time_class_labels = self.tasks["time_class"]["labels"]
        self.nutrient_targets = self.tasks.get("nutrients", DEFAULT_TASKS["nutrients"])["targets"]

        self._load_models()

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
        suffix = "classifier" if task_type == "classification" else "regressor"
        key = f"{model_name}_{suffix}"
        return MODEL_REGISTRY.get(key, MODEL_REGISTRY[f"lightgbm_{suffix}"])

    def _task_model_name(self, task: str) -> str:
        """Per-task model family from meta, falling back to the global name."""
        default_key = "nutrients_model_name" if task == "nutrients" else "model_name"
        return self.tasks[task].get("model_name", self.meta.get(default_key, "lightgbm"))

    def _load_models(self):
        """Load the trained models for difficulty, meal type, time class, and nutrients."""
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

        if "nutrients" in self.tasks:
            model_dir = self.tasks["nutrients"].get("model_dir")
            if model_dir and (self.models_path / model_dir).exists():
                try:
                    self.nutrients_hf = HFNutrientRegressor(
                        self.models_path / model_dir, device=self._device)
                except Exception as e:
                    logging.error(f"Error loading fine-tuned nutrients model: {e}")
            try:
                nutrients_model_name = self._task_model_name("nutrients")
                path_start = str(self.models_path / self.tasks["nutrients"]["path_start"])
                model = load_model_by_name(
                    self._registry_class(nutrients_model_name, "regression"),
                    nutrients_model_name,
                    {},
                    path_start=path_start,
                    task_type="regression",
                )
                self.nutrients_pipeline = GeneralPipelineSklearn(model=model, task_type="regression")
            except Exception as e:
                logging.error(f"Error loading nutrients model: {e}")

    # --- LLM helpers ---

    def _generate_text(self, prompt: str) -> Optional[str]:
        """Generate text with Google Gemini if configured, else OpenRouter free tier."""
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                logger.warning("Gemini generation failed: %s", e)

        if self.openrouter_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=self.openrouter_api_key)
                response = client.chat.completions.create(
                    model=OPENROUTER_TEXT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
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

    def predict_nutrients_from_embedding(self, embedding: List[float]) -> Dict[str, Any]:
        """Predict per-serving nutrient values (kcal, fat, ...) from text embedding."""
        if self.nutrients_pipeline is None:
            return {"error": "Model not loaded"}
        try:
            embedding_array = np.array(embedding).reshape(1, -1)
            preds = np.asarray(self.nutrients_pipeline.model.predict(embedding_array))
            preds = np.clip(preds.reshape(-1), 0, None)
            return {
                "per_serving": {
                    t: round(float(v), 1) for t, v in zip(self.nutrient_targets, preds)
                }
            }
        except Exception as e:
            return {"error": str(e)}

    def predict_nutrients_from_text(self, text: str,
                                    embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """Predict per-serving nutrients from recipe text.

        Uses the fine-tuned bge regression head when available (much better
        test MAE than embedding-based regression), else the registry model
        on the provided or freshly computed embedding.
        """
        if self.nutrients_hf is not None:
            try:
                preds = np.clip(self.nutrients_hf.predict(text), 0, None)
                return {
                    "per_serving": {
                        t: round(float(v), 1)
                        for t, v in zip(self.nutrients_hf.targets, preds)
                    },
                    "method": "bge_finetune",
                }
            except Exception as e:
                logger.warning("Fine-tuned nutrients prediction failed: %s", e)
        if embedding is None:
            embedding = self.get_text_embedding(text)
        result = self.predict_nutrients_from_embedding(embedding)
        if "per_serving" in result:
            result["method"] = self._task_model_name("nutrients")
        return result

    def analyze_recipe(self, recipe_description: str) -> Dict[str, Any]:
        """Perform complete analysis: enhance -> embed -> predict difficulty/meal_type/time_class/nutrients."""
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
                "nutrients": self.predict_nutrients_from_text(
                    formatted_text, embedding=class_embedding),
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
