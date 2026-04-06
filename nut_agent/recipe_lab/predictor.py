"""
ML-powered recipe analysis using text embeddings and trained LightGBM models.
Cleaned version of model_predictor.py with dead code removed.
"""

import os
import sys
import numpy as np
import torch
from google import genai
from google.genai import types
from google.api_core import retry
from typing import Dict, Any, List, Union
from pathlib import Path
import json

# Add parent paths for model imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from food_preds.pipelines_torch.base import GeneralPipelineSklearn
from food_preds.utils.utils import load_model
from food_preds.pipelines_torch.models import MODEL_REGISTRY


class FoodModelPredictor:
    """
    Wrapper class to load and use the trained food prediction models with text embeddings.
    Supports: difficulty, meal type, and time class prediction via LightGBM on embeddings.
    """

    def __init__(self, models_path: str = None, api_key: str = None):
        if models_path is None:
            models_path = Path(__file__).parent.parent.parent / "food_preds" / "results"
        self.models_path = Path(models_path)

        # Initialize Google API for embeddings and text enhancement
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

            if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
                genai.models.Models.generate_content = retry.Retry(
                    predicate=self.is_retriable, timeout=600.0
                )(genai.models.Models.generate_content)
        else:
            self.client = None

        # Initialize pipelines
        self.difficulty_pipeline = None
        self.meal_type_pipeline = None
        self.time_class_pipeline = None

        # Label mappings
        self.difficulty_labels = ['Easy', 'More effort', 'A challenge']
        self.meal_type_labels = ['breakfast', 'lunch', 'dinner', 'snack', 'dessert']
        self.time_class_labels = ['<15 min', '15-30 min', '30-60 min', '>60 min']

        self._load_models()

    def _load_models(self):
        """Load LightGBM models for difficulty, meal type, and time class."""
        try:
            diff_path = str(self.models_path / 'difficulty_train')
            model = load_model(MODEL_REGISTRY['lightgbm_classifier'], 'lightgbm_classifier', {}, path_start=diff_path)
            self.difficulty_pipeline = GeneralPipelineSklearn(model=model, task_type='classification')

            meal_path = str(self.models_path / 'meal_train')
            model = load_model(MODEL_REGISTRY['lightgbm_classifier'], 'lightgbm_classifier', {}, path_start=meal_path)
            self.meal_type_pipeline = GeneralPipelineSklearn(model=model, task_type='classification')

            timec_path = str(self.models_path / 'total_time_class_train')
            model = load_model(MODEL_REGISTRY['lightgbm_classifier'], 'lightgbm_classifier', {}, path_start=timec_path)
            self.time_class_pipeline = GeneralPipelineSklearn(model=model, task_type='classification')
        except Exception as e:
            import logging
            logging.error(f"Error loading models: {e}")

    def enhance_recipe_description(self, user_description: str) -> Dict[str, Union[str, List[str]]]:
        """Use LLM to enhance user's recipe description and extract structured information."""
        fallback_data = {
            'name': user_description.split(',')[0].strip(),
            'ingredients': 'Not specified',
            'steps': 'Not specified'
        }

        if not self.client:
            return fallback_data

        try:
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

            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )

            raw_text = response.text
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
        """Generate text embedding using Google's gemini-embedding-2-preview model."""
        try:
            if not self.client:
                raise Exception("Google API client not available")

            @retry.Retry(predicate=self.is_retriable, timeout=600.0)
            def embed_with_retry(text: str) -> List[float]:
                response = self.client.models.embed_content(
                    model="gemini-embedding-2-preview",
                    contents=text,
                )
                return response.embeddings[0].values

            return embed_with_retry(text)

        except Exception:
            return [0.0] * 3072

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
        """Predict meal type from text embedding, grouped as 'breakfast' or 'lunch/dinner'."""
        if self.meal_type_pipeline is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        try:
            embedding_array = np.array(embedding).reshape(1, -1)
            probabilities = self.meal_type_pipeline.model.predict_proba(embedding_array)
            predicted_class = np.argmax(probabilities[0])
            confidence = float(probabilities[0][predicted_class])
            label = self.meal_type_labels[predicted_class]
            grouped_label = "breakfast" if label == "breakfast" else "lunch/dinner"
            return {
                "prediction": grouped_label,
                "confidence": confidence,
                "class_index": int(predicted_class)
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

    def analyze_recipe(self, recipe_description: str) -> Dict[str, Any]:
        """Perform complete analysis: enhance -> embed -> predict difficulty/meal_type/time_class."""
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
            }
        except Exception as e:
            return {
                "error": f"Error analyzing recipe: {str(e)}",
                "original_description": recipe_description
            }

    def generate_llm_interpretation(self, analysis_results: Dict[str, Any]) -> str:
        """Use LLM to interpret and explain the model results."""
        if not self.client:
            return "LLM interpretation not available (no API key provided)."
        try:
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
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating LLM interpretation: {str(e)}"
