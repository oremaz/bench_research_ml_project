import os
import sys
import numpy as np
import torch
from google import genai
from google.genai import types
from google.api_core import retry
from typing import Dict, Any, List, Union
from pathlib import Path
from food_preds.pipelines_torch.base import GeneralPipelineSklearn
from food_preds.utils.utils import load_model
from food_preds.pipelines_torch.models import MODEL_REGISTRY
import json

sys.path.append(str(Path(__file__).parent))


class FoodModelPredictor:
    """
    Wrapper class to load and use the trained food prediction models with text embeddings.
    """
    
    def __init__(self, models_path: str = None, api_key: str = None):
        if models_path is None:
            models_path = Path(__file__).parent.parent / "food_preds" / "results"
        self.models_path = Path(models_path)
        
        # Initialize Google API for embeddings and text enhancement
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            
            # Set up retry configuration for API calls
            self.is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
            
            # Apply retry to generate_content if not already wrapped
            if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
                genai.models.Models.generate_content = retry.Retry(
                    predicate=self.is_retriable, timeout=600.0
                )(genai.models.Models.generate_content)
        else:
            self.client = None
            print("Warning: No Google API key provided. Text enhancement will be limited.")
        
        # Initialize model containers
        self.difficulty_model = None
        self.meal_type_model = None
        self.time_class_model = None
        self.difficulty_pipeline = None
        self.meal_type_pipeline = None
        self.time_class_pipeline = None
            
        # Load models
        self._load_models()
        
        # Define label mappings
        self.difficulty_labels = ['Easy', 'More effort', 'A challenge']
        self.meal_type_labels = ['breakfast', 'lunch', 'dinner', 'snack', 'dessert']
        
    def _load_models(self):
        """Load only LightGBM models for difficulty, meal type, and total time."""
        try:
            diff_path = str(self.models_path / 'difficulty_train')
            self.difficulty_model = load_model(MODEL_REGISTRY['lightgbm_classifier'], 'lightgbm_classifier', {}, path_start=diff_path)
            self.difficulty_pipeline = GeneralPipelineSklearn(model=self.difficulty_model, task_type='classification')

            meal_path = str(self.models_path / 'meal_train')
            self.meal_type_model = load_model(MODEL_REGISTRY['lightgbm_classifier'], 'lightgbm_classifier', {}, path_start=meal_path)
            self.meal_type_pipeline = GeneralPipelineSklearn(model=self.meal_type_model, task_type='classification')

            # Load time classification model
            timec_path = str(self.models_path / 'total_time_class_train')
            self.time_class_model = load_model(MODEL_REGISTRY['lightgbm_classifier'], 'lightgbm_classifier', {}, path_start=timec_path)
            self.time_class_pipeline = GeneralPipelineSklearn(model=self.time_class_model, task_type='classification')
        except Exception as e:
            print(f"Error loading models: {e}")
    def predict_time_class_from_embedding(self, embedding: list) -> dict:
        """
        Predict total time class from text embedding using LightGBM pipeline.
        """
        if self.time_class_pipeline is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        try:
            embedding_array = np.array(embedding).reshape(1, -1)
            probabilities = self.time_class_pipeline.model.predict_proba(embedding_array)
            predicted_class = int(np.argmax(probabilities[0]))
            confidence = float(probabilities[0][predicted_class])
            # Map class index to label
            class_labels = ['<15 min', '15-30 min', '30-60 min', '>60 min']
            # If more than 4 classes, fallback to string index
            if predicted_class < len(class_labels):
                label = class_labels[predicted_class]
            else:
                label = str(predicted_class)
            all_probs = {class_labels[i] if i < len(class_labels) else str(i): float(prob) for i, prob in enumerate(probabilities[0])}
            return {
                "prediction": label,
                "confidence": confidence,
                "class_index": predicted_class,
                "all_probabilities": all_probs
            }
        except Exception as e:
            return {"prediction": "Unknown", "confidence": 0.0, "error": str(e)}

    def enhance_recipe_description(self, user_description: str) -> Dict[str, Union[str, List[str]]]:
        """
        Use LLM to enhance user's recipe description and extract structured information.
        """
        # Fallback for when the LLM client isn't available
        fallback_data = {
            'name': user_description.split(',')[0].strip(),
            'ingredients': 'Not specified',
            'steps': 'Not specified'
        }

        if not self.client:
            return fallback_data
        
        try:
            # Refined prompt to explicitly forbid markdown
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
            
            # --- START: FIX ---
            # Find the start of the JSON object
            json_start_index = raw_text.find('{')
            # Find the end of the JSON object
            json_end_index = raw_text.rfind('}') + 1
            
            if json_start_index == -1 or json_end_index == 0:
                # If no JSON object is found, return the fallback
                return fallback_data
            
            # Extract the JSON string
            json_string = raw_text[json_start_index:json_end_index]
            # --- END: FIX ---

            # Parse the cleaned JSON string
            recipe_data = json.loads(json_string)
            return recipe_data

        except (json.JSONDecodeError, AttributeError, Exception) as e:
            # Catch potential errors during API call or JSON parsing
            print(f"An error occurred: {e}")
            return fallback_data
    
    def format_recipe_text(self, recipe_data: Dict[str, str]) -> str:
        """
        Format recipe data in the same way as the training data.
        Format: "name: [name] ingredients: [ingredients] steps: [steps]"
        """
        name = recipe_data.get('name', '').strip() if isinstance(recipe_data.get('name', ''), str) else str(recipe_data.get('name', ''))
        
        # Handle ingredients - could be string or list
        ingredients = recipe_data.get('ingredients', '')
        if isinstance(ingredients, list):
            ingredients = ', '.join(ingredients)
        elif isinstance(ingredients, str):
            ingredients = ingredients.strip()
        else:
            ingredients = str(ingredients)
        
        # Handle steps - could be string or list
        steps = recipe_data.get('steps', '')
        if isinstance(steps, list):
            steps = '. '.join(steps)
        elif isinstance(steps, str):
            steps = steps.strip()
        else:
            steps = str(steps)
        
        # Clean up the text
        def clean_text(text):
            return ' '.join(text.replace('\n', ' ').split())
        
        formatted_text = f"name: {clean_text(name)} ingredients: {clean_text(ingredients)} steps: {clean_text(steps)}"
        return formatted_text
    
    def get_text_embedding(self, text: str, task_type: str = "classification") -> List[float]:
        """
        Generate text embedding using Google's text-embedding-004 model.
        
        Args:
            text: Text to embed
            task_type: Either "classification" or "regression" based on the ML task
        """
        try:
            if not self.client:
                raise Exception("Google API client not available")
            
            # Create retry decorator for embed_content
            @retry.Retry(predicate=self.is_retriable, timeout=600.0)
            def embed_with_retry(text: str, task_type: str) -> List[float]:
                # Map task types to Google's expected values
                google_task_type = "classification" if task_type == "classification" else "SEMANTIC_SIMILARITY"
                
                # Try the newer API first
                try:
                    response = self.client.models.embed_content(
                        model="models/text-embedding-004",
                        contents=text,
                        config=types.EmbedContentConfig(
                            task_type=google_task_type,
                        ),
                    )
                    return response.embeddings[0].values
                except TypeError as e:
                    # If the newer API fails, try the older format
                    print(f"New API failed, trying older format: {e}")
                    response = self.client.models.embed_content(
                        model="models/text-embedding-004",
                        content=text,
                        task_type=google_task_type,
                    )
                    if hasattr(response, 'embeddings'):
                        return response.embeddings[0].values
                    else:
                        return response.embedding
            
            return embed_with_retry(text, task_type)
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a zero vector as fallback (this won't give good predictions)
            return [0.0] * 768  # Standard embedding dimension
        
    def predict_difficulty(self, recipe_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict recipe difficulty.
        
        Args:
            recipe_features: Dictionary with recipe features
            
        Returns:
            Dictionary with prediction and confidence
        """
        if self.difficulty_model is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        
        try:
            # Convert features to the expected format
            feature_array = self._prepare_features(recipe_features)
            
            # Make prediction
            if hasattr(self.difficulty_model, 'predict_proba'):
                probabilities = self.difficulty_model.predict_proba(feature_array)
                predicted_class = np.argmax(probabilities[0])
                confidence = float(probabilities[0][predicted_class])
            elif hasattr(self.difficulty_model, 'predict'):
                prediction = self.difficulty_model.predict(feature_array)
                predicted_class = int(prediction[0])
                confidence = 0.8  # Default confidence for models without probability
            else:
                # For raw boosting models
                import xgboost as xgb
                dtest = xgb.DMatrix(feature_array)
                prediction = self.difficulty_model.predict(dtest)
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
            
            return {
                "prediction": self.difficulty_labels[predicted_class],
                "confidence": confidence,
                "class_index": predicted_class
            }
            
        except Exception as e:
            return {"prediction": "Unknown", "confidence": 0.0, "error": str(e)}
    
    def predict_meal_type(self, recipe_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict meal type.
        
        Args:
            recipe_features: Dictionary with recipe features
            
        Returns:
            Dictionary with prediction and confidence
        """
        if self.meal_type_model is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        
        try:
            feature_array = self._prepare_features(recipe_features)
            
            if hasattr(self.meal_type_model, 'predict_proba'):
                probabilities = self.meal_type_model.predict_proba(feature_array)
                predicted_class = np.argmax(probabilities[0])
                confidence = float(probabilities[0][predicted_class])
            elif hasattr(self.meal_type_model, 'predict'):
                prediction = self.meal_type_model.predict(feature_array)
                predicted_class = int(prediction[0])
                confidence = 0.8
            else:
                import xgboost as xgb
                dtest = xgb.DMatrix(feature_array)
                prediction = self.meal_type_model.predict(dtest)
                predicted_class = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
            
            return {
                "prediction": self.meal_type_labels[predicted_class],
                "confidence": confidence,
                "class_index": predicted_class
            }
            
        except Exception as e:
            return {"prediction": "Unknown", "confidence": 0.0, "error": str(e)}
    
    # Removed predict_nutrients method - not needed for simplified version
    
    # ...removed predict_total_time method...
    
    def _prepare_features(self, recipe_features: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for model prediction.
        
        Args:
            recipe_features: Dictionary with recipe features
            
        Returns:
            Numpy array with features in the correct order
        """
        # Create feature array with default values
        feature_array = []
        
        for col in self.feature_columns:
            if col in recipe_features:
                feature_array.append(recipe_features[col])
            else:
                # Default values for missing features
                default_values = {
                    'calories': 300,
                    'protein': 10,
                    'fat': 10,
                    'sodium': 500,
                    'rating': 4.0,
                    'prep_time': 15,
                    'cook_time': 20,
                    'total_time': 35,
                    'servings': 4
                }
                feature_array.append(default_values.get(col, 0))
        
        return np.array(feature_array).reshape(1, -1)
    
    def analyze_recipe(self, recipe_description: str) -> Dict[str, Any]:
        """
        Perform complete analysis of a recipe using text embeddings.
        Only returns difficulty and grouped meal type using LightGBM models.
        """
        try:
            enhanced_recipe = self.enhance_recipe_description(recipe_description)
            formatted_text = self.format_recipe_text(enhanced_recipe)
            class_embedding = self.get_text_embedding(formatted_text, "classification")
            difficulty_result = self.predict_difficulty_from_embedding(class_embedding)
            meal_type_result = self.predict_meal_type_from_embedding(class_embedding)
            time_class_result = self.predict_time_class_from_embedding(class_embedding)
            analysis = {
                "original_description": recipe_description,
                "enhanced_recipe": enhanced_recipe,
                "difficulty": difficulty_result,
                "meal_type": meal_type_result,
                "time_class": time_class_result
            }
            return analysis
        except Exception as e:
            return {
                "error": f"Error analyzing recipe: {str(e)}",
                "original_description": recipe_description
            }
    
    def predict_difficulty_from_embedding(self, embedding: List[float]) -> Dict[str, Any]:
        """
        Predict cooking difficulty from text embedding using LightGBM pipeline.
        """
        if self.difficulty_pipeline is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        try:
            embedding_array = np.array(embedding).reshape(1, -1)
            # LightGBM: confidence is the probability of the predicted class
            probabilities = self.difficulty_pipeline.model.predict_proba(embedding_array)
            predicted_class = np.argmax(probabilities[0])
            confidence = float(probabilities[0][predicted_class])
            all_probs = {label: float(prob) for label, prob in zip(self.difficulty_labels, probabilities[0])}
            return {
                "prediction": self.difficulty_labels[predicted_class],
                "confidence": confidence,
                "class_index": predicted_class,
                "all_probabilities": all_probs
            }
        except Exception as e:
            return {"prediction": "Unknown", "confidence": 0.0, "error": str(e)}
    
    def predict_meal_type_from_embedding(self, embedding: List[float]) -> Dict[str, Any]:
        """
        Predict meal type from text embedding using LightGBM pipeline, grouped as 'breakfast' or 'lunch/dinner'.
        """
        if self.meal_type_pipeline is None:
            return {"prediction": "Unknown", "confidence": 0.0, "error": "Model not loaded"}
        try:
            embedding_array = np.array(embedding).reshape(1, -1)
            probabilities = self.meal_type_pipeline.model.predict_proba(embedding_array)
            predicted_class = np.argmax(probabilities[0])
            confidence = float(probabilities[0][predicted_class])
            label = self.meal_type_labels[predicted_class]
            if label == "breakfast":
                grouped_label = "breakfast"
            else:
                grouped_label = "lunch/dinner"
            return {
                "prediction": grouped_label,
                "confidence": confidence,
                "class_index": predicted_class
            }
        except Exception as e:
            return {"prediction": "Unknown", "confidence": 0.0, "error": str(e)}
    
    # ...removed predict_total_time_from_embedding method...

    def generate_llm_interpretation(self, analysis_results: Dict[str, Any]) -> str:
        """
        Use LLM to interpret and explain the model results.
        """
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
