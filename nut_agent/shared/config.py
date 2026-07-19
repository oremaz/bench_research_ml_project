"""
Configuration file for the nutritionist agent.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
ML_PIPELINE_DIR = BASE_DIR.parent / "ml_pipeline"
MODELS_DIR = ML_PIPELINE_DIR / "results"
SECRETS_DIR = BASE_DIR / "secrets"

# Model configurations
MODEL_CONFIGS = {
    "difficulty": {
        "model_paths": [
            MODELS_DIR / "difficulty_train" / "xgboost_classifier.pt",
            MODELS_DIR / "difficulty_train" / "lightgbm_classifier.pt",
            MODELS_DIR / "difficulty_train" / "mlp_classifier.pt",
        ],
        "labels": ["Easy", "More effort", "A challenge"],
        "default_confidence": 0.8
    },
    "meal_type": {
        "model_paths": [
            MODELS_DIR / "meal_train" / "xgboost_classifier.pt",
            MODELS_DIR / "meal_train" / "lightgbm_classifier.pt",
            MODELS_DIR / "meal_train" / "mlp_classifier.pt",
        ],
        "labels": ["breakfast", "lunch", "dinner", "snack", "dessert"],
        "default_confidence": 0.8
    },
    "nutrients": {
        "model_paths": [
            MODELS_DIR / "nutrient_train" / "xgboost_regressor.pt",
            MODELS_DIR / "nutrient_train" / "lightgbm_regressor.pt",
            MODELS_DIR / "nutrient_train" / "mlp_regressor.pt",
        ],
        "output_names": ["calories", "protein", "carbs", "fat", "sodium"]
    }
}

# Feature configurations
FEATURE_COLUMNS = [
    'calories', 'protein', 'fat', 'sodium', 'rating',
    'prep_time', 'cook_time', 'total_time', 'servings'
]

DEFAULT_FEATURE_VALUES = {
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

# Nutrition calculation constants
BMR_CONSTANTS = {
    'male': {'base': 10, 'weight': 6.25, 'height': 5, 'age': 5},
    'female': {'base': 10, 'weight': 6.25, 'height': 5, 'age': -161}
}

ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'light': 1.375,
    'moderate': 1.55,
    'active': 1.725,
    'very_active': 1.9
}

WEIGHT_GOAL_ADJUSTMENTS = {
    'lose': -500,    # 500 calorie deficit
    'maintain': 0,   # No adjustment
    'gain': 500      # 500 calorie surplus
}

# Macronutrient ratios
MACRO_RATIOS = {
    'protein_per_kg': 1.6,      # grams per kg body weight
    'fat_percentage': 0.25,      # 25% of total calories
    'protein_calories_per_g': 4,
    'carb_calories_per_g': 4,
    'fat_calories_per_g': 9
}

# Water intake calculation
WATER_ML_PER_KG = 35  # ml per kg body weight

# LLM configurations
GEMINI_MODEL = "gemini-2.0-flash"
MAX_RECURSION_LIMIT = 50

# OpenRouter fallback (used when GOOGLE_API_KEY is not configured).
# Defaults are free-tier models; override via env for paid tiers.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_AGENT_MODEL = os.environ.get("OPENROUTER_AGENT_MODEL", "google/gemma-4-31b-it:free")
OPENROUTER_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "google/gemma-4-31b-it:free")
# Server-side fallbacks when the primary free model is rate-limited upstream.
# All support tool calling; the vision list additionally supports images.
OPENROUTER_AGENT_FALLBACKS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "google/gemma-4-26b-a4b-it:free",
]
OPENROUTER_VISION_FALLBACKS = [
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
]

# Streamlit configurations
STREAMLIT_CONFIG = {
    'page_title': "AI Nutritionist Agent",
    'page_icon': "🥗",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

# Prompt templates
PROMPTS = {
    'profile_extraction': """
    Extract the following information from the user's message and format as structured data:
    - height_cm: height in centimeters
    - weight_kg: weight in kilograms
    - age: age in years
    - gender: 'male' or 'female'
    - activity_level: 'sedentary', 'light', 'moderate', 'active', or 'very_active'
    - weight_goal: 'lose', 'maintain', or 'gain'
    - dietary_restrictions: list of restrictions (vegetarian, vegan, gluten-free, etc.)
    - preferred_cuisines: list of preferred cuisines

    User message: {user_message}

    If any information is missing, ask the user to provide it.
    """,

    'daily_constraints_extraction': """
    Extract daily constraints and previous day's results from the user's message:

    Daily constraints to look for:
    - Available cooking time
    - Dietary preferences for today
    - Special requirements or restrictions
    - Preferred meal times

    Previous day's results to look for:
    - Weight change
    - Energy levels
    - Adherence to previous plan
    - How they felt

    User message: {user_message}
    """,

    'meal_suggestions': """
    Generate {num_suggestions} meal suggestions for {meal_type} with the following criteria:
    - Target calories: approximately {target_calories}
    - Dietary restrictions: {dietary_restrictions}
    - Preferred cuisines: {preferred_cuisines}
    - Maximum cooking time: {max_cooking_time} minutes

    For each suggestion, provide:
    1. Recipe name
    2. Brief description
    3. Estimated prep and cook time
    4. Main ingredients
    5. Approximate nutritional info

    Format as a structured list.
    """,

    'nutritional_analysis': """
    Calculate the approximate nutritional values for this entire meal plan:

    {meal_plan}

    Provide:
    - Total calories
    - Total protein (g)
    - Total carbohydrates (g)
    - Total fat (g)

    Also provide a brief analysis of how well this meal plan aligns with the user's goals.
    """
}

# Error messages
ERROR_MESSAGES = {
    'model_not_loaded': "Model not loaded. Please check model files.",
    'api_key_missing': "Google API key is required.",
    'profile_incomplete': "Please complete your profile setup first.",
    'invalid_input': "Invalid input provided. Please try again.",
    'prediction_failed': "Failed to make prediction. Using default values."
}
