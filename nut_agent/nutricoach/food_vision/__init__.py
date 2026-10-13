"""
Food Vision module: multiple methods for food image analysis.

Primary benchmark: single-shot VLM, RGB regression and specialized Food-R1.
"""

from .base import FoodAnalysisResult, FoodAnalyzer
from .nutrition_db import NutritionDB

__all__ = ["FoodAnalysisResult", "FoodAnalyzer", "NutritionDB"]
