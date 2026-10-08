"""
Food Vision module: multiple methods for food image analysis.

Primary benchmark: single-shot VLM, RGB regression, specialized Food-R1,
and RGB geometry with nutrition lookup. Legacy research analyzers are retained.
"""

from .base import FoodAnalysisResult, FoodAnalyzer
from .nutrition_db import NutritionDB

__all__ = ["FoodAnalysisResult", "FoodAnalyzer", "NutritionDB"]
