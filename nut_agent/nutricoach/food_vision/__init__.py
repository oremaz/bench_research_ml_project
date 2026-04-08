"""
Food Vision module — multiple methods for food image analysis.

Methods:
1. RF-DETR fine-tuning pipeline (detection + nutrition DB)
2. Pure vLLM via OpenRouter (Claude Opus chained prompts)
3. CLIP zero-shot + LLM ensemble
4. RAG-enhanced VLM (DietAI24-inspired)
"""

from .base import FoodAnalysisResult, FoodAnalyzer
from .nutrition_db import NutritionDB

__all__ = ["FoodAnalysisResult", "FoodAnalyzer", "NutritionDB"]
