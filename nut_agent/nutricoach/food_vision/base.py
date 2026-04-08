"""
Base class and data models for food image analysis.
"""

import time
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class FoodItem:
    """A single detected food item with nutritional estimates."""
    name: str
    quantity_grams: float
    confidence: float = 0.0
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    portion_description: str = ""  # e.g. "1 cup", "half plate"


@dataclass
class FoodAnalysisResult:
    """Complete result from analyzing a food image."""
    method: str
    food_items: List[FoodItem] = field(default_factory=list)
    total_calories: float = 0.0
    total_protein_g: float = 0.0
    total_carbs_g: float = 0.0
    total_fat_g: float = 0.0
    elapsed_seconds: float = 0.0
    raw_response: str = ""
    error: Optional[str] = None

    def compute_totals(self):
        """Recompute totals from individual food items."""
        self.total_calories = sum(f.calories for f in self.food_items)
        self.total_protein_g = sum(f.protein_g for f in self.food_items)
        self.total_carbs_g = sum(f.carbs_g for f in self.food_items)
        self.total_fat_g = sum(f.fat_g for f in self.food_items)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "food_items": [
                {
                    "name": f.name,
                    "quantity_grams": f.quantity_grams,
                    "confidence": f.confidence,
                    "calories": f.calories,
                    "protein_g": f.protein_g,
                    "carbs_g": f.carbs_g,
                    "fat_g": f.fat_g,
                    "portion_description": f.portion_description,
                }
                for f in self.food_items
            ],
            "total_calories": self.total_calories,
            "total_protein_g": self.total_protein_g,
            "total_carbs_g": self.total_carbs_g,
            "total_fat_g": self.total_fat_g,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "error": self.error,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        if self.error:
            return f"[{self.method}] Error: {self.error}"
        lines = [f"[{self.method}] ({self.elapsed_seconds:.1f}s)"]
        for f in self.food_items:
            lines.append(
                f"  - {f.name}: {f.quantity_grams:.0f}g "
                f"({f.calories:.0f} kcal, P:{f.protein_g:.1f}g, "
                f"C:{f.carbs_g:.1f}g, F:{f.fat_g:.1f}g)"
            )
        lines.append(
            f"  TOTAL: {self.total_calories:.0f} kcal, "
            f"P:{self.total_protein_g:.1f}g, C:{self.total_carbs_g:.1f}g, "
            f"F:{self.total_fat_g:.1f}g"
        )
        return "\n".join(lines)


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Infer MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")


class FoodAnalyzer(ABC):
    """Abstract base class for all food image analyzers."""

    method_name: str = "base"

    @abstractmethod
    def analyze(self, image_path: str) -> FoodAnalysisResult:
        """Analyze a food image and return structured results."""
        ...

    def _timed_analyze(self, image_path: str) -> FoodAnalysisResult:
        """Wrapper that times the analysis."""
        start = time.time()
        result = self.analyze(image_path)
        result.elapsed_seconds = time.time() - start
        return result
