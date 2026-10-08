"""
Method 1: Offline food detection with food-specific portion assumptions.

A food-fine-tuned RF-DETR checkpoint identifies foods. Local serving defaults
and nutrition entries produce automatic estimates without user measurements.
Bounding boxes and detection counts are not converted to mass.
Generic COCO inference is not supported.
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional, List, Dict

from .base import FoodAnalyzer, FoodAnalysisResult, FoodItem
from .nutrition_db import NutritionDB, default_portion_grams

logger = logging.getLogger(__name__)
FOOD_VISION_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = FOOD_VISION_DIR / "results" / "rf_detr_food" / "checkpoint_best_total.pth"
DEFAULT_NUTRITION_NAMES = {
    "rice": "white rice", "chicken": "chicken breast", "beef": "beef steak",
    "carrot": "carrot", "potato": "potato", "pasta": "pasta",
}


class RFDETRAnalyzer(FoodAnalyzer):
    """Use a food checkpoint and local serving defaults for automatic predictions."""

    method_name = "rf_detr"

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        model_size: str = "base",
        num_classes: Optional[int] = None,
        portion_weights: Optional[Dict[str, float]] = None,
        labels_path: Optional[str] = None,
        nutrition_names: Optional[Dict[str, str]] = None,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        self.num_classes = num_classes
        self.portion_weights = {
            name.lower().strip(): float(grams)
            for name, grams in (portion_weights or {}).items()
        }
        if any(not math.isfinite(g) or g <= 0 for g in self.portion_weights.values()):
            raise ValueError("Measured food weights must be finite and positive")
        self.labels_path = Path(labels_path) if labels_path else self.model_path.parent / "food_classes.json"
        self.class_names = {}
        self.nutrition_names = {name.lower().strip(): entry.lower().strip()
                                for name, entry in (nutrition_names or {}).items()}
        self.nutrition_db = NutritionDB()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Food-fine-tuned RF-DETR checkpoint required: {self.model_path}. "
                "Generic COCO fallback is disabled."
            )
        if not self.labels_path.is_file():
            raise FileNotFoundError(f"Food class mapping required: {self.labels_path}")
        with self.labels_path.open() as source:
            labels = json.load(source)
        if "categories" in labels:
            categories = sorted(labels["categories"], key=lambda item: int(item["id"]))
            self.class_names = {index: item["name"] for index, item in enumerate(categories)}
        else:
            self.class_names = {int(key): name for key, name in labels.items()}
        if not self.class_names or any(not isinstance(name, str) or not name.strip()
                                       for name in self.class_names.values()):
            raise ValueError("Food class mapping must contain non-empty labels")
        class_count = self.num_classes or len(self.class_names)
        if class_count != len(self.class_names):
            raise ValueError("num_classes must match the food class mapping")
        from rfdetr import RFDETRBase, RFDETRLarge

        ModelClass = RFDETRLarge if self.model_size == "large" else RFDETRBase
        self._model = ModelClass(pretrain_weights=str(self.model_path), num_classes=class_count)

    def analyze(self, image_path: str) -> FoodAnalysisResult:
        start = time.time()
        result = FoodAnalysisResult(method=self.method_name)
        try:
            self._load_model()
            detections = self._model.predict(image_path, threshold=self.confidence_threshold)
            result.food_items = self._parse_detections(detections)
            result.compute_totals()
        except Exception as exc:
            logger.error("RF-DETR analysis failed: %s", exc)
            result.error = str(exc)
        result.elapsed_seconds = time.time() - start
        return result

    def _parse_detections(self, detections) -> List[FoodItem]:
        confidences = {}
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            if int(class_id) not in self.class_names:
                raise ValueError(f"No food label for detected class {class_id}")
            name = self.class_names[int(class_id)].lower().strip()
            confidences[name] = max(confidences.get(name, 0.0), float(confidence))
        if not confidences:
            raise ValueError("No food detected by the food-fine-tuned checkpoint")

        food_items = []
        for name, confidence in confidences.items():
            query = self.nutrition_names.get(name, DEFAULT_NUTRITION_NAMES.get(name, name))
            db_name, info = self.nutrition_db.lookup_with_name(query)
            grams = self.portion_weights.get(name, default_portion_grams(db_name or name))
            nutrients = self.nutrition_db.enrich_food_item(db_name or query, grams)
            portion_source = "supplied total weight" if name in self.portion_weights else "assumed serving"
            nutrition_source = f"nutrition entry: {db_name}" if info else "generic average nutrition fallback"
            food_items.append(FoodItem(
                name=name,
                quantity_grams=grams,
                confidence=confidence,
                calories=nutrients["calories"],
                protein_g=nutrients["protein_g"],
                carbs_g=nutrients["carbs_g"],
                fat_g=nutrients["fat_g"],
                portion_description=f"{grams:g}g {portion_source}; {nutrition_source}",
            ))
        return food_items


class RFDETRFoodTrainer:
    """
    Fine-tuning helper for RF-DETR on food detection datasets.

    Supports:
    - FoodSeg103 (103 ingredient classes, segmentation masks)
    - UEC-FoodPix Complete (100 food categories)
    - Custom Roboflow datasets (COCO format)

    Usage:
        trainer = RFDETRFoodTrainer()
        trainer.train(
            dataset_dir="./food_coco/",
            epochs=50,
            output_dir="./rf_detr_food_weights/",
        )
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size

    def train(
        self,
        dataset_dir: str,
        epochs: int = 50,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        lr: float = 1e-4,
        output_dir: str = "./rf_detr_food_weights",
        resume: Optional[str] = None,
    ) -> str:
        """
        Fine-tune RF-DETR on a food detection dataset in COCO format.

        Expected dataset structure:
            dataset_dir/
              train/
                images/
                _annotations.coco.json
              valid/
                images/
                _annotations.coco.json
              test/  (optional)
                images/
                _annotations.coco.json

        Returns:
            Path to the best checkpoint.
        """
        try:
            from rfdetr import RFDETRBase, RFDETRLarge
        except ImportError:
            raise ImportError("rfdetr is required. Install with: pip install rfdetr")

        ModelClass = RFDETRLarge if self.model_size == "large" else RFDETRBase
        model = ModelClass()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting RF-DETR fine-tuning: %s, %d epochs, bs=%d, lr=%s",
            self.model_size, epochs, batch_size, lr,
        )

        model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            output_dir=str(output_path),
            **({"resume": resume} if resume else {}),
        )

        with (Path(dataset_dir) / "train" / "_annotations.coco.json").open() as source:
            categories = json.load(source)["categories"]
        with (output_path / "food_classes.json").open("w") as target:
            categories = sorted(categories, key=lambda item: int(item["id"]))
            json.dump({str(index): item["name"] for index, item in enumerate(categories)}, target, indent=2)

        # Find best checkpoint
        checkpoints = sorted(output_path.glob("*.pth"))
        best = str(checkpoints[-1]) if checkpoints else str(output_path / "model_final.pth")
        logger.info("Training complete. Best checkpoint: %s", best)
        return best

    @staticmethod
    def download_food_dataset(
        dataset_name: str = "food-detection",
        workspace: str = "roboflow-universe",
        version: int = 1,
        output_dir: str = "./food_coco",
        api_key: Optional[str] = None,
    ) -> str:
        """
        Download a food detection dataset from Roboflow Universe in COCO format.

        Popular food datasets on Roboflow:
        - "food-detection" (general food items)
        - "food-items-detection" (packaged foods)
        - "fruits-and-vegetables" (produce)

        Returns:
            Path to the downloaded dataset directory.
        """
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError(
                "roboflow is required. Install with: pip install roboflow"
            )

        rf = Roboflow(api_key=api_key or "")
        project = rf.workspace(workspace).project(dataset_name)
        dataset = project.version(version).download("coco", location=output_dir)
        logger.info("Dataset downloaded to %s", output_dir)
        return output_dir
