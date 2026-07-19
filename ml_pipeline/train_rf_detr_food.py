"""
Fine-tune RF-DETR (base) on the FoodSeg103-derived COCO detection dataset for
the NutriCoach food photo analyzer (method 1, offline).

Uses nut_agent's RFDETRFoodTrainer so the training path is the same one
documented in nut_agent/nutricoach/food_vision/README.md.

Run (after prepare_foodseg103_coco.py):
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. uv run python ml_pipeline/train_rf_detr_food.py
"""

import sys
from pathlib import Path

ML_PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(ML_PIPELINE_DIR.parent / "nut_agent"))

from nutricoach.food_vision.rf_detr_analyzer import RFDETRFoodTrainer

DATASET_DIR = ML_PIPELINE_DIR / "data" / "foodseg103_coco"
OUTPUT_DIR = ML_PIPELINE_DIR / "results" / "rf_detr_food"
EPOCHS = 8
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 1e-4


def main():
    trainer = RFDETRFoodTrainer(model_size="base")
    best = trainer.train(
        dataset_dir=str(DATASET_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM,
        lr=LR,
        output_dir=str(OUTPUT_DIR),
    )
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
