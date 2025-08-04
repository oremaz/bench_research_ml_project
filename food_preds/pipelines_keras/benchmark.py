from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from pipelines_keras.base import GeneralPipeline

class BenchmarkRunner:
    """
    Runs grid search over models, augmentations, and metrics, collects results in a DataFrame.
    """
    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        augmentations: List[Optional[Any]],
        metrics: List[Any],
        task_type: str = "classification",
        device: str = "/CPU:0",
        epochs: int = 10,
        batch_size: int = 32,
        early_stopping: Optional[int] = None,
        use_class_weights: bool = True,
    ):
        self.model_configs = model_configs
        self.augmentations = augmentations
        self.metrics = metrics
        self.task_type = task_type
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.use_class_weights = use_class_weights

    def run(self, X_train, y_train, X_val, y_val) -> pd.DataFrame:
        results = []
        for model_cfg in tqdm(self.model_configs, desc="Models"):
            model_name = model_cfg["name"]
            model_class = model_cfg["class"]
            model_params = model_cfg.get("params", {})
            for aug in tqdm(self.augmentations, desc="Augmentations", leave=False):
                aug_name = aug.__name__ if aug is not None else "none"
                # Apply augmentation if any
                if aug is not None:
                    X_aug, y_aug = aug(X_train, y_train)
                else:
                    X_aug, y_aug = X_train, y_train
                # Instantiate model
                model = model_class(**model_params)
                # Choose loss and optimizer
                import tensorflow as tf
                if self.task_type == "classification":
                    loss_fn = "sparse_categorical_crossentropy"
                    optimizer = "adam"
                    metrics = ["accuracy"]
                else:
                    loss_fn = "mse"
                    optimizer = "adam"
                    metrics = []
                # Create pipeline
                pipeline = GeneralPipeline(
                    model=model,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    augmentations=None,  # Already applied
                    metrics=metrics,
                    task_type=self.task_type,
                    device=self.device,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    early_stopping=self.early_stopping,
                    use_class_weights=self.use_class_weights,
                )
                # Train
                pipeline.fit(X_aug, y_aug, X_val, y_val)
                # Evaluate
                eval_results = pipeline.evaluate(X_val, y_val)
                for metric_name, score in eval_results.items():
                    results.append({
                        "model": model_name,
                        "augmentation": aug_name,
                        "metric": metric_name,
                        "score": score
                    })
        return pd.DataFrame(results)
