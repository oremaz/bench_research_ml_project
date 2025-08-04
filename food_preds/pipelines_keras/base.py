import numpy as np
from typing import Callable, List, Optional, Dict, Any, Tuple, Union
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

class GeneralPipeline:
    """
    General pipeline for Keras models supporting classification and regression.
    """
    def __init__(
        self,
        model: tf.keras.Model,
        loss_fn: Union[str, Callable],
        optimizer: Union[str, Callable],
        augmentations: Optional[Callable] = None,
        metrics: Optional[List[Union[str, Callable]]] = None,
        task_type: str = "classification",
        device: str = "/CPU:0",
        epochs: int = 10,
        batch_size: int = 32,
        early_stopping: Optional[int] = None,
        use_class_weights: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.augmentations = augmentations
        self.metrics = metrics or []
        self.task_type = task_type
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.use_class_weights = use_class_weights
        self.class_weights = None
        self.history = {"train_loss": [], "val_loss": [], "metrics": []}

    def _compute_class_weights(self, y: np.ndarray) -> Optional[Dict[int, float]]:
        """Compute class weights for imbalanced classification."""
        if not self.use_class_weights or self.task_type != "classification":
            return None
        
        # Get unique classes and their counts
        unique_classes = np.unique(y)
        if len(unique_classes) <= 1:
            return None  # No imbalance if only one class
        
        # Compute balanced class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y
        )
        
        # Convert to dictionary format for Keras
        weights_dict = {int(k): float(v) for k, v in zip(unique_classes, class_weights)}
        print(f"Computed class weights: {weights_dict}")
        return weights_dict

    def prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, train: bool = True) -> tf.data.Dataset:
        if self.augmentations and train:
            X, y = self.augmentations(X, y)
        X = np.array(X, dtype=np.float32)
        if y is not None:
            y = np.array(y)
            ds = tf.data.Dataset.from_tensor_slices((X, y))
        else:
            ds = tf.data.Dataset.from_tensor_slices(X)
        if train:
            ds = ds.shuffle(buffer_size=len(X))
        ds = ds.batch(self.batch_size)
        return ds

    def _create_loss_function(self) -> Union[str, Callable]:
        """Create loss function with class weights if applicable."""
        import tensorflow as tf
        
        # If loss_fn is a string, instantiate it
        if isinstance(self.loss_fn, str):
            # For Keras, most loss functions are strings that work with class_weight parameter
            # So we return the string as-is, and let Keras handle weights via class_weight
            return self.loss_fn
        else:
            # If loss_fn is already a callable, return as-is
            return self.loss_fn

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        # Compute class weights from training data (before augmentation)
        self.class_weights = self._compute_class_weights(y_train)
        
        # Create loss function with weights if applicable
        loss_fn = self._create_loss_function()
        
        train_ds = self.prepare_data(X_train, y_train, train=True)
        val_ds = self.prepare_data(X_val, y_val, train=False) if X_val is not None and y_val is not None else None
        callbacks = []
        if self.early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=self.early_stopping, restore_best_weights=True))
        
        with tf.device(self.device):
            self.model.compile(optimizer=self.optimizer, loss=loss_fn, metrics=self.metrics)
            
            # Use class weights in fit if available
            fit_kwargs = {
                'x': train_ds,
                'validation_data': val_ds,
                'epochs': self.epochs,
                'callbacks': callbacks,
                'verbose': 1
            }
            
            if self.class_weights is not None:
                fit_kwargs['class_weight'] = self.class_weights
            
            history = self.model.fit(**fit_kwargs)
        self.history = history.history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        ds = self.prepare_data(X, y, train=False)
        results = self.model.evaluate(ds, verbose=0, return_dict=True)
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        ds = self.prepare_data(X, train=False)
        preds = self.model.predict(ds, verbose=0)
        if self.task_type == "classification":
            if preds.shape[-1] > 1:
                return np.argmax(preds, axis=1)
            else:
                return (preds > 0.5).astype(int).flatten()
        return preds.flatten() if preds.ndim > 1 and preds.shape[1] == 1 else preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")
        ds = self.prepare_data(X, train=False)
        preds = self.model.predict(ds, verbose=0)
        if preds.shape[-1] == 1:
            return np.hstack([1 - preds, preds])
        return preds
