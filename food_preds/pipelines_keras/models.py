import numpy as np
from typing import Dict, Callable, Any, Optional
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from sklearn.ensemble import RandomForestClassifier

# --- MLPs ---
def build_mlp(input_dim: int, hidden_dims: list = [256, 128, 64], output_dim: int = 1, dropout: float = 0.3, batchnorm: bool = False, activation: str = 'relu') -> Model:
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for h in hidden_dims:
        x = layers.Dense(h, activation=activation)(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(output_dim, activation=None)(x)
    return Model(inputs, outputs)

def build_mlp_classifier(input_dim: int, hidden_dims: list = [256, 128, 64], num_classes: int = 2, dropout: float = 0.3, batchnorm: bool = False) -> Model:
    activation = 'softmax' if num_classes > 1 else 'sigmoid'
    model = build_mlp(input_dim, hidden_dims, num_classes, dropout, batchnorm)
    model.layers[-1].activation = tf.keras.activations.get(activation)
    return model

def build_mlp_regressor(input_dim: int, hidden_dims: list = [256, 128, 64], dropout: float = 0.3, batchnorm: bool = False) -> Model:
    return build_mlp(input_dim, hidden_dims, 1, dropout, batchnorm)

# --- RandomForest Wrapper ---
class SklearnRandomForestClassifierWrapper:
    """
    Wrapper for sklearn RandomForestClassifier to be compatible with GeneralPipeline.
    """
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)

# --- Transformer (Keras) ---
def build_transformer_classifier(input_dim: int, num_classes: int = 2, seq_len: int = 16, nhead: int = 4, num_layers: int = 2, ff_dim: int = 128, dropout: float = 0.1) -> Model:
    feature_dim = input_dim // seq_len
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((seq_len, feature_dim))(inputs)
    for _ in range(num_layers):
        x1 = layers.MultiHeadAttention(num_heads=nhead, key_dim=feature_dim, dropout=dropout)(x, x)
        x2 = layers.Add()([x, x1])
        x2 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(ff_dim, activation='relu')(x2)
        x3 = layers.Dense(feature_dim)(x3)
        x = layers.Add()([x2, x3])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')(x)
    return Model(inputs, outputs)

def build_transformer_regressor(input_dim: int, seq_len: int = 16, nhead: int = 4, num_layers: int = 2, ff_dim: int = 128, dropout: float = 0.1) -> Model:
    feature_dim = input_dim // seq_len
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((seq_len, feature_dim))(inputs)
    for _ in range(num_layers):
        x1 = layers.MultiHeadAttention(num_heads=nhead, key_dim=feature_dim, dropout=dropout)(x, x)
        x2 = layers.Add()([x, x1])
        x2 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(ff_dim, activation='relu')(x2)
        x3 = layers.Dense(feature_dim)(x3)
        x = layers.Add()([x2, x3])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation=None)(x)
    return Model(inputs, outputs)

# --- HuggingFace/LoRA Wrapper ---
class HuggingFaceLoRAWrapper:
    """
    Wrapper for any HuggingFace model with LoRA/PEFT fine-tuning.
    Accepts model_name, tokenizer_name, and LoRA config.
    Note: Input must be tokenized text.
    """
    def __init__(self, model_name: str, tokenizer_name: Optional[str] = None, lora_config: Optional[Dict[str, Any]] = None, num_labels: int = 2, device: str = 'cpu'):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if lora_config is None:
            lora_config = {
                'r': 8,
                'lora_alpha': 16,
                'target_modules': ["q_proj", "v_proj"],
                'lora_dropout': 0.05,
                'bias': "none",
                'task_type': TaskType.SEQ_CLS
            }
        lora_cfg = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, lora_cfg)
        self.device = device
        self.model.to(device)
    def fit(self, X, y, *args, **kwargs):
        from transformers import TrainingArguments, Trainer
        import torch
        from datasets import Dataset
        ds = Dataset.from_dict({'text': X, 'labels': y})
        def tokenize_fn(batch):
            return self.tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)
        ds = ds.map(tokenize_fn, batched=True)
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=kwargs.get('epochs', 2),
            per_device_train_batch_size=kwargs.get('batch_size', 8),
            logging_steps=10,
            report_to='none',
            fp16=True if self.device == 'cuda' else False
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=ds,
            eval_dataset=None,
            tokenizer=self.tokenizer
        )
        trainer.train()
    def predict(self, X):
        import torch
        inputs = self.tokenizer(X, return_tensors='pt', truncation=True, padding='max_length', max_length=256).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        return logits.cpu().numpy()
    def predict_proba(self, X):
        logits = self.predict(X)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

class LlamaCppClassifier:
    """
    Adapter-style classifier for llama.cpp models.
    Trains a dense layer on top of frozen llama.cpp outputs for classification.
    Note: This class uses torch for the dense head, as llama.cpp is not available in Keras/TensorFlow.
    """
    def __init__(self, gguf_path, num_classes=2, device="cpu"):
        from llama_cpp import Llama  # type: ignore
        import torch
        import torch.nn as nn
        self.llama = Llama(model_path=gguf_path, n_ctx=2048, logits_all=True)
        self.hidden_size = self.llama.config["hidden_size"]
        self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
        self.device = device

    def _get_logits(self, prompt):
        import torch
        output = self.llama(prompt)
        last_logits = torch.tensor(output["logits"][-1]).to(self.device)
        return last_logits

    def fit(self, prompts, labels, epochs=3, lr=1e-3):
        import torch
        import torch.optim as optim
        import torch.nn as nn
        self.classifier.train()
        optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            total_loss = 0
            for prompt, label in zip(prompts, labels):
                logits = self._get_logits(prompt)
                pred = self.classifier(logits.unsqueeze(0))
                loss = loss_fn(pred, torch.tensor([label]).to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} loss: {total_loss/len(prompts):.4f}")

    def predict(self, prompts):
        import torch
        self.classifier.eval()
        preds = []
        with torch.no_grad():
            for prompt in prompts:
                logits = self._get_logits(prompt)
                pred = self.classifier(logits.unsqueeze(0))
                preds.append(pred.argmax(dim=-1).item())
        return preds

    def predict_proba(self, prompts):
        import torch
        self.classifier.eval()
        probs = []
        with torch.no_grad():
            for prompt in prompts:
                logits = self._get_logits(prompt)
                pred = self.classifier(logits.unsqueeze(0))
                prob = torch.softmax(pred, dim=-1).cpu().numpy()[0]
                probs.append(prob)
        return probs

class HuggingFaceLoRAWrapperQuantized:
    """
    Wrapper for LoRA fine-tuning with HuggingFace, then quantization to llama.cpp GGUF format for low-memory inference.
    Workflow:
      1. Fine-tune with LoRA (HuggingFaceLoRAWrapper)
      2. Merge LoRA weights into base model
      3. Convert to GGUF (llama.cpp quantized)
      4. Load with LlamaCppClassifier for inference
    Requirements:
      - You must have the necessary conversion tools (e.g., llama.cpp's convert script, transformers, peft, etc.)
      - Quantization step may require manual intervention or a shell command
    Note: This class uses torch for the quantized inference head, as llama.cpp is not available in Keras/TensorFlow.
    """
    def __init__(self, model_name: str, tokenizer_name: str = '', lora_config: dict = {}, num_labels: int = 2, device: str = 'cpu', gguf_path: str = "quantized_model.gguf", quantize_command: str = ''):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else None
        self.lora_config = lora_config if lora_config else None
        self.num_labels = num_labels
        self.device = device
        self.gguf_path = gguf_path
        self.quantize_command = quantize_command
        self.hf_lora = HuggingFaceLoRAWrapper(model_name, self.tokenizer_name, self.lora_config, num_labels, device)
        self.cpp_model = None

    def fit(self, X, y, *args, **kwargs):
        self.hf_lora.fit(X, y, *args, **kwargs)
        self._merge_lora_weights()
        self._convert_to_gguf()
        self.cpp_model = LlamaCppClassifier(self.gguf_path, num_classes=self.num_labels, device=self.device)  # type: ignore

    def predict(self, X):
        if self.cpp_model is None:
            raise RuntimeError("Model not quantized/loaded yet. Call fit() first.")
        return self.cpp_model.predict(X)

    def predict_proba(self, X):
        if self.cpp_model is None:
            raise RuntimeError("Model not quantized/loaded yet. Call fit() first.")
        return self.cpp_model.predict_proba(X)

    def _merge_lora_weights(self):
        try:
            from peft import merge_and_unload  # type: ignore
        except ImportError:
            raise ImportError("peft is required for merging LoRA weights. Please install peft.")
        self.hf_lora.model = merge_and_unload(self.hf_lora.model)

    def _convert_to_gguf(self):
        import os
        if self.quantize_command:
            print(f"Running quantization command: {self.quantize_command}")
            os.system(self.quantize_command)
        else:
            print("Please manually convert the merged model to GGUF format using llama.cpp's convert script or other tools.")
            print(f"Save the GGUF file as: {self.gguf_path}")
            input("Press Enter after you have created the GGUF file...")

# --- Registry ---
MODEL_REGISTRY: Dict[str, Callable] = {
    "mlp_classifier": build_mlp_classifier,
    "mlp_regressor": build_mlp_regressor,
    "deep_mlp_classifier": lambda input_dim, num_classes=2: build_mlp_classifier(input_dim, [512, 256, 128, 64], num_classes=num_classes, dropout=0.4, batchnorm=True),
    "deep_mlp_regressor": lambda input_dim: build_mlp_regressor(input_dim, [512, 256, 128, 64], dropout=0.4, batchnorm=True),
    "transformer_classifier": build_transformer_classifier,
    "transformer_regressor": build_transformer_regressor,
    "advanced_transformer_classifier": lambda input_dim, num_classes=2: build_transformer_classifier(input_dim, num_classes=num_classes, nhead=8, num_layers=6, ff_dim=512, dropout=0.2, seq_len=16),
    "advanced_transformer_regressor": lambda input_dim: build_transformer_regressor(input_dim, nhead=8, num_layers=6, ff_dim=512, dropout=0.2, seq_len=16),
    "random_forest_classifier": SklearnRandomForestClassifierWrapper,
    "hf_lora": HuggingFaceLoRAWrapper,
    "llama_cpp_classifier": LlamaCppClassifier,
    "hf_lora_quantized": HuggingFaceLoRAWrapperQuantized,
}

"""
MODEL_REGISTRY keys:
- 'mlp_classifier', 'mlp_regressor', 'deep_mlp_classifier', 'deep_mlp_regressor', 'transformer_classifier', 'transformer_regressor', 'advanced_transformer_classifier', 'advanced_transformer_regressor', 'random_forest_classifier', 'hf_lora', 'llama_cpp_classifier', 'hf_lora_quantized'
- 'hf_lora' requires text input and HuggingFace/PEFT dependencies.
- All models are compatible with GeneralPipeline, but some (sklearn, hf_lora) may require special handling in fit/predict.
"""
