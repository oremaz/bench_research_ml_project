import importlib
import warnings
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Optional, Type

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A very small CNN for quick experiments on vision datasets."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),  # assumes input images of size 32x32
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AdaptiveCNN(nn.Module):
    """CNN that adapts to different input sizes using adaptive pooling with modern GELU activations."""

    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling to fixed size
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return nn.functional.relu(out)


class ResidualCNN(nn.Module):
    """A small residual network inspired by ResNet architecture."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ResidualBlock(64)
        self.layer2 = ResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet50(nn.Module):
    """Transfer learning using a ResNet-50 backbone pretrained on ImageNet with adaptive input size."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, input_size: int = 224, freeze_backbone: bool = True):
        super().__init__()
        from torchvision import models

        self.input_size = input_size
        
        try:  # torchvision >= 0.13
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
        except AttributeError:  # older torchvision versions
            self.model = models.resnet50(pretrained=pretrained)
        
        # Optionally freeze backbone for faster training
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # Don't freeze the final classification layer
                    param.requires_grad = False
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # If input size doesn't match expected size (224), resize it
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)


class CLIPClassifier(nn.Module):
    """Fine-tuned CLIP vision encoder with a linear classification head using Hugging Face transformers."""

    def __init__(self, num_classes: int = 2, model_name: str = "openai/clip-vit-base-patch32", input_size: int = 224, unfreeze_layers: int = 0):
        super().__init__()
        try:
            from transformers import CLIPVisionModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for CLIPClassifier. Install with: pip install transformers"
            )

        self.input_size = input_size
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        
        # Freeze all vision model parameters first
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        # Optionally unfreeze the last N transformer layers for fine-tuning
        if unfreeze_layers > 0:
            # Get the transformer layers (encoder.layers)
            encoder_layers = self.vision_model.vision_model.encoder.layers
            total_layers = len(encoder_layers)
            
            # Unfreeze the last N layers
            for i in range(max(0, total_layers - unfreeze_layers), total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
            
            # Also unfreeze the final layer norm
            if hasattr(self.vision_model.vision_model, 'post_layernorm'):
                for param in self.vision_model.vision_model.post_layernorm.parameters():
                    param.requires_grad = True
            
            print(f"Unfroze last {unfreeze_layers} transformer layers for fine-tuning")
            
        # Add classification head
        hidden_size = self.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Convert tensor to PIL images for processor
        # Assuming x is in range [0, 1] or already normalized
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Process through CLIP vision model
        # Note: This expects input in the format expected by CLIP (RGB, normalized)
        vision_outputs = self.vision_model(pixel_values=x)
        pooled_output = vision_outputs.pooler_output  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits


class Qwen2VLQLoRA(nn.Module):
    """QLoRA fine-tuning wrapper for the Qwen 2.5 Vision-Language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct",
        num_classes: int = 2,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        import torch as _torch

        self.processor = AutoProcessor.from_pretrained(model_name)

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch.bfloat16 if _torch.cuda.is_available() else _torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
        )

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.num_classes = num_classes

    def forward(self, images, prompts):
        inputs = self.processor(images=images, text=prompts, return_tensors="pt").to(self.model.device)
        output = self.model(**inputs)
        return output.logits[:, -1, : self.num_classes]


def _import_optional_module(candidates: Iterable[str]) -> Optional[ModuleType]:
    """Try to import the first available module from *candidates*.

    Parameters
    ----------
    candidates:
        Potential fully-qualified module names.

    Returns
    -------
    ModuleType or ``None``
        The imported module, or ``None`` if none of the candidates could be
        imported. No exception is raised so that callers can provide helpful
        installation hints.
    """

    for name in candidates:
        if not name:
            continue
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _load_attr(module: ModuleType, attr_candidates: Iterable[str]):
    for attr in attr_candidates:
        if not attr:
            continue
        target = module
        try:
            for part in attr.split("."):
                target = getattr(target, part)
        except AttributeError:
            continue
        return target
    raise AttributeError(f"None of {list(attr_candidates)} found in module {module.__name__}.")


class FatFormerWrapper(nn.Module):
    """Adapter around the official `Michel-liu/FatFormer` implementation.

    The wrapper dynamically imports the GitHub repository (either installed via
    ``pip install git+https://github.com/Michel-liu/FatFormer`` or added to the
    ``PYTHONPATH``) and instantiates the :class:`FatFormer` model exposed there.

    Parameters
    ----------
    num_classes:
        Number of target classes for the downstream benchmark.
    init_kwargs:
        Optional keyword arguments forwarded to the constructor provided by the
        official repository. Different checkpoints/configurations may require
        specific arguments. When left ``None`` the wrapper only attempts to set
        ``num_classes``.
    checkpoint_path:
        Optional path to a checkpoint produced by the official training code.
        If supplied the weights are loaded with ``strict=False`` to remain
        robust against minor key mismatches.
    module_candidates / class_candidates:
        Advanced users can provide alternative import strings should the
        repository structure change. By default we cover the most common
        layouts used in the upstream project.
    """

    def __init__(
        self,
        num_classes: int = 2,
        init_kwargs: Optional[dict] = None,
        checkpoint_path: Optional[str] = None,
        module_candidates: Optional[Iterable[str]] = None,
        class_candidates: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        module = _import_optional_module(
            module_candidates
            or (
                "FatFormer.models.fatformer",
                "fatformer.models.fatformer",
                "models.fatformer",
            )
        )
        if module is None:
            raise ImportError(
                "FatFormer repository is not available. Install it with "
                "`pip install git+https://github.com/Michel-liu/FatFormer` "
                "or add the cloned repo to PYTHONPATH."
            )

        FatFormerCls = _load_attr(module, class_candidates or ("FatFormer", "FatFormerModel"))
        init_kwargs = dict(init_kwargs or {})

        if "num_classes" not in init_kwargs:
            init_kwargs["num_classes"] = num_classes

        try:
            self.model = FatFormerCls(**init_kwargs)
        except TypeError as exc:  # pragma: no cover - depends on upstream signature
            raise TypeError(
                "Unable to instantiate FatFormer with the provided arguments. "
                "Check the upstream constructor signature and supply the "
                "required values via `init_kwargs`."
            ) from exc

        if checkpoint_path:
            ckpt = torch.load(Path(checkpoint_path), map_location="cpu")
            state = ckpt.get("state_dict") if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing or unexpected:
                warnings.warn(
                    "Loaded FatFormer checkpoint with mismatched keys: "
                    f"missing={missing}, unexpected={unexpected}",
                    RuntimeWarning,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DiffusionFakeWrapper(nn.Module):
    """Adapter for the official `skJack/DiffusionFake` classifier implementation.

    The upstream repository exposes several network architectures geared towards
    diffusion-based forgery detection. The wrapper loads the default classifier
    and forwards inputs to it. Users can customise the instantiated architecture
    via ``builder`` and ``builder_kwargs`` should they rely on a specific
    configuration file from the original code base.
    """

    def __init__(
        self,
        num_classes: int = 2,
        builder: Optional[str] = None,
        builder_kwargs: Optional[dict] = None,
        module_candidates: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        module = _import_optional_module(
            module_candidates
            or (
                "DiffusionFake.models.networks",
                "diffusionfake.models.networks",
                "models.networks",
            )
        )
        if module is None:
            raise ImportError(
                "DiffusionFake repository is not available. Install it with "
                "`pip install git+https://github.com/skJack/DiffusionFake` "
                "or add the cloned repo to PYTHONPATH."
            )

        if builder is None:
            # try the default classifier exposed in upstream repo
            builder_candidates = (
                "Classifier",
                "DiffusionFakeClassifier",
                "build_classifier",
            )
        else:
            builder_candidates = (builder,)

        target = None
        for candidate in builder_candidates:
            try:
                target = _load_attr(module, (candidate,))
                break
            except AttributeError:
                continue
        if target is None:
            raise AttributeError(
                "Could not locate a classifier constructor inside the "
                "DiffusionFake repository. Provide `builder` with the fully "
                "qualified callable name from the upstream project."
            )

        builder_kwargs = dict(builder_kwargs or {})
        if "num_classes" not in builder_kwargs:
            builder_kwargs["num_classes"] = num_classes

        try:
            self.model = target(**builder_kwargs)
        except TypeError as exc:  # pragma: no cover - depends on upstream signature
            raise TypeError(
                "Unable to instantiate the DiffusionFake classifier. "
                "Supply the appropriate keyword arguments via `builder_kwargs`."
            ) from exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "adaptive_cnn": AdaptiveCNN,
    "residual_cnn": ResidualCNN,
    "resnet50": ResNet50,
    "clip_classifier": CLIPClassifier,
    "qwen2_vl_qlora": Qwen2VLQLoRA,
    "fatformer": FatFormerWrapper,
    "diffusionfake": DiffusionFakeWrapper,
}


def get_model(name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """Retrieve a vision model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
