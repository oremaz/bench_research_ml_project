"""Semi-supervised vision algorithms.

This module complements :mod:`vision_models` by providing training utilities for
semi-supervised learning (SSL). Classical approaches are implemented from
scratch whereas recent 2024/2025 A* methods (e.g. CDMAD) integrate the official
GitHub repositories when available. The abstractions are deliberately light so
they slot into the existing benchmarking pipelines without code duplication.
"""

from __future__ import annotations

import importlib
import math
from copy import deepcopy
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _optional_import(candidates: Iterable[str]):
    for name in candidates:
        if not name:
            continue
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def _cosine_rampup(cur: int, rampup: int) -> float:
    if rampup <= 0:
        return 1.0
    cur = max(0, min(cur, rampup))
    return 0.5 - 0.5 * math.cos(math.pi * cur / rampup)


def _update_ema(ema_model: nn.Module, student: nn.Module, decay: float = 0.999):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), student.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)


class SSLImageClassifier(nn.Module):
    """Base class wrapping a supervised backbone with SSL-specific losses."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple delegation
        return self.backbone(x)

    # --- hooks for pipelines -------------------------------------------------
    def ssl_loss(
        self,
        batch_l: Tuple[torch.Tensor, torch.Tensor],
        batch_u: Tuple[torch.Tensor, torch.Tensor],
        epoch: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def post_step(self) -> None:
        """Optional hook executed after the optimiser step."""

    # The weak/strong augmentation hooks are intentionally trivial so that the
    # training script can inject domain-specific transforms at runtime.
    def weak_augment(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - overridden in user code
        return x

    def strong_augment(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


class PseudoLabel(SSLImageClassifier):
    """Classical pseudo-labeling (Lee, 2013)."""

    def __init__(self, backbone: nn.Module, threshold: float = 0.95, unsup_weight: float = 1.0, rampup: int = 5):
        super().__init__(backbone)
        self.threshold = threshold
        self.unsup_weight = unsup_weight
        self.rampup = rampup

    def ssl_loss(self, batch_l, batch_u, epoch, **_):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        logits_l = self.backbone(x_l)
        sup = F.cross_entropy(logits_l, y_l)

        with torch.no_grad():
            logits_w = self.backbone(self.weak_augment(x_u))
            probs = torch.softmax(logits_w, dim=1)
            conf, labels = probs.max(dim=1)
            mask = (conf >= self.threshold).float()

        logits_s = self.backbone(self.strong_augment(x_u))
        unsup = (F.cross_entropy(logits_s, labels, reduction="none") * mask).mean()
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        return sup, weight * unsup, {"mask_rate": mask.mean().item(), "weight": weight}


class PiModel(SSLImageClassifier):
    """Consistency regularisation using two augmented views (Laine & Aila)."""

    def __init__(self, backbone: nn.Module, unsup_weight: float = 1.0, rampup: int = 5, temperature: float = 1.0):
        super().__init__(backbone)
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.temperature = temperature

    def ssl_loss(self, batch_l, batch_u, epoch, **_):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        sup = F.cross_entropy(self.backbone(x_l), y_l)

        p1 = torch.softmax(self.backbone(self.weak_augment(x_u)) / self.temperature, dim=1)
        p2 = torch.softmax(self.backbone(self.strong_augment(x_u)) / self.temperature, dim=1)
        unsup = F.mse_loss(p1, p2)
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        return sup, weight * unsup, {"weight": weight}


class MeanTeacher(SSLImageClassifier):
    """Mean Teacher (Tarvainen & Valpola, 2017)."""

    def __init__(self, backbone: nn.Module, ema_decay: float = 0.999, unsup_weight: float = 1.0, rampup: int = 5, temperature: float = 1.0):
        super().__init__(backbone)
        self.teacher = deepcopy(backbone).eval()
        for param in self.teacher.parameters():  # pragma: no cover - simple attribute access
            param.requires_grad_(False)
        self.ema_decay = ema_decay
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.temperature = temperature

    def ssl_loss(self, batch_l, batch_u, epoch, **_):
        x_l, y_l = batch_l
        x_u, _ = batch_u
        sup = F.cross_entropy(self.backbone(x_l), y_l)

        with torch.no_grad():
            teacher_logits = self.teacher(self.weak_augment(x_u))
            teacher_prob = torch.softmax(teacher_logits / self.temperature, dim=1)

        student_prob = torch.softmax(self.backbone(self.strong_augment(x_u)) / self.temperature, dim=1)
        unsup = F.mse_loss(student_prob, teacher_prob)
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        return sup, weight * unsup, {"weight": weight, "_update_ema": True}

    def post_step(self) -> None:
        _update_ema(self.teacher, self.backbone, self.ema_decay)


class STUCSSIC(SSLImageClassifier):
    """Self-adaptive thresholding with unreliable-sample contrastive learning.

    The authors have not released official code (Papers with Code lists none),
    so this class provides an in-house implementation of the loss described in
    the 2024 manuscript.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        base_threshold: float = 0.95,
        min_threshold: float = 0.6,
        projection_dim: int = 128,
        temperature: float = 0.5,
        unsup_weight: float = 1.0,
        rampup: int = 10,
    ) -> None:
        super().__init__(backbone)
        self.num_classes = num_classes
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.temperature = temperature
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.projection = nn.Linear(num_classes, projection_dim)
        self.register_buffer("class_threshold", torch.full((num_classes,), base_threshold))

    def _update_threshold(self, epoch: int) -> None:
        relax = _cosine_rampup(epoch, self.rampup)
        # relax=0 at start -> threshold=base_threshold; relax=1 -> min_threshold
        new_thresh = self.min_threshold + (self.base_threshold - self.min_threshold) * (1 - relax)
        self.class_threshold.fill_(new_thresh)

    def ssl_loss(self, batch_l, batch_u, epoch, **_):
        self._update_threshold(epoch)
        x_l, y_l = batch_l
        x_u, _ = batch_u

        sup = F.cross_entropy(self.backbone(x_l), y_l)

        with torch.no_grad():
            logits_w = self.backbone(self.weak_augment(x_u))
            probs_w = torch.softmax(logits_w, dim=1)
            conf, labels = probs_w.max(dim=1)
            thr = self.class_threshold[labels]
            reliable = (conf >= thr).float()

        logits_s = self.backbone(self.strong_augment(x_u))
        ce_loss = (F.cross_entropy(logits_s, labels, reduction="none") * reliable).mean()

        proj_w = F.normalize(self.projection(probs_w), dim=1)
        proj_s = F.normalize(self.projection(torch.softmax(logits_s, dim=1)), dim=1)
        unreliable_idx = (reliable < 0.5).nonzero(as_tuple=True)[0]
        contrast = torch.tensor(0.0, device=x_u.device)
        if unreliable_idx.numel() > 0:
            zw = proj_w[unreliable_idx]
            zs = proj_s[unreliable_idx]
            logits_con = torch.matmul(zw, zs.t()) / self.temperature
            targets = torch.arange(logits_con.size(0), device=x_u.device)
            contrast = F.cross_entropy(logits_con, targets)

        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        unsup = ce_loss + contrast
        return sup, weight * unsup, {
            "reliable_rate": reliable.mean().item(),
            "contrastive_active": float(unreliable_idx.numel() > 0),
            "weight": weight,
        }


class CDMADSemiSupervised(PseudoLabel):
    """CDMAD debiased pseudo-labelling (CVPR 2024).

    The official repository (``LeeHyuck/CDMAD``) exposes utility functions for
    computing the bias logits. When the package is available we use it directly;
    otherwise we fall back to the formula from the paper.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        threshold: float = 0.95,
        unsup_weight: float = 1.0,
        rampup: int = 8,
        bias_input_shape: Tuple[int, int, int] = (3, 32, 32),
        repo_bias_function: Optional[str] = None,
    ) -> None:
        super().__init__(backbone, threshold=threshold, unsup_weight=unsup_weight, rampup=rampup)
        self.num_classes = num_classes
        self.bias_input_shape = bias_input_shape
        self.register_buffer("bias_logits", torch.zeros(num_classes))

        module = _optional_import(("CDMAD", "cdmad", "CDMAD.core", "CDMAD.model"))
        bias_fn = None
        if module is not None:
            candidates = (repo_bias_function,) if repo_bias_function else (
                "utils.compute_bias",
                "core.compute_bias",
                "bias.compute_bias",
            )
            for candidate in candidates:
                if not candidate:
                    continue
                try:
                    bias_fn = module
                    for part in candidate.split("."):
                        bias_fn = getattr(bias_fn, part)
                    break
                except AttributeError:
                    continue
        self._repo_bias_fn = bias_fn

    @torch.no_grad()
    def update_bias(self, device: torch.device) -> None:
        dummy = torch.zeros((1,) + self.bias_input_shape, device=device)
        logits = self.backbone(dummy)
        if self._repo_bias_fn is not None:
            try:  # pragma: no cover - depends on external repo
                bias = self._repo_bias_fn(logits)
                if isinstance(bias, torch.Tensor):
                    self.bias_logits.copy_(bias.squeeze())
                    return
            except Exception:
                pass
        self.bias_logits.copy_(logits.squeeze())

    def ssl_loss(self, batch_l, batch_u, epoch, **kwargs):
        device = batch_l[0].device
        if not torch.any(self.bias_logits):
            self.update_bias(device)

        x_l, y_l = batch_l
        x_u, _ = batch_u

        sup = F.cross_entropy(self.backbone(x_l), y_l)

        with torch.no_grad():
            logits_w = self.backbone(self.weak_augment(x_u)) - self.bias_logits.unsqueeze(0)
            probs = torch.softmax(logits_w, dim=1)
            conf, labels = probs.max(dim=1)
            mask = (conf >= self.threshold).float()

        logits_s = self.backbone(self.strong_augment(x_u))
        unsup = (F.cross_entropy(logits_s, labels, reduction="none") * mask).mean()
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        return sup, weight * unsup, {"mask_rate": mask.mean().item(), "weight": weight}


__all__ = [
    "SSLImageClassifier",
    "PseudoLabel",
    "PiModel",
    "MeanTeacher",
    "STUCSSIC",
    "CDMADSemiSupervised",
]

