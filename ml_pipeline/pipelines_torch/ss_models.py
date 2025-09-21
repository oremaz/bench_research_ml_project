"""Semi-supervised utilities for tabular models."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine_rampup(cur: int, rampup: int) -> float:
    if rampup <= 0:
        return 1.0
    cur = max(0, min(cur, rampup))
    return 0.5 - 0.5 * math.cos(math.pi * cur / rampup)


def _maybe_apply(transform: Optional[Callable[[torch.Tensor], torch.Tensor]], x: torch.Tensor) -> torch.Tensor:
    if transform is None:
        return x
    return transform(x)


class SemiSupervisedTabular(nn.Module):
    """Generic pseudo-labelling/mean-teacher wrapper for tabular networks."""

    def __init__(
        self,
        base: nn.Module,
        num_classes: int,
        threshold: float = 0.95,
        unsup_weight: float = 1.0,
        rampup: int = 5,
        use_mean_teacher: bool = True,
        ema_decay: float = 0.999,
        weak_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        strong_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.num_classes = num_classes
        self.threshold = threshold
        self.unsup_weight = unsup_weight
        self.rampup = rampup
        self.use_mean_teacher = use_mean_teacher
        self.ema_decay = ema_decay
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        if use_mean_teacher:
            self.teacher = deepcopy(base).eval()
            for param in self.teacher.parameters():  # pragma: no cover - simple setter
                param.requires_grad_(False)
        else:
            self.teacher = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.base(x)

    def _predict(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        logits = model(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        return logits

    def step(
        self,
        batch_l: Tuple[torch.Tensor, torch.Tensor],
        batch_u: Tuple[torch.Tensor, torch.Tensor],
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x_l, y_l = batch_l
        x_u, _ = batch_u

        logits_l = self._predict(self.base, x_l)
        loss_sup = F.cross_entropy(logits_l, y_l)

        weak_u = _maybe_apply(self.weak_transform, x_u)
        strong_u = _maybe_apply(self.strong_transform, x_u)

        with torch.no_grad():
            teacher_model = self.teacher if self.teacher is not None else self.base
            logits_u = self._predict(teacher_model, weak_u)
            probs = torch.softmax(logits_u, dim=1)
            conf, pseudo = probs.max(dim=1)
            mask = (conf >= self.threshold).float()

        logits_s = self._predict(self.base, strong_u)
        loss_unsup = (F.cross_entropy(logits_s, pseudo, reduction="none") * mask).mean()
        weight = self.unsup_weight * _cosine_rampup(epoch, self.rampup)
        loss = loss_sup + weight * loss_unsup

        stats = {
            "loss_sup": loss_sup.item(),
            "loss_unsup": loss_unsup.item(),
            "mask_rate": mask.mean().item(),
            "weight": weight,
        }
        return loss, stats

    def post_optimizer_step(self) -> None:
        if self.teacher is not None:
            with torch.no_grad():
                for p_t, p_s in zip(self.teacher.parameters(), self.base.parameters()):
                    p_t.data.mul_(self.ema_decay).add_(p_s.data, alpha=1 - self.ema_decay)


__all__ = ["SemiSupervisedTabular"]

