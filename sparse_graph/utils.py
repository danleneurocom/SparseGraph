from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class MetricTracker:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, metrics: dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self.totals[key] += float(value) * n
            self.counts[key] += n

    def averages(self) -> dict[str, float]:
        return {
            key: self.totals[key] / max(self.counts[key], 1)
            for key in self.totals
        }


def to_float_dict(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            result[key] = float(value.detach().cpu().item())
        else:
            result[key] = float(value)
    return result
