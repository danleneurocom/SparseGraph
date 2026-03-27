from __future__ import annotations

import torch
from torch import nn

from .blocks import make_norm


class PredictionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = hidden_channels or in_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
