from __future__ import annotations

import torch
from torch import nn

from .backbone import HRUNetBackbone
from .heads import PredictionHead


class TopoSparseNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        use_attention: bool = True,
        affinity_channels: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = HRUNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            use_attention=use_attention,
        )
        feature_channels = self.backbone.out_channels
        self.mask_head = PredictionHead(feature_channels, 1)
        self.skeleton_head = PredictionHead(feature_channels, 1)
        self.junction_head = PredictionHead(feature_channels, 1)
        self.endpoint_head = PredictionHead(feature_channels, 1)
        self.affinity_head = PredictionHead(feature_channels, affinity_channels)
        self.uncertainty_head = PredictionHead(feature_channels, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "mask_logits": self.mask_head(features),
            "skeleton_logits": self.skeleton_head(features),
            "junction_logits": self.junction_head(features),
            "endpoint_logits": self.endpoint_head(features),
            "affinity": self.affinity_head(features),
            "uncertainty_logits": self.uncertainty_head(features),
        }
