from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def dice_loss_from_probs(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1.0 - probs) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * bce
    return loss.mean()


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    erode_x = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    erode_y = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(erode_x, erode_y)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    img = img.clamp(0.0, 1.0)
    skeleton = F.relu(img - soft_open(img))
    for _ in range(iterations):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def cldice_loss(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    iterations: int = 10,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred_skeleton = soft_skeletonize(pred_probs, iterations)
    target_skeleton = soft_skeletonize(target_probs, iterations)

    topology_precision = (pred_skeleton * target_probs).flatten(1).sum(dim=1)
    topology_precision = topology_precision / (pred_skeleton.flatten(1).sum(dim=1) + eps)

    topology_sensitivity = (target_skeleton * pred_probs).flatten(1).sum(dim=1)
    topology_sensitivity = topology_sensitivity / (target_skeleton.flatten(1).sum(dim=1) + eps)

    cl_dice = (2.0 * topology_precision * topology_sensitivity + eps) / (
        topology_precision + topology_sensitivity + eps
    )
    return 1.0 - cl_dice.mean()


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(pred)
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction="sum")
    return loss / mask.sum().clamp_min(1.0)


def binary_dice_score_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    score = (2.0 * intersection + eps) / (denominator + eps)
    return score.mean()


class TopoSparseObjective(nn.Module):
    def __init__(self, weights: dict[str, Any]) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mask_probs = torch.sigmoid(predictions["mask_logits"])
        skeleton_probs = torch.sigmoid(predictions["skeleton_logits"])
        uncertainty_probs = torch.sigmoid(predictions["uncertainty_logits"])

        mask_loss = dice_loss_from_probs(mask_probs, batch["mask"]) + binary_focal_loss_with_logits(
            predictions["mask_logits"], batch["mask"]
        )
        skeleton_loss = dice_loss_from_probs(skeleton_probs, batch["skeleton"])
        topology_loss = cldice_loss(
            mask_probs,
            batch["mask"],
            iterations=int(self.weights.get("skeleton_iterations", 10)),
        )
        junction_loss = binary_focal_loss_with_logits(predictions["junction_logits"], batch["junction"])
        endpoint_loss = binary_focal_loss_with_logits(predictions["endpoint_logits"], batch["endpoint"])
        node_loss = 0.5 * (junction_loss + endpoint_loss)
        affinity_loss = masked_smooth_l1(
            predictions["affinity"],
            batch["affinity"],
            batch["skeleton"],
        )
        uncertainty_loss = F.mse_loss(uncertainty_probs, batch["uncertainty"])

        if "graph_target" in batch and "graph_logits" in predictions:
            graph_loss = F.binary_cross_entropy_with_logits(
                predictions["graph_logits"], batch["graph_target"]
            )
        else:
            graph_loss = predictions["mask_logits"].new_tensor(0.0)

        total = (
            self.weights.get("mask", 1.0) * mask_loss
            + self.weights.get("skeleton", 1.0) * skeleton_loss
            + self.weights.get("topology", 0.5) * topology_loss
            + self.weights.get("node", 0.5) * node_loss
            + self.weights.get("affinity", 0.5) * affinity_loss
            + self.weights.get("uncertainty", 0.1) * uncertainty_loss
            + self.weights.get("graph", 0.0) * graph_loss
        )

        metrics = {
            "loss": total,
            "mask_loss": mask_loss,
            "skeleton_loss": skeleton_loss,
            "topology_loss": topology_loss,
            "node_loss": node_loss,
            "affinity_loss": affinity_loss,
            "uncertainty_loss": uncertainty_loss,
            "graph_loss": graph_loss,
            "mask_dice": binary_dice_score_from_logits(predictions["mask_logits"], batch["mask"]),
            "skeleton_dice": binary_dice_score_from_logits(
                predictions["skeleton_logits"], batch["skeleton"]
            ),
        }
        return total, metrics
