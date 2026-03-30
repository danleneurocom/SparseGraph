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


def soft_neighbor_degree(img: torch.Tensor) -> torch.Tensor:
    kernel = img.new_ones((1, 1, 3, 3))
    kernel[:, :, 1, 1] = 0.0
    return F.conv2d(img, kernel, padding=1)


def soft_branchpoint_map(img: torch.Tensor, slope: float = 4.0) -> torch.Tensor:
    degree = soft_neighbor_degree(img)
    branch_gate = torch.sigmoid((degree - 2.25) * slope)
    return (img * branch_gate).clamp(0.0, 1.0)


def soft_endpoint_map(img: torch.Tensor, slope: float = 4.0) -> torch.Tensor:
    degree = soft_neighbor_degree(img)
    low_gate = torch.sigmoid((degree - 0.25) * slope)
    high_gate = torch.sigmoid((1.75 - degree) * slope)
    return (img * low_gate * high_gate).clamp(0.0, 1.0)


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


def _gather_point_features(feature_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = feature_map.shape
    y = points[..., 0].long().clamp(0, height - 1)
    x = points[..., 1].long().clamp(0, width - 1)
    flat_index = (y * width + x).view(batch_size, 1, -1).expand(-1, channels, -1)
    gathered = torch.gather(feature_map.view(batch_size, channels, -1), 2, flat_index)
    return gathered.view(batch_size, channels, *points.shape[1:-1]).permute(0, *range(2, points.ndim), 1)


def _masked_path_mean(feature_map: torch.Tensor, path_points: torch.Tensor, path_mask: torch.Tensor) -> torch.Tensor:
    gathered = _gather_point_features(feature_map, path_points)
    while path_mask.ndim < gathered.ndim:
        path_mask = path_mask.unsqueeze(-1)
    weighted = gathered * path_mask
    denominator = path_mask.sum(dim=-2).clamp_min(1.0)
    return weighted.sum(dim=-2) / denominator


def compute_graph_relation_outputs(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    required_prediction_keys = {"graph_query_embeddings", "graph_key_embeddings", "path_memory_logits"}
    required_batch_keys = {
        "graph_pair_points",
        "graph_pair_path_points",
        "graph_pair_path_mask",
        "graph_pair_valid",
    }
    if not required_prediction_keys.issubset(predictions) or not required_batch_keys.issubset(batch):
        return None

    pair_points = batch["graph_pair_points"]
    pair_path_points = batch["graph_pair_path_points"]
    pair_path_mask = batch["graph_pair_path_mask"]
    valid_mask = batch["graph_pair_valid"].float()

    query_embeddings = F.normalize(predictions["graph_query_embeddings"], dim=1)
    key_embeddings = F.normalize(predictions["graph_key_embeddings"], dim=1)
    path_memory_logits = predictions["path_memory_logits"]
    skeleton_prob = torch.sigmoid(predictions["skeleton_logits"])
    relay_prob = (
        torch.sigmoid(predictions["relay_logits"]) if "relay_logits" in predictions else skeleton_prob
    )
    bridge_prob = (
        torch.sigmoid(predictions["bridge_logits"])
        if "bridge_logits" in predictions
        else torch.zeros_like(skeleton_prob)
    )
    uncertainty_prob = (
        torch.sigmoid(predictions["uncertainty_logits"])
        if "uncertainty_logits" in predictions
        else torch.zeros_like(skeleton_prob)
    )
    node_prob = torch.maximum(
        torch.sigmoid(predictions["junction_logits"]),
        torch.sigmoid(predictions["endpoint_logits"]),
    )
    counterfactual_prob = (
        torch.sigmoid(predictions["counterfactual_gate_logits"])
        if "counterfactual_gate_logits" in predictions
        else torch.zeros_like(skeleton_prob)
    )
    node_capacity_prob = (
        torch.sigmoid(predictions["node_capacity_logits"])
        if "node_capacity_logits" in predictions
        else torch.zeros_like(skeleton_prob)
    )
    causal_saliency_prob = (
        torch.sigmoid(predictions["causal_saliency_logits"])
        if "causal_saliency_logits" in predictions
        else torch.sigmoid(path_memory_logits)
    )
    dense_sparse_projection_prob = (
        torch.sigmoid(predictions["dense_sparse_projection_logits"])
        if "dense_sparse_projection_logits" in predictions
        else skeleton_prob
    )

    src_points = pair_points[:, :, 0]
    dst_points = pair_points[:, :, 1]
    q_src = _gather_point_features(query_embeddings, src_points)
    q_dst = _gather_point_features(query_embeddings, dst_points)
    k_src = _gather_point_features(key_embeddings, src_points)
    k_dst = _gather_point_features(key_embeddings, dst_points)
    node_support = 0.5 * (
        _gather_point_features(node_prob, src_points).squeeze(-1)
        + _gather_point_features(node_prob, dst_points).squeeze(-1)
    )
    node_capacity = 0.5 * (
        _gather_point_features(node_capacity_prob, src_points).squeeze(-1)
        + _gather_point_features(node_capacity_prob, dst_points).squeeze(-1)
    )
    path_memory = _masked_path_mean(path_memory_logits, pair_path_points, pair_path_mask).squeeze(-1)
    path_skeleton = _masked_path_mean(skeleton_prob, pair_path_points, pair_path_mask).squeeze(-1)
    path_relay = _masked_path_mean(relay_prob, pair_path_points, pair_path_mask).squeeze(-1)
    path_bridge = _masked_path_mean(bridge_prob, pair_path_points, pair_path_mask).squeeze(-1)
    path_uncertainty = _masked_path_mean(uncertainty_prob, pair_path_points, pair_path_mask).squeeze(-1)
    path_causal = _masked_path_mean(causal_saliency_prob, pair_path_points, pair_path_mask).squeeze(-1)
    path_projection = _masked_path_mean(
        dense_sparse_projection_prob,
        pair_path_points,
        pair_path_mask,
    ).squeeze(-1)
    counterfactual_support = _masked_path_mean(
        counterfactual_prob,
        pair_path_points,
        pair_path_mask,
    ).squeeze(-1)

    embedding_dim = max(int(q_src.shape[-1]), 1)
    compatibility = 0.5 * (
        (q_src * k_dst).sum(dim=-1) + (q_dst * k_src).sum(dim=-1)
    ) / (embedding_dim**0.5)

    spatial_extent = float(max(predictions["mask_logits"].shape[-2:]))
    distance = torch.linalg.norm(
        src_points.float() - dst_points.float(),
        dim=-1,
    ) / max(spatial_extent, 1.0)

    graph_logits = (
        1.35 * compatibility
        + 1.00 * path_memory
        + 0.85 * path_skeleton
        + 0.35 * path_relay
        + 0.20 * path_bridge
        + 0.25 * counterfactual_support
        + 0.20 * node_support
        + 0.35 * node_capacity
        + 0.70 * path_causal
        + 0.50 * path_projection
        - 0.45 * path_uncertainty
        - 0.35 * distance
    )
    importance_pred = torch.clamp(
        0.42 * path_causal
        + 0.18 * node_capacity
        + 0.15 * counterfactual_support
        + 0.10 * path_memory
        + 0.15 * path_projection,
        min=0.0,
        max=1.0,
    )
    return {
            "graph_logits": graph_logits,
            "valid_mask": valid_mask,
            "importance_pred": importance_pred,
            "node_capacity": node_capacity,
            "path_causal": path_causal,
            "path_projection": path_projection,
        }


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
        node_tolerance_loss = 0.5 * (
            dice_loss_from_probs(
                torch.sigmoid(predictions["junction_logits"]),
                F.max_pool2d(batch["junction"], kernel_size=5, stride=1, padding=2),
            )
            + dice_loss_from_probs(
                torch.sigmoid(predictions["endpoint_logits"]),
                F.max_pool2d(batch["endpoint"], kernel_size=5, stride=1, padding=2),
            )
        )
        affinity_loss = masked_smooth_l1(
            predictions["affinity"],
            batch["affinity"],
            batch["skeleton"],
        )
        uncertainty_loss = F.mse_loss(uncertainty_probs, batch["uncertainty"])
        consistency_loss = F.relu(skeleton_probs - mask_probs).mean()
        node_probs = torch.maximum(
            torch.sigmoid(predictions["junction_logits"]),
            torch.sigmoid(predictions["endpoint_logits"]),
        )
        consistency_loss = consistency_loss + (node_probs * (1.0 - skeleton_probs)).mean()
        target_degree = soft_neighbor_degree(batch["skeleton"])
        pred_degree = soft_neighbor_degree(skeleton_probs)
        branchpoint_probs = soft_branchpoint_map(skeleton_probs)
        endpoint_from_skeleton = soft_endpoint_map(skeleton_probs)
        node_focus = F.max_pool2d(
            torch.maximum(batch["junction"], batch["endpoint"]),
            kernel_size=5,
            stride=1,
            padding=2,
        )
        support_loss = predictions["mask_logits"].new_tensor(0.0)
        bio_prior_loss = predictions["mask_logits"].new_tensor(0.0)
        causal_loss = predictions["mask_logits"].new_tensor(0.0)
        relay_loss = predictions["mask_logits"].new_tensor(0.0)
        node_policy_loss = predictions["mask_logits"].new_tensor(0.0)
        graph_importance_loss = predictions["mask_logits"].new_tensor(0.0)
        branch_budget_loss = predictions["mask_logits"].new_tensor(0.0)
        branch_austerity_loss = predictions["mask_logits"].new_tensor(0.0)

        branch_budget_loss = branch_budget_loss + dice_loss_from_probs(
            branchpoint_probs,
            batch["junction"],
        )
        branch_budget_loss = branch_budget_loss + 0.5 * dice_loss_from_probs(
            endpoint_from_skeleton,
            batch["endpoint"],
        )
        branch_budget_loss = branch_budget_loss + 0.35 * masked_smooth_l1(
            pred_degree,
            target_degree,
            node_focus,
        )
        pred_branch_count = branchpoint_probs.flatten(1).sum(dim=1)
        target_branch_count = batch["junction"].flatten(1).sum(dim=1)
        pred_endpoint_count = endpoint_from_skeleton.flatten(1).sum(dim=1)
        target_endpoint_count = batch["endpoint"].flatten(1).sum(dim=1)
        branch_count_scale = target_branch_count.detach().clamp_min(1.0)
        endpoint_count_scale = target_endpoint_count.detach().clamp_min(1.0)
        branch_budget_loss = branch_budget_loss + 0.20 * F.smooth_l1_loss(
            pred_branch_count / branch_count_scale,
            target_branch_count / branch_count_scale,
        )
        branch_budget_loss = branch_budget_loss + 0.10 * F.smooth_l1_loss(
            pred_endpoint_count / endpoint_count_scale,
            target_endpoint_count / endpoint_count_scale,
        )
        if "branch_keep_logits" in predictions and "branch_prune_logits" in predictions:
            graph_causal_target = batch.get("graph_causal_path", batch["skeleton"])
            node_context = F.max_pool2d(
                torch.maximum(batch["junction"], batch["endpoint"]),
                kernel_size=7,
                stride=1,
                padding=3,
            )
            keep_target = torch.clamp(
                0.65 * graph_causal_target + 0.35 * node_context,
                0.0,
                1.0,
            )
            prune_target = torch.clamp(
                F.max_pool2d(batch["mask"], kernel_size=5, stride=1, padding=2)
                * (1.0 - F.max_pool2d(graph_causal_target, kernel_size=5, stride=1, padding=2))
                * (1.0 - F.max_pool2d(batch["junction"], kernel_size=5, stride=1, padding=2)),
                0.0,
                1.0,
            )
            branch_keep_probs = torch.sigmoid(predictions["branch_keep_logits"])
            branch_prune_probs = torch.sigmoid(predictions["branch_prune_logits"])
            branch_austerity_loss = branch_austerity_loss + dice_loss_from_probs(
                branch_keep_probs,
                keep_target,
            )
            branch_austerity_loss = branch_austerity_loss + 0.5 * binary_focal_loss_with_logits(
                predictions["branch_keep_logits"],
                keep_target,
            )
            branch_austerity_loss = branch_austerity_loss + dice_loss_from_probs(
                branch_prune_probs,
                prune_target,
            )
            branch_austerity_loss = branch_austerity_loss + 0.5 * binary_focal_loss_with_logits(
                predictions["branch_prune_logits"],
                prune_target,
            )
            allowed_branch_zone = F.max_pool2d(batch["junction"], kernel_size=5, stride=1, padding=2)
            branch_austerity_loss = branch_austerity_loss + 0.40 * (
                soft_branchpoint_map(skeleton_probs) * (1.0 - allowed_branch_zone) * (1.0 - keep_target)
            ).mean()
            branch_austerity_loss = branch_austerity_loss + 0.20 * (
                skeleton_probs
                * branch_prune_probs
                * (1.0 - F.max_pool2d(batch["skeleton"], kernel_size=3, stride=1, padding=1))
            ).mean()

        auxiliary_loss = predictions["mask_logits"].new_tensor(0.0)
        if "coarse_mask_logits" in predictions:
            coarse_mask_probs = torch.sigmoid(predictions["coarse_mask_logits"])
            auxiliary_loss = auxiliary_loss + dice_loss_from_probs(coarse_mask_probs, batch["mask"])
            auxiliary_loss = auxiliary_loss + binary_focal_loss_with_logits(
                predictions["coarse_mask_logits"], batch["mask"]
            )
        if "coarse_skeleton_logits" in predictions:
            coarse_skeleton_probs = torch.sigmoid(predictions["coarse_skeleton_logits"])
            auxiliary_loss = auxiliary_loss + dice_loss_from_probs(coarse_skeleton_probs, batch["skeleton"])
        if "coarse_junction_logits" in predictions and "coarse_endpoint_logits" in predictions:
            coarse_junction_loss = binary_focal_loss_with_logits(
                predictions["coarse_junction_logits"], batch["junction"]
            )
            coarse_endpoint_loss = binary_focal_loss_with_logits(
                predictions["coarse_endpoint_logits"], batch["endpoint"]
            )
            auxiliary_loss = auxiliary_loss + 0.5 * (coarse_junction_loss + coarse_endpoint_loss)
        if "rollout_mask_logits" in predictions:
            rollout_mask_probs = torch.sigmoid(predictions["rollout_mask_logits"])
            auxiliary_loss = auxiliary_loss + dice_loss_from_probs(rollout_mask_probs, batch["mask"])
            auxiliary_loss = auxiliary_loss + binary_focal_loss_with_logits(
                predictions["rollout_mask_logits"], batch["mask"]
            )
        if "rollout_skeleton_logits" in predictions:
            rollout_skeleton_probs = torch.sigmoid(predictions["rollout_skeleton_logits"])
            auxiliary_loss = auxiliary_loss + dice_loss_from_probs(
                rollout_skeleton_probs, batch["skeleton"]
            )
        if "rollout_junction_logits" in predictions and "rollout_endpoint_logits" in predictions:
            rollout_junction_loss = binary_focal_loss_with_logits(
                predictions["rollout_junction_logits"], batch["junction"]
            )
            rollout_endpoint_loss = binary_focal_loss_with_logits(
                predictions["rollout_endpoint_logits"], batch["endpoint"]
            )
            auxiliary_loss = auxiliary_loss + 0.5 * (rollout_junction_loss + rollout_endpoint_loss)
        if "relay_logits" in predictions:
            relay_probs = torch.sigmoid(predictions["relay_logits"])
            relay_loss = relay_loss + dice_loss_from_probs(relay_probs, batch["skeleton"])
            relay_loss = relay_loss + 0.5 * binary_focal_loss_with_logits(
                predictions["relay_logits"], batch["skeleton"]
            )
            consistency_loss = consistency_loss + (skeleton_probs * (1.0 - relay_probs)).mean()
        if "bridge_logits" in predictions:
            bridge_target = (
                F.relu(F.max_pool2d(batch["skeleton"], kernel_size=7, stride=1, padding=3) - batch["skeleton"])
                * batch["mask"]
            )
            bridge_probs = torch.sigmoid(predictions["bridge_logits"])
            relay_loss = relay_loss + dice_loss_from_probs(bridge_probs, bridge_target)
            relay_loss = relay_loss + 0.5 * binary_focal_loss_with_logits(
                predictions["bridge_logits"], bridge_target
            )
        if "counterfactual_bridge_logits" in predictions:
            counterfactual_bridge_probs = torch.sigmoid(predictions["counterfactual_bridge_logits"])
            relay_loss = relay_loss + dice_loss_from_probs(counterfactual_bridge_probs, batch["skeleton"])
            relay_loss = relay_loss + 0.5 * binary_focal_loss_with_logits(
                predictions["counterfactual_bridge_logits"], batch["skeleton"]
            )
        if "endpoint_relay_logits" in predictions:
            endpoint_relay_target = F.max_pool2d(batch["endpoint"], kernel_size=5, stride=1, padding=2)
            endpoint_relay_probs = torch.sigmoid(predictions["endpoint_relay_logits"])
            relay_loss = relay_loss + 0.5 * dice_loss_from_probs(
                endpoint_relay_probs,
                endpoint_relay_target,
            )
        if "junction_relay_logits" in predictions:
            junction_relay_target = F.max_pool2d(batch["junction"], kernel_size=5, stride=1, padding=2)
            junction_relay_probs = torch.sigmoid(predictions["junction_relay_logits"])
            relay_loss = relay_loss + 0.5 * dice_loss_from_probs(
                junction_relay_probs,
                junction_relay_target,
            )
        if "node_capacity_logits" in predictions and "graph_node_capacity" in batch:
            node_mask = torch.maximum(batch["junction"], batch["endpoint"])
            node_policy_loss = node_policy_loss + masked_smooth_l1(
                torch.sigmoid(predictions["node_capacity_logits"]),
                batch["graph_node_capacity"],
                node_mask,
            )
        if "causal_saliency_logits" in predictions and "graph_causal_path" in batch:
            causal_saliency_probs = torch.sigmoid(predictions["causal_saliency_logits"])
            node_policy_loss = node_policy_loss + dice_loss_from_probs(
                causal_saliency_probs,
                batch["graph_causal_path"],
            )
            node_policy_loss = node_policy_loss + 0.5 * F.smooth_l1_loss(
                causal_saliency_probs,
                batch["graph_causal_path"],
            )
        if "dense_sparse_projection_logits" in predictions:
            dense_sparse_projection_probs = torch.sigmoid(predictions["dense_sparse_projection_logits"])
            projection_target = torch.clamp(
                0.75 * batch["skeleton"] + 0.25 * batch.get("graph_causal_path", batch["skeleton"]),
                0.0,
                1.0,
            )
            dense_mask_projection = soft_skeletonize(
                mask_probs,
                iterations=max(int(self.weights.get("skeleton_iterations", 10)) // 2, 4),
            )
            branch_austerity_loss = branch_austerity_loss + 0.75 * dice_loss_from_probs(
                dense_sparse_projection_probs,
                projection_target,
            )
            branch_austerity_loss = branch_austerity_loss + 0.25 * binary_focal_loss_with_logits(
                predictions["dense_sparse_projection_logits"],
                projection_target,
            )
            consistency_loss = consistency_loss + 0.35 * dice_loss_from_probs(
                dense_sparse_projection_probs,
                dense_mask_projection,
            )
            consistency_loss = consistency_loss + 0.20 * dice_loss_from_probs(
                skeleton_probs,
                dense_sparse_projection_probs.detach(),
            )
            consistency_loss = consistency_loss + 0.15 * (
                dense_sparse_projection_probs
                * (1.0 - F.max_pool2d(mask_probs, kernel_size=3, stride=1, padding=1))
            ).mean()

        if (
            "morphology_prior_logits" in predictions
            and "activity_prior_logits" in predictions
            and "agreement_logits" in predictions
            and "conflict_logits" in predictions
        ):
            morphology_prior_probs = torch.sigmoid(predictions["morphology_prior_logits"])
            activity_prior_probs = torch.sigmoid(predictions["activity_prior_logits"])
            agreement_probs = torch.sigmoid(predictions["agreement_logits"])
            conflict_probs = torch.sigmoid(predictions["conflict_logits"])

            morphology_loss = dice_loss_from_probs(morphology_prior_probs, batch["mask"]) + (
                binary_focal_loss_with_logits(predictions["morphology_prior_logits"], batch["mask"])
            )
            activity_loss = dice_loss_from_probs(activity_prior_probs, batch["skeleton"])
            agreement_loss = dice_loss_from_probs(agreement_probs, batch["skeleton"])
            conflict_loss = F.mse_loss(conflict_probs, batch["uncertainty"])
            bio_prior_loss = morphology_loss + activity_loss + agreement_loss + conflict_loss

            mask_support = torch.maximum(morphology_prior_probs, activity_prior_probs)
            skeleton_support = torch.sqrt(
                morphology_prior_probs.clamp_min(1e-6) * activity_prior_probs.clamp_min(1e-6)
            ) * agreement_probs
            support_loss = (mask_probs * (1.0 - mask_support)).mean()
            support_loss = support_loss + 0.5 * (skeleton_probs * (1.0 - skeleton_support)).mean()

            morphology_only = morphology_prior_probs * (1.0 - activity_prior_probs)
            activity_only = activity_prior_probs * (1.0 - morphology_prior_probs)
            causal_loss = (skeleton_probs * morphology_only).mean()
            causal_loss = causal_loss + (skeleton_probs * activity_only).mean()
            causal_loss = causal_loss + 0.5 * (mask_probs * conflict_probs).mean()

        graph_outputs = compute_graph_relation_outputs(predictions, batch)
        if graph_outputs is not None and "graph_target" in batch:
            graph_logits = graph_outputs["graph_logits"]
            graph_valid = graph_outputs["valid_mask"]
            graph_target = batch["graph_target"].float()
            graph_weight = graph_valid.float()
            graph_loss_map = F.binary_cross_entropy_with_logits(
                graph_logits,
                graph_target,
                reduction="none",
            )
            graph_loss = (graph_loss_map * graph_weight).sum() / graph_weight.sum().clamp_min(1.0)
            predictions["graph_logits"] = graph_logits.detach()
            graph_probs = torch.sigmoid(graph_logits)
            graph_pred = (graph_probs >= 0.5).float()
            graph_correct = ((graph_pred == graph_target).float() * graph_weight).sum()
            graph_accuracy = graph_correct / graph_weight.sum().clamp_min(1.0)
            positive_weight = (graph_target * graph_weight).sum().clamp_min(1.0)
            graph_recall = ((graph_pred * graph_target) * graph_weight).sum() / positive_weight
            if "graph_pair_importance" in batch:
                importance_target = batch["graph_pair_importance"].float()
                importance_pred = graph_outputs["importance_pred"]
                graph_importance_loss = F.smooth_l1_loss(
                    importance_pred * graph_weight,
                    importance_target * graph_weight,
                    reduction="sum",
                ) / graph_weight.sum().clamp_min(1.0)
                pair_importance_mae = (
                    (importance_pred - importance_target).abs() * graph_weight
                ).sum() / graph_weight.sum().clamp_min(1.0)
            else:
                pair_importance_mae = predictions["mask_logits"].new_tensor(0.0)
            predictions["graph_logits"] = graph_logits.detach()
        else:
            graph_loss = predictions["mask_logits"].new_tensor(0.0)
            graph_accuracy = predictions["mask_logits"].new_tensor(0.0)
            graph_recall = predictions["mask_logits"].new_tensor(0.0)
            pair_importance_mae = predictions["mask_logits"].new_tensor(0.0)

        total = (
            self.weights.get("mask", 1.0) * mask_loss
            + self.weights.get("skeleton", 1.0) * skeleton_loss
            + self.weights.get("topology", 0.5) * topology_loss
            + self.weights.get("node", 0.5) * node_loss
            + self.weights.get("node_tolerance", 0.0) * node_tolerance_loss
            + self.weights.get("affinity", 0.5) * affinity_loss
            + self.weights.get("uncertainty", 0.1) * uncertainty_loss
            + self.weights.get("auxiliary", 0.0) * auxiliary_loss
            + self.weights.get("consistency", 0.0) * consistency_loss
            + self.weights.get("support", 0.0) * support_loss
            + self.weights.get("bio_prior", 0.0) * bio_prior_loss
            + self.weights.get("causal", 0.0) * causal_loss
            + self.weights.get("relay", 0.0) * relay_loss
            + self.weights.get("node_policy", 0.0) * node_policy_loss
            + self.weights.get("branch_budget", 0.0) * branch_budget_loss
            + self.weights.get("branch_austerity", 0.0) * branch_austerity_loss
            + self.weights.get("graph_importance", 0.0) * graph_importance_loss
            + self.weights.get("graph", 0.0) * graph_loss
        )

        metrics = {
            "loss": total,
            "mask_loss": mask_loss,
            "skeleton_loss": skeleton_loss,
            "topology_loss": topology_loss,
            "node_loss": node_loss,
            "node_tolerance_loss": node_tolerance_loss,
            "affinity_loss": affinity_loss,
            "uncertainty_loss": uncertainty_loss,
            "auxiliary_loss": auxiliary_loss,
            "consistency_loss": consistency_loss,
            "support_loss": support_loss,
            "bio_prior_loss": bio_prior_loss,
            "causal_loss": causal_loss,
            "relay_loss": relay_loss,
            "node_policy_loss": node_policy_loss,
            "branch_budget_loss": branch_budget_loss,
            "branch_austerity_loss": branch_austerity_loss,
            "graph_loss": graph_loss,
            "graph_importance_loss": graph_importance_loss,
            "graph_accuracy": graph_accuracy,
            "graph_recall": graph_recall,
            "pair_importance_mae": pair_importance_mae,
            "mask_dice": binary_dice_score_from_logits(predictions["mask_logits"], batch["mask"]),
            "skeleton_dice": binary_dice_score_from_logits(
                predictions["skeleton_logits"], batch["skeleton"]
            ),
            "cldice": 1.0 - topology_loss.detach(),
        }
        metrics["selection_score"] = (
            0.4 * metrics["mask_dice"] + 0.35 * metrics["skeleton_dice"] + 0.25 * metrics["cldice"]
        )
        return total, metrics
