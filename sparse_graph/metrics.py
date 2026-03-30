from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree
from torch.nn import functional as F

from .losses import soft_skeletonize


def _safe_divide(numerator: float, denominator: float, eps: float = 1e-6) -> float:
    return float(numerator) / float(max(denominator, eps))


def _connected_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    binary = binary.astype(bool)
    structure = np.ones((3, 3), dtype=np.int8)
    labels, component_index = ndimage.label(binary, structure=structure)
    return labels.astype(np.int32, copy=False), int(component_index)


def _neighbor_degree(binary: np.ndarray) -> np.ndarray:
    padded = np.pad(binary.astype(np.float32), 1)
    degree = np.zeros_like(binary, dtype=np.float32)
    height, width = binary.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            degree += padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
    return degree * binary.astype(np.float32)


def _peak_points(logits: torch.Tensor, threshold: float = 0.5) -> list[tuple[int, int]]:
    if logits.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap, got shape {tuple(logits.shape)}")
    pooled = F.max_pool2d(logits[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    maxima = (logits >= threshold) & (logits == pooled)
    ys, xs = torch.nonzero(maxima, as_tuple=True)
    return list(zip(ys.tolist(), xs.tolist()))


def _brier_score_from_probs(pred_probs: torch.Tensor, target_probs: torch.Tensor) -> float:
    return float(torch.mean((pred_probs - target_probs) ** 2).detach().cpu().item())


def _expected_calibration_error_from_probs(
    pred_probs: torch.Tensor,
    target_probs: torch.Tensor,
    num_bins: int = 10,
) -> float:
    probs = pred_probs.detach().flatten().cpu()
    target = target_probs.detach().flatten().cpu()
    if probs.numel() == 0:
        return 0.0

    ece = 0.0
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1)
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not torch.any(mask):
            continue
        bin_confidence = probs[mask].mean()
        bin_accuracy = target[mask].mean()
        ece += float(mask.float().mean().item()) * float(abs(bin_confidence - bin_accuracy).item())
    return ece


def _match_points(
    pred_points: list[tuple[int, int]],
    target_points: list[tuple[int, int]],
    tolerance: float,
) -> tuple[int, int, int]:
    if not pred_points and not target_points:
        return 0, 0, 0

    pred_array = np.asarray(pred_points, dtype=np.float32)
    target_array = np.asarray(target_points, dtype=np.float32)
    tree = cKDTree(target_array)

    candidate_matches: list[tuple[float, int, int]] = []
    for pred_index, neighbors in enumerate(tree.query_ball_point(pred_array, r=tolerance)):
        if not neighbors:
            continue
        deltas = target_array[neighbors] - pred_array[pred_index]
        distances = np.linalg.norm(deltas, axis=1)
        for neighbor_index, distance in zip(neighbors, distances.tolist()):
            candidate_matches.append((float(distance), pred_index, int(neighbor_index)))

    candidate_matches.sort(key=lambda item: item[0])
    matched_preds: set[int] = set()
    matched_targets: set[int] = set()
    true_positive = 0
    for _, pred_index, target_index in candidate_matches:
        if pred_index in matched_preds or target_index in matched_targets:
            continue
        matched_preds.add(pred_index)
        matched_targets.add(target_index)
        true_positive += 1

    false_positive = max(len(pred_points) - true_positive, 0)
    false_negative = max(len(target_points) - true_positive, 0)
    return true_positive, false_positive, false_negative


def binary_precision_recall_f1(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
) -> tuple[float, float, float]:
    pred_binary = pred_binary.astype(bool)
    target_binary = target_binary.astype(bool)
    true_positive = float(np.logical_and(pred_binary, target_binary).sum())
    false_positive = float(np.logical_and(pred_binary, np.logical_not(target_binary)).sum())
    false_negative = float(np.logical_and(np.logical_not(pred_binary), target_binary).sum())

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def cldice_score_from_probs(
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
    return cl_dice.mean()


@dataclass
class GraphProxyMetrics:
    junction_f1: float
    endpoint_f1: float
    component_error: float
    branch_count_error: float
    length_error: float


def graph_proxy_metrics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    thresholds: dict[str, float] | None = None,
    node_tolerance: float = 3.0,
) -> GraphProxyMetrics:
    thresholds = thresholds or {}
    mask_threshold = float(thresholds.get("mask_threshold", 0.5))
    skeleton_threshold = float(thresholds.get("skeleton_threshold", 0.5))
    junction_threshold = float(thresholds.get("junction_threshold", 0.5))
    endpoint_threshold = float(thresholds.get("endpoint_threshold", 0.5))

    mask_prob = torch.sigmoid(predictions["mask_logits"])
    skeleton_prob = torch.sigmoid(predictions["skeleton_logits"])
    junction_prob = torch.sigmoid(predictions["junction_logits"])
    endpoint_prob = torch.sigmoid(predictions["endpoint_logits"])

    junction_f1_values: list[float] = []
    endpoint_f1_values: list[float] = []
    component_errors: list[float] = []
    branch_errors: list[float] = []
    length_errors: list[float] = []

    batch_size = int(mask_prob.shape[0])
    for index in range(batch_size):
        pred_skeleton = (
            (skeleton_prob[index, 0] >= skeleton_threshold).detach().cpu().numpy().astype(np.uint8)
        )
        target_skeleton = (
            (batch["skeleton"][index, 0] >= skeleton_threshold).detach().cpu().numpy().astype(np.uint8)
        )

        pred_mask = (mask_prob[index, 0] >= mask_threshold).detach().cpu().numpy().astype(np.uint8)
        target_mask = (batch["mask"][index, 0] >= mask_threshold).detach().cpu().numpy().astype(np.uint8)

        _, pred_components = _connected_components(pred_skeleton)
        _, target_components = _connected_components(target_skeleton)
        component_errors.append(abs(float(pred_components) - float(target_components)))

        pred_degree = _neighbor_degree(pred_skeleton)
        target_degree = _neighbor_degree(target_skeleton)
        pred_branches = int(((pred_skeleton > 0) & (pred_degree >= 3)).sum())
        target_branches = int(((target_skeleton > 0) & (target_degree >= 3)).sum())
        branch_errors.append(abs(float(pred_branches) - float(target_branches)))

        pred_length = float(pred_skeleton.sum())
        target_length = float(target_skeleton.sum())
        length_errors.append(abs(pred_length - target_length) / max(target_length, 1.0))

        pred_junction_points = _peak_points(junction_prob[index, 0], threshold=junction_threshold)
        target_junction_points = _peak_points(batch["junction"][index, 0], threshold=junction_threshold)
        junction_tp, junction_fp, junction_fn = _match_points(
            pred_junction_points,
            target_junction_points,
            tolerance=node_tolerance,
        )
        junction_precision = _safe_divide(junction_tp, junction_tp + junction_fp)
        junction_recall = _safe_divide(junction_tp, junction_tp + junction_fn)
        junction_f1_values.append(
            _safe_divide(2.0 * junction_precision * junction_recall, junction_precision + junction_recall)
        )

        pred_endpoint_points = _peak_points(endpoint_prob[index, 0], threshold=endpoint_threshold)
        target_endpoint_points = _peak_points(batch["endpoint"][index, 0], threshold=endpoint_threshold)
        endpoint_tp, endpoint_fp, endpoint_fn = _match_points(
            pred_endpoint_points,
            target_endpoint_points,
            tolerance=node_tolerance,
        )
        endpoint_precision = _safe_divide(endpoint_tp, endpoint_tp + endpoint_fp)
        endpoint_recall = _safe_divide(endpoint_tp, endpoint_tp + endpoint_fn)
        endpoint_f1_values.append(
            _safe_divide(2.0 * endpoint_precision * endpoint_recall, endpoint_precision + endpoint_recall)
        )

        if target_mask.sum() == 0 and pred_mask.sum() == 0:
            length_errors[-1] = 0.0

    return GraphProxyMetrics(
        junction_f1=float(np.mean(junction_f1_values)) if junction_f1_values else 0.0,
        endpoint_f1=float(np.mean(endpoint_f1_values)) if endpoint_f1_values else 0.0,
        component_error=float(np.mean(component_errors)) if component_errors else 0.0,
        branch_count_error=float(np.mean(branch_errors)) if branch_errors else 0.0,
        length_error=float(np.mean(length_errors)) if length_errors else 0.0,
    )


def publication_metrics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    thresholds: dict[str, float] | None = None,
    skeleton_iterations: int = 10,
) -> dict[str, float]:
    thresholds = thresholds or {}
    mask_threshold = float(thresholds.get("mask_threshold", 0.5))
    skeleton_threshold = float(thresholds.get("skeleton_threshold", 0.5))
    junction_threshold = float(thresholds.get("junction_threshold", 0.5))
    endpoint_threshold = float(thresholds.get("endpoint_threshold", 0.5))

    mask_prob = torch.sigmoid(predictions["mask_logits"])
    skeleton_prob = torch.sigmoid(predictions["skeleton_logits"])
    junction_prob = torch.sigmoid(predictions["junction_logits"])
    endpoint_prob = torch.sigmoid(predictions["endpoint_logits"])

    mask_binary = (mask_prob >= mask_threshold).detach().cpu().numpy()
    skeleton_binary = (skeleton_prob >= skeleton_threshold).detach().cpu().numpy()
    target_mask = (batch["mask"] >= mask_threshold).detach().cpu().numpy()
    target_skeleton = (batch["skeleton"] >= skeleton_threshold).detach().cpu().numpy()

    mask_precision_values: list[float] = []
    mask_recall_values: list[float] = []
    mask_f1_values: list[float] = []
    skeleton_precision_values: list[float] = []
    skeleton_recall_values: list[float] = []
    skeleton_f1_values: list[float] = []

    batch_size = int(mask_prob.shape[0])
    for index in range(batch_size):
        mask_precision, mask_recall, mask_f1 = binary_precision_recall_f1(
            mask_binary[index, 0],
            target_mask[index, 0],
        )
        mask_precision_values.append(mask_precision)
        mask_recall_values.append(mask_recall)
        mask_f1_values.append(mask_f1)

        skeleton_precision, skeleton_recall, skeleton_f1 = binary_precision_recall_f1(
            skeleton_binary[index, 0],
            target_skeleton[index, 0],
        )
        skeleton_precision_values.append(skeleton_precision)
        skeleton_recall_values.append(skeleton_recall)
        skeleton_f1_values.append(skeleton_f1)

    graph_metrics = graph_proxy_metrics(predictions, batch, thresholds=thresholds)
    cldice = float(
        cldice_score_from_probs(mask_prob, batch["mask"], iterations=skeleton_iterations).detach().cpu().item()
    )

    return {
        "mask_precision": float(np.mean(mask_precision_values)) if mask_precision_values else 0.0,
        "mask_recall": float(np.mean(mask_recall_values)) if mask_recall_values else 0.0,
        "mask_f1": float(np.mean(mask_f1_values)) if mask_f1_values else 0.0,
        "skeleton_precision": float(np.mean(skeleton_precision_values)) if skeleton_precision_values else 0.0,
        "skeleton_recall": float(np.mean(skeleton_recall_values)) if skeleton_recall_values else 0.0,
        "skeleton_f1": float(np.mean(skeleton_f1_values)) if skeleton_f1_values else 0.0,
        "cldice": cldice,
        "mask_brier": _brier_score_from_probs(mask_prob, batch["mask"]),
        "skeleton_brier": _brier_score_from_probs(skeleton_prob, batch["skeleton"]),
        "junction_brier": _brier_score_from_probs(junction_prob, batch["junction"]),
        "endpoint_brier": _brier_score_from_probs(endpoint_prob, batch["endpoint"]),
        "mask_ece": _expected_calibration_error_from_probs(mask_prob, batch["mask"]),
        "skeleton_ece": _expected_calibration_error_from_probs(skeleton_prob, batch["skeleton"]),
        "junction_ece": _expected_calibration_error_from_probs(
            junction_prob,
            (batch["junction"] >= junction_threshold).float(),
        ),
        "endpoint_ece": _expected_calibration_error_from_probs(
            endpoint_prob,
            (batch["endpoint"] >= endpoint_threshold).float(),
        ),
        "junction_f1": graph_metrics.junction_f1,
        "endpoint_f1": graph_metrics.endpoint_f1,
        "component_error": graph_metrics.component_error,
        "branch_count_error": graph_metrics.branch_count_error,
        "length_error": graph_metrics.length_error,
        "publish_score": (
            0.25 * cldice
            + 0.20 * (float(np.mean(mask_f1_values)) if mask_f1_values else 0.0)
            + 0.20 * (float(np.mean(skeleton_f1_values)) if skeleton_f1_values else 0.0)
            + 0.15 * graph_metrics.junction_f1
            + 0.10 * graph_metrics.endpoint_f1
            + 0.10 * max(0.0, 1.0 - graph_metrics.component_error)
        ),
    }
