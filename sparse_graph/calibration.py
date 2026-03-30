from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


HEAD_TO_LOGIT_KEY = {
    "mask": "mask_logits",
    "skeleton": "skeleton_logits",
    "junction": "junction_logits",
    "endpoint": "endpoint_logits",
    "uncertainty": "uncertainty_logits",
}


def load_calibration(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_temperature_scaling(
    predictions: dict[str, torch.Tensor],
    calibration: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    if not calibration:
        return predictions
    temperatures = calibration.get("temperatures", {})
    scaled = dict(predictions)
    for head, logit_key in HEAD_TO_LOGIT_KEY.items():
        if logit_key not in predictions:
            continue
        temperature = float(temperatures.get(head, 1.0))
        scaled[logit_key] = predictions[logit_key] / max(temperature, 1e-6)
    return scaled


def resolve_metric_thresholds(
    config: dict[str, Any],
    calibration: dict[str, Any] | None = None,
) -> dict[str, float]:
    post_cfg = dict(config.get("postprocess", {}))
    thresholds = {
        "mask_threshold": float(post_cfg.get("mask_threshold", 0.5)),
        "skeleton_threshold": float(post_cfg.get("skeleton_threshold", 0.5)),
        "junction_threshold": float(post_cfg.get("junction_threshold", post_cfg.get("node_threshold", 0.5))),
        "endpoint_threshold": float(post_cfg.get("endpoint_threshold", post_cfg.get("node_threshold", 0.5))),
    }
    if calibration:
        thresholds.update({key: float(value) for key, value in calibration.get("thresholds", {}).items()})
    return thresholds


def resolve_postprocess_config(
    config: dict[str, Any],
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    post_cfg = dict(config.get("postprocess", {}))
    if calibration:
        for key, value in calibration.get("postprocess", {}).items():
            post_cfg[key] = value
        thresholds = calibration.get("thresholds", {})
        for key in ("mask_threshold", "skeleton_threshold", "junction_threshold", "endpoint_threshold"):
            if key in thresholds:
                post_cfg[key] = float(thresholds[key])
    return post_cfg
