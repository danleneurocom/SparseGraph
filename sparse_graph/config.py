from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "device": "cpu",
    "output_dir": "runs/toposparsenet_base",
    "model": {
        "in_channels": 4,
        "base_channels": 32,
        "channel_multipliers": [1, 2, 4, 8],
        "use_attention": True,
        "affinity_channels": 2,
        "use_topology_refinement": False,
        "topology_variant": "none",
        "morphology_channels": [0, 1],
        "activity_channels": [2, 3],
        "graph_embedding_channels": 16,
    },
    "data": {
        "dataset_type": "synthetic",
        "train_dir": "",
        "val_dir": "",
        "image_size": 256,
        "train_samples": 128,
        "val_samples": 32,
        "batch_size": 4,
        "num_workers": 0,
    },
    "training": {
        "epochs": 5,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "log_every": 10,
        "init_checkpoint": "",
    },
    "loss": {
        "mask": 1.0,
        "skeleton": 1.0,
        "topology": 0.5,
        "node": 0.5,
        "node_tolerance": 0.0,
        "affinity": 0.5,
        "uncertainty": 0.1,
        "auxiliary": 0.0,
        "consistency": 0.0,
        "support": 0.0,
        "bio_prior": 0.0,
        "causal": 0.0,
        "relay": 0.0,
        "node_policy": 0.0,
        "branch_budget": 0.0,
        "branch_austerity": 0.0,
        "graph_importance": 0.0,
        "graph_minimal": 0.0,
        "graph": 0.0,
        "skeleton_iterations": 10,
    },
    "postprocess": {
        "node_threshold": 0.5,
        "mask_threshold": 0.5,
        "skeleton_threshold": 0.5,
        "junction_threshold": 0.5,
        "endpoint_threshold": 0.5,
        "max_neighbor_distance": 24,
        "min_path_support": 0.35,
        "max_neighbors_per_node": 2,
        "decode_mode": "geodesic",
        "nms_radius": 4,
        "max_path_ratio": 1.85,
        "affinity_weight": 0.2,
        "uncertainty_weight": 0.75,
        "bridge_weight": 0.2,
        "relay_weight": 0.15,
        "mask_weight": 0.1,
        "relation_weight": 0.55,
        "causal_weight": 0.25,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        deep_update(config, loaded)
    return config
