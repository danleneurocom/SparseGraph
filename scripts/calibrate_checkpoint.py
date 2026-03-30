from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sparse_graph.calibration import apply_temperature_scaling, resolve_metric_thresholds
from sparse_graph.metrics import publication_metrics
from sparse_graph.train import build_datasets, build_model
from sparse_graph.utils import MetricTracker, move_batch_to_device, set_seed


TEMPERATURE_KEYS = ("mask", "skeleton", "junction", "endpoint")
THRESHOLD_KEYS = ("mask_threshold", "skeleton_threshold", "junction_threshold", "endpoint_threshold")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune post-hoc temperatures and thresholds on a split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        requested = "cpu"
    return torch.device(requested)


def detach_to_cpu(items: dict[str, Any]) -> dict[str, Any]:
    detached: dict[str, Any] = {}
    for key, value in items.items():
        detached[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return detached


def evaluate_cached(
    cached_batches: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    temperatures: dict[str, float],
    thresholds: dict[str, float],
    skeleton_iterations: int,
) -> dict[str, float]:
    tracker = MetricTracker()
    calibration = {"temperatures": temperatures}
    for predictions, batch in cached_batches:
        calibrated = apply_temperature_scaling(predictions, calibration)
        metrics = publication_metrics(
            calibrated,
            batch,
            thresholds=thresholds,
            skeleton_iterations=skeleton_iterations,
        )
        tracker.update(metrics, n=int(batch["image"].shape[0]))

    averaged = tracker.averages()
    averaged["calibration_objective"] = (
        averaged["publish_score"]
        - 0.05 * (averaged["mask_ece"] + averaged["skeleton_ece"])
        - 0.025 * (averaged["junction_ece"] + averaged["endpoint_ece"])
    )
    return averaged


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size

    set_seed(int(config["seed"]))
    device = resolve_device(args.device or config.get("device"))
    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    train_dataset, val_dataset = build_datasets(config)
    dataset = train_dataset if args.split == "train" else val_dataset
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
    )

    cached_batches: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"cache-{args.split}", leave=False):
            device_batch = move_batch_to_device(batch, device)
            predictions = model(device_batch["image"])
            cached_batches.append((detach_to_cpu(predictions), detach_to_cpu(batch)))

    temperatures = {key: 1.0 for key in TEMPERATURE_KEYS}
    thresholds = resolve_metric_thresholds(config)
    search_space = {
        "mask": [0.85, 1.0, 1.15, 1.3],
        "skeleton": [0.75, 0.9, 1.0, 1.1, 1.25],
        "junction": [0.7, 0.85, 1.0, 1.15, 1.3, 1.5],
        "endpoint": [0.7, 0.85, 1.0, 1.15, 1.3, 1.5],
        "mask_threshold": [0.35, 0.45, 0.5, 0.55, 0.65],
        "skeleton_threshold": [0.25, 0.35, 0.45, 0.5, 0.55, 0.65],
        "junction_threshold": [0.15, 0.25, 0.35, 0.45, 0.55],
        "endpoint_threshold": [0.15, 0.25, 0.35, 0.45, 0.55],
    }

    best_metrics = evaluate_cached(
        cached_batches,
        temperatures,
        thresholds,
        skeleton_iterations=int(config["loss"].get("skeleton_iterations", 10)),
    )
    best_objective = best_metrics["calibration_objective"]

    for _ in range(max(args.rounds, 1)):
        for key in TEMPERATURE_KEYS:
            local_best_value = temperatures[key]
            local_best_metrics = best_metrics
            local_best_objective = best_objective
            for candidate in search_space[key]:
                trial_temperatures = dict(temperatures)
                trial_temperatures[key] = float(candidate)
                trial_metrics = evaluate_cached(
                    cached_batches,
                    trial_temperatures,
                    thresholds,
                    skeleton_iterations=int(config["loss"].get("skeleton_iterations", 10)),
                )
                if trial_metrics["calibration_objective"] > local_best_objective:
                    local_best_value = float(candidate)
                    local_best_metrics = trial_metrics
                    local_best_objective = trial_metrics["calibration_objective"]
            temperatures[key] = local_best_value
            best_metrics = local_best_metrics
            best_objective = local_best_objective

        for key in THRESHOLD_KEYS:
            local_best_value = thresholds[key]
            local_best_metrics = best_metrics
            local_best_objective = best_objective
            for candidate in search_space[key]:
                trial_thresholds = dict(thresholds)
                trial_thresholds[key] = float(candidate)
                trial_metrics = evaluate_cached(
                    cached_batches,
                    temperatures,
                    trial_thresholds,
                    skeleton_iterations=int(config["loss"].get("skeleton_iterations", 10)),
                )
                if trial_metrics["calibration_objective"] > local_best_objective:
                    local_best_value = float(candidate)
                    local_best_metrics = trial_metrics
                    local_best_objective = trial_metrics["calibration_objective"]
            thresholds[key] = local_best_value
            best_metrics = local_best_metrics
            best_objective = local_best_objective

    report = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "temperatures": temperatures,
        "thresholds": thresholds,
        "postprocess": {
            "mask_threshold": thresholds["mask_threshold"],
            "skeleton_threshold": thresholds["skeleton_threshold"],
            "junction_threshold": thresholds["junction_threshold"],
            "endpoint_threshold": thresholds["endpoint_threshold"],
        },
        "metrics": best_metrics,
    }

    output_path = (
        Path(args.json_out)
        if args.json_out
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_{args.split}_calibration.json")
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
