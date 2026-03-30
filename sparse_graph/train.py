from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import CalciumSummaryNpzDataset, SyntheticCalciumDataset
from .losses import TopoSparseObjective
from .metrics import publication_metrics
from .models import TopoSparseNet
from .utils import MetricTracker, move_batch_to_device, set_seed, to_float_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TopoSparseNet.")
    parser.add_argument("--config", default="configs/toposparsenet_base.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def build_datasets(config: dict[str, Any]):
    data_cfg = config["data"]
    dataset_type = data_cfg["dataset_type"]
    if dataset_type == "synthetic":
        train_dataset = SyntheticCalciumDataset(
            num_samples=int(data_cfg["train_samples"]),
            image_size=int(data_cfg["image_size"]),
            in_channels=int(config["model"]["in_channels"]),
            seed=int(config["seed"]),
        )
        val_dataset = SyntheticCalciumDataset(
            num_samples=int(data_cfg["val_samples"]),
            image_size=int(data_cfg["image_size"]),
            in_channels=int(config["model"]["in_channels"]),
            seed=int(config["seed"]) + 1,
        )
        return train_dataset, val_dataset

    if dataset_type == "npz":
        return CalciumSummaryNpzDataset(data_cfg["train_dir"]), CalciumSummaryNpzDataset(
            data_cfg["val_dir"]
        )

    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def build_dataloaders(config: dict[str, Any]):
    train_dataset, val_dataset = build_datasets(config)
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config["model"]
    return TopoSparseNet(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_multipliers=tuple(model_cfg["channel_multipliers"]),
        use_attention=bool(model_cfg["use_attention"]),
        affinity_channels=int(model_cfg["affinity_channels"]),
        use_topology_refinement=bool(model_cfg.get("use_topology_refinement", False)),
        topology_variant=str(model_cfg.get("topology_variant", "none")),
        morphology_channels=tuple(model_cfg.get("morphology_channels", [0, 1])),
        activity_channels=tuple(model_cfg.get("activity_channels", [2, 3])),
        graph_embedding_channels=int(model_cfg.get("graph_embedding_channels", 16)),
    )


def load_compatible_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str]]:
    model_state = model.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    skipped_mismatch: list[str] = []
    unexpected: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_mismatch.append(key)
            continue
        filtered_state[key] = value
    missing = sorted(set(model_state.keys()) - set(filtered_state.keys()))
    model.load_state_dict(filtered_state, strict=False)
    return missing, unexpected, skipped_mismatch


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    objective: TopoSparseObjective,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    split: str,
    log_every: int,
    include_publication_metrics: bool = False,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    tracker = MetricTracker()
    progress = tqdm(loader, desc=f"{split} {epoch:02d}", leave=False)

    for step, batch in enumerate(progress, start=1):
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(is_train):
            predictions = model(batch["image"])
            loss, metrics = objective(predictions, batch)
            if include_publication_metrics:
                metrics.update(
                    publication_metrics(
                        predictions,
                        batch,
                        skeleton_iterations=int(objective.weights.get("skeleton_iterations", 10)),
                    )
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        float_metrics = to_float_dict(metrics)
        batch_size = int(batch["image"].shape[0])
        tracker.update(float_metrics, n=batch_size)

        if step % max(log_every, 1) == 0 or step == len(loader):
            avg = tracker.averages()
            progress.set_postfix(
                loss=f"{avg['loss']:.4f}",
                mask_dice=f"{avg['mask_dice']:.4f}",
                skeleton_dice=f"{avg['skeleton_dice']:.4f}",
                cldice=f"{avg.get('cldice', 0.0):.4f}",
            )

    return tracker.averages()


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    metrics: dict[str, float],
    is_best: bool,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
    }
    torch.save(checkpoint, output_dir / "last.pt")
    if is_best:
        torch.save(checkpoint, output_dir / "best.pt")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.train_samples is not None:
        config["data"]["train_samples"] = args.train_samples
    if args.val_samples is not None:
        config["data"]["val_samples"] = args.val_samples
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size

    set_seed(int(config["seed"]))

    requested_device = str(config["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    init_checkpoint = str(config["training"].get("init_checkpoint", "")).strip()
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location=device)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected, skipped_mismatch = load_compatible_state_dict(model, state_dict)
        print(
            f"Initialized model from {init_checkpoint} | "
            f"missing={len(missing)} unexpected={len(unexpected)} skipped_mismatch={len(skipped_mismatch)}"
        )
    objective = TopoSparseObjective(config["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_val = float("-inf")
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            objective=objective,
            optimizer=optimizer,
            device=device,
            grad_clip=float(config["training"]["grad_clip"]),
            epoch=epoch,
            split="train",
            log_every=int(config["training"]["log_every"]),
            include_publication_metrics=False,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            objective=objective,
            optimizer=None,
            device=device,
            grad_clip=0.0,
            epoch=epoch,
            split="val",
            log_every=int(config["training"]["log_every"]),
            include_publication_metrics=True,
        )

        selection_metric = float(val_metrics.get("publish_score", val_metrics.get("selection_score", 0.0)))
        is_best = selection_metric > best_val
        if is_best:
            best_val = selection_metric
        save_checkpoint(output_dir, epoch, model, optimizer, config, val_metrics, is_best=is_best)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val mask dice {val_metrics['mask_dice']:.4f} | "
            f"val skeleton dice {val_metrics['skeleton_dice']:.4f} | "
            f"val cldice {val_metrics.get('cldice', 0.0):.4f} | "
            f"val publish score {val_metrics.get('publish_score', selection_metric):.4f}"
        )
