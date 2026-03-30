from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sparse_graph.config import deep_update
from sparse_graph.losses import TopoSparseObjective
from sparse_graph.metrics import publication_metrics
from sparse_graph.train import build_datasets, build_model
from sparse_graph.utils import MetricTracker, move_batch_to_device, set_seed, to_float_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint with publication-facing metrics.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Which configured split to evaluate.",
    )
    parser.add_argument("--train-dir", default=None, help="Optional override for config data.train_dir.")
    parser.add_argument("--val-dir", default=None, help="Optional override for config data.val_dir.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", default=None, help="Optional report path. Defaults next to the checkpoint.")
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


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    overrides = {"data": {}}
    if args.train_dir is not None:
        overrides["data"]["train_dir"] = args.train_dir
    if args.val_dir is not None:
        overrides["data"]["val_dir"] = args.val_dir
    if args.batch_size is not None:
        overrides["data"]["batch_size"] = args.batch_size
    deep_update(config, overrides)

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

    objective = TopoSparseObjective(config["loss"])
    tracker = MetricTracker()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"eval-{args.split}", leave=False):
            batch = move_batch_to_device(batch, device)
            predictions = model(batch["image"])
            _, metrics = objective(predictions, batch)
            metrics.update(
                publication_metrics(
                    predictions,
                    batch,
                    skeleton_iterations=int(config["loss"].get("skeleton_iterations", 10)),
                )
            )
            tracker.update(to_float_dict(metrics), n=int(batch["image"].shape[0]))

    report = {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "metrics": tracker.averages(),
    }

    output_path = Path(args.json_out) if args.json_out else checkpoint_path.with_name(
        f"{checkpoint_path.stem}_{args.split}_report.json"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
