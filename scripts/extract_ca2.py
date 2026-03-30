from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sparse_graph.graph.builder import GraphBuilder
from sparse_graph.models import TopoSparseNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deployable sparse extraction on unlabeled Ca2 input."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint (.pt).")
    parser.add_argument("--input", required=True, help="Path to input movie or summary stack.")
    parser.add_argument("--output-dir", required=True, help="Folder to save extraction outputs.")
    parser.add_argument(
        "--input-mode",
        choices=("auto", "movie", "summary"),
        default="auto",
        help="Whether the input is a raw time movie or already-built summary channels.",
    )
    parser.add_argument(
        "--time-axis",
        type=int,
        default=0,
        help="Time axis for raw movies. Default assumes [T, H, W].",
    )
    parser.add_argument("--device", default=None, help="Override checkpoint device, e.g. cpu or mps.")
    parser.add_argument(
        "--tta",
        choices=("none", "flips"),
        default="flips",
        help="Use simple flip test-time augmentation for unlabeled robustness.",
    )
    parser.add_argument("--node-threshold", type=float, default=None)
    parser.add_argument("--skeleton-threshold", type=float, default=None)
    parser.add_argument("--max-neighbor-distance", type=int, default=None)
    parser.add_argument("--min-path-support", type=float, default=None)
    parser.add_argument("--max-neighbors-per-node", type=int, default=None)
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


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as data:
            if "movie" in data:
                return data["movie"]
            if "image" in data:
                return data["image"]
            first_key = next(iter(data.keys()))
            return data[first_key]
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading TIFF stacks requires tifffile. Install requirements.txt first."
            ) from exc
        return tifffile.imread(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def to_channel_first_summary(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        raise ValueError("A summary stack must have channels, not a single 2D frame.")
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D summary stack, got shape {array.shape}.")
    if array.shape[0] <= 8:
        return array.astype(np.float32)
    if array.shape[-1] <= 8:
        return np.moveaxis(array, -1, 0).astype(np.float32)
    raise ValueError(
        "Could not infer summary channels. Use --input-mode movie for raw time stacks."
    )


def percentile_normalize(image: np.ndarray) -> np.ndarray:
    low = float(np.percentile(image, 1.0))
    high = float(np.percentile(image, 99.0))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    image = (image - low) / (high - low)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def local_correlation_map(movie: np.ndarray) -> np.ndarray:
    centered = movie - movie.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True) + 1e-6
    z_movie = centered / scale

    _, height, width = z_movie.shape
    neighbor_sum = np.zeros_like(z_movie, dtype=np.float32)
    neighbor_count = np.zeros((height, width), dtype=np.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        src_y = slice(max(0, -dy), height - max(0, dy))
        src_x = slice(max(0, -dx), width - max(0, dx))
        dst_y = slice(max(0, dy), height - max(0, -dy))
        dst_x = slice(max(0, dx), width - max(0, -dx))
        neighbor_sum[:, dst_y, dst_x] += z_movie[:, src_y, src_x]
        neighbor_count[dst_y, dst_x] += 1.0

    neighbor_mean = neighbor_sum / np.clip(neighbor_count[None, ...], 1.0, None)
    return (z_movie * neighbor_mean).mean(axis=0).astype(np.float32)


def build_summary_from_movie(movie: np.ndarray, time_axis: int) -> np.ndarray:
    if movie.ndim != 3:
        raise ValueError(f"Expected a 3D raw movie, got shape {movie.shape}.")
    movie = np.moveaxis(movie, time_axis, 0).astype(np.float32)
    movie = np.nan_to_num(movie, copy=False)

    mean_map = percentile_normalize(movie.mean(axis=0))
    max_map = percentile_normalize(movie.max(axis=0))
    std_map = percentile_normalize(movie.std(axis=0))
    corr_map = percentile_normalize(local_correlation_map(movie))
    return np.stack([mean_map, max_map, std_map, corr_map], axis=0).astype(np.float32)


def infer_input_mode(array: np.ndarray, requested: str) -> str:
    if requested != "auto":
        return requested
    if array.ndim == 3 and (array.shape[0] <= 8 or array.shape[-1] <= 8):
        return "summary"
    return "movie"


def resize_summary(summary: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(summary[None, ...]).float()
    if tensor.shape[-2:] != (image_size, image_size):
        tensor = F.interpolate(
            tensor,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
    return tensor


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[TopoSparseNet, dict[str, Any]]:
    config = checkpoint["config"]
    model_cfg = config["model"]
    model = TopoSparseNet(
        in_channels=int(model_cfg["in_channels"]),
        base_channels=int(model_cfg["base_channels"]),
        channel_multipliers=tuple(model_cfg["channel_multipliers"]),
        use_attention=bool(model_cfg["use_attention"]),
        affinity_channels=int(model_cfg["affinity_channels"]),
        use_topology_refinement=bool(model_cfg.get("use_topology_refinement", False)),
        topology_variant=str(model_cfg.get("topology_variant", "none")),
        morphology_channels=tuple(model_cfg.get("morphology_channels", [0, 1])),
        activity_channels=tuple(model_cfg.get("activity_channels", [2, 3])),
    )
    model.load_state_dict(checkpoint["model"])
    return model, config


def predict_with_tta(
    model: TopoSparseNet,
    image: torch.Tensor,
    device: torch.device,
    tta: str,
) -> dict[str, torch.Tensor]:
    variants: list[tuple[torch.Tensor, tuple[int, ...]]] = [(image, ())]
    if tta == "flips":
        variants.extend(
            [
                (torch.flip(image, dims=(-1,)), (-1,)),
                (torch.flip(image, dims=(-2,)), (-2,)),
                (torch.flip(image, dims=(-2, -1)), (-2, -1)),
            ]
        )

    aggregated: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for augmented, flip_dims in variants:
            predictions = model(augmented.to(device))
            for key, value in predictions.items():
                restored = torch.flip(value, dims=flip_dims) if flip_dims else value
                aggregated[key] = aggregated.get(key, torch.zeros_like(restored)) + restored

    factor = float(len(variants))
    return {key: value / factor for key, value in aggregated.items()}


def graph_to_json(graph_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "index": int(node.index),
                "y": int(node.y),
                "x": int(node.x),
                "kind": node.kind,
                "score": float(node.score),
            }
            for node in graph_result["nodes"]
        ],
        "edges": [
            {
                "source": int(edge.source),
                "target": int(edge.target),
                "score": float(edge.score),
            }
            for edge in graph_result["edges"]
        ],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model, config = build_model_from_checkpoint(checkpoint)
    device = resolve_device(args.device or config.get("device"))
    model = model.to(device)
    model.eval()

    array = load_array(input_path)
    input_mode = infer_input_mode(array, args.input_mode)
    if input_mode == "summary":
        summary = to_channel_first_summary(array)
    else:
        summary = build_summary_from_movie(array, time_axis=args.time_axis)

    image_size = int(config["data"]["image_size"])
    image = resize_summary(summary, image_size=image_size)
    predictions = predict_with_tta(model, image, device=device, tta=args.tta)

    post_cfg = dict(config.get("postprocess", {}))
    overrides = {
        "node_threshold": args.node_threshold,
        "skeleton_threshold": args.skeleton_threshold,
        "max_neighbor_distance": args.max_neighbor_distance,
        "min_path_support": args.min_path_support,
        "max_neighbors_per_node": args.max_neighbors_per_node,
    }
    for key, value in overrides.items():
        if value is not None:
            post_cfg[key] = value
    graph_builder = GraphBuilder(**post_cfg)
    graph_result = graph_builder(predictions)

    mask_prob = torch.sigmoid(predictions["mask_logits"])[0, 0].detach().cpu().numpy()
    skeleton_prob = torch.sigmoid(predictions["skeleton_logits"])[0, 0].detach().cpu().numpy()
    junction_prob = torch.sigmoid(predictions["junction_logits"])[0, 0].detach().cpu().numpy()
    endpoint_prob = torch.sigmoid(predictions["endpoint_logits"])[0, 0].detach().cpu().numpy()
    uncertainty_prob = torch.sigmoid(predictions["uncertainty_logits"])[0, 0].detach().cpu().numpy()
    affinity = predictions["affinity"][0].detach().cpu().numpy()
    confidence_map = np.clip((mask_prob + skeleton_prob) * 0.5 - 0.5 * uncertainty_prob, 0.0, 1.0)

    extra_outputs: dict[str, np.ndarray] = {}
    for key in (
        "morphology_prior_logits",
        "activity_prior_logits",
        "agreement_logits",
        "conflict_logits",
    ):
        if key in predictions:
            export_key = key.replace("_logits", "_prob")
            extra_outputs[export_key] = (
                torch.sigmoid(predictions[key])[0, 0].detach().cpu().numpy().astype(np.float32)
            )

    np.savez_compressed(
        output_dir / "predictions.npz",
        summary=image[0].cpu().numpy(),
        mask_prob=mask_prob.astype(np.float32),
        skeleton_prob=skeleton_prob.astype(np.float32),
        junction_prob=junction_prob.astype(np.float32),
        endpoint_prob=endpoint_prob.astype(np.float32),
        uncertainty_prob=uncertainty_prob.astype(np.float32),
        confidence_map=confidence_map.astype(np.float32),
        affinity=affinity.astype(np.float32),
        **extra_outputs,
    )

    with (output_dir / "graph.json").open("w", encoding="utf-8") as handle:
        json.dump(graph_to_json(graph_result), handle, indent=2)

    metadata = {
        "input_path": str(input_path),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "input_mode": input_mode,
        "tta": args.tta,
        "num_nodes": len(graph_result["nodes"]),
        "num_edges": len(graph_result["edges"]),
        "mean_confidence": float(confidence_map.mean()),
        "model_config": config["model"],
        "postprocess": post_cfg,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"Saved extraction outputs to {output_dir} | "
        f"nodes={len(graph_result['nodes'])} | edges={len(graph_result['edges'])} | "
        f"mean_confidence={confidence_map.mean():.4f}"
    )


if __name__ == "__main__":
    main()
