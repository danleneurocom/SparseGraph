from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from skimage.io import imsave
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sparse_graph.calibration import (
    apply_temperature_scaling,
    load_calibration,
    resolve_postprocess_config,
)
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
        "--calibration-json",
        default=None,
        help="Optional temperature/threshold calibration file.",
    )
    parser.add_argument(
        "--tta",
        choices=("none", "flips"),
        default="flips",
        help="Use simple flip test-time augmentation for unlabeled robustness.",
    )
    parser.add_argument(
        "--preset",
        choices=("default", "conservative", "reviewer"),
        default="default",
        help="Use a tuned extraction preset. Reviewer favors a cleaner minimal graph.",
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
        graph_embedding_channels=int(model_cfg.get("graph_embedding_channels", 16)),
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
                "degree_hint": float(getattr(node, "degree_hint", 0.0)),
            }
            for node in graph_result["nodes"]
        ],
        "edges": [
            {
                "source": int(edge.source),
                "target": int(edge.target),
                "score": float(edge.score),
                "path": [[int(y), int(x)] for y, x in edge.path],
            }
            for edge in graph_result["edges"]
        ],
    }


def candidate_bridges_to_json(
    graph_result: dict[str, Any],
) -> dict[str, Any]:
    node_lookup = {int(node.index): node for node in graph_result.get("raw_nodes", graph_result.get("nodes", []))}
    return {
        "candidate_bridges": [
            {
                "source": int(bridge.source),
                "target": int(bridge.target),
                "score": float(bridge.score),
                "relation_prob": float(bridge.relation_prob),
                "path_support": float(bridge.path_support),
                "uncertainty": float(bridge.uncertainty),
                "source_kind": node_lookup.get(int(bridge.source)).kind if int(bridge.source) in node_lookup else "unknown",
                "target_kind": node_lookup.get(int(bridge.target)).kind if int(bridge.target) in node_lookup else "unknown",
                "source_y": int(node_lookup.get(int(bridge.source)).y) if int(bridge.source) in node_lookup else -1,
                "source_x": int(node_lookup.get(int(bridge.source)).x) if int(bridge.source) in node_lookup else -1,
                "target_y": int(node_lookup.get(int(bridge.target)).y) if int(bridge.target) in node_lookup else -1,
                "target_x": int(node_lookup.get(int(bridge.target)).x) if int(bridge.target) in node_lookup else -1,
                "path": [[int(y), int(x)] for y, x in bridge.path],
            }
            for bridge in graph_result.get("candidate_bridges", [])
        ]
    }


def apply_preset(post_cfg: dict[str, Any], preset: str) -> dict[str, Any]:
    if preset == "default":
        post_cfg.setdefault("prune_mode", "none")
        return post_cfg

    if preset == "conservative":
        post_cfg.update(
            {
                "node_threshold": max(float(post_cfg.get("node_threshold", 0.4)), 0.48),
                "skeleton_threshold": max(float(post_cfg.get("skeleton_threshold", 0.45)), 0.54),
                "max_neighbor_distance": min(int(post_cfg.get("max_neighbor_distance", 20)), 16),
                "min_path_support": max(float(post_cfg.get("min_path_support", 0.28)), 0.38),
                "max_neighbors_per_node": min(int(post_cfg.get("max_neighbors_per_node", 3)), 2),
                "prune_mode": "conservative",
                "prune_terminal_score": 0.72,
                "prune_terminal_node_score": 0.84,
                "prune_min_spur_length": 36,
                "prune_keep_trunk_length": 96,
                "prune_keep_edge_score": 0.92,
                "prune_min_component_length": 88,
                "prune_component_score": 0.80,
                "enforce_tree": False,
            }
        )
        return post_cfg

    if preset == "reviewer":
        post_cfg.update(
            {
                "node_threshold": max(float(post_cfg.get("node_threshold", 0.4)), 0.58),
                "skeleton_threshold": max(float(post_cfg.get("skeleton_threshold", 0.45)), 0.64),
                "max_neighbor_distance": min(int(post_cfg.get("max_neighbor_distance", 20)), 12),
                "min_path_support": max(float(post_cfg.get("min_path_support", 0.28)), 0.52),
                "max_neighbors_per_node": min(int(post_cfg.get("max_neighbors_per_node", 3)), 2),
                "prune_mode": "reviewer",
                "prune_terminal_score": 0.82,
                "prune_terminal_node_score": 0.90,
                "prune_min_spur_length": 52,
                "prune_keep_trunk_length": 112,
                "prune_keep_edge_score": 0.95,
                "prune_min_component_length": 120,
                "prune_component_score": 0.88,
                "enforce_tree": True,
            }
        )
        return post_cfg

    return post_cfg


def _normalize_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.nan_to_num(image, copy=False)
    low = float(np.percentile(image, 1.0))
    high = float(np.percentile(image, 99.0))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def summary_to_rgb(summary: np.ndarray) -> np.ndarray:
    summary = np.asarray(summary, dtype=np.float32)
    if summary.ndim != 3:
        raise ValueError(f"Expected [C, H, W] summary, got shape {summary.shape}")

    if summary.shape[0] >= 4:
        channels = [summary[1], summary[0], summary[3]]
    elif summary.shape[0] == 3:
        channels = [summary[0], summary[1], summary[2]]
    elif summary.shape[0] == 2:
        channels = [summary[0], summary[1], summary[0]]
    else:
        channels = [summary[0], summary[0], summary[0]]
    return np.stack([_normalize_display(channel) for channel in channels], axis=-1)


def _paint_square(
    canvas: np.ndarray,
    y: int,
    x: int,
    color: tuple[float, float, float],
    radius: int,
) -> None:
    height, width, _ = canvas.shape
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    canvas[y0:y1, x0:x1] = np.asarray(color, dtype=np.float32)


def _stack_panels(panels: list[np.ndarray], separator_width: int = 8) -> np.ndarray:
    separator = np.ones((panels[0].shape[0], separator_width, 3), dtype=np.float32)
    stitched: list[np.ndarray] = []
    for index, panel in enumerate(panels):
        stitched.append(np.clip(panel, 0.0, 1.0))
        if index < len(panels) - 1:
            stitched.append(separator)
    return np.concatenate(stitched, axis=1)


def render_extraction_visual(
    summary: np.ndarray,
    mask_prob: np.ndarray,
    skeleton_prob: np.ndarray,
    graph_result: dict[str, Any],
    candidate_bridges: list[Any] | None = None,
    dense_sparse_projection_prob: np.ndarray | None = None,
    skeleton_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    original_rgb = summary_to_rgb(summary)
    dense_rgb = original_rgb * 0.78
    sparse_rgb = original_rgb * 0.60
    graph_rgb = original_rgb * 0.72
    candidate_rgb = original_rgb * 0.72
    sparse_only_rgb = np.zeros_like(original_rgb, dtype=np.float32)
    graph_only_rgb = np.zeros_like(original_rgb, dtype=np.float32)
    candidate_only_rgb = np.zeros_like(original_rgb, dtype=np.float32)

    mask_alpha = np.clip(mask_prob * 0.22, 0.0, 0.22)[..., None]
    mask_color = np.array([1.0, 0.45, 0.12], dtype=np.float32)
    dense_rgb = dense_rgb * (1.0 - mask_alpha) + mask_alpha * mask_color

    skeleton_alpha = np.clip(skeleton_prob * 0.90, 0.0, 0.90)[..., None]
    skeleton_color = np.array([0.08, 0.92, 1.0], dtype=np.float32)
    sparse_rgb = sparse_rgb * (1.0 - 0.40 * skeleton_alpha) + skeleton_alpha * skeleton_color
    graph_rgb = graph_rgb * (1.0 - 0.40 * skeleton_alpha) + skeleton_alpha * skeleton_color
    sparse_only_rgb = sparse_only_rgb + skeleton_alpha * skeleton_color

    if dense_sparse_projection_prob is not None:
        projection_alpha = np.clip(dense_sparse_projection_prob * 0.25, 0.0, 0.25)[..., None]
        projection_color = np.array([0.92, 0.20, 0.88], dtype=np.float32)
        sparse_rgb = sparse_rgb * (1.0 - projection_alpha) + projection_alpha * projection_color
        graph_rgb = graph_rgb * (1.0 - 0.5 * projection_alpha) + 0.5 * projection_alpha * projection_color
        candidate_rgb = candidate_rgb * (1.0 - 0.5 * projection_alpha) + 0.5 * projection_alpha * projection_color
        sparse_only_rgb = sparse_only_rgb + np.clip(dense_sparse_projection_prob * 0.45, 0.0, 0.45)[..., None] * projection_color

    binary_skeleton = (skeleton_prob >= float(skeleton_threshold)).astype(np.float32)
    binary_skeleton_rgb = np.repeat(binary_skeleton[..., None], 3, axis=2)

    for edge in graph_result["edges"]:
        for y, x in edge.path:
            _paint_square(graph_rgb, int(y), int(x), color=(1.0, 0.88, 0.18), radius=1)
            _paint_square(graph_only_rgb, int(y), int(x), color=(1.0, 0.88, 0.18), radius=1)
    for node in graph_result["nodes"]:
        color = (1.0, 0.18, 0.20) if node.kind == "junction" else (0.15, 1.0, 0.35)
        _paint_square(graph_rgb, int(node.y), int(node.x), color=color, radius=2)
        _paint_square(graph_only_rgb, int(node.y), int(node.x), color=color, radius=2)
        _paint_square(candidate_rgb, int(node.y), int(node.x), color=color, radius=2)

    for bridge in candidate_bridges or []:
        for y, x in bridge.path:
            _paint_square(candidate_rgb, int(y), int(x), color=(0.96, 0.38, 1.0), radius=1)
            _paint_square(candidate_only_rgb, int(y), int(x), color=(0.96, 0.38, 1.0), radius=1)

    dense_rgb = np.clip(dense_rgb, 0.0, 1.0)
    sparse_rgb = np.clip(sparse_rgb, 0.0, 1.0)
    graph_rgb = np.clip(graph_rgb, 0.0, 1.0)
    candidate_rgb = np.clip(candidate_rgb, 0.0, 1.0)
    sparse_only_rgb = np.clip(sparse_only_rgb, 0.0, 1.0)
    graph_only_rgb = np.clip(graph_only_rgb, 0.0, 1.0)
    candidate_only_rgb = np.clip(candidate_only_rgb, 0.0, 1.0)
    comparison_rgb = _stack_panels([original_rgb, graph_rgb])
    storyboard_rgb = _stack_panels([original_rgb, dense_rgb, sparse_rgb, graph_rgb])
    sparse_storyboard_rgb = _stack_panels([sparse_only_rgb, binary_skeleton_rgb, graph_only_rgb])
    mapping_storyboard_rgb = _stack_panels([original_rgb, graph_rgb, candidate_rgb, candidate_only_rgb])
    return {
        "original": original_rgb,
        "dense": dense_rgb,
        "sparse": sparse_rgb,
        "sparse_only": sparse_only_rgb,
        "binary_skeleton": binary_skeleton_rgb,
        "graph": graph_rgb,
        "graph_only": graph_only_rgb,
        "candidate": candidate_rgb,
        "candidate_only": candidate_only_rgb,
        "comparison": comparison_rgb,
        "storyboard": storyboard_rgb,
        "sparse_storyboard": sparse_storyboard_rgb,
        "mapping_storyboard": mapping_storyboard_rgb,
    }


def save_visual_outputs(
    output_dir: Path,
    summary: np.ndarray,
    mask_prob: np.ndarray,
    skeleton_prob: np.ndarray,
    graph_result: dict[str, Any],
    candidate_bridges: list[Any] | None = None,
    dense_sparse_projection_prob: np.ndarray | None = None,
    skeleton_threshold: float = 0.5,
) -> dict[str, str]:
    visuals = render_extraction_visual(
        summary=summary,
        mask_prob=mask_prob,
        skeleton_prob=skeleton_prob,
        graph_result=graph_result,
        candidate_bridges=candidate_bridges,
        dense_sparse_projection_prob=dense_sparse_projection_prob,
        skeleton_threshold=skeleton_threshold,
    )
    visual_paths = {
        "original_view": output_dir / "original_view.png",
        "dense_view": output_dir / "dense_view.png",
        "sparse_view": output_dir / "sparse_view.png",
        "sparse_only_view": output_dir / "sparse_only.png",
        "binary_skeleton_view": output_dir / "binary_skeleton.png",
        "graph_view": output_dir / "graph_view.png",
        "graph_only_view": output_dir / "graph_only.png",
        "candidate_view": output_dir / "candidate_view.png",
        "candidate_only_view": output_dir / "candidate_only.png",
        "extraction_view": output_dir / "extraction_view.png",
        "comparison_view": output_dir / "comparison.png",
        "storyboard_view": output_dir / "storyboard.png",
        "sparse_storyboard_view": output_dir / "sparse_storyboard.png",
        "mapping_storyboard_view": output_dir / "mapping_storyboard.png",
    }
    visual_arrays = {
        "original_view": visuals["original"],
        "dense_view": visuals["dense"],
        "sparse_view": visuals["sparse"],
        "sparse_only_view": visuals["sparse_only"],
        "binary_skeleton_view": visuals["binary_skeleton"],
        "graph_view": visuals["graph"],
        "graph_only_view": visuals["graph_only"],
        "candidate_view": visuals["candidate"],
        "candidate_only_view": visuals["candidate_only"],
        "extraction_view": visuals["graph"],
        "comparison_view": visuals["comparison"],
        "storyboard_view": visuals["storyboard"],
        "sparse_storyboard_view": visuals["sparse_storyboard"],
        "mapping_storyboard_view": visuals["mapping_storyboard"],
    }
    for key, path in visual_paths.items():
        imsave(path, (visual_arrays[key] * 255.0).astype(np.uint8), check_contrast=False)
    return {key: str(path) for key, path in visual_paths.items()}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model, config = build_model_from_checkpoint(checkpoint)
    calibration = load_calibration(args.calibration_json)
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
    predictions = apply_temperature_scaling(
        predict_with_tta(model, image, device=device, tta=args.tta),
        calibration,
    )

    post_cfg = apply_preset(resolve_postprocess_config(config, calibration), args.preset)
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
    dense_sparse_projection_prob = None
    if "dense_sparse_projection_logits" in predictions:
        dense_sparse_projection_prob = (
            torch.sigmoid(predictions["dense_sparse_projection_logits"])[0, 0].detach().cpu().numpy()
        )

    extra_outputs: dict[str, np.ndarray] = {}
    for key in (
        "morphology_prior_logits",
        "activity_prior_logits",
        "agreement_logits",
        "conflict_logits",
        "node_capacity_logits",
        "causal_saliency_logits",
        "dense_sparse_projection_logits",
        "path_memory_logits",
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

    visual_paths = save_visual_outputs(
        output_dir=output_dir,
        summary=image[0].cpu().numpy(),
        mask_prob=mask_prob.astype(np.float32),
        skeleton_prob=skeleton_prob.astype(np.float32),
        graph_result=graph_result,
        candidate_bridges=graph_result.get("candidate_bridges", []),
        dense_sparse_projection_prob=dense_sparse_projection_prob.astype(np.float32)
        if dense_sparse_projection_prob is not None
        else None,
        skeleton_threshold=float(post_cfg.get("skeleton_threshold", 0.5)),
    )

    with (output_dir / "graph.json").open("w", encoding="utf-8") as handle:
        json.dump(graph_to_json(graph_result), handle, indent=2)
    if "raw_nodes" in graph_result and "raw_edges" in graph_result:
        with (output_dir / "graph_raw.json").open("w", encoding="utf-8") as handle:
            json.dump(
                graph_to_json(
                    {
                        "nodes": graph_result["raw_nodes"],
                        "edges": graph_result["raw_edges"],
                    }
                ),
                handle,
                indent=2,
            )
    if graph_result.get("candidate_bridges"):
        with (output_dir / "candidate_bridges.json").open("w", encoding="utf-8") as handle:
            json.dump(candidate_bridges_to_json(graph_result), handle, indent=2)

    graph_mask = np.zeros_like(confidence_map, dtype=bool)
    for edge in graph_result["edges"]:
        for y, x in edge.path:
            graph_mask[int(y), int(x)] = True
    structure_mask = skeleton_prob >= float(post_cfg.get("skeleton_threshold", 0.5))
    candidate_mask = np.zeros_like(confidence_map, dtype=bool)
    for bridge in graph_result.get("candidate_bridges", []):
        for y, x in bridge.path:
            candidate_mask[int(y), int(x)] = True

    mean_structure_confidence = float(confidence_map[structure_mask].mean()) if np.any(structure_mask) else 0.0
    mean_graph_confidence = float(confidence_map[graph_mask].mean()) if np.any(graph_mask) else 0.0
    mean_edge_score = (
        float(np.mean([edge.score for edge in graph_result["edges"]])) if graph_result["edges"] else 0.0
    )
    mean_node_score = (
        float(np.mean([node.score for node in graph_result["nodes"]])) if graph_result["nodes"] else 0.0
    )
    mean_candidate_bridge_score = (
        float(np.mean([bridge.score for bridge in graph_result.get("candidate_bridges", [])]))
        if graph_result.get("candidate_bridges")
        else 0.0
    )
    mean_candidate_confidence = (
        float(confidence_map[candidate_mask].mean()) if np.any(candidate_mask) else 0.0
    )

    metadata = {
        "input_path": str(input_path),
        "checkpoint": str(args.checkpoint),
        "calibration": str(args.calibration_json) if args.calibration_json else None,
        "device": str(device),
        "input_mode": input_mode,
        "tta": args.tta,
        "preset": args.preset,
        "num_nodes": len(graph_result["nodes"]),
        "num_edges": len(graph_result["edges"]),
        "raw_num_nodes": len(graph_result.get("raw_nodes", graph_result["nodes"])),
        "raw_num_edges": len(graph_result.get("raw_edges", graph_result["edges"])),
        "num_candidate_bridges": len(graph_result.get("candidate_bridges", [])),
        "mean_confidence": float(confidence_map.mean()),
        "mean_confidence_all_pixels": float(confidence_map.mean()),
        "mean_confidence_structure": mean_structure_confidence,
        "mean_confidence_graph": mean_graph_confidence,
        "mean_confidence_candidate_bridges": mean_candidate_confidence,
        "mean_edge_score": mean_edge_score,
        "mean_node_score": mean_node_score,
        "mean_candidate_bridge_score": mean_candidate_bridge_score,
        "model_config": config["model"],
        "postprocess": post_cfg,
        "visuals": visual_paths,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"Saved extraction outputs to {output_dir} | "
        f"nodes={len(graph_result['nodes'])} | edges={len(graph_result['edges'])} | "
        f"candidate_bridges={len(graph_result.get('candidate_bridges', []))} | "
        f"graph_confidence={mean_graph_confidence:.4f}"
    )


if __name__ == "__main__":
    main()
