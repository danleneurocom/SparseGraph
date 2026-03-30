from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.morphology import skeletonize
from skimage.transform import resize
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the public axon dataset archives into SparseGraph .npz samples."
    )
    parser.add_argument("--dataset-root", required=True, help="Folder containing images.rar, mask.rar, swc.rar.")
    parser.add_argument("--output-root", required=True, help="Destination folder for train/val/test .npz files.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--projection-axis", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split sample limit for smoke runs.")
    return parser.parse_args()


def percentile_normalize(image: np.ndarray) -> np.ndarray:
    low = float(np.percentile(image, 1.0))
    high = float(np.percentile(image, 99.0))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    image = (image - low) / (high - low)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def resize_2d(array: np.ndarray, image_size: int, order: int) -> np.ndarray:
    preserve_range = True
    anti_aliasing = order > 0
    return resize(
        array,
        (image_size, image_size),
        order=order,
        preserve_range=preserve_range,
        anti_aliasing=anti_aliasing,
    ).astype(np.float32)


def neighbor_degree(skeleton: np.ndarray) -> np.ndarray:
    padded = np.pad(skeleton, 1)
    degree = np.zeros_like(skeleton, dtype=np.float32)
    height, width = skeleton.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            degree += padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
    return degree * skeleton


def estimate_affinity(skeleton: np.ndarray) -> np.ndarray:
    height, width = skeleton.shape
    affinity = np.zeros((2, height, width), dtype=np.float32)
    ys, xs = np.where(skeleton > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        neighbors: list[np.ndarray] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width and skeleton[ny, nx] > 0:
                    neighbors.append(np.array([dx, dy], dtype=np.float32))
        if not neighbors:
            continue
        if len(neighbors) == 1:
            vector = neighbors[0]
        else:
            vector = max(neighbors, key=lambda item: float(np.linalg.norm(item)))
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            affinity[0, y, x] = vector[0] / norm
            affinity[1, y, x] = vector[1] / norm
    return affinity


def line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    y0, x0 = start
    y1, x1 = end
    steps = max(abs(y1 - y0), abs(x1 - x0)) + 1
    ys = np.linspace(y0, y1, steps).round().astype(int)
    xs = np.linspace(x0, x1, steps).round().astype(int)
    return list(zip(ys.tolist(), xs.tolist()))


def read_archive_member(archive_path: Path, member_path: str) -> bytes:
    result = subprocess.run(
        ["bsdtar", "-xOf", str(archive_path), member_path],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def read_tiff_from_archive(archive_path: Path, member_path: str) -> np.ndarray:
    data = read_archive_member(archive_path, member_path)
    return tifffile.imread(io.BytesIO(data))


def read_text_from_archive(archive_path: Path, member_path: str) -> str:
    return read_archive_member(archive_path, member_path).decode("utf-8")


def load_split_list(path: Path, limit: int) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit > 0:
        names = names[:limit]
    return names


def build_image_summary(volume: np.ndarray, projection_axis: int, image_size: int) -> np.ndarray:
    volume = volume.astype(np.float32)
    mean_proj = percentile_normalize(volume.mean(axis=projection_axis))
    max_proj = percentile_normalize(volume.max(axis=projection_axis))
    std_proj = percentile_normalize(volume.std(axis=projection_axis))
    argmax_proj = np.argmax(volume, axis=projection_axis).astype(np.float32)
    argmax_proj = percentile_normalize(argmax_proj)
    summary = np.stack([mean_proj, max_proj, std_proj, argmax_proj], axis=0)
    return np.stack([resize_2d(channel, image_size=image_size, order=1) for channel in summary], axis=0)


def parse_swc_projected_skeleton(
    swc_text: str,
    spatial_shape: tuple[int, int],
    projection_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    axis_lookup = {
        0: (1, 0),  # project z, keep y/x
        1: (2, 0),  # project y, keep z/x
        2: (2, 1),  # project x, keep z/y
    }
    keep_axes = axis_lookup[projection_axis]

    nodes: dict[int, tuple[float, float, float]] = {}
    parents: dict[int, int] = {}
    for line in swc_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        node_id = int(float(parts[0]))
        x = float(parts[2])
        y = float(parts[3])
        z = float(parts[4])
        parent = int(float(parts[6]))
        nodes[node_id] = (x, y, z)
        parents[node_id] = parent

    height, width = spatial_shape
    skeleton = np.zeros((height, width), dtype=np.float32)
    overlap = np.zeros((height, width), dtype=np.float32)
    for node_id, parent_id in parents.items():
        if parent_id < 0 or node_id not in nodes or parent_id not in nodes:
            continue
        start_xyz = nodes[parent_id]
        end_xyz = nodes[node_id]
        start = np.array(start_xyz)[list(keep_axes)]
        end = np.array(end_xyz)[list(keep_axes)]
        start_yx = (int(round(start[0])), int(round(start[1])))
        end_yx = (int(round(end[0])), int(round(end[1])))
        for y, x in line_points(start_yx, end_yx):
            if 0 <= y < height and 0 <= x < width:
                skeleton[y, x] = 1.0
                overlap[y, x] += 1.0
    return skeleton, overlap


def build_targets(
    mask_volume: np.ndarray,
    swc_text: str,
    projection_axis: int,
    image_size: int,
) -> dict[str, np.ndarray]:
    mask_volume = (mask_volume > 0).astype(np.float32)
    mask_2d = mask_volume.max(axis=projection_axis)
    swc_skeleton, overlap = parse_swc_projected_skeleton(
        swc_text,
        spatial_shape=mask_2d.shape,
        projection_axis=projection_axis,
    )
    if swc_skeleton.sum() == 0:
        swc_skeleton = skeletonize(mask_2d > 0).astype(np.float32)

    mask_2d = resize_2d(mask_2d, image_size=image_size, order=0) > 0.5
    skeleton_2d = resize_2d(swc_skeleton, image_size=image_size, order=0) > 0.5
    overlap_2d = resize_2d(overlap, image_size=image_size, order=1)

    degree = neighbor_degree(skeleton_2d.astype(np.float32))
    junction = ((skeleton_2d > 0) & (degree >= 3)).astype(np.float32)
    endpoint = ((skeleton_2d > 0) & (degree == 1)).astype(np.float32)
    affinity = estimate_affinity(skeleton_2d.astype(np.float32))
    uncertainty = np.maximum((degree >= 4).astype(np.float32), (overlap_2d > 1.5).astype(np.float32))

    return {
        "mask": mask_2d.astype(np.float32)[None, ...],
        "skeleton": skeleton_2d.astype(np.float32)[None, ...],
        "junction": junction.astype(np.float32)[None, ...],
        "endpoint": endpoint.astype(np.float32)[None, ...],
        "affinity": affinity.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32)[None, ...],
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    images_archive = dataset_root / "images.rar"
    mask_archive = dataset_root / "mask.rar"
    swc_archive = dataset_root / "swc.rar"
    if not images_archive.is_file() or not mask_archive.is_file() or not swc_archive.is_file():
        raise FileNotFoundError("Expected images.rar, mask.rar, and swc.rar under dataset-root.")

    manifests: list[dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        split_file = dataset_root / f"{split_name}.txt"
        if not split_file.is_file():
            continue
        split_root = output_root / split_name
        split_root.mkdir(parents=True, exist_ok=True)
        sample_names = load_split_list(split_file, limit=args.limit)
        for sample_name in tqdm(sample_names, desc=f"prepare-axon-{split_name}"):
            sample_key = sample_name.strip()
            image_volume = read_tiff_from_archive(images_archive, f"images/{sample_key}")
            mask_volume = read_tiff_from_archive(mask_archive, f"mask/{sample_key}")
            swc_text = read_text_from_archive(swc_archive, f"swc/{sample_key[:-4]}.swc")

            image = build_image_summary(
                image_volume,
                projection_axis=args.projection_axis,
                image_size=args.image_size,
            )
            targets = build_targets(
                mask_volume,
                swc_text,
                projection_axis=args.projection_axis,
                image_size=args.image_size,
            )
            output_path = split_root / f"{sample_key}.npz"
            np.savez_compressed(
                output_path,
                image=image.astype(np.float32),
                mask=targets["mask"],
                skeleton=targets["skeleton"],
                junction=targets["junction"],
                endpoint=targets["endpoint"],
                affinity=targets["affinity"],
                uncertainty=targets["uncertainty"],
            )
            manifests.append(
                {
                    "split": split_name,
                    "sample": sample_key,
                    "image_shape": list(map(int, image_volume.shape)),
                    "mask_shape": list(map(int, mask_volume.shape)),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifests, handle, indent=2)


if __name__ == "__main__":
    main()
