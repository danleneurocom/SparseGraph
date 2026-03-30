from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import zarr
from skimage.morphology import skeletonize
from skimage.transform import resize
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FISBe zarr volumes into SparseGraph .npz samples."
    )
    parser.add_argument(
        "--zip-path",
        default=None,
        help="Path to the FISBe zip archive. Use this if you want direct zipped access.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Path to an extracted FISBe root containing train/val/test split folders.",
    )
    parser.add_argument("--output-root", required=True, help="Destination folder for train/val/test .npz samples.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split sample limit for smoke runs.")
    parser.add_argument("--z-chunk", type=int, default=32, help="Chunk size for z-axis streaming.")
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


def discover_samples_from_zip(zip_path: Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith("/volumes/raw/.zarray"):
                continue
            path = PurePosixPath(name)
            if len(path.parts) < 5:
                continue
            split_name = path.parts[1]
            sample_name = path.parts[2]
            if split_name in splits:
                splits[split_name].append(sample_name)
    return {key: sorted(set(value)) for key, value in splits.items() if value}


def discover_samples_from_directory(dataset_root: Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    for split_dir in dataset_root.iterdir():
        if not split_dir.is_dir():
            continue
        sample_names = sorted(path.name for path in split_dir.glob("*.zarr"))
        if sample_names:
            splits[split_dir.name] = sample_names
    return splits


def open_sample_group(
    split_name: str,
    sample_name: str,
    dataset_root: Path | None,
    store: zarr.storage.ZipStore | None,
):
    if store is not None:
        return zarr.open_group(store=store, path=f"completely/{split_name}/{sample_name}", mode="r")
    if dataset_root is None:
        raise ValueError("Either dataset_root or store must be provided.")
    return zarr.open_group(store=str(dataset_root / split_name / sample_name), mode="r")


def build_image_summary(raw: zarr.Array, image_size: int, z_chunk: int) -> np.ndarray:
    channels = min(int(raw.shape[0]), 3)
    depth = int(raw.shape[1])
    height = int(raw.shape[2])
    width = int(raw.shape[3])
    max_projections = np.zeros((channels, height, width), dtype=np.float32)
    for start in range(0, depth, z_chunk):
        stop = min(start + z_chunk, depth)
        chunk = np.asarray(raw[:channels, start:stop, :, :], dtype=np.float32)
        max_projections = np.maximum(max_projections, chunk.max(axis=1))

    if channels == 1:
        max_projections = np.repeat(max_projections, 3, axis=0)
    elif channels == 2:
        max_projections = np.concatenate([max_projections, max_projections[-1:]], axis=0)
    summary = np.stack(
        [
            percentile_normalize(max_projections[0]),
            percentile_normalize(max_projections[1]),
            percentile_normalize(max_projections[2]),
            percentile_normalize(max_projections.mean(axis=0)),
        ],
        axis=0,
    )
    return np.stack([resize_2d(channel, image_size=image_size, order=1) for channel in summary], axis=0)


def build_targets(instances: zarr.Array, image_size: int, z_chunk: int) -> dict[str, np.ndarray]:
    neuron_count = int(instances.shape[0])
    depth = int(instances.shape[1])
    height = int(instances.shape[2])
    width = int(instances.shape[3])

    union_projection = np.zeros((height, width), dtype=bool)
    overlap_count = np.zeros((height, width), dtype=np.float32)
    skeleton_projection = np.zeros((height, width), dtype=bool)

    for neuron_index in range(neuron_count):
        neuron_projection = np.zeros((height, width), dtype=bool)
        for start in range(0, depth, z_chunk):
            stop = min(start + z_chunk, depth)
            chunk = np.asarray(instances[neuron_index, start:stop, :, :], dtype=np.uint8)
            neuron_projection |= chunk.max(axis=0) > 0
        if neuron_projection.any():
            union_projection |= neuron_projection
            overlap_count += neuron_projection.astype(np.float32)
            skeleton_projection |= skeletonize(neuron_projection)

    if not skeleton_projection.any():
        skeleton_projection = skeletonize(union_projection)

    mask_2d = resize_2d(union_projection.astype(np.float32), image_size=image_size, order=0) > 0.5
    skeleton_2d = resize_2d(skeleton_projection.astype(np.float32), image_size=image_size, order=0) > 0.5
    overlap_2d = resize_2d(overlap_count, image_size=image_size, order=1)

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
    zip_path = Path(args.zip_path) if args.zip_path else None
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    if zip_path is None and dataset_root is None:
        raise ValueError("Pass either --zip-path or --dataset-root.")
    if zip_path is not None and not zip_path.is_file():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")
    if dataset_root is not None and not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    store = zarr.storage.ZipStore(zip_path, mode="r") if zip_path is not None else None
    splits = discover_samples_from_zip(zip_path) if zip_path is not None else discover_samples_from_directory(dataset_root)
    manifests: list[dict[str, object]] = []
    try:
        for split_name, sample_names in splits.items():
            if args.limit > 0:
                sample_names = sample_names[: args.limit]
            split_root = Path(args.output_root) / split_name
            split_root.mkdir(parents=True, exist_ok=True)
            for sample_name in tqdm(sample_names, desc=f"prepare-fisbe-{split_name}"):
                group = open_sample_group(split_name, sample_name, dataset_root=dataset_root, store=store)
                raw = group["volumes"]["raw"]
                instances = group["volumes"]["gt_instances"]
                image = build_image_summary(raw, image_size=args.image_size, z_chunk=args.z_chunk)
                targets = build_targets(instances, image_size=args.image_size, z_chunk=args.z_chunk)
                output_path = split_root / f"{sample_name}.npz"
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
                        "sample": sample_name,
                        "raw_shape": list(map(int, raw.shape)),
                        "instance_shape": list(map(int, instances.shape)),
                    }
                )
    finally:
        if store is not None:
            store.close()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifests, handle, indent=2)


if __name__ == "__main__":
    main()
