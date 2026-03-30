from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Neurofinder calcium-imaging datasets into SparseGraph .npz samples."
    )
    parser.add_argument("--dataset-root", required=True, help="Folder containing Neurofinder dataset folders.")
    parser.add_argument("--output-root", required=True, help="Destination folder for train/val .npz samples.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


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


def build_summary(movie: np.ndarray) -> np.ndarray:
    movie = movie.astype(np.float32)
    return np.stack(
        [
            percentile_normalize(movie.mean(axis=0)),
            percentile_normalize(movie.max(axis=0)),
            percentile_normalize(movie.std(axis=0)),
            percentile_normalize(local_correlation_map(movie)),
        ],
        axis=0,
    ).astype(np.float32)


def resize_image(array: np.ndarray, image_size: int, order: int) -> np.ndarray:
    from skimage.transform import resize

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


def rasterize_regions(regions: list[dict[str, object]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    height, width = shape
    for region in regions:
        coordinates = region.get("coordinates", [])
        for coord in coordinates:
            if len(coord) < 2:
                continue
            y = int(coord[0])
            x = int(coord[1])
            if 0 <= y < height and 0 <= x < width:
                mask[y, x] = 1.0
    return mask


def build_targets(mask: np.ndarray, image_size: int) -> dict[str, np.ndarray]:
    from skimage.morphology import skeletonize

    mask = resize_image(mask, image_size=image_size, order=0)
    mask = (mask > 0.5).astype(np.float32)
    skeleton = skeletonize(mask > 0).astype(np.float32)
    degree = neighbor_degree(skeleton)
    junction = ((skeleton > 0) & (degree >= 3)).astype(np.float32)
    endpoint = ((skeleton > 0) & (degree == 1)).astype(np.float32)
    affinity = estimate_affinity(skeleton)
    uncertainty = ((degree >= 4).astype(np.float32))[None, ...]
    return {
        "mask": mask[None, ...].astype(np.float32),
        "skeleton": skeleton[None, ...].astype(np.float32),
        "junction": junction[None, ...].astype(np.float32),
        "endpoint": endpoint[None, ...].astype(np.float32),
        "affinity": affinity.astype(np.float32),
        "uncertainty": uncertainty.astype(np.float32),
    }


def discover_datasets(root: Path) -> list[Path]:
    datasets: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_dir():
            continue
        if (candidate / "images").is_dir() and (candidate / "regions" / "regions.json").is_file():
            datasets.append(candidate)
    return datasets


def load_movie(dataset_dir: Path) -> np.ndarray:
    import tifffile

    frame_paths = sorted((dataset_dir / "images").glob("*.tif*"))
    if not frame_paths:
        raise FileNotFoundError(f"No TIFF frames found in {dataset_dir / 'images'}")
    frames = [tifffile.imread(path).astype(np.float32) for path in frame_paths]
    return np.stack(frames, axis=0)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    train_root = output_root / "train"
    val_root = output_root / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    dataset_dirs = discover_datasets(dataset_root)
    if not dataset_dirs:
        raise FileNotFoundError(f"No Neurofinder training datasets found under {dataset_root}")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(dataset_dirs))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(dataset_dirs) * args.val_ratio)))
    val_indices = set(indices[:val_count].tolist())

    manifest: list[dict[str, object]] = []
    for index, dataset_dir in enumerate(tqdm(dataset_dirs, desc="prepare-neurofinder")):
        movie = load_movie(dataset_dir)
        summary = build_summary(movie)
        summary = np.stack([resize_image(channel, args.image_size, order=1) for channel in summary], axis=0)

        with (dataset_dir / "regions" / "regions.json").open("r", encoding="utf-8") as handle:
            regions = json.load(handle)
        mask = rasterize_regions(regions, shape=movie.shape[-2:])
        targets = build_targets(mask, image_size=args.image_size)

        split_root = val_root if index in val_indices else train_root
        output_name = dataset_dir.name.replace(".", "_") + ".npz"
        np.savez_compressed(
            split_root / output_name,
            image=summary.astype(np.float32),
            mask=targets["mask"],
            skeleton=targets["skeleton"],
            junction=targets["junction"],
            endpoint=targets["endpoint"],
            affinity=targets["affinity"],
            uncertainty=targets["uncertainty"],
        )
        manifest.append(
            {
                "dataset": dataset_dir.name,
                "split": "val" if index in val_indices else "train",
                "frames": int(movie.shape[0]),
                "height": int(movie.shape[1]),
                "width": int(movie.shape[2]),
                "regions": int(len(regions)),
            }
        )

    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
