from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects, skeletonize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert IXI MRA volumes into SparseGraph .npz samples with pseudo labels."
    )
    parser.add_argument("--dataset-root", required=True, help="Folder containing IXI *.nii.gz MRA files.")
    parser.add_argument("--output-root", required=True, help="Destination folder for train/val .npz samples.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--projection-axis",
        default="auto",
        choices=["auto", "0", "1", "2"],
        help="3D axis to collapse. 'auto' selects the smallest dimension.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-object-size", type=int, default=64)
    parser.add_argument("--min-hole-size", type=int, default=64)
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=0.9,
        help="Multiplier applied to the Otsu threshold on the vesselness map.",
    )
    return parser.parse_args()


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def resize_2d(array: np.ndarray, image_size: int, mode: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    zoom_factors = (image_size / array.shape[0], image_size / array.shape[1])
    order = 0 if mode == "nearest" else 1
    resized = zoom(array, zoom_factors, order=order).astype(np.float32)
    if resized.shape == (image_size, image_size):
        return resized
    output = np.zeros((image_size, image_size), dtype=np.float32)
    height = min(image_size, resized.shape[0])
    width = min(image_size, resized.shape[1])
    output[:height, :width] = resized[:height, :width]
    return output


def choose_projection_axis(shape: tuple[int, ...], axis_arg: str) -> int:
    if axis_arg == "auto":
        return int(np.argmin(shape))
    return int(axis_arg)


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    image = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        rows, cols = image.shape
        for step in (0, 1):
            to_remove: list[tuple[int, int]] = []
            for y in range(1, rows - 1):
                for x in range(1, cols - 1):
                    if image[y, x] != 1:
                        continue
                    p2 = image[y - 1, x]
                    p3 = image[y - 1, x + 1]
                    p4 = image[y, x + 1]
                    p5 = image[y + 1, x + 1]
                    p6 = image[y + 1, x]
                    p7 = image[y + 1, x - 1]
                    p8 = image[y, x - 1]
                    p9 = image[y - 1, x - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    count = int(sum(neighbors))
                    if count < 2 or count > 6:
                        continue
                    transitions = 0
                    for i in range(8):
                        if neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1:
                            transitions += 1
                    if transitions != 1:
                        continue
                    if step == 0:
                        if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                            continue
                    to_remove.append((y, x))
            if to_remove:
                changed = True
                for y, x in to_remove:
                    image[y, x] = 0
    return image.astype(np.float32)


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
            best_distance = -1.0
            vector = neighbors[0]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    candidate = neighbors[j] - neighbors[i]
                    distance = float(np.linalg.norm(candidate))
                    if distance > best_distance:
                        best_distance = distance
                        vector = candidate
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            affinity[0, y, x] = vector[0] / norm
            affinity[1, y, x] = vector[1] / norm
    return affinity


def build_image_channels(volume: np.ndarray, axis: int, image_size: int) -> np.ndarray:
    max_proj = normalize(np.max(volume, axis=axis))
    mean_proj = normalize(np.mean(volume, axis=axis))
    std_proj = normalize(np.std(volume, axis=axis))
    p95_proj = normalize(np.percentile(volume, 95, axis=axis))
    return np.stack(
        [
            resize_2d(max_proj, image_size, "bilinear"),
            resize_2d(mean_proj, image_size, "bilinear"),
            resize_2d(std_proj, image_size, "bilinear"),
            resize_2d(p95_proj, image_size, "bilinear"),
        ],
        axis=0,
    ).astype(np.float32)


def build_pseudo_targets(
    volume: np.ndarray,
    axis: int,
    image_size: int,
    min_object_size: int,
    min_hole_size: int,
    threshold_scale: float,
) -> dict[str, np.ndarray]:
    projection = normalize(np.max(volume, axis=axis))
    vesselness = frangi(projection)
    vesselness = normalize(vesselness)
    threshold = threshold_otsu(vesselness) * threshold_scale
    mask = vesselness > threshold
    mask = remove_small_objects(mask, min_size=min_object_size)
    mask = binary_closing(mask, footprint=disk(1))
    mask = remove_small_holes(mask, area_threshold=min_hole_size)
    mask = resize_2d(mask.astype(np.float32), image_size, "nearest") > 0.5

    skeleton = skeletonize(mask)
    if skeleton.sum() == 0:
        skeleton = zhang_suen_thinning(mask.astype(np.float32)) > 0
    degree = neighbor_degree(skeleton.astype(np.float32))
    junction = ((skeleton > 0) & (degree >= 3)).astype(np.float32)
    endpoint = ((skeleton > 0) & (degree == 1)).astype(np.float32)
    affinity = estimate_affinity(skeleton.astype(np.float32))
    uncertainty = np.zeros((1, image_size, image_size), dtype=np.float32)

    return {
        "mask": mask.astype(np.float32)[None, ...],
        "skeleton": skeleton.astype(np.float32)[None, ...],
        "junction": junction[None, ...],
        "endpoint": endpoint[None, ...],
        "affinity": affinity.astype(np.float32),
        "uncertainty": uncertainty,
    }


def stable_split(files: list[Path], val_ratio: float) -> tuple[list[Path], list[Path]]:
    files = sorted(files)
    if not files:
        return [], []
    val_count = max(1, int(round(len(files) * val_ratio))) if len(files) > 1 else 0
    val_files = files[-val_count:] if val_count > 0 else []
    train_files = files[:-val_count] if val_count > 0 else files
    if not train_files and val_files:
        train_files = [val_files[0]]
        val_files = val_files[1:]
    return train_files, val_files


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "train").mkdir(exist_ok=True)
    (output_root / "val").mkdir(exist_ok=True)

    files = sorted(dataset_root.glob("*.nii.gz"))
    if args.max_samples is not None:
        files = files[: args.max_samples]
    if not files:
        raise SystemExit("No IXI MRA .nii.gz files were found.")

    train_files, val_files = stable_split(files, args.val_ratio)
    manifest = []
    skipped = []

    for split, split_files in (("train", train_files), ("val", val_files)):
        for path in split_files:
            try:
                volume = nib.load(str(path)).get_fdata(dtype=np.float32)
            except Exception as exc:
                skipped.append(
                    {
                        "split": split,
                        "source": str(path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            axis = choose_projection_axis(volume.shape[:3], args.projection_axis)
            image = build_image_channels(volume, axis=axis, image_size=args.image_size)
            targets = build_pseudo_targets(
                volume,
                axis=axis,
                image_size=args.image_size,
                min_object_size=args.min_object_size,
                min_hole_size=args.min_hole_size,
                threshold_scale=args.threshold_scale,
            )
            output_dir = output_root / split
            output_dir.mkdir(parents=True, exist_ok=True)
            output_name = path.name.replace(".nii.gz", "") + ".npz"
            output_path = output_dir / output_name
            np.savez_compressed(
                output_path,
                image=image,
                mask=targets["mask"],
                skeleton=targets["skeleton"],
                junction=targets["junction"],
                endpoint=targets["endpoint"],
                affinity=targets["affinity"],
                uncertainty=targets["uncertainty"],
            )
            manifest.append(
                {
                    "split": split,
                    "source": str(path),
                    "output": str(output_path),
                    "projection_axis": axis,
                }
            )

    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with (output_root / "skipped.json").open("w", encoding="utf-8") as handle:
        json.dump(skipped, handle, indent=2)

    print(f"Prepared {len(manifest)} IXI pseudo-labeled samples into {output_root}")
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")
    if skipped:
        print(f"Skipped {len(skipped)} corrupt or unreadable files")


if __name__ == "__main__":
    main()
