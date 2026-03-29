from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import zoom


@dataclass
class SampleSpec:
    subject: str
    session: str
    image_path: Path
    with_parent_path: Path
    aneurysm_only_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenNeuro ds005096 into SparseGraph .npz training samples."
    )
    parser.add_argument("--dataset-root", required=True, help="Local ds005096 root with actual NIfTI payloads.")
    parser.add_argument("--output-root", required=True, help="Output folder for train/val .npz samples.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--projection-axis",
        default="auto",
        choices=["auto", "0", "1", "2"],
        help="3D axis to collapse into 2D. 'auto' selects the smallest spatial dimension.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of subjects reserved for validation.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--include-aneurysm",
        action="store_true",
        help="Keep aneurysm voxels in the temporary target mask instead of subtracting them.",
    )
    return parser.parse_args()


def load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "nibabel is required to read NIfTI files. Install dependencies with `pip install -r requirements.txt`."
        ) from exc
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def choose_projection_axis(shape: tuple[int, ...], axis_arg: str) -> int:
    if axis_arg == "auto":
        return int(np.argmin(shape))
    return int(axis_arg)


def normalize_map(array: np.ndarray) -> np.ndarray:
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
    resized = zoom(array, zoom_factors, order=order)
    resized = resized.astype(np.float32)
    if resized.shape == (image_size, image_size):
        return resized

    output = np.zeros((image_size, image_size), dtype=np.float32)
    height = min(image_size, resized.shape[0])
    width = min(image_size, resized.shape[1])
    output[:height, :width] = resized[:height, :width]
    return output


def zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    image = (binary > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        to_remove: list[tuple[int, int]] = []
        rows, cols = image.shape
        for step in (0, 1):
            to_remove.clear()
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
                        if p2 * p4 * p6 != 0:
                            continue
                        if p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0:
                            continue
                        if p2 * p6 * p8 != 0:
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
    volume = np.asarray(volume, dtype=np.float32)
    max_proj = normalize_map(np.max(volume, axis=axis))
    mean_proj = normalize_map(np.mean(volume, axis=axis))
    std_proj = normalize_map(np.std(volume, axis=axis))
    p95_proj = normalize_map(np.percentile(volume, 95, axis=axis))
    channels = [
        resize_2d(max_proj, image_size, "bilinear"),
        resize_2d(mean_proj, image_size, "bilinear"),
        resize_2d(std_proj, image_size, "bilinear"),
        resize_2d(p95_proj, image_size, "bilinear"),
    ]
    return np.stack(channels, axis=0).astype(np.float32)


def build_targets(
    vessel_mask_3d: np.ndarray,
    axis: int,
    image_size: int,
) -> dict[str, np.ndarray]:
    mask_2d = (np.max(vessel_mask_3d, axis=axis) > 0).astype(np.float32)
    mask_2d = resize_2d(mask_2d, image_size, "nearest")
    mask_2d = (mask_2d > 0.5).astype(np.float32)

    skeleton = zhang_suen_thinning(mask_2d)
    degree = neighbor_degree(skeleton)
    junction = ((skeleton > 0) & (degree >= 3)).astype(np.float32)
    endpoint = ((skeleton > 0) & (degree == 1)).astype(np.float32)
    affinity = estimate_affinity(skeleton)
    uncertainty = np.zeros((1, image_size, image_size), dtype=np.float32)

    return {
        "mask": mask_2d[None, ...].astype(np.float32),
        "skeleton": skeleton[None, ...].astype(np.float32),
        "junction": junction[None, ...].astype(np.float32),
        "endpoint": endpoint[None, ...].astype(np.float32),
        "affinity": affinity.astype(np.float32),
        "uncertainty": uncertainty,
    }


def pick_best_path(paths: Iterable[Path], prefer: tuple[str, ...] = (), avoid: tuple[str, ...] = ()) -> Path | None:
    candidates = [path for path in paths if path.suffix in {".nii", ".gz"} or path.name.endswith(".nii.gz")]
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        bad = sum(1 for token in avoid if token.lower() in name)
        good = sum(1 for token in prefer if token.lower() in name)
        return (bad, -good, name)

    candidates.sort(key=score)
    return candidates[0]


def file_payload_exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_symlink():
        return path.resolve().exists()
    return True


def collect_samples(dataset_root: Path) -> list[SampleSpec]:
    samples: list[SampleSpec] = []
    derivatives_root = dataset_root / "derivatives"
    for derivative_subject in sorted(derivatives_root.glob("sub-*")):
        if not derivative_subject.is_dir():
            continue
        subject = derivative_subject.name
        original_subject = dataset_root / subject
        if not original_subject.exists():
            continue

        for session_dir in sorted(derivative_subject.glob("ses-*")):
            session = session_dir.name
            original_anat = original_subject / session / "anat"
            if not original_anat.exists():
                continue

            angio_path = pick_best_path(original_anat.glob("*angio.nii*"))
            if angio_path is None:
                continue

            derivative_leaf = None
            for candidate in ("anat", "ct", "ds"):
                path = session_dir / candidate
                if path.exists():
                    derivative_leaf = path
                    break
            if derivative_leaf is None:
                continue

            with_parent_dir = derivative_leaf / "Nifti with Parent Artery"
            aneurysm_dir = derivative_leaf / "Nifti Aneurysm Only"
            if not with_parent_dir.exists():
                continue

            with_parent_path = pick_best_path(
                with_parent_dir.glob("*.nii*"),
                prefer=("parent", "artery", "vessel"),
                avoid=("duplicate",),
            )
            aneurysm_only_path = None
            if aneurysm_dir.exists():
                aneurysm_only_path = pick_best_path(
                    aneurysm_dir.glob("*.nii*"),
                    prefer=("aneurysm",),
                    avoid=("duplicate",),
                )

            if with_parent_path is None:
                continue
            if not file_payload_exists(angio_path) or not file_payload_exists(with_parent_path):
                continue
            if aneurysm_only_path is not None and not file_payload_exists(aneurysm_only_path):
                aneurysm_only_path = None

            samples.append(
                SampleSpec(
                    subject=subject,
                    session=session,
                    image_path=angio_path,
                    with_parent_path=with_parent_path,
                    aneurysm_only_path=aneurysm_only_path,
                )
            )
    return samples


def stable_subject_split(subjects: list[str], val_ratio: float) -> tuple[set[str], set[str]]:
    unique_subjects = sorted(set(subjects))
    if not unique_subjects:
        return set(), set()
    val_count = max(1, int(round(len(unique_subjects) * val_ratio))) if len(unique_subjects) > 1 else 0
    val_subjects = set(unique_subjects[-val_count:]) if val_count > 0 else set()
    train_subjects = set(unique_subjects) - val_subjects
    if not train_subjects and val_subjects:
        train_subjects.add(sorted(val_subjects)[0])
        val_subjects.remove(sorted(val_subjects)[0])
    return train_subjects, val_subjects


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "train").mkdir(exist_ok=True)
    (output_root / "val").mkdir(exist_ok=True)

    samples = collect_samples(dataset_root)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    if not samples:
        raise SystemExit(
            "No usable samples were found. Make sure the real NIfTI payloads are present locally, not just annex symlinks."
        )

    train_subjects, val_subjects = stable_subject_split(
        [sample.subject for sample in samples],
        args.val_ratio,
    )

    manifest: list[dict[str, str]] = []
    for sample in samples:
        image = load_nifti(sample.image_path)
        with_parent = load_nifti(sample.with_parent_path) > 0
        aneurysm_only = None
        if sample.aneurysm_only_path is not None:
            aneurysm_only = load_nifti(sample.aneurysm_only_path) > 0

        if args.include_aneurysm or aneurysm_only is None:
            vessel_mask = with_parent.astype(np.float32)
        else:
            vessel_mask = np.logical_and(with_parent, np.logical_not(aneurysm_only)).astype(np.float32)
            if vessel_mask.sum() == 0:
                vessel_mask = with_parent.astype(np.float32)

        axis = choose_projection_axis(image.shape[:3], args.projection_axis)
        image_channels = build_image_channels(image, axis=axis, image_size=args.image_size)
        targets = build_targets(vessel_mask, axis=axis, image_size=args.image_size)

        split = "val" if sample.subject in val_subjects else "train"
        output_name = f"{sample.subject}_{sample.session}.npz"
        output_path = output_root / split / output_name
        np.savez_compressed(
            output_path,
            image=image_channels,
            mask=targets["mask"],
            skeleton=targets["skeleton"],
            junction=targets["junction"],
            endpoint=targets["endpoint"],
            affinity=targets["affinity"],
            uncertainty=targets["uncertainty"],
        )
        manifest.append(
            {
                "subject": sample.subject,
                "session": sample.session,
                "split": split,
                "image": str(sample.image_path),
                "with_parent": str(sample.with_parent_path),
                "aneurysm_only": str(sample.aneurysm_only_path) if sample.aneurysm_only_path else "",
                "output": str(output_path),
            }
        )

    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Prepared {len(manifest)} samples into {output_root}")
    print(f"Train subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)}")


if __name__ == "__main__":
    main()
