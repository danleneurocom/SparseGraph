from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def _ensure_channel_first(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, ...]
    return array


def _line_points(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    y0, x0 = start
    y1, x1 = end
    steps = max(abs(y1 - y0), abs(x1 - x0)) + 1
    ys = np.linspace(y0, y1, steps).round().astype(int)
    xs = np.linspace(x0, x1, steps).round().astype(int)
    return list(zip(ys.tolist(), xs.tolist()))


class CalciumSummaryNpzDataset(Dataset):
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.files[index]
        with np.load(path) as sample:
            image = _ensure_channel_first(sample["image"]).astype(np.float32)
            mask = _ensure_channel_first(sample["mask"]).astype(np.float32)
            skeleton = _ensure_channel_first(sample["skeleton"]).astype(np.float32)
            junction = _ensure_channel_first(sample["junction"]).astype(np.float32)
            endpoint = _ensure_channel_first(sample["endpoint"]).astype(np.float32)

            if "affinity" in sample:
                affinity = _ensure_channel_first(sample["affinity"]).astype(np.float32)
            else:
                affinity = np.zeros((2, image.shape[-2], image.shape[-1]), dtype=np.float32)

            if "uncertainty" in sample:
                uncertainty = _ensure_channel_first(sample["uncertainty"]).astype(np.float32)
            else:
                uncertainty = np.zeros((1, image.shape[-2], image.shape[-1]), dtype=np.float32)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "skeleton": torch.from_numpy(skeleton),
            "junction": torch.from_numpy(junction),
            "endpoint": torch.from_numpy(endpoint),
            "affinity": torch.from_numpy(affinity),
            "uncertainty": torch.from_numpy(uncertainty),
        }


class SyntheticCalciumDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        image_size: int = 256,
        in_channels: int = 4,
        seed: int = 7,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.in_channels = in_channels
        rng = np.random.default_rng(seed)
        self.sample_seeds = rng.integers(0, 1_000_000_000, size=num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(int(self.sample_seeds[index]))
        skeleton = self._sample_tree(rng)
        degree = self._neighbor_degree(skeleton)
        junction = ((skeleton > 0) & (degree >= 3)).astype(np.float32)
        endpoint = ((skeleton > 0) & (degree == 1)).astype(np.float32)
        uncertainty = ((degree >= 4).astype(np.float32))[None, ...]
        affinity = self._estimate_affinity(skeleton)

        skeleton_tensor = torch.from_numpy(skeleton[None, ...].astype(np.float32))
        kernel_size = int(rng.integers(5, 10))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = F.max_pool2d(
            skeleton_tensor.unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ).squeeze(0)
        mask = (mask > 0).float()
        image = self._render_input(
            mask.squeeze(0).numpy(),
            skeleton,
            junction,
            rng,
        )

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": mask,
            "skeleton": skeleton_tensor,
            "junction": torch.from_numpy(junction[None, ...].astype(np.float32)),
            "endpoint": torch.from_numpy(endpoint[None, ...].astype(np.float32)),
            "affinity": torch.from_numpy(affinity.astype(np.float32)),
            "uncertainty": torch.from_numpy(uncertainty.astype(np.float32)),
        }

    def _sample_tree(self, rng: np.random.Generator) -> np.ndarray:
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        margin = max(self.image_size // 12, 8)
        center = np.array(
            [
                rng.integers(margin * 2, self.image_size - margin * 2),
                rng.integers(margin * 2, self.image_size - margin * 2),
            ]
        )

        num_primary = int(rng.integers(2, 4))
        branches: list[tuple[np.ndarray, float, int, int]] = []
        for _ in range(num_primary):
            angle = float(rng.uniform(0, 2 * np.pi))
            length = int(rng.integers(self.image_size // 8, self.image_size // 5))
            branches.append((center.copy(), angle, length, 0))

        max_depth = 3
        while branches:
            start, angle, length, depth = branches.pop()
            end = np.array(
                [
                    int(start[0] + np.sin(angle) * length),
                    int(start[1] + np.cos(angle) * length),
                ]
            )
            end = np.clip(end, margin, self.image_size - margin - 1)
            for y, x in _line_points((int(start[0]), int(start[1])), (int(end[0]), int(end[1]))):
                image[y, x] = 1.0

            if depth < max_depth:
                num_children = int(rng.integers(1, 3))
                for _ in range(num_children):
                    next_angle = angle + float(rng.uniform(-1.05, 1.05))
                    next_length = max(10, int(length * rng.uniform(0.5, 0.8)))
                    branches.append((end.copy(), next_angle, next_length, depth + 1))

        if rng.random() < 0.3:
            gap_count = int(rng.integers(1, 4))
            ys, xs = np.where(image > 0)
            if len(ys) > gap_count:
                gap_indices = rng.choice(len(ys), size=gap_count, replace=False)
                image[ys[gap_indices], xs[gap_indices]] = 0.0

        return image

    def _neighbor_degree(self, skeleton: np.ndarray) -> np.ndarray:
        padded = np.pad(skeleton, 1)
        degree = np.zeros_like(skeleton, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                degree += padded[1 + dy : 1 + dy + self.image_size, 1 + dx : 1 + dx + self.image_size]
        return degree * skeleton

    def _estimate_affinity(self, skeleton: np.ndarray) -> np.ndarray:
        affinity = np.zeros((2, self.image_size, self.image_size), dtype=np.float32)
        ys, xs = np.where(skeleton > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            neighbors: list[np.ndarray] = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < self.image_size and 0 <= nx < self.image_size and skeleton[ny, nx] > 0:
                        neighbors.append(np.array([dx, dy], dtype=np.float32))
            if not neighbors:
                continue
            if len(neighbors) == 1:
                vector = neighbors[0]
            else:
                max_distance = -1.0
                vector = neighbors[0]
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        candidate = neighbors[j] - neighbors[i]
                        distance = float(np.linalg.norm(candidate))
                        if distance > max_distance:
                            max_distance = distance
                            vector = candidate
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                affinity[0, y, x] = vector[0] / norm
                affinity[1, y, x] = vector[1] / norm
        return affinity

    def _render_input(
        self,
        mask: np.ndarray,
        skeleton: np.ndarray,
        junction: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mask_tensor = torch.from_numpy(mask[None, None, ...])
        skeleton_tensor = torch.from_numpy(skeleton[None, None, ...])
        junction_tensor = torch.from_numpy(junction[None, None, ...])

        thick = F.max_pool2d(mask_tensor, kernel_size=5, stride=1, padding=2).squeeze().numpy()
        soft = F.avg_pool2d(mask_tensor, kernel_size=9, stride=1, padding=4).squeeze().numpy()
        branch = F.max_pool2d(junction_tensor, kernel_size=7, stride=1, padding=3).squeeze().numpy()
        center = F.avg_pool2d(skeleton_tensor, kernel_size=5, stride=1, padding=2).squeeze().numpy()

        noise = rng.normal(0.0, 0.05, size=(self.in_channels, self.image_size, self.image_size)).astype(
            np.float32
        )
        image = np.zeros((self.in_channels, self.image_size, self.image_size), dtype=np.float32)
        image[0] = np.clip(0.75 * mask + 0.25 * soft + noise[0] + 0.05, 0.0, 1.0)
        image[1] = np.clip(0.55 * thick + 0.35 * center + noise[1], 0.0, 1.0)
        image[2] = np.clip(0.45 * soft + 0.4 * branch + np.abs(noise[2]) * 0.5, 0.0, 1.0)
        image[3] = np.clip(0.35 * mask + 0.15 * center + noise[3] + rng.random() * 0.05, 0.0, 1.0)
        return image
