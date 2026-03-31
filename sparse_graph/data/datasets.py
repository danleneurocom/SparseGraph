from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
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


_NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _sample_path_points(path: list[tuple[int, int]], path_length: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.zeros((path_length, 2), dtype=np.int64)
    mask = np.zeros((path_length,), dtype=np.float32)
    if not path:
        return coords, mask
    if len(path) == 1:
        coords[0] = np.asarray(path[0], dtype=np.int64)
        mask[0] = 1.0
        return coords, mask
    sample_indices = np.linspace(0, len(path) - 1, path_length).round().astype(int)
    sampled = np.asarray([path[index] for index in sample_indices], dtype=np.int64)
    valid = min(len(sampled), path_length)
    coords[:valid] = sampled[:valid]
    mask[:valid] = 1.0
    return coords, mask


def _maximum_spanning_edge_subset(
    num_nodes: int,
    weighted_edges: list[tuple[int, int, float]],
) -> set[tuple[int, int]]:
    parent = list(range(max(num_nodes, 0)))
    rank = [0 for _ in range(max(num_nodes, 0))]

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return False
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1
        return True

    selected: set[tuple[int, int]] = set()
    for source, target, weight in sorted(weighted_edges, key=lambda item: item[2], reverse=True):
        if union(int(source), int(target)):
            selected.add(tuple(sorted((int(source), int(target)))))
    return selected


def _graph_nodes_from_targets(
    skeleton: np.ndarray,
    junction: np.ndarray,
    endpoint: np.ndarray,
) -> list[dict[str, Any]]:
    skeleton_binary = skeleton > 0.5
    node_mask = ((junction > 0.5) | (endpoint > 0.5)) & skeleton_binary
    if not np.any(node_mask):
        return []

    structure = np.ones((3, 3), dtype=np.int8)
    labels, num_labels = ndimage.label(node_mask.astype(np.uint8), structure=structure)
    nodes: list[dict[str, Any]] = []
    for label_index in range(1, int(num_labels) + 1):
        coords = np.argwhere(labels == label_index)
        if coords.size == 0:
            continue
        junction_votes = float(junction[labels == label_index].sum())
        endpoint_votes = float(endpoint[labels == label_index].sum())
        kind = "junction" if junction_votes >= endpoint_votes else "endpoint"
        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords - centroid[None, :], axis=1)
        point = coords[int(np.argmin(distances))]
        nodes.append(
            {
                "index": len(nodes),
                "kind": kind,
                "point": (int(point[0]), int(point[1])),
                "pixels": {tuple(map(int, coord)) for coord in coords.tolist()},
            }
        )
    return nodes


def _skeleton_neighbors(skeleton: np.ndarray, y: int, x: int) -> list[tuple[int, int]]:
    height, width = skeleton.shape
    neighbors: list[tuple[int, int]] = []
    for dy, dx in _NEIGHBOR_OFFSETS:
        ny = y + dy
        nx = x + dx
        if 0 <= ny < height and 0 <= nx < width and skeleton[ny, nx] > 0:
            neighbors.append((ny, nx))
    return neighbors


def _trace_graph_edges(
    skeleton: np.ndarray,
    nodes: list[dict[str, Any]],
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    if not nodes:
        return []

    node_lookup = -np.ones(skeleton.shape, dtype=np.int32)
    for node in nodes:
        for y, x in node["pixels"]:
            node_lookup[y, x] = int(node["index"])

    visited_segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edge_paths: dict[tuple[int, int], list[tuple[int, int]]] = {}
    skeleton_binary = (skeleton > 0.5).astype(np.uint8)

    for node in nodes:
        source_index = int(node["index"])
        for y, x in node["pixels"]:
            for neighbor in _skeleton_neighbors(skeleton_binary, y, x):
                if node_lookup[neighbor] == source_index:
                    continue
                segment_key = tuple(sorted(((y, x), neighbor)))
                if segment_key in visited_segments:
                    continue

                path = [(y, x), neighbor]
                visited_segments.add(segment_key)
                previous = (y, x)
                current = neighbor

                while True:
                    current_node = int(node_lookup[current])
                    if current_node >= 0 and current_node != source_index:
                        pair = tuple(sorted((source_index, current_node)))
                        if pair not in edge_paths or len(path) < len(edge_paths[pair]):
                            edge_paths[pair] = path.copy()
                        break

                    next_candidates = [
                        candidate
                        for candidate in _skeleton_neighbors(skeleton_binary, current[0], current[1])
                        if candidate != previous
                    ]
                    if not next_candidates:
                        break

                    if len(next_candidates) > 1:
                        labeled_candidates = [candidate for candidate in next_candidates if node_lookup[candidate] >= 0]
                        if len(labeled_candidates) == 1:
                            next_point = labeled_candidates[0]
                        else:
                            unlabeled_candidates = [
                                candidate
                                for candidate in next_candidates
                                if node_lookup[candidate] < 0
                                and tuple(sorted((current, candidate))) not in visited_segments
                            ]
                            if len(unlabeled_candidates) != 1:
                                break
                            next_point = unlabeled_candidates[0]
                    else:
                        next_point = next_candidates[0]

                    segment_key = tuple(sorted((current, next_point)))
                    if segment_key in visited_segments and node_lookup[next_point] < 0:
                        break
                    visited_segments.add(segment_key)
                    previous, current = current, next_point
                    path.append(current)

    traced_edges = [
        (int(source), int(target), path)
        for (source, target), path in sorted(edge_paths.items(), key=lambda item: item[0])
    ]
    return traced_edges


def _edge_component_sizes(
    adjacency: dict[int, set[int]],
    source: int,
    target: int,
) -> tuple[int, int]:
    def bfs(start: int, blocked: tuple[int, int]) -> set[int]:
        queue = [start]
        visited = {start}
        while queue:
            current = queue.pop()
            for neighbor in adjacency.get(current, set()):
                if (current == blocked[0] and neighbor == blocked[1]) or (
                    current == blocked[1] and neighbor == blocked[0]
                ):
                    continue
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return visited

    source_component = bfs(source, (source, target))
    target_component = bfs(target, (source, target))
    return len(source_component), len(target_component)


def _causal_edge_importance(
    adjacency: dict[int, set[int]],
    source: int,
    target: int,
    path_length: int,
    diagonal_length: float,
) -> float:
    component_a, component_b = _edge_component_sizes(adjacency, source, target)
    total_component = max(component_a + component_b, 1)
    split_importance = min(component_a, component_b) / max(total_component - 1, 1)
    length_importance = min(float(path_length) / max(diagonal_length, 1.0), 1.0)
    causal_importance = 0.65 * split_importance + 0.35 * length_importance
    return float(np.clip(causal_importance, 0.0, 1.0))


def _build_graph_training_targets(
    skeleton: np.ndarray,
    junction: np.ndarray,
    endpoint: np.ndarray,
    max_pairs: int = 96,
    path_length: int = 48,
    max_pair_distance: float = 96.0,
) -> dict[str, np.ndarray]:
    nodes = _graph_nodes_from_targets(skeleton, junction, endpoint)
    positive_edges = _trace_graph_edges(skeleton, nodes)

    graph_node_capacity = np.zeros_like(skeleton, dtype=np.float32)
    graph_causal_path = np.zeros_like(skeleton, dtype=np.float32)
    graph_backbone_path = np.zeros_like(skeleton, dtype=np.float32)
    graph_redundant_path = np.zeros_like(skeleton, dtype=np.float32)
    pair_points = np.zeros((max_pairs, 2, 2), dtype=np.int64)
    pair_path_points = np.zeros((max_pairs, path_length, 2), dtype=np.int64)
    pair_path_mask = np.zeros((max_pairs, path_length), dtype=np.float32)
    pair_target = np.zeros((max_pairs,), dtype=np.float32)
    pair_importance = np.zeros((max_pairs,), dtype=np.float32)
    pair_valid = np.zeros((max_pairs,), dtype=np.float32)
    pair_minimal_target = np.zeros((max_pairs,), dtype=np.float32)
    pair_minimal_valid = np.zeros((max_pairs,), dtype=np.float32)

    positive_lookup = {
        tuple(sorted((source, target))): path for source, target, path in positive_edges
    }

    if not nodes:
        return {
            "graph_pair_points": pair_points,
            "graph_pair_path_points": pair_path_points,
            "graph_pair_path_mask": pair_path_mask,
            "graph_target": pair_target,
            "graph_pair_importance": pair_importance,
            "graph_pair_valid": pair_valid,
            "graph_minimal_target": pair_minimal_target,
            "graph_minimal_valid": pair_minimal_valid,
            "graph_node_capacity": graph_node_capacity[None, ...],
            "graph_causal_path": graph_causal_path[None, ...],
            "graph_backbone_path": graph_backbone_path[None, ...],
            "graph_redundant_path": graph_redundant_path[None, ...],
        }

    adjacency: dict[int, set[int]] = {int(node["index"]): set() for node in nodes}
    positive_importance: dict[tuple[int, int], float] = {}
    diagonal_length = float(np.hypot(*skeleton.shape))
    for source, target, path in positive_edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    for source, target, path in positive_edges:
        pair = tuple(sorted((source, target)))
        importance = _causal_edge_importance(
            adjacency=adjacency,
            source=source,
            target=target,
            path_length=len(path),
            diagonal_length=diagonal_length,
        )
        positive_importance[pair] = importance
        for y, x in path:
            graph_causal_path[y, x] = max(graph_causal_path[y, x], importance)
    for node in nodes:
        node_degree = len(adjacency.get(int(node["index"]), set()))
        capacity = float(np.clip((node_degree - 1) / 3.0, 0.0, 1.0))
        for y, x in node["pixels"]:
            graph_node_capacity[y, x] = capacity

    weighted_positive_edges: list[tuple[int, int, float]] = []
    for source, target, path in positive_edges:
        pair = tuple(sorted((source, target)))
        path_bonus = float(min(len(path) / max(diagonal_length, 1.0), 1.0))
        edge_score = float(positive_importance.get(pair, 0.0) + 0.10 * path_bonus)
        weighted_positive_edges.append((source, target, edge_score))
    forest_positive_pairs = _maximum_spanning_edge_subset(len(nodes), weighted_positive_edges)
    if positive_importance:
        importance_values = np.asarray(list(positive_importance.values()), dtype=np.float32)
        backbone_threshold = float(np.quantile(importance_values, 0.70))
        minimal_positive_pairs = {
            pair for pair, importance in positive_importance.items() if float(importance) >= backbone_threshold
        }
        for node in nodes:
            if node["kind"] != "junction":
                continue
            node_index = int(node["index"])
            incident_pairs = [
                (pair, importance)
                for pair, importance in positive_importance.items()
                if node_index in pair
            ]
            if incident_pairs:
                best_pair = max(incident_pairs, key=lambda item: item[1])[0]
                minimal_positive_pairs.add(best_pair)
        if not minimal_positive_pairs:
            minimal_positive_pairs = set(forest_positive_pairs)
    else:
        minimal_positive_pairs = set()

    for source, target, path in positive_edges:
        pair = tuple(sorted((source, target)))
        importance = float(positive_importance.get(pair, 0.0))
        if pair in minimal_positive_pairs:
            for y, x in path:
                graph_backbone_path[y, x] = max(graph_backbone_path[y, x], importance)
        else:
            redundancy_risk = float(np.clip(1.0 - importance, 0.0, 1.0))
            for y, x in path:
                graph_redundant_path[y, x] = max(graph_redundant_path[y, x], redundancy_risk)

    positive_pairs: list[tuple[tuple[int, int], list[tuple[int, int]]]] = list(positive_lookup.items())
    negative_pairs: list[tuple[int, int, float]] = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            pair = (left, right)
            if pair in positive_lookup:
                continue
            point_left = np.asarray(nodes[left]["point"], dtype=np.float32)
            point_right = np.asarray(nodes[right]["point"], dtype=np.float32)
            distance = float(np.linalg.norm(point_left - point_right))
            if distance <= max_pair_distance:
                negative_pairs.append((left, right, distance))
    negative_pairs.sort(key=lambda item: item[2])

    max_positive = min(len(positive_pairs), max((3 * max_pairs) // 4, 1))
    minimal_positive = [
        item for item in positive_pairs if tuple(sorted(item[0])) in minimal_positive_pairs
    ]
    redundant_positive = [
        item for item in positive_pairs if tuple(sorted(item[0])) not in minimal_positive_pairs
    ]
    if len(minimal_positive) > max_positive > 0:
        minimal_positive.sort(
            key=lambda item: positive_importance.get(tuple(sorted(item[0])), 0.0),
            reverse=True,
        )
        selected_positive = minimal_positive[:max_positive]
    else:
        selected_positive = list(minimal_positive)
        remaining_positive = max_positive - len(selected_positive)
        if len(redundant_positive) > remaining_positive > 0:
            sample_indices = np.linspace(
                0,
                len(redundant_positive) - 1,
                remaining_positive,
            ).round().astype(int)
            selected_positive.extend([redundant_positive[index] for index in sample_indices.tolist()])
        else:
            selected_positive.extend(redundant_positive)

    pair_records: list[tuple[int, int, float, float, float, float, list[tuple[int, int]]]] = []
    for (source, target), path in selected_positive:
        pair_key = tuple(sorted((source, target)))
        pair_records.append(
            (
                source,
                target,
                1.0,
                float(positive_importance.get(pair_key, 1.0)),
                1.0 if pair_key in minimal_positive_pairs else 0.0,
                1.0,
                path,
            )
        )

    remaining = max_pairs - len(pair_records)
    for source, target, _ in negative_pairs[: max(remaining, 0)]:
        pair_records.append(
            (
                source,
                target,
                0.0,
                0.0,
                0.0,
                0.0,
                _line_points(nodes[source]["point"], nodes[target]["point"]),
            )
        )

    for pair_index, (source, target, label, importance, minimal_label, minimal_valid, path) in enumerate(
        pair_records[:max_pairs]
    ):
        pair_points[pair_index, 0] = np.asarray(nodes[source]["point"], dtype=np.int64)
        pair_points[pair_index, 1] = np.asarray(nodes[target]["point"], dtype=np.int64)
        sampled_path, sampled_mask = _sample_path_points(path, path_length)
        pair_path_points[pair_index] = sampled_path
        pair_path_mask[pair_index] = sampled_mask
        pair_target[pair_index] = float(label)
        pair_importance[pair_index] = float(importance)
        pair_valid[pair_index] = 1.0
        pair_minimal_target[pair_index] = float(minimal_label)
        pair_minimal_valid[pair_index] = float(minimal_valid)

    return {
        "graph_pair_points": pair_points,
        "graph_pair_path_points": pair_path_points,
        "graph_pair_path_mask": pair_path_mask,
        "graph_target": pair_target,
        "graph_pair_importance": pair_importance,
        "graph_pair_valid": pair_valid,
        "graph_minimal_target": pair_minimal_target,
        "graph_minimal_valid": pair_minimal_valid,
        "graph_node_capacity": graph_node_capacity[None, ...],
        "graph_causal_path": graph_causal_path[None, ...],
        "graph_backbone_path": graph_backbone_path[None, ...],
        "graph_redundant_path": graph_redundant_path[None, ...],
    }


class CalciumSummaryNpzDataset(Dataset):
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        candidate_files = sorted(self.root.glob("*.npz"))
        required_keys = {"image", "mask", "skeleton", "junction", "endpoint"}
        self.files: list[Path] = []
        skipped_files: list[Path] = []
        for path in candidate_files:
            try:
                with np.load(path) as sample:
                    if required_keys.issubset(sample.files):
                        self.files.append(path)
                    else:
                        skipped_files.append(path)
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                skipped_files.append(path)
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.root}")
        if skipped_files:
            print(f"Skipping {len(skipped_files)} unreadable samples from {self.root}")

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

        graph_targets = _build_graph_training_targets(
            skeleton=skeleton[0],
            junction=junction[0],
            endpoint=endpoint[0],
        )

        batch = {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "skeleton": torch.from_numpy(skeleton),
            "junction": torch.from_numpy(junction),
            "endpoint": torch.from_numpy(endpoint),
            "affinity": torch.from_numpy(affinity),
            "uncertainty": torch.from_numpy(uncertainty),
        }
        batch.update({key: torch.from_numpy(value) for key, value in graph_targets.items()})
        return batch


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
        graph_targets = _build_graph_training_targets(
            skeleton=skeleton,
            junction=junction,
            endpoint=endpoint,
        )

        batch = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": mask,
            "skeleton": skeleton_tensor,
            "junction": torch.from_numpy(junction[None, ...].astype(np.float32)),
            "endpoint": torch.from_numpy(endpoint[None, ...].astype(np.float32)),
            "affinity": torch.from_numpy(affinity.astype(np.float32)),
            "uncertainty": torch.from_numpy(uncertainty.astype(np.float32)),
        }
        batch.update({key: torch.from_numpy(value) for key, value in graph_targets.items()})
        return batch

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
