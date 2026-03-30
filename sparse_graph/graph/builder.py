from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch.nn import functional as F

try:
    from skimage.graph import route_through_array
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    route_through_array = None


@dataclass
class GraphNode:
    index: int
    y: int
    x: int
    kind: str
    score: float


@dataclass
class GraphEdge:
    source: int
    target: int
    score: float
    path: list[tuple[int, int]]


class GraphBuilder:
    def __init__(
        self,
        node_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        skeleton_threshold: float = 0.5,
        junction_threshold: float | None = None,
        endpoint_threshold: float | None = None,
        max_neighbor_distance: int = 24,
        min_path_support: float = 0.35,
        max_neighbors_per_node: int = 2,
        decode_mode: str = "geodesic",
        nms_radius: int = 4,
        max_path_ratio: float = 1.85,
        affinity_weight: float = 0.2,
        uncertainty_weight: float = 0.75,
        bridge_weight: float = 0.2,
        relay_weight: float = 0.15,
        mask_weight: float = 0.1,
        relation_weight: float = 0.55,
    ) -> None:
        self.node_threshold = node_threshold
        self.mask_threshold = mask_threshold
        self.skeleton_threshold = skeleton_threshold
        self.junction_threshold = junction_threshold if junction_threshold is not None else node_threshold
        self.endpoint_threshold = endpoint_threshold if endpoint_threshold is not None else node_threshold
        self.max_neighbor_distance = max_neighbor_distance
        self.min_path_support = min_path_support
        self.max_neighbors_per_node = max_neighbors_per_node
        self.decode_mode = decode_mode
        self.nms_radius = nms_radius
        self.max_path_ratio = max_path_ratio
        self.affinity_weight = affinity_weight
        self.uncertainty_weight = uncertainty_weight
        self.bridge_weight = bridge_weight
        self.relay_weight = relay_weight
        self.mask_weight = mask_weight
        self.relation_weight = relation_weight

    def __call__(self, predictions: dict[str, torch.Tensor]) -> dict[str, Any]:
        mask = torch.sigmoid(predictions["mask_logits"][0, 0])
        skeleton = torch.sigmoid(predictions["skeleton_logits"][0, 0])
        junction = torch.sigmoid(predictions["junction_logits"][0, 0])
        endpoint = torch.sigmoid(predictions["endpoint_logits"][0, 0])
        uncertainty = (
            torch.sigmoid(predictions["uncertainty_logits"][0, 0])
            if "uncertainty_logits" in predictions
            else torch.zeros_like(skeleton)
        )
        affinity = predictions["affinity"][0] if "affinity" in predictions else torch.zeros((2,) + skeleton.shape)
        relay = torch.sigmoid(predictions["relay_logits"][0, 0]) if "relay_logits" in predictions else skeleton
        bridge = (
            torch.sigmoid(predictions["bridge_logits"][0, 0])
            if "bridge_logits" in predictions
            else torch.zeros_like(skeleton)
        )
        graph_query = predictions["graph_query_embeddings"][0] if "graph_query_embeddings" in predictions else None
        graph_key = predictions["graph_key_embeddings"][0] if "graph_key_embeddings" in predictions else None
        path_memory = (
            torch.sigmoid(predictions["path_memory_logits"][0, 0])
            if "path_memory_logits" in predictions
            else None
        )
        counterfactual_gate = (
            torch.sigmoid(predictions["counterfactual_gate_logits"][0, 0])
            if "counterfactual_gate_logits" in predictions
            else None
        )

        nodes = self._extract_nodes(junction, endpoint, skeleton, relay, bridge)
        edges = self._propose_edges(
            nodes,
            mask,
            skeleton,
            relay,
            bridge,
            uncertainty,
            affinity,
            graph_query,
            graph_key,
            path_memory,
            counterfactual_gate,
        )

        graph = nx.Graph()
        for node in nodes:
            graph.add_node(
                node.index,
                y=node.y,
                x=node.x,
                kind=node.kind,
                score=node.score,
            )
        for edge in edges:
            graph.add_edge(edge.source, edge.target, score=edge.score, path=edge.path)

        return {"nodes": nodes, "edges": edges, "graph": graph}

    def _extract_nodes(
        self,
        junction: torch.Tensor,
        endpoint: torch.Tensor,
        skeleton: torch.Tensor,
        relay: torch.Tensor,
        bridge: torch.Tensor,
    ) -> list[GraphNode]:
        support = torch.maximum(skeleton, torch.maximum(relay, bridge))
        candidates: list[tuple[str, float, int, int]] = []
        for kind, heatmap, threshold in (
            ("junction", junction, self.junction_threshold),
            ("endpoint", endpoint, self.endpoint_threshold),
        ):
            pooled = F.max_pool2d(heatmap[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
            maxima = (heatmap >= threshold) & (heatmap == pooled)
            ys, xs = torch.nonzero(maxima, as_tuple=True)
            for y, x in zip(ys.tolist(), xs.tolist()):
                score = 0.7 * float(heatmap[y, x].item()) + 0.3 * float(support[y, x].item())
                candidates.append((kind, score, int(y), int(x)))

        # Greedy NMS stops repeated heatmap peaks from exploding the decoded graph.
        candidates.sort(key=lambda item: (item[1], 1 if item[0] == "junction" else 0), reverse=True)
        nodes: list[GraphNode] = []
        for kind, score, y, x in candidates:
            if any(np.hypot(node.y - y, node.x - x) <= float(self.nms_radius) for node in nodes):
                continue
            nodes.append(
                GraphNode(
                    index=len(nodes),
                    y=y,
                    x=x,
                    kind=kind,
                    score=score,
                )
            )
        return nodes

    def _propose_edges(
        self,
        nodes: list[GraphNode],
        mask: torch.Tensor,
        skeleton: torch.Tensor,
        relay: torch.Tensor,
        bridge: torch.Tensor,
        uncertainty: torch.Tensor,
        affinity: torch.Tensor,
        graph_query: torch.Tensor | None,
        graph_key: torch.Tensor | None,
        path_memory: torch.Tensor | None,
        counterfactual_gate: torch.Tensor | None,
    ) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        used_pairs: set[tuple[int, int]] = set()
        degree_budget = {
            node.index: (1 if node.kind == "endpoint" else max(self.max_neighbors_per_node, 3))
            for node in nodes
        }
        current_degree = {node.index: 0 for node in nodes}

        structure_support = torch.clamp(
            0.55 * skeleton
            + self.bridge_weight * bridge
            + self.relay_weight * relay
            + self.mask_weight * mask,
            min=1e-4,
            max=1.0,
        )
        cost_map = (-torch.log(structure_support) + self.uncertainty_weight * uncertainty).detach().cpu().numpy()
        support_np = structure_support.detach().cpu().numpy()
        uncertainty_np = uncertainty.detach().cpu().numpy()
        affinity_np = affinity.detach().cpu().numpy()
        graph_query_np = graph_query.detach().cpu().numpy() if graph_query is not None else None
        graph_key_np = graph_key.detach().cpu().numpy() if graph_key is not None else None
        path_memory_np = path_memory.detach().cpu().numpy() if path_memory is not None else None
        counterfactual_np = (
            counterfactual_gate.detach().cpu().numpy() if counterfactual_gate is not None else None
        )

        for node in nodes:
            distances: list[tuple[float, GraphNode]] = []
            for other in nodes:
                if node.index == other.index:
                    continue
                distance = float(np.hypot(node.y - other.y, node.x - other.x))
                distances.append((distance, other))
            distances.sort(key=lambda item: item[0])

            for distance, other in distances:
                if current_degree[node.index] >= degree_budget[node.index]:
                    break
                if current_degree[other.index] >= degree_budget[other.index]:
                    continue
                if distance > self.max_neighbor_distance:
                    break

                pair = tuple(sorted((node.index, other.index)))
                if pair in used_pairs:
                    continue

                path = self._route_path((node.y, node.x), (other.y, other.x), cost_map)
                if not path:
                    continue
                path_ratio = len(path) / max(distance, 1.0)
                if path_ratio > self.max_path_ratio:
                    continue

                path_support = self._path_support(path, support_np)
                affinity_alignment = self._path_affinity_alignment(path, affinity_np)
                mean_uncertainty = self._path_support(path, uncertainty_np)
                heuristic_score = (
                    0.55 * path_support
                    + self.affinity_weight * affinity_alignment
                    + 0.15 * 0.5 * (node.score + other.score)
                    + 0.10 * (1.0 - mean_uncertainty)
                )
                relation_prob = self._edge_relation_probability(
                    node,
                    other,
                    path,
                    graph_query_np,
                    graph_key_np,
                    path_memory_np,
                    counterfactual_np,
                    support_np,
                    uncertainty_np,
                )
                edge_score = (1.0 - self.relation_weight) * heuristic_score + self.relation_weight * relation_prob
                if path_support < self.min_path_support or edge_score < self.min_path_support:
                    continue

                edges.append(
                    GraphEdge(
                        source=pair[0],
                        target=pair[1],
                        score=float(edge_score),
                        path=path,
                    )
                )
                used_pairs.add(pair)
                current_degree[node.index] += 1
                current_degree[other.index] += 1
        return edges

    def _route_path(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        cost_map: np.ndarray,
    ) -> list[tuple[int, int]]:
        if self.decode_mode == "line" or route_through_array is None:
            return self._straight_line_path(start, end)
        try:
            path, _ = route_through_array(
                cost_map,
                start,
                end,
                fully_connected=True,
                geometric=True,
            )
        except Exception:
            return self._straight_line_path(start, end)
        return [(int(y), int(x)) for y, x in path]

    def _straight_line_path(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[tuple[int, int]]:
        y0, x0 = start
        y1, x1 = end
        steps = max(abs(y1 - y0), abs(x1 - x0)) + 1
        ys = np.linspace(y0, y1, steps).round().astype(int)
        xs = np.linspace(x0, x1, steps).round().astype(int)
        return list(zip(ys.tolist(), xs.tolist()))

    def _path_support(
        self,
        path: list[tuple[int, int]],
        score_map: np.ndarray,
    ) -> float:
        if not path:
            return 0.0
        ys = np.array([point[0] for point in path], dtype=np.int32)
        xs = np.array([point[1] for point in path], dtype=np.int32)
        values = score_map[ys, xs]
        return float(values.mean()) if len(values) else 0.0

    def _path_affinity_alignment(
        self,
        path: list[tuple[int, int]],
        affinity: np.ndarray,
    ) -> float:
        if len(path) < 2:
            return 0.0
        alignments: list[float] = []
        for (y0, x0), (y1, x1) in zip(path[:-1], path[1:]):
            step = np.array([x1 - x0, y1 - y0], dtype=np.float32)
            norm = float(np.linalg.norm(step))
            if norm == 0.0:
                continue
            step = step / norm
            flow = affinity[:, y0, x0]
            flow_norm = float(np.linalg.norm(flow))
            if flow_norm == 0.0:
                continue
            flow = flow / flow_norm
            alignments.append(float(abs(np.dot(step, flow))))
        return float(np.mean(alignments)) if alignments else 0.0

    def _edge_relation_probability(
        self,
        node: GraphNode,
        other: GraphNode,
        path: list[tuple[int, int]],
        graph_query: np.ndarray | None,
        graph_key: np.ndarray | None,
        path_memory: np.ndarray | None,
        counterfactual_gate: np.ndarray | None,
        support_map: np.ndarray,
        uncertainty_map: np.ndarray,
    ) -> float:
        if graph_query is None or graph_key is None or path_memory is None:
            return 0.5

        q_src = graph_query[:, node.y, node.x]
        q_dst = graph_query[:, other.y, other.x]
        k_src = graph_key[:, node.y, node.x]
        k_dst = graph_key[:, other.y, other.x]
        embedding_dim = max(int(graph_query.shape[0]), 1)
        compatibility = 0.5 * (
            float(np.dot(q_src, k_dst)) + float(np.dot(q_dst, k_src))
        ) / float(embedding_dim**0.5)

        path_memory_score = self._path_support(path, path_memory)
        path_support = self._path_support(path, support_map)
        uncertainty = self._path_support(path, uncertainty_map)
        counterfactual_score = (
            self._path_support(path, counterfactual_gate) if counterfactual_gate is not None else 0.0
        )
        spatial_extent = max(float(graph_query.shape[-2]), float(graph_query.shape[-1]), 1.0)
        distance = float(np.hypot(node.y - other.y, node.x - other.x)) / spatial_extent
        relation_logit = (
            1.35 * compatibility
            + 1.00 * path_memory_score
            + 0.90 * path_support
            + 0.20 * counterfactual_score
            + 0.10 * 0.5 * (node.score + other.score)
            - 0.45 * uncertainty
            - 0.30 * distance
        )
        return float(1.0 / (1.0 + np.exp(-relation_logit)))
