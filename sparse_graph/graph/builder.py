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


@dataclass
class CandidateBridge:
    source: int
    target: int
    score: float
    relation_prob: float
    path_support: float
    uncertainty: float
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
        causal_weight: float = 0.25,
        prune_mode: str = "none",
        prune_terminal_score: float = 0.68,
        prune_terminal_node_score: float = 0.80,
        prune_min_spur_length: int = 32,
        prune_keep_trunk_length: int = 96,
        prune_keep_edge_score: float = 0.90,
        prune_min_component_length: int = 80,
        prune_component_score: float = 0.78,
        enforce_tree: bool = False,
        candidate_distance_ratio: float = 1.35,
        candidate_support_ratio: float = 0.78,
        candidate_score_ratio: float = 0.82,
        candidate_relation_threshold: float = 0.52,
        max_candidate_bridges: int = 96,
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
        self.causal_weight = causal_weight
        self.prune_mode = prune_mode
        self.prune_terminal_score = prune_terminal_score
        self.prune_terminal_node_score = prune_terminal_node_score
        self.prune_min_spur_length = prune_min_spur_length
        self.prune_keep_trunk_length = prune_keep_trunk_length
        self.prune_keep_edge_score = prune_keep_edge_score
        self.prune_min_component_length = prune_min_component_length
        self.prune_component_score = prune_component_score
        self.enforce_tree = enforce_tree
        self.candidate_distance_ratio = candidate_distance_ratio
        self.candidate_support_ratio = candidate_support_ratio
        self.candidate_score_ratio = candidate_score_ratio
        self.candidate_relation_threshold = candidate_relation_threshold
        self.max_candidate_bridges = max_candidate_bridges

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
        node_capacity = (
            torch.sigmoid(predictions["node_capacity_logits"][0, 0])
            if "node_capacity_logits" in predictions
            else None
        )
        causal_saliency = (
            torch.sigmoid(predictions["causal_saliency_logits"][0, 0])
            if "causal_saliency_logits" in predictions
            else None
        )
        dense_sparse_projection = (
            torch.sigmoid(predictions["dense_sparse_projection_logits"][0, 0])
            if "dense_sparse_projection_logits" in predictions
            else None
        )

        nodes = self._extract_nodes(junction, endpoint, skeleton, relay, bridge, dense_sparse_projection)
        edges, candidate_bridges = self._propose_edges(
            nodes,
            mask,
            skeleton,
            relay,
            bridge,
            dense_sparse_projection,
            uncertainty,
            affinity,
            graph_query,
            graph_key,
            path_memory,
            counterfactual_gate,
            node_capacity,
            causal_saliency,
        )

        raw_nodes = list(nodes)
        raw_edges = list(edges)
        nodes, edges = self._prune_graph(nodes, edges)
        candidate_bridges = self._filter_candidate_bridges(nodes, edges, candidate_bridges)

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

        return {
            "nodes": nodes,
            "edges": edges,
            "graph": graph,
            "raw_nodes": raw_nodes,
            "raw_edges": raw_edges,
            "candidate_bridges": candidate_bridges,
        }

    def _extract_nodes(
        self,
        junction: torch.Tensor,
        endpoint: torch.Tensor,
        skeleton: torch.Tensor,
        relay: torch.Tensor,
        bridge: torch.Tensor,
        dense_sparse_projection: torch.Tensor | None,
    ) -> list[GraphNode]:
        support = torch.maximum(skeleton, torch.maximum(relay, bridge))
        if dense_sparse_projection is not None:
            support = torch.maximum(support, dense_sparse_projection)
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
        dense_sparse_projection: torch.Tensor | None,
        uncertainty: torch.Tensor,
        affinity: torch.Tensor,
        graph_query: torch.Tensor | None,
        graph_key: torch.Tensor | None,
        path_memory: torch.Tensor | None,
        counterfactual_gate: torch.Tensor | None,
        node_capacity: torch.Tensor | None,
        causal_saliency: torch.Tensor | None,
    ) -> tuple[list[GraphEdge], list[CandidateBridge]]:
        edges: list[GraphEdge] = []
        candidate_bridges: list[CandidateBridge] = []
        candidate_pairs: set[tuple[int, int]] = set()
        used_pairs: set[tuple[int, int]] = set()
        degree_budget = {}
        for node in nodes:
            if node.kind == "endpoint":
                degree_budget[node.index] = 1
                continue
            if node_capacity is not None:
                capacity_value = float(node_capacity[node.y, node.x].item())
                predicted_budget = int(round(1.0 + 3.0 * capacity_value))
                degree_budget[node.index] = max(2, min(max(self.max_neighbors_per_node, 4), predicted_budget))
            else:
                degree_budget[node.index] = max(self.max_neighbors_per_node, 3)
        current_degree = {node.index: 0 for node in nodes}

        structure_support = torch.clamp(
            0.45 * skeleton
            + self.bridge_weight * bridge
            + self.relay_weight * relay
            + 0.20 * (dense_sparse_projection if dense_sparse_projection is not None else skeleton)
            + self.mask_weight * mask,
            min=1e-4,
            max=1.0,
        )
        if causal_saliency is not None:
            structure_support = torch.clamp(
                structure_support + self.causal_weight * causal_saliency,
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
        node_capacity_np = node_capacity.detach().cpu().numpy() if node_capacity is not None else None
        causal_saliency_np = causal_saliency.detach().cpu().numpy() if causal_saliency is not None else None
        candidate_distance_limit = float(self.max_neighbor_distance) * float(self.candidate_distance_ratio)
        candidate_support_threshold = float(self.min_path_support) * float(self.candidate_support_ratio)
        candidate_score_threshold = float(self.min_path_support) * float(self.candidate_score_ratio)

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
                    over_budget = True
                else:
                    over_budget = False
                if distance > candidate_distance_limit:
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
                    node_capacity_np,
                    causal_saliency_np,
                    support_np,
                    uncertainty_np,
                )
                edge_score = (1.0 - self.relation_weight) * heuristic_score + self.relation_weight * relation_prob
                meets_confirmed = (
                    not over_budget
                    and distance <= float(self.max_neighbor_distance)
                    and path_support >= float(self.min_path_support)
                    and edge_score >= float(self.min_path_support)
                )
                if not meets_confirmed:
                    if (
                        pair not in candidate_pairs
                        and len(candidate_bridges) < int(self.max_candidate_bridges)
                        and distance <= candidate_distance_limit
                        and path_support >= candidate_support_threshold
                        and edge_score >= candidate_score_threshold
                        and relation_prob >= float(self.candidate_relation_threshold)
                        and ("endpoint" in {node.kind, other.kind} or over_budget)
                    ):
                        candidate_bridges.append(
                            CandidateBridge(
                                source=pair[0],
                                target=pair[1],
                                score=float(edge_score),
                                relation_prob=float(relation_prob),
                                path_support=float(path_support),
                                uncertainty=float(mean_uncertainty),
                                path=path,
                            )
                        )
                        candidate_pairs.add(pair)
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
        candidate_bridges.sort(
            key=lambda bridge: (bridge.score, bridge.relation_prob, bridge.path_support),
            reverse=True,
        )
        return edges, candidate_bridges[: int(self.max_candidate_bridges)]

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
        node_capacity: np.ndarray | None,
        causal_saliency: np.ndarray | None,
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
        node_capacity_score = (
            0.5 * (float(node_capacity[node.y, node.x]) + float(node_capacity[other.y, other.x]))
            if node_capacity is not None
            else 0.0
        )
        causal_score = self._path_support(path, causal_saliency) if causal_saliency is not None else 0.0
        counterfactual_score = (
            self._path_support(path, counterfactual_gate) if counterfactual_gate is not None else 0.0
        )
        spatial_extent = max(float(graph_query.shape[-2]), float(graph_query.shape[-1]), 1.0)
        distance = float(np.hypot(node.y - other.y, node.x - other.x)) / spatial_extent
        relation_logit = (
            1.35 * compatibility
            + 1.00 * path_memory_score
            + 0.90 * path_support
            + 0.75 * causal_score
            + 0.30 * node_capacity_score
            + 0.20 * counterfactual_score
            + 0.10 * 0.5 * (node.score + other.score)
            - 0.45 * uncertainty
            - 0.30 * distance
        )
        return float(1.0 / (1.0 + np.exp(-relation_logit)))

    def _prune_graph(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        if self.prune_mode == "none" or not edges:
            kept_indices = {node.index for edge in edges for node in nodes if node.index in (edge.source, edge.target)}
            if not kept_indices:
                return nodes, edges
            kept_nodes = [node for node in nodes if node.index in kept_indices]
            return kept_nodes, edges

        node_lookup = {node.index: node for node in nodes}
        working_edges = list(edges)
        working_edges = self._remove_weak_components(working_edges)

        for _ in range(8):
            degrees = self._node_degrees(working_edges)
            filtered_edges: list[GraphEdge] = []
            changed = False
            for edge in working_edges:
                if self._should_prune_terminal_edge(edge, degrees, node_lookup):
                    changed = True
                    continue
                filtered_edges.append(edge)
            filtered_edges = self._remove_weak_components(filtered_edges)
            if len(filtered_edges) != len(working_edges):
                changed = True
            working_edges = filtered_edges
            if not changed:
                break

        if self.enforce_tree and working_edges:
            working_edges = self._maximum_spanning_forest(working_edges)

        kept_indices = {index for edge in working_edges for index in (edge.source, edge.target)}
        kept_nodes = [node for node in nodes if node.index in kept_indices]
        return kept_nodes, working_edges

    def _filter_candidate_bridges(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        candidate_bridges: list[CandidateBridge],
    ) -> list[CandidateBridge]:
        if not candidate_bridges:
            return []

        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node.index)
        for edge in edges:
            graph.add_edge(edge.source, edge.target)

        component_lookup: dict[int, int] = {}
        for component_index, component in enumerate(nx.connected_components(graph)):
            for node_index in component:
                component_lookup[int(node_index)] = component_index

        filtered: list[CandidateBridge] = []
        seen_pairs: set[tuple[int, int]] = set()
        for bridge in sorted(
            candidate_bridges,
            key=lambda item: (item.score, item.relation_prob, item.path_support),
            reverse=True,
        ):
            pair = tuple(sorted((bridge.source, bridge.target)))
            if pair in seen_pairs:
                continue
            left_component = component_lookup.get(bridge.source, -1)
            right_component = component_lookup.get(bridge.target, -2)
            if left_component == right_component and left_component >= 0:
                continue
            filtered.append(bridge)
            seen_pairs.add(pair)
            if len(filtered) >= int(self.max_candidate_bridges):
                break
        return filtered

    def _node_degrees(self, edges: list[GraphEdge]) -> dict[int, int]:
        degrees: dict[int, int] = {}
        for edge in edges:
            degrees[edge.source] = degrees.get(edge.source, 0) + 1
            degrees[edge.target] = degrees.get(edge.target, 0) + 1
        return degrees

    def _remove_weak_components(self, edges: list[GraphEdge]) -> list[GraphEdge]:
        if self.prune_mode == "none" or not edges:
            return edges

        graph = nx.Graph()
        edge_lookup: dict[tuple[int, int], GraphEdge] = {}
        for edge in edges:
            pair = tuple(sorted((edge.source, edge.target)))
            edge_lookup[pair] = edge
            graph.add_edge(edge.source, edge.target)

        kept_pairs: set[tuple[int, int]] = set()
        for component in nx.connected_components(graph):
            component_edges = [
                edge_lookup[tuple(sorted((left, right)))]
                for left, right in graph.subgraph(component).edges()
            ]
            if not component_edges:
                continue
            component_length = float(sum(len(edge.path) for edge in component_edges))
            component_max_score = float(max(edge.score for edge in component_edges))
            if (
                component_length < float(self.prune_min_component_length)
                and component_max_score < float(self.prune_component_score)
            ):
                continue
            for edge in component_edges:
                kept_pairs.add(tuple(sorted((edge.source, edge.target))))

        return [edge for edge in edges if tuple(sorted((edge.source, edge.target))) in kept_pairs]

    def _maximum_spanning_forest(self, edges: list[GraphEdge]) -> list[GraphEdge]:
        graph = nx.Graph()
        edge_lookup: dict[tuple[int, int], GraphEdge] = {}
        for edge in edges:
            pair = tuple(sorted((edge.source, edge.target)))
            edge_lookup[pair] = edge
            length_bonus = min(len(edge.path) / max(float(self.prune_keep_trunk_length), 1.0), 1.0)
            graph.add_edge(edge.source, edge.target, weight=float(edge.score) + 0.12 * length_bonus)

        kept_pairs: set[tuple[int, int]] = set()
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component).copy()
            forest = nx.maximum_spanning_tree(subgraph, weight="weight")
            for left, right in forest.edges():
                kept_pairs.add(tuple(sorted((int(left), int(right)))))
        return [edge for edge in edges if tuple(sorted((edge.source, edge.target))) in kept_pairs]

    def _should_prune_terminal_edge(
        self,
        edge: GraphEdge,
        degrees: dict[int, int],
        node_lookup: dict[int, GraphNode],
    ) -> bool:
        if self.prune_mode == "none":
            return False

        edge_length = len(edge.path)
        if edge_length >= int(self.prune_keep_trunk_length) or edge.score >= float(self.prune_keep_edge_score):
            return False

        leaf_indices = [index for index in (edge.source, edge.target) if degrees.get(index, 0) <= 1]
        if not leaf_indices:
            return False

        leaf_nodes = [node_lookup[index] for index in leaf_indices if index in node_lookup]
        leaf_score = max((node.score for node in leaf_nodes), default=0.0)
        leaf_kinds = {node.kind for node in leaf_nodes}

        if edge_length <= int(self.prune_min_spur_length) and edge.score < float(self.prune_terminal_score):
            return True
        if (
            "endpoint" in leaf_kinds
            and edge.score < float(self.prune_terminal_score)
            and leaf_score < float(self.prune_terminal_node_score)
        ):
            return True
        if self.prune_mode == "reviewer":
            if edge_length <= int(self.prune_min_spur_length * 1.5) and edge.score < float(self.prune_terminal_score + 0.05):
                return True
        return False
