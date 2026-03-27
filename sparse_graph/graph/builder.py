from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch.nn import functional as F


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


class GraphBuilder:
    def __init__(
        self,
        node_threshold: float = 0.5,
        skeleton_threshold: float = 0.5,
        max_neighbor_distance: int = 24,
        min_path_support: float = 0.35,
        max_neighbors_per_node: int = 2,
    ) -> None:
        self.node_threshold = node_threshold
        self.skeleton_threshold = skeleton_threshold
        self.max_neighbor_distance = max_neighbor_distance
        self.min_path_support = min_path_support
        self.max_neighbors_per_node = max_neighbors_per_node

    def __call__(self, predictions: dict[str, torch.Tensor]) -> dict[str, Any]:
        skeleton = (torch.sigmoid(predictions["skeleton_logits"][0, 0]) >= self.skeleton_threshold).float()
        junction = torch.sigmoid(predictions["junction_logits"][0, 0])
        endpoint = torch.sigmoid(predictions["endpoint_logits"][0, 0])

        nodes = self._extract_nodes(junction, endpoint)
        edges = self._propose_edges(nodes, skeleton)

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
            graph.add_edge(edge.source, edge.target, score=edge.score)

        return {"nodes": nodes, "edges": edges, "graph": graph}

    def _extract_nodes(self, junction: torch.Tensor, endpoint: torch.Tensor) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        for kind, heatmap in (("junction", junction), ("endpoint", endpoint)):
            pooled = F.max_pool2d(heatmap[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
            maxima = (heatmap >= self.node_threshold) & (heatmap == pooled)
            ys, xs = torch.nonzero(maxima, as_tuple=True)
            for y, x in zip(ys.tolist(), xs.tolist()):
                nodes.append(
                    GraphNode(
                        index=len(nodes),
                        y=int(y),
                        x=int(x),
                        kind=kind,
                        score=float(heatmap[y, x].item()),
                    )
                )
        return nodes

    def _propose_edges(self, nodes: list[GraphNode], skeleton: torch.Tensor) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        used_pairs: set[tuple[int, int]] = set()
        skeleton_np = skeleton.detach().cpu().numpy()
        for node in nodes:
            distances = []
            for other in nodes:
                if node.index == other.index:
                    continue
                distance = float(np.hypot(node.y - other.y, node.x - other.x))
                distances.append((distance, other))
            distances.sort(key=lambda item: item[0])

            added = 0
            for distance, other in distances:
                if distance > self.max_neighbor_distance:
                    break
                pair = tuple(sorted((node.index, other.index)))
                if pair in used_pairs:
                    continue
                support = self._line_support((node.y, node.x), (other.y, other.x), skeleton_np)
                if support < self.min_path_support:
                    continue
                edges.append(GraphEdge(source=pair[0], target=pair[1], score=float(support)))
                used_pairs.add(pair)
                added += 1
                if added >= self.max_neighbors_per_node:
                    break
        return edges

    def _line_support(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        skeleton: np.ndarray,
    ) -> float:
        y0, x0 = start
        y1, x1 = end
        steps = max(abs(y1 - y0), abs(x1 - x0)) + 1
        ys = np.linspace(y0, y1, steps).round().astype(int)
        xs = np.linspace(x0, x1, steps).round().astype(int)
        values = skeleton[ys, xs]
        return float(values.mean()) if len(values) else 0.0
