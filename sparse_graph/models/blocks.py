from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..losses import soft_skeletonize


def make_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            make_norm(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.act(x + residual)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.block(x)


class TopologyAwareRefinement(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        topology_channels: int = 4,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        hidden = max(feature_channels // reduction, 8)
        self.topology_encoder = nn.Sequential(
            nn.Conv2d(topology_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            make_norm(feature_channels),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            make_norm(feature_channels),
            nn.SiLU(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(feature_channels * 2, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_channels * 2, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mix = ResidualBlock(feature_channels * 2, feature_channels)

    def forward(self, features: torch.Tensor, topology_logits: list[torch.Tensor]) -> torch.Tensor:
        topology_prior = torch.cat([torch.sigmoid(logits) for logits in topology_logits], dim=1)
        topology_features = self.topology_encoder(topology_prior)
        joint = torch.cat([features, topology_features], dim=1)
        spatial = self.spatial_gate(joint)
        channel = self.channel_gate(joint)
        gated = features * (1.0 + spatial) * (1.0 + channel)
        return self.mix(torch.cat([gated, topology_features], dim=1))


class DenseSparseCoupling(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.dense_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sparse_gate = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dense_mix = ResidualBlock(channels * 2, channels)
        self.sparse_mix = ResidualBlock(channels * 2, channels)

    def forward(self, dense_features: torch.Tensor, sparse_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint = torch.cat([dense_features, sparse_features], dim=1)
        dense_gate = self.dense_gate(joint)
        sparse_gate = self.sparse_gate(joint)
        dense_out = self.dense_mix(
            torch.cat([dense_features * (1.0 + dense_gate), sparse_features], dim=1)
        )
        sparse_out = self.sparse_mix(
            torch.cat([sparse_features * (1.0 + sparse_gate), dense_features], dim=1)
        )
        return dense_out, sparse_out


class TopologyConsensusRollout(nn.Module):
    def __init__(self, channels: int, topology_channels: int = 4, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.topology_encoder = nn.Sequential(
            nn.Conv2d(topology_channels + 2, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.consensus_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.conflict_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.dense_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sparse_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dense_update = ResidualBlock(channels * 2, channels)
        self.sparse_update = ResidualBlock(channels * 2, channels)

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: torch.Tensor,
        topology_logits: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topology_probs = [torch.sigmoid(logits) for logits in topology_logits]
        mask_prob, skeleton_prob, junction_prob, endpoint_prob = topology_probs
        consensus = torch.sqrt(mask_prob.clamp_min(1e-6) * skeleton_prob.clamp_min(1e-6))
        consensus = consensus * (1.0 + torch.maximum(junction_prob, endpoint_prob))
        conflict = torch.abs(mask_prob - skeleton_prob) + torch.relu(
            torch.maximum(junction_prob, endpoint_prob) - skeleton_prob
        )
        topology_state = self.topology_encoder(torch.cat(topology_probs + [consensus, conflict], dim=1))
        consensus_features = self.consensus_to_features(consensus)
        conflict_features = self.conflict_to_features(conflict)

        dense_gate = self.dense_gate(
            torch.cat(
                [dense_features, sparse_features, topology_state, consensus_features],
                dim=1,
            )
        )
        sparse_gate = self.sparse_gate(
            torch.cat(
                [sparse_features, dense_features, topology_state, conflict_features],
                dim=1,
            )
        )

        shared_dense = topology_state + consensus_features - 0.5 * conflict_features
        shared_sparse = topology_state + consensus_features - conflict_features
        dense_out = self.dense_update(
            torch.cat([dense_features * (1.0 + dense_gate), shared_dense], dim=1)
        )
        sparse_out = self.sparse_update(
            torch.cat([sparse_features * (1.0 + sparse_gate), shared_sparse], dim=1)
        )
        return dense_out, sparse_out


class BranchRelayReasoning(nn.Module):
    def __init__(
        self,
        channels: int,
        affinity_channels: int = 2,
        relay_kernel_sizes: tuple[int, ...] = (5, 9, 13),
        counterfactual_drop_prob: float = 0.3,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.relay_kernel_sizes = tuple(relay_kernel_sizes)
        self.counterfactual_drop_prob = float(counterfactual_drop_prob)
        self.topology_encoder = nn.Sequential(
            nn.Conv2d(8, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.affinity_encoder = nn.Sequential(
            nn.Conv2d(affinity_channels, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.endpoint_relay_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.junction_relay_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.bridge_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.counterfactual_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.dense_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sparse_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dense_update = ResidualBlock(channels * 2, channels)
        self.sparse_update = ResidualBlock(channels * 2, channels)
        for kernel_size in self.relay_kernel_sizes:
            self.register_buffer(
                f"line_kernels_{kernel_size}",
                self._make_line_kernels(kernel_size),
                persistent=False,
            )

    @staticmethod
    def _make_line_kernels(kernel_size: int) -> torch.Tensor:
        if kernel_size % 2 == 0:
            raise ValueError("relay_kernel_size must be odd.")
        kernels = torch.zeros((4, 1, kernel_size, kernel_size), dtype=torch.float32)
        center = kernel_size // 2
        kernels[0, 0, center, :] = 1.0
        kernels[1, 0, :, center] = 1.0
        for index in range(kernel_size):
            kernels[2, 0, index, index] = 1.0
            kernels[3, 0, index, kernel_size - 1 - index] = 1.0
        kernels = kernels / kernels.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        return kernels

    def _propagate_with_orientation(
        self,
        seed_map: torch.Tensor,
        orientation_weights: torch.Tensor,
    ) -> torch.Tensor:
        responses: list[torch.Tensor] = []
        for kernel_size in self.relay_kernel_sizes:
            line_kernels = getattr(self, f"line_kernels_{kernel_size}")
            line_response = F.conv2d(
                seed_map,
                line_kernels.to(device=seed_map.device, dtype=seed_map.dtype),
                padding=kernel_size // 2,
            )
            responses.append((orientation_weights * line_response).sum(dim=1, keepdim=True))
        return torch.stack(responses, dim=0).mean(dim=0)

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: torch.Tensor,
        topology_logits: list[torch.Tensor],
        affinity_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        mask_prob, skeleton_prob, junction_prob, endpoint_prob = [
            torch.sigmoid(logits) for logits in topology_logits
        ]
        node_seed = torch.maximum(junction_prob, endpoint_prob)
        affinity = torch.tanh(affinity_logits)

        norm = torch.sqrt((affinity**2).sum(dim=1, keepdim=True) + 1e-6)
        dir_x = affinity[:, 0:1] / norm
        dir_y = affinity[:, 1:2] / norm
        inv_sqrt_two = 0.7071067811865476
        orientation_logits = torch.cat(
            [
                dir_x.abs(),
                dir_y.abs(),
                ((dir_x + dir_y) * inv_sqrt_two).abs(),
                ((dir_x - dir_y) * inv_sqrt_two).abs(),
            ],
            dim=1,
        )
        orientation_weights = orientation_logits / orientation_logits.sum(dim=1, keepdim=True).clamp_min(1e-6)

        endpoint_relay = self._propagate_with_orientation(endpoint_prob, orientation_weights)
        junction_relay = self._propagate_with_orientation(junction_prob, orientation_weights)
        node_context = F.max_pool2d(node_seed, kernel_size=3, stride=1, padding=1)
        relay_map = torch.clamp(
            0.45 * endpoint_relay
            + 0.35 * junction_relay
            + 0.35 * skeleton_prob
            + 0.2 * node_context,
            0.0,
            1.0,
        )

        max_kernel = max(self.relay_kernel_sizes)
        bridge_seed = F.max_pool2d(
            node_seed,
            kernel_size=max_kernel,
            stride=1,
            padding=max_kernel // 2,
        )
        bridge_map = torch.clamp(
            mask_prob * (1.0 - skeleton_prob) * (0.6 * relay_map + 0.4 * bridge_seed),
            0.0,
            1.0,
        )

        if self.training and self.counterfactual_drop_prob > 0:
            drop_focus = (relay_map > relay_map.mean(dim=(2, 3), keepdim=True)).float()
            drop_mask = (torch.rand_like(relay_map) < self.counterfactual_drop_prob).float() * drop_focus
        else:
            drop_mask = torch.zeros_like(relay_map)
        counterfactual_skeleton = skeleton_prob * (1.0 - drop_mask)
        counterfactual_bridge = torch.clamp(
            mask_prob
            * (1.0 - counterfactual_skeleton)
            * (0.55 * endpoint_relay + 0.3 * junction_relay + 0.35 * node_context),
            0.0,
            1.0,
        )
        bridge_consensus = torch.maximum(bridge_map, counterfactual_bridge)
        relay_conflict = torch.abs(endpoint_relay - junction_relay)

        topology_state = self.topology_encoder(
            torch.cat(
                [
                    mask_prob,
                    skeleton_prob,
                    node_seed,
                    endpoint_relay,
                    junction_relay,
                    bridge_consensus,
                    counterfactual_bridge,
                    relay_conflict,
                ],
                dim=1,
            )
        )
        affinity_features = self.affinity_encoder(affinity)
        endpoint_relay_features = self.endpoint_relay_to_features(endpoint_relay)
        junction_relay_features = self.junction_relay_to_features(junction_relay)
        bridge_features = self.bridge_to_features(bridge_consensus)
        counterfactual_features = self.counterfactual_to_features(counterfactual_bridge)

        dense_gate = self.dense_gate(
            torch.cat(
                [
                    dense_features,
                    sparse_features,
                    topology_state,
                    bridge_features + counterfactual_features,
                ],
                dim=1,
            )
        )
        sparse_context = endpoint_relay_features + junction_relay_features + affinity_features + counterfactual_features
        sparse_gate = self.sparse_gate(
            torch.cat([sparse_features, dense_features, topology_state, sparse_context], dim=1)
        )

        dense_out = self.dense_update(
            torch.cat(
                [
                    dense_features * (1.0 + dense_gate),
                    topology_state + bridge_features + counterfactual_features,
                ],
                dim=1,
            )
        )
        sparse_out = self.sparse_update(
            torch.cat(
                [
                    sparse_features * (1.0 + sparse_gate) * (1.0 + relay_map),
                    topology_state + sparse_context,
                ],
                dim=1,
            )
        )

        outputs = {
            "relay_logits": torch.logit(relay_map.clamp(1e-4, 1.0 - 1e-4)),
            "bridge_logits": torch.logit(bridge_consensus.clamp(1e-4, 1.0 - 1e-4)),
            "counterfactual_bridge_logits": torch.logit(
                counterfactual_bridge.clamp(1e-4, 1.0 - 1e-4)
            ),
            "endpoint_relay_logits": torch.logit(endpoint_relay.clamp(1e-4, 1.0 - 1e-4)),
            "junction_relay_logits": torch.logit(junction_relay.clamp(1e-4, 1.0 - 1e-4)),
        }
        return dense_out, sparse_out, outputs


class CounterfactualGeodesicReasoning(nn.Module):
    def __init__(
        self,
        channels: int,
        embedding_channels: int = 16,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.topology_encoder = nn.Sequential(
            nn.Conv2d(8, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.prior_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.counterfactual_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.graph_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dense_update = ResidualBlock(channels * 2, channels)
        self.sparse_update = ResidualBlock(channels * 2, channels)
        self.graph_update = ResidualBlock(channels * 2, channels)
        self.query_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, embedding_channels, kernel_size=1),
        )
        self.key_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, embedding_channels, kernel_size=1),
        )
        self.path_memory_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.counterfactual_gate_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: torch.Tensor,
        topology_logits: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        topology_probs = [torch.sigmoid(logits) for logits in topology_logits]
        mask_prob, skeleton_prob, junction_prob, endpoint_prob, relay_prob, bridge_prob, counterfactual_prob = (
            topology_probs
        )
        node_support = torch.maximum(junction_prob, endpoint_prob)
        geodesic_prior = torch.clamp(
            0.35 * mask_prob
            + 0.45 * skeleton_prob
            + 0.35 * relay_prob
            + 0.30 * bridge_prob
            + 0.20 * counterfactual_prob
            + 0.15 * node_support,
            0.0,
            1.0,
        )
        counterfactual_gap = torch.clamp(counterfactual_prob - skeleton_prob, min=0.0, max=1.0)
        topology_state = self.topology_encoder(
            torch.cat(
                [
                    mask_prob,
                    skeleton_prob,
                    junction_prob,
                    endpoint_prob,
                    relay_prob,
                    bridge_prob,
                    counterfactual_prob,
                    counterfactual_gap,
                ],
                dim=1,
            )
        )
        prior_features = self.prior_to_features(geodesic_prior)
        counterfactual_features = self.counterfactual_to_features(counterfactual_gap)
        graph_gate = self.graph_gate(
            torch.cat(
                [
                    dense_features,
                    sparse_features,
                    topology_state,
                    prior_features + counterfactual_features,
                ],
                dim=1,
            )
        )

        shared_dense = topology_state + prior_features - 0.25 * counterfactual_features
        shared_sparse = topology_state + prior_features + 0.35 * counterfactual_features
        dense_out = self.dense_update(
            torch.cat([dense_features * (1.0 + graph_gate), shared_dense], dim=1)
        )
        sparse_out = self.sparse_update(
            torch.cat(
                [sparse_features * (1.0 + graph_gate) * (1.0 + geodesic_prior), shared_sparse],
                dim=1,
            )
        )
        graph_features = self.graph_update(
            torch.cat([0.5 * (dense_out + sparse_out), topology_state + prior_features], dim=1)
        )
        outputs = {
            "graph_query_embeddings": self.query_head(graph_features),
            "graph_key_embeddings": self.key_head(graph_features),
            "path_memory_logits": self.path_memory_head(graph_features),
            "counterfactual_gate_logits": self.counterfactual_gate_head(
                graph_features + counterfactual_features
            ),
        }
        return dense_out, sparse_out, outputs


class NodeAwareCausalPolicy(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.policy_encoder = nn.Sequential(
            nn.Conv2d(9, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.node_capacity_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.causal_saliency_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.node_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.causal_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.node_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.causal_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.dense_update = ResidualBlock(channels * 2, channels)
        self.sparse_update = ResidualBlock(channels * 2, channels)

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: torch.Tensor,
        topology_logits: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        topology_probs = [torch.sigmoid(logits) for logits in topology_logits]
        (
            mask_prob,
            skeleton_prob,
            junction_prob,
            endpoint_prob,
            relay_prob,
            bridge_prob,
            counterfactual_prob,
            path_memory_prob,
        ) = topology_probs
        node_support = torch.maximum(junction_prob, endpoint_prob)
        counterfactual_gap = torch.clamp(counterfactual_prob - skeleton_prob, min=0.0, max=1.0)
        policy_state = self.policy_encoder(
            torch.cat(
                [
                    mask_prob,
                    skeleton_prob,
                    junction_prob,
                    endpoint_prob,
                    relay_prob,
                    bridge_prob,
                    counterfactual_prob,
                    path_memory_prob,
                    counterfactual_gap,
                ],
                dim=1,
            )
        )
        node_capacity_logits = self.node_capacity_head(policy_state)
        causal_saliency_logits = self.causal_saliency_head(policy_state)
        node_capacity = torch.sigmoid(node_capacity_logits)
        causal_saliency = torch.sigmoid(causal_saliency_logits)
        node_prior = torch.clamp(0.65 * node_support + 0.35 * node_capacity, 0.0, 1.0)
        causal_prior = torch.clamp(
            0.40 * skeleton_prob
            + 0.25 * relay_prob
            + 0.20 * path_memory_prob
            + 0.15 * bridge_prob
            + 0.35 * causal_saliency
            - 0.10 * counterfactual_gap,
            0.0,
            1.0,
        )
        node_features = self.node_to_features(node_prior)
        causal_features = self.causal_to_features(causal_prior)
        node_gate = self.node_gate(
            torch.cat([dense_features, sparse_features, policy_state, node_features], dim=1)
        )
        causal_gate = self.causal_gate(
            torch.cat([sparse_features, dense_features, policy_state, causal_features], dim=1)
        )
        dense_out = self.dense_update(
            torch.cat([dense_features * (1.0 + node_gate), policy_state + node_features], dim=1)
        )
        sparse_out = self.sparse_update(
            torch.cat(
                [
                    sparse_features * (1.0 + causal_gate) * (1.0 + causal_prior),
                    policy_state + causal_features + 0.5 * node_features,
                ],
                dim=1,
            )
        )
        outputs = {
            "node_capacity_logits": node_capacity_logits,
            "causal_saliency_logits": causal_saliency_logits,
        }
        return dense_out, sparse_out, outputs


class CausalBranchAusterity(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.register_buffer(
            "neighbor_kernel",
            torch.tensor(
                [[[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.policy_encoder = nn.Sequential(
            nn.Conv2d(8, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.node_capacity_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.causal_saliency_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.branch_keep_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.branch_prune_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.dense_sparse_projection_head = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.keep_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.node_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.causal_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.prune_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.projection_to_features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(),
        )
        self.keep_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.prune_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dense_update = ResidualBlock(channels * 2, channels)
        self.sparse_update = ResidualBlock(channels * 2, channels)

    def _soft_branch_proxy(self, skeleton_prob: torch.Tensor) -> torch.Tensor:
        degree = F.conv2d(
            skeleton_prob,
            self.neighbor_kernel.to(device=skeleton_prob.device, dtype=skeleton_prob.dtype),
            padding=1,
        )
        branch_gate = torch.sigmoid((degree - 2.25) * 4.0)
        return torch.clamp(skeleton_prob * branch_gate, 0.0, 1.0)

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: torch.Tensor,
        topology_logits: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        topology_probs = [torch.sigmoid(logits) for logits in topology_logits]
        (
            mask_prob,
            skeleton_prob,
            junction_prob,
            endpoint_prob,
            relay_prob,
            bridge_prob,
            counterfactual_prob,
            path_memory_prob,
        ) = topology_probs
        node_support = torch.maximum(junction_prob, endpoint_prob)
        junction_context = F.max_pool2d(junction_prob, kernel_size=5, stride=1, padding=2)
        endpoint_context = F.max_pool2d(endpoint_prob, kernel_size=5, stride=1, padding=2)
        counterfactual_gap = torch.clamp(counterfactual_prob - skeleton_prob, min=0.0, max=1.0)
        branch_proxy = self._soft_branch_proxy(skeleton_prob)
        policy_state = self.policy_encoder(
            torch.cat(
                [
                    mask_prob,
                    skeleton_prob,
                    junction_prob,
                    endpoint_prob,
                    relay_prob,
                    bridge_prob,
                    counterfactual_prob,
                    path_memory_prob,
                ],
                dim=1,
            )
        )
        dense_sparse_projection_logits = self.dense_sparse_projection_head(
            torch.cat([policy_state, dense_features], dim=1)
        )
        dense_mask_prior = soft_skeletonize(mask_prob, iterations=6)
        dense_sparse_projection_prob = torch.clamp(
            0.60 * torch.sigmoid(dense_sparse_projection_logits) + 0.40 * dense_mask_prior,
            0.0,
            1.0,
        )
        node_capacity_logits = self.node_capacity_head(policy_state)
        causal_saliency_logits = self.causal_saliency_head(policy_state)
        node_capacity_prob = torch.sigmoid(node_capacity_logits)
        causal_saliency_prob = torch.sigmoid(causal_saliency_logits)
        essential_prior = torch.clamp(
            0.30 * skeleton_prob
            + 0.20 * relay_prob
            + 0.15 * path_memory_prob
            + 0.20 * dense_sparse_projection_prob
            + 0.25 * causal_saliency_prob
            + 0.15 * endpoint_context
            + 0.10 * junction_context
            + 0.10 * node_capacity_prob,
            0.0,
            1.0,
        )
        surplus_risk = torch.clamp(
            0.55 * branch_proxy * (1.0 - junction_context)
            + 0.20 * bridge_prob * (1.0 - path_memory_prob)
            + 0.15 * counterfactual_gap
            + 0.10 * relay_prob * (1.0 - node_capacity_prob),
            0.0,
            1.0,
        )
        branch_keep_logits = self.branch_keep_head(policy_state)
        branch_prune_logits = self.branch_prune_head(policy_state)
        branch_keep = torch.clamp(0.55 * torch.sigmoid(branch_keep_logits) + 0.45 * essential_prior, 0.0, 1.0)
        branch_prune = torch.clamp(
            torch.sigmoid(branch_prune_logits) * surplus_risk * (1.0 - 0.55 * dense_sparse_projection_prob),
            0.0,
            1.0,
        )
        keep_features = self.keep_to_features(branch_keep)
        node_features = self.node_to_features(node_capacity_prob)
        causal_features = self.causal_to_features(causal_saliency_prob)
        prune_features = self.prune_to_features(branch_prune)
        projection_features = self.projection_to_features(dense_sparse_projection_prob)
        keep_gate = self.keep_gate(
            torch.cat([dense_features, sparse_features, policy_state, keep_features + node_features], dim=1)
        )
        prune_gate = self.prune_gate(
            torch.cat([sparse_features, dense_features, policy_state, prune_features + causal_features], dim=1)
        )
        dense_out = self.dense_update(
            torch.cat(
                [
                    dense_features
                    * (1.0 + keep_gate * (0.5 * branch_keep + 0.3 * node_capacity_prob + 0.2 * dense_sparse_projection_prob)),
                    policy_state + keep_features + 0.5 * node_features + 0.35 * projection_features - 0.25 * prune_features,
                ],
                dim=1,
            )
        )
        sparse_out = self.sparse_update(
            torch.cat(
                [
                    sparse_features
                    * (1.0 + keep_gate * branch_keep)
                    * (1.0 + 0.35 * causal_saliency_prob)
                    * (1.0 + 0.30 * dense_sparse_projection_prob)
                    * (1.0 - 0.65 * prune_gate * branch_prune)
                    * (1.0 + 0.20 * node_support),
                    policy_state
                    + keep_features
                    + causal_features
                    + 0.35 * node_features
                    + 0.50 * projection_features
                    - prune_features,
                ],
                dim=1,
            )
        )
        outputs = {
            "node_capacity_logits": node_capacity_logits,
            "causal_saliency_logits": causal_saliency_logits,
            "branch_keep_logits": branch_keep_logits,
            "branch_prune_logits": branch_prune_logits,
            "dense_sparse_projection_logits": dense_sparse_projection_logits,
        }
        return dense_out, sparse_out, outputs


class BioCausalRouting(nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_channels: int,
        morphology_channels: tuple[int, ...] = (0, 1),
        activity_channels: tuple[int, ...] = (2, 3),
        reduction: int = 4,
    ) -> None:
        super().__init__()
        if not morphology_channels or not activity_channels:
            raise ValueError("BioCausalRouting requires non-empty morphology and activity channel groups.")

        self.morphology_channels = tuple(morphology_channels)
        self.activity_channels = tuple(activity_channels)
        hidden = max(feature_channels // reduction, 8)

        self.morphology_prior = nn.Sequential(
            nn.Conv2d(len(self.morphology_channels), hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.activity_prior = nn.Sequential(
            nn.Conv2d(len(self.activity_channels), hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.agreement_head = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.conflict_head = nn.Sequential(
            nn.Conv2d(4, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

        self.morphology_to_features = nn.Sequential(
            nn.Conv2d(1, feature_channels, kernel_size=1, bias=False),
            make_norm(feature_channels),
            nn.SiLU(),
        )
        self.activity_to_features = nn.Sequential(
            nn.Conv2d(1, feature_channels, kernel_size=1, bias=False),
            make_norm(feature_channels),
            nn.SiLU(),
        )
        self.agreement_to_features = nn.Sequential(
            nn.Conv2d(1, feature_channels, kernel_size=1, bias=False),
            make_norm(feature_channels),
            nn.SiLU(),
        )
        self.conflict_to_features = nn.Sequential(
            nn.Conv2d(1, feature_channels, kernel_size=1, bias=False),
            make_norm(feature_channels),
            nn.SiLU(),
        )

        self.dense_gate = nn.Sequential(
            nn.Conv2d(feature_channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sparse_gate = nn.Sequential(
            nn.Conv2d(feature_channels * 4, hidden, kernel_size=1, bias=False),
            make_norm(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, feature_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.dense_mix = ResidualBlock(feature_channels * 2, feature_channels)
        self.sparse_mix = ResidualBlock(feature_channels * 2, feature_channels)

    def forward(
        self,
        x: torch.Tensor,
        dense_features: torch.Tensor,
        sparse_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        morphology_input = x[:, self.morphology_channels, ...]
        activity_input = x[:, self.activity_channels, ...]

        morphology_prior_logits = self.morphology_prior(morphology_input)
        activity_prior_logits = self.activity_prior(activity_input)
        morphology_prior = torch.sigmoid(morphology_prior_logits)
        activity_prior = torch.sigmoid(activity_prior_logits)

        agreement_logits = self.agreement_head(
            torch.cat(
                [
                    morphology_prior,
                    activity_prior,
                    morphology_prior * activity_prior,
                ],
                dim=1,
            )
        )
        conflict_logits = self.conflict_head(
            torch.cat(
                [
                    torch.abs(morphology_prior - activity_prior),
                    morphology_prior * (1.0 - activity_prior),
                    activity_prior * (1.0 - morphology_prior),
                    torch.maximum(morphology_prior, activity_prior),
                ],
                dim=1,
            )
        )
        agreement = torch.sigmoid(agreement_logits)
        conflict = torch.sigmoid(conflict_logits)

        morphology_features = self.morphology_to_features(morphology_prior)
        activity_features = self.activity_to_features(activity_prior)
        agreement_features = self.agreement_to_features(agreement)
        conflict_features = self.conflict_to_features(conflict)

        dense_gate = self.dense_gate(
            torch.cat(
                [dense_features, morphology_features, agreement_features, conflict_features],
                dim=1,
            )
        )
        sparse_gate = self.sparse_gate(
            torch.cat(
                [sparse_features, activity_features, agreement_features, conflict_features],
                dim=1,
            )
        )

        dense_conditioned = dense_features * (1.0 + dense_gate) * (1.0 + agreement)
        sparse_conditioned = sparse_features * (1.0 + sparse_gate) * (1.0 + agreement) * (1.0 - 0.5 * conflict)

        dense_out = self.dense_mix(torch.cat([dense_conditioned, morphology_features], dim=1))
        sparse_out = self.sparse_mix(torch.cat([sparse_conditioned, activity_features], dim=1))

        priors = {
            "morphology_prior_logits": morphology_prior_logits,
            "activity_prior_logits": activity_prior_logits,
            "agreement_logits": agreement_logits,
            "conflict_logits": conflict_logits,
        }
        return dense_out, sparse_out, priors


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, stride=2),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)
