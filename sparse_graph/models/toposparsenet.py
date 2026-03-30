from __future__ import annotations

import torch
from torch import nn

from .backbone import HRUNetBackbone
from .blocks import (
    BioCausalRouting,
    BranchRelayReasoning,
    CausalBranchAusterity,
    CounterfactualGeodesicReasoning,
    DenseSparseCoupling,
    NodeAwareCausalPolicy,
    ResidualBlock,
    TopologyAwareRefinement,
    TopologyConsensusRollout,
)
from .heads import PredictionHead


class TopoSparseNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        use_attention: bool = True,
        affinity_channels: int = 2,
        use_topology_refinement: bool = False,
        topology_variant: str = "none",
        morphology_channels: tuple[int, ...] = (0, 1),
        activity_channels: tuple[int, ...] = (2, 3),
        graph_embedding_channels: int = 16,
    ) -> None:
        super().__init__()
        if topology_variant == "none" and use_topology_refinement:
            topology_variant = "refine"
        self.topology_variant = topology_variant
        self.backbone = HRUNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            use_attention=use_attention,
        )
        feature_channels = self.backbone.out_channels
        if self.topology_variant in {
            "dual_branch",
            "bio_causal_dual",
            "consensus_rollout",
            "relay_consensus",
            "relay_geodesic",
            "relay_geodesic_nodecausal",
            "relay_geodesic_nodecausal_austerity",
            "relay_geodesic_policy",
        }:
            self.dense_stem = ResidualBlock(feature_channels, feature_channels)
            self.sparse_stem = ResidualBlock(feature_channels, feature_channels)
            self.coarse_mask_head = PredictionHead(feature_channels, 1)
            self.coarse_skeleton_head = PredictionHead(feature_channels, 1)
            self.coarse_junction_head = PredictionHead(feature_channels, 1)
            self.coarse_endpoint_head = PredictionHead(feature_channels, 1)
            self.dense_refinement = TopologyAwareRefinement(feature_channels, topology_channels=4)
            self.sparse_refinement = TopologyAwareRefinement(feature_channels, topology_channels=4)
            self.branch_coupling = DenseSparseCoupling(feature_channels)
            self.consensus_rollout = (
                TopologyConsensusRollout(feature_channels, topology_channels=4)
                if self.topology_variant in {"consensus_rollout", "relay_consensus"}
                else None
            )
            self.branch_relay = (
                BranchRelayReasoning(feature_channels, affinity_channels=affinity_channels)
                if self.topology_variant
                in {
                    "relay_consensus",
                    "relay_geodesic",
                    "relay_geodesic_nodecausal",
                    "relay_geodesic_nodecausal_austerity",
                    "relay_geodesic_policy",
                }
                else None
            )
            self.counterfactual_geodesic = (
                CounterfactualGeodesicReasoning(
                    feature_channels,
                    embedding_channels=graph_embedding_channels,
                )
                if self.topology_variant
                in {
                    "relay_geodesic",
                    "relay_geodesic_nodecausal",
                    "relay_geodesic_nodecausal_austerity",
                    "relay_geodesic_policy",
                }
                else None
            )
            self.node_causal_policy = (
                NodeAwareCausalPolicy(feature_channels)
                if self.topology_variant == "relay_geodesic_nodecausal"
                else None
            )
            self.causal_branch_austerity = (
                CausalBranchAusterity(feature_channels)
                if self.topology_variant in {"relay_geodesic_nodecausal_austerity", "relay_geodesic_policy"}
                else None
            )
            if self.topology_variant == "bio_causal_dual":
                self.bio_causal_router = BioCausalRouting(
                    in_channels=in_channels,
                    feature_channels=feature_channels,
                    morphology_channels=morphology_channels,
                    activity_channels=activity_channels,
                )
            else:
                self.bio_causal_router = None
        elif self.topology_variant == "refine":
            self.coarse_mask_head = PredictionHead(feature_channels, 1)
            self.coarse_skeleton_head = PredictionHead(feature_channels, 1)
            self.coarse_junction_head = PredictionHead(feature_channels, 1)
            self.coarse_endpoint_head = PredictionHead(feature_channels, 1)
            self.topology_refinement = TopologyAwareRefinement(feature_channels, topology_channels=4)
            self.dense_stem = None
            self.sparse_stem = None
            self.dense_refinement = None
            self.sparse_refinement = None
            self.branch_coupling = None
            self.bio_causal_router = None
            self.consensus_rollout = None
            self.branch_relay = None
            self.counterfactual_geodesic = None
            self.node_causal_policy = None
            self.causal_branch_austerity = None
        elif self.topology_variant == "none":
            self.coarse_mask_head = None
            self.coarse_skeleton_head = None
            self.coarse_junction_head = None
            self.coarse_endpoint_head = None
            self.topology_refinement = None
            self.dense_stem = None
            self.sparse_stem = None
            self.dense_refinement = None
            self.sparse_refinement = None
            self.branch_coupling = None
            self.bio_causal_router = None
            self.consensus_rollout = None
            self.branch_relay = None
            self.counterfactual_geodesic = None
            self.node_causal_policy = None
            self.causal_branch_austerity = None
        else:
            raise ValueError(f"Unsupported topology_variant: {self.topology_variant}")

        self.mask_head = PredictionHead(feature_channels, 1)
        self.skeleton_head = PredictionHead(feature_channels, 1)
        self.junction_head = PredictionHead(feature_channels, 1)
        self.endpoint_head = PredictionHead(feature_channels, 1)
        self.affinity_head = PredictionHead(feature_channels, affinity_channels)
        self.uncertainty_head = PredictionHead(feature_channels, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        outputs: dict[str, torch.Tensor] = {}
        dense_features = features
        sparse_features = features

        if self.topology_variant in {
            "dual_branch",
            "bio_causal_dual",
            "consensus_rollout",
            "relay_consensus",
            "relay_geodesic",
            "relay_geodesic_nodecausal",
            "relay_geodesic_nodecausal_austerity",
            "relay_geodesic_policy",
        }:
            dense_features = self.dense_stem(features)
            sparse_features = self.sparse_stem(features)
            coarse_mask_logits = self.coarse_mask_head(dense_features)
            coarse_skeleton_logits = self.coarse_skeleton_head(sparse_features)
            coarse_junction_logits = self.coarse_junction_head(sparse_features)
            coarse_endpoint_logits = self.coarse_endpoint_head(sparse_features)
            topology_logits = [
                coarse_mask_logits,
                coarse_skeleton_logits,
                coarse_junction_logits,
                coarse_endpoint_logits,
            ]
            dense_features = self.dense_refinement(dense_features, topology_logits)
            sparse_features = self.sparse_refinement(sparse_features, topology_logits)
            if self.bio_causal_router is not None:
                dense_features, sparse_features, bio_outputs = self.bio_causal_router(
                    x,
                    dense_features,
                    sparse_features,
                )
                outputs.update(bio_outputs)
            dense_features, sparse_features = self.branch_coupling(dense_features, sparse_features)
            if self.consensus_rollout is not None:
                rollout_mask_logits = self.mask_head(dense_features)
                rollout_skeleton_logits = self.skeleton_head(sparse_features)
                rollout_junction_logits = self.junction_head(sparse_features)
                rollout_endpoint_logits = self.endpoint_head(sparse_features)
                dense_features, sparse_features = self.consensus_rollout(
                    dense_features,
                    sparse_features,
                    [
                        rollout_mask_logits,
                        rollout_skeleton_logits,
                        rollout_junction_logits,
                        rollout_endpoint_logits,
                    ],
                )
                outputs.update(
                    {
                        "rollout_mask_logits": rollout_mask_logits,
                        "rollout_skeleton_logits": rollout_skeleton_logits,
                        "rollout_junction_logits": rollout_junction_logits,
                        "rollout_endpoint_logits": rollout_endpoint_logits,
                    }
                )
            if self.branch_relay is not None:
                relay_topology_logits = [
                    outputs.get("rollout_mask_logits", coarse_mask_logits),
                    outputs.get("rollout_skeleton_logits", coarse_skeleton_logits),
                    outputs.get("rollout_junction_logits", coarse_junction_logits),
                    outputs.get("rollout_endpoint_logits", coarse_endpoint_logits),
                ]
                relay_affinity_logits = self.affinity_head(sparse_features)
                dense_features, sparse_features, relay_outputs = self.branch_relay(
                    dense_features,
                    sparse_features,
                    relay_topology_logits,
                    relay_affinity_logits,
                )
                outputs.update(relay_outputs)
            if self.counterfactual_geodesic is not None:
                geodesic_topology_logits = [
                    outputs.get("rollout_mask_logits", coarse_mask_logits),
                    outputs.get("rollout_skeleton_logits", coarse_skeleton_logits),
                    outputs.get("rollout_junction_logits", coarse_junction_logits),
                    outputs.get("rollout_endpoint_logits", coarse_endpoint_logits),
                    outputs.get("relay_logits", coarse_skeleton_logits),
                    outputs.get("bridge_logits", coarse_skeleton_logits),
                    outputs.get("counterfactual_bridge_logits", coarse_skeleton_logits),
                ]
                dense_features, sparse_features, graph_outputs = self.counterfactual_geodesic(
                    dense_features,
                    sparse_features,
                    geodesic_topology_logits,
                )
                outputs.update(graph_outputs)
            if self.node_causal_policy is not None:
                policy_topology_logits = [
                    outputs.get("rollout_mask_logits", coarse_mask_logits),
                    outputs.get("rollout_skeleton_logits", coarse_skeleton_logits),
                    outputs.get("rollout_junction_logits", coarse_junction_logits),
                    outputs.get("rollout_endpoint_logits", coarse_endpoint_logits),
                    outputs.get("relay_logits", coarse_skeleton_logits),
                    outputs.get("bridge_logits", coarse_skeleton_logits),
                    outputs.get("counterfactual_bridge_logits", coarse_skeleton_logits),
                    outputs.get("path_memory_logits", coarse_skeleton_logits),
                ]
                dense_features, sparse_features, policy_outputs = self.node_causal_policy(
                    dense_features,
                    sparse_features,
                    policy_topology_logits,
                )
                outputs.update(policy_outputs)
            if self.causal_branch_austerity is not None:
                austerity_topology_logits = [
                    outputs.get("rollout_mask_logits", coarse_mask_logits),
                    outputs.get("rollout_skeleton_logits", coarse_skeleton_logits),
                    outputs.get("rollout_junction_logits", coarse_junction_logits),
                    outputs.get("rollout_endpoint_logits", coarse_endpoint_logits),
                    outputs.get("relay_logits", coarse_skeleton_logits),
                    outputs.get("bridge_logits", coarse_skeleton_logits),
                    outputs.get("counterfactual_bridge_logits", coarse_skeleton_logits),
                    outputs.get("path_memory_logits", coarse_skeleton_logits),
                ]
                dense_features, sparse_features, austerity_outputs = self.causal_branch_austerity(
                    dense_features,
                    sparse_features,
                    austerity_topology_logits,
                )
                outputs.update(austerity_outputs)
            outputs.update(
                {
                    "coarse_mask_logits": coarse_mask_logits,
                    "coarse_skeleton_logits": coarse_skeleton_logits,
                    "coarse_junction_logits": coarse_junction_logits,
                    "coarse_endpoint_logits": coarse_endpoint_logits,
                }
            )
        elif self.topology_variant == "refine":
            coarse_mask_logits = self.coarse_mask_head(features)
            coarse_skeleton_logits = self.coarse_skeleton_head(features)
            coarse_junction_logits = self.coarse_junction_head(features)
            coarse_endpoint_logits = self.coarse_endpoint_head(features)
            dense_features = self.topology_refinement(
                features,
                [
                    coarse_mask_logits,
                    coarse_skeleton_logits,
                    coarse_junction_logits,
                    coarse_endpoint_logits,
                ],
            )
            outputs.update(
                {
                    "coarse_mask_logits": coarse_mask_logits,
                    "coarse_skeleton_logits": coarse_skeleton_logits,
                    "coarse_junction_logits": coarse_junction_logits,
                    "coarse_endpoint_logits": coarse_endpoint_logits,
                }
            )

        if self.topology_variant in {
            "dual_branch",
            "bio_causal_dual",
            "consensus_rollout",
            "relay_consensus",
            "relay_geodesic",
            "relay_geodesic_nodecausal",
            "relay_geodesic_nodecausal_austerity",
            "relay_geodesic_policy",
        }:
            mask_logits = self.mask_head(dense_features)
            skeleton_logits = self.skeleton_head(sparse_features)
            junction_logits = self.junction_head(sparse_features)
            endpoint_logits = self.endpoint_head(sparse_features)
            affinity_logits = self.affinity_head(sparse_features)
            uncertainty_logits = self.uncertainty_head(dense_features)
            if "bridge_logits" in outputs:
                skeleton_logits = skeleton_logits + 0.3 * outputs["bridge_logits"]
            if "counterfactual_bridge_logits" in outputs:
                skeleton_logits = skeleton_logits + 0.2 * outputs["counterfactual_bridge_logits"]
            if "junction_relay_logits" in outputs:
                junction_logits = junction_logits + 0.25 * outputs["junction_relay_logits"]
            if "endpoint_relay_logits" in outputs:
                endpoint_logits = endpoint_logits + 0.25 * outputs["endpoint_relay_logits"]
            if "path_memory_logits" in outputs:
                skeleton_logits = skeleton_logits + 0.20 * outputs["path_memory_logits"]
            if "counterfactual_gate_logits" in outputs:
                skeleton_logits = skeleton_logits + 0.10 * outputs["counterfactual_gate_logits"]
            if "causal_saliency_logits" in outputs:
                skeleton_logits = skeleton_logits + 0.15 * outputs["causal_saliency_logits"]
            if "dense_sparse_projection_logits" in outputs:
                skeleton_logits = skeleton_logits + 0.25 * outputs["dense_sparse_projection_logits"]
            if "branch_keep_logits" in outputs:
                skeleton_logits = skeleton_logits + 0.15 * outputs["branch_keep_logits"]
            if "branch_prune_logits" in outputs:
                skeleton_logits = skeleton_logits - 0.20 * outputs["branch_prune_logits"]
            if "conflict_logits" in outputs:
                uncertainty_logits = uncertainty_logits + outputs["conflict_logits"]
            outputs.update(
                {
                    "mask_logits": mask_logits,
                    "skeleton_logits": skeleton_logits,
                    "junction_logits": junction_logits,
                    "endpoint_logits": endpoint_logits,
                    "affinity": affinity_logits,
                    "uncertainty_logits": uncertainty_logits,
                }
            )
        else:
            outputs.update(
                {
                    "mask_logits": self.mask_head(dense_features),
                    "skeleton_logits": self.skeleton_head(dense_features),
                    "junction_logits": self.junction_head(dense_features),
                    "endpoint_logits": self.endpoint_head(dense_features),
                    "affinity": self.affinity_head(dense_features),
                    "uncertainty_logits": self.uncertainty_head(dense_features),
                }
            )
        return outputs
