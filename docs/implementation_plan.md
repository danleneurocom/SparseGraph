# Implementation Plan

## Goal

This document translates `TopoSparseNet` into a concrete V1 implementation plan. The emphasis is a codebase that is easy to train, inspect, and extend during early research iterations.

## V1 scope

The current scaffold implements:

- 2D multi-channel summary-map input
- residual U-Net backbone
- multitask prediction heads
- Dice, focal, and clDice-style losses
- synthetic dataset generation for smoke tests
- `.npz` dataset loader for real experiments
- simple graph-builder hook for sparse post-processing

The current scaffold does not yet implement:

- full video or 3D temporal encoder
- learned graph neural network refinement
- exact graph-supervised loss
- benchmark-specific data preprocessing

## Module map

| File | Responsibility |
| --- | --- |
| `sparse_graph/models/blocks.py` | Reusable residual, downsampling, upsampling, and attention blocks |
| `sparse_graph/models/backbone.py` | High-resolution residual U-Net backbone |
| `sparse_graph/models/heads.py` | Lightweight prediction heads for each structural target |
| `sparse_graph/models/toposparsenet.py` | Full multitask model assembly |
| `sparse_graph/losses.py` | Dice, focal, clDice, and aggregate multitask objective |
| `sparse_graph/data/datasets.py` | Synthetic-data generator and `.npz` dataset loader |
| `sparse_graph/graph/builder.py` | Simple node and edge extraction from sparse predictions |
| `sparse_graph/train.py` | Training loop, validation loop, checkpointing, and CLI wiring |
| `configs/toposparsenet_base.json` | Default experiment configuration |

## Default tensor shapes

The default config assumes:

- batch size `B`
- input channels `C = 4`
- spatial size `H = W = 256`
- base channels `32`

### Backbone

| Stage | Tensor shape |
| --- | --- |
| Input | `B x 4 x 256 x 256` |
| Stem | `B x 32 x 256 x 256` |
| Down 1 | `B x 64 x 128 x 128` |
| Down 2 | `B x 128 x 64 x 64` |
| Down 3 | `B x 256 x 32 x 32` |
| Bottleneck | `B x 256 x 32 x 32` |
| Up 1 | `B x 128 x 64 x 64` |
| Up 2 | `B x 64 x 128 x 128` |
| Up 3 | `B x 32 x 256 x 256` |
| Shared decoder feature | `B x 32 x 256 x 256` |

### Heads

| Head | Output shape |
| --- | --- |
| Mask logits | `B x 1 x 256 x 256` |
| Skeleton logits | `B x 1 x 256 x 256` |
| Junction logits | `B x 1 x 256 x 256` |
| Endpoint logits | `B x 1 x 256 x 256` |
| Affinity field | `B x 2 x 256 x 256` |
| Uncertainty logits | `B x 1 x 256 x 256` |

## Data contract

Each training sample should expose:

| Key | Shape | Notes |
| --- | --- | --- |
| `image` | `C x H x W` | Summary maps from Ca2+ imaging |
| `mask` | `1 x H x W` | Dense neurite occupancy |
| `skeleton` | `1 x H x W` | Sparse centerline target |
| `junction` | `1 x H x W` | Branch-point target |
| `endpoint` | `1 x H x W` | Endpoint target |
| `affinity` | `2 x H x W` | Optional local tangent target |
| `uncertainty` | `1 x H x W` | Optional ambiguity supervision |

If `affinity` or `uncertainty` are not available, they can be initialized to zeros during early training.

## Loss map

| Prediction | Target | Loss |
| --- | --- | --- |
| Mask | `mask` | Dice + focal |
| Skeleton | `skeleton` | Dice |
| Topology | `mask` and `skeleton` | clDice-style soft skeleton loss |
| Junction | `junction` | Focal |
| Endpoint | `endpoint` | Focal |
| Affinity | `affinity` | Masked smooth L1 on skeleton support |
| Uncertainty | `uncertainty` | MSE after sigmoid |

## Training stages

### Stage A: sanity pass

- train on synthetic data
- verify all heads learn non-trivial outputs
- confirm loss terms are numerically stable

### Stage B: summary-map experiments

- switch to real `.npz` data
- optimize dense mask and skeleton heads first
- inspect false breaks and false merges

### Stage C: topology ablations

- compare with and without `clDice`
- compare dense-only vs dense-plus-skeleton supervision
- quantify connected-component improvements

### Stage D: graph-aware refinement

- add a learned edge classifier or GNN
- promote graph-level metrics to primary model-selection criteria

## Recommended near-term extensions

1. Replace heuristic summary maps with a temporal encoder over short movie clips.
2. Add exact graph labels and a learned edge-refinement module.
3. Introduce weak or centerline-only supervision for low-annotation settings.
4. Add connectome-specific downstream evaluation.
