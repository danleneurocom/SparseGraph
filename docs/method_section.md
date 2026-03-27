# Method

## 1. Overview

We propose `TopoSparseNet`, a topology-aware dense-to-sparse architecture for Ca2+ neuronal microscopy. The aim is not only to segment neuronal material, but to infer a sparse structural representation that preserves branch topology and can be converted into a connectome-ready graph with limited post-processing.

Given a calcium-imaging recording, the model predicts:

- a dense neurite occupancy mask
- a sparse centerline map
- junction heatmaps
- endpoint heatmaps
- a local tangent or affinity field
- an uncertainty map for ambiguous regions

The dense mask is treated as an auxiliary representation that stabilizes training, while the sparse outputs are the primary target.

## 2. Input Representation

The current implementation focuses on a practical 2D summary-map regime. Instead of processing the entire video cube directly, each recording is converted into a compact multi-channel image stack. A typical input contains:

- mean projection
- max projection
- temporal variation map
- local correlation or activity map

This design preserves temporal cues that are relevant for calcium imaging while keeping the memory footprint manageable during early experimentation.

## 3. Network Architecture

### 3.1 Shared encoder-decoder

The backbone is a high-resolution residual U-Net with limited downsampling. This choice is motivated by the thin and fragmented nature of neurites: aggressive pooling often destroys the continuity cues needed for sparse reconstruction.

The backbone contains:

- a residual stem
- three downsampling stages
- a bottleneck attention block
- three upsampling stages with skip connections

The decoder returns a full-resolution feature map that is shared by all prediction heads.

### 3.2 Multitask sparse heads

From the shared decoder feature map, the network predicts multiple structural outputs in parallel:

- `Mask head`: dense neurite occupancy
- `Skeleton head`: one-pixel-wide centerline map
- `Junction head`: branch-point heatmap
- `Endpoint head`: terminal-point heatmap
- `Affinity head`: 2D local tangent field that helps connect nearby centerline segments
- `Uncertainty head`: region-level ambiguity estimate, useful near crossings, blur, and weak signal

This multitask design follows the central lesson of topology-aware vascular segmentation: topology is easier to learn when the network is asked to predict explicit structural intermediates instead of only a dense binary mask.

## 4. Sparse Graph Construction

The final sparse representation is obtained by converting the predicted structural maps into a graph.

First, candidate nodes are detected from the junction and endpoint heatmaps using thresholding and local-maxima filtering. Next, centerline support is extracted from the skeleton map. Candidate node pairs are then connected when their straight-line path has sufficient support in the sparse prediction and the inferred distance remains within a biologically plausible radius.

This stage yields a graph with:

- node positions
- node types such as junction or endpoint
- edge candidates with confidence scores

The current scaffold uses a lightweight heuristic graph builder. In later versions, this module can be replaced with a learned graph refiner or a graph neural network.

## 5. Training Objective

Let the model output be:

```math
\hat{Y} = \{\hat{M}, \hat{S}, \hat{J}, \hat{E}, \hat{A}, \hat{U}\}
```

where `M` is the dense mask, `S` the sparse skeleton, `J` the junction map, `E` the endpoint map, `A` the affinity field, and `U` the uncertainty map.

The total loss is:

```math
\mathcal{L}_{total} =
\lambda_{mask}\mathcal{L}_{mask} +
\lambda_{skel}\mathcal{L}_{skel} +
\lambda_{topo}\mathcal{L}_{topo} +
\lambda_{node}\mathcal{L}_{node} +
\lambda_{aff}\mathcal{L}_{aff} +
\lambda_{unc}\mathcal{L}_{unc} +
\lambda_{graph}\mathcal{L}_{graph}
```

The terms are instantiated as follows:

- `L_mask`: Dice plus focal-style binary supervision on the dense mask
- `L_skel`: Dice loss on the sparse centerline target
- `L_topo`: clDice-style topology loss computed from differentiable soft skeletons
- `L_node`: focal loss on junction and endpoint heatmaps
- `L_aff`: masked regression loss on the local tangent field
- `L_unc`: uncertainty calibration loss
- `L_graph`: reserved hook for explicit graph supervision when graph labels are available

In the present V1 scaffold, `L_graph` is included as an architectural hook but defaults to zero when direct graph labels are not provided.

## 6. Training Strategy

The recommended development sequence is:

1. Train on 2D summary maps with dense mask and skeleton supervision.
2. Add topology-aware loss to improve branch continuity.
3. Add node heatmaps for junction and endpoint localization.
4. Add the affinity head and graph-builder-based evaluation.
5. Replace the heuristic graph stage with a learned edge refiner if the dataset scale permits it.

This staged approach keeps the first model publishable while leaving room for stronger graph-centric contributions later.

## 7. Evaluation

Pixel overlap alone is insufficient for this problem. The model should be evaluated with both segmentation and topology-sensitive metrics:

- Dice and IoU for dense mask quality
- skeleton Dice or centerline F1
- clDice for topology preservation
- junction and endpoint F1
- branch count error
- graph edit distance
- total neurite length error
- connected-component error

When possible, the strongest validation is downstream: measure whether the sparse output reduces manual proofreading effort or improves connectome extraction quality.
