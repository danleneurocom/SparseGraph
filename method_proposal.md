# TopoSparseNet

## A topology-aware dense-to-sparse model for Ca2+ neuronal microscopy

### Problem statement
Current biomedical vision models mostly learn `sparse -> dense` mappings, such as vessel enhancement or topology-aware dense segmentation. In this project, the desired direction is the reverse: given dense Ca2+ microscopy observations of neuronal structures, predict a sparse 2D representation that preserves neuronal topology and is easier to convert into a connectome-ready graph.

The target output is not just a binary mask. It is a sparse structural representation:

- neurite centerlines
- branch points
- endpoints
- local connectivity cues
- an exportable node-edge graph

## Proposed method
We propose `TopoSparseNet`, a multitask topology-aware network that predicts dense and sparse representations jointly, but treats the sparse graph as the primary output.

```text
Input Ca2+ microscopy
movie clip or 2D summary maps
    |
    v
Temporal-spatial encoder
3D CNN / 2.5D CNN / HR-U-Net backbone
    |
    v
Shared high-resolution decoder
    |
    +------------------> Dense neurite mask head
    |
    +------------------> Skeleton / centerline head
    |
    +------------------> Junction heatmap head
    |
    +------------------> Endpoint heatmap head
    |
    +------------------> Tangent / affinity field head
    |
    +------------------> Uncertainty head
    |
    v
Differentiable graph builder
node detection + edge proposal + gap bridging
    |
    v
Graph refinement module
small GNN or edge classifier
    |
    v
Sparse 2D neurite graph
for connectome mapping
```

## Architecture details

### 1. Input representation
If full movies are available, use short temporal clips centered on stable activity windows. If only 2D processing is practical, convert each clip into a compact multi-channel summary image:

- mean projection
- max projection
- standard deviation map
- local correlation map
- temporal activity map

This is important because Ca2+ imaging contains temporal cues that help separate active structure from background haze.

### 2. Backbone
Use a high-resolution `U-Net`-style encoder-decoder with residual blocks and limited pooling. Thin neurites are easily lost when the network downsamples too aggressively, so the backbone should preserve fine spatial detail. A practical starting point is:

- residual HR-U-Net or nnU-Net-style backbone
- 4 resolution levels at most
- strided convolutions only where necessary
- light attention block at the bottleneck

### 3. Multitask prediction heads
The network should emit several complementary outputs:

- `Dense mask head`: predicts full neurite occupancy
- `Skeleton head`: predicts a one-pixel-wide sparse centerline map
- `Junction head`: predicts branch-point heatmaps
- `Endpoint head`: predicts neurite endpoints
- `Affinity head`: predicts local continuation direction or pairwise edge affinity
- `Uncertainty head`: highlights regions with crossings, blur, or ambiguous overlap

This follows the same principle that made topology-aware vessel models effective: topology is easier to learn when the network is asked to predict explicit structural intermediates instead of only a dense mask.

### 4. Graph construction
After dense prediction, convert the sparse outputs into a graph:

1. Non-maximum suppress the junction and endpoint heatmaps to obtain candidate nodes.
2. Trace local centerline segments from the skeleton map.
3. Use the affinity or tangent field to connect nearby nodes through plausible paths.
4. Reject unlikely edges with a small graph neural network or edge MLP.
5. Export the final sparse output as a node-edge graph, optionally with branch length and confidence.

This graph layer is the key difference from standard segmentation pipelines. The final product is already close to the representation needed downstream for connectomics.

## Training objective
Train the model jointly with dense, sparse, and graph-level supervision:

```math
\mathcal{L}_{total} =
\lambda_{mask}\mathcal{L}_{mask} +
\lambda_{skel}\mathcal{L}_{skel} +
\lambda_{topo}\mathcal{L}_{topo} +
\lambda_{node}\mathcal{L}_{node} +
\lambda_{aff}\mathcal{L}_{aff} +
\lambda_{graph}\mathcal{L}_{graph} +
\lambda_{unc}\mathcal{L}_{unc}
```

Recommended components:

- `L_mask`: Dice + BCE or Dice + Focal loss on dense neurite segmentation
- `L_skel`: Dice or soft Dice on the centerline target
- `L_topo`: `clDice` or a differentiable skeleton-based topology loss
- `L_node`: focal loss on junction and endpoint heatmaps
- `L_aff`: cosine or regression loss on local tangent vectors, or BCE on edge affinities
- `L_graph`: binary edge classification loss over candidate node pairs, optionally with path consistency regularization
- `L_unc`: uncertainty calibration loss so low-confidence regions can be flagged during graph extraction

A good starting weighting is:

```text
lambda_mask  = 1.0
lambda_skel  = 1.0
lambda_topo  = 0.5
lambda_node  = 0.5
lambda_aff   = 0.5
lambda_graph = 1.0
lambda_unc   = 0.1
```

These values should be tuned after the first pilot experiments.

## Why this architecture is a strong fit

### 1. It matches the real output you want
The scientific goal is not just segmenting neurons. It is producing a sparse map that preserves branch structure and is easier to convert into a connectome.

### 2. It borrows the right idea from topology-aware vessel work
The strongest transferable lesson from recent vascular papers is not the exact backbone. It is the use of explicit skeleton learning and topology-aware objectives. That idea transfers well to neurites, which are also thin, branching, and connectivity-sensitive.

### 3. It is safer than direct image-to-graph prediction
An end-to-end transformer that predicts the full graph directly is interesting, but it is higher risk and usually more data-hungry. The proposed multitask route is more practical for a first publishable model.

### 4. It handles imperfect annotation better
If dense masks are expensive to annotate, centerline supervision can still be useful. This is valuable because neurite tracing is often easier than pixel-perfect dense labeling.

## Suggested ablation plan
To make the paper convincing, the ablation should test whether each structural component improves topology rather than only pixel overlap.

### Baselines

- plain U-Net dense segmentation
- U-Net + skeleton head
- U-Net + skeleton head + topology loss
- full `TopoSparseNet` without graph refinement
- full `TopoSparseNet` with graph refinement

### Core ablations

1. `Dense only` vs `dense + skeleton`
   Test whether explicit sparse supervision improves neurite continuity.

2. `No topology loss` vs `clDice / differentiable skeleton loss`
   Test whether topology-aware loss reduces broken branches.

3. `No temporal channels` vs `summary-map input`
   Test whether temporal information improves sparse extraction.

4. `No affinity head` vs `affinity head`
   Test whether local direction cues improve node-to-node connectivity.

5. `No graph refinement` vs `graph refinement`
   Test whether the graph module reduces false merges and missed links.

6. `Dense target primary` vs `sparse target primary`
   Test whether treating the sparse graph as the main task is better aligned with the final connectomics objective.

## Evaluation metrics
Pixel IoU alone will be too weak for this problem. Use structural metrics:

- Dice and IoU for dense mask quality
- skeleton Dice or centerline F1
- `clDice` for topology preservation
- branch count error
- endpoint detection F1
- junction detection F1
- graph edit distance
- connectivity score or number of disconnected components
- total neurite length error
- shortest-path length error between matched branch nodes

If possible, include downstream utility:

- improvement in connectome graph extraction quality
- reduction in manual proofreading time

## Proposed research claim
The strongest paper claim is:

`We propose a topology-aware dense-to-sparse neuronal mapping network that transforms Ca2+ microscopy observations into a connectome-ready sparse 2D graph by jointly learning dense masks, centerlines, branch landmarks, and graph connectivity.`

This is safer and more defensible than claiming to be the first dense-to-sparse model in all of vision, because related work already exists in vessel centerlines and neuron reconstruction.

## Practical starting version
If you want the first implementation to be manageable, start with:

- 2D multi-channel summary-map input instead of full video
- residual U-Net backbone
- dense mask head
- skeleton head
- junction and endpoint heads
- `clDice` + Dice losses
- simple post-processing graph builder without a GNN

That gives you a clear version 1. Then add the affinity field and graph refinement as version 2.

## References

- Rougé P, Passat N, Merveille O. `Topology aware multitask cascaded U-Net for cerebrovascular segmentation`. PLOS ONE, 2024. https://doi.org/10.1371/journal.pone.0311439
- Shit S, Paetzold JC, Sekuboyina A, et al. `clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation`. CVPR, 2021. https://doi.org/10.1109/CVPR46437.2021.01629
- Menten MJ, Paetzold JC, Zimmer VA, et al. `A skeletonization algorithm for gradient-based optimization`. ICCV, 2023. https://doi.org/10.1109/ICCV51070.2023.01956
- Tetteh G, Efremov V, Forkert ND, et al. `DeepVesselNet: Vessel Segmentation, Centerline Prediction, and Bifurcation Detection in 3-D Angiographic Volumes`. Frontiers in Neuroscience, 2020. https://doi.org/10.3389/fnins.2020.592352
- Soltanian-Zadeh S, Sahingur K, Blau S, et al. `Fast and robust active neuron segmentation in two-photon calcium imaging using spatiotemporal deep learning`. PNAS, 2019. https://doi.org/10.1073/pnas.1812995116
- Wang Y, Liu J, Zha H, et al. `NeuroSeg-II: A deep learning approach for generalized neuron segmentation in two-photon Ca2+ imaging`. Frontiers in Cellular Neuroscience, 2023. https://doi.org/10.3389/fncel.2023.1127847
- Li R, Zeng T, Peng H, Ji S. `Deep Learning Segmentation of Optical Microscopy Images Improves 3-D Neuron Reconstruction`. IEEE TMI, 2017. https://doi.org/10.1109/TMI.2017.2679713
- Li Y, Wan Y, Guo X, et al. `Deep-Learning-Based Automated Neuron Reconstruction From 3D Microscopy Images Using Synthetic Training Images`. IEEE TMI, 2022. https://doi.org/10.1109/TMI.2021.3130934
- Peng J, et al. `NRTR: Neuron Reconstruction With Transformer From 3D Optical Microscopy Images`. IEEE TMI, 2024. https://doi.org/10.1109/TMI.2023.3323466
- Chen J, Zhang R, et al. `Quantifying morphologies of developing neuronal cells using deep learning with imperfect annotations`. IBRO Neuroscience Reports, 2024. https://doi.org/10.1016/j.ibneur.2023.12.009
