# Results

## 1. Public Axon Benchmark

We evaluated the proposed dense-to-sparse framework on the public axon benchmark using the same local train/validation split, preprocessing pipeline, and publication-facing metric suite across all experiments. Table 1 summarizes the three strongest late-stage variants of the model. `policy_dense2sparse` is the cleanest deployment-oriented checkpoint, `minimalgraph` adds explicit supervision for essential graph edges, and `pruneaware` further penalizes redundant branch paths during training.

| Model | publish_score | mask_f1 | skeleton_f1 | clDice | junction_f1 | endpoint_f1 | graph_accuracy | graph_recall | component_error | branch_count_error | length_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| policy_dense2sparse | 0.5685 | 0.8449 | 0.5219 | 0.8888 | 0.2476 | 0.3585 | 0.7960 | 0.7771 | 23.4471 | 1334.8353 | 0.3036 |
| minimalgraph | 0.5693 | 0.8413 | 0.5270 | 0.8857 | 0.2451 | 0.3741 | 0.8280 | 0.9422 | 22.2235 | 1612.4353 | 0.3419 |
| pruneaware | 0.5702 | 0.8437 | 0.5276 | 0.8879 | 0.2459 | 0.3706 | 0.8249 | 0.9449 | 21.5765 | 1523.6118 | 0.3307 |

**Table 1.** Comparison of the strongest dense-to-sparse checkpoints on the public axon validation set.

The results show a consistent pattern. First, dense structural segmentation is already strong, with `mask_f1` around `0.84` across all three checkpoints. Second, topology preservation is one of the strongest aspects of the current system, with `clDice` close to `0.89`, indicating that the extracted sparse representation usually follows the global continuity of the underlying arbor. Third, sparse centerline extraction remains moderate rather than excellent: `skeleton_f1` improves only modestly from `0.5219` to `0.5276`, suggesting that the dense-to-sparse conversion is working, but still leaves room for more precise pruning and branch control.

The trade-off between the variants is also informative. `policy_dense2sparse` remains the cleanest deployment checkpoint because it gives the best branch-count and length behavior among the three strongest models. `minimalgraph` improves graph-level recall and essential-edge retention, but keeps too many local branches. `pruneaware` partially corrects that trend: compared with `minimalgraph`, it reduces `graph_surplus_prob`, lowers `component_error`, and improves `branch_count_error`, while preserving the best overall `publish_score`. This indicates that explicit redundant-path suppression is beneficial, but not yet sufficient to fully solve branch explosion.

## 2. Structural Interpretation

The current system is strongest at answering the question, “where is the neuronal structure and how does it continue?”, and weaker at answering, “which local branches are truly important enough to keep in the final sparse graph?” This is reflected in the metric split:

- Strong: dense mask quality, topology preservation, graph recall, and component continuity.
- Moderate: sparse centerline precision.
- Weakest: junction fidelity and branch-count control.

In practical terms, this means the model usually captures the major backbone correctly, but may still retain extra side branches or local over-segmentation around branching regions. For downstream connectome-style use, this is an important distinction: false positive connections are often more harmful than small discontinuities. As a result, the current extractor is better viewed as a conservative structural proposal generator than as a final connectome reconstruction system.

## 3. Deployment Readout

To better interpret inference quality, we also measured graph-specific confidence on a dense reviewer-style extraction example. The summary is shown below:

- `mean_confidence_all_pixels = 0.4383`
- `mean_confidence_structure = 0.8872`
- `mean_confidence_graph = 0.8767`
- `mean_edge_score = 0.8928`
- `mean_candidate_bridge_score = 0.8298`
- `num_candidate_bridges = 10`

The whole-image confidence value appears low because it averages over background as well as structure. The more meaningful quantities are `mean_confidence_graph` and `mean_edge_score`, both of which are high. This indicates that the confirmed sparse graph is considerably more reliable than the raw image-wide confidence average suggests. The newly added candidate-bridge output also provides a cleaner downstream interface: instead of forcing uncertain reconnections into the confirmed graph, the model exposes them separately as plausible mapping hypotheses.

## 4. Takeaway

The main conclusion is that the proposed dense-to-sparse framework is already strong enough to support a defensible research claim on public data: it produces accurate dense segmentation, strong topology preservation, and a practically usable sparse structural graph. However, the current limitation is equally clear. The system is not yet fully connectome-ready because local branch selection, junction accuracy, and branch-count suppression remain the dominant failure modes. The most honest paper claim at this stage is therefore:

> The proposed model is effective for topology-aware dense-to-sparse neuronal structure extraction, with strong continuity preservation and graph recall, but further improvements are required for reviewer-ready minimal graph reconstruction.

## 5. Recommended Checkpoints

- For the strongest benchmark checkpoint: [best.pt](/Users/lenguyenlinhdan/Desktop/SparseGraph/runs/toposparsenet_axon_policy_pruneaware_mps/best.pt)
- For the cleanest current reviewer-style extraction: [best.pt](/Users/lenguyenlinhdan/Desktop/SparseGraph/runs/toposparsenet_axon_policy_dense2sparse_mps/best.pt)
- For the deployment example with confirmed graph plus candidate bridges: [metadata.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/runs/dense_extract_reviewer_mapping/metadata.json)
