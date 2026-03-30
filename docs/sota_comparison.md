# TopoSparseNet vs Relevant SOTA

This note separates three different comparisons:

1. `Direct internal comparison` on the same axon benchmark pipeline used in this repo.
2. `External adjacent baselines` from published papers on closely related public datasets.
3. `Fairness caveats` about what we can and cannot currently claim.

## Internal Axon Comparison

All rows below use the same local axon benchmark pipeline in this repo:

- dataset root converted by [prepare_axon.py](/Users/lenguyenlinhdan/Desktop/SparseGraph/scripts/prepare_axon.py)
- same train/val split in `data/axon_npz`
- same publication-facing metrics from [metrics.py](/Users/lenguyenlinhdan/Desktop/SparseGraph/sparse_graph/metrics.py)

| Model | publish_score | mask_f1 | skeleton_f1 | clDice | junction_f1 | endpoint_f1 | component_error | branch_count_error | length_error | graph_accuracy | graph_recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_unet | 0.4259 | 0.7247 | 0.3231 | 0.7759 | 0.1482 | 0.0020 | 44.5882 | 6506.4000 | 4.5690 | 0.0000 | 0.0000 |
| relay | 0.4840 | 0.7298 | 0.4214 | 0.7879 | 0.2271 | 0.2268 | 67.9412 | 3361.9765 | 2.7676 | NA | NA |
| relay_geodesic | 0.5230 | 0.7855 | 0.4743 | 0.8308 | 0.2286 | 0.2902 | 40.6706 | 1160.2941 | 0.6020 | 0.7598 | 0.6692 |
| nodecausal | 0.5317 | 0.7987 | 0.4895 | 0.8399 | 0.2260 | 0.3019 | 28.4353 | 1429.5882 | 0.5119 | 0.7924 | 0.7563 |
| nodecausal_budget | 0.5490 | 0.8220 | 0.5074 | 0.8635 | 0.2450 | 0.3054 | 25.0235 | 1624.6118 | 0.4497 | 0.7958 | 0.7626 |
| nodecausal_austerity | 0.5485 | 0.8276 | 0.5117 | 0.8736 | 0.2347 | 0.2703 | 28.5647 | 1373.2235 | 0.3515 | 0.7954 | 0.7917 |
| policy | 0.5657 | 0.8392 | 0.5197 | 0.8858 | 0.2521 | 0.3470 | 25.6000 | 2080.3059 | 0.4588 | 0.7967 | 0.8012 |
| policy_dense2sparse | 0.5685 | 0.8449 | 0.5219 | 0.8888 | 0.2476 | 0.3585 | 23.4471 | 1334.8353 | 0.3036 | 0.7960 | 0.7771 |

### Readout

- `shared_unet` is the strict local baseline on the exact same split, and it is far behind the topology-aware family:
  - `publish_score 0.4259` vs `0.5685` for `policy_dense2sparse`
  - `skeleton_f1 0.3231` vs `0.5219`
  - `endpoint_f1 0.0020` vs `0.3585`
  - `branch_count_error 6506.4` vs `1334.8`
- `policy_dense2sparse` is now the best checkpoint by the repo's current composite `publish_score`.
- `policy_dense2sparse` is also the cleanest flagship mechanism because it enhances the same unified policy block with an explicit dense-to-sparse projection prior, instead of adding another separate novelty module.
- Compared with the previous `policy`, the dense-to-sparse enhancement improves the parts that matter most for the original project goal:
  - better `mask_f1`, `skeleton_f1`, and `clDice`
  - much better `branch_count_error`
  - much better `length_error`
  - better `endpoint_f1`
- `policy_dense2sparse` is now also the strongest dense-to-sparse structural model in this table:
  - best `component_error`
  - best `branch_count_error`
  - best `length_error`
  - best `endpoint_f1`
- `nodecausal_austerity` remains the clearest sparse-suppression precursor ablation:
  - still very strong on `clDice`
  - still better aligned with the final model than the older generic `policy`
- `nodecausal_budget` remains a useful component-stability ablation:
  - it explains why explicit branch budgeting helped before the dense-to-sparse prior was added

### Current Recommendation

- Use [best.pt](/Users/lenguyenlinhdan/Desktop/SparseGraph/runs/toposparsenet_axon_policy_dense2sparse_mps/best.pt) as the main paper checkpoint and flagship architecture.
- Use `shared_unet` as the direct local baseline under the same split and preprocessing.
- Keep `policy`, `nodecausal_budget`, and `nodecausal_austerity` in the paper as ablations that explain how the final dense-to-sparse policy evolved:
  - `policy` explains the consolidated causal topology block
  - `budget` explains component stability
  - `austerity` explains branch suppression and path fidelity
- Do not present these as multiple flagship "novel mechanisms." The cleaner story is:
  - `policy_dense2sparse` is the single proposed mechanism
  - `policy`, `budget`, and `austerity` are development-stage ablations that motivated it

## Relevant Published Baselines

These are the most relevant public baselines and adjacent SOTA references for the current paper direction.

### 1. Axon Reconstruction Dataset Paper

Closest external reference for the current public dataset:

- Paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12361129/>
- The paper reports that a `distance field supervised U-Net` followed by `PointTree` achieved `precision 0.82`, `recall 0.97`, and `F1 0.87` on `90 image blocks`.
- The same paper also reports cross-domain light-sheet transfer results around `F1 0.78-0.90` with `AJI 0.39-0.51`.

Why it matters:

- This is the strongest directly relevant public reference for axon tracing.
- It is still **not directly apples-to-apples** with our current numbers:
  - their pipeline is 3D reconstruction/tracing oriented
  - their evaluation is point/tree oriented
  - our current repo evaluation is a 2D projection-based dense-to-sparse extraction benchmark with topology proxies

Implication:

- We should treat the axon paper as a `target bar`, not as something we can claim to have beaten yet.

### 2. FISBe Benchmark

Strong public benchmark for long-range projection instance segmentation:

- Benchmark site: <https://kainmueller-lab.github.io/fisbe/>
- Paper: <https://openaccess.thecvf.com/content/CVPR2024/html/Mais_FISBe_A_Real-World_Benchmark_Dataset_for_Instance_Segmentation_of_Long-Range_CVPR_2024_paper.html>
- The public benchmark page reports `PatchPerPix` around `avF1 0.29-0.34` and `clDice_TP 0.80-0.81` across FISBe subsets.

Why it matters:

- FISBe is a realistic, public, neurite/axon-style benchmark where topology and instance continuity matter.
- It is highly relevant for publication because it tests structural generalization beyond one dataset.

Current limitation:

- We have FISBe data prep and training scaffolding, but we are not yet running the official FISBe metric suite in this repo.
- So today we can only use FISBe as an adjacent benchmark, not yet as a direct leaderboard comparison.

### 3. Topology-Aware Cascaded U-Net

Relevant topology-aware medical baseline:

- Paper: <https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0311439>
- Search snippet and paper summary report approximately `DSC 0.83 ± 0.03` and `clDice 0.88 ± 0.03` on IXI.

Why it matters:

- It is one of the clearest examples of topology-aware supervision improving centerline fidelity on public data.
- It is a useful conceptual baseline for our topology modules, even though the task is vessel segmentation rather than neuron extraction.

Current limitation:

- This is a different imaging domain and evaluation setup.
- It is a `related topology baseline`, not a direct benchmark opponent on our axon pipeline.

## What We Can Claim Today

Safe claims:

- Our consolidated `policy_dense2sparse` model improves substantially over a strict shared-U-Net baseline on the same public axon split.
- The paper can defensibly claim one flagship mechanism:
  - a unified causal topology policy that jointly reasons about node capacity, path importance, branch keep/prune evidence, and dense-to-sparse projection
- The older `policy`, `budget`, and `austerity` variants can be framed as ablations that shaped the final policy design.

Unsafe claims:

- We cannot yet say we beat the published axon paper.
- We cannot yet say we beat FISBe SOTA.
- We should not call the current comparison directly apples-to-apples with 3D tracing papers or official FISBe leaderboard methods.

## What Would Make the Comparison Publishable

1. Add a `direct benchmark` section with the official metric protocol for at least one public dataset.
2. Evaluate our best model with the official FISBe metric suite if we want a clean public leaderboard-style comparison.
3. Add or reproduce one strong public axon baseline under our local data split, ideally:
   - U-Net style segmentation baseline
   - a PointTree-style or tracing-style baseline if feasible
4. Keep the paper story centered on one flagship architecture:
   - `policy_dense2sparse` as the proposed mechanism
   - `shared_unet` as the strict local baseline
   - `policy`, `budget`, and `austerity` as focused ablations

## Files

- Shared-U-Net baseline report: [best_val_report.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/runs/toposparsenet_axon_shared_unet_mps/best_val_report.json)
- Shared-U-Net baseline config: [toposparsenet_axon_shared_unet_mps.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/configs/toposparsenet_axon_shared_unet_mps.json)
- Dense-to-sparse policy best report: [best_val_report.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/runs/toposparsenet_axon_policy_dense2sparse_mps/best_val_report.json)
- Dense-to-sparse policy config: [toposparsenet_axon_policy_dense2sparse_mps.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/configs/toposparsenet_axon_policy_dense2sparse_mps.json)
- Policy best report: [best_val_report.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/runs/toposparsenet_axon_policy_mps/best_val_report.json)
- Policy config: [toposparsenet_axon_policy_mps.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/configs/toposparsenet_axon_policy_mps.json)
- Budget best report: [best_val_report.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/runs/toposparsenet_axon_nodecausal_budget_mps/best_val_report.json)
- Austerity best report: [best_val_report.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/runs/toposparsenet_axon_nodecausal_austerity_mps/best_val_report.json)
- Austerity config: [toposparsenet_axon_nodecausal_austerity_mps.json](/Users/lenguyenlinhdan/Desktop/SparseGraph/configs/toposparsenet_axon_nodecausal_austerity_mps.json)
- Main model: [toposparsenet.py](/Users/lenguyenlinhdan/Desktop/SparseGraph/sparse_graph/models/toposparsenet.py)
- New mechanism blocks: [blocks.py](/Users/lenguyenlinhdan/Desktop/SparseGraph/sparse_graph/models/blocks.py)
- Objective: [losses.py](/Users/lenguyenlinhdan/Desktop/SparseGraph/sparse_graph/losses.py)
