# SparseGraph

`SparseGraph` is a research scaffold for `TopoSparseNet`, a topology-aware dense-to-sparse model for Ca2+ neuronal microscopy. The goal is to transform dense calcium-imaging observations into a sparse 2D structural representation that is easier to convert into a connectome-ready graph.

This repo now includes:

- a proposal-oriented overview in [method_proposal.md](/Users/lenguyenlinhdan/Desktop/SparseGraph/method_proposal.md)
- a paper-style Method section in [docs/method_section.md](/Users/lenguyenlinhdan/Desktop/SparseGraph/docs/method_section.md)
- an implementation blueprint with module and tensor-shape details in [docs/implementation_plan.md](/Users/lenguyenlinhdan/Desktop/SparseGraph/docs/implementation_plan.md)
- a benchmark strategy note in [docs/public_benchmark_plan.md](/Users/lenguyenlinhdan/Desktop/SparseGraph/docs/public_benchmark_plan.md)
- a PyTorch training scaffold for a practical V1 model based on 2D multi-channel summary maps

## Repository layout

```text
SparseGraph/
├── configs/
│   └── toposparsenet_base.json
├── docs/
│   ├── implementation_plan.md
│   └── method_section.md
├── sparse_graph/
│   ├── config.py
│   ├── losses.py
│   ├── train.py
│   ├── utils.py
│   ├── data/
│   │   └── datasets.py
│   ├── graph/
│   │   └── builder.py
│   └── models/
│       ├── backbone.py
│       ├── blocks.py
│       ├── heads.py
│       └── toposparsenet.py
├── method_proposal.md
├── requirements.txt
└── train.py
```

## Model summary

The scaffolded V1 model predicts:

- dense neurite mask
- sparse skeleton / centerline map
- junction heatmap
- endpoint heatmap
- local 2D tangent field
- uncertainty map

The training objective combines dense segmentation loss, sparse supervision, and topology-aware loss. A simple graph builder is included for post-processing hooks and later connectomics work.

The current codebase also includes multiple topology-aware variants:

- self-conditioned topology refinement
- dual-route dense/sparse decoding
- consensus rollout refinement, where an initial structural prediction is fed back into the network to revise both dense and sparse branches before the final output

## Quickstart

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a synthetic-data training pass:

```bash
python train.py --config configs/toposparsenet_base.json --epochs 1 --train-samples 16 --val-samples 4
```

Outputs will be written to `runs/toposparsenet_base/` by default.

## Temporary dataset: `ds005096`

You can use [OpenNeuro `ds005096`](https://github.com/OpenNeuroDatasets/ds005096) as a temporary development dataset, but it should be treated as a topology and pipeline pretraining source, not as a substitute for real Ca2+ neuronal data.

What it is:

- TOF-MRA intracranial aneurysm data
- BIDS-like subject/session layout
- original angiography volumes
- derivative aneurysm and parent-artery segmentations

What it is good for:

- NIfTI I/O and preprocessing
- dense-to-sparse post-processing
- skeleton, junction, and endpoint target generation
- topology-aware training and graph-building experiments

What it is not good for:

- calcium-imaging temporal modeling
- neuron-specific morphology and branching statistics
- final model selection for your target problem

Important download note:

- cloning the GitHub repo gives you metadata and annexed placeholders
- the actual `.nii/.nii.gz` image payloads must be fetched separately from OpenNeuro or DataLad-compatible storage

Once the real volumes are present locally, convert them into the repo's `.npz` format with:

```bash
python scripts/prepare_ds005096.py \
  --dataset-root /path/to/ds005096 \
  --output-root data/ds005096_npz \
  --image-size 256
```

This creates:

- `data/ds005096_npz/train/*.npz`
- `data/ds005096_npz/val/*.npz`

Then update the config:

```json
{
  "data": {
    "dataset_type": "npz",
    "train_dir": "data/ds005096_npz/train",
    "val_dir": "data/ds005096_npz/val"
  }
}
```

## Temporary dataset: `IXI`

The local IXI MRA collection is a better temporary fit operationally than the OpenNeuro clone because it contains real `.nii.gz` volumes directly. The tradeoff is that IXI MRA is unlabeled, so it can only support weak supervision or pseudo-label development unless you add manual annotations.

Use IXI for:

- preprocessing and NIfTI pipeline development
- 3D-to-2D projection experiments
- pseudo-label experiments using vesselness filtering and skeletonization
- early graph-building and topology debugging

Do not use IXI pseudo-label performance as your final scientific result. It is a bootstrap dataset, not the target task.

Prepare IXI `.npz` samples with:

```bash
python scripts/prepare_ixi.py \
  --dataset-root /path/to/IXI-MRA \
  --output-root data/ixi_npz \
  --image-size 256
```

The script builds:

- 4-channel 2D summary images from each 3D MRA volume
- pseudo vessel masks from Frangi vesselness + thresholding
- pseudo skeleton, junction, endpoint, and affinity targets

There is also a starter config at `configs/toposparsenet_ixi_pseudo.json`.

If you are training on Apple Silicon, use the dedicated MPS config:

```bash
python train.py --config configs/toposparsenet_ixi_mps.json
```

To train the topology-refinement variant instead:

```bash
python train.py --config configs/toposparsenet_ixi_toporefine_mps.json
```

To train the stronger dual-route dense/sparse topology model:

```bash
python train.py --config configs/toposparsenet_ixi_dualroute_mps.json
```

To train the stronger iterative consensus-rollout variant:

```bash
python train.py --config configs/toposparsenet_ixi_consensus_mps.json
```

An additional experimental config, `configs/toposparsenet_ixi_biocausal_mps.json`, adds biology-aware causal routing based on morphology-vs-activity channel groups. It is the more novel mechanism, but the dual-route model is currently the strongest validated checkpoint on the IXI proxy dataset.

## Public Ca2 benchmark: `Neurofinder`

The repo now includes `scripts/prepare_neurofinder.py` to convert Neurofinder training datasets into the repo's `.npz` format.

Example:

```bash
python scripts/prepare_neurofinder.py \
  --dataset-root /path/to/neurofinder \
  --output-root data/neurofinder_npz \
  --image-size 256
```

Then train with:

```bash
python train.py --config configs/toposparsenet_neurofinder_consensus_mps.json
```

This gives us a public Ca2 benchmark track while we wait for wet-lab data. Keep in mind that Neurofinder is more soma-focused than branch-topology-focused, so it should be paired with a structural topology benchmark for the full paper.

## Public topology benchmark: `Axon`

The repo now includes `scripts/prepare_axon.py` for the public axon dataset packaged as `images.rar`, `mask.rar`, `swc.rar`, plus `train.txt`, `val.txt`, and `test.txt`.

Example:

```bash
python scripts/prepare_axon.py \
  --dataset-root /path/to/15372438 \
  --output-root data/axon_npz \
  --image-size 256
```

Then train with:

```bash
python train.py --config configs/toposparsenet_axon_consensus_mps.json
```

This benchmark is especially useful because it provides both dense masks and sparse SWC structure.

## Public topology benchmark: `FISBe`

The repo also includes `scripts/prepare_fisbe.py`, which can read either an extracted FISBe folder or the original zip archive directly through `zarr`.

Example:

```bash
python scripts/prepare_fisbe.py \
  --zip-path /path/to/fisbe_v1.0_completely.zip \
  --output-root data/fisbe_npz \
  --image-size 256
```

Then train with:

```bash
python train.py --config configs/toposparsenet_fisbe_consensus_mps.json
```

FISBe gives us a stronger structure-focused public benchmark alongside Neurofinder.

## Evaluation

Training now tracks a broader validation signal than mask Dice alone, including `clDice` and a composite `publish_score`.

To generate a standalone evaluation report for a checkpoint:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint runs/toposparsenet_ixi_dualroute_mps/best.pt
```

The report includes:

- mask precision / recall / F1
- skeleton precision / recall / F1
- `clDice`
- junction and endpoint F1
- component error
- branch-count error
- length error
- a composite `publish_score`

## Unlabeled Ca2 extraction

You can now run extraction on unlabeled Ca2 input directly. The extractor accepts:

- raw movies as `.npy`, `.npz`, `.tif`, or `.tiff`
- summary stacks as `.npy` or `.npz`

For raw movies, the script builds a 4-channel summary stack using:

- mean projection
- max projection
- temporal standard deviation
- local correlation map

Then it runs the model, applies optional flip-based test-time augmentation, and saves:

- dense and sparse probability maps
- uncertainty and confidence maps
- graph nodes and edges as JSON

Example:

```bash
python scripts/extract_ca2.py \
  --checkpoint runs/toposparsenet_ixi_pseudo/best.pt \
  --input /path/to/ca2_movie.tif \
  --output-dir runs/ca2_extraction_sample
```

Important note:

- this makes the model deployable for unlabeled extraction
- it does not by itself guarantee scientific validity on Ca2 without validation data
- use the confidence and uncertainty outputs for triage and wet-lab review

## Real-data format

The `CalciumSummaryNpzDataset` expects one `.npz` file per sample with at least:

- `image`: shape `[C, H, W]`
- `mask`: shape `[1, H, W]` or `[H, W]`
- `skeleton`: shape `[1, H, W]` or `[H, W]`
- `junction`: shape `[1, H, W]` or `[H, W]`
- `endpoint`: shape `[1, H, W]` or `[H, W]`

Optional keys:

- `affinity`: shape `[2, H, W]`
- `uncertainty`: shape `[1, H, W]` or `[H, W]`

Update the config to switch `"dataset_type"` from `"synthetic"` to `"npz"` and point `"train_dir"` and `"val_dir"` to your sample folders.

## Current scope

This is a research starter, not a finished benchmark package. The current implementation intentionally favors:

- clarity over maximal complexity
- a practical 2D summary-map pipeline before full video models
- explicit sparse supervision before a heavier end-to-end graph predictor

That makes it a good base for ablations and for the first paper iteration.
