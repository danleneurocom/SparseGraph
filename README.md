# SparseGraph

`SparseGraph` is a research scaffold for `TopoSparseNet`, a topology-aware dense-to-sparse model for Ca2+ neuronal microscopy. The goal is to transform dense calcium-imaging observations into a sparse 2D structural representation that is easier to convert into a connectome-ready graph.

This repo now includes:

- a proposal-oriented overview in [method_proposal.md](/Users/lenguyenlinhdan/Desktop/SparseGraph/method_proposal.md)
- a paper-style Method section in [docs/method_section.md](/Users/lenguyenlinhdan/Desktop/SparseGraph/docs/method_section.md)
- an implementation blueprint with module and tensor-shape details in [docs/implementation_plan.md](/Users/lenguyenlinhdan/Desktop/SparseGraph/docs/implementation_plan.md)
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
