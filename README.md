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
