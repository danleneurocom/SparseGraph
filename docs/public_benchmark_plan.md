# Public Benchmark Plan

This repo should be evaluated in two stages if the goal is a publishable paper and later wet-lab deployment.

## Stage 1: Strong supervised public benchmarks

Use labeled public datasets where we can report clear quantitative gains over strong baselines.

### 1. Neurofinder

Why use it:

- public calcium-imaging benchmark
- established evaluation culture for neuron finding
- directly relevant to later Ca2 deployment

What to report:

- mask F1 / precision / recall
- Neurofinder-style region metrics if exported as regions
- calibration and uncertainty maps for deployment triage

Limitation:

- mostly soma-level rather than branching-topology supervision

### 2. FISBe

Why use it:

- public neuron microscopy benchmark with pixel-wise instance masks
- good fit for thin-structure and morphology generalization
- stronger structural challenge than cell-body-only Ca2 datasets

What to report:

- mask Dice / F1
- clDice
- topology proxy metrics from this repo
- cross-split generalization

### 3. Axon reconstruction dataset

Why use it:

- public microscopy dataset with explicit axonal reconstruction focus
- closer to connectome-style sparse structure extraction
- useful for demonstrating dense-to-sparse structural recovery

What to report:

- mask Dice / F1
- skeleton F1
- clDice
- component error, branch-count error, and length error

## Stage 2: Unlabeled wet-lab adaptation

Once the public benchmark model is strong, move to wet-lab Ca2 data with:

- unlabeled extraction using `scripts/extract_ca2.py`
- confidence and uncertainty triage
- optional self-training or test-time adaptation
- a small expert-reviewed validation subset whenever possible

## Publishable metric targets

These are practical targets, not hard rules:

- mask F1 at or above `0.80` on in-domain public data is strong
- clDice at or above `0.80` starts to look publication-ready
- skeleton F1 in the `0.60+` range is a healthier topology baseline than the current IXI proxy result
- component and branch-count errors should be low enough that qualitative reconstructions do not frequently fragment or over-merge

## Baseline set

At minimum compare against:

- plain U-Net or nnU-Net
- U-Net + clDice
- current dual-route TopoSparseNet
- current consensus-rollout TopoSparseNet
- dataset-specific baselines from the benchmark literature

## Current repo recommendation

- keep IXI as proxy pretraining only
- use Neurofinder for public Ca2 benchmarking
- use a structural dataset such as FISBe or the public axon dataset for topology-heavy evaluation
- treat unlabeled wet-lab deployment as a second-stage validation problem, not the only evaluation
