from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
FIGURES_DIR = DOCS_DIR / "figures"


@dataclass(frozen=True)
class BenchmarkRun:
    key: str
    label: str
    report_path: Path


@dataclass(frozen=True)
class QualitativeRun:
    key: str
    label: str
    metadata_path: Path
    image_path: Path


BENCHMARK_RUNS = [
    BenchmarkRun(
        key="shared_unet",
        label="Shared U-Net",
        report_path=ROOT / "runs/toposparsenet_axon_shared_unet_mps/best_val_report.json",
    ),
    BenchmarkRun(
        key="relay",
        label="Relay",
        report_path=ROOT / "runs/toposparsenet_axon_relay_mps/best_val_report.json",
    ),
    BenchmarkRun(
        key="relay_geodesic",
        label="Relay + Geodesic",
        report_path=ROOT / "runs/toposparsenet_axon_relay_geodesic_mps/best_val_report.json",
    ),
    BenchmarkRun(
        key="policy",
        label="Policy",
        report_path=ROOT / "runs/toposparsenet_axon_policy_mps/best_val_report.json",
    ),
    BenchmarkRun(
        key="policy_dense2sparse",
        label="Policy + D2S",
        report_path=ROOT / "runs/toposparsenet_axon_policy_dense2sparse_mps/best_val_report.json",
    ),
    BenchmarkRun(
        key="policy_pruneaware",
        label="Policy + D2S + Prune",
        report_path=ROOT / "runs/toposparsenet_axon_policy_pruneaware_mps/best_val_report.json",
    ),
    BenchmarkRun(
        key="policy_nodeaware",
        label="Policy + Node-Aware",
        report_path=ROOT / "runs/toposparsenet_axon_policy_nodeaware_mps/best_val_report.json",
    ),
    BenchmarkRun(
        key="policy_nodeaware_refined",
        label="Policy + Refined",
        report_path=ROOT / "runs/toposparsenet_axon_policy_nodeaware_refined_mps/best_val_report.json",
    ),
]

ABLATION_ROWS = [
    {
        "id": "A0",
        "variant": "Shared U-Net",
        "delta": "Baseline segmentation-only shared decoder",
        "key": "shared_unet",
    },
    {
        "id": "A1",
        "variant": "+ Relay / Geodesic",
        "delta": "Add relay reasoning and geodesic relation decoding",
        "key": "relay_geodesic",
    },
    {
        "id": "A2",
        "variant": "+ Unified Policy",
        "delta": "Add unified causal policy with keep/prune routing",
        "key": "policy",
    },
    {
        "id": "A3",
        "variant": "+ Dense-to-Sparse Projection",
        "delta": "Inject dense-to-sparse projection prior into final skeleton prediction",
        "key": "policy_dense2sparse",
    },
    {
        "id": "A4",
        "variant": "+ Prune-Aware Supervision",
        "delta": "Train branch keep/prune heads against redundant graph paths",
        "key": "policy_pruneaware",
    },
    {
        "id": "A5",
        "variant": "+ Node Heatmaps / Degree",
        "delta": "Use Gaussian node supervision and node-degree prediction",
        "key": "policy_nodeaware",
    },
    {
        "id": "A6",
        "variant": "+ Refined Degree-Aware Austerity",
        "delta": "Tighten degree-aware pruning around endpoints and low-capacity nodes",
        "key": "policy_nodeaware_refined",
    },
]

QUALITATIVE_RUNS = [
    QualitativeRun(
        key="policy_dense2sparse",
        label="Policy + D2S reviewer extraction",
        metadata_path=ROOT / "runs/dense_extract_reviewer_mapping/metadata.json",
        image_path=ROOT / "runs/dense_extract_reviewer_mapping/mapping_storyboard.png",
    ),
    QualitativeRun(
        key="policy_nodeaware_refined",
        label="Node-aware refined reviewer extraction",
        metadata_path=ROOT / "runs/dense_extract_reviewer_nodeaware_refined/metadata.json",
        image_path=ROOT / "runs/dense_extract_reviewer_nodeaware_refined/mapping_storyboard.png",
    ),
]

EXTERNAL_REFERENCES = [
    {
        "reference": "Axon dataset tracing paper",
        "dataset": "Public single-neuron axon dataset",
        "reported_metrics": "Precision 0.82, Recall 0.97, F1 0.87",
        "protocol_note": "3D tracing/tree protocol; not directly comparable to our 2D proxy evaluation",
    },
    {
        "reference": "FISBe / PatchPerPix",
        "dataset": "FISBe benchmark",
        "reported_metrics": "avF1 0.29-0.34, clDice_TP 0.80-0.81",
        "protocol_note": "Official FISBe instance protocol; different target and metric suite",
    },
    {
        "reference": "Topology-aware cascaded U-Net",
        "dataset": "IXI vessels",
        "reported_metrics": "DSC 0.83, clDice 0.88",
        "protocol_note": "Medical vessel segmentation; relevant topology reference, different domain",
    },
]


def ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(report_path: Path) -> dict[str, float]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    return data["metrics"]


def load_metadata(metadata_path: Path) -> dict[str, float]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def round_metric(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_benchmark_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in BENCHMARK_RUNS:
        metrics = load_metrics(run.report_path)
        rows.append(
            {
                "key": run.key,
                "label": run.label,
                "publish_score": round_metric(metrics.get("publish_score")),
                "mask_f1": round_metric(metrics.get("mask_f1")),
                "skeleton_f1": round_metric(metrics.get("skeleton_f1")),
                "cldice": round_metric(metrics.get("cldice")),
                "junction_f1": round_metric(metrics.get("junction_f1")),
                "endpoint_f1": round_metric(metrics.get("endpoint_f1")),
                "graph_accuracy": round_metric(metrics.get("graph_accuracy")),
                "graph_recall": round_metric(metrics.get("graph_recall")),
                "component_error": round_metric(metrics.get("component_error")),
                "branch_count_error": round_metric(metrics.get("branch_count_error")),
                "length_error": round_metric(metrics.get("length_error")),
                "graph_surplus_prob": round_metric(metrics.get("graph_surplus_prob")),
            }
        )
    return rows


def build_ablation_rows(benchmark_lookup: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in ABLATION_ROWS:
        metrics = benchmark_lookup[item["key"]]
        rows.append(
            {
                "id": item["id"],
                "variant": item["variant"],
                "delta": item["delta"],
                "publish_score": metrics["publish_score"],
                "skeleton_f1": metrics["skeleton_f1"],
                "junction_f1": metrics["junction_f1"],
                "endpoint_f1": metrics["endpoint_f1"],
                "branch_count_error": metrics["branch_count_error"],
                "length_error": metrics["length_error"],
            }
        )
    return rows


def build_qualitative_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in QUALITATIVE_RUNS:
        metadata = load_metadata(run.metadata_path)
        raw_nodes = int(metadata.get("raw_num_nodes", 0) or 0)
        raw_edges = int(metadata.get("raw_num_edges", 0) or 0)
        confirmed_nodes = int(metadata.get("num_nodes", 0) or 0)
        confirmed_edges = int(metadata.get("num_edges", 0) or 0)
        rows.append(
            {
                "key": run.key,
                "label": run.label,
                "raw_nodes": raw_nodes,
                "raw_edges": raw_edges,
                "confirmed_nodes": confirmed_nodes,
                "confirmed_edges": confirmed_edges,
                "candidate_bridges": int(metadata.get("num_candidate_bridges", 0) or 0),
                "node_reduction_pct": round_metric(
                    100.0 * (raw_nodes - confirmed_nodes) / max(raw_nodes, 1),
                    digits=2,
                ),
                "edge_reduction_pct": round_metric(
                    100.0 * (raw_edges - confirmed_edges) / max(raw_edges, 1),
                    digits=2,
                ),
                "mean_confidence_graph": round_metric(metadata.get("mean_confidence_graph")),
                "mean_edge_score": round_metric(metadata.get("mean_edge_score")),
                "mean_candidate_bridge_score": round_metric(metadata.get("mean_candidate_bridge_score")),
            }
        )
    return rows


def plot_benchmark_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["key"]
        in {
            "shared_unet",
            "relay_geodesic",
            "policy_dense2sparse",
            "policy_pruneaware",
            "policy_nodeaware",
            "policy_nodeaware_refined",
        }
    ]
    labels = [str(row["label"]) for row in selected]
    colors = []
    for row in selected:
        key = str(row["key"])
        if key == "shared_unet":
            colors.append("#8f96a3")
        elif key == "policy_dense2sparse":
            colors.append("#2b6cb0")
        elif key == "policy_nodeaware_refined":
            colors.append("#d97706")
        else:
            colors.append("#5b8c5a")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=180)
    metric_specs = [
        ("publish_score", "Publish Score", False),
        ("skeleton_f1", "Skeleton F1", False),
        ("junction_f1", "Junction F1", False),
        ("branch_count_error", "Branch Count Error", True),
    ]
    for ax, (metric, title, invert) in zip(axes.flat, metric_specs):
        values = [float(row[metric]) for row in selected]
        ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
        if invert:
            ax.invert_yaxis()
            ax.set_ylabel("Lower is better")
        else:
            ax.set_ylabel("Higher is better")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    fig.suptitle("Axon Benchmark Summary", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=180)
    for row in rows:
        x = float(row["junction_f1"])
        y = float(row["branch_count_error"])
        score = float(row["publish_score"])
        size = 220 + 520 * max(score - 0.40, 0.0)
        key = str(row["key"])
        color = "#d97706" if key == "policy_nodeaware_refined" else "#2b6cb0" if key == "policy_dense2sparse" else "#5b8c5a"
        if key == "shared_unet":
            color = "#8f96a3"
        ax.scatter(x, y, s=size, color=color, alpha=0.82, edgecolor="black", linewidth=0.7)
        ax.text(x + 0.002, y + 24.0, str(row["label"]), fontsize=8)

    ax.set_xlabel("Junction F1 (higher is better)")
    ax.set_ylabel("Branch Count Error (lower is better)")
    ax.set_title("Node Quality vs Branch Control Trade-off")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def compose_qualitative_panel(rows: list[dict[str, object]], output_path: Path) -> None:
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    sections: list[tuple[str, Image.Image, list[str]]] = []
    for run in QUALITATIVE_RUNS:
        row = next(item for item in rows if item["key"] == run.key)
        image = Image.open(run.image_path).convert("RGB")
        lines = [
            f"Confirmed nodes/edges: {row['confirmed_nodes']} / {row['confirmed_edges']}",
            f"Raw nodes/edges: {row['raw_nodes']} / {row['raw_edges']}",
            f"Candidate bridges: {row['candidate_bridges']}",
            f"Graph confidence: {row['mean_confidence_graph']}",
            f"Edge score: {row['mean_edge_score']}",
        ]
        if row["mean_candidate_bridge_score"] is not None:
            lines.append(f"Candidate score: {row['mean_candidate_bridge_score']}")
        sections.append((run.label, image, lines))

    section_width = 1440
    padded_images: list[Image.Image] = []
    for _, image, _ in sections:
        scale = min(section_width / image.width, 1.0)
        resized = image.resize((int(image.width * scale), int(image.height * scale)))
        padded_images.append(resized)

    draw_probe = ImageDraw.Draw(Image.new("RGB", (10, 10), "white"))
    margin = 36
    gap = 28
    title_height = _text_size(draw_probe, "X", title_font)[1] + 12
    line_height = _text_size(draw_probe, "X", font)[1] + 6
    canvas_width = section_width + 2 * margin
    canvas_height = margin
    for (_, image, lines), padded_image in zip(sections, padded_images):
        canvas_height += title_height + padded_image.height + 12 + len(lines) * line_height + gap
    canvas_height += margin

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    y = margin
    header = "Dense-to-sparse extraction on a dense validation sample"
    draw.text((margin, y), header, fill="black", font=title_font)
    y += title_height + 10

    for (label, _, lines), image in zip(sections, padded_images):
        draw.text((margin, y), label, fill="black", font=title_font)
        y += title_height
        x = margin + (section_width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + 12
        for line in lines:
            draw.text((margin, y), line, fill="black", font=font)
            y += line_height
        y += gap

    canvas.save(output_path)


def main() -> None:
    ensure_dirs()

    benchmark_rows = build_benchmark_rows()
    benchmark_lookup = {str(row["key"]): row for row in benchmark_rows}
    ablation_rows = build_ablation_rows(benchmark_lookup)
    qualitative_rows = build_qualitative_rows()

    write_csv(
        ASSETS_DIR / "axon_benchmark.csv",
        benchmark_rows,
        [
            "key",
            "label",
            "publish_score",
            "mask_f1",
            "skeleton_f1",
            "cldice",
            "junction_f1",
            "endpoint_f1",
            "graph_accuracy",
            "graph_recall",
            "component_error",
            "branch_count_error",
            "length_error",
            "graph_surplus_prob",
        ],
    )
    write_csv(
        ASSETS_DIR / "axon_ablation.csv",
        ablation_rows,
        [
            "id",
            "variant",
            "delta",
            "publish_score",
            "skeleton_f1",
            "junction_f1",
            "endpoint_f1",
            "branch_count_error",
            "length_error",
        ],
    )
    write_csv(
        ASSETS_DIR / "extraction_graph_stats.csv",
        qualitative_rows,
        [
            "key",
            "label",
            "raw_nodes",
            "raw_edges",
            "confirmed_nodes",
            "confirmed_edges",
            "candidate_bridges",
            "node_reduction_pct",
            "edge_reduction_pct",
            "mean_confidence_graph",
            "mean_edge_score",
            "mean_candidate_bridge_score",
        ],
    )
    write_csv(
        ASSETS_DIR / "external_reference_bars.csv",
        EXTERNAL_REFERENCES,
        ["reference", "dataset", "reported_metrics", "protocol_note"],
    )

    plot_benchmark_summary(benchmark_rows, FIGURES_DIR / "axon_benchmark_summary.png")
    plot_tradeoff(benchmark_rows, FIGURES_DIR / "axon_tradeoff.png")
    compose_qualitative_panel(qualitative_rows, FIGURES_DIR / "extraction_qualitative.png")

    print(f"Wrote benchmark CSV: {ASSETS_DIR / 'axon_benchmark.csv'}")
    print(f"Wrote ablation CSV: {ASSETS_DIR / 'axon_ablation.csv'}")
    print(f"Wrote extraction CSV: {ASSETS_DIR / 'extraction_graph_stats.csv'}")
    print(f"Wrote benchmark figure: {FIGURES_DIR / 'axon_benchmark_summary.png'}")
    print(f"Wrote tradeoff figure: {FIGURES_DIR / 'axon_tradeoff.png'}")
    print(f"Wrote qualitative figure: {FIGURES_DIR / 'extraction_qualitative.png'}")


if __name__ == "__main__":
    main()
