from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ARCH_LABELS = {
    "layerwise_dirac_aux": "Auxiliary",
    "layerwise_dirac_adapter": "Adapter",
    "layerwise_dirac_residual": "Residual",
}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def save_accuracy_by_arch(root: Path, fig_dir: Path) -> None:
    rows = read_csv(root / "architecture_summary.csv")
    if not rows:
        return
    names = [ARCH_LABELS.get(row["architecture"], row["architecture"]) for row in rows]
    held = [f(row, "held_out") for row in rows]
    wrap = [f(row, "wrap") for row in rows]
    nowrap = [f(row, "no_wrap") for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    plt.figure(figsize=(8, 4))
    plt.bar(x - width, held, width, label="held out")
    plt.bar(x, wrap, width, label="wrap")
    plt.bar(x + width, nowrap, width, label="no wrap")
    plt.xticks(x, names)
    plt.ylim(0, 1.02)
    plt.ylabel("accuracy")
    plt.title("Layerwise Architectures")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "qnn_architecture_accuracy.png", dpi=180)
    plt.close()


def component_rows(root: Path, arch_dir: str) -> list[dict[str, Any]]:
    return read_csv(root / arch_dir / "component_metrics.csv")


def save_layerwise_accuracy(root: Path, fig_dir: Path) -> None:
    arch_dirs = {
        "Auxiliary": "modular_addition_qnn_mod97_layerwise_dirac_aux_scratch",
        "Adapter": "modular_addition_qnn_mod97_layerwise_dirac_adapter_scratch",
        "Residual": "modular_addition_qnn_mod97_layerwise_dirac_residual_scratch",
    }
    variants = ["layer_0_full", "layer_1_full", "layer_2_full", "layer_3_full", "final_full", "layer_uniform_mean"]
    labels = ["L0", "L1", "L2", "L3", "Final", "Mean"]
    plt.figure(figsize=(8, 4.5))
    for name, directory in arch_dirs.items():
        rows = [row for row in component_rows(root, directory) if row.get("split") == "held_out"]
        by_variant = {row["variant"]: row for row in rows}
        y = [f(by_variant.get(v, {}), "accuracy") for v in variants]
        plt.plot(labels, y, marker="o", linewidth=2, label=name)
    plt.ylim(0, 1.02)
    plt.ylabel("held-out accuracy")
    plt.title("Layerwise Readouts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "qnn_layerwise_accuracy.png", dpi=180)
    plt.close()


def save_frequency_cutoff(root: Path, fig_dir: Path) -> None:
    specs = {
        "Auxiliary final": ("modular_addition_qnn_mod97_layerwise_dirac_aux_scratch", "final_k"),
        "Adapter final": ("modular_addition_qnn_mod97_layerwise_dirac_adapter_scratch", "final_k"),
        "Residual layer 3": ("modular_addition_qnn_mod97_layerwise_dirac_residual_scratch", "layer_3_k"),
    }
    plt.figure(figsize=(8, 4.5))
    for label, (directory, prefix) in specs.items():
        rows = [
            row
            for row in read_csv(root / directory / "cutoff_metrics.csv")
            if row.get("split") == "held_out" and row.get("variant", "").startswith(prefix)
        ]
        parsed = []
        for row in rows:
            try:
                k = int(row["variant"].split("k")[-1])
            except ValueError:
                continue
            parsed.append((k, f(row, "accuracy"), f(row, "within_1")))
        parsed.sort()
        if parsed:
            plt.plot([x[0] for x in parsed], [x[1] for x in parsed], marker="o", linewidth=2, label=label)
    plt.ylim(0, 1.02)
    plt.xlabel("maximum Fourier frequency")
    plt.ylabel("held-out exact accuracy")
    plt.title("Frequency Sharpening")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "qnn_frequency_cutoff.png", dpi=180)
    plt.close()


def save_offset_plot(root: Path, fig_dir: Path) -> None:
    specs = {
        "Auxiliary": "modular_addition_qnn_mod97_layerwise_dirac_aux_scratch",
        "Adapter": "modular_addition_qnn_mod97_layerwise_dirac_adapter_scratch",
        "Residual": "modular_addition_qnn_mod97_layerwise_dirac_residual_scratch",
    }
    offsets = [-2, -1, 0, 1, 2]
    x = np.arange(len(offsets))
    width = 0.22
    plt.figure(figsize=(8, 4.5))
    for i, (label, directory) in enumerate(specs.items()):
        rows = [
            row
            for row in read_csv(root / directory / "offset_summary.csv")
            if row.get("variant") == "final_full"
        ]
        by_offset = {int(float(row["signed_offset"])): f(row, "fraction") for row in rows}
        y = [by_offset.get(offset, 0.0) for offset in offsets]
        plt.bar(x + (i - 1) * width, y, width, label=label)
    plt.xticks(x, [str(o) for o in offsets])
    plt.yscale("log")
    plt.xlabel("signed prediction offset")
    plt.ylabel("held-out fraction")
    plt.title("Adjacent Errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "qnn_error_offsets.png", dpi=180)
    plt.close()


def save_adapter_ablation(root: Path, fig_dir: Path) -> None:
    rows = read_csv(root / "modular_addition_qnn_mod97_layerwise_dirac_adapter_scratch" / "adapter_disabled_metrics.csv")
    if not rows:
        return
    held = [row for row in rows if row.get("split") == "held_out"]
    if not held:
        return
    disabled = f(held[0], "accuracy")
    comp = [
        row
        for row in component_rows(root, "modular_addition_qnn_mod97_layerwise_dirac_adapter_scratch")
        if row.get("split") == "held_out" and row.get("variant") == "final_full"
    ]
    enabled = f(comp[0], "accuracy") if comp else float("nan")
    plt.figure(figsize=(5, 4))
    plt.bar(["enabled", "disabled"], [enabled, disabled], color=["#4c78a8", "#f58518"])
    plt.ylim(0, 1.02)
    plt.ylabel("held-out accuracy")
    plt.title("Adapter Feedback")
    plt.tight_layout()
    plt.savefig(Path(fig_dir) / "qnn_adapter_ablation.png", dpi=180)
    plt.close()


def save_sweep_status(analysis_dir: Path, fig_dir: Path) -> None:
    rows = read_csv(analysis_dir / "qnn_publication_sweep" / "sweep_status.csv")
    completed = [row for row in rows if row.get("status") == "complete"]
    if not completed:
        return
    labels = [f"{row['kind']}\nmod{row['modulus']}\n{row['architecture']}\ns{row['seed']}" for row in completed]
    vals = [f(row, "best_test_accuracy") for row in completed]
    plt.figure(figsize=(max(8, 0.45 * len(labels)), 4))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylim(0, 1.02)
    plt.ylabel("best held-out accuracy")
    plt.title("Completed Sweep Runs")
    plt.tight_layout()
    plt.savefig(fig_dir / "qnn_publication_sweep_completed.png", dpi=180)
    plt.close()


def save_sweep_aggregate(analysis_dir: Path, fig_dir: Path) -> None:
    rows = read_csv(analysis_dir / "qnn_publication_sweep" / "sweep_aggregate.csv")
    rows.extend(read_csv(analysis_dir / "qnn_layerwise_mean_sweep" / "sweep_aggregate.csv"))
    selected = [
        ("qnn", "aux", "QNN aux"),
        ("qnn", "adapter", "QNN adapter"),
        ("qnn", "residual", "QNN residual"),
        ("qnn", "mean", "QNN mean"),
        ("classical", "fourier_delta_matched", "Fourier delta"),
        ("classical", "fourier_product_delta", "Product Fourier"),
    ]
    if not rows:
        return
    moduli = sorted({int(row["modulus"]) for row in rows if row.get("modulus")})
    x = np.arange(len(moduli))
    width = 0.14
    plt.figure(figsize=(10, 4.8))
    for idx, (kind, arch, label) in enumerate(selected):
        y = []
        err = []
        by_modulus = {
            int(row["modulus"]): row
            for row in rows
            if row.get("kind") == kind and row.get("architecture") == arch
        }
        for modulus in moduli:
            row = by_modulus.get(modulus, {})
            y.append(f(row, "heldout_mean"))
            err.append(f(row, "heldout_std", 0.0))
        plt.errorbar(
            x + (idx - (len(selected) - 1) / 2) * width,
            y,
            yerr=err,
            fmt="o",
            capsize=3,
            linewidth=2,
            markersize=6,
            label=label,
        )
    plt.xticks(x, [str(m) for m in moduli])
    plt.ylim(-0.02, 1.04)
    plt.xlabel("modulus")
    plt.ylabel("held-out accuracy")
    plt.title("Seed Sweep")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "qnn_seed_modulus_sweep.png", dpi=180)
    plt.close()


def write_story(root: Path, out_dir: Path) -> None:
    arch_rows = read_csv(root / "architecture_summary.csv")
    arch_table = []
    for row in arch_rows:
        arch_table.append(
            f"| {ARCH_LABELS.get(row['architecture'], row['architecture'])} | {f(row, 'held_out'):.6f} | "
            f"{f(row, 'within_1'):.6f} | {f(row, 'mod_add_r2'):.6f} |"
        )
    aggregate_rows = read_csv(root.parent / "qnn_publication_sweep" / "sweep_aggregate.csv")
    aggregate_rows.extend(read_csv(root.parent / "qnn_layerwise_mean_sweep" / "sweep_aggregate.csv"))
    selected = {
        ("qnn", "aux"),
        ("qnn", "adapter"),
        ("qnn", "residual"),
        ("qnn", "mean"),
        ("classical", "fourier_delta_matched"),
        ("classical", "fourier_product_delta"),
    }
    sweep_table = []
    for row in aggregate_rows:
        if (row.get("kind"), row.get("architecture")) not in selected:
            continue
        sweep_table.append(
            f"| {row.get('kind', '')} | {row.get('modulus', '')} | {row.get('architecture', '')} | "
            f"{row.get('seeds_completed', '')} | {f(row, 'heldout_mean'):.6f} | "
            f"{f(row, 'heldout_std'):.6f} | {f(row, 'heldout_min'):.6f} | {f(row, 'heldout_max'):.6f} |"
        )
    lines = [
        "# QNN Publication Story",
        "",
        "## Short Claim",
        "",
        "Data-reuploading QNNs learn a smooth cyclic representation of modular addition. The hard part is sharpening that representation into exact finite-residue decisions.",
        "",
        "## Current Evidence",
        "",
        "| architecture | held-out exact | within one | mod-add R2 |",
        "| --- | ---: | ---: | ---: |",
        *arch_table,
        "",
        "The residual layerwise QNN is the best scratch architecture. The auxiliary model contains a stronger layer-ensemble solution than its final head, and the adapter intervention shows that residue feedback is causally used in that model while still producing a worse algorithm.",
        "",
        "## Seed and Modulus Sweep",
        "",
        "| kind | modulus | architecture | seeds | held-out mean | held-out std | min | max |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        *sweep_table,
        "",
        "The p31 QNNs overfit the training split but generalize poorly. The p97 and p127 QNNs are strong across seeds. The fixed layerwise mean is the strongest QNN variant in this sweep at p97 and p127. The matched Fourier-delta classical baseline is weak at p31, good at p97, and near-perfect at p127, so the QNN result is not a quantum-advantage claim.",
        "",
        "## Figure Index",
        "",
        "- `qnn_architecture_accuracy.png`",
        "- `qnn_layerwise_accuracy.png`",
        "- `qnn_frequency_cutoff.png`",
        "- `qnn_error_offsets.png`",
        "- `qnn_adapter_ablation.png`",
        "- `qnn_seed_modulus_sweep.png`",
        "- `qnn_publication_sweep_completed.png` if sweep runs exist",
        "",
        "## Claim Boundary",
        "",
        "This is not evidence for quantum advantage. The selected sweep includes seed and modulus controls plus matched classical Fourier baselines, and those baselines show that Fourier-feature classical models can match or exceed the QNNs when given enough structure.",
        "",
    ]
    (out_dir / "QNN_PUBLICATION_STORY.md").write_text("\n".join(lines), encoding="utf-8")


def generate(args: argparse.Namespace) -> None:
    analysis_dir = Path(args.analysis_dir)
    root = analysis_dir / "qnn_layerwise_dirac_exhaustive"
    fig_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    save_accuracy_by_arch(root, fig_dir)
    save_layerwise_accuracy(root, fig_dir)
    save_frequency_cutoff(root, fig_dir)
    save_offset_plot(root, fig_dir)
    save_adapter_ablation(root, fig_dir)
    save_sweep_status(analysis_dir, fig_dir)
    save_sweep_aggregate(analysis_dir, fig_dir)
    write_story(root, fig_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create consolidated QNN publication figures and short story.")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--out-dir", default="figures/qnn_publication")
    return parser.parse_args()


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
