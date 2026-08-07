from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import MODALITIES, ModularPairsDataset
from .models import TrimodalQNNModel


SECTOR_PAIRS = (("T", "N"), ("T", "I"), ("N", "I"))


@dataclass
class CCAFit:
    mean_x: np.ndarray
    mean_y: np.ndarray
    wx: np.ndarray
    wy: np.ndarray
    train_corrs: np.ndarray


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def layer_name(idx: int) -> str:
    return "initial" if idx == 0 else f"layer_{idx}"


def load_model(run_dir: Path, checkpoint_step: int, device: torch.device) -> tuple[TrimodalQNNModel, dict[str, Any], Path]:
    cfg = load_config(run_dir / "config.yaml")
    checkpoint_path = run_dir / f"checkpoint_{checkpoint_step}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = TrimodalQNNModel(cfg["model"], modulus=int(cfg["data"]["modulus"])).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg, checkpoint_path


@torch.no_grad()
def collect_split(
    model: TrimodalQNNModel,
    cfg: dict[str, Any],
    *,
    split: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    dataset = ModularPairsDataset(cfg["data"], split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rep_parts: dict[str, dict[str, list[np.ndarray]]] = {
        "encoder": {"input": []},
        "complex_state": {"initial": []},
        "prob_state": {"initial": []},
    }
    for idx in range(len(model.layers)):
        rep_parts["complex_state"][layer_name(idx + 1)] = []
        rep_parts["prob_state"][layer_name(idx + 1)] = []
    labels = []

    for batch_cpu in loader:
        batch = batch_to_device(batch_cpu, device)
        labels.append(batch_cpu["y"].numpy())
        state, features = model.initial_state_and_features(batch)
        rep_parts["encoder"]["input"].append(features.detach().cpu().numpy())
        rep_parts["complex_state"]["initial"].append(complex_features(state))
        rep_parts["prob_state"]["initial"].append(prob_features(state))
        for idx, layer in enumerate(model.layers, start=1):
            state = layer(state, features, ablate_cross=False)
            rep_parts["complex_state"][layer_name(idx)].append(complex_features(state))
            rep_parts["prob_state"][layer_name(idx)].append(prob_features(state))

    reps: dict[str, dict[str, np.ndarray]] = {}
    for rep_name, by_layer in rep_parts.items():
        reps[rep_name] = {name: np.concatenate(parts, axis=0) for name, parts in by_layer.items()}
    return reps, np.concatenate(labels, axis=0)


def complex_features(state: torch.Tensor) -> np.ndarray:
    return torch.cat([state.real, state.imag], dim=-1).detach().cpu().numpy()


def prob_features(state: torch.Tensor) -> np.ndarray:
    probs = state.abs().pow(2)
    mass = probs.sum(dim=-1, keepdim=True)
    return torch.cat([probs, mass], dim=-1).detach().cpu().numpy()


def invsqrt_cov(cov: np.ndarray, *, eps: float) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    return (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T


def fit_cca(x: np.ndarray, y: np.ndarray, *, n_components: int, reg: float) -> CCAFit:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    mean_x = x.mean(axis=0, keepdims=True)
    mean_y = y.mean(axis=0, keepdims=True)
    xc = x - mean_x
    yc = y - mean_y
    denom = max(x.shape[0] - 1, 1)
    cxx = (xc.T @ xc) / denom
    cyy = (yc.T @ yc) / denom
    cxy = (xc.T @ yc) / denom
    rx = float(reg) * (float(np.trace(cxx)) / max(cxx.shape[0], 1) + 1e-12)
    ry = float(reg) * (float(np.trace(cyy)) / max(cyy.shape[0], 1) + 1e-12)
    cxx = cxx + rx * np.eye(cxx.shape[0], dtype=np.float64)
    cyy = cyy + ry * np.eye(cyy.shape[0], dtype=np.float64)
    inv_x = invsqrt_cov(cxx, eps=max(rx * 1e-3, 1e-12))
    inv_y = invsqrt_cov(cyy, eps=max(ry * 1e-3, 1e-12))
    u, s, vt = np.linalg.svd(inv_x @ cxy @ inv_y, full_matrices=False)
    k = min(int(n_components), u.shape[1], vt.shape[0])
    return CCAFit(mean_x=mean_x, mean_y=mean_y, wx=inv_x @ u[:, :k], wy=inv_y @ vt.T[:, :k], train_corrs=s[:k])


def project_pair(fit: CCAFit, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (x.astype(np.float64, copy=False) - fit.mean_x) @ fit.wx, (y.astype(np.float64, copy=False) - fit.mean_y) @ fit.wy


def heldout_corrs(zx: np.ndarray, zy: np.ndarray) -> np.ndarray:
    zx = zx - zx.mean(axis=0, keepdims=True)
    zy = zy - zy.mean(axis=0, keepdims=True)
    num = (zx * zy).sum(axis=0)
    den = np.sqrt(np.square(zx).sum(axis=0) * np.square(zy).sum(axis=0)).clip(1e-12)
    return np.abs(num / den)


def ridge_classifier_fit(x: np.ndarray, labels: np.ndarray, *, n_classes: int, reg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x.astype(np.float64, copy=False)
    mean = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    z = (x - mean) / scale
    z = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)
    y = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    y[np.arange(labels.shape[0]), labels.astype(np.int64)] = 1.0
    xtx = z.T @ z
    xtx += float(reg) * np.eye(xtx.shape[0], dtype=np.float64)
    w = np.linalg.solve(xtx, z.T @ y)
    return w, mean, scale


def ridge_classifier_accuracy(x: np.ndarray, labels: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    w, mean, scale = fit
    z = (x.astype(np.float64, copy=False) - mean) / scale
    z = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)
    pred = (z @ w).argmax(axis=1)
    return float((pred == labels.astype(np.int64)).mean())


def sector_index(sector: str) -> int:
    return MODALITIES.index(sector)


def summarize_corrs(corrs: np.ndarray, prefix: str = "heldout") -> dict[str, float]:
    out = {}
    for k in (1, 3, 5, 10, 20):
        kk = min(k, corrs.shape[0])
        out[f"{prefix}_mean_top{k}"] = float(np.mean(corrs[:kk])) if kk else float("nan")
    out[f"{prefix}_top1"] = float(corrs[0]) if corrs.shape[0] else float("nan")
    return out


def score_subspace_overlap(x: np.ndarray, wa: np.ndarray, wb: np.ndarray, *, k: int) -> dict[str, float]:
    za = (x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)) @ wa[:, :k]
    zb = (x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)) @ wb[:, :k]
    qa, _ = np.linalg.qr(za)
    qb, _ = np.linalg.qr(zb)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return {
        "overlap_mean_cos": float(np.mean(s)),
        "overlap_min_cos": float(np.min(s)),
        "overlap_top_cos": float(np.max(s)),
    }


def run_cca_analysis(
    *,
    model: TrimodalQNNModel,
    cfg: dict[str, Any],
    batch_size: int,
    device: torch.device,
    n_components: int,
    cca_reg: float,
    probe_reg: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train_reps, train_y = collect_split(model, cfg, split="train", batch_size=batch_size, device=device)
    heldout_reps, heldout_y = collect_split(model, cfg, split="heldout", batch_size=batch_size, device=device)
    modulus = int(cfg["data"]["modulus"])
    pair_rows: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []

    for rep_name, train_layers in train_reps.items():
        for layer in train_layers:
            train_arr = train_reps[rep_name][layer]
            heldout_arr = heldout_reps[rep_name][layer]
            fits: dict[tuple[str, str], CCAFit] = {}
            pair_metrics: dict[str, dict[str, float]] = {}

            for sector in MODALITIES:
                train_x = train_arr[:, sector_index(sector), :]
                heldout_x = heldout_arr[:, sector_index(sector), :]
                fit = ridge_classifier_fit(train_x, train_y, n_classes=modulus, reg=probe_reg)
                probe_rows.append(
                    {
                        "representation": rep_name,
                        "layer": layer,
                        "sector": sector,
                        "feature_dim": train_x.shape[1],
                        "heldout_accuracy": ridge_classifier_accuracy(heldout_x, heldout_y, fit),
                    }
                )

            for left, right in SECTOR_PAIRS:
                x_train = train_arr[:, sector_index(left), :]
                y_train = train_arr[:, sector_index(right), :]
                x_held = heldout_arr[:, sector_index(left), :]
                y_held = heldout_arr[:, sector_index(right), :]
                fit = fit_cca(x_train, y_train, n_components=n_components, reg=cca_reg)
                fits[(left, right)] = fit
                zx_train, zy_train = project_pair(fit, x_train, y_train)
                zx_held, zy_held = project_pair(fit, x_held, y_held)
                corrs = heldout_corrs(zx_held, zy_held)
                pair_feature_train = np.concatenate([zx_train[:, :20], zy_train[:, :20]], axis=1)
                pair_feature_held = np.concatenate([zx_held[:, :20], zy_held[:, :20]], axis=1)
                probe_fit = ridge_classifier_fit(pair_feature_train, train_y, n_classes=modulus, reg=probe_reg)
                row: dict[str, Any] = {
                    "representation": rep_name,
                    "layer": layer,
                    "pair": f"{left}{right}",
                    "left": left,
                    "right": right,
                    "feature_dim": x_train.shape[1],
                    "train_n": train_arr.shape[0],
                    "heldout_n": heldout_arr.shape[0],
                    "train_top1": float(fit.train_corrs[0]),
                    "train_mean_top5": float(np.mean(fit.train_corrs[: min(5, fit.train_corrs.shape[0])])),
                    "cca_sum_probe_heldout_accuracy": ridge_classifier_accuracy(pair_feature_held, heldout_y, probe_fit),
                    **summarize_corrs(corrs),
                }
                pair_rows.append(row)
                pair_metrics[f"{left}{right}"] = row

            for metric in ("heldout_top1", "heldout_mean_top5", "heldout_mean_top10", "heldout_mean_top20", "cca_sum_probe_heldout_accuracy"):
                tn = float(pair_metrics["TN"][metric])
                ti = float(pair_metrics["TI"][metric])
                ni = float(pair_metrics["NI"][metric])
                binding_rows.append(
                    {
                        "representation": rep_name,
                        "layer": layer,
                        "metric": metric,
                        "TN": tn,
                        "TI": ti,
                        "NI": ni,
                        "text_hub_score": 0.5 * (tn + ti) - ni,
                    }
                )

            for anchor in MODALITIES:
                if anchor == "T":
                    x_anchor = train_arr[:, sector_index("T"), :]
                    wa = fits[("T", "N")].wx
                    wb = fits[("T", "I")].wx
                    compared = "TN_text_vs_TI_text"
                elif anchor == "N":
                    x_anchor = train_arr[:, sector_index("N"), :]
                    wa = fits[("T", "N")].wy
                    wb = fits[("N", "I")].wx
                    compared = "TN_number_vs_NI_number"
                else:
                    x_anchor = train_arr[:, sector_index("I"), :]
                    wa = fits[("T", "I")].wy
                    wb = fits[("N", "I")].wy
                    compared = "TI_image_vs_NI_image"
                for k in (3, 5, 10, 20):
                    kk = min(k, wa.shape[1], wb.shape[1])
                    overlap_rows.append(
                        {
                            "representation": rep_name,
                            "layer": layer,
                            "anchor": anchor,
                            "comparison": compared,
                            "k": kk,
                            **score_subspace_overlap(x_anchor, wa, wb, k=kk),
                        }
                    )

    return pair_rows, binding_rows, overlap_rows, probe_rows


def ordered_layers(rows: list[dict[str, Any]]) -> list[str]:
    names = sorted({str(row["layer"]) for row in rows})
    def key(name: str) -> int:
        if name == "input":
            return -1
        if name == "initial":
            return 0
        return int(name.split("_")[1])
    return sorted(names, key=key)


def make_figures(out_dir: Path, pair_rows: list[dict[str, Any]], binding_rows: list[dict[str, Any]], overlap_rows: list[dict[str, Any]], probe_rows: list[dict[str, Any]]) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for rep in ("encoder", "complex_state", "prob_state"):
        rows = [row for row in pair_rows if row["representation"] == rep]
        if not rows:
            continue
        layers = ordered_layers(rows)
        plt.figure(figsize=(9.5, 5.0))
        for pair in ("TN", "TI", "NI"):
            values = []
            xs = []
            for idx, layer in enumerate(layers):
                match = [row for row in rows if row["pair"] == pair and row["layer"] == layer]
                if match:
                    xs.append(idx)
                    values.append(float(match[0]["heldout_mean_top10"]))
            plt.plot(xs, values, marker="o", linewidth=1.6, markersize=4, label=pair)
        plt.xticks(range(len(layers)), layers, rotation=25, ha="right")
        plt.ylim(0.0, 1.02)
        plt.ylabel("mean top-10 CCA")
        plt.title(f"{rep} CCA")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / f"{rep}_pairwise_cca.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    rows = [row for row in binding_rows if row["metric"] == "heldout_mean_top10"]
    reps = ["encoder", "complex_state", "prob_state"]
    layers = ordered_layers(rows)
    matrix = np.full((len(reps), len(layers)), np.nan)
    for row in rows:
        if row["representation"] in reps and row["layer"] in layers:
            matrix[reps.index(str(row["representation"])), layers.index(str(row["layer"]))] = float(row["text_hub_score"])
    plt.figure(figsize=(10, 3.8))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 1e-6)
    plt.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="text hub score")
    plt.yticks(range(len(reps)), reps)
    plt.xticks(range(len(layers)), layers, rotation=25, ha="right")
    plt.title("CCA Text Hub")
    plt.tight_layout()
    path = fig_dir / "text_hub_score_heatmap.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    for rep in ("encoder", "complex_state", "prob_state"):
        rows = [row for row in probe_rows if row["representation"] == rep]
        if not rows:
            continue
        layers = ordered_layers(rows)
        plt.figure(figsize=(9.5, 5.0))
        for sector in MODALITIES:
            values = []
            xs = []
            for idx, layer in enumerate(layers):
                match = [row for row in rows if row["sector"] == sector and row["layer"] == layer]
                if match:
                    xs.append(idx)
                    values.append(float(match[0]["heldout_accuracy"]))
            plt.plot(xs, values, marker="o", linewidth=1.6, markersize=4, label=sector)
        plt.xticks(range(len(layers)), layers, rotation=25, ha="right")
        plt.ylim(0.0, 1.02)
        plt.ylabel("held-out accuracy")
        plt.title(f"{rep} Sector Probe")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / f"{rep}_sector_sum_probe.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    rows = [row for row in overlap_rows if int(row["k"]) == 10]
    for rep in ("encoder", "complex_state", "prob_state"):
        rep_rows = [row for row in rows if row["representation"] == rep]
        if not rep_rows:
            continue
        layers = ordered_layers(rep_rows)
        plt.figure(figsize=(9.5, 5.0))
        for anchor in MODALITIES:
            values = []
            xs = []
            for idx, layer in enumerate(layers):
                match = [row for row in rep_rows if row["anchor"] == anchor and row["layer"] == layer]
                if match:
                    xs.append(idx)
                    values.append(float(match[0]["overlap_mean_cos"]))
            plt.plot(xs, values, marker="o", linewidth=1.6, markersize=4, label=anchor)
        plt.xticks(range(len(layers)), layers, rotation=25, ha="right")
        plt.ylim(0.0, 1.02)
        plt.ylabel("mean principal cosine")
        plt.title(f"{rep} Anchor Overlap")
        plt.legend()
        plt.tight_layout()
        path = fig_dir / f"{rep}_anchor_overlap.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    return paths


def best_rows(rows: list[dict[str, Any]], key: str, *, n: int = 10, reverse: bool = True) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key, 0.0)), reverse=reverse)[:n]


def write_report(
    out_dir: Path,
    *,
    run_name: str,
    checkpoint_step: int,
    checkpoint_path: Path,
    pair_rows: list[dict[str, Any]],
    binding_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    figure_paths: list[str],
    elapsed_sec: float,
) -> None:
    lines: list[str] = []
    lines.append("# Sector CCA Binding Analysis")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Run: `{run_name}`")
    lines.append(f"Checkpoint: `{checkpoint_step}`")
    lines.append(f"Checkpoint path: `{checkpoint_path}`")
    lines.append(f"Elapsed seconds: `{elapsed_sec:.3f}`")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("CCA maps are fit on strict training pairs and evaluated on held-out pairs. Representations are encoder sector features, complex sector statevectors, and per-sector probability features. A positive text hub score means average `T-N`/`T-I` CCA exceeds `N-I` CCA.")
    lines.append("")
    lines.append("## Pairwise CCA")
    lines.append("")
    lines.append("Final-layer held-out mean top-10 CCA:")
    lines.append("")
    lines.append("| representation | T-N | T-I | N-I | text hub score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for rep in ("encoder", "complex_state", "prob_state"):
        layer = "input" if rep == "encoder" else "layer_4"
        rows = [row for row in pair_rows if row["representation"] == rep and row["layer"] == layer]
        by_pair = {row["pair"]: float(row["heldout_mean_top10"]) for row in rows}
        if {"TN", "TI", "NI"}.issubset(by_pair):
            hub = 0.5 * (by_pair["TN"] + by_pair["TI"]) - by_pair["NI"]
            lines.append(f"| {rep} `{layer}` | {by_pair['TN']:.6f} | {by_pair['TI']:.6f} | {by_pair['NI']:.6f} | {hub:.6f} |")
    lines.append("")
    lines.append("Strongest text-hub CCA rows:")
    lines.append("")
    lines.append("| representation | layer | metric | T-N | T-I | N-I | hub |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in best_rows([row for row in binding_rows if row["metric"] == "heldout_mean_top10"], "text_hub_score", n=10):
        lines.append(
            f"| {row['representation']} | {row['layer']} | {row['metric']} | {float(row['TN']):.6f} | "
            f"{float(row['TI']):.6f} | {float(row['NI']):.6f} | {float(row['text_hub_score']):.6f} |"
        )
    lines.append("")
    lines.append("## Anchor Subspace Overlap")
    lines.append("")
    lines.append("Top anchor overlaps at `k=10`:")
    lines.append("")
    lines.append("| representation | layer | anchor | mean cosine | min cosine |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for row in best_rows([row for row in overlap_rows if int(row["k"]) == 10], "overlap_mean_cos", n=12):
        lines.append(
            f"| {row['representation']} | {row['layer']} | {row['anchor']} | "
            f"{float(row['overlap_mean_cos']):.6f} | {float(row['overlap_min_cos']):.6f} |"
        )
    lines.append("")
    lines.append("## Single-Sector Sum Probes")
    lines.append("")
    lines.append("Held-out residue accuracy from one sector's raw subspace:")
    lines.append("")
    lines.append("| representation | layer | T | N | I |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for rep in ("encoder", "complex_state", "prob_state"):
        relevant = [row for row in probe_rows if row["representation"] == rep]
        for layer in ordered_layers(relevant):
            vals = {row["sector"]: float(row["heldout_accuracy"]) for row in relevant if row["layer"] == layer}
            if set(MODALITIES).issubset(vals):
                lines.append(f"| {rep} | {layer} | {vals['T']:.6f} | {vals['N']:.6f} | {vals['I']:.6f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If text is the binding modality, CCA should show `T-N` and `T-I` alignment at least as high as `N-I`, and the text-side CCA subspaces for number and image should overlap.")
    lines.append("- If text merely dominates classification without binding, text single-sector probes can be high while pairwise CCA and anchor overlap remain weak.")
    lines.append("- If all modalities share a balanced code, all three pairwise CCA values and all three anchor overlaps should be similar.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for path in figure_paths:
        lines.append(f"- `{Path(path).relative_to(out_dir)}`")
    lines.append("")
    lines.append("## Tables")
    lines.append("")
    lines.append("- `pairwise_cca.csv`")
    lines.append("- `text_binding_scores.csv`")
    lines.append("- `anchor_subspace_overlap.csv`")
    lines.append("- `sector_sum_probe.csv`")
    lines.append("- `manifest.json`")
    lines.append("")
    (out_dir / "TRIMODAL_QNN_SECTOR_CCA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CCA analysis of trimodal QNN sector subspaces.")
    parser.add_argument("--run-dir", default="trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean")
    parser.add_argument("--checkpoint-step", type=int, default=2000)
    parser.add_argument("--out-dir", default="trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_sector_cca")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--cca-reg", type=float, default=1e-3)
    parser.add_argument("--probe-reg", type=float, default=1e-2)
    args = parser.parse_args()

    root = Path.cwd()
    run_dir = Path(args.run_dir)
    run_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir
    out_dir = Path(args.out_dir)
    out_dir = (root / out_dir).resolve() if not out_dir.is_absolute() else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    start = time.time()
    model, cfg, checkpoint_path = load_model(run_dir, int(args.checkpoint_step), device)
    pair_rows, binding_rows, overlap_rows, probe_rows = run_cca_analysis(
        model=model,
        cfg=cfg,
        batch_size=int(args.batch_size),
        device=device,
        n_components=int(args.n_components),
        cca_reg=float(args.cca_reg),
        probe_reg=float(args.probe_reg),
    )
    figure_paths = make_figures(out_dir, pair_rows, binding_rows, overlap_rows, probe_rows)

    write_csv(out_dir / "pairwise_cca.csv", pair_rows)
    write_csv(out_dir / "text_binding_scores.csv", binding_rows)
    write_csv(out_dir / "anchor_subspace_overlap.csv", overlap_rows)
    write_csv(out_dir / "sector_sum_probe.csv", probe_rows)
    manifest = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "checkpoint_step": int(args.checkpoint_step),
        "checkpoint_path": str(checkpoint_path),
        "out_dir": str(out_dir),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "n_components": int(args.n_components),
        "cca_reg": float(args.cca_reg),
        "probe_reg": float(args.probe_reg),
        "elapsed_sec": time.time() - start,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_report(
        out_dir,
        run_name=run_dir.name,
        checkpoint_step=int(args.checkpoint_step),
        checkpoint_path=checkpoint_path,
        pair_rows=pair_rows,
        binding_rows=binding_rows,
        overlap_rows=overlap_rows,
        probe_rows=probe_rows,
        figure_paths=figure_paths,
        elapsed_sec=manifest["elapsed_sec"],
    )
    print(json.dumps({"event": "sector_cca_done", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
