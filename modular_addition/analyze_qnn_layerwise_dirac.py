from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .analyze_qnn_delta_rescue import (
    calibration,
    coeff_frequency_diagnostics,
    coefficient_norm_rows,
    metric_rows_for_logits,
    signed_offset,
    split_masks,
    topk_accuracy,
    within_accuracy,
)
from .analyze_qnn_mech import (
    acc,
    addition_fourier,
    all_pairs,
    behavior_strata,
    cross_entropy,
    feature_fourier_probes,
    feature_grid_fft,
    logsumexp_np,
    margin_stats,
    markdown_table,
    nearest_train_diagnostics,
    residual_fit,
    to_jsonable,
    train_mask_for_split,
)
from .qnn_mod97 import DiracDeltaHead, FourierDeltaHead, QNNClassifier
from .train import resolve_device


def checkpoint_config(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint, checkpoint["config"]


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[QNNClassifier, dict[str, Any], dict[str, Any]]:
    checkpoint, config = checkpoint_config(checkpoint_path)
    data_cfg = config["dataset"]
    model_cfg = config["model"]
    model = QNNClassifier(
        variant=checkpoint["variant"],
        n_qubits=int(model_cfg.get("n_qubits", 7)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        modulus=int(data_cfg.get("modulus", 97)),
        input_frequencies=model_cfg.get("input_frequencies"),
        auxiliary_head_moduli=model_cfg.get("auxiliary_head_moduli"),
        readout_type=str(model_cfg.get("readout_type", "linear")),
        fourier_max_frequency=model_cfg.get("fourier_max_frequency"),
        fourier_kernel_init=str(model_cfg.get("fourier_kernel_init", "fejer")),
        fourier_residual_linear=bool(model_cfg.get("fourier_residual_linear", False)),
        fourier_residual_scale=float(model_cfg.get("fourier_residual_scale", 1.0)),
        fourier_kernel_trainable=bool(model_cfg.get("fourier_kernel_trainable", True)),
        dirac_coefficient_mode=str(model_cfg.get("dirac_coefficient_mode", "unit")),
        dirac_coefficient_eps=float(model_cfg.get("dirac_coefficient_eps", 1e-6)),
        dirac_kernel_trainable=bool(model_cfg.get("dirac_kernel_trainable", False)),
        dirac_sharpen_kernel_init=str(model_cfg.get("dirac_sharpen_kernel_init", "dirichlet")),
        dirac_sharpen_kernel_trainable=bool(model_cfg.get("dirac_sharpen_kernel_trainable", True)),
        dirac_sharpen_strength_init=float(model_cfg.get("dirac_sharpen_strength_init", 0.0)),
        dirac_sharpen_strength_max=float(model_cfg.get("dirac_sharpen_strength_max", 0.25)),
        dirac_primary_base_scale_init=float(model_cfg.get("dirac_primary_base_scale_init", 0.2)),
        dirac_primary_base_scale_trainable=bool(model_cfg.get("dirac_primary_base_scale_trainable", True)),
        layerwise_dirac_adapter_scale=float(model_cfg.get("layerwise_dirac_adapter_scale", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config, checkpoint


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def head_components(
    head: FourierDeltaHead,
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    normed = head.norm(features)
    raw_coeffs = head.raw_fourier_features(features)
    coeffs = head.coefficients_from_normed(normed)
    weights = head.kernel_weights()
    scale = head.scale.exp().clamp(max=100.0)
    delta = scale * (coeffs @ (head.class_basis * weights).T)
    if head.residual is None:
        residual = torch.zeros_like(delta)
    else:
        residual = head.residual_scale * head.residual(normed)
    logits = delta + residual + head.bias
    return {
        "raw_coeffs": raw_coeffs,
        "coeffs": coeffs,
        "delta": delta,
        "residual": residual,
        "bias": head.bias.detach().expand_as(delta),
        "logits": logits,
    }


@torch.no_grad()
def collect_layerwise(
    model: QNNClassifier,
    *,
    modulus: int,
    device: torch.device,
    batch_size: int,
    adapter_scale_override: float | None = None,
) -> dict[str, Any]:
    if model.readout_type not in {"layerwise_dirac_aux", "layerwise_dirac_adapter", "layerwise_dirac_residual"}:
        raise TypeError(f"unsupported readout_type: {model.readout_type}")
    if not hasattr(model, "layerwise_heads"):
        raise TypeError("checkpoint does not have layerwise heads")

    a, b, _ = all_pairs(modulus)
    n_layers = len(model.layerwise_heads)
    saved_scale = float(getattr(model, "layerwise_dirac_adapter_scale", 0.0))
    if adapter_scale_override is not None:
        model.layerwise_dirac_adapter_scale = float(adapter_scale_override)

    final_logits_parts: list[torch.Tensor] = []
    final_features_parts: list[torch.Tensor] = []
    layer_logits_parts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    layer_features_parts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    layer_delta_parts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    layer_residual_parts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    layer_raw_coeff_parts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    layer_coeff_parts: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    final_head_parts: dict[str, list[torch.Tensor]] = {
        "raw_coeffs": [],
        "coeffs": [],
        "delta": [],
        "residual": [],
        "bias": [],
        "logits": [],
    }

    try:
        for start in range(0, len(a), batch_size):
            aa = torch.tensor(a[start : start + batch_size], dtype=torch.long, device=device)
            bb = torch.tensor(b[start : start + batch_size], dtype=torch.long, device=device)
            logits, features = model.forward_with_features(aa, bb)
            final_logits_parts.append(logits.detach().cpu())
            final_features_parts.append(features.detach().cpu())
            for layer, head in enumerate(model.layerwise_heads):
                feats = model._last_layerwise_features[layer]
                comps = head_components(head, feats)
                layer_features_parts[layer].append(feats.detach().cpu())
                layer_logits_parts[layer].append(model._last_layerwise_logits[layer].detach().cpu())
                layer_delta_parts[layer].append(comps["delta"].detach().cpu())
                layer_residual_parts[layer].append(comps["residual"].detach().cpu())
                layer_raw_coeff_parts[layer].append(comps["raw_coeffs"].detach().cpu())
                layer_coeff_parts[layer].append(comps["coeffs"].detach().cpu())
            if isinstance(model.head, FourierDeltaHead):
                comps = head_components(model.head, features)
                for key in final_head_parts:
                    final_head_parts[key].append(comps[key].detach().cpu())
    finally:
        if adapter_scale_override is not None:
            model.layerwise_dirac_adapter_scale = saved_scale

    out: dict[str, Any] = {
        "final_logits": torch.cat(final_logits_parts).numpy(),
        "final_features": torch.cat(final_features_parts).numpy(),
        "layers": [],
        "residual_weights": None,
    }
    for layer in range(n_layers):
        out["layers"].append(
            {
                "features": torch.cat(layer_features_parts[layer]).numpy(),
                "logits": torch.cat(layer_logits_parts[layer]).numpy(),
                "delta": torch.cat(layer_delta_parts[layer]).numpy(),
                "residual": torch.cat(layer_residual_parts[layer]).numpy(),
                "raw_coeffs": torch.cat(layer_raw_coeff_parts[layer]).numpy(),
                "coeffs": torch.cat(layer_coeff_parts[layer]).numpy(),
                "frequency_weights": model.layerwise_heads[layer].kernel_weights().detach().cpu().numpy()[::2],
                "bias": model.layerwise_heads[layer].bias.detach().cpu().numpy(),
                "class_basis": model.layerwise_heads[layer].class_basis.detach().cpu().numpy(),
                "scale": float(model.layerwise_heads[layer].scale.exp().clamp(max=100.0).detach().cpu()),
                "max_frequency": int(model.layerwise_heads[layer].max_frequency),
            }
        )
    if final_head_parts["logits"]:
        out["final_head"] = {key: torch.cat(parts).numpy() for key, parts in final_head_parts.items()}
        out["final_head"].update(
            {
                "frequency_weights": model.head.kernel_weights().detach().cpu().numpy()[::2],
                "bias_vector": model.head.bias.detach().cpu().numpy(),
                "class_basis": model.head.class_basis.detach().cpu().numpy(),
                "scale": float(model.head.scale.exp().clamp(max=100.0).detach().cpu()),
                "max_frequency": int(model.head.max_frequency),
            }
        )
    if hasattr(model, "layerwise_logit_weights"):
        out["residual_weights"] = torch.softmax(model.layerwise_logit_weights, dim=0).detach().cpu().numpy()
    return out


def cutoff_logits(
    coeffs: np.ndarray,
    class_basis: np.ndarray,
    frequency_weights: np.ndarray,
    scale: float,
    bias: np.ndarray,
    cutoffs: list[int],
) -> dict[str, np.ndarray]:
    out = {}
    for cutoff in cutoffs:
        k = min(int(cutoff), int(len(frequency_weights)))
        cols = 2 * k
        weights = np.repeat(frequency_weights[:k], 2)
        logits = float(scale) * (coeffs[:, :cols] @ (class_basis[:, :cols] * weights).T)
        out[f"k{cutoff}"] = logits + bias[None, :]
    return out


def variant_logits(
    collected: dict[str, Any],
    *,
    readout_type: str,
    labels: np.ndarray,
    cutoffs: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    variants: dict[str, np.ndarray] = {"final_full": collected["final_logits"]}
    cutoff_variants: dict[str, np.ndarray] = {}
    layer_logits = [layer["logits"] for layer in collected["layers"]]
    for i, layer in enumerate(collected["layers"]):
        variants[f"layer_{i}_full"] = layer["logits"]
        variants[f"layer_{i}_delta_only"] = layer["delta"]
        variants[f"layer_{i}_delta_bias"] = layer["delta"] + layer["bias"][None, :]
        for name, logits in cutoff_logits(
            layer["coeffs"],
            layer["class_basis"],
            layer["frequency_weights"],
            layer["scale"],
            layer["bias"],
            cutoffs,
        ).items():
            cutoff_variants[f"layer_{i}_{name}"] = logits
    if "final_head" in collected:
        fh = collected["final_head"]
        variants["final_head_recomputed"] = fh["logits"]
        variants["final_delta_only"] = fh["delta"]
        variants["final_delta_bias"] = fh["delta"] + fh["bias"]
        for name, logits in cutoff_logits(
            fh["coeffs"],
            fh["class_basis"],
            fh["frequency_weights"],
            fh["scale"],
            fh["bias_vector"],
            cutoffs,
        ).items():
            cutoff_variants[f"final_{name}"] = logits
    if layer_logits:
        variants["layer_uniform_mean"] = np.mean(np.stack(layer_logits, axis=0), axis=0)
        weights = collected.get("residual_weights")
        if weights is not None:
            stacked = np.stack(layer_logits, axis=0)
            variants["residual_weighted"] = (weights[:, None, None] * stacked).sum(axis=0)
            for i in range(len(layer_logits)):
                variants[f"residual_only_layer_{i}"] = layer_logits[i]
                keep = weights.copy()
                keep[i] = 0.0
                if keep.sum() > 0:
                    keep = keep / keep.sum()
                variants[f"residual_without_layer_{i}_renorm"] = (keep[:, None, None] * stacked).sum(axis=0)
            for i in range(len(layer_logits)):
                variants[f"residual_cumulative_to_layer_{i}"] = np.mean(stacked[: i + 1], axis=0)
    return variants, cutoff_variants


def rows_for_variant_set(
    variants: dict[str, np.ndarray],
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    modulus: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, logits in variants.items():
        rows.extend(metric_rows_for_logits(name, logits, labels, masks, modulus))
    return rows


def residual_fit_rows(
    variants: dict[str, np.ndarray],
    train_mask: np.ndarray,
    modulus: int,
    selected: list[str],
) -> list[dict[str, Any]]:
    rows = []
    masks = {"all": None, "train": train_mask, "held_out": ~train_mask}
    for name in selected:
        if name not in variants:
            continue
        cube = variants[name].reshape(modulus, modulus, modulus)
        for split, mask in masks.items():
            for kind in ["mod_add", "integer_add", "a_minus_c", "b_minus_c", "class_only", "sum_only"]:
                fit = residual_fit(cube, modulus, kind, pair_mask=mask)
                rows.append(
                    {
                        "variant": name,
                        "split": split,
                        "fit": kind,
                        "r2": fit["r2"],
                        "correct_residual_margin": fit.get("correct_residual_margin", float("nan")),
                    }
                )
    return rows


def layer_overlap_rows(
    variants: dict[str, np.ndarray],
    labels: np.ndarray,
    train_mask: np.ndarray,
    n_layers: int,
) -> list[dict[str, Any]]:
    held = ~train_mask
    final_pred = variants["final_full"].argmax(axis=1)
    final_correct = final_pred == labels
    rows = []
    for i in range(n_layers):
        name = f"layer_{i}_full"
        pred = variants[name].argmax(axis=1)
        correct = pred == labels
        rows.append(
            {
                "layer": i,
                "held_out_layer_accuracy": acc(pred, labels, held),
                "final_wrong_layer_correct_count": int((held & ~final_correct & correct).sum()),
                "final_correct_layer_wrong_count": int((held & final_correct & ~correct).sum()),
                "held_out_agreement_with_final": float(np.mean(pred[held] == final_pred[held])),
                "held_out_both_correct": int((held & final_correct & correct).sum()),
                "held_out_both_wrong": int((held & ~final_correct & ~correct).sum()),
            }
        )
    return rows


def failure_cases(
    variants: dict[str, np.ndarray],
    labels: np.ndarray,
    train_mask: np.ndarray,
    modulus: int,
    n_layers: int,
    limit: int = 300,
) -> list[dict[str, Any]]:
    a, b, _ = all_pairs(modulus)
    held = ~train_mask
    full = variants["final_full"]
    pred = full.argmax(axis=1)
    correct = pred == labels
    probs = softmax_np(full)
    x = full.copy()
    x[np.arange(x.shape[0]), labels] = -np.inf
    competitor = x.argmax(axis=1)
    margin = full[np.arange(full.shape[0]), labels] - full[np.arange(full.shape[0]), competitor]
    idxs = np.where(held & ~correct)[0]
    idxs = idxs[np.argsort(margin[idxs])]
    offsets = signed_offset(pred, labels, modulus)
    rows = []
    for idx in idxs[:limit]:
        row = {
            "a": int(a[idx]),
            "b": int(b[idx]),
            "integer_sum": int(a[idx] + b[idx]),
            "wrap": bool(a[idx] + b[idx] >= modulus),
            "true_sum": int(labels[idx]),
            "prediction": int(pred[idx]),
            "signed_offset": int(offsets[idx]),
            "confidence": float(probs[idx, pred[idx]]),
            "true_probability": float(probs[idx, labels[idx]]),
            "competitor": int(competitor[idx]),
            "final_margin": float(margin[idx]),
        }
        for layer in range(n_layers):
            lp = variants[f"layer_{layer}_full"].argmax(axis=1)
            row[f"layer_{layer}_prediction"] = int(lp[idx])
            row[f"layer_{layer}_correct"] = bool(lp[idx] == labels[idx])
        rows.append(row)
    return rows


def offset_summary(
    variants: dict[str, np.ndarray],
    labels: np.ndarray,
    train_mask: np.ndarray,
    modulus: int,
    names: list[str],
) -> list[dict[str, Any]]:
    held = ~train_mask
    rows = []
    for name in names:
        if name not in variants:
            continue
        pred = variants[name].argmax(axis=1)
        offsets = signed_offset(pred[held], labels[held], modulus)
        values, counts = np.unique(offsets, return_counts=True)
        top = sorted(zip(values.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:12]
        for offset, count in top:
            rows.append({"variant": name, "signed_offset": int(offset), "count": int(count), "fraction": float(count / held.sum())})
    return rows


def layer_head_summaries(collected: dict[str, Any], train_mask: np.ndarray, labels: np.ndarray, modulus: int) -> list[dict[str, Any]]:
    rows = []
    for i, layer in enumerate(collected["layers"]):
        norms = coefficient_norm_rows({"raw_coeffs": layer["raw_coeffs"], "coeffs": layer["coeffs"]})
        for row in norms:
            row = dict(row)
            row["head"] = f"layer_{i}"
            rows.append(row)
    if "final_head" in collected:
        norms = coefficient_norm_rows({"raw_coeffs": collected["final_head"]["raw_coeffs"], "coeffs": collected["final_head"]["coeffs"]})
        for row in norms:
            row = dict(row)
            row["head"] = "final"
            rows.append(row)
    return rows


def frequency_rows(collected: dict[str, Any], labels: np.ndarray, train_mask: np.ndarray, modulus: int) -> list[dict[str, Any]]:
    rows = []
    for i, layer in enumerate(collected["layers"]):
        for row in coeff_frequency_diagnostics(layer["coeffs"], labels, train_mask, layer["frequency_weights"], modulus):
            row = dict(row)
            row["head"] = f"layer_{i}"
            rows.append(row)
    if "final_head" in collected:
        for row in coeff_frequency_diagnostics(
            collected["final_head"]["coeffs"],
            labels,
            train_mask,
            collected["final_head"]["frequency_weights"],
            modulus,
        ):
            row = dict(row)
            row["head"] = "final"
            rows.append(row)
    return rows


def make_plots(
    out_dir: Path,
    result: dict[str, Any],
    variants: dict[str, np.ndarray],
    cutoff_rows: list[dict[str, Any]],
    labels: np.ndarray,
    train_mask: np.ndarray,
    modulus: int,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    held_rows = [row for row in result["component_metrics"] if row["split"] == "held_out"]
    important = [
        "final_full",
        "final_head_recomputed",
        "final_delta_bias",
        "layer_0_full",
        "layer_1_full",
        "layer_2_full",
        "layer_3_full",
        "layer_uniform_mean",
        "residual_weighted",
    ]
    rows = [row for row in held_rows if row["variant"] in important]
    rows = sorted(rows, key=lambda r: important.index(r["variant"]) if r["variant"] in important else 999)
    if rows:
        plt.figure(figsize=(10, 4))
        plt.bar([row["variant"] for row in rows], [row["accuracy"] for row in rows])
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 1.02)
        plt.ylabel("held-out accuracy")
        plt.title("Layerwise Accuracy")
        plt.tight_layout()
        plt.savefig(fig_dir / "layerwise_accuracy.png", dpi=170)
        plt.close()

    full = variants["final_full"]
    pred = full.argmax(axis=1)
    held = ~train_mask
    offsets = signed_offset(pred[held], labels[held], modulus)
    values, counts = np.unique(offsets, return_counts=True)
    plt.figure(figsize=(9, 4))
    plt.bar(values, counts / counts.sum(), width=0.9)
    plt.xlabel("signed offset")
    plt.ylabel("held-out fraction")
    plt.title("Prediction Offsets")
    plt.tight_layout()
    plt.savefig(fig_dir / "final_error_offsets.png", dpi=170)
    plt.close()

    cutoff_final = [row for row in cutoff_rows if row["split"] == "held_out" and row["variant"].startswith("final_k")]
    if not cutoff_final:
        cutoff_final = [row for row in cutoff_rows if row["split"] == "held_out" and row["variant"].startswith("layer_3_k")]
    cutoff_final = sorted(cutoff_final, key=lambda row: int(row["variant"].split("k")[-1]))
    if cutoff_final:
        plt.figure(figsize=(8, 4))
        plt.plot([int(row["variant"].split("k")[-1]) for row in cutoff_final], [row["accuracy"] for row in cutoff_final], marker="o", label="exact")
        plt.plot([int(row["variant"].split("k")[-1]) for row in cutoff_final], [row["within_1"] for row in cutoff_final], marker="o", label="within one")
        plt.xlabel("maximum Fourier frequency")
        plt.ylabel("held-out rate")
        plt.ylim(0, 1.02)
        plt.title("Frequency Cutoff")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "frequency_cutoff.png", dpi=170)
        plt.close()

    freq_rows = [row for row in result["frequency_diagnostics"] if row["split"] == "held_out" and row["frequency"] <= 21]
    if freq_rows:
        heads = sorted(set(row["head"] for row in freq_rows), key=lambda x: (x != "final", x))
        plt.figure(figsize=(10, 5))
        for head in heads:
            rows_h = [row for row in freq_rows if row["head"] == head]
            rows_h = sorted(rows_h, key=lambda row: row["frequency"])
            plt.plot([row["frequency"] for row in rows_h], [row["r2"] for row in rows_h], marker=".", linewidth=1.1, label=head)
        plt.xlabel("frequency")
        plt.ylabel("held-out coefficient R2")
        plt.title("Coefficient R2")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "coefficient_r2_by_head.png", dpi=170)
        plt.close()

    by_sum = result["behavior"]["accuracy_by_integer_sum"]
    plt.figure(figsize=(9, 4))
    plt.plot([row["integer_sum"] for row in by_sum], [row["held_out_accuracy"] for row in by_sum], linewidth=1.4)
    plt.axvline(50, color="black", linestyle="--", linewidth=1)
    plt.axvline(modulus - 1, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("ordinary sum")
    plt.ylabel("held-out accuracy")
    plt.ylim(0, 1.02)
    plt.title("Accuracy By Sum")
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_by_sum.png", dpi=170)
    plt.close()


def summarize_limited_range(result: dict[str, Any]) -> dict[str, Any]:
    rows = {row["name"]: row for row in result["behavior"]["summary_rows"]}
    held = rows["held_out"]["accuracy"]
    le50 = rows["held_out_sum_le_50"]["accuracy"]
    thresholds = result["behavior"]["threshold_accuracy"]
    best_gap = max(row["held_out_accuracy"] - held for row in thresholds if not math.isnan(row["held_out_accuracy"]))
    return {
        "held_out_accuracy": float(held),
        "held_out_sum_le_50_accuracy": float(le50),
        "sum_le_50_minus_overall": float(le50 - held),
        "largest_threshold_minus_overall": float(best_gap),
        "limited_to_sum_50_supported": bool(le50 > held + 0.15),
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    rows = {row["name"]: row for row in result["behavior"]["summary_rows"]}
    component_held = [row for row in result["component_metrics"] if row["split"] == "held_out"]
    comp = {row["variant"]: row for row in component_held}
    selected_component_names = [
        "final_full",
        "final_head_recomputed",
        "final_delta_bias",
        "layer_0_full",
        "layer_1_full",
        "layer_2_full",
        "layer_3_full",
        "layer_uniform_mean",
        "residual_weighted",
    ]
    selected_components = [comp[name] for name in selected_component_names if name in comp]
    cutoffs = [row for row in result["cutoff_metrics"] if row["split"] == "held_out"]
    cutoff_focus = [row for row in cutoffs if row["variant"].startswith("final_k")]
    if not cutoff_focus:
        cutoff_focus = [row for row in cutoffs if row["variant"].startswith("layer_3_k")]
    cutoff_focus = sorted(cutoff_focus, key=lambda row: int(row["variant"].split("k")[-1]))
    residual_rows = [row for row in result["residual_fit_rows"] if row["split"] == "held_out" and row["fit"] in {"mod_add", "integer_add", "class_only"}]
    residual_focus = [row for row in residual_rows if row["variant"] in {"final_full", "final_delta_bias", "layer_3_full", "residual_weighted"}]
    freq_rows = [
        row
        for row in result["frequency_diagnostics"]
        if row["split"] == "held_out" and row["frequency"] <= 8 and row["head"] in {"final", "layer_0", "layer_1", "layer_2", "layer_3"}
    ]
    limited = result["limited_range"]
    lines = [
        "# QNN Layerwise Dirac Exhaustive Analysis",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Run directory: `{result['run_dir']}`",
        f"Readout type: `{result['readout_type']}`",
        f"Variant: `{result['variant']}`",
        f"Checkpoint step: `{result['checkpoint_step']}`",
        f"Parameters: `{result['parameter_count']}`",
        "",
        "## Executive Summary",
        "",
        f"Held-out exact accuracy is `{rows['held_out']['accuracy']:.6f}` and train exact accuracy is `{rows['train']['accuracy']:.6f}`.",
        f"Wrap accuracy is `{rows['held_out_wrap']['accuracy']:.6f}` and no-wrap accuracy is `{rows['held_out_no_wrap']['accuracy']:.6f}`.",
        f"The ordinary-sum `<=50` slice is `{limited['held_out_sum_le_50_accuracy']:.6f}`, so the limited-to-50 hypothesis is `{limited['limited_to_sum_50_supported']}`.",
        "",
        "## Behavioral Evaluation",
        "",
        *markdown_table(
            result["behavior"]["summary_rows"],
            [("split", "name"), ("count", "count"), ("accuracy", "accuracy"), ("loss", "loss")],
        ),
        "",
        "## Layer And Head Components",
        "",
        *markdown_table(
            selected_components,
            [
                ("variant", "variant"),
                ("accuracy", "accuracy"),
                ("loss", "loss"),
                ("top2", "top2_accuracy"),
                ("within1", "within_1"),
                ("within2", "within_2"),
                ("margin", "margin_mean"),
            ],
        ),
        "",
        "## Frequency Cutoffs",
        "",
        *markdown_table(
            cutoff_focus,
            [("variant", "variant"), ("accuracy", "accuracy"), ("within1", "within_1"), ("within2", "within_2"), ("margin", "margin_mean")],
        ),
        "",
        "## Residual Fits",
        "",
        *markdown_table(
            residual_focus,
            [("variant", "variant"), ("fit", "fit"), ("R2", "r2"), ("correct residual margin", "correct_residual_margin")],
        ),
        "",
        "## Layer Correctness Overlap",
        "",
        *markdown_table(
            result["layer_overlap"],
            [
                ("layer", "layer"),
                ("layer acc", "held_out_layer_accuracy"),
                ("final wrong layer correct", "final_wrong_layer_correct_count"),
                ("final correct layer wrong", "final_correct_layer_wrong_count"),
                ("agreement", "held_out_agreement_with_final"),
            ],
        ),
        "",
        "## Coefficient Norms",
        "",
        *markdown_table(
            result["coefficient_norms"],
            [
                ("head", "head"),
                ("kind", "kind"),
                ("mean", "mean"),
                ("std", "std"),
                ("p05", "p05"),
                ("median", "median"),
                ("p95", "p95"),
                ("near zero", "near_zero_fraction"),
            ],
        ),
        "",
        "## Low-Frequency Coefficient Recovery",
        "",
        *markdown_table(
            freq_rows,
            [
                ("head", "head"),
                ("freq", "frequency"),
                ("weight", "head_weight"),
                ("R2", "r2"),
                ("scaled R2", "scalar_fit_r2"),
                ("pair cosine", "mean_pair_cosine"),
            ],
            limit=60,
        ),
        "",
        "## Error Offsets",
        "",
        *markdown_table(
            result["offset_summary"],
            [("variant", "variant"), ("offset", "signed_offset"), ("count", "count"), ("fraction", "fraction")],
            limit=80,
        ),
        "",
        "## Failure Audit",
        "",
        *markdown_table(
            result["failure_cases"],
            [
                ("a", "a"),
                ("b", "b"),
                ("sum", "true_sum"),
                ("pred", "prediction"),
                ("offset", "signed_offset"),
                ("confidence", "confidence"),
                ("margin", "final_margin"),
                ("l0", "layer_0_prediction"),
                ("l1", "layer_1_prediction"),
                ("l2", "layer_2_prediction"),
                ("l3", "layer_3_prediction"),
            ],
            limit=30,
        ),
        "",
        "## Feature Geometry",
        "",
        f"Final feature addition-diagonal energy: `{result['feature_grid_fft']['energy_fractions']['addition_diagonal']:.6f}`.",
        f"Final feature nearest-train same-sum rate: `{result['nearest_train']['feature_nearest_train_same_sum_rate']:.6f}`.",
        f"Final-logit addition residual energy: `{result['logit_fourier']['energy_fractions']['addition_residual']:.6f}`.",
        f"Held-out calibration ECE: `{result['calibration']['ece']:.6f}`.",
        "",
        "## Adapter Control",
        "",
    ]
    if "adapter_disabled_metrics" in result:
        lines.extend(
            markdown_table(
                result["adapter_disabled_metrics"],
                [("split", "split"), ("accuracy", "accuracy"), ("loss", "loss"), ("within1", "within_1")],
            )
        )
    else:
        lines.append("No adapter-disable control applies to this readout.")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/layerwise_accuracy.png`",
            "- `figures/final_error_offsets.png`",
            "- `figures/frequency_cutoff.png`",
            "- `figures/coefficient_r2_by_head.png`",
            "- `figures/accuracy_by_sum.png`",
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def interpretation_text(result: dict[str, Any]) -> str:
    readout_type = result["readout_type"]
    rows = {row["name"]: row for row in result["behavior"]["summary_rows"]}
    held = rows["held_out"]["accuracy"]
    wrap = rows["held_out_wrap"]["accuracy"]
    nowrap = rows["held_out_no_wrap"]["accuracy"]
    layer_rows = {row["variant"]: row for row in result["component_metrics"] if row["split"] == "held_out"}
    if readout_type == "layerwise_dirac_adapter":
        return (
            "The adapter architecture learns real cyclic structure, but the feedback path is not benign. "
            f"Held-out accuracy is {held:.6f}, with a large wrap/no-wrap gap ({wrap:.6f} vs {nowrap:.6f}). "
            "The layer heads contain useful residue predictors, but feeding their coefficients back into the next layer appears to make optimization and wrap generalization worse. "
            "The failure mode is mostly local adjacent-residue confusion, not a small ordinary-sum cutoff."
        )
    if readout_type == "layerwise_dirac_residual":
        weights = result.get("residual_weights") or []
        weight_text = ", ".join(f"{w:.3f}" for w in weights)
        return (
            "The residual layerwise architecture is the strongest scratch result. "
            f"Held-out accuracy is {held:.6f}, and the learned layer weights are [{weight_text}], so the final layer dominates while earlier layer logits still contribute. "
            "This supports a final-layer cyclic rule regularized by intermediate Dirac/Fourier heads, not a uniformly distributed vote across all circuit layers. "
            "Remaining failures are mostly adjacent-margin errors with very few nonlocal aliases."
        )
    final_delta = layer_rows.get("final_delta_bias", {}).get("accuracy", float("nan"))
    last_layer = layer_rows.get("layer_3_full", {}).get("accuracy", float("nan"))
    return (
        "The auxiliary architecture shows that per-layer Dirac/Fourier supervision helps train a strong global cyclic rule from scratch. "
        f"The final head reaches {held:.6f}, final delta-plus-bias reaches {final_delta:.6f}, and the last auxiliary layer reaches {last_layer:.6f}. "
        "The model is not limited to sums below 50; its weakness is local residue-boundary precision, especially around wrap and high ordinary sums."
    )


def analyze_checkpoint(args: argparse.Namespace, checkpoint: Path, out_dir: Path) -> dict[str, Any]:
    device = resolve_device(args.device)
    model, config, checkpoint_obj = load_model(checkpoint, device)
    modulus = int(config["dataset"].get("modulus", 97))
    train_fraction = float(config["dataset"].get("train_fraction", 0.3))
    split_seed = int(config["dataset"].get("seed", 0))
    train_mask = train_mask_for_split(modulus, train_fraction, split_seed)
    a, b, labels = all_pairs(modulus)
    masks = split_masks(train_mask, modulus)
    cutoffs = [int(x) for x in str(args.cutoffs).split(",") if x.strip()]

    collected = collect_layerwise(model, modulus=modulus, device=device, batch_size=int(args.batch_size))
    variants, cutoff_variants = variant_logits(collected, readout_type=model.readout_type, labels=labels, cutoffs=cutoffs)
    component_rows = rows_for_variant_set(variants, labels, masks, modulus)
    cutoff_rows = rows_for_variant_set(cutoff_variants, labels, masks, modulus)
    selected_for_fits = ["final_full", "final_delta_bias", "final_head_recomputed", "layer_0_full", "layer_1_full", "layer_2_full", "layer_3_full", "layer_uniform_mean", "residual_weighted"]
    residual_rows = residual_fit_rows(variants, train_mask, modulus, selected_for_fits)
    n_layers = len(collected["layers"])
    overlap = layer_overlap_rows(variants, labels, train_mask, n_layers)
    failures = failure_cases(variants, labels, train_mask, modulus, n_layers)
    offset_rows = offset_summary(
        variants,
        labels,
        train_mask,
        modulus,
        ["final_full", "final_delta_bias", "layer_3_full", "layer_uniform_mean", "residual_weighted"],
    )
    coeff_norms = layer_head_summaries(collected, train_mask, labels, modulus)
    freq = frequency_rows(collected, labels, train_mask, modulus)
    pred = variants["final_full"].argmax(axis=1)
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "run_dir": str(Path(args.run_dir) if args.run_dir else checkpoint.parent),
        "checkpoint_step": int(checkpoint_obj.get("step", -1)),
        "variant": str(model.variant),
        "readout_type": str(model.readout_type),
        "modulus": int(modulus),
        "n_layers": int(n_layers),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "residual_weights": collected.get("residual_weights").tolist() if collected.get("residual_weights") is not None else None,
        "behavior": behavior_strata(variants["final_full"], labels, train_mask, modulus=modulus),
        "component_metrics": component_rows,
        "cutoff_metrics": cutoff_rows,
        "residual_fit_rows": residual_rows,
        "layer_overlap": overlap,
        "failure_cases": failures,
        "offset_summary": offset_rows,
        "coefficient_norms": coeff_norms,
        "frequency_diagnostics": freq,
        "calibration": calibration(variants["final_full"], labels, ~train_mask, bins=int(args.calibration_bins)),
        "logit_fourier": addition_fourier(variants["final_full"].reshape(modulus, modulus, modulus), modulus),
        "feature_grid_fft": feature_grid_fft(collected["final_features"], modulus),
        "feature_fourier_probes": feature_fourier_probes(collected["final_features"], train_mask, modulus),
        "nearest_train": nearest_train_diagnostics(collected["final_features"], pred, labels, train_mask, modulus),
    }
    result["limited_range"] = summarize_limited_range(result)

    if model.readout_type == "layerwise_dirac_adapter":
        disabled = collect_layerwise(model, modulus=modulus, device=device, batch_size=int(args.batch_size), adapter_scale_override=0.0)
        disabled_logits = {"adapter_disabled_final": disabled["final_logits"]}
        result["adapter_disabled_metrics"] = rows_for_variant_set(disabled_logits, labels, masks, modulus)
    result["interpretation"] = interpretation_text(result)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "component_metrics.csv", component_rows)
    write_csv(out_dir / "cutoff_metrics.csv", cutoff_rows)
    write_csv(out_dir / "residual_fit_rows.csv", residual_rows)
    write_csv(out_dir / "layer_overlap.csv", overlap)
    write_csv(out_dir / "failure_cases.csv", failures)
    write_csv(out_dir / "offset_summary.csv", offset_rows)
    write_csv(out_dir / "coefficient_norms.csv", coeff_norms)
    write_csv(out_dir / "frequency_diagnostics.csv", freq)
    if "adapter_disabled_metrics" in result:
        write_csv(out_dir / "adapter_disabled_metrics.csv", result["adapter_disabled_metrics"])
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(result), f, indent=2)
    make_plots(out_dir, result, variants, cutoff_rows, labels, train_mask, modulus)
    write_report(out_dir / "QNN_LAYERWISE_DIRAC_EXHAUSTIVE_REPORT.md", result)
    return result


def aggregate_report(out_dir: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        summary = {row["name"]: row for row in result["behavior"]["summary_rows"]}
        comp = {row["variant"]: row for row in result["component_metrics"] if row["split"] == "held_out"}
        rows.append(
            {
                "architecture": result["readout_type"],
                "held_out": summary["held_out"]["accuracy"],
                "train": summary["train"]["accuracy"],
                "wrap": summary["held_out_wrap"]["accuracy"],
                "no_wrap": summary["held_out_no_wrap"]["accuracy"],
                "sum_le_50": summary["held_out_sum_le_50"]["accuracy"],
                "within_1": comp["final_full"]["within_1"],
                "top2": comp["final_full"]["top2_accuracy"],
                "mod_add_r2": next(
                    row["r2"]
                    for row in result["residual_fit_rows"]
                    if row["variant"] == "final_full" and row["split"] == "held_out" and row["fit"] == "mod_add"
                ),
                "addition_energy": result["logit_fourier"]["energy_fractions"]["addition_residual"],
                "feature_addition_energy": result["feature_grid_fft"]["energy_fractions"]["addition_diagonal"],
                "report": str(Path(result["out_dir"]) / "QNN_LAYERWISE_DIRAC_EXHAUSTIVE_REPORT.md") if "out_dir" in result else "",
            }
        )
    write_csv(out_dir / "architecture_summary.csv", rows)
    lines = [
        "# QNN Layerwise Dirac Architecture Comparison",
        "",
        "This report compares the three from-scratch layerwise Dirac/Fourier QNN architectures.",
        "",
        *markdown_table(
            rows,
            [
                ("architecture", "architecture"),
                ("held-out", "held_out"),
                ("train", "train"),
                ("wrap", "wrap"),
                ("no-wrap", "no_wrap"),
                ("within one", "within_1"),
                ("mod-add R2", "mod_add_r2"),
                ("addition energy", "addition_energy"),
                ("feature add energy", "feature_addition_energy"),
            ],
        ),
        "",
        "## Interpretation",
        "",
        "The residual layerwise architecture is the strongest scratch architecture. It keeps the final layer dominant while letting earlier Dirac/Fourier heads contribute regularizing evidence. The auxiliary architecture is also strong and shows that deep residue supervision is useful. The adapter architecture is the negative control: feeding residue coefficients back into the next quantum layer reduces held-out accuracy and hurts wrap generalization.",
        "",
        "Across all three, the small ordinary-sum cutoff hypothesis is not supported. The failures are primarily adjacent residue margins and a few nonlocal aliases, consistent with imperfect sharpening of a global cyclic rule.",
        "",
    ]
    (out_dir / "QNN_LAYERWISE_DIRAC_ARCHITECTURE_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exhaustive mech interp for layerwise Dirac/Fourier QNN checkpoints.")
    parser.add_argument("--checkpoint", action="append", default=None, help="Checkpoint path. May be passed multiple times.")
    parser.add_argument("--run-dir", default=None, help="Run directory for a single checkpoint.")
    parser.add_argument("--out-dir", default="analysis/qnn_layerwise_dirac_exhaustive")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cutoffs", default="1,2,3,5,8,13,21")
    parser.add_argument("--calibration-bins", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")
    root = Path(args.out_dir)
    results = []
    for ckpt_str in args.checkpoint:
        ckpt = Path(ckpt_str)
        name = ckpt.parent.name
        out_dir = root / name
        result = analyze_checkpoint(args, ckpt, out_dir)
        result["out_dir"] = str(out_dir)
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(result), f, indent=2)
        results.append(result)
    if len(results) > 1:
        aggregate_report(root, results)


if __name__ == "__main__":
    main()
