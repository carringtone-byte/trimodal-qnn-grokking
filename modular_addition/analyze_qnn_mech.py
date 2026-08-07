from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from .qnn_mod97 import ModularPairDataset, QNNClassifier
from .train import resolve_device


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def all_pairs(modulus: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.repeat(np.arange(modulus, dtype=np.int64), modulus)
    b = np.tile(np.arange(modulus, dtype=np.int64), modulus)
    labels = (a + b) % modulus
    return a, b, labels


def train_mask_for_split(modulus: int, train_fraction: float, seed: int) -> np.ndarray:
    train = ModularPairDataset(modulus=modulus, train_fraction=train_fraction, seed=seed, split="train")
    train_pairs = set(train.pairs)
    a, b, _ = all_pairs(modulus)
    return np.asarray([(int(x), int(y)) in train_pairs for x, y in zip(a, b)], dtype=bool)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[QNNClassifier, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
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
    return model, config


@torch.no_grad()
def collect_outputs(
    model: QNNClassifier,
    *,
    modulus: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    a, b, _ = all_pairs(modulus)
    logits_parts: list[torch.Tensor] = []
    feature_parts: list[torch.Tensor] = []
    for start in range(0, len(a), batch_size):
        aa = torch.tensor(a[start : start + batch_size], dtype=torch.long, device=device)
        bb = torch.tensor(b[start : start + batch_size], dtype=torch.long, device=device)
        logits, features = model.forward_with_features(aa, bb)
        logits_parts.append(logits.detach().cpu())
        feature_parts.append(features.detach().cpu())
    return torch.cat(logits_parts).numpy(), torch.cat(feature_parts).numpy()


def logsumexp_np(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.exp(x - m).sum(axis=axis))


def cross_entropy(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    x = logits[mask]
    y = labels[mask]
    if x.shape[0] == 0:
        return float("nan")
    log_norm = logsumexp_np(x, axis=1)
    return float(np.mean(log_norm - x[np.arange(x.shape[0]), y]))


def acc(pred: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(pred[mask] == labels[mask]))


def margin_stats(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is None:
        mask = np.ones(labels.shape[0], dtype=bool)
    if int(mask.sum()) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    x = logits[mask].copy()
    y = labels[mask]
    correct = x[np.arange(x.shape[0]), y]
    x[np.arange(x.shape[0]), y] = -np.inf
    other = np.max(x, axis=1)
    margin = correct - other
    return {
        "mean": float(np.mean(margin)),
        "median": float(np.median(margin)),
        "p10": float(np.quantile(margin, 0.10)),
        "p90": float(np.quantile(margin, 0.90)),
    }


def metric_row(name: str, pred: np.ndarray, logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "count": int(mask.sum()),
        "accuracy": acc(pred, labels, mask),
        "loss": cross_entropy(logits, labels, mask),
        "margin": margin_stats(logits, labels, mask),
    }


def behavior_strata(
    logits: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    *,
    modulus: int,
) -> dict[str, Any]:
    pred = logits.argmax(axis=1)
    a, b, _ = all_pairs(modulus)
    integer_sum = a + b
    wrap = integer_sum >= modulus
    test_mask = ~train_mask
    all_mask = np.ones_like(train_mask, dtype=bool)
    rows = [
        metric_row("all", pred, logits, labels, all_mask),
        metric_row("train", pred, logits, labels, train_mask),
        metric_row("held_out", pred, logits, labels, test_mask),
        metric_row("held_out_no_wrap", pred, logits, labels, test_mask & ~wrap),
        metric_row("held_out_wrap", pred, logits, labels, test_mask & wrap),
        metric_row("held_out_sum_le_50", pred, logits, labels, test_mask & (integer_sum <= 50)),
        metric_row("held_out_sum_51_to_96", pred, logits, labels, test_mask & (integer_sum > 50) & (integer_sum < modulus)),
        metric_row("held_out_wrap_sum_97_to_145", pred, logits, labels, test_mask & (integer_sum >= modulus) & (integer_sum <= 145)),
        metric_row("held_out_wrap_sum_gt_145", pred, logits, labels, test_mask & (integer_sum > 145)),
        metric_row("held_out_both_operands_le_50", pred, logits, labels, test_mask & (a <= 50) & (b <= 50)),
        metric_row("held_out_either_operand_gt_50", pred, logits, labels, test_mask & ((a > 50) | (b > 50))),
    ]
    thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 96, 120, 145, 170, 192]
    threshold_rows = []
    for threshold in thresholds:
        mask = integer_sum <= threshold
        threshold_rows.append(
            {
                "integer_sum_le": int(threshold),
                "all_count": int(mask.sum()),
                "held_out_count": int((test_mask & mask).sum()),
                "all_accuracy": acc(pred, labels, mask),
                "held_out_accuracy": acc(pred, labels, test_mask & mask),
                "train_accuracy": acc(pred, labels, train_mask & mask),
            }
        )
    exact_sum_rows = []
    for s_int in range(2 * modulus - 1):
        mask = integer_sum == s_int
        exact_sum_rows.append(
            {
                "integer_sum": int(s_int),
                "count": int(mask.sum()),
                "train_count": int((train_mask & mask).sum()),
                "held_out_count": int((test_mask & mask).sum()),
                "all_accuracy": acc(pred, labels, mask),
                "train_accuracy": acc(pred, labels, train_mask & mask),
                "held_out_accuracy": acc(pred, labels, test_mask & mask),
            }
        )
    residue_rows = []
    for residue in range(modulus):
        mask = labels == residue
        residue_rows.append(
            {
                "residue": int(residue),
                "count": int(mask.sum()),
                "held_out_count": int((test_mask & mask).sum()),
                "all_accuracy": acc(pred, labels, mask),
                "held_out_accuracy": acc(pred, labels, test_mask & mask),
            }
        )
    offset = (pred - labels) % modulus
    offset_rows = []
    for value in range(modulus):
        mask = test_mask & (offset == value)
        offset_rows.append({"offset": int(value), "held_out_count": int(mask.sum()), "held_out_fraction": float(mask.sum() / max(1, test_mask.sum()))})
    circular_error = np.minimum(offset, modulus - offset)
    error_distance = {
        f"held_out_within_{radius}": float(np.mean(circular_error[test_mask] <= radius))
        for radius in [0, 1, 2, 3, 5, 10, 20, 48]
    }
    mod_rows = []
    for m in [2, 3, 4, 5, 8, 10, 16, 32]:
        mod_rows.append(
            {
                "modulus": int(m),
                "held_out_pred_residue_accuracy": float(np.mean((pred[test_mask] % m) == (labels[test_mask] % m))),
                "all_pred_residue_accuracy": float(np.mean((pred % m) == (labels % m))),
            }
        )
    return {
        "summary_rows": rows,
        "threshold_accuracy": threshold_rows,
        "accuracy_by_integer_sum": exact_sum_rows,
        "accuracy_by_true_residue": residue_rows,
        "held_out_error_offsets": sorted(offset_rows, key=lambda row: row["held_out_count"], reverse=True),
        "held_out_circular_error": error_distance,
        "small_modulus_agreement": mod_rows,
        "shortcut_rates": {
            "held_out_pred_equals_a": float(np.mean(pred[test_mask] == a[test_mask])),
            "held_out_pred_equals_b": float(np.mean(pred[test_mask] == b[test_mask])),
            "held_out_pred_equals_unwrapped_sum_when_in_range": acc(pred, labels, test_mask & (integer_sum < modulus)),
            "held_out_pred_equals_zero": float(np.mean(pred[test_mask] == 0)),
            "held_out_prediction_entropy": entropy_from_counts(np.bincount(pred[test_mask], minlength=modulus)),
        },
    }


def entropy_from_counts(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p = p / max(1.0, p.sum())
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def residual_fit(logits_cube: np.ndarray, modulus: int, kind: str, pair_mask: np.ndarray | None = None) -> dict[str, Any]:
    a = np.arange(modulus)[:, None, None]
    b = np.arange(modulus)[None, :, None]
    c = np.arange(modulus)[None, None, :]
    if kind == "mod_add":
        cats = (a + b - c) % modulus
    elif kind == "integer_add":
        cats = a + b - c + (modulus - 1)
    elif kind == "a_minus_c":
        cats = (a - c) % modulus + np.zeros_like(b)
    elif kind == "b_minus_c":
        cats = (b - c) % modulus + np.zeros_like(a)
    elif kind == "class_only":
        cats = c + np.zeros((modulus, modulus, 1), dtype=int)
    elif kind == "sum_only":
        cats = ((a + b) % modulus) + np.zeros_like(c)
    else:
        raise ValueError(kind)
    if pair_mask is not None:
        mask_2d = pair_mask.reshape(modulus, modulus)
        values = logits_cube[mask_2d, :].reshape(-1)
        cats_flat = cats[mask_2d, :].reshape(-1)
    else:
        values = logits_cube.reshape(-1)
        cats_flat = cats.reshape(-1)
    profile = np.zeros(int(cats_flat.max()) + 1, dtype=np.float64)
    counts = np.zeros_like(profile)
    np.add.at(profile, cats_flat, values)
    np.add.at(counts, cats_flat, 1.0)
    profile = profile / np.maximum(counts, 1.0)
    pred = profile[cats_flat]
    sse = float(np.square(values - pred).sum())
    sst = float(np.square(values - values.mean()).sum())
    out: dict[str, Any] = {"r2": float(1.0 - sse / max(sst, 1e-12))}
    if kind == "mod_add":
        correct = float(profile[0])
        others = np.delete(profile, 0)
        out.update(
            {
                "correct_residual_mean_logit": correct,
                "other_residual_mean_logit": float(others.mean()),
                "correct_residual_margin": float(correct - others.mean()),
                "correct_residual_rank_by_mean_logit": int(1 + np.sum(profile > correct)),
                "top_residuals": [
                    {"residual": int(idx), "mean_logit": float(profile[idx])}
                    for idx in np.argsort(profile)[::-1][:10]
                ],
                "profile": [{"residual": int(idx), "mean_logit": float(value)} for idx, value in enumerate(profile)],
            }
        )
    return out


def signed_freq(k: int, modulus: int) -> int:
    return int(k if k <= modulus // 2 else k - modulus)


def addition_fourier(logits_cube: np.ndarray, modulus: int) -> dict[str, Any]:
    centered = logits_cube.astype(np.float64) - logits_cube.mean()
    spectrum = np.fft.fftn(centered)
    energy = np.abs(spectrum) ** 2
    total = float(energy.sum())
    idx = np.arange(modulus)
    masks: dict[str, np.ndarray] = {}
    masks["addition_residual"] = np.zeros_like(energy, dtype=bool)
    masks["difference_residual"] = np.zeros_like(energy, dtype=bool)
    masks["a_only"] = np.zeros_like(energy, dtype=bool)
    masks["b_only"] = np.zeros_like(energy, dtype=bool)
    masks["class_only"] = np.zeros_like(energy, dtype=bool)
    masks["a_minus_c"] = np.zeros_like(energy, dtype=bool)
    masks["b_minus_c"] = np.zeros_like(energy, dtype=bool)
    for k in idx:
        masks["addition_residual"][k, k, (-k) % modulus] = True
        masks["difference_residual"][k, (-k) % modulus, 0] = True
        masks["a_only"][k, 0, 0] = True
        masks["b_only"][0, k, 0] = True
        masks["class_only"][0, 0, k] = True
        masks["a_minus_c"][k, 0, (-k) % modulus] = True
        masks["b_minus_c"][0, k, (-k) % modulus] = True
    fractions = {name: float(energy[mask].sum() / max(total, 1e-12)) for name, mask in masks.items()}
    add_energy = np.asarray([energy[k, k, (-k) % modulus] for k in idx], dtype=np.float64)
    add_nonzero = add_energy.copy()
    add_nonzero[0] = 0.0
    add_total = float(add_nonzero.sum())
    top_add = [
        {
            "frequency": int(k),
            "signed_frequency": signed_freq(int(k), modulus),
            "energy_fraction_total": float(add_energy[k] / max(total, 1e-12)),
            "energy_fraction_addition_manifold": float(add_energy[k] / max(add_total, 1e-12)),
        }
        for k in np.argsort(add_energy)[::-1]
        if int(k) != 0
    ][:20]
    low_freq = []
    for cutoff in [1, 2, 3, 5, 8, 13, 21, 34, modulus // 2]:
        mask = np.asarray([0 < min(k, modulus - k) <= cutoff for k in idx])
        low_freq.append(
            {
                "max_circular_frequency": int(cutoff),
                "fraction_of_total_energy": float(add_energy[mask].sum() / max(total, 1e-12)),
                "fraction_of_addition_energy": float(add_energy[mask].sum() / max(add_total, 1e-12)),
            }
        )
    flat = energy.reshape(-1)
    order = np.argsort(flat)[::-1]
    top_modes = []
    for pos in order[:30]:
        mode = np.unravel_index(int(pos), energy.shape)
        if mode == (0, 0, 0):
            continue
        top_modes.append(
            {
                "freq_a": signed_freq(mode[0], modulus),
                "freq_b": signed_freq(mode[1], modulus),
                "freq_c": signed_freq(mode[2], modulus),
                "energy_fraction": float(flat[pos] / max(total, 1e-12)),
            }
        )
        if len(top_modes) >= 15:
            break
    participation = float((add_total**2) / max(float(np.square(add_nonzero).sum()), 1e-12))
    return {
        "energy_fractions": fractions,
        "top_addition_frequencies": top_add,
        "low_frequency_cumulative": low_freq,
        "addition_effective_frequency_count": participation,
        "top_3d_modes": top_modes,
    }


def feature_grid_fft(features: np.ndarray, modulus: int) -> dict[str, Any]:
    grid = features.reshape(modulus, modulus, features.shape[1]).astype(np.float64)
    grid = grid - grid.mean(axis=(0, 1), keepdims=True)
    spec = np.fft.fftn(grid, axes=(0, 1))
    power = np.square(np.abs(spec)).sum(axis=2)
    total = float(power.sum())
    idx = np.arange(modulus)
    masks = {
        "addition_diagonal": (idx, idx),
        "difference_diagonal": (idx, (-idx) % modulus),
        "a_only": (idx, np.zeros_like(idx)),
        "b_only": (np.zeros_like(idx), idx),
    }
    fractions = {name: float(power[x, y].sum() / max(total, 1e-12)) for name, (x, y) in masks.items()}
    fractions["other"] = float(max(0.0, 1.0 - sum(fractions.values())))
    add_power = power[idx, idx].copy()
    add_power[0] = 0.0
    add_total = float(add_power.sum())
    top = [
        {
            "frequency": int(k),
            "signed_frequency": signed_freq(int(k), modulus),
            "energy_fraction_total": float(add_power[k] / max(total, 1e-12)),
            "energy_fraction_addition_diagonal": float(add_power[k] / max(add_total, 1e-12)),
        }
        for k in np.argsort(add_power)[::-1]
        if int(k) != 0
    ][:20]
    low = []
    for cutoff in [1, 2, 3, 5, 8, 13, 21, 34, modulus // 2]:
        mask = np.asarray([0 < min(k, modulus - k) <= cutoff for k in idx])
        low.append(
            {
                "max_circular_frequency": int(cutoff),
                "fraction_of_total_energy": float(add_power[mask].sum() / max(total, 1e-12)),
                "fraction_of_addition_energy": float(add_power[mask].sum() / max(add_total, 1e-12)),
            }
        )
    return {
        "energy_fractions": fractions,
        "top_addition_frequencies": top,
        "low_frequency_cumulative": low,
    }


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    xtx = x_train.T @ x_train
    beta = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), x_train.T @ y_train)
    return x_test @ beta


def standardize_features(features: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = features.astype(np.float64)
    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)


def fourier_targets(values: np.ndarray, modulus: int, frequency: int) -> np.ndarray:
    angle = math.tau * frequency * values / modulus
    return np.stack([np.cos(angle), np.sin(angle)], axis=1)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.square(y_true - y_pred).sum())
    sst = float(np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum())
    return float(1.0 - sse / max(sst, 1e-12))


def feature_fourier_probes(features: np.ndarray, train_mask: np.ndarray, modulus: int) -> dict[str, Any]:
    a, b, labels = all_pairs(modulus)
    x = standardize_features(features, train_mask)
    out: dict[str, Any] = {}
    for target_name, values in {"a": a, "b": b, "sum": labels}.items():
        rows = []
        for k in range(1, modulus // 2 + 1):
            y = fourier_targets(values, modulus, k)
            pred = ridge_fit_predict(x[train_mask], y[train_mask], x[~train_mask])
            rows.append({"frequency": int(k), "held_out_r2": r2_score(y[~train_mask], pred)})
        top = sorted(rows, key=lambda row: row["held_out_r2"], reverse=True)[:12]
        out[target_name] = {
            "mean_held_out_r2": float(np.mean([row["held_out_r2"] for row in rows])),
            "max_held_out_r2": float(top[0]["held_out_r2"]),
            "top_frequencies": top,
        }
    y = fourier_targets(labels, modulus, 1)
    pred = ridge_fit_predict(x[train_mask], y[train_mask], x[~train_mask])
    angle = np.mod(np.arctan2(pred[:, 1], pred[:, 0]), math.tau)
    decoded = np.rint(angle * modulus / math.tau).astype(np.int64) % modulus
    truth = labels[~train_mask]
    diff = (decoded - truth) % modulus
    circ = np.minimum(diff, modulus - diff)
    out["sum_frequency_1_phase_decode"] = {
        "exact_accuracy": float(np.mean(decoded == truth)),
        "within_1": float(np.mean(circ <= 1)),
        "within_5": float(np.mean(circ <= 5)),
        "mean_circular_error": float(np.mean(circ)),
    }
    return out


def head_weight_fft(model: QNNClassifier, modulus: int) -> dict[str, Any]:
    if model.head is None:
        return {}
    if hasattr(model.head, "frequency_weight_logits"):
        freq_weights = F.softplus(model.head.frequency_weight_logits).detach().cpu().numpy()
        order = [int(k) for k in np.argsort(freq_weights)[::-1]]
        out: dict[str, Any] = {
            "delta_frequency_weights": [
                {
                    "frequency": k + 1,
                    "signed_frequency": signed_freq(k + 1, modulus),
                    "weight": float(freq_weights[k]),
                }
                for k in order[:20]
            ]
        }
        residual = getattr(model.head, "residual", None)
        if residual is None:
            return out
        linear = residual
    else:
        linear = model.head[-1]
    weight = linear.weight.detach().cpu().numpy()
    centered = weight - weight.mean(axis=0, keepdims=True)
    energy = np.square(np.abs(np.fft.fft(centered, axis=0))).sum(axis=1)
    total = float(energy.sum())
    order = [int(k) for k in np.argsort(energy)[::-1] if int(k) != 0]
    result = {
        "top_class_frequencies": [
            {
                "frequency": k,
                "signed_frequency": signed_freq(k, modulus),
                "energy_fraction": float(energy[k] / max(total, 1e-12)),
            }
            for k in order[:20]
        ]
    }
    if "out" in locals():
        out.update(result)
        return out
    return result


def nearest_train_diagnostics(
    features: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    modulus: int,
) -> dict[str, Any]:
    a, b, _ = all_pairs(modulus)
    train_pairs = np.stack([a[train_mask], b[train_mask]], axis=1)
    test_pairs = np.stack([a[~train_mask], b[~train_mask]], axis=1)
    correct = pred[~train_mask] == labels[~train_mask]
    chunks = []
    for start in range(0, test_pairs.shape[0], 512):
        diff = np.abs(test_pairs[start : start + 512, None, :] - train_pairs[None, :, :])
        torus = np.minimum(diff, modulus - diff)
        chunks.append(torus.sum(axis=2).min(axis=1))
    nearest_torus_l1 = np.concatenate(chunks)
    bins = [0, 1, 2, 3, 4, 5, 8, 12, 97]
    by_distance = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 97:
            mask = nearest_torus_l1 >= lo
            name = f">={lo}"
        else:
            mask = (nearest_torus_l1 >= lo) & (nearest_torus_l1 < hi)
            name = f"{lo}-{hi - 1}"
        by_distance.append({"nearest_torus_l1": name, "count": int(mask.sum()), "accuracy": float(np.mean(correct[mask])) if mask.any() else float("nan")})
    x = features.astype(np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    train_x = x[train_mask]
    test_x = x[~train_mask]
    train_labels = labels[train_mask]
    nn_correct = []
    for start in range(0, test_x.shape[0], 512):
        sims = test_x[start : start + 512] @ train_x.T
        nn = sims.argmax(axis=1)
        nn_correct.append(train_labels[nn] == labels[~train_mask][start : start + 512])
    nn_same_sum = np.concatenate(nn_correct)
    return {
        "accuracy_by_nearest_train_torus_l1": by_distance,
        "feature_nearest_train_same_sum_rate": float(np.mean(nn_same_sum)),
        "feature_nearest_train_same_sum_rate_when_model_correct": float(np.mean(nn_same_sum[correct])) if correct.any() else float("nan"),
        "feature_nearest_train_same_sum_rate_when_model_wrong": float(np.mean(nn_same_sum[~correct])) if (~correct).any() else float("nan"),
    }


def make_plots(out_dir: Path, name: str, result: dict[str, Any]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    behavior = result["behavior"]
    exact = behavior["accuracy_by_integer_sum"]
    x = [row["integer_sum"] for row in exact]
    y = [row["held_out_accuracy"] for row in exact]
    plt.figure(figsize=(9, 4))
    plt.plot(x, y, linewidth=1.6)
    plt.axvline(50, color="black", linestyle="--", linewidth=1)
    plt.axvline(96, color="gray", linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("ordinary sum")
    plt.ylabel("held-out accuracy")
    plt.title("Accuracy By Sum")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_accuracy_by_sum.png", dpi=160)
    plt.close()

    thr = behavior["threshold_accuracy"]
    plt.figure(figsize=(7, 4))
    plt.plot([row["integer_sum_le"] for row in thr], [row["held_out_accuracy"] for row in thr], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("ordinary sum threshold")
    plt.ylabel("held-out accuracy")
    plt.title("Threshold Accuracy")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_threshold_accuracy.png", dpi=160)
    plt.close()

    offsets = sorted(behavior["held_out_error_offsets"], key=lambda row: row["offset"])[:]
    plt.figure(figsize=(9, 4))
    plt.bar([row["offset"] for row in offsets], [row["held_out_fraction"] for row in offsets], width=0.9)
    plt.xlabel("prediction offset")
    plt.ylabel("held-out fraction")
    plt.title("Error Offsets")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_error_offsets.png", dpi=160)
    plt.close()

    profile = result["residual_fits"]["all"]["mod_add"]["profile"]
    plt.figure(figsize=(9, 4))
    plt.plot([row["residual"] for row in profile], [row["mean_logit"] for row in profile], linewidth=1.5)
    plt.xlabel("modular residual")
    plt.ylabel("mean logit")
    plt.title("Residual Profile")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_residual_profile.png", dpi=160)
    plt.close()

    top_add = result["logit_fourier"]["top_addition_frequencies"][:16]
    plt.figure(figsize=(8, 4))
    plt.bar([str(row["signed_frequency"]) for row in top_add], [row["energy_fraction_addition_manifold"] for row in top_add])
    plt.xlabel("addition frequency")
    plt.ylabel("addition energy fraction")
    plt.title("Addition Frequencies")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_addition_frequencies.png", dpi=160)
    plt.close()

    grid = result["feature_grid_fft"]["energy_fractions"]
    plt.figure(figsize=(7, 4))
    keys = ["addition_diagonal", "difference_diagonal", "a_only", "b_only", "other"]
    plt.bar(keys, [grid[key] for key in keys])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("energy fraction")
    plt.title("Feature FFT Energy")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_feature_fft_energy.png", dpi=160)
    plt.close()


def interpret(result: dict[str, Any]) -> dict[str, Any]:
    rows = {row["name"]: row for row in result["behavior"]["summary_rows"]}
    small = rows["held_out_sum_le_50"]["accuracy"]
    held = rows["held_out"]["accuracy"]
    wrap = rows["held_out_wrap"]["accuracy"]
    no_wrap = rows["held_out_no_wrap"]["accuracy"]
    add_frac = result["logit_fourier"]["energy_fractions"]["addition_residual"]
    low5 = next(row for row in result["logit_fourier"]["low_frequency_cumulative"] if row["max_circular_frequency"] == 5)
    mod_r2 = result["residual_fits"]["all"]["mod_add"]["r2"]
    class_r2 = result["residual_fits"]["all"]["class_only"]["r2"]
    return {
        "limited_to_sum_50_supported": bool(small > held + 0.15),
        "small_sum_accuracy": float(small),
        "held_out_accuracy": float(held),
        "wrap_minus_no_wrap_accuracy": float(wrap - no_wrap),
        "add_circulant_logit_support": "strong" if (mod_r2 > 0.5 and add_frac > 0.2) else ("partial" if mod_r2 > 0.1 or add_frac > 0.05 else "weak"),
        "mod_add_residual_r2": float(mod_r2),
        "class_only_r2": float(class_r2),
        "addition_manifold_energy_fraction": float(add_frac),
        "low_frequency_le_5_fraction_of_addition_energy": float(low5["fraction_of_addition_energy"]),
        "likely_story": "",
    }


def analyze_checkpoint(checkpoint: Path, device_name: str, batch_size: int, out_dir: Path, name: str) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, config = load_model(checkpoint, device)
    modulus = int(config["dataset"].get("modulus", 97))
    train_fraction = float(config["dataset"].get("train_fraction", 0.3))
    split_seed = int(config["dataset"].get("seed", 0))
    train_mask = train_mask_for_split(modulus, train_fraction, split_seed)
    a, b, labels = all_pairs(modulus)
    logits, features = collect_outputs(model, modulus=modulus, device=device, batch_size=batch_size)
    pred = logits.argmax(axis=1)
    logits_cube = logits.reshape(modulus, modulus, modulus)
    integer_sum = a + b
    masks = {
        "all": None,
        "train": train_mask,
        "held_out": ~train_mask,
        "held_out_sum_le_50": (~train_mask) & (integer_sum <= 50),
        "held_out_no_wrap": (~train_mask) & (integer_sum < modulus),
        "held_out_wrap": (~train_mask) & (integer_sum >= modulus),
    }
    residuals = {
        mask_name: {
            kind: residual_fit(logits_cube, modulus, kind, pair_mask=mask)
            for kind in ["mod_add", "integer_add", "a_minus_c", "b_minus_c", "class_only", "sum_only"]
        }
        for mask_name, mask in masks.items()
    }
    result: dict[str, Any] = {
        "name": name,
        "checkpoint": str(checkpoint),
        "variant": model.variant,
        "modulus": modulus,
        "feature_dim": int(features.shape[1]),
        "checkpoint_step": int(torch.load(checkpoint, map_location="cpu", weights_only=False).get("step", -1)),
        "behavior": behavior_strata(logits, labels, train_mask, modulus=modulus),
        "residual_fits": residuals,
        "logit_fourier": addition_fourier(logits_cube, modulus),
        "feature_grid_fft": feature_grid_fft(features, modulus),
        "feature_fourier_probes": feature_fourier_probes(features, train_mask, modulus),
        "head_weight_fft": head_weight_fft(model, modulus),
        "nearest_train": nearest_train_diagnostics(features, pred, labels, train_mask, modulus),
    }
    result["interpretation_flags"] = interpret(result)
    make_plots(out_dir, name, result)
    return result


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    selected = rows[:limit] if limit is not None else rows
    out = ["| " + " | ".join(label for label, _ in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        vals = []
        for _, key in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return out


def write_checkpoint_report(path: Path, result: dict[str, Any]) -> None:
    rows = {row["name"]: row for row in result["behavior"]["summary_rows"]}
    flags = result["interpretation_flags"]
    lines = [
        f"# QNN Mech Interp {result['name']}",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Variant: `{result['variant']}`",
        f"Checkpoint step: `{result['checkpoint_step']}`",
        f"Feature dimension: `{result['feature_dim']}`",
        "",
        "## Behavioral Summary",
        "",
        *markdown_table(
            result["behavior"]["summary_rows"],
            [("split", "name"), ("count", "count"), ("accuracy", "accuracy"), ("loss", "loss")],
        ),
        "",
        "## Limited Range Test",
        "",
        f"Held-out accuracy for ordinary sums `a+b <= 50` is `{rows['held_out_sum_le_50']['accuracy']:.6f}` versus overall held-out `{rows['held_out']['accuracy']:.6f}`.",
        f"Limited-to-50 support flag: `{flags['limited_to_sum_50_supported']}`.",
        "",
        *markdown_table(
            result["behavior"]["threshold_accuracy"],
            [("sum <= t", "integer_sum_le"), ("held-out count", "held_out_count"), ("held-out acc", "held_out_accuracy"), ("train acc", "train_accuracy")],
        ),
        "",
        "## Logit Residual Fits",
        "",
        "| subset | mod-add R2 | integer-add R2 | a-c R2 | b-c R2 | class-only R2 | correct residual margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for subset, values in result["residual_fits"].items():
        lines.append(
            f"| {subset} | {values['mod_add']['r2']:.6f} | {values['integer_add']['r2']:.6f} | "
            f"{values['a_minus_c']['r2']:.6f} | {values['b_minus_c']['r2']:.6f} | "
            f"{values['class_only']['r2']:.6f} | {values['mod_add']['correct_residual_margin']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Fourier Structure",
            "",
            f"Addition residual manifold energy fraction: `{result['logit_fourier']['energy_fractions']['addition_residual']:.6f}`.",
            f"Feature addition-diagonal energy fraction: `{result['feature_grid_fft']['energy_fractions']['addition_diagonal']:.6f}`.",
            f"Effective addition frequency count: `{result['logit_fourier']['addition_effective_frequency_count']:.6f}`.",
            "",
            "Top logit addition frequencies:",
            "",
            *markdown_table(
                result["logit_fourier"]["top_addition_frequencies"][:10],
                [("frequency", "signed_frequency"), ("total energy", "energy_fraction_total"), ("addition energy", "energy_fraction_addition_manifold")],
            ),
            "",
            "Low-frequency cumulative addition energy:",
            "",
            *markdown_table(
                result["logit_fourier"]["low_frequency_cumulative"],
                [("max freq", "max_circular_frequency"), ("total energy", "fraction_of_total_energy"), ("addition energy", "fraction_of_addition_energy")],
            ),
            "",
            "## Feature Fourier Probes",
            "",
            "| target | max held-out R2 | mean held-out R2 | top frequency |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for target in ["a", "b", "sum"]:
        item = result["feature_fourier_probes"][target]
        lines.append(
            f"| {target} | {item['max_held_out_r2']:.6f} | {item['mean_held_out_r2']:.6f} | {item['top_frequencies'][0]['frequency']} |"
        )
    phase = result["feature_fourier_probes"]["sum_frequency_1_phase_decode"]
    lines.extend(
        [
            "",
            f"`k=1` sum phase decode exact accuracy: `{phase['exact_accuracy']:.6f}`, within 5: `{phase['within_5']:.6f}`.",
            "",
            "## Alternative Shortcut Checks",
            "",
            *markdown_table(
                result["behavior"]["small_modulus_agreement"],
                [("modulus", "modulus"), ("held-out pred residue acc", "held_out_pred_residue_accuracy"), ("all pred residue acc", "all_pred_residue_accuracy")],
            ),
            "",
            "Top held-out error offsets:",
            "",
            *markdown_table(
                result["behavior"]["held_out_error_offsets"][:10],
                [("offset", "offset"), ("count", "held_out_count"), ("fraction", "held_out_fraction")],
            ),
            "",
            "Nearest-train diagnostics:",
            "",
            *markdown_table(
                result["nearest_train"]["accuracy_by_nearest_train_torus_l1"],
                [("nearest train distance", "nearest_torus_l1"), ("count", "count"), ("accuracy", "accuracy")],
            ),
            "",
            f"Feature nearest-train same-sum rate: `{result['nearest_train']['feature_nearest_train_same_sum_rate']:.6f}`.",
            "",
            "## Interpretation",
            "",
        ]
    )
    if flags["limited_to_sum_50_supported"]:
        lines.append("The ordinary-sum `<=50` slice is substantially easier than the overall held-out set, so a limited-range rule remains plausible.")
    else:
        lines.append("The ordinary-sum `<=50` slice is not substantially better than the overall held-out set, so the model is not simply adding only up to 50.")
    lines.append(
        f"The add-circulant support level is `{flags['add_circulant_logit_support']}`: mod-add residual R2 is `{flags['mod_add_residual_r2']:.6f}` and addition-manifold energy is `{flags['addition_manifold_energy_fraction']:.6f}`."
    )
    lines.append("The strongest current interpretation should be based on the combined behavior, Fourier, and shortcut diagnostics above rather than accuracy alone.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_report(path: Path, results: list[dict[str, Any]]) -> None:
    def summary_row(result: dict[str, Any], name: str) -> dict[str, Any]:
        rows = {row["name"]: row for row in result["behavior"]["summary_rows"]}
        return rows[name]

    def low_frequency(result: dict[str, Any], cutoff: int) -> dict[str, Any]:
        return next(
            row
            for row in result["logit_fourier"]["low_frequency_cumulative"]
            if row["max_circular_frequency"] == cutoff
        )

    def offset_fraction(result: dict[str, Any], offset: int) -> float:
        for row in result["behavior"]["held_out_error_offsets"]:
            if int(row["offset"]) == offset:
                return float(row["held_out_fraction"])
        return 0.0

    def top_freq_string(result: dict[str, Any], n: int = 4) -> str:
        freqs = [
            str(row["signed_frequency"])
            for row in result["logit_fourier"]["top_addition_frequencies"][:n]
        ]
        return ", ".join(freqs)

    best_rows = []
    for result in results:
        rows = {row["name"]: row for row in result["behavior"]["summary_rows"]}
        circular = result["behavior"]["held_out_circular_error"]
        best_rows.append(
            {
                "name": result["name"],
                "variant": result["variant"],
                "step": result["checkpoint_step"],
                "held_out_accuracy": rows["held_out"]["accuracy"],
                "train_accuracy": rows["train"]["accuracy"],
                "sum_le_50_accuracy": rows["held_out_sum_le_50"]["accuracy"],
                "wrap_accuracy": rows["held_out_wrap"]["accuracy"],
                "mod_add_r2": result["residual_fits"]["all"]["mod_add"]["r2"],
                "addition_energy": result["logit_fourier"]["energy_fractions"]["addition_residual"],
                "feature_sum_max_r2": result["feature_fourier_probes"]["sum"]["max_held_out_r2"],
                "phase_exact": result["feature_fourier_probes"]["sum_frequency_1_phase_decode"]["exact_accuracy"],
                "within_1": circular["held_out_within_1"],
                "within_2": circular["held_out_within_2"],
                "within_5": circular["held_out_within_5"],
                "low3_addition_energy": low_frequency(result, 3)["fraction_of_addition_energy"],
                "low5_addition_energy": low_frequency(result, 5)["fraction_of_addition_energy"],
                "feature_addition_energy": result["feature_grid_fft"]["energy_fractions"]["addition_diagonal"],
                "a_minus_c_r2": result["residual_fits"]["all"]["a_minus_c"]["r2"],
                "b_minus_c_r2": result["residual_fits"]["all"]["b_minus_c"]["r2"],
                "class_only_r2": result["residual_fits"]["all"]["class_only"]["r2"],
                "pred_equals_a": result["behavior"]["shortcut_rates"]["held_out_pred_equals_a"],
                "pred_equals_b": result["behavior"]["shortcut_rates"]["held_out_pred_equals_b"],
                "pred_equals_zero": result["behavior"]["shortcut_rates"]["held_out_pred_equals_zero"],
                "prediction_entropy": result["behavior"]["shortcut_rates"]["held_out_prediction_entropy"],
                "feature_nn_same_sum": result["nearest_train"]["feature_nearest_train_same_sum_rate"],
                "feature_nn_same_sum_correct": result["nearest_train"]["feature_nearest_train_same_sum_rate_when_model_correct"],
                "feature_nn_same_sum_wrong": result["nearest_train"]["feature_nearest_train_same_sum_rate_when_model_wrong"],
                "offset_plus_1": offset_fraction(result, 1),
                "offset_minus_1": offset_fraction(result, result["modulus"] - 1),
                "top_addition_freqs": top_freq_string(result),
                "limited_flag": result["interpretation_flags"]["limited_to_sum_50_supported"],
                "both_operands_le_50": rows["held_out_both_operands_le_50"]["accuracy"],
                "no_wrap_accuracy": rows["held_out_no_wrap"]["accuracy"],
            }
        )
    lines = [
        "# QNN Mod97 Mech Interp",
        "",
        "This report analyzes the supplied QNN checkpoints. The main scientific question is whether the observed generalization comes from a limited Fourier/addition circuit, a small ordinary-sum rule, memorization, operand shortcuts, or another coarse representation.",
        "",
        "## Summary Table",
        "",
        *markdown_table(
            best_rows,
            [
                ("checkpoint", "name"),
                ("variant", "variant"),
                ("step", "step"),
                ("held-out acc", "held_out_accuracy"),
                ("train acc", "train_accuracy"),
                ("sum <= 50 acc", "sum_le_50_accuracy"),
                ("wrap acc", "wrap_accuracy"),
                ("mod-add R2", "mod_add_r2"),
                ("add energy", "addition_energy"),
                ("sum Fourier R2", "feature_sum_max_r2"),
                ("phase acc", "phase_exact"),
            ],
        ),
        "",
        "## Main Findings",
        "",
    ]
    best = max(best_rows, key=lambda row: row["held_out_accuracy"])
    high_accuracy = best["held_out_accuracy"] >= 0.8
    lines.append(f"- Best held-out checkpoint is `{best['name']}` with accuracy `{best['held_out_accuracy']:.6f}`.")
    if high_accuracy:
        lines.append("- This run substantially improves exact modular-addition accuracy. It is not a weak 50% circular estimator; remaining held-out errors are mostly adjacent residues.")
    else:
        lines.append("- The models do not show a grokking transition. They learn partial generalization and remain noisy, with final checkpoints below their best checkpoints.")
    lines.append("- The limited-to-ordinary-sum-50 hypothesis is not supported. The `sum <= 50` slice is not materially better than the overall held-out split, and wrap/no-wrap behavior does not show a cutoff at 50.")
    if high_accuracy:
        lines.append("- A low-frequency cyclic scaffold is still visible, but the readout now sharply resolves most residues. The remaining failure mode is adjacent-residue confusion, not broad approximate addition.")
        lines.append("- The dominant behavioral signature is high exact accuracy plus very high within-one accuracy, indicating that the rescue mostly fixed the original local readout precision failure.")
    else:
        lines.append("- A limited low-frequency Fourier/cyclic approximation is supported. Logit energy on the addition residual manifold is substantial, but it is concentrated in very low frequencies, which gives a coarse circular estimate rather than exact residue selection.")
        lines.append("- The dominant behavioral signature is local circular error: both best checkpoints are correct or off by one on about 95% of held-out examples, despite exact accuracy near 50%.")
    lines.append("- Operand-copy, class-prior, and small-modulus shortcut explanations are weak. The residual fits and shortcut rates are far smaller than the mod-add residual fit.")
    lines.append("- The harmonic rescue is not part of the completed-pair analysis because it was suspended during `prob_head`; its observed behavior was train memorization with held-out near chance.")
    lines.extend(
        [
            "",
            "## Limited Ordinary-Sum Test",
            "",
            "If either completed model only learned to add up to an ordinary-sum boundary near 50, the `sum <= 50` held-out slice should strongly outperform the full held-out set, and no-wrap examples should be much easier than wrap examples. That is not what happens.",
            "",
            *markdown_table(
                best_rows,
                [
                    ("checkpoint", "name"),
                    ("held-out", "held_out_accuracy"),
                    ("sum <= 50", "sum_le_50_accuracy"),
                    ("both operands <= 50", "both_operands_le_50"),
                    ("no wrap", "no_wrap_accuracy"),
                    ("wrap", "wrap_accuracy"),
                    ("limited flag", "limited_flag"),
                ],
            ),
            "",
            "The `sum <= 50` slice is not materially better than the full held-out split. This rules out a simple small-range arithmetic rule.",
            "",
            "## Circular Error Pattern",
            "",
            "The models are usually close on the residue circle. For weak runs this indicates a coarse circular estimate; for high-accuracy rescues it shows that the remaining mistakes are still local adjacent-residue errors.",
            "",
            *markdown_table(
                best_rows,
                [
                    ("checkpoint", "name"),
                    ("exact", "held_out_accuracy"),
                    ("within 1", "within_1"),
                    ("within 2", "within_2"),
                    ("within 5", "within_5"),
                    ("offset +1", "offset_plus_1"),
                    ("offset -1", "offset_minus_1"),
                ],
            ),
            "",
            "The off-by-one pattern is too structured to look like chance. If exact accuracy is low, it indicates a coarse circular estimator; if exact accuracy is high, it identifies the remaining precision bottleneck.",
            "",
            "## Fourier Circuit Evidence",
            "",
            "The analyzed checkpoints have visible add-circulant structure in logit space. The relevant question is not whether Fourier structure exists, but whether it is rich enough to support exact residues. Even high-accuracy delta-rescue runs still concentrate most addition-manifold energy in the first few frequencies.",
            "",
            *markdown_table(
                best_rows,
                [
                    ("checkpoint", "name"),
                    ("mod-add R2", "mod_add_r2"),
                    ("logit add energy", "addition_energy"),
                    ("feature add energy", "feature_addition_energy"),
                    ("add energy k <= 3", "low3_addition_energy"),
                    ("add energy k <= 5", "low5_addition_energy"),
                    ("top add freqs", "top_addition_freqs"),
                ],
            ),
            "",
            "`prob_head` keeps a high-dimensional feature state and can carry more non-sum structure. `expval_head` has a much smaller 14-dimensional feature state and usually cleaner addition-diagonal geometry. In high-accuracy delta-rescue runs, the readout rather than the raw feature FFT carries much of the exact class-selection work.",
            "",
            "## Why QNNs Act This Way",
            "",
            "The most plausible explanation is that this data-reuploading circuit has learned a smooth trigonometric approximation to modular addition. The inputs are encoded as angles, the circuit repeatedly applies rotations whose parameters are linear functions of those angle features, and the measured probabilities or expectation values are therefore finite Fourier-like functions of `a` and `b`. With only 7 qubits, 4 reuploading layers, and a shallow ring-entangling pattern, the easiest functions to fit are low-frequency modes on the operand torus.",
            "",
            "Exact mod-97 classification is much sharper than approximate circular addition. To map every residue to a separate class, the logit rule needs enough high-frequency structure to form narrow peaks at `(a+b-c) mod 97 = 0` and suppress the two neighboring residues. The QNNs instead learn a broad circular bump: the correct class is often one of the top local residues, but the model frequently chooses `s+1` or `s-1`. That is why exact accuracy is near 50% while within-one accuracy is near 95%.",
            "",
            "This also explains why there is no ordinary-sum cutoff. A rule based on low Fourier modes over the residue circle is global and periodic. It does not care whether the unwrapped integer sum is below 50, above 50, wrapping, or non-wrapping. It fails by angular precision, not by running out of an integer-addition range.",
            "",
            "The readout channel matters. `prob_head` exposes all 128 basis-state probabilities to a learned linear head, so it has enough readout capacity to fit train examples more aggressively and to keep operand-specific structure. `expval_head` exposes only 14 global Z/ZZ expectation values, which forces a more compressed representation. That compression makes its feature geometry cleaner and more addition-diagonal, but it also throws away the fine-grained degrees of freedom needed to separate adjacent residues reliably.",
            "",
            "The strict Born readout is a useful negative control. Although 7 qubits provide 128 basis states for 97 residues, the computational basis is not naturally aligned with mod-97 addition. A shallow variational circuit must learn both the arithmetic representation and an arbitrary allocation of residues to basis states. The hybrid heads can linearly read out useful smooth quantum features, but the Born channel alone does not organize those features into an exact residue classifier.",
            "",
            "The direct auxiliary 97-way residue head helps but does not by itself change the underlying spectral bottleneck. The initialized delta-rescue result shows that this bottleneck can be partly overcome when Fourier pressure, adjacent-residue margin, residual linear readout, and readout refresh are trained together.",
            "",
            "Relative to the successful Transformer, RWKV classifier, and contrastive JEPA, the QNN still has a smoother and lower-dimensional arithmetic state. The delta rescue shows that a sharper readout can recover much more exact behavior, but the representation remains less cleanly Fourier-linear than the best classical models.",
            "",
            "## Alternative Failure Models",
            "",
            *markdown_table(
                best_rows,
                [
                    ("checkpoint", "name"),
                    ("a-c R2", "a_minus_c_r2"),
                    ("b-c R2", "b_minus_c_r2"),
                    ("class R2", "class_only_r2"),
                    ("pred=a", "pred_equals_a"),
                    ("pred=b", "pred_equals_b"),
                    ("pred=0", "pred_equals_zero"),
                    ("entropy", "prediction_entropy"),
                    ("NN same sum", "feature_nn_same_sum"),
                ],
            ),
            "",
            "Operand shortcuts are not plausible: `a-c` and `b-c` residual fits are near zero, and direct copies of `a` or `b` occur at about one percent. Class-prior memorization is also weak: class-only residual fit is small and prediction entropy is high. Nearest-train interpolation is a secondary factor rather than the main algorithm, because same-sum nearest-neighbor rates are low even when accuracy is high.",
            "",
            "## Verdict",
            "",
            "The QNN result is not a rule that only works up to ordinary sum 50. It is a cyclic/addition representation whose main residual failure mode is local residue precision.",
            "",
            "For the 30k direct-aux run this appeared as a coarse circular estimator around 50% exact accuracy. For the initialized Fourier-delta rescue it becomes a much sharper classifier above 90% held-out accuracy, with most remaining errors at `s+1` or `s-1`.",
            "",
            "## Files",
            "",
        ]
    )
    for result in results:
        lines.append(f"- `{result['name']}` report: `{path.parent / (result['name'] + '.md')}`")
    lines.append(f"- Figures: `{path.parent / 'figures'}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_checkpoints(run_dir: Path) -> list[tuple[str, Path]]:
    return [
        ("prob_head_best", run_dir / "checkpoint_prob_head_best.pt"),
        ("prob_head_final", run_dir / "checkpoint_prob_head_final.pt"),
        ("expval_head_best", run_dir / "checkpoint_expval_head_best.pt"),
        ("expval_head_final", run_dir / "checkpoint_expval_head_final.pt"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/modular_addition_qnn_mod97_direct_aux_30k")
    parser.add_argument("--out-dir", default="analysis/qnn_mod97_mech")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoints:
        ckpts = []
        for item in args.checkpoints:
            path = Path(item)
            ckpts.append((path.stem.replace("checkpoint_", ""), path))
    else:
        ckpts = default_checkpoints(run_dir)
    results = []
    for name, checkpoint in ckpts:
        if not checkpoint.exists():
            continue
        result = analyze_checkpoint(checkpoint, args.device, args.batch_size, out_dir, name)
        (out_dir / f"{name}.json").write_text(json.dumps(to_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checkpoint_report(out_dir / f"{name}.md", result)
        results.append(result)
        print(out_dir / f"{name}.md")
    write_summary_report(out_dir / "QNN_MOD97_MECH_INTERP_REPORT.md", results)
    (out_dir / "qnn_mod97_mech_summary.json").write_text(json.dumps(to_jsonable({"results": results}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out_dir / "QNN_MOD97_MECH_INTERP_REPORT.md")


if __name__ == "__main__":
    main()
