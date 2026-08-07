from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ORDERED_ROUTES, ROUTE_PROBLEM_MODES, ModularPairsDataset


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: ModularPairsDataset, *, batch_size: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    correct = 0
    loss_sum = 0.0
    route_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0.0, "correct": 0.0, "loss": 0.0})
    q_rows = []
    feature_rows = []
    a_rows = []
    b_rows = []
    y_rows = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        logits = out["logits"]
        loss = F.cross_entropy(logits, batch["y"], reduction="none")
        pred = logits.argmax(dim=-1)
        ok = pred.eq(batch["y"])
        total += int(batch["y"].numel())
        correct += int(ok.sum().item())
        loss_sum += float(loss.sum().item())
        q_rows.append(out["q_hat"].detach().cpu())
        feature_rows.append(out["features"].detach().cpu())
        a_rows.append(batch["a"].detach().cpu())
        b_rows.append(batch["b"].detach().cpu())
        y_rows.append(batch["y"].detach().cpu())
        if dataset.problem_mode in ROUTE_PROBLEM_MODES:
            for route_idx in batch["route_id"].detach().cpu().unique().tolist():
                mask = batch["route_id"].detach().cpu().eq(int(route_idx))
                route = ORDERED_ROUTES[int(route_idx)]
                stats = route_stats[route]
                stats["n"] += float(mask.sum().item())
                stats["correct"] += float(ok.detach().cpu()[mask].sum().item())
                stats["loss"] += float(loss.detach().cpu()[mask].sum().item())

    result: dict[str, Any] = {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "n": total,
    }
    if route_stats:
        result["routes"] = {
            route: {
                "accuracy": stats["correct"] / max(stats["n"], 1.0),
                "loss": stats["loss"] / max(stats["n"], 1.0),
                "n": int(stats["n"]),
            }
            for route, stats in sorted(route_stats.items())
        }
    q_hat = torch.cat(q_rows, dim=0).numpy()
    features = torch.cat(feature_rows, dim=0).numpy()
    records = {
        "a": torch.cat(a_rows, dim=0).numpy(),
        "b": torch.cat(b_rows, dim=0).numpy(),
        "y": torch.cat(y_rows, dim=0).numpy(),
    }
    result["fourier_addition_energy"] = fourier_addition_energy(q_hat, records, modulus=dataset.modulus)
    result["same_sum_feature_ratio"] = same_sum_variance_ratio(features, records, modulus=dataset.modulus)
    result["phase_mae"] = phase_mean_absolute_error(q_hat, records, modulus=dataset.modulus)
    return result


def fourier_addition_energy(values: np.ndarray, records: dict[str, np.ndarray], *, modulus: int) -> float:
    """Energy ratio on the addition diagonal of route-agnostic feature grids."""

    flat = values.reshape(values.shape[0], -1)
    grids = np.full((flat.shape[1], modulus, modulus), np.nan, dtype=np.float64)
    counts = np.zeros((modulus, modulus), dtype=np.float64)
    for row, a, b in zip(flat, records["a"], records["b"]):
        grids[:, int(a), int(b)] = row
        counts[int(a), int(b)] += 1.0
    observed = counts > 0
    if not np.any(observed):
        return float("nan")
    # Strict held-out splits do not cover the full p x p grid. For progress
    # telemetry we fill unobserved cells with the observed feature mean; full
    # all-pair audits are exact because no fill is applied when coverage is 1.
    if not np.all(observed):
        for idx, grid in enumerate(grids):
            mean_value = float(np.nanmean(grid))
            grids[idx] = np.where(np.isnan(grid), mean_value, grid)
    total = 0.0
    diagonal = 0.0
    for grid in grids:
        coeffs = np.fft.fft2(grid)
        energy = np.abs(coeffs) ** 2
        total += float(energy.sum())
        diagonal += float(sum(energy[k, k] for k in range(modulus)))
    return diagonal / total if total > 0.0 else float("nan")


def same_sum_variance_ratio(features: np.ndarray, records: dict[str, np.ndarray], *, modulus: int) -> float:
    sums = records["y"]
    global_center = features.mean(axis=0, keepdims=True)
    total_var = float(((features - global_center) ** 2).sum(axis=1).mean())
    within = 0.0
    n = 0
    for residue in range(modulus):
        mask = sums == residue
        if mask.sum() < 2:
            continue
        group = features[mask]
        center = group.mean(axis=0, keepdims=True)
        within += float(((group - center) ** 2).sum(axis=1).sum())
        n += int(mask.sum())
    return within / max(n, 1) / max(total_var, 1e-12)


def phase_mean_absolute_error(q_hat: np.ndarray, records: dict[str, np.ndarray], *, modulus: int) -> float:
    q = q_hat[..., 0] + 1j * q_hat[..., 1]
    ks = np.arange(1, q_hat.shape[1] + 1, dtype=np.float64)
    target = np.exp(2j * np.pi * records["y"][:, None] * ks[None, :] / float(modulus))
    err = np.angle(q * np.conj(target))
    return float(np.abs(err).mean())


@torch.no_grad()
def cross_ablation_accuracy(model: torch.nn.Module, dataset: ModularPairsDataset, *, batch_size: int, device: torch.device) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0
    correct = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch, ablate_cross=True)
        pred = out["logits"].argmax(dim=-1)
        correct += int(pred.eq(batch["y"]).sum().item())
        total += int(batch["y"].numel())
    return correct / max(total, 1)
