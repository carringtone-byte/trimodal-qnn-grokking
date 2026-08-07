from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import load_config
from .data import ORDERED_ROUTES, ROUTE_PROBLEM_MODES, ModularPairsDataset, split_pairs
from .models import TrimodalQNNModel


CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pt$")
NONFINITE_RE = re.compile(r"checkpoint_nonfinite_(\d+)\.pt$")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def discover_checkpoints(run_dir: Path, *, include_nonfinite: bool = False) -> list[tuple[int, Path, bool]]:
    rows: list[tuple[int, Path, bool]] = []
    for path in run_dir.glob("checkpoint_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            rows.append((int(match.group(1)), path, False))
    if include_nonfinite:
        for path in run_dir.glob("checkpoint_nonfinite_*.pt"):
            match = NONFINITE_RE.match(path.name)
            if match:
                rows.append((int(match.group(1)), path, True))
    return sorted(rows, key=lambda item: (item[0], item[2]))


def parse_failure_events(run_name: str, run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event = row.get("event")
        if event in {"nonfinite_loss", "nonfinite_gradient"}:
            rows.append(
                {
                    "run": run_name,
                    "event": event,
                    "step": int(row.get("step", -1)),
                    "elapsed_sec": safe_float(row.get("elapsed_sec")),
                    "loss": safe_float(row.get("loss")),
                    "ce_loss": safe_float(row.get("ce_loss")),
                    "fourier_auxiliary_loss": safe_float(row.get("fourier_auxiliary_loss")),
                    "hard_neighbor_margin_loss": safe_float(row.get("hard_neighbor_margin_loss")),
                    "same_sum_loss": safe_float(row.get("same_sum_loss")),
                }
            )
    return rows


@dataclass
class SplitStats:
    n: int = 0
    correct: int = 0
    loss_sum: float = 0.0
    phase_abs_sum: float = 0.0
    q_norm_sum: float = 0.0

    def update(
        self,
        *,
        correct: np.ndarray,
        loss: np.ndarray,
        phase_abs: np.ndarray | None = None,
        q_norm: np.ndarray | None = None,
    ) -> None:
        self.n += int(correct.size)
        self.correct += int(correct.sum())
        self.loss_sum += float(loss.sum())
        if phase_abs is not None:
            self.phase_abs_sum += float(phase_abs.sum())
        if q_norm is not None:
            self.q_norm_sum += float(q_norm.sum())

    def as_dict(self, prefix: str) -> dict[str, float]:
        n = max(self.n, 1)
        return {
            f"{prefix}_n": float(self.n),
            f"{prefix}_accuracy": self.correct / n,
            f"{prefix}_loss": self.loss_sum / n,
            f"{prefix}_phase_mae": self.phase_abs_sum / n,
            f"{prefix}_q_norm": self.q_norm_sum / n,
        }


class MetricAccumulator:
    def __init__(self) -> None:
        self.stats: dict[str, SplitStats] = defaultdict(SplitStats)

    def update(
        self,
        *,
        labels: torch.Tensor,
        logits: torch.Tensor,
        split_names: np.ndarray,
        q_hat: torch.Tensor | None = None,
        route_names: np.ndarray | None = None,
    ) -> None:
        loss = F.cross_entropy(logits, labels, reduction="none").detach().cpu().numpy()
        pred = logits.argmax(dim=-1).detach().cpu().numpy()
        y_np = labels.detach().cpu().numpy()
        correct = pred == y_np
        phase_abs = None
        q_norm = None
        if q_hat is not None:
            q_np = q_hat.detach().cpu().numpy()
            phase_abs = phase_abs_error(q_np, y_np, modulus=logits.shape[-1]).mean(axis=1)
            q_norm = np.linalg.norm(q_np.reshape(q_np.shape[0], -1), axis=1)
        for split in ("all", "train", "heldout"):
            mask = np.ones_like(correct, dtype=bool) if split == "all" else split_names == split
            if mask.any():
                self.stats[split].update(
                    correct=correct[mask],
                    loss=loss[mask],
                    phase_abs=None if phase_abs is None else phase_abs[mask],
                    q_norm=None if q_norm is None else q_norm[mask],
                )
        if route_names is not None:
            for route in np.unique(route_names):
                route_mask = route_names == route
                for split in ("all", "train", "heldout"):
                    split_mask = np.ones_like(correct, dtype=bool) if split == "all" else split_names == split
                    mask = route_mask & split_mask
                    if mask.any():
                        self.stats[f"route:{route}:{split}"].update(
                            correct=correct[mask],
                            loss=loss[mask],
                            phase_abs=None if phase_abs is None else phase_abs[mask],
                            q_norm=None if q_norm is None else q_norm[mask],
                        )

    def row(self, prefixes: Iterable[str] = ("all", "train", "heldout")) -> dict[str, float]:
        out: dict[str, float] = {}
        for prefix in prefixes:
            out.update(self.stats[prefix].as_dict(prefix))
        return out


class SameSumAccumulator:
    def __init__(self, modulus: int, feature_dim: int):
        self.modulus = int(modulus)
        self.feature_dim = int(feature_dim)
        self.n = 0
        self.sum = np.zeros(feature_dim, dtype=np.float64)
        self.sumsq = 0.0
        self.group_n = np.zeros(modulus, dtype=np.int64)
        self.group_sum = np.zeros((modulus, feature_dim), dtype=np.float64)
        self.group_sumsq = np.zeros(modulus, dtype=np.float64)

    def update(self, features: np.ndarray, labels: np.ndarray) -> None:
        features = features.astype(np.float64, copy=False)
        self.n += int(features.shape[0])
        self.sum += features.sum(axis=0)
        norms = np.einsum("ij,ij->i", features, features)
        self.sumsq += float(norms.sum())
        for residue in np.unique(labels):
            mask = labels == residue
            self.group_n[int(residue)] += int(mask.sum())
            self.group_sum[int(residue)] += features[mask].sum(axis=0)
            self.group_sumsq[int(residue)] += float(norms[mask].sum())

    def ratio(self) -> float:
        if self.n <= 1:
            return float("nan")
        total_var = self.sumsq / self.n - float(np.dot(self.sum, self.sum)) / (self.n * self.n)
        within = 0.0
        for residue in range(self.modulus):
            n = int(self.group_n[residue])
            if n <= 1:
                continue
            within += self.group_sumsq[residue] - float(np.dot(self.group_sum[residue], self.group_sum[residue])) / n
        return within / self.n / max(total_var, 1e-12)


class FourierGridAccumulator:
    def __init__(self, *, modulus: int, feature_dim: int, routes: list[str]):
        self.modulus = int(modulus)
        self.feature_dim = int(feature_dim)
        self.routes = list(routes)
        self.route_grids = {
            route: np.zeros((self.feature_dim, self.modulus, self.modulus), dtype=np.float32) for route in self.routes
        }
        self.route_counts = {route: np.zeros((self.modulus, self.modulus), dtype=np.int16) for route in self.routes}
        self.mean_grid_sum = np.zeros((self.feature_dim, self.modulus, self.modulus), dtype=np.float32)
        self.mean_grid_count = np.zeros((self.modulus, self.modulus), dtype=np.int16)

    def update(self, values: np.ndarray, a: np.ndarray, b: np.ndarray, routes: np.ndarray) -> None:
        flat = values.reshape(values.shape[0], -1).astype(np.float32, copy=False)
        for route in np.unique(routes):
            mask = routes == route
            aa = a[mask].astype(np.int64)
            bb = b[mask].astype(np.int64)
            self.route_grids[str(route)][:, aa, bb] = flat[mask].T
            self.route_counts[str(route)][aa, bb] += 1
        aa_all = a.astype(np.int64)
        bb_all = b.astype(np.int64)
        self.mean_grid_sum[:, aa_all, bb_all] += flat.T
        self.mean_grid_count[aa_all, bb_all] += 1

    def energies(self) -> dict[str, float]:
        out = {}
        route_values = []
        for route in self.routes:
            if not np.all(self.route_counts[route] > 0):
                continue
            energy = addition_diagonal_energy(self.route_grids[route])
            out[f"route:{route}"] = energy
            route_values.append(energy)
        count = np.maximum(self.mean_grid_count, 1)
        mean_grid = self.mean_grid_sum / count[None, :, :]
        out["mean_route"] = addition_diagonal_energy(mean_grid)
        if route_values:
            out["route_mean"] = float(np.mean(route_values))
            out["route_min"] = float(np.min(route_values))
            out["route_max"] = float(np.max(route_values))
        return out


def addition_diagonal_energy(grids: np.ndarray) -> float:
    coeffs = np.fft.fft2(grids, axes=(-2, -1))
    energy = np.abs(coeffs) ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return float("nan")
    diag = 0.0
    for k in range(grids.shape[-1]):
        diag += float(energy[:, k, k].sum())
    return diag / total


def phase_abs_error(q_hat: np.ndarray, labels: np.ndarray, *, modulus: int) -> np.ndarray:
    q = q_hat[..., 0] + 1j * q_hat[..., 1]
    ks = np.arange(1, q_hat.shape[1] + 1, dtype=np.float64)
    target = np.exp(2j * np.pi * labels[:, None] * ks[None, :] / float(modulus))
    return np.abs(np.angle(q * np.conj(target)))


def readout_heads(model: TrimodalQNNModel) -> list[torch.nn.Module]:
    if getattr(model, "head_type", "fourier_delta") == "layerwise_dirac_mean":
        return list(model.layerwise_heads)
    return [model.head]


def positive_frequency_weights(head: torch.nn.Module) -> torch.Tensor:
    if hasattr(head, "kernel_weights"):
        weights = head.kernel_weights().detach().cpu()
    else:
        weights = torch.softmax(head.frequency_weight_logits.detach().cpu(), dim=0)
    return weights / weights.sum().clamp_min(1e-12)


def mean_frequency_weights(heads: list[torch.nn.Module]) -> torch.Tensor:
    weights = torch.stack([positive_frequency_weights(head) for head in heads], dim=0).mean(dim=0)
    return weights / weights.sum().clamp_min(1e-12)


def head_for_state(model: TrimodalQNNModel, layer_idx: int) -> torch.nn.Module:
    if getattr(model, "head_type", "fourier_delta") == "layerwise_dirac_mean":
        return model.layerwise_heads[max(0, min(layer_idx - 1, len(model.layerwise_heads) - 1))]
    return model.head


def cutoff_logits_from_head(head: torch.nn.Module, q_hat: torch.Tensor, cutoff: int) -> torch.Tensor:
    cutoff = min(int(cutoff), int(head.max_frequency))
    q = q_hat[:, :cutoff, :]
    if hasattr(head, "kernel_weights"):
        weights = head.kernel_weights()[:cutoff].to(q.device)
    else:
        weights = torch.softmax(head.frequency_weight_logits[:cutoff], dim=0)
    class_cos = head.class_cos[:, :cutoff].to(q.device)
    class_sin = head.class_sin[:, :cutoff].to(q.device)
    matched = q[:, :, 0] @ (class_cos * weights).T + q[:, :, 1] @ (class_sin * weights).T
    return torch.exp(head.scale_log) * matched + head.bias.to(q.device)


def route_names_for_batch(route_ids: torch.Tensor, problem_mode: str) -> np.ndarray:
    if problem_mode in ROUTE_PROBLEM_MODES:
        return np.array([ORDERED_ROUTES[int(idx)] for idx in route_ids.detach().cpu().numpy()], dtype=object)
    return np.array(["three_sector"] * int(route_ids.shape[0]), dtype=object)


def split_names_for_batch(a: torch.Tensor, b: torch.Tensor, train_pairs: set[tuple[int, int]]) -> np.ndarray:
    a_np = a.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()
    return np.array(["train" if (int(x), int(y)) in train_pairs else "heldout" for x, y in zip(a_np, b_np)], dtype=object)


def forward_states(model: TrimodalQNNModel, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    state, features = model.initial_state_and_features(batch)
    states = [state]
    for layer in model.layers:
        state = layer(state, features, ablate_cross=False)
        states.append(state)
    return states


def layer_name(layer_idx: int) -> str:
    return "initial" if layer_idx == 0 else f"layer_{layer_idx}"


def analyze_checkpoint(
    *,
    run_name: str,
    checkpoint_step: int,
    checkpoint_path: Path,
    is_nonfinite: bool,
    cfg: dict[str, Any],
    all_dataset: ModularPairsDataset,
    train_pairs: set[tuple[int, int]],
    device: torch.device,
    batch_size: int,
    cutoffs: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    modulus = int(cfg["data"]["modulus"])
    model = TrimodalQNNModel(cfg["model"], modulus=modulus).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    problem_mode = str(cfg["model"]["problem_mode"])
    sectors = list(model.sectors)
    routes = list(ORDERED_ROUTES) if problem_mode in ROUTE_PROBLEM_MODES else ["three_sector"]
    loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
    n_layers = len(model.layers) + 1
    dummy_state = torch.zeros(1, model.sector_count, model.state_dim, dtype=torch.complex64, device=device)
    feature_dim = int(model.measure(dummy_state).shape[-1])
    heads = readout_heads(model)
    max_frequency = int(heads[0].max_frequency)
    q_dim = max_frequency * 2

    full_metrics = MetricAccumulator()
    base_metrics = MetricAccumulator()
    cross_metrics = MetricAccumulator()
    residual_metrics = MetricAccumulator()
    cutoff_metrics = {cutoff: MetricAccumulator() for cutoff in cutoffs if cutoff <= max_frequency}
    layer_metrics = [MetricAccumulator() for _ in range(n_layers)]
    final_same_sum = {
        "all": SameSumAccumulator(modulus, feature_dim),
        "train": SameSumAccumulator(modulus, feature_dim),
        "heldout": SameSumAccumulator(modulus, feature_dim),
    }
    final_fourier = FourierGridAccumulator(modulus=modulus, feature_dim=q_dim, routes=routes)
    layer_fourier = [FourierGridAccumulator(modulus=modulus, feature_dim=q_dim, routes=routes) for _ in range(n_layers)]
    sector_mass_sum = np.zeros((n_layers, model.sector_count), dtype=np.float64)
    sector_mass_n = np.zeros(n_layers, dtype=np.int64)

    start = time.time()
    with torch.no_grad():
        for batch_cpu in loader:
            batch = batch_to_device(batch_cpu, device)
            labels = batch["y"]
            a_np = batch_cpu["a"].numpy()
            b_np = batch_cpu["b"].numpy()
            y_np = batch_cpu["y"].numpy()
            splits = split_names_for_batch(batch_cpu["a"], batch_cpu["b"], train_pairs)
            route_names = route_names_for_batch(batch_cpu["route_id"], problem_mode)

            states = forward_states(model, batch)
            final_features = None
            final_q_hat = None
            final_logits = None
            state_logits: list[torch.Tensor] = []
            state_q_hats: list[torch.Tensor] = []
            for idx, state in enumerate(states):
                measured = model.measure(state)
                head = head_for_state(model, idx)
                out = head(measured)
                logits = out["logits"]
                q_hat = out["q_hat"]
                state_logits.append(logits)
                state_q_hats.append(q_hat)
                layer_metrics[idx].update(labels=labels, logits=logits, split_names=splits, q_hat=q_hat, route_names=route_names)
                q_np = q_hat.detach().cpu().numpy()
                layer_fourier[idx].update(q_np, a_np, b_np, route_names)
                masses = state.abs().pow(2).sum(dim=-1).detach().cpu().numpy()
                sector_mass_sum[idx] += masses.sum(axis=0)
                sector_mass_n[idx] += masses.shape[0]
                if idx == n_layers - 1:
                    final_features = measured

            if getattr(model, "head_type", "fourier_delta") == "layerwise_dirac_mean":
                final_logits = torch.stack(state_logits[1:], dim=0).mean(dim=0)
                final_q_hat = torch.stack(state_q_hats[1:], dim=0).mean(dim=0)
            else:
                final_logits = state_logits[-1]
                final_q_hat = state_q_hats[-1]

            assert final_features is not None and final_q_hat is not None and final_logits is not None
            full_metrics.update(labels=labels, logits=final_logits, split_names=splits, q_hat=final_q_hat, route_names=route_names)
            if getattr(model, "head_type", "fourier_delta") == "layerwise_dirac_mean":
                base_logits = final_logits
            else:
                base_logits = model.head.logits_from_coefficients(final_q_hat)
            base_metrics.update(labels=labels, logits=base_logits, split_names=splits, q_hat=final_q_hat, route_names=route_names)

            residual_head = getattr(getattr(model, "head", None), "residual_head", None)
            if residual_head is not None:
                residual_logits = residual_head(final_features)
                residual_metrics.update(labels=labels, logits=residual_logits, split_names=splits, route_names=route_names)
            for cutoff, metric in cutoff_metrics.items():
                if getattr(model, "head_type", "fourier_delta") == "layerwise_dirac_mean":
                    cutoff_logits = torch.stack(
                        [
                            cutoff_logits_from_head(head, q_hat, cutoff)
                            for head, q_hat in zip(model.layerwise_heads, state_q_hats[1:])
                        ],
                        dim=0,
                    ).mean(dim=0)
                else:
                    cutoff_logits = cutoff_logits_from_head(model.head, final_q_hat, cutoff)
                metric.update(labels=labels, logits=cutoff_logits, split_names=splits, q_hat=final_q_hat)

            cross_out = model(batch, ablate_cross=True)
            cross_metrics.update(
                labels=labels,
                logits=cross_out["logits"],
                split_names=splits,
                q_hat=cross_out["q_hat"],
                route_names=route_names,
            )

            features_np = final_features.detach().cpu().numpy()
            final_fourier.update(final_q_hat.detach().cpu().numpy(), a_np, b_np, route_names)
            final_same_sum["all"].update(features_np, y_np)
            for split in ("train", "heldout"):
                mask = splits == split
                if mask.any():
                    final_same_sum[split].update(features_np[mask], y_np[mask])

    full_row = full_metrics.row()
    base_row = base_metrics.row()
    cross_row = cross_metrics.row()
    residual_head = getattr(getattr(model, "head", None), "residual_head", None)
    residual_row = residual_metrics.row() if residual_head is not None else {}
    final_energy = final_fourier.energies()
    freq_weights = mean_frequency_weights(heads).numpy()
    top_freq = int(np.argmax(freq_weights) + 1)
    head_scales = np.array([float(torch.exp(head.scale_log.detach().cpu())) for head in heads], dtype=np.float64)
    summary_row: dict[str, Any] = {
        "run": run_name,
        "checkpoint_step": checkpoint_step,
        "checkpoint_path": str(checkpoint_path),
        "is_nonfinite_checkpoint": is_nonfinite,
        "problem_mode": problem_mode,
        "n_qubits": int(model.n_qubits),
        "n_layers": int(len(model.layers)),
        "sector_count": int(model.sector_count),
        "state_dim": int(model.state_dim),
        "parameter_count": int(sum(param.numel() for param in model.parameters())),
        "analysis_elapsed_sec": time.time() - start,
        "head_type": str(getattr(model, "head_type", "fourier_delta")),
        "head_scale": float(head_scales.mean()),
        "head_scale_min": float(head_scales.min()),
        "head_scale_max": float(head_scales.max()),
        "frequency_entropy": float(-(freq_weights * np.log(np.clip(freq_weights, 1e-12, 1.0))).sum()),
        "top_frequency": top_freq,
        "top_frequency_weight": float(freq_weights[top_freq - 1]),
        "final_same_sum_ratio_all": final_same_sum["all"].ratio(),
        "final_same_sum_ratio_train": final_same_sum["train"].ratio(),
        "final_same_sum_ratio_heldout": final_same_sum["heldout"].ratio(),
        "final_fourier_energy_mean_route": final_energy.get("mean_route", float("nan")),
        "final_fourier_energy_route_mean": final_energy.get("route_mean", float("nan")),
        "final_fourier_energy_route_min": final_energy.get("route_min", float("nan")),
        "final_fourier_energy_route_max": final_energy.get("route_max", float("nan")),
    }
    for key, value in full_row.items():
        summary_row[f"full_{key}"] = value
    for key, value in base_row.items():
        summary_row[f"fourier_only_{key}"] = value
    for key, value in cross_row.items():
        summary_row[f"cross_ablate_{key}"] = value
    for key, value in residual_row.items():
        summary_row[f"residual_only_{key}"] = value

    route_rows: list[dict[str, Any]] = []
    for route in routes:
        row = {
            "run": run_name,
            "checkpoint_step": checkpoint_step,
            "route": route,
            "fourier_energy": final_energy.get(f"route:{route}", float("nan")),
        }
        for split in ("all", "train", "heldout"):
            stats = full_metrics.stats[f"route:{route}:{split}"].as_dict(split)
            for key, value in stats.items():
                row[f"full_{key}"] = value
            base_stats = base_metrics.stats[f"route:{route}:{split}"].as_dict(split)
            for key, value in base_stats.items():
                row[f"fourier_only_{key}"] = value
            cross_stats = cross_metrics.stats[f"route:{route}:{split}"].as_dict(split)
            for key, value in cross_stats.items():
                row[f"cross_ablate_{key}"] = value
        route_rows.append(row)

    layer_rows: list[dict[str, Any]] = []
    for idx, metric in enumerate(layer_metrics):
        energies = layer_fourier[idx].energies()
        row = {
            "run": run_name,
            "checkpoint_step": checkpoint_step,
            "layer_index": idx,
            "layer_name": layer_name(idx),
            "fourier_energy_mean_route": energies.get("mean_route", float("nan")),
            "fourier_energy_route_mean": energies.get("route_mean", float("nan")),
            "fourier_energy_route_min": energies.get("route_min", float("nan")),
            "fourier_energy_route_max": energies.get("route_max", float("nan")),
        }
        for key, value in metric.row().items():
            row[f"logit_lens_{key}"] = value
        layer_rows.append(row)

    cutoff_rows: list[dict[str, Any]] = []
    for cutoff, metric in cutoff_metrics.items():
        row = {"run": run_name, "checkpoint_step": checkpoint_step, "cutoff": cutoff}
        for key, value in metric.row().items():
            row[key] = value
        cutoff_rows.append(row)

    sector_rows: list[dict[str, Any]] = []
    for idx in range(n_layers):
        denom = max(int(sector_mass_n[idx]), 1)
        masses = sector_mass_sum[idx] / denom
        entropy = -float((masses * np.log(np.clip(masses, 1e-12, 1.0))).sum())
        for sector, mass in zip(sectors, masses):
            sector_rows.append(
                {
                    "run": run_name,
                    "checkpoint_step": checkpoint_step,
                    "layer_index": idx,
                    "layer_name": layer_name(idx),
                    "sector": sector,
                    "mean_mass": float(mass),
                    "mass_entropy": entropy,
                }
            )

    return summary_row, route_rows, layer_rows, cutoff_rows, sector_rows


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compact_filename_slug(value: str, max_len: int = 32) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if len(slug) <= max_len:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
    return f"{slug[: max_len - 9]}_{digest}"


def make_figures(out_dir: Path, summary_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]], route_rows: list[dict[str, Any]]) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    runs = sorted({str(row["run"]) for row in summary_rows})

    plt.figure(figsize=(11, 6))
    for run in runs:
        rows = sorted([row for row in summary_rows if row["run"] == run], key=lambda item: int(item["checkpoint_step"]))
        xs = [int(row["checkpoint_step"]) for row in rows]
        ys = [safe_float(row.get("full_heldout_accuracy")) for row in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=run)
    plt.xlabel("checkpoint step")
    plt.ylabel("held-out accuracy")
    plt.title("QNN checkpoint accuracy")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = fig_dir / "checkpoint_heldout_accuracy.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    plt.figure(figsize=(11, 6))
    for run in runs:
        rows = sorted([row for row in summary_rows if row["run"] == run], key=lambda item: int(item["checkpoint_step"]))
        xs = [int(row["checkpoint_step"]) for row in rows]
        ys = [safe_float(row.get("final_fourier_energy_mean_route")) for row in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=run)
    plt.xlabel("checkpoint step")
    plt.ylabel("addition-diagonal Fourier energy")
    plt.title("QNN Fourier dynamics")
    plt.legend(fontsize=7)
    plt.tight_layout()
    path = fig_dir / "checkpoint_fourier_energy.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    for run in runs:
        rows = [row for row in layer_rows if row["run"] == run]
        if not rows:
            continue
        steps = sorted({int(row["checkpoint_step"]) for row in rows})
        layers = sorted({int(row["layer_index"]) for row in rows})
        matrix = np.full((len(layers), len(steps)), np.nan)
        for row in rows:
            i = layers.index(int(row["layer_index"]))
            j = steps.index(int(row["checkpoint_step"]))
            matrix[i, j] = safe_float(row.get("logit_lens_heldout_accuracy"))
        plt.figure(figsize=(max(8, len(steps) * 0.35), 4.2))
        plt.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=max(0.65, np.nanmax(matrix)))
        plt.colorbar(label="held-out accuracy")
        plt.yticks(range(len(layers)), [layer_name(layer) for layer in layers])
        plt.xticks(range(len(steps)), steps, rotation=90, fontsize=7)
        plt.xlabel("checkpoint step")
        plt.title(f"{run}: layer logit lens")
        plt.tight_layout()
        path = fig_dir / f"{compact_filename_slug(run)}_layer_logit_lens.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    ordered_runs = [run for run in runs if "ordered_route" in run]
    for run in ordered_runs:
        rows = [row for row in route_rows if row["run"] == run]
        steps = sorted({int(row["checkpoint_step"]) for row in rows})
        routes = list(ORDERED_ROUTES)
        matrix = np.full((len(routes), len(steps)), np.nan)
        for row in rows:
            if row["route"] not in routes:
                continue
            i = routes.index(str(row["route"]))
            j = steps.index(int(row["checkpoint_step"]))
            matrix[i, j] = safe_float(row.get("full_heldout_accuracy"))
        plt.figure(figsize=(max(8, len(steps) * 0.35), 4.8))
        plt.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=max(0.45, np.nanmax(matrix)))
        plt.colorbar(label="held-out accuracy")
        plt.yticks(range(len(routes)), routes)
        plt.xticks(range(len(steps)), steps, rotation=90, fontsize=7)
        plt.xlabel("checkpoint step")
        plt.title(f"{run}: route accuracy")
        plt.tight_layout()
        path = fig_dir / f"{compact_filename_slug(run)}_route_accuracy.png"
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(str(path))

    return paths


def summarize_best(rows: list[dict[str, Any]], run: str, key: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["run"] == run and math.isfinite(safe_float(row.get(key)))]
    if not candidates:
        return None
    return max(candidates, key=lambda row: safe_float(row.get(key)))


def write_report(out_dir: Path, summary_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]], figure_paths: list[str]) -> None:
    runs = sorted({str(row["run"]) for row in summary_rows})
    lines: list[str] = []
    lines.append("# Trimodal QNN Checkpoint Mechanistic Analysis")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    if len(summary_rows) == 1:
        row = summary_rows[0]
        lines.append(
            "This focused pass evaluates one requested QNN checkpoint "
            f"(`{row['run']}` step `{int(row['checkpoint_step'])}`) with exact all-pair diagnostics, "
            "layerwise logit lenses, Fourier-energy grids, readout ablations, cross-sector ablations, "
            "route-local metrics, and sector-mass dynamics."
        )
    else:
        lines.append("This pass evaluates saved QNN checkpoints with exact all-pair diagnostics, layerwise logit lenses, Fourier-energy grids, readout ablations, cross-sector ablations, route-local metrics, and sector-mass dynamics.")
    lines.append("")
    lines.append("## Best Checkpoints")
    lines.append("")
    lines.append("| run | best step | held-out acc | Fourier-only held-out | cross-ablate held-out | Fourier energy | same-sum ratio | top frequency | head scale |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in runs:
        best = summarize_best(summary_rows, run, "full_heldout_accuracy")
        if best is None:
            continue
        lines.append(
            "| {run} | {step} | {held:.6f} | {base:.6f} | {cross:.6f} | {energy:.6f} | {ss:.6f} | {freq} | {scale:.6f} |".format(
                run=run,
                step=int(best["checkpoint_step"]),
                held=safe_float(best.get("full_heldout_accuracy")),
                base=safe_float(best.get("fourier_only_heldout_accuracy")),
                cross=safe_float(best.get("cross_ablate_heldout_accuracy")),
                energy=safe_float(best.get("final_fourier_energy_mean_route")),
                ss=safe_float(best.get("final_same_sum_ratio_heldout")),
                freq=int(safe_float(best.get("top_frequency"))),
                scale=safe_float(best.get("head_scale")),
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The exact all-pair metrics test whether the learned QNN state generalizes beyond the strict training-pair split.")
    lines.append("- Fourier-only/readout ablations show how much of the solution is carried by the explicit cyclic delta readout rather than any residual class head. For layerwise Dirac-mean checkpoints this equals the full readout because there is no residual class head.")
    lines.append("- Cross-sector ablations distinguish genuinely sector-interactive solutions from route-local or single-sector solutions.")
    lines.append("- The layerwise logit lens tests when the trained cyclic readouts can decode the sum from intermediate QNN states; this is a stricter formation diagnostic than final accuracy alone.")
    if any("ordered_route" in run for run in runs):
        lines.append("- Ordered-route models remain the harder route-composition target: they improve with 7 qubits but stay route-fragile and numerically unstable.")
    lines.append("")
    if failure_rows:
        lines.append("## Failure Events")
        lines.append("")
        lines.append("| run | event | step | loss | CE | Fourier aux | neighbor margin |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in failure_rows:
            lines.append(
                "| {run} | {event} | {step} | {loss:.6f} | {ce:.6f} | {fourier:.6f} | {neighbor:.6f} |".format(
                    run=row["run"],
                    event=row["event"],
                    step=int(row["step"]),
                    loss=safe_float(row.get("loss")),
                    ce=safe_float(row.get("ce_loss")),
                    fourier=safe_float(row.get("fourier_auxiliary_loss")),
                    neighbor=safe_float(row.get("hard_neighbor_margin_loss")),
                )
            )
        lines.append("")
    lines.append("## Figures")
    lines.append("")
    for path in figure_paths:
        rel = Path(path).relative_to(out_dir)
        lines.append(f"- `{rel}`")
    lines.append("")
    lines.append("## Tables")
    lines.append("")
    lines.append("- `checkpoint_summary.csv`: one row per checkpoint with exact behavior, ablations, Fourier energy, same-sum ratio, head scale, and frequency concentration.")
    lines.append("- `layer_logit_lens.csv`: one row per checkpoint and layer with frozen final-head decode accuracy and Fourier-energy diagnostics.")
    lines.append("- `route_summary.csv`: route-local exact behavior and Fourier-energy summaries.")
    lines.append("- `frequency_cutoffs.csv`: final-state Fourier cutoff diagnostics.")
    lines.append("- `sector_masses.csv`: mean sector mass per checkpoint and layer.")
    lines.append("- `failure_events.csv`: nonfinite events parsed from training metrics.")
    lines.append("")
    (out_dir / "TRIMODAL_QNN_CHECKPOINT_MECH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run checkpoint-level mechanistic analysis for trimodal QNN runs.")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="QNN run directory. May be repeated. Defaults to the four phase-1 lesson runs.",
    )
    parser.add_argument("--out-dir", default="trimodal_qnn_codex/analysis/phase1_checkpoint_mech")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--include-nonfinite", action="store_true")
    parser.add_argument("--cutoffs", default="1,2,3,5,8,13,21")
    parser.add_argument("--checkpoint-step", type=int, action="append", default=[], help="Analyze only the given checkpoint step. May be repeated.")
    args = parser.parse_args()

    root = Path.cwd()
    run_dirs = [Path(item) for item in args.run_dir] or [
        Path("trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons"),
        Path("trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons"),
        Path("trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons_v2"),
        Path("trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons_v2"),
    ]
    run_dirs = [(root / path).resolve() if not path.is_absolute() else path for path in run_dirs]
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    cutoffs = [int(item) for item in args.cutoffs.split(",") if item.strip()]

    summary_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    cutoff_rows: list[dict[str, Any]] = []
    sector_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"runs": [], "device": str(device), "batch_size": args.batch_size}

    for run_dir in run_dirs:
        run_name = run_dir.name
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"missing config.yaml in {run_dir}")
        cfg = load_config(config_path)
        all_dataset = ModularPairsDataset(cfg["data"], split="all")
        train_pairs, _ = split_pairs(
            int(cfg["data"]["modulus"]),
            float(cfg["data"].get("train_fraction", 0.30)),
            int(cfg["data"].get("split_seed", cfg["data"].get("seed", 0))),
        )
        train_pair_set = {(int(a), int(b)) for a, b in train_pairs}
        checkpoints = discover_checkpoints(run_dir, include_nonfinite=args.include_nonfinite)
        if args.checkpoint_step:
            wanted_steps = {int(step) for step in args.checkpoint_step}
            checkpoints = [item for item in checkpoints if int(item[0]) in wanted_steps]
            if not checkpoints:
                raise FileNotFoundError(f"no requested checkpoints {sorted(wanted_steps)} found in {run_dir}")
        failure_rows.extend(parse_failure_events(run_name, run_dir))
        manifest["runs"].append(
            {
                "run": run_name,
                "run_dir": str(run_dir),
                "checkpoints": [step for step, _, is_nonfinite in checkpoints if not is_nonfinite],
                "nonfinite_checkpoints": [step for step, _, is_nonfinite in checkpoints if is_nonfinite],
                "records": len(all_dataset),
            }
        )
        for step, checkpoint_path, is_nonfinite in checkpoints:
            print(json.dumps({"event": "analyze_checkpoint", "run": run_name, "step": step, "nonfinite": is_nonfinite}, sort_keys=True))
            try:
                summary, routes, layers, cut, sectors = analyze_checkpoint(
                    run_name=run_name,
                    checkpoint_step=step,
                    checkpoint_path=checkpoint_path,
                    is_nonfinite=is_nonfinite,
                    cfg=cfg,
                    all_dataset=all_dataset,
                    train_pairs=train_pair_set,
                    device=device,
                    batch_size=int(args.batch_size),
                    cutoffs=cutoffs,
                )
            except Exception as exc:
                failure_rows.append(
                    {
                        "run": run_name,
                        "event": "analysis_error",
                        "step": step,
                        "elapsed_sec": float("nan"),
                        "loss": float("nan"),
                        "ce_loss": float("nan"),
                        "fourier_auxiliary_loss": float("nan"),
                        "hard_neighbor_margin_loss": float("nan"),
                        "same_sum_loss": float("nan"),
                        "error": repr(exc),
                    }
                )
                if not is_nonfinite:
                    raise
                continue
            summary_rows.append(summary)
            route_rows.extend(routes)
            layer_rows.extend(layers)
            cutoff_rows.extend(cut)
            sector_rows.extend(sectors)

    write_csv(out_dir / "checkpoint_summary.csv", summary_rows)
    write_csv(out_dir / "route_summary.csv", route_rows)
    write_csv(out_dir / "layer_logit_lens.csv", layer_rows)
    write_csv(out_dir / "frequency_cutoffs.csv", cutoff_rows)
    write_csv(out_dir / "sector_masses.csv", sector_rows)
    write_csv(out_dir / "failure_events.csv", failure_rows)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    figures = make_figures(out_dir, summary_rows, layer_rows, route_rows)
    write_report(out_dir, summary_rows, failure_rows, figures)
    print(json.dumps({"event": "analysis_done", "out_dir": str(out_dir), "checkpoints": len(summary_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
