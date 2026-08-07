from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import load_config
from .data import MODALITIES, ModularPairsDataset
from .models import TrimodalQNNModel


@dataclass
class ScoreStat:
    n: int = 0
    clean_correct: int = 0
    corrupt_correct: int = 0
    loss_sum: float = 0.0
    margin_sum: float = 0.0
    clean_logprob_sum: float = 0.0
    corrupt_logprob_sum: float = 0.0

    def update(self, logits: torch.Tensor, clean_y: torch.Tensor, corrupt_y: torch.Tensor | None = None) -> None:
        if corrupt_y is None:
            corrupt_y = clean_y
        idx = torch.arange(clean_y.shape[0], device=clean_y.device)
        log_probs = F.log_softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits, clean_y, reduction="sum")
        self.n += int(clean_y.numel())
        self.clean_correct += int(pred.eq(clean_y).sum().item())
        self.corrupt_correct += int(pred.eq(corrupt_y).sum().item())
        self.loss_sum += float(loss.detach().cpu())
        self.margin_sum += float((logits[idx, clean_y] - logits[idx, corrupt_y]).detach().sum().cpu())
        self.clean_logprob_sum += float(log_probs[idx, clean_y].detach().sum().cpu())
        self.corrupt_logprob_sum += float(log_probs[idx, corrupt_y].detach().sum().cpu())

    def row(self) -> dict[str, float]:
        n = max(self.n, 1)
        return {
            "n": float(self.n),
            "clean_accuracy": self.clean_correct / n,
            "corrupt_accuracy": self.corrupt_correct / n,
            "clean_loss": self.loss_sum / n,
            "clean_minus_corrupt_margin": self.margin_sum / n,
            "clean_logprob": self.clean_logprob_sum / n,
            "corrupt_logprob": self.corrupt_logprob_sum / n,
        }


@dataclass
class FreqEffectStat:
    n: int
    sumsq: np.ndarray
    sumabs: np.ndarray

    @classmethod
    def create(cls, max_frequency: int) -> "FreqEffectStat":
        return cls(n=0, sumsq=np.zeros(max_frequency, dtype=np.float64), sumabs=np.zeros(max_frequency, dtype=np.float64))

    def update(self, diff: torch.Tensor) -> None:
        values = diff.detach().cpu().numpy()
        per_freq_sq = np.square(values).sum(axis=-1)
        per_freq_abs = np.linalg.norm(values, axis=-1)
        self.n += int(values.shape[0])
        self.sumsq += per_freq_sq.sum(axis=0)
        self.sumabs += per_freq_abs.sum(axis=0)

    def rows(self) -> list[dict[str, float]]:
        denom = max(self.n, 1)
        return [
            {
                "frequency": float(idx + 1),
                "q_mse": float(self.sumsq[idx] / denom),
                "q_l2": float(self.sumabs[idx] / denom),
                "n": float(self.n),
            }
            for idx in range(self.sumsq.shape[0])
        ]


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
    return f"layer_{idx + 1}"


def load_model(run_dir: Path, checkpoint_step: int, device: torch.device) -> tuple[TrimodalQNNModel, dict[str, Any], Path]:
    cfg = load_config(run_dir / "config.yaml")
    checkpoint_path = run_dir / f"checkpoint_{checkpoint_step}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = TrimodalQNNModel(cfg["model"], modulus=int(cfg["data"]["modulus"])).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if getattr(model, "head_type", None) != "layerwise_dirac_mean":
        raise ValueError("this causal analysis currently expects head_type=layerwise_dirac_mean")
    return model, cfg, checkpoint_path


@torch.no_grad()
def layer_readouts(
    model: TrimodalQNNModel,
    batch: dict[str, torch.Tensor],
    *,
    ablate_cross: bool = False,
    sector_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    state, features = model.initial_state_and_features(batch)
    if sector_mask is not None:
        mask = sector_mask.to(state.device, dtype=state.real.dtype).reshape(1, model.sector_count, 1)
        state = state * mask
        norm = state.abs().pow(2).sum(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-8)
        state = state / norm
    states_before = []
    states_after = []
    logits = []
    q_hats = []
    for idx, layer in enumerate(model.layers):
        states_before.append(state)
        state = layer(state, features, ablate_cross=ablate_cross)
        states_after.append(state)
        out = model.layerwise_heads[idx](model.measure(state))
        logits.append(out["logits"])
        q_hats.append(out["q_hat"])
    return {
        "features": features,
        "states_before": states_before,
        "states_after": states_after,
        "layer_logits": logits,
        "layer_q_hats": q_hats,
        "logits": torch.stack(logits, dim=0).mean(dim=0),
        "q_hat": torch.stack(q_hats, dim=0).mean(dim=0),
    }


def logits_from_q(head: torch.nn.Module, q_hat: torch.Tensor, *, cutoff: int | None = None) -> torch.Tensor:
    if cutoff is None or cutoff >= int(head.max_frequency):
        return head.logits_from_coefficients(q_hat)
    q = q_hat.clone()
    q[:, int(cutoff) :, :] = 0.0
    return head.logits_from_coefficients(q)


def corrupted_batch(batch: dict[str, torch.Tensor], *, delta: int, modulus: int) -> dict[str, torch.Tensor]:
    out = dict(batch)
    out["b"] = (batch["b"] + int(delta)) % int(modulus)
    out["y"] = (batch["y"] + int(delta)) % int(modulus)
    return out


def score_recovery(row: dict[str, Any], clean_base: dict[str, float], corrupt_base: dict[str, float]) -> dict[str, Any]:
    clean_margin = clean_base["clean_minus_corrupt_margin"]
    corrupt_margin = corrupt_base["clean_minus_corrupt_margin"]
    denom = clean_margin - corrupt_margin
    if abs(denom) < 1e-12:
        margin_recovery = float("nan")
    else:
        margin_recovery = (float(row["clean_minus_corrupt_margin"]) - corrupt_margin) / denom
    row["margin_recovery"] = margin_recovery
    row["accuracy_gain_vs_corrupt"] = float(row["clean_accuracy"]) - corrupt_base["clean_accuracy"]
    row["accuracy_drop_vs_clean"] = clean_base["clean_accuracy"] - float(row["clean_accuracy"])
    row["margin_gain_vs_corrupt"] = float(row["clean_minus_corrupt_margin"]) - corrupt_margin
    row["margin_drop_vs_clean"] = clean_margin - float(row["clean_minus_corrupt_margin"])
    return row


def frequency_band_specs(max_frequency: int) -> dict[str, list[int]]:
    specs = {
        "low_1_5": list(range(0, min(5, max_frequency))),
        "mid_6_13": list(range(5, min(13, max_frequency))),
        "high_14_21": list(range(13, min(21, max_frequency))),
        "lowmid_1_13": list(range(0, min(13, max_frequency))),
        "core_3_6": list(range(2, min(6, max_frequency))),
        "full_1_21": list(range(0, max_frequency)),
    }
    return {name: idxs for name, idxs in specs.items() if idxs}


@torch.no_grad()
def run_frequency_patching(
    model: TrimodalQNNModel,
    loader: DataLoader,
    *,
    modulus: int,
    deltas: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_frequency = int(model.layerwise_heads[0].max_frequency)
    n_layers = len(model.layerwise_heads)
    stats: dict[tuple[Any, ...], ScoreStat] = defaultdict(ScoreStat)
    baseline: dict[tuple[int, str], ScoreStat] = defaultdict(ScoreStat)

    for batch_cpu in loader:
        batch = batch_to_device(batch_cpu, next(model.parameters()).device)
        clean_y = batch["y"]
        clean = layer_readouts(model, batch)
        for delta in deltas:
            corrupt_batch_delta = corrupted_batch(batch, delta=delta, modulus=modulus)
            corrupt_y = corrupt_batch_delta["y"]
            corrupt = layer_readouts(model, corrupt_batch_delta)
            baseline[(delta, "clean")].update(clean["logits"], clean_y, corrupt_y)
            baseline[(delta, "corrupt")].update(corrupt["logits"], clean_y, corrupt_y)

            for freq_idx in range(max_frequency):
                patched_layer_logits = []
                ablated_layer_logits = []
                for layer_idx, head in enumerate(model.layerwise_heads):
                    q_restore = corrupt["layer_q_hats"][layer_idx].clone()
                    q_restore[:, freq_idx, :] = clean["layer_q_hats"][layer_idx][:, freq_idx, :]
                    patched_layer_logits.append(head.logits_from_coefficients(q_restore))

                    q_ablate = clean["layer_q_hats"][layer_idx].clone()
                    q_ablate[:, freq_idx, :] = corrupt["layer_q_hats"][layer_idx][:, freq_idx, :]
                    ablated_layer_logits.append(head.logits_from_coefficients(q_ablate))

                stats[(delta, "restore_all_layers", "all", freq_idx + 1)].update(
                    torch.stack(patched_layer_logits, dim=0).mean(dim=0), clean_y, corrupt_y
                )
                stats[(delta, "ablate_all_layers", "all", freq_idx + 1)].update(
                    torch.stack(ablated_layer_logits, dim=0).mean(dim=0), clean_y, corrupt_y
                )

                for layer_idx, head in enumerate(model.layerwise_heads):
                    restore_logits = list(corrupt["layer_logits"])
                    q_restore = corrupt["layer_q_hats"][layer_idx].clone()
                    q_restore[:, freq_idx, :] = clean["layer_q_hats"][layer_idx][:, freq_idx, :]
                    restore_logits[layer_idx] = head.logits_from_coefficients(q_restore)
                    stats[(delta, "restore_single_layer", layer_name(layer_idx), freq_idx + 1)].update(
                        torch.stack(restore_logits, dim=0).mean(dim=0), clean_y, corrupt_y
                    )

                    ablate_logits = list(clean["layer_logits"])
                    q_ablate = clean["layer_q_hats"][layer_idx].clone()
                    q_ablate[:, freq_idx, :] = corrupt["layer_q_hats"][layer_idx][:, freq_idx, :]
                    ablate_logits[layer_idx] = head.logits_from_coefficients(q_ablate)
                    stats[(delta, "ablate_single_layer", layer_name(layer_idx), freq_idx + 1)].update(
                        torch.stack(ablate_logits, dim=0).mean(dim=0), clean_y, corrupt_y
                    )

    baseline_rows: dict[str, Any] = {}
    for (delta, name), stat in baseline.items():
        baseline_rows[f"delta_{delta}_{name}"] = stat.row()

    rows = []
    for (delta, intervention, layer, frequency), stat in sorted(stats.items(), key=lambda item: item[0]):
        clean_base = baseline[(delta, "clean")].row()
        corrupt_base = baseline[(delta, "corrupt")].row()
        row: dict[str, Any] = {
            "delta": int(delta),
            "intervention": intervention,
            "layer": str(layer),
            "frequency": int(frequency),
            **stat.row(),
            "clean_baseline_accuracy": clean_base["clean_accuracy"],
            "corrupt_baseline_clean_accuracy": corrupt_base["clean_accuracy"],
            "clean_baseline_margin": clean_base["clean_minus_corrupt_margin"],
            "corrupt_baseline_margin": corrupt_base["clean_minus_corrupt_margin"],
        }
        rows.append(score_recovery(row, clean_base, corrupt_base))
    return rows, baseline_rows


@torch.no_grad()
def run_frequency_band_patching(
    model: TrimodalQNNModel,
    loader: DataLoader,
    *,
    modulus: int,
    deltas: list[int],
) -> list[dict[str, Any]]:
    max_frequency = int(model.layerwise_heads[0].max_frequency)
    bands = frequency_band_specs(max_frequency)
    stats: dict[tuple[Any, ...], ScoreStat] = defaultdict(ScoreStat)
    baseline: dict[tuple[int, str], ScoreStat] = defaultdict(ScoreStat)

    for batch_cpu in loader:
        batch = batch_to_device(batch_cpu, next(model.parameters()).device)
        clean_y = batch["y"]
        clean = layer_readouts(model, batch)
        for delta in deltas:
            corrupt_batch_delta = corrupted_batch(batch, delta=delta, modulus=modulus)
            corrupt_y = corrupt_batch_delta["y"]
            corrupt = layer_readouts(model, corrupt_batch_delta)
            baseline[(delta, "clean")].update(clean["logits"], clean_y, corrupt_y)
            baseline[(delta, "corrupt")].update(corrupt["logits"], clean_y, corrupt_y)

            for band_name, freq_indices in bands.items():
                restore_logits = []
                ablate_logits = []
                for layer_idx, head in enumerate(model.layerwise_heads):
                    q_restore = corrupt["layer_q_hats"][layer_idx].clone()
                    q_restore[:, freq_indices, :] = clean["layer_q_hats"][layer_idx][:, freq_indices, :]
                    restore_logits.append(head.logits_from_coefficients(q_restore))

                    q_ablate = clean["layer_q_hats"][layer_idx].clone()
                    q_ablate[:, freq_indices, :] = corrupt["layer_q_hats"][layer_idx][:, freq_indices, :]
                    ablate_logits.append(head.logits_from_coefficients(q_ablate))

                stats[(delta, "restore_band_all_layers", band_name)].update(
                    torch.stack(restore_logits, dim=0).mean(dim=0), clean_y, corrupt_y
                )
                stats[(delta, "ablate_band_all_layers", band_name)].update(
                    torch.stack(ablate_logits, dim=0).mean(dim=0), clean_y, corrupt_y
                )

    rows = []
    for (delta, intervention, band), stat in sorted(stats.items(), key=lambda item: item[0]):
        clean_base = baseline[(delta, "clean")].row()
        corrupt_base = baseline[(delta, "corrupt")].row()
        row: dict[str, Any] = {
            "delta": int(delta),
            "intervention": intervention,
            "band": str(band),
            **stat.row(),
            "clean_baseline_accuracy": clean_base["clean_accuracy"],
            "corrupt_baseline_clean_accuracy": corrupt_base["clean_accuracy"],
            "clean_baseline_margin": clean_base["clean_minus_corrupt_margin"],
            "corrupt_baseline_margin": corrupt_base["clean_minus_corrupt_margin"],
        }
        rows.append(score_recovery(row, clean_base, corrupt_base))
    return rows


def continue_from_state(
    model: TrimodalQNNModel,
    state: torch.Tensor,
    features: torch.Tensor,
    *,
    start_layer_idx: int,
    prefix_logits: list[torch.Tensor],
    prefix_q_hats: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = list(prefix_logits)
    q_hats = list(prefix_q_hats)
    out = model.layerwise_heads[start_layer_idx](model.measure(state))
    logits.append(out["logits"])
    q_hats.append(out["q_hat"])
    for layer_idx in range(start_layer_idx + 1, len(model.layers)):
        state = model.layers[layer_idx](state, features, ablate_cross=False)
        out = model.layerwise_heads[layer_idx](model.measure(state))
        logits.append(out["logits"])
        q_hats.append(out["q_hat"])
    return torch.stack(logits, dim=0).mean(dim=0), torch.stack(q_hats, dim=0).mean(dim=0)


def renormalize_state(state: torch.Tensor) -> torch.Tensor:
    norm = state.abs().pow(2).sum(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-8)
    return state / norm


@torch.no_grad()
def sector_scattering_for_layer(
    model: TrimodalQNNModel,
    layer_idx: int,
    state_before: torch.Tensor,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = model.layers[layer_idx]
    pre_mix = layer.apply_entanglement(layer.apply_rotations(state_before, features))
    if layer.sector_mixer is None:
        eye = torch.eye(model.sector_count, dtype=torch.complex64, device=pre_mix.device)
        u = eye
    else:
        u = layer.sector_mixer.unitary()
    path_components = u[None, :, :, None] * pre_mix[:, None, :, :]
    post_mix = path_components.sum(dim=2)
    component_power = path_components.abs().pow(2).sum(dim=-1)
    return pre_mix, post_mix, path_components, component_power


@torch.no_grad()
def run_sector_tomography(
    model: TrimodalQNNModel,
    loader: DataLoader,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    sectors = list(model.sectors)
    max_frequency = int(model.layerwise_heads[0].max_frequency)
    baseline = ScoreStat()
    path_stats: dict[tuple[int, str, str], ScoreStat] = defaultdict(ScoreStat)
    freq_stats: dict[tuple[int, str, str], FreqEffectStat] = {}
    power_sums: dict[tuple[int, str, str], float] = defaultdict(float)
    power_n: dict[tuple[int, str, str], int] = defaultdict(int)
    unitary_rows = []
    mask_stats: dict[str, ScoreStat] = defaultdict(ScoreStat)

    for layer_idx, layer in enumerate(model.layers):
        if layer.sector_mixer is None:
            u = torch.eye(model.sector_count, dtype=torch.complex64)
        else:
            u = layer.sector_mixer.unitary().detach().cpu()
        for target_idx, target in enumerate(sectors):
            for source_idx, source in enumerate(sectors):
                value = complex(u[target_idx, source_idx].item())
                unitary_rows.append(
                    {
                        "layer": layer_name(layer_idx),
                        "layer_index": layer_idx + 1,
                        "source": source,
                        "target": target,
                        "amplitude_real": value.real,
                        "amplitude_imag": value.imag,
                        "amplitude_abs": abs(value),
                        "power": abs(value) ** 2,
                        "phase": math.atan2(value.imag, value.real),
                    }
                )

    sector_masks = {}
    for idx, sector in enumerate(sectors):
        mask = torch.zeros(model.sector_count, device=device)
        mask[idx] = 1.0
        sector_masks[sector] = mask
    for i, first in enumerate(sectors):
        for j, second in enumerate(sectors):
            if j <= i:
                continue
            mask = torch.zeros(model.sector_count, device=device)
            mask[i] = 1.0
            mask[j] = 1.0
            sector_masks[f"{first}{second}"] = mask
    sector_masks["all"] = torch.ones(model.sector_count, device=device)

    for batch_cpu in loader:
        batch = batch_to_device(batch_cpu, device)
        labels = batch["y"]
        clean = layer_readouts(model, batch)
        baseline.update(clean["logits"], labels)

        for mask_name, mask in sector_masks.items():
            out = layer_readouts(model, batch, sector_mask=mask)
            mask_stats[mask_name].update(out["logits"], labels)

        for layer_idx in range(len(model.layers)):
            _, post_mix, path_components, component_power = sector_scattering_for_layer(
                model, layer_idx, clean["states_before"][layer_idx], clean["features"]
            )
            prefix_logits = clean["layer_logits"][:layer_idx]
            prefix_q_hats = clean["layer_q_hats"][:layer_idx]
            for target_idx, target in enumerate(sectors):
                for source_idx, source in enumerate(sectors):
                    key = (layer_idx + 1, source, target)
                    ablated = post_mix.clone()
                    ablated[:, target_idx, :] = ablated[:, target_idx, :] - path_components[:, target_idx, source_idx, :]
                    ablated = renormalize_state(ablated)
                    logits, q_hat = continue_from_state(
                        model,
                        ablated,
                        clean["features"],
                        start_layer_idx=layer_idx,
                        prefix_logits=prefix_logits,
                        prefix_q_hats=prefix_q_hats,
                    )
                    path_stats[key].update(logits, labels)
                    if key not in freq_stats:
                        freq_stats[key] = FreqEffectStat.create(max_frequency)
                    freq_stats[key].update(q_hat - clean["q_hat"])
                    power_sums[key] += float(component_power[:, target_idx, source_idx].sum().detach().cpu())
                    power_n[key] += int(labels.numel())

    base_row = baseline.row()
    path_rows = []
    for (layer_idx, source, target), stat in sorted(path_stats.items()):
        row = {
            "layer": f"layer_{layer_idx}",
            "layer_index": layer_idx,
            "source": source,
            "target": target,
            **stat.row(),
            "baseline_accuracy": base_row["clean_accuracy"],
            "baseline_margin": base_row["clean_minus_corrupt_margin"],
            "baseline_clean_logprob": base_row["clean_logprob"],
            "accuracy_drop": base_row["clean_accuracy"] - stat.row()["clean_accuracy"],
            "margin_drop": base_row["clean_minus_corrupt_margin"] - stat.row()["clean_minus_corrupt_margin"],
            "clean_logprob_drop": base_row["clean_logprob"] - stat.row()["clean_logprob"],
            "removed_component_power": power_sums[(layer_idx, source, target)] / max(power_n[(layer_idx, source, target)], 1),
        }
        path_rows.append(row)

    freq_rows = []
    for (layer_idx, source, target), stat in sorted(freq_stats.items()):
        for row in stat.rows():
            freq_rows.append({"layer": f"layer_{layer_idx}", "layer_index": layer_idx, "source": source, "target": target, **row})

    mask_rows = []
    for mask_name, stat in sorted(mask_stats.items()):
        row = {"mask": mask_name, **stat.row(), "baseline_accuracy": base_row["clean_accuracy"]}
        row["accuracy_drop"] = base_row["clean_accuracy"] - row["clean_accuracy"]
        mask_rows.append(row)

    return path_rows, freq_rows, unitary_rows, mask_rows


def pivot_frequency_patch(rows: list[dict[str, Any]], *, delta: int, intervention: str, value: str) -> tuple[list[int], list[str], np.ndarray]:
    subset = [row for row in rows if int(row["delta"]) == delta and row["intervention"] == intervention]
    freqs = sorted({int(row["frequency"]) for row in subset})
    layers = sorted({str(row["layer"]) for row in subset}, key=lambda item: 99 if item == "all" else int(item.split("_")[1]))
    matrix = np.full((len(layers), len(freqs)), np.nan)
    for row in subset:
        i = layers.index(str(row["layer"]))
        j = freqs.index(int(row["frequency"]))
        matrix[i, j] = float(row[value])
    return freqs, layers, matrix


def make_figures(
    out_dir: Path,
    freq_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    path_rows: list[dict[str, Any]],
    freq_effect_rows: list[dict[str, Any]],
    unitary_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
) -> list[str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    sectors = sorted(
        {str(row["source"]) for row in unitary_rows}
        | {str(row["target"]) for row in unitary_rows}
        | {str(row["source"]) for row in path_rows}
        | {str(row["target"]) for row in path_rows}
    )
    if not sectors:
        sectors = list(MODALITIES)

    all_layer_rows = [
        row
        for row in freq_rows
        if row["intervention"] == "restore_all_layers" and row["layer"] == "all" and int(row["delta"]) in {1, 5, 13}
    ]
    plt.figure(figsize=(10, 5.5))
    for delta in sorted({int(row["delta"]) for row in all_layer_rows}):
        rows = sorted([row for row in all_layer_rows if int(row["delta"]) == delta], key=lambda item: int(item["frequency"]))
        plt.plot([int(row["frequency"]) for row in rows], [float(row["margin_recovery"]) for row in rows], marker="o", linewidth=1.5, markersize=3, label=f"d={delta}")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("frequency")
    plt.ylabel("margin recovery")
    plt.title("Frequency Patch")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = fig_dir / "frequency_patch_all_layers.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    freqs, layers, matrix = pivot_frequency_patch(freq_rows, delta=1, intervention="restore_single_layer", value="margin_recovery")
    plt.figure(figsize=(11, 4.5))
    plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="margin recovery")
    plt.yticks(range(len(layers)), layers)
    plt.xticks(range(len(freqs)), freqs, fontsize=7)
    plt.xlabel("frequency")
    plt.title("Layer Frequency Patch")
    plt.tight_layout()
    path = fig_dir / "frequency_patch_layer_heatmap_delta1.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    freqs, layers, matrix = pivot_frequency_patch(freq_rows, delta=1, intervention="ablate_single_layer", value="margin_drop_vs_clean")
    plt.figure(figsize=(11, 4.5))
    plt.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="margin drop")
    plt.yticks(range(len(layers)), layers)
    plt.xticks(range(len(freqs)), freqs, fontsize=7)
    plt.xlabel("frequency")
    plt.title("Layer Frequency Ablation")
    plt.tight_layout()
    path = fig_dir / "frequency_ablation_layer_heatmap_delta1.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    restore_bands = [row for row in band_rows if row["intervention"] == "restore_band_all_layers" and int(row["delta"]) in {1, 5, 13}]
    bands = ["low_1_5", "core_3_6", "mid_6_13", "high_14_21", "lowmid_1_13", "full_1_21"]
    x = np.arange(len(bands))
    width = 0.24
    plt.figure(figsize=(10, 4.8))
    for offset_idx, delta in enumerate(sorted({int(row["delta"]) for row in restore_bands})):
        values = []
        for band in bands:
            match = [row for row in restore_bands if int(row["delta"]) == delta and row["band"] == band]
            values.append(float(match[0]["clean_accuracy"]) if match else np.nan)
        plt.bar(x + (offset_idx - 1) * width, values, width=width, label=f"d={delta}")
    plt.xticks(x, bands, rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("patched accuracy")
    plt.title("Frequency Band Patch")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = fig_dir / "frequency_band_patch_accuracy.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    layers_unique = sorted({int(row["layer_index"]) for row in unitary_rows})
    fig, axes = plt.subplots(1, len(layers_unique), figsize=(3.2 * len(layers_unique), 3.2), constrained_layout=True)
    if len(layers_unique) == 1:
        axes = [axes]
    for ax, layer_idx in zip(axes, layers_unique):
        matrix = np.zeros((len(sectors), len(sectors)))
        for row in unitary_rows:
            if int(row["layer_index"]) == layer_idx:
                i = sectors.index(str(row["target"]))
                j = sectors.index(str(row["source"]))
                matrix[i, j] = float(row["power"])
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="cividis")
        ax.set_title(f"L{layer_idx}")
        ax.set_xticks(range(len(sectors)), sectors, rotation=90, fontsize=7)
        ax.set_yticks(range(len(sectors)), sectors, fontsize=7)
        ax.set_xlabel("source")
        ax.set_ylabel("target")
    fig.colorbar(im, ax=axes, shrink=0.8, label="power")
    path = fig_dir / "sector_mixer_power.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, len(layers_unique), figsize=(3.2 * len(layers_unique), 3.2), constrained_layout=True)
    if len(layers_unique) == 1:
        axes = [axes]
    vmax = max(abs(float(row["accuracy_drop"])) for row in path_rows) if path_rows else 1.0
    for ax, layer_idx in zip(axes, layers_unique):
        matrix = np.zeros((len(sectors), len(sectors)))
        for row in path_rows:
            if int(row["layer_index"]) == layer_idx:
                i = sectors.index(str(row["target"]))
                j = sectors.index(str(row["source"]))
                matrix[i, j] = float(row["accuracy_drop"])
        im = ax.imshow(matrix, vmin=-vmax, vmax=vmax, cmap="coolwarm")
        ax.set_title(f"L{layer_idx}")
        ax.set_xticks(range(len(sectors)), sectors, rotation=90, fontsize=7)
        ax.set_yticks(range(len(sectors)), sectors, fontsize=7)
        ax.set_xlabel("source")
        ax.set_ylabel("target")
    fig.colorbar(im, ax=axes, shrink=0.8, label="accuracy drop")
    path = fig_dir / "sector_path_accuracy_drop.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    aggregate: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in freq_effect_rows:
        aggregate[(str(row["source"]), str(row["target"]), int(float(row["frequency"])))].append(float(row["q_mse"]))
    freqs = sorted({key[2] for key in aggregate})
    paths_labels = [f"{source}->{target}" for target in sectors for source in sectors]
    matrix = np.full((len(paths_labels), len(freqs)), np.nan)
    for path_idx, label in enumerate(paths_labels):
        source, target = label.split("->")
        for freq_idx, freq in enumerate(freqs):
            vals = aggregate.get((source, target, freq), [])
            if vals:
                matrix[path_idx, freq_idx] = float(np.mean(vals))
    plt.figure(figsize=(11, 5.2))
    plt.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="q MSE")
    plt.yticks(range(len(paths_labels)), paths_labels)
    plt.xticks(range(len(freqs)), freqs, fontsize=7)
    plt.xlabel("frequency")
    plt.title("Path Frequency Effect")
    plt.tight_layout()
    path = fig_dir / "sector_path_frequency_effect.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    rows = sorted(mask_rows, key=lambda row: (len(str(row["mask"])), str(row["mask"])))
    plt.figure(figsize=(8, 4.2))
    plt.bar([str(row["mask"]) for row in rows], [float(row["clean_accuracy"]) for row in rows], color="#4C78A8")
    plt.ylim(0.0, 1.0)
    plt.xlabel("sector mask")
    plt.ylabel("accuracy")
    plt.title("Sector Mask")
    plt.tight_layout()
    path = fig_dir / "sector_mask_accuracy.png"
    plt.savefig(path, dpi=180)
    plt.close()
    paths.append(str(path))

    return paths


def top_rows(rows: list[dict[str, Any]], key: str, n: int = 8, reverse: bool = True) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(key, 0.0)), reverse=reverse)[:n]


def write_report(
    out_dir: Path,
    *,
    run_name: str,
    checkpoint_step: int,
    checkpoint_path: Path,
    freq_rows: list[dict[str, Any]],
    band_rows: list[dict[str, Any]],
    freq_baselines: dict[str, Any],
    path_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    figure_paths: list[str],
    elapsed_sec: float,
) -> None:
    lines: list[str] = []
    lines.append("# Dirac-Mean Causal Mechanistic Analysis")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Run: `{run_name}`")
    lines.append(f"Checkpoint: `{checkpoint_step}`")
    lines.append(f"Checkpoint path: `{checkpoint_path}`")
    lines.append(f"Elapsed seconds: `{elapsed_sec:.3f}`")
    lines.append("")
    lines.append("## Frequency Patching")
    lines.append("")
    lines.append("Top all-layer single-frequency restorations by margin recovery:")
    lines.append("")
    lines.append("| delta | frequency | clean acc | margin recovery | acc gain |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    all_layer_restore = [row for row in freq_rows if row["intervention"] == "restore_all_layers"]
    for row in top_rows(all_layer_restore, "margin_recovery", n=10):
        lines.append(
            f"| {int(row['delta'])} | {int(row['frequency'])} | {float(row['clean_accuracy']):.6f} | "
            f"{float(row['margin_recovery']):.6f} | {float(row['accuracy_gain_vs_corrupt']):.6f} |"
        )
    lines.append("")
    lines.append("Top single-layer frequency necessities by clean-margin drop:")
    lines.append("")
    lines.append("| delta | layer | frequency | acc drop | margin drop |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    single_layer_ablate = [row for row in freq_rows if row["intervention"] == "ablate_single_layer"]
    for row in top_rows(single_layer_ablate, "margin_drop_vs_clean", n=10):
        lines.append(
            f"| {int(row['delta'])} | {row['layer']} | {int(row['frequency'])} | "
            f"{float(row['accuracy_drop_vs_clean']):.6f} | {float(row['margin_drop_vs_clean']):.6f} |"
        )
    lines.append("")
    lines.append("Frequency-band all-layer restorations:")
    lines.append("")
    lines.append("| delta | band | clean acc | margin recovery | acc gain |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    band_restore = [row for row in band_rows if row["intervention"] == "restore_band_all_layers"]
    for row in top_rows(band_restore, "clean_accuracy", n=12):
        lines.append(
            f"| {int(row['delta'])} | {row['band']} | {float(row['clean_accuracy']):.6f} | "
            f"{float(row['margin_recovery']):.6f} | {float(row['accuracy_gain_vs_corrupt']):.6f} |"
        )
    lines.append("")
    lines.append("Frequency-band all-layer necessities:")
    lines.append("")
    lines.append("| delta | band | acc drop | margin drop |")
    lines.append("| ---: | --- | ---: | ---: |")
    band_ablate = [row for row in band_rows if row["intervention"] == "ablate_band_all_layers"]
    for row in top_rows(band_ablate, "accuracy_drop_vs_clean", n=12):
        lines.append(
            f"| {int(row['delta'])} | {row['band']} | {float(row['accuracy_drop_vs_clean']):.6f} | "
            f"{float(row['margin_drop_vs_clean']):.6f} |"
        )
    lines.append("")
    lines.append("## Sector Scattering")
    lines.append("")
    lines.append("Largest sector-path ablation effects:")
    lines.append("")
    lines.append("| layer | source | target | acc drop | ablated acc | logprob drop | removed power |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for row in top_rows(path_rows, "accuracy_drop", n=12):
        lines.append(
            f"| {row['layer']} | {row['source']} | {row['target']} | {float(row['accuracy_drop']):.6f} | "
            f"{float(row['clean_accuracy']):.6f} | {float(row['clean_logprob_drop']):.6f} | "
            f"{float(row['removed_component_power']):.6f} |"
        )
    lines.append("")
    lines.append("Sector-mask readout baselines:")
    lines.append("")
    lines.append("| mask | accuracy | drop |")
    lines.append("| --- | ---: | ---: |")
    for row in sorted(mask_rows, key=lambda item: (len(str(item["mask"])), str(item["mask"]))):
        lines.append(f"| {row['mask']} | {float(row['clean_accuracy']):.6f} | {float(row['accuracy_drop']):.6f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Frequency patching tests causal sufficiency of individual cyclic readout coordinates by moving one clean Fourier pair into a corrupted example.")
    lines.append("- Frequency ablation tests necessity by replacing one clean Fourier pair with the corresponding corrupted pair.")
    lines.append("- Sector scattering ablates a single source-to-target contribution at one learned sector mixer, then continues the QNN and measures the final layerwise Dirac-mean readout.")
    lines.append("- Sector masks test whether the final behavior can be carried by one input modality sector alone.")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(freq_baselines, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for path in figure_paths:
        lines.append(f"- `{Path(path).relative_to(out_dir)}`")
    lines.append("")
    lines.append("## Tables")
    lines.append("")
    lines.append("- `frequency_patch.csv`")
    lines.append("- `frequency_band_patch.csv`")
    lines.append("- `sector_path_ablation.csv`")
    lines.append("- `sector_path_frequency_effect.csv`")
    lines.append("- `sector_mixer_unitaries.csv`")
    lines.append("- `sector_mask_ablation.csv`")
    lines.append("- `manifest.json`")
    lines.append("")
    (out_dir / "TRIMODAL_QNN_DIRAC_CAUSAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel causal mech-interp analyses for trimodal layerwise Dirac-mean QNN checkpoints.")
    parser.add_argument("--run-dir", default="trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean")
    parser.add_argument("--checkpoint-step", type=int, default=2000)
    parser.add_argument("--out-dir", default="trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_causal")
    parser.add_argument("--split", default="heldout", choices=["heldout", "all", "train"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--deltas", default="1,2,5,13")
    args = parser.parse_args()

    root = Path.cwd()
    run_dir = Path(args.run_dir)
    run_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir
    out_dir = Path(args.out_dir)
    out_dir = (root / out_dir).resolve() if not out_dir.is_absolute() else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    deltas = [int(item) for item in str(args.deltas).split(",") if item.strip()]

    start = time.time()
    model, cfg, checkpoint_path = load_model(run_dir, int(args.checkpoint_step), device)
    dataset = ModularPairsDataset(cfg["data"], split=args.split)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False)
    modulus = int(cfg["data"]["modulus"])

    freq_rows, freq_baselines = run_frequency_patching(model, loader, modulus=modulus, deltas=deltas)
    band_rows = run_frequency_band_patching(model, loader, modulus=modulus, deltas=deltas)
    path_rows, freq_effect_rows, unitary_rows, mask_rows = run_sector_tomography(model, loader)
    figure_paths = make_figures(out_dir, freq_rows, band_rows, path_rows, freq_effect_rows, unitary_rows, mask_rows)

    write_csv(out_dir / "frequency_patch.csv", freq_rows)
    write_csv(out_dir / "frequency_band_patch.csv", band_rows)
    write_csv(out_dir / "sector_path_ablation.csv", path_rows)
    write_csv(out_dir / "sector_path_frequency_effect.csv", freq_effect_rows)
    write_csv(out_dir / "sector_mixer_unitaries.csv", unitary_rows)
    write_csv(out_dir / "sector_mask_ablation.csv", mask_rows)
    manifest = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "checkpoint_step": int(args.checkpoint_step),
        "checkpoint_path": str(checkpoint_path),
        "out_dir": str(out_dir),
        "split": args.split,
        "records": len(dataset),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "deltas": deltas,
        "elapsed_sec": time.time() - start,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_report(
        out_dir,
        run_name=run_dir.name,
        checkpoint_step=int(args.checkpoint_step),
        checkpoint_path=checkpoint_path,
        freq_rows=freq_rows,
        band_rows=band_rows,
        freq_baselines=freq_baselines,
        path_rows=path_rows,
        mask_rows=mask_rows,
        figure_paths=figure_paths,
        elapsed_sec=manifest["elapsed_sec"],
    )
    print(json.dumps({"event": "causal_analysis_done", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
