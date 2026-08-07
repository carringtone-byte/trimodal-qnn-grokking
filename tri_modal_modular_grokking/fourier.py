from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def grid_fft_energy(grid: np.ndarray) -> dict[str, Any]:
    modulus = grid.shape[0]
    centered = grid - grid.mean(axis=(0, 1), keepdims=True)
    spectrum = np.fft.fftn(centered, axes=(0, 1))
    power = np.square(np.abs(spectrum)).sum(axis=2)
    total = float(power.sum())
    if total <= 1e-12:
        return {
            "addition_diag": 0.0,
            "difference_diag": 0.0,
            "a_only": 0.0,
            "b_only": 0.0,
            "other": 0.0,
            "top_freq": 0,
            "top_freq_r2": 0.0,
            "mean_freq_r2": 0.0,
        }
    mask = np.zeros_like(power, dtype=bool)
    idx = np.arange(1, modulus)
    add_positions = (idx, idx)
    diff_positions = (idx, (-idx) % modulus)
    a_positions = (idx, np.zeros_like(idx))
    b_positions = (np.zeros_like(idx), idx)
    add_power = power[add_positions]
    fractions = {
        "addition_diag": float(add_power.sum() / total),
        "difference_diag": float(power[diff_positions].sum() / total),
        "a_only": float(power[a_positions].sum() / total),
        "b_only": float(power[b_positions].sum() / total),
    }
    mask[add_positions] = True
    mask[diff_positions] = True
    mask[a_positions] = True
    mask[b_positions] = True
    fractions["other"] = float(power[~mask].sum() / total)
    top_idx = int(idx[int(np.argmax(add_power))]) if len(add_power) else 0
    fractions["top_freq"] = top_idx
    fractions["top_freq_r2"] = float(add_power.max() / max(float(add_power.sum()), 1e-12)) if len(add_power) else 0.0
    fractions["mean_freq_r2"] = float(add_power.mean() / max(float(add_power.sum()), 1e-12)) if len(add_power) else 0.0
    return fractions


def build_grid(slots: torch.Tensor, records: dict[str, list[int]], *, cell_id: int, modulus: int) -> np.ndarray | None:
    grid = np.zeros((modulus, modulus, slots.shape[1]), dtype=np.float64)
    seen = np.zeros((modulus, modulus), dtype=bool)
    cell_records = np.asarray(records["cell_id"]) == cell_id
    indices = np.where(cell_records)[0]
    for idx in indices:
        a = int(records["a"][idx])
        b = int(records["b"][idx])
        grid[a, b] = slots[idx].double().numpy()
        seen[a, b] = True
    if not seen.all():
        return None
    return grid


def run_fourier(answer_slots_path: Path, out_dir: Path) -> Path:
    data = torch.load(answer_slots_path, map_location="cpu")
    slots = data["slots"].float()
    records = data["records"]
    metadata = data["metadata"]
    modulus = int(metadata["config"]["modulus"])
    rows = []
    for cell in metadata["cells"]:
        grid = build_grid(slots, records, cell_id=int(cell["id"]), modulus=modulus)
        if grid is None:
            continue
        metrics = grid_fft_energy(grid)
        rows.append(
            {
                "layer": data.get("layer", -1),
                "mode_a": cell["mode_a"],
                "mode_b": cell["mode_b"],
                "output_mode": cell["output_mode"],
                **metrics,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fourier_energy_by_layer.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["layer", "mode_a", "mode_b", "output_mode", "addition_diag", "difference_diag", "a_only", "b_only", "other", "top_freq", "top_freq_r2", "mean_freq_r2"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "num_cells": len(rows),
        "mean_addition_diagonal_energy": float(np.mean([row["addition_diag"] for row in rows])) if rows else 0.0,
        "rows": rows,
    }
    (out_dir / "fourier_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tri-modal answer-slot Fourier diagnostics.")
    parser.add_argument("--answer-slots", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(run_fourier(args.answer_slots, args.out_dir))


if __name__ == "__main__":
    main()
