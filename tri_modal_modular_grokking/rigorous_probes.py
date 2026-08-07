from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .checkpoint_dynamics import CheckpointInfo, list_numeric_checkpoints, load_metrics, load_model_from_checkpoint
from .data import MultiModalModularConfig, MultiModalModularDataset, split_pairs
from .models import move_batch
from .train import resolve_device


TARGET_NAMES = ("s", "a", "b", "wrap", "mode_a", "mode_b", "output_mode", "cell_id")


@dataclass(frozen=True)
class AnalysisUnit:
    checkpoint: CheckpointInfo
    layers: tuple[int, ...]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def pair_key(pair: tuple[int, int], modulus: int) -> int:
    return int(pair[0]) * modulus + int(pair[1])


def residue(pair: tuple[int, int], modulus: int) -> int:
    return (int(pair[0]) + int(pair[1])) % modulus


def stratified_sample_pairs(
    pairs: list[tuple[int, int]],
    *,
    n: int,
    modulus: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair in pairs:
        groups[residue(pair, modulus)].append(pair)
    for group in groups.values():
        rng.shuffle(group)
    residue_order = np.arange(modulus)
    selected: list[tuple[int, int]] = []
    while len(selected) < n:
        rng.shuffle(residue_order)
        made_progress = False
        for r in residue_order:
            group = groups[int(r)]
            if group:
                selected.append(group.pop())
                made_progress = True
                if len(selected) >= n:
                    break
        if not made_progress:
            break
    if len(selected) != n:
        raise ValueError(f"could only sample {len(selected)} pairs, requested {n}")
    return selected


def train_val_split_keep_train(
    pairs: list[tuple[int, int]],
    *,
    val_fraction: float,
    modulus: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair in pairs:
        groups[residue(pair, modulus)].append(pair)
    train: list[tuple[int, int]] = []
    val: list[tuple[int, int]] = []
    for group in groups.values():
        rng.shuffle(group)
        if len(group) <= 1:
            train.extend(group)
            continue
        n_val = int(round(len(group) * val_fraction))
        n_val = min(max(n_val, 1), len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


def build_pair_sample(
    cfg: MultiModalModularConfig,
    *,
    sample_pairs: int,
    sample_seed: int,
) -> dict[str, Any]:
    train_pairs, heldout_pairs = split_pairs(cfg.modulus, cfg.train_fraction, cfg.seed)
    n_train = int(round(sample_pairs * cfg.train_fraction))
    n_heldout = sample_pairs - n_train
    rng = np.random.default_rng(sample_seed)
    selected_train = stratified_sample_pairs(train_pairs, n=n_train, modulus=cfg.modulus, rng=rng)
    selected_heldout = stratified_sample_pairs(heldout_pairs, n=n_heldout, modulus=cfg.modulus, rng=rng)
    selected = selected_train + selected_heldout
    return {
        "sample_seed": sample_seed,
        "sample_pairs": sample_pairs,
        "selected_train_pairs": selected_train,
        "selected_heldout_pairs": selected_heldout,
        "selected_pairs": selected,
    }


def split_maps_for_seeds(
    cfg: MultiModalModularConfig,
    selected_train: list[tuple[int, int]],
    selected_heldout: list[tuple[int, int]],
    *,
    seeds: list[int],
    val_fraction: float,
) -> dict[int, dict[str, set[int]]]:
    out: dict[int, dict[str, set[int]]] = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        train_pairs, val_pairs = train_val_split_keep_train(selected_train, val_fraction=val_fraction, modulus=cfg.modulus, rng=rng)
        out[seed] = {
            "train": {pair_key(pair, cfg.modulus) for pair in train_pairs},
            "val": {pair_key(pair, cfg.modulus) for pair in val_pairs},
            "test": {pair_key(pair, cfg.modulus) for pair in selected_heldout},
        }
    return out


def selected_dataset_indices(dataset: MultiModalModularDataset, selected_pairs: list[tuple[int, int]]) -> list[int]:
    pair_index = {pair: idx for idx, pair in enumerate(dataset.pairs)}
    per_pair = len(dataset.cells) * dataset.cfg.examples_per_pair_per_cell
    indices: list[int] = []
    for pair in selected_pairs:
        idx = pair_index[pair]
        base = idx * per_pair
        for cell_idx in range(len(dataset.cells)):
            indices.append(base + cell_idx * dataset.cfg.examples_per_pair_per_cell)
    return indices


@torch.no_grad()
def collect_slots(
    model: torch.nn.Module,
    dataset: MultiModalModularDataset,
    indices: list[int],
    device: torch.device,
    *,
    layers: list[int],
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], dict[str, np.ndarray]]:
    need_hidden = any(layer != -1 for layer in layers)
    slots: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    records: dict[str, list[int]] = {key: [] for key in ["pair_key", "a", "b", "s", "wrap", "cell_id", "mode_a", "mode_b", "output_mode"]}
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch, return_hidden=need_hidden)
        for layer in layers:
            if layer == -1:
                slot = outputs["answer_slot"]
            else:
                by_layer = outputs["answer_slots_by_layer"]
                if not isinstance(by_layer, list):
                    raise TypeError("missing answer_slots_by_layer")
                slot = by_layer[layer]
            if not isinstance(slot, torch.Tensor):
                raise TypeError("slot must be tensor")
            slots[layer].append(slot.detach().cpu().float())
        a = batch["a"].detach().cpu().long().numpy()
        b = batch["b"].detach().cpu().long().numpy()
        records["pair_key"].extend((a * dataset.cfg.modulus + b).astype(int).tolist())
        for key in ["a", "b", "s", "wrap", "cell_id"]:
            records[key].extend(batch[key].detach().cpu().long().tolist())
        records["output_mode"].extend(batch["output_mode_id"].detach().cpu().long().tolist())
        records["mode_a"].extend(batch["operand_a_mode_id"].detach().cpu().long().tolist())
        records["mode_b"].extend(batch["operand_b_mode_id"].detach().cpu().long().tolist())
    return {layer: torch.cat(parts, dim=0) for layer, parts in slots.items()}, {key: np.asarray(value, dtype=np.int64) for key, value in records.items()}


def mask_for_keys(pair_keys: np.ndarray, keys: set[int]) -> np.ndarray:
    return np.isin(pair_keys, np.fromiter(keys, dtype=np.int64))


def standardize_from_train(x_train: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    return tuple((x - mean) / std for x in (x_train, *others))


def add_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)


def ridge_weights(x_train: torch.Tensor, y_train: torch.Tensor, *, num_classes: int, lam: float) -> torch.Tensor:
    x = add_bias(x_train)
    y = F.one_hot(y_train.long(), num_classes=num_classes).float()
    xtx = x.T @ x
    reg = torch.eye(xtx.shape[0], dtype=x.dtype)
    reg[-1, -1] = 0.0
    xty = x.T @ y
    return torch.linalg.solve(xtx + float(lam) * reg, xty)


def accuracy_from_weights(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> float:
    logits = add_bias(x) @ weights
    return float((logits.argmax(dim=-1) == y.long()).float().mean().item())


def fit_select_ridge(
    x_train_raw: torch.Tensor,
    y_train: torch.Tensor,
    x_val_raw: torch.Tensor,
    y_val: torch.Tensor,
    x_test_raw: torch.Tensor,
    y_test: torch.Tensor,
    *,
    num_classes: int,
    lambdas: list[float],
) -> dict[str, Any]:
    x_train, x_val, x_test = standardize_from_train(x_train_raw.float(), x_val_raw.float(), x_test_raw.float())
    best: dict[str, Any] | None = None
    for lam in lambdas:
        weights = ridge_weights(x_train, y_train, num_classes=num_classes, lam=lam)
        val_acc = accuracy_from_weights(x_val, y_val, weights)
        train_acc = accuracy_from_weights(x_train, y_train, weights)
        if best is None or val_acc > best["val_accuracy"] or (val_acc == best["val_accuracy"] and lam < best["lambda"]):
            best = {"lambda": lam, "weights": weights, "train_accuracy": train_acc, "val_accuracy": val_acc, "x_train": x_train, "x_val": x_val, "x_test": x_test}
    if best is None:
        raise ValueError("empty lambda grid")
    test_acc = accuracy_from_weights(best["x_test"], y_test, best["weights"])
    return {
        "lambda": float(best["lambda"]),
        "train_accuracy": float(best["train_accuracy"]),
        "val_accuracy": float(best["val_accuracy"]),
        "test_accuracy": float(test_acc),
        "weights": best["weights"],
        "x_train": best["x_train"],
        "x_val": best["x_val"],
        "x_test": best["x_test"],
    }


def permutation_control_accuracy(
    x_train_raw: torch.Tensor,
    y_train: torch.Tensor,
    x_test_raw: torch.Tensor,
    y_test: torch.Tensor,
    *,
    num_classes: int,
    lam: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    perm = torch.tensor(rng.permutation(len(y_train)), dtype=torch.long)
    y_perm = y_train[perm]
    x_train, x_test = standardize_from_train(x_train_raw.float(), x_test_raw.float())
    weights = ridge_weights(x_train, y_perm, num_classes=num_classes, lam=lam)
    return accuracy_from_weights(x_test, y_test, weights)


def run_global_probes(
    slots: torch.Tensor,
    records: dict[str, np.ndarray],
    split_maps: dict[int, dict[str, set[int]]],
    target_classes: dict[str, int],
    *,
    seeds: list[int],
    lambdas: list[float],
    step: int,
    layer: int,
) -> list[dict[str, Any]]:
    x = slots.float()
    pair_keys = records["pair_key"]
    rows = []
    for seed in seeds:
        split = split_maps[seed]
        train_mask = torch.tensor(mask_for_keys(pair_keys, split["train"]), dtype=torch.bool)
        val_mask = torch.tensor(mask_for_keys(pair_keys, split["val"]), dtype=torch.bool)
        test_mask = torch.tensor(mask_for_keys(pair_keys, split["test"]), dtype=torch.bool)
        for target, num_classes in target_classes.items():
            labels = torch.tensor(records[target], dtype=torch.long)
            result = fit_select_ridge(
                x[train_mask],
                labels[train_mask],
                x[val_mask],
                labels[val_mask],
                x[test_mask],
                labels[test_mask],
                num_classes=num_classes,
                lambdas=lambdas,
            )
            control = ""
            if target == "s":
                control = permutation_control_accuracy(
                    x[train_mask],
                    labels[train_mask],
                    x[test_mask],
                    labels[test_mask],
                    num_classes=num_classes,
                    lam=float(result["lambda"]),
                    seed=100000 + seed + step + (layer + 9) * 997,
                )
            rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "seed": seed,
                    "target": target,
                    "lambda": result["lambda"],
                    "train_accuracy": result["train_accuracy"],
                    "val_accuracy": result["val_accuracy"],
                    "test_accuracy": result["test_accuracy"],
                    "permutation_control_test_accuracy": control,
                    "n_train": int(train_mask.sum()),
                    "n_val": int(val_mask.sum()),
                    "n_test": int(test_mask.sum()),
                }
            )
    return rows


def run_cross_cell_transfer(
    slots: torch.Tensor,
    records: dict[str, np.ndarray],
    split_maps: dict[int, dict[str, set[int]]],
    metadata: dict[str, Any],
    modulus: int,
    *,
    seeds: list[int],
    lambdas: list[float],
    step: int,
    layer: int,
) -> list[dict[str, Any]]:
    x = slots.float()
    y = torch.tensor(records["s"], dtype=torch.long)
    pair_keys = records["pair_key"]
    cell_ids = records["cell_id"]
    rows = []
    for seed in seeds:
        split = split_maps[seed]
        train_base = mask_for_keys(pair_keys, split["train"])
        val_base = mask_for_keys(pair_keys, split["val"])
        test_base = mask_for_keys(pair_keys, split["test"])
        for source_cell in range(len(metadata["cells"])):
            source_train_mask_np = train_base & (cell_ids == source_cell)
            source_val_mask_np = val_base & (cell_ids == source_cell)
            source_train_mask = torch.tensor(source_train_mask_np, dtype=torch.bool)
            source_val_mask = torch.tensor(source_val_mask_np, dtype=torch.bool)
            if int(source_train_mask.sum()) == 0 or int(source_val_mask.sum()) == 0:
                continue
            # Use the source cell's validation set to choose the ridge penalty.
            source_result = fit_select_ridge(
                x[source_train_mask],
                y[source_train_mask],
                x[source_val_mask],
                y[source_val_mask],
                x[source_val_mask],
                y[source_val_mask],
                num_classes=modulus,
                lambdas=lambdas,
            )
            x_train_source, _x_val_source = standardize_from_train(x[source_train_mask].float(), x[source_val_mask].float())
            # Refit with the selected lambda so target-cell test states are transformed by source statistics.
            source_mean = x[source_train_mask].float().mean(dim=0, keepdim=True)
            source_std = x[source_train_mask].float().std(dim=0, keepdim=True).clamp_min(1e-6)
            weights = ridge_weights(x_train_source, y[source_train_mask], num_classes=modulus, lam=float(source_result["lambda"]))
            source_meta = metadata["cells"][source_cell]
            for target_cell in range(len(metadata["cells"])):
                target_test_mask_np = test_base & (cell_ids == target_cell)
                target_test_mask = torch.tensor(target_test_mask_np, dtype=torch.bool)
                x_target = (x[target_test_mask].float() - source_mean) / source_std
                acc = accuracy_from_weights(x_target, y[target_test_mask], weights)
                target_meta = metadata["cells"][target_cell]
                rows.append(
                    {
                        "step": step,
                        "layer": layer,
                        "seed": seed,
                        "source_cell": source_cell,
                        "target_cell": target_cell,
                        "source_key": source_meta["key"],
                        "target_key": target_meta["key"],
                        "source_mode_a": source_meta["mode_a"],
                        "source_mode_b": source_meta["mode_b"],
                        "source_output_mode": source_meta["output_mode"],
                        "target_mode_a": target_meta["mode_a"],
                        "target_mode_b": target_meta["mode_b"],
                        "target_output_mode": target_meta["output_mode"],
                        "lambda": source_result["lambda"],
                        "source_train_accuracy": source_result["train_accuracy"],
                        "source_val_accuracy": source_result["val_accuracy"],
                        "target_test_accuracy": acc,
                        "n_source_train": int(source_train_mask.sum()),
                        "n_source_val": int(source_val_mask.sum()),
                        "n_target_test": int(target_test_mask.sum()),
                    }
                )
    return rows


def summarize_unit(
    *,
    step: int,
    layer: int,
    metrics: dict[str, Any],
    global_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    target_names: tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "step": step,
        "layer": layer,
        "heldout_accuracy": metrics.get("heldout_accuracy", ""),
        "heldout_loss": metrics.get("heldout_loss", ""),
        "train_eval_accuracy": metrics.get("train_eval_accuracy", ""),
        "train_eval_loss": metrics.get("train_eval_loss", ""),
    }
    by_target: dict[str, list[float]] = defaultdict(list)
    perm_s = []
    for row in global_rows:
        by_target[str(row["target"])].append(float(row["test_accuracy"]))
        if row["target"] == "s" and row["permutation_control_test_accuracy"] != "":
            perm_s.append(float(row["permutation_control_test_accuracy"]))
    for target in target_names:
        vals = by_target.get(target, [])
        if vals:
            out[f"{target}_probe_test_mean"] = float(np.mean(vals))
            out[f"{target}_probe_test_std"] = float(np.std(vals, ddof=0))
    if perm_s:
        out["s_permutation_control_mean"] = float(np.mean(perm_s))
    transfer_vals = [float(row["target_test_accuracy"]) for row in transfer_rows]
    diag_vals = [float(row["target_test_accuracy"]) for row in transfer_rows if int(row["source_cell"]) == int(row["target_cell"])]
    offdiag_vals = [float(row["target_test_accuracy"]) for row in transfer_rows if int(row["source_cell"]) != int(row["target_cell"])]
    if transfer_vals:
        out["cross_cell_s_transfer_mean"] = float(np.mean(transfer_vals))
        out["cross_cell_s_transfer_min"] = float(np.min(transfer_vals))
        out["cross_cell_s_transfer_std"] = float(np.std(transfer_vals, ddof=0))
    if diag_vals:
        out["same_cell_s_transfer_mean"] = float(np.mean(diag_vals))
    if offdiag_vals:
        out["offdiag_cell_s_transfer_mean"] = float(np.mean(offdiag_vals))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_analysis_units(
    run_dir: Path,
    *,
    transition_steps: set[int],
    max_checkpoints: int | None,
) -> list[AnalysisUnit]:
    checkpoints = list_numeric_checkpoints(run_dir)
    if max_checkpoints is not None:
        checkpoints = checkpoints[:max_checkpoints]
    units = []
    for checkpoint in checkpoints:
        if checkpoint.step in transition_steps:
            units.append(AnalysisUnit(checkpoint=checkpoint, layers=(0, 1, 2, 3, -1)))
        else:
            units.append(AnalysisUnit(checkpoint=checkpoint, layers=(-1,)))
    return units


def target_class_counts(cfg: MultiModalModularConfig, metadata: dict[str, Any]) -> dict[str, int]:
    return {
        "s": cfg.modulus,
        "a": cfg.modulus,
        "b": cfg.modulus,
        "wrap": 2,
        "mode_a": len(metadata.get("input_modes", cfg.input_modes)),
        "mode_b": len(metadata.get("input_modes", cfg.input_modes)),
        "output_mode": len(metadata.get("output_modes", cfg.output_modes)),
        "cell_id": len(metadata["cells"]),
    }


def write_manifest(
    out_dir: Path,
    cfg: MultiModalModularConfig,
    sample: dict[str, Any],
    split_maps: dict[int, dict[str, set[int]]],
    *,
    states: int,
    seeds: list[int],
    lambdas: list[float],
    transition_steps: list[int],
) -> None:
    manifest = {
        "modulus": cfg.modulus,
        "train_fraction": cfg.train_fraction,
        "sample_seed": sample["sample_seed"],
        "sample_pairs": sample["sample_pairs"],
        "states": states,
        "selected_train_pairs": len(sample["selected_train_pairs"]),
        "selected_heldout_pairs": len(sample["selected_heldout_pairs"]),
        "seeds": seeds,
        "lambdas": lambdas,
        "transition_steps": transition_steps,
        "split_counts": {
            str(seed): {role: len(keys) for role, keys in split.items()}
            for seed, split in split_maps.items()
        },
    }
    (out_dir / "probe_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_report(out_dir: Path, summary_rows: list[dict[str, Any]]) -> Path:
    final = [row for row in summary_rows if int(row["step"]) == 20000 and int(row["layer"]) == -1]
    final_row = final[0] if final else summary_rows[-1]
    transfer_hit = next((row for row in summary_rows if int(row["layer"]) == -1 and float(row.get("cross_cell_s_transfer_mean", 0.0)) >= 0.9), None)
    s_hit = next((row for row in summary_rows if int(row["layer"]) == -1 and float(row.get("s_probe_test_mean", 0.0)) >= 0.9), None)
    lines = [
        "# Rigorous Tri-Modal Probe Report",
        "",
        "Probe protocol: sampled ~25k answer states, pair-disjoint train/validation/test splits, five split seeds, ridge-classifier lambda sweep, permutation controls, and full 27x27 cross-cell residue transfer.",
        "",
        "## Final Checkpoint",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| final-slot `s` probe test mean | {float(final_row.get('s_probe_test_mean', 0.0)):.6f} |",
        f"| final-slot `s` probe test std | {float(final_row.get('s_probe_test_std', 0.0)):.6f} |",
        f"| final-slot `s` permutation control | {float(final_row.get('s_permutation_control_mean', 0.0)):.6f} |",
        f"| cross-cell `s` transfer mean | {float(final_row.get('cross_cell_s_transfer_mean', 0.0)):.6f} |",
        f"| cross-cell `s` transfer min | {float(final_row.get('cross_cell_s_transfer_min', 0.0)):.6f} |",
        f"| output-mode probe test mean | {float(final_row.get('output_mode_probe_test_mean', 0.0)):.6f} |",
        "",
        "## Timing Markers",
        "",
        "| event | first final-slot checkpoint | value |",
        "| --- | ---: | ---: |",
        f"| `s` probe test mean >= 0.90 | {int(s_hit['step']) if s_hit else ''} | {float(s_hit.get('s_probe_test_mean', 0.0)) if s_hit else ''} |",
        f"| cross-cell `s` transfer mean >= 0.90 | {int(transfer_hit['step']) if transfer_hit else ''} | {float(transfer_hit.get('cross_cell_s_transfer_mean', 0.0)) if transfer_hit else ''} |",
        "",
        "## Outputs",
        "",
        "```text",
        "probe_manifest.json",
        "probe_summary.csv",
        "global_probe_results.csv",
        "cross_cell_transfer.csv",
        "```",
        "",
    ]
    path = out_dir / "RIGOROUS_PROBE_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_rigorous_probes(
    run_dir: Path,
    out_dir: Path,
    *,
    device_name: str,
    sample_pairs: int,
    sample_seed: int,
    seeds: list[int],
    val_fraction: float,
    lambdas: list[float],
    transition_steps: list[int],
    batch_size: int,
    max_checkpoints: int | None,
) -> Path:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    units = selected_analysis_units(run_dir, transition_steps=set(transition_steps), max_checkpoints=max_checkpoints)
    if not units:
        raise ValueError(f"no checkpoints found in {run_dir}")
    _model, first_config, first_metadata = load_model_from_checkpoint(units[0].checkpoint.path, device)
    cfg = MultiModalModularConfig.from_dict(first_config.get("dataset", {}))
    target_classes = target_class_counts(cfg, first_metadata)
    del _model
    dataset = MultiModalModularDataset(cfg, split="all")
    sample = build_pair_sample(cfg, sample_pairs=sample_pairs, sample_seed=sample_seed)
    split_maps = split_maps_for_seeds(cfg, sample["selected_train_pairs"], sample["selected_heldout_pairs"], seeds=seeds, val_fraction=val_fraction)
    indices = selected_dataset_indices(dataset, sample["selected_pairs"])
    write_manifest(out_dir, cfg, sample, split_maps, states=len(indices), seeds=seeds, lambdas=lambdas, transition_steps=transition_steps)
    metrics_by_step = load_metrics(run_dir)

    global_rows_all: list[dict[str, Any]] = []
    transfer_rows_all: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for unit in units:
        checkpoint = unit.checkpoint
        print(f"checkpoint {checkpoint.step} layers {','.join(str(layer) for layer in unit.layers)}", flush=True)
        model, _config, metadata = load_model_from_checkpoint(checkpoint.path, device)
        slots_by_layer, records = collect_slots(model, dataset, indices, device, layers=list(unit.layers), batch_size=batch_size)
        for layer, slots in slots_by_layer.items():
            print(f"  probes layer {layer}", flush=True)
            global_rows = run_global_probes(slots, records, split_maps, target_classes, seeds=seeds, lambdas=lambdas, step=checkpoint.step, layer=layer)
            transfer_rows = run_cross_cell_transfer(slots, records, split_maps, metadata, cfg.modulus, seeds=seeds, lambdas=lambdas, step=checkpoint.step, layer=layer)
            summary_rows.append(
                summarize_unit(
                    step=checkpoint.step,
                    layer=layer,
                    metrics=metrics_by_step.get(checkpoint.step, {}),
                    global_rows=global_rows,
                    transfer_rows=transfer_rows,
                    target_names=TARGET_NAMES,
                )
            )
            global_rows_all.extend(global_rows)
            transfer_rows_all.extend(transfer_rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(out_dir / "global_probe_results.csv", global_rows_all)
    write_csv(out_dir / "cross_cell_transfer.csv", transfer_rows_all)
    write_csv(out_dir / "probe_summary.csv", summary_rows)
    (out_dir / "probe_summary.json").write_text(json.dumps({"elapsed_sec": time.time() - start, "rows": summary_rows}, indent=2, sort_keys=True), encoding="utf-8")
    report = write_report(out_dir, summary_rows)
    print(report, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rigorous sampled linear probes across tri-modal checkpoints.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-pairs", type=int, default=927)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lambdas", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--transition-steps", default="10000,11000,12000,13000,20000")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-checkpoints", type=int, default=None)
    args = parser.parse_args()
    run_rigorous_probes(
        args.run_dir,
        args.out_dir,
        device_name=args.device,
        sample_pairs=args.sample_pairs,
        sample_seed=args.sample_seed,
        seeds=parse_int_list(args.seeds),
        val_fraction=args.val_fraction,
        lambdas=parse_float_list(args.lambdas),
        transition_steps=parse_int_list(args.transition_steps),
        batch_size=args.batch_size,
        max_checkpoints=args.max_checkpoints,
    )


if __name__ == "__main__":
    main()
