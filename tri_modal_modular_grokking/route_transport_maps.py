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

from .checkpoint_dynamics import load_model_from_checkpoint
from .data import INPUT_MODES, OUTPUT_MODES, MultiModalModularConfig, MultiModalModularDataset
from .leave_combo_mech_interp import trained_combo_set
from .models import move_batch
from .patching import correct_against
from .rigorous_probes import build_pair_sample, mask_for_keys, parse_float_list, parse_int_list, split_maps_for_seeds
from .route_operand_probes import (
    RouteJob,
    collect_route_states,
    combo_code,
    parse_jobs,
    save_heatmap,
    selected_cell_ids,
    selected_indices_for_cells,
)
from .route_path_tracing import (
    DEFAULT_GOOD_COMBOS,
    apply_site_patch,
    batch_for_cell,
    cell_id,
    collect_source_sites,
    extract_site,
    pair_indices_for_split,
    sequence_spans,
)
from .route_subspace_patching import (
    BASELINE_RUN_NAME,
    PatchSpec,
    batch_accuracy,
    parse_combos,
    patch_specs_for_route,
    stable_seed,
)
from .train import resolve_device


MAP_KINDS = ("pca_ridge", "procrustes", "identity_coord", "random_coord")


@dataclass(frozen=True)
class TransportSpec:
    kind: str
    rank: int

    @property
    def label(self) -> str:
        return f"{self.kind}_r{self.rank}" if self.rank else self.kind


@dataclass
class TransportMap:
    target_mean: torch.Tensor
    target_basis: torch.Tensor
    source_mean: torch.Tensor
    source_basis: torch.Tensor
    coord_map: torch.Tensor

    def apply(self, target_values: torch.Tensor) -> torch.Tensor:
        dtype = target_values.dtype
        device = target_values.device
        target_mean = self.target_mean.to(device=device, dtype=dtype)
        target_basis = self.target_basis.to(device=device, dtype=dtype)
        source_mean = self.source_mean.to(device=device, dtype=dtype)
        source_basis = self.source_basis.to(device=device, dtype=dtype)
        coord_map = self.coord_map.to(device=device, dtype=dtype)
        target_coords = (target_values - target_mean) @ target_basis
        source_coords = target_coords @ coord_map
        return source_mean + source_coords @ source_basis.T


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_map_kinds(text: str) -> tuple[str, ...]:
    if text.strip().lower() in {"all", "*"}:
        return MAP_KINDS
    kinds = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = sorted(set(kinds).difference(MAP_KINDS))
    if unknown:
        raise ValueError(f"unknown map kinds: {unknown}")
    return kinds


def combo_name_from_code(code: int) -> str:
    a_idx = int(code) // len(INPUT_MODES)
    b_idx = int(code) % len(INPUT_MODES)
    return f"{INPUT_MODES[a_idx]}+{INPUT_MODES[b_idx]}"


def paired_indices(
    records: dict[str, np.ndarray],
    *,
    source_combo: str,
    target_combo: str,
    output_mode: str,
    pair_keys: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not pair_keys:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    source_code = combo_code(source_combo)
    target_code = combo_code(target_combo)
    output_id = OUTPUT_MODES.index(output_mode)
    key_mask = mask_for_keys(records["pair_key"], pair_keys)
    source_mask = key_mask & (records["input_combo_code"] == source_code) & (records["output_mode"] == output_id)
    target_mask = key_mask & (records["input_combo_code"] == target_code) & (records["output_mode"] == output_id)
    source_by_pair = {int(records["pair_key"][idx]): int(idx) for idx in np.flatnonzero(source_mask)}
    target_by_pair = {int(records["pair_key"][idx]): int(idx) for idx in np.flatnonzero(target_mask)}
    common = sorted(set(source_by_pair).intersection(target_by_pair))
    if not common:
        raise ValueError(f"no paired rows for {source_combo}->{target_combo} output={output_mode}")
    source_idx = torch.tensor([source_by_pair[key] for key in common], dtype=torch.long)
    target_idx = torch.tensor([target_by_pair[key] for key in common], dtype=torch.long)
    return source_idx, target_idx


def pca_fit(x: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    _u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    effective_rank = min(rank, vh.shape[0], vh.shape[1])
    basis = vh[:effective_rank].T.contiguous()
    denom = float((s.square()).sum().item())
    explained = float((s[:effective_rank].square()).sum().item() / denom) if denom > 0.0 else 0.0
    return mean, basis, explained, effective_rank


def ridge_coord_map(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    lambdas: list[float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if x_val.numel() == 0:
        x_val = x_train
        y_val = y_train
    best: tuple[float, torch.Tensor, float] | None = None
    failures: list[str] = []
    eye = torch.eye(x_train.shape[1], dtype=x_train.dtype)
    xtx = x_train.T @ x_train
    xty = x_train.T @ y_train
    for lam in lambdas:
        try:
            weights = torch.linalg.solve(xtx + float(lam) * eye, xty)
        except RuntimeError as exc:
            failures.append(f"{lam}: {exc}")
            continue
        val_mse = float(((x_val @ weights - y_val).square()).mean().item())
        if best is None or val_mse < best[2] or (val_mse == best[2] and lam < best[0]):
            best = (float(lam), weights, val_mse)
    if best is None:
        raise RuntimeError("all ridge transport fits failed: " + " | ".join(failures[:5]))
    return best[1].contiguous(), {"lambda": best[0], "map_val_coord_mse": best[2]}


def procrustes_coord_map(x_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
    cross = x_train.T @ y_train
    u, _s, vh = torch.linalg.svd(cross, full_matrices=False)
    return (u @ vh).contiguous()


def random_coord_map(rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q, _r = torch.linalg.qr(torch.randn(rank, rank, generator=generator), mode="reduced")
    return q.contiguous()


def evaluate_map(tmap: TransportMap, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    if x.numel() == 0:
        return {"map_mse": float("nan"), "map_rel_mse": float("nan"), "map_cosine": float("nan")}
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y.float(), nan=0.0, posinf=0.0, neginf=0.0)
    y_hat = tmap.apply(x)
    mse = float((y_hat - y).square().mean().item())
    denom = float((y - y.mean(dim=0, keepdim=True)).square().mean().item())
    cosine = float(torch.nn.functional.cosine_similarity(y_hat, y, dim=1).mean().item())
    return {"map_mse": mse, "map_rel_mse": mse / denom if denom > 0.0 else float("nan"), "map_cosine": cosine}


def fit_transport_map(
    states: dict[tuple[int, str, str], torch.Tensor],
    records: dict[str, np.ndarray],
    split: dict[str, set[int]],
    *,
    patch_spec: PatchSpec,
    source_combo: str,
    target_combo: str,
    output_mode: str,
    spec: TransportSpec,
    lambdas: list[float],
    seed: int,
) -> tuple[TransportMap, dict[str, Any]]:
    key = (patch_spec.layer, patch_spec.component, patch_spec.site)
    x_all = torch.nan_to_num(states[key].float(), nan=0.0, posinf=0.0, neginf=0.0)
    src_train, tgt_train = paired_indices(records, source_combo=source_combo, target_combo=target_combo, output_mode=output_mode, pair_keys=split["train"])
    src_val, tgt_val = paired_indices(records, source_combo=source_combo, target_combo=target_combo, output_mode=output_mode, pair_keys=split["val"])
    src_test, tgt_test = paired_indices(records, source_combo=source_combo, target_combo=target_combo, output_mode=output_mode, pair_keys=split["test"])

    source_train = x_all[src_train]
    target_train = x_all[tgt_train]
    source_val = x_all[src_val]
    target_val = x_all[tgt_val]
    source_test = x_all[src_test]
    target_test = x_all[tgt_test]

    target_mean, target_basis, target_explained, effective_rank = pca_fit(target_train, spec.rank)
    source_mean, source_basis, source_explained, _source_rank = pca_fit(source_train, effective_rank)
    zt_train = (target_train - target_mean) @ target_basis
    zs_train = (source_train - source_mean) @ source_basis
    zt_val = (target_val - target_mean) @ target_basis
    zs_val = (source_val - source_mean) @ source_basis

    stats: dict[str, Any] = {
        "n_train_pairs": int(len(src_train)),
        "n_val_pairs": int(len(src_val)),
        "n_test_pairs": int(len(src_test)),
        "effective_rank": int(effective_rank),
        "target_pca_explained": target_explained,
        "source_pca_explained": source_explained,
    }
    if spec.kind == "pca_ridge":
        coord_map, ridge_stats = ridge_coord_map(zt_train, zs_train, zt_val, zs_val, lambdas=lambdas)
        stats.update(ridge_stats)
    elif spec.kind == "procrustes":
        coord_map = procrustes_coord_map(zt_train, zs_train)
        stats["lambda"] = ""
    elif spec.kind == "identity_coord":
        coord_map = torch.eye(effective_rank, dtype=torch.float32)
        stats["lambda"] = ""
    elif spec.kind == "random_coord":
        coord_map = random_coord_map(effective_rank, seed)
        stats["lambda"] = ""
    else:
        raise ValueError(f"unknown transport map kind {spec.kind}")

    tmap = TransportMap(
        target_mean=target_mean,
        target_basis=target_basis,
        source_mean=source_mean,
        source_basis=source_basis,
        coord_map=coord_map,
    )
    for split_name, x, y in (
        ("train", target_train, source_train),
        ("val", target_val, source_val),
        ("test", target_test, source_test),
    ):
        eval_stats = evaluate_map(tmap, x, y)
        for key_name, value in eval_stats.items():
            stats[f"{split_name}_{key_name}"] = value
    return tmap, stats


@torch.no_grad()
def forward_with_transport_patch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    patch: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    backbone = model.backbone
    x, key_padding, answer_positions = backbone.build_sequence(batch)
    spans = sequence_spans(model, batch)
    rows = torch.arange(x.shape[0], device=x.device)

    def maybe_patch(tensor: torch.Tensor, layer: int, component: str) -> None:
        if patch is None:
            return
        if int(patch["layer"]) != layer or str(patch["component"]) != component:
            return
        site = str(patch["site"])
        mode = str(patch["transport_mode"])
        if mode == "source_full":
            patched = patch["source_values"].to(device=tensor.device, dtype=tensor.dtype)
        elif mode == "learned_map":
            target_values = extract_site(tensor, spans, site)
            patched = patch["transport_map"].apply(target_values)
        else:
            raise ValueError(f"unknown transport patch mode {mode}")
        apply_site_patch(tensor, spans, site, patched)

    maybe_patch(x, -1, "embed")
    for layer_idx, block in enumerate(backbone.layers):
        maybe_patch(x, layer_idx, "resid_pre")
        attn_out = block._sa_block(block.norm1(x), None, key_padding, is_causal=False)
        maybe_patch(attn_out, layer_idx, "attn_out")
        x_mid = x + attn_out
        maybe_patch(x_mid, layer_idx, "resid_mid")
        mlp_out = block._ff_block(block.norm2(x_mid))
        maybe_patch(mlp_out, layer_idx, "mlp_out")
        x = x_mid + mlp_out
        maybe_patch(x, layer_idx, "resid_post")
    x = backbone.final_norm(x)
    answer_slot = x[rows, answer_positions]
    return {
        "answer_slot": answer_slot,
        "answer_positions": answer_positions,
        "number_logits": model.number_head(answer_slot),
        "text_logits": model.text_head(answer_slot).view(answer_slot.shape[0], model.cfg.max_answer_len, model.cfg.text_vocab_size),
        "image_class_logits": model.image_class_head(answer_slot),
    }


def transport_specs(kinds: tuple[str, ...], ranks: list[int]) -> list[TransportSpec]:
    return [TransportSpec(kind, rank) for kind in kinds for rank in ranks]


def run_model_route(
    *,
    model_group: str,
    run_dir: Path,
    route_combo: str,
    device: torch.device,
    source_combos: tuple[str, ...],
    good_combos: tuple[str, ...],
    patch_specs: tuple[PatchSpec, ...],
    patch_pairs: int,
    patch_split: str,
    patch_seed: int,
    sample_pairs: int,
    sample_seed: int,
    val_fraction: float,
    split_seed: int,
    ranks: list[int],
    map_kinds: tuple[str, ...],
    lambdas: list[float],
    batch_size: int,
) -> list[dict[str, Any]]:
    model, config, _metadata = load_model_from_checkpoint(run_dir / "checkpoint_final.pt", device)
    model.eval()
    cfg = MultiModalModularConfig.from_dict(config.get("dataset", {}))
    dataset = MultiModalModularDataset(cfg, split="all")
    trained_combos = trained_combo_set(cfg)
    sample = build_pair_sample(cfg, sample_pairs=sample_pairs, sample_seed=sample_seed)
    split_maps = split_maps_for_seeds(
        cfg,
        sample["selected_train_pairs"],
        sample["selected_heldout_pairs"],
        seeds=[split_seed],
        val_fraction=val_fraction,
    )
    combos = tuple(dict.fromkeys((*good_combos, route_combo)))
    cell_ids = selected_cell_ids(dataset, combos)
    indices = selected_indices_for_cells(dataset, sample["selected_pairs"], cell_ids)
    sites = tuple(sorted({spec.site for spec in patch_specs}))
    states, records = collect_route_states(model, dataset, indices, device, sites=sites, batch_size=batch_size)
    split = split_maps[split_seed]
    specs = transport_specs(map_kinds, ranks)
    fitted: dict[tuple[str, str, str, str], tuple[TransportMap, dict[str, Any]]] = {}
    fit_errors: dict[tuple[str, str, str, str], str] = {}
    for patch_spec in patch_specs:
        for source_combo in source_combos:
            for output_mode in OUTPUT_MODES:
                for spec in specs:
                    fit_key = (patch_spec.key, source_combo, output_mode, spec.label)
                    try:
                        tmap, stats = fit_transport_map(
                            states,
                            records,
                            split,
                            patch_spec=patch_spec,
                            source_combo=source_combo,
                            target_combo=route_combo,
                            output_mode=output_mode,
                            spec=spec,
                            lambdas=lambdas,
                            seed=patch_seed + stable_seed(run_dir.name, route_combo, patch_spec.key, source_combo, output_mode, spec.label),
                        )
                    except (RuntimeError, ValueError) as exc:
                        fit_errors[fit_key] = str(exc)
                        continue
                    fitted[fit_key] = (tmap, stats)
    del states
    del records

    pair_indices = pair_indices_for_split(cfg, split=patch_split, n=patch_pairs, seed=patch_seed + len(route_combo))
    rows: list[dict[str, Any]] = []
    for output_mode in OUTPUT_MODES:
        target_key = f"{route_combo}->{output_mode}"
        target_idx = cell_id(dataset, target_key)
        target_batch = move_batch(batch_for_cell(dataset, pair_indices, target_idx), device)
        target_acc = batch_accuracy(model, target_batch)
        for source_combo in source_combos:
            source_key = f"{source_combo}->{output_mode}"
            source_idx = cell_id(dataset, source_key)
            source_batch = move_batch(batch_for_cell(dataset, pair_indices, source_idx), device)
            source_acc = batch_accuracy(model, source_batch)
            source_sites = collect_source_sites(model, source_batch)
            for patch_spec in patch_specs:
                source_values = source_sites[(patch_spec.layer, patch_spec.component, patch_spec.site)]
                outputs = forward_with_transport_patch(
                    model,
                    target_batch,
                    patch={
                        "layer": patch_spec.layer,
                        "component": patch_spec.component,
                        "site": patch_spec.site,
                        "transport_mode": "source_full",
                        "source_values": source_values,
                    },
                )
                patched_acc = float(correct_against(outputs, target_batch, target_batch).float().mean().detach().cpu())
                rows.append(
                    {
                        "model_group": model_group,
                        "run_name": run_dir.name,
                        "route_combo": route_combo,
                        "source_combo": source_combo,
                        "target_combo": route_combo,
                        "output_mode": output_mode,
                        "patch_spec": patch_spec.name,
                        "layer": patch_spec.layer,
                        "component": patch_spec.component,
                        "site": patch_spec.site,
                        "transport_label": "source_full",
                        "transport_kind": "source_full",
                        "rank": 0,
                        "source_accuracy": source_acc,
                        "target_baseline_accuracy": target_acc,
                        "patched_target_accuracy": patched_acc,
                        "patch_delta_vs_target": patched_acc - target_acc,
                        "n_examples": len(pair_indices),
                        "source_trained_input_combo": source_combo in trained_combos,
                        "target_trained_input_combo": route_combo in trained_combos,
                    }
                )
                for spec in specs:
                    fit_key = (patch_spec.key, source_combo, output_mode, spec.label)
                    base = {
                        "model_group": model_group,
                        "run_name": run_dir.name,
                        "route_combo": route_combo,
                        "source_combo": source_combo,
                        "target_combo": route_combo,
                        "output_mode": output_mode,
                        "patch_spec": patch_spec.name,
                        "layer": patch_spec.layer,
                        "component": patch_spec.component,
                        "site": patch_spec.site,
                        "transport_label": spec.label,
                        "transport_kind": spec.kind,
                        "rank": spec.rank,
                        "source_accuracy": source_acc,
                        "target_baseline_accuracy": target_acc,
                        "n_examples": len(pair_indices),
                        "source_trained_input_combo": source_combo in trained_combos,
                        "target_trained_input_combo": route_combo in trained_combos,
                    }
                    if fit_key not in fitted:
                        rows.append({**base, "transport_error": fit_errors.get(fit_key, "transport map was not fitted"), "patched_target_accuracy": "", "patch_delta_vs_target": ""})
                        continue
                    tmap, stats = fitted[fit_key]
                    outputs = forward_with_transport_patch(
                        model,
                        target_batch,
                        patch={
                            "layer": patch_spec.layer,
                            "component": patch_spec.component,
                            "site": patch_spec.site,
                            "transport_mode": "learned_map",
                            "transport_map": tmap,
                        },
                    )
                    patched_acc = float(correct_against(outputs, target_batch, target_batch).float().mean().detach().cpu())
                    rows.append(
                        {
                            **base,
                            "patched_target_accuracy": patched_acc,
                            "patch_delta_vs_target": patched_acc - target_acc,
                            **stats,
                        }
                    )
        del target_batch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def mean_by(rows: list[dict[str, Any]], keys: tuple[str, ...], value: str) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get(value) == "":
            continue
        grouped[tuple(row[key] for key in keys)].append(float(row[value]))
    return {key: float(np.mean(vals)) for key, vals in grouped.items()}


def summarize_rows(rows: list[dict[str, Any]], *, elapsed_sec: float, patch_pairs: int, sample_pairs: int) -> dict[str, Any]:
    numeric = [row for row in rows if row.get("patched_target_accuracy") != ""]
    route_summaries: list[dict[str, Any]] = []
    for route_combo in sorted({str(row["route_combo"]) for row in numeric}):
        for model_group in sorted({str(row["model_group"]) for row in numeric if row["route_combo"] == route_combo}):
            subset = [row for row in numeric if row["route_combo"] == route_combo and row["model_group"] == model_group]
            learned = [row for row in subset if row["transport_kind"] not in {"source_full", "random_coord"}]
            random_rows = [row for row in subset if row["transport_kind"] == "random_coord"]
            full_rows = [row for row in subset if row["transport_kind"] == "source_full"]
            best = max(subset, key=lambda row: float(row["patched_target_accuracy"]))
            best_learned = max(learned, key=lambda row: float(row["patched_target_accuracy"])) if learned else best
            best_full = max(full_rows, key=lambda row: float(row["patched_target_accuracy"])) if full_rows else best
            route_summaries.append(
                {
                    "model_group": model_group,
                    "route_combo": route_combo,
                    "mean_target_baseline_accuracy": float(np.mean([float(row["target_baseline_accuracy"]) for row in subset])),
                    "best_full_accuracy": float(best_full["patched_target_accuracy"]),
                    "best_full_site": f"L{best_full['layer']}:{best_full['component']}:{best_full['site']}",
                    "best_learned_accuracy": float(best_learned["patched_target_accuracy"]),
                    "best_learned_label": best_learned["transport_label"],
                    "best_learned_site": f"L{best_learned['layer']}:{best_learned['component']}:{best_learned['site']}",
                    "mean_learned_accuracy": float(np.mean([float(row["patched_target_accuracy"]) for row in learned])) if learned else float("nan"),
                    "mean_random_accuracy": float(np.mean([float(row["patched_target_accuracy"]) for row in random_rows])) if random_rows else float("nan"),
                }
            )
    return {
        "elapsed_sec": elapsed_sec,
        "patch_pairs": patch_pairs,
        "sample_pairs": sample_pairs,
        "n_rows": len(rows),
        "n_numeric_rows": len(numeric),
        "n_failed_rows": len(rows) - len(numeric),
        "route_summaries": route_summaries,
    }


def graph_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric = [row for row in rows if row.get("patched_target_accuracy") != ""]
    out: list[dict[str, Any]] = []
    for model_group in sorted({row["model_group"] for row in numeric}):
        for target_combo in sorted({row["target_combo"] for row in numeric if row["model_group"] == model_group}):
            for source_combo in sorted({row["source_combo"] for row in numeric if row["model_group"] == model_group and row["target_combo"] == target_combo}):
                subset = [row for row in numeric if row["model_group"] == model_group and row["target_combo"] == target_combo and row["source_combo"] == source_combo]
                learned = [row for row in subset if row["transport_kind"] not in {"source_full", "random_coord"}]
                if not learned:
                    continue
                best = max(learned, key=lambda row: float(row["patched_target_accuracy"]))
                out.append(
                    {
                        "model_group": model_group,
                        "source_combo": source_combo,
                        "target_combo": target_combo,
                        "best_learned_accuracy": float(best["patched_target_accuracy"]),
                        "best_learned_delta": float(best["patch_delta_vs_target"]),
                        "best_label": best["transport_label"],
                        "best_patch_spec": best["patch_spec"],
                        "best_output_mode": best["output_mode"],
                        "best_site": f"L{best['layer']}:{best['component']}:{best['site']}",
                    }
                )
    return out


def save_figures(out_dir: Path, rows: list[dict[str, Any]], graph: list[dict[str, Any]]) -> None:
    numeric = [row for row in rows if row.get("patched_target_accuracy") != ""]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for model_group in sorted({row["model_group"] for row in numeric}):
        for route_combo in sorted({row["route_combo"] for row in numeric if row["model_group"] == model_group}):
            route_rows = [row for row in numeric if row["model_group"] == model_group and row["route_combo"] == route_combo]
            for patch_name in sorted({row["patch_spec"] for row in route_rows}):
                subset = [row for row in route_rows if row["patch_spec"] == patch_name]
                labels = sorted({row["transport_label"] for row in subset})
                modes = list(OUTPUT_MODES)
                means = mean_by(subset, ("transport_label", "output_mode"), "patched_target_accuracy")
                matrix = np.full((len(labels), len(modes)), np.nan, dtype=np.float64)
                for i, label in enumerate(labels):
                    for j, mode in enumerate(modes):
                        value = means.get((label, mode))
                        if value is not None:
                            matrix[i, j] = value
                safe_route = route_combo.replace("+", "_")
                save_heatmap(fig_dir / f"{model_group}_{safe_route}_{patch_name}_transport_accuracy.png", matrix, modes, labels, f"{model_group} {route_combo} {patch_name}")
    for model_group in sorted({row["model_group"] for row in graph}):
        subset = [row for row in graph if row["model_group"] == model_group]
        sources = list(DEFAULT_GOOD_COMBOS)
        targets = sorted({row["target_combo"] for row in subset})
        matrix = np.full((len(sources), len(targets)), np.nan, dtype=np.float64)
        lookup = {(row["source_combo"], row["target_combo"]): float(row["best_learned_accuracy"]) for row in subset}
        for i, source in enumerate(sources):
            for j, target in enumerate(targets):
                value = lookup.get((source, target))
                if value is not None:
                    matrix[i, j] = value
        save_heatmap(fig_dir / f"{model_group}_learned_transport_route_graph.png", matrix, targets, sources, f"{model_group} learned transport graph")


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Route Transport Maps",
        "",
        "This experiment replaces PCA-only subspace patching with explicit low-dimensional transport maps from an omitted route's local activation manifold into a mature no-image source route manifold. The same tests are run against the fully grokked `phase4_full_crossmodal` baseline as trained-route controls.",
        "",
        "## Headline",
        "",
        "| measurement | value |",
        "| --- | ---: |",
        f"| rows | {summary['n_rows']} |",
        f"| numeric rows | {summary['n_numeric_rows']} |",
        f"| failed rows | {summary['n_failed_rows']} |",
        f"| patch pairs | {summary['patch_pairs']} |",
        f"| map sample pairs | {summary['sample_pairs']} |",
        f"| elapsed seconds | {summary['elapsed_sec']:.2f} |",
        "",
        "## Route Summary",
        "",
        "| model group | route | target baseline | best full | best learned transport | learned site | mean learned | random control |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary["route_summaries"]:
        lines.append(
            f"| `{row['model_group']}` | `{row['route_combo']}` | "
            f"{row['mean_target_baseline_accuracy']:.6f} | "
            f"{row['best_full_accuracy']:.6f} | "
            f"{row['best_learned_accuracy']:.6f} `{row['best_learned_label']}` | "
            f"`{row['best_learned_site']}` | "
            f"{row['mean_learned_accuracy']:.6f} | "
            f"{row['mean_random_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Map Families",
            "",
            "| family | intervention |",
            "| --- | --- |",
            "| `pca_ridge` | PCA coordinates for target and source, ridge map from target coordinates to source coordinates |",
            "| `procrustes` | PCA coordinates with an orthogonal Procrustes map, preserving geometry as much as possible |",
            "| `identity_coord` | target PCA coordinates copied into source PCA axes; alignment control |",
            "| `random_coord` | random orthogonal coordinate map; rank-matched negative control |",
            "| `source_full` | complete mature source vector copied at the site; positive control |",
            "",
            "## Artifacts",
            "",
            "```text",
            "route_transport_rows.csv",
            "route_transport_graph_rows.csv",
            "summary.json",
            "figures/*transport_accuracy.png",
            "figures/*learned_transport_route_graph.png",
            "```",
            "",
        ]
    )
    (out_dir / "ROUTE_TRANSPORT_MAPS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    *,
    runs_root: Path,
    out_dir: Path,
    jobs: list[RouteJob],
    baseline_run_name: str,
    include_baseline: bool,
    device_name: str,
    source_combos: tuple[str, ...],
    good_combos: tuple[str, ...],
    patch_spec_text: str,
    patch_pairs: int,
    patch_split: str,
    patch_seed: int,
    sample_pairs: int,
    sample_seed: int,
    val_fraction: float,
    split_seed: int,
    ranks: list[int],
    map_kinds: tuple[str, ...],
    lambdas: list[float],
    batch_size: int,
) -> Path:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pipeline.log"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = resolve_device(device_name)
    all_rows: list[dict[str, Any]] = []
    log(
        "started route transport maps "
        f"jobs={','.join(job.omitted_combo for job in jobs)} baseline={include_baseline} "
        f"patch_pairs={patch_pairs} sample_pairs={sample_pairs} ranks={','.join(str(rank) for rank in ranks)}"
    )
    for job in jobs:
        patch_specs = patch_specs_for_route(job.omitted_combo, patch_spec_text)
        route_start = time.time()
        log(f"leaveout {job.run_name} route={job.omitted_combo} specs={','.join(spec.name for spec in patch_specs)}")
        all_rows.extend(
            run_model_route(
                model_group="leaveout",
                run_dir=runs_root / job.run_name,
                route_combo=job.omitted_combo,
                device=device,
                source_combos=source_combos,
                good_combos=good_combos,
                patch_specs=patch_specs,
                patch_pairs=patch_pairs,
                patch_split=patch_split,
                patch_seed=patch_seed,
                sample_pairs=sample_pairs,
                sample_seed=sample_seed,
                val_fraction=val_fraction,
                split_seed=split_seed,
                ranks=ranks,
                map_kinds=map_kinds,
                lambdas=lambdas,
                batch_size=batch_size,
            )
        )
        log(f"finished leaveout {job.run_name} elapsed_sec={time.time() - route_start:.2f}")
        if include_baseline:
            route_start = time.time()
            log(f"baseline {baseline_run_name} pseudo_route={job.omitted_combo}")
            all_rows.extend(
                run_model_route(
                    model_group="baseline_full",
                    run_dir=runs_root / baseline_run_name,
                    route_combo=job.omitted_combo,
                    device=device,
                    source_combos=source_combos,
                    good_combos=good_combos,
                    patch_specs=patch_specs,
                    patch_pairs=patch_pairs,
                    patch_split=patch_split,
                    patch_seed=patch_seed,
                    sample_pairs=sample_pairs,
                    sample_seed=sample_seed,
                    val_fraction=val_fraction,
                    split_seed=split_seed,
                    ranks=ranks,
                    map_kinds=map_kinds,
                    lambdas=lambdas,
                    batch_size=batch_size,
                )
            )
            log(f"finished baseline {baseline_run_name} pseudo_route={job.omitted_combo} elapsed_sec={time.time() - route_start:.2f}")
    graph = graph_rows(all_rows)
    write_csv(out_dir / "route_transport_rows.csv", all_rows)
    write_csv(out_dir / "route_transport_graph_rows.csv", graph)
    summary = summarize_rows(all_rows, elapsed_sec=time.time() - start, patch_pairs=patch_pairs, sample_pairs=sample_pairs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    save_figures(out_dir, all_rows, graph)
    write_report(out_dir, summary)
    log(f"complete rows={len(all_rows)} graph_rows={len(graph)} elapsed_sec={time.time() - start:.2f}")
    return out_dir / "ROUTE_TRANSPORT_MAPS_REPORT.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned low-dimensional route transport maps with causal patching and baseline controls.")
    parser.add_argument("--runs-root", type=Path, default=Path("tri_modal_modular_grokking/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("tri_modal_modular_grokking/analysis/phase6_route_transport_maps"))
    parser.add_argument("--jobs", default="all")
    parser.add_argument("--baseline-run-name", default=BASELINE_RUN_NAME)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-combos", default="good")
    parser.add_argument("--good-combos", default="good")
    parser.add_argument("--patch-specs", default="hotspots")
    parser.add_argument("--patch-pairs", type=int, default=256)
    parser.add_argument("--patch-split", default="heldout_pair")
    parser.add_argument("--patch-seed", type=int, default=742109)
    parser.add_argument("--sample-pairs", type=int, default=927)
    parser.add_argument("--sample-seed", type=int, default=714203)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--ranks", default="8,16,32,64")
    parser.add_argument("--map-kinds", default="all")
    parser.add_argument("--lambdas", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    print(
        run_pipeline(
            runs_root=args.runs_root,
            out_dir=args.out_dir,
            jobs=parse_jobs(args.jobs),
            baseline_run_name=args.baseline_run_name,
            include_baseline=not args.no_baseline,
            device_name=args.device,
            source_combos=parse_combos(args.source_combos),
            good_combos=parse_combos(args.good_combos),
            patch_spec_text=args.patch_specs,
            patch_pairs=args.patch_pairs,
            patch_split=args.patch_split,
            patch_seed=args.patch_seed,
            sample_pairs=args.sample_pairs,
            sample_seed=args.sample_seed,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            ranks=parse_int_list(args.ranks),
            map_kinds=parse_map_kinds(args.map_kinds),
            lambdas=parse_float_list(args.lambdas),
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
