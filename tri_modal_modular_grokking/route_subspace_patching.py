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
from .data import OUTPUT_MODES, MultiModalModularConfig, MultiModalModularDataset
from .leave_combo_mech_interp import trained_combo_set
from .models import move_batch
from .patching import correct_against
from .rigorous_probes import build_pair_sample, mask_for_keys, parse_float_list, parse_int_list, split_maps_for_seeds
from .route_operand_probes import (
    RouteJob,
    collect_route_states,
    combo_code,
    fit_select_ridge_robust,
    parse_jobs,
    save_heatmap,
    selected_cell_ids,
    selected_indices_for_cells,
)
from .route_path_tracing import (
    DEFAULT_GOOD_COMBOS,
    SITES,
    apply_site_patch,
    batch_for_cell,
    cell_id,
    collect_source_sites,
    extract_site,
    pair_indices_for_split,
    sequence_spans,
)
from .train import resolve_device


TARGETS = ("a", "b", "s")
BASELINE_RUN_NAME = "phase4_full_crossmodal"


@dataclass(frozen=True)
class PatchSpec:
    name: str
    layer: int
    component: str
    site: str

    @property
    def key(self) -> str:
        return f"L{self.layer}:{self.component}:{self.site}"


@dataclass(frozen=True)
class BasisSpec:
    family: str
    role: str
    target: str
    rank: int
    patch_mode: str

    @property
    def label(self) -> str:
        parts = [self.family]
        if self.role:
            parts.append(self.role)
        if self.target:
            parts.append(self.target)
        if self.rank:
            parts.append(f"r{self.rank}")
        if self.patch_mode == "orthogonal":
            parts.append("orth")
        return "_".join(parts)


HOTSPOTS: dict[str, tuple[PatchSpec, ...]] = {
    "image+text": (
        PatchSpec("early_b_image_text", 0, "mlp_out", "operand_b_pool"),
        PatchSpec("late_answer_query", 3, "mlp_out", "answer_query"),
    ),
    "text+image": (
        PatchSpec("early_b_text_image", 0, "mlp_out", "operand_b_pool"),
        PatchSpec("late_answer_query", 3, "mlp_out", "answer_query"),
    ),
    "number+image": (
        PatchSpec("early_plus_attn", 0, "attn_out", "plus"),
        PatchSpec("early_plus_resid_mid", 0, "resid_mid", "plus"),
        PatchSpec("late_answer_query", 3, "mlp_out", "answer_query"),
    ),
    "image+number": (
        PatchSpec("early_a_image_number", 0, "mlp_out", "operand_a_pool"),
        PatchSpec("early_b_image_number_mid", 0, "resid_mid", "operand_b_pool"),
        PatchSpec("early_b_image_number_mlp", 0, "mlp_out", "operand_b_pool"),
        PatchSpec("late_answer_query", 3, "mlp_out", "answer_query"),
    ),
}


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


def parse_combos(text: str) -> tuple[str, ...]:
    if text.strip().lower() in {"default", "good"}:
        return DEFAULT_GOOD_COMBOS
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_targets(text: str) -> tuple[str, ...]:
    targets = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = sorted(set(targets).difference(TARGETS))
    if unknown:
        raise ValueError(f"unknown targets: {unknown}")
    return targets


def parse_patch_spec(text: str) -> PatchSpec:
    try:
        layer_text, component, site = text.split(":", 2)
    except ValueError as exc:
        raise ValueError(f"patch spec must be layer:component:site, got {text!r}") from exc
    site = site.strip()
    if site not in SITES:
        raise ValueError(f"unknown site {site!r}")
    return PatchSpec(f"custom_{layer_text}_{component}_{site}", int(layer_text), component.strip(), site)


def patch_specs_for_route(route_combo: str, spec_text: str) -> tuple[PatchSpec, ...]:
    if spec_text.strip().lower() in {"hotspots", "default"}:
        if route_combo not in HOTSPOTS:
            raise ValueError(f"no hotspots registered for {route_combo}")
        return HOTSPOTS[route_combo]
    return tuple(parse_patch_spec(part.strip()) for part in spec_text.split(",") if part.strip())


def targets_for_site(site: str, requested: tuple[str, ...]) -> tuple[str, ...]:
    if site == "operand_a_pool":
        preferred = ("a", "s")
    elif site == "operand_b_pool":
        preferred = ("b", "s")
    elif site == "plus":
        preferred = ("a", "b", "s")
    elif site == "answer_query":
        preferred = ("s", "a", "b")
    else:
        preferred = requested
    return tuple(target for target in preferred if target in requested)


def basis_specs_for_site(
    site: str,
    *,
    ranks: list[int],
    targets: tuple[str, ...],
    include_pca: bool,
    include_random: bool,
    include_complements: bool,
) -> list[BasisSpec]:
    specs = [BasisSpec("full", "source", "", 0, "full")]
    site_targets = targets_for_site(site, targets)
    for rank in ranks:
        if include_random:
            specs.append(BasisSpec("random", "control", "", rank, "projected"))
        if include_pca:
            specs.append(BasisSpec("pca", "mature_no_image", "", rank, "projected"))
            specs.append(BasisSpec("pca", "local_target", "", rank, "projected"))
        for target in site_targets:
            specs.append(BasisSpec("probe", "mature_no_image", target, rank, "projected"))
            specs.append(BasisSpec("probe", "local_target", target, rank, "projected"))
    if include_complements and ranks:
        rank = ranks[0]
        for target in site_targets:
            specs.append(BasisSpec("probe", "mature_no_image", target, rank, "orthogonal"))
            specs.append(BasisSpec("probe", "local_target", target, rank, "orthogonal"))
    deduped: list[BasisSpec] = []
    seen: set[tuple[str, str, str, int, str]] = set()
    for spec in specs:
        key = (spec.family, spec.role, spec.target, spec.rank, spec.patch_mode)
        if key not in seen:
            deduped.append(spec)
            seen.add(key)
    return deduped


def role_mask(records: dict[str, np.ndarray], *, route_combo: str, good_combos: tuple[str, ...], role: str) -> np.ndarray:
    if role == "local_target":
        return records["input_combo_code"] == combo_code(route_combo)
    if role == "mature_no_image":
        good_codes = np.asarray([combo_code(combo) for combo in good_combos], dtype=np.int64)
        return np.isin(records["input_combo_code"], good_codes)
    raise ValueError(f"unknown basis role {role}")


def orthonormal_basis(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    matrix = torch.nan_to_num(matrix.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.ndim != 2:
        raise ValueError(f"expected matrix, got shape {tuple(matrix.shape)}")
    if matrix.numel() == 0:
        raise ValueError("empty basis matrix")
    q, _r = torch.linalg.qr(matrix, mode="reduced")
    return q[:, : min(rank, q.shape[1])].contiguous()


def pca_basis(x: torch.Tensor, train_mask: np.ndarray, rank: int) -> tuple[torch.Tensor, dict[str, Any]]:
    mask = torch.tensor(train_mask, dtype=torch.bool)
    if int(mask.sum()) <= 1:
        raise ValueError("not enough rows for PCA basis")
    x_train = torch.nan_to_num(x[mask].float(), nan=0.0, posinf=0.0, neginf=0.0)
    x_centered = x_train - x_train.mean(dim=0, keepdim=True)
    _u, s, vh = torch.linalg.svd(x_centered, full_matrices=False)
    use_rank = min(rank, vh.shape[0])
    basis = vh[:use_rank].T.contiguous()
    denom = float((s.square()).sum().item())
    explained = float((s[:use_rank].square()).sum().item() / denom) if denom > 0.0 else 0.0
    return basis, {"n_fit": int(mask.sum()), "pca_explained_variance": explained}


def random_basis(d_model: int, rank: int, seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(d_model, rank, generator=generator)
    return orthonormal_basis(matrix, rank), {"n_fit": 0}


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 2_147_483_647
    return value


def probe_basis(
    x: torch.Tensor,
    records: dict[str, np.ndarray],
    split: dict[str, set[int]],
    role_train_mask: np.ndarray,
    *,
    target: str,
    num_classes: int,
    lambdas: list[float],
    rank: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    pair_keys = records["pair_key"]
    train_np = mask_for_keys(pair_keys, split["train"]) & role_train_mask
    val_np = mask_for_keys(pair_keys, split["val"]) & role_train_mask
    test_np = mask_for_keys(pair_keys, split["test"]) & role_train_mask
    counts = {"n_train": int(train_np.sum()), "n_val": int(val_np.sum()), "n_test": int(test_np.sum())}
    if min(counts.values()) <= 0:
        raise ValueError(f"empty probe split: {counts}")
    labels = torch.tensor(records[target], dtype=torch.long)
    train_mask = torch.tensor(train_np, dtype=torch.bool)
    val_mask = torch.tensor(val_np, dtype=torch.bool)
    test_mask = torch.tensor(test_np, dtype=torch.bool)
    x_float = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x_train = x_float[train_mask]
    std = x_train.std(dim=0).clamp_min(1e-6)
    result = fit_select_ridge_robust(
        x_train,
        labels[train_mask],
        x_float[val_mask],
        labels[val_mask],
        x_float[test_mask],
        labels[test_mask],
        num_classes=num_classes,
        lambdas=lambdas,
    )
    weights = result["weights"][:-1].float()
    raw_weights = weights / std.view(-1, 1)
    u, _s, _vh = torch.linalg.svd(torch.nan_to_num(raw_weights), full_matrices=False)
    basis = u[:, : min(rank, u.shape[1])].contiguous()
    return basis, {
        **counts,
        "lambda": float(result["lambda"]),
        "basis_probe_train_accuracy": float(result["train_accuracy"]),
        "basis_probe_val_accuracy": float(result["val_accuracy"]),
        "basis_probe_test_accuracy": float(result["test_accuracy"]),
    }


def make_basis_bank(
    states: dict[tuple[int, str, str], torch.Tensor],
    records: dict[str, np.ndarray],
    split: dict[str, set[int]],
    *,
    route_combo: str,
    good_combos: tuple[str, ...],
    patch_specs: tuple[PatchSpec, ...],
    ranks: list[int],
    targets: tuple[str, ...],
    lambdas: list[float],
    modulus: int,
    random_seed: int,
    include_pca: bool,
    include_random: bool,
    include_complements: bool,
) -> tuple[dict[tuple[str, BasisSpec], torch.Tensor], dict[tuple[str, BasisSpec], dict[str, Any]]]:
    bank: dict[tuple[str, BasisSpec], torch.Tensor] = {}
    meta: dict[tuple[str, BasisSpec], dict[str, Any]] = {}
    role_masks = {
        "mature_no_image": role_mask(records, route_combo=route_combo, good_combos=good_combos, role="mature_no_image"),
        "local_target": role_mask(records, route_combo=route_combo, good_combos=good_combos, role="local_target"),
    }
    fit_key_mask = mask_for_keys(records["pair_key"], split["train"])
    for patch_spec in patch_specs:
        x = torch.nan_to_num(states[(patch_spec.layer, patch_spec.component, patch_spec.site)].float(), nan=0.0, posinf=0.0, neginf=0.0)
        d_model = x.shape[1]
        for basis_spec in basis_specs_for_site(
            patch_spec.site,
            ranks=ranks,
            targets=targets,
            include_pca=include_pca,
            include_random=include_random,
            include_complements=include_complements,
        ):
            if basis_spec.family == "full":
                continue
            try:
                if basis_spec.family == "random":
                    basis, stats = random_basis(d_model, basis_spec.rank, random_seed + stable_seed(patch_spec.key, basis_spec.rank))
                elif basis_spec.family == "pca":
                    basis, stats = pca_basis(x, fit_key_mask & role_masks[basis_spec.role], basis_spec.rank)
                elif basis_spec.family == "probe":
                    basis, stats = probe_basis(
                        x,
                        records,
                        split,
                        role_masks[basis_spec.role],
                        target=basis_spec.target,
                        num_classes=modulus,
                        lambdas=lambdas,
                        rank=basis_spec.rank,
                    )
                else:
                    raise ValueError(f"unknown basis family {basis_spec.family}")
            except (RuntimeError, ValueError) as exc:
                meta[(patch_spec.key, basis_spec)] = {"basis_error": str(exc)}
                continue
            bank[(patch_spec.key, basis_spec)] = basis
            meta[(patch_spec.key, basis_spec)] = stats
    return bank, meta


def project_delta(source: torch.Tensor, target: torch.Tensor, basis: torch.Tensor | None, patch_mode: str) -> torch.Tensor:
    if patch_mode == "full":
        return source
    if basis is None:
        raise ValueError("basis is required for projected or orthogonal patching")
    basis = basis.to(device=target.device, dtype=target.dtype)
    delta = source - target
    projected = (delta @ basis) @ basis.T
    if patch_mode == "orthogonal":
        projected = delta - projected
    return target + projected


@torch.no_grad()
def forward_with_subspace_patch(
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
        source = patch["source_values"].to(device=tensor.device, dtype=tensor.dtype)
        target = extract_site(tensor, spans, site)
        patched = project_delta(source, target, patch.get("basis"), str(patch["patch_mode"]))
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


@torch.no_grad()
def batch_accuracy(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> float:
    outputs = model(batch)
    return float(correct_against(outputs, batch, batch).float().mean().detach().cpu())


def mean_by(rows: list[dict[str, Any]], keys: tuple[str, ...], value: str) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[value]))
    return {key: float(np.mean(vals)) for key, vals in grouped.items()}


def run_model_route(
    *,
    model_group: str,
    run_dir: Path,
    route_combo: str,
    out_dir: Path,
    device: torch.device,
    source_combos: tuple[str, ...],
    good_combos: tuple[str, ...],
    patch_specs: tuple[PatchSpec, ...],
    patch_pairs: int,
    patch_split: str,
    patch_seed: int,
    basis_sample_pairs: int,
    basis_sample_seed: int,
    val_fraction: float,
    probe_seed: int,
    ranks: list[int],
    targets: tuple[str, ...],
    lambdas: list[float],
    batch_size: int,
    include_pca: bool,
    include_random: bool,
    include_complements: bool,
) -> list[dict[str, Any]]:
    model, config, _metadata = load_model_from_checkpoint(run_dir / "checkpoint_final.pt", device)
    model.eval()
    cfg = MultiModalModularConfig.from_dict(config.get("dataset", {}))
    dataset = MultiModalModularDataset(cfg, split="all")
    trained_combos = trained_combo_set(cfg)

    sample = build_pair_sample(cfg, sample_pairs=basis_sample_pairs, sample_seed=basis_sample_seed)
    split_maps = split_maps_for_seeds(
        cfg,
        sample["selected_train_pairs"],
        sample["selected_heldout_pairs"],
        seeds=[probe_seed],
        val_fraction=val_fraction,
    )
    combos = tuple(dict.fromkeys((*good_combos, route_combo)))
    cell_ids = selected_cell_ids(dataset, combos)
    indices = selected_indices_for_cells(dataset, sample["selected_pairs"], cell_ids)
    sites = tuple(sorted({spec.site for spec in patch_specs}, key=SITES.index))
    states, records = collect_route_states(model, dataset, indices, device, sites=sites, batch_size=batch_size)
    basis_bank, basis_meta = make_basis_bank(
        states,
        records,
        split_maps[probe_seed],
        route_combo=route_combo,
        good_combos=good_combos,
        patch_specs=patch_specs,
        ranks=ranks,
        targets=targets,
        lambdas=lambdas,
        modulus=cfg.modulus,
        random_seed=patch_seed + probe_seed,
        include_pca=include_pca,
        include_random=include_random,
        include_complements=include_complements,
    )
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
                for basis_spec in basis_specs_for_site(
                    patch_spec.site,
                    ranks=ranks,
                    targets=targets,
                    include_pca=include_pca,
                    include_random=include_random,
                    include_complements=include_complements,
                ):
                    if basis_spec.family == "full":
                        basis = None
                        stats: dict[str, Any] = {}
                    else:
                        bank_key = (patch_spec.key, basis_spec)
                        if bank_key not in basis_bank:
                            err = basis_meta.get(bank_key, {}).get("basis_error", "basis was not fitted")
                            rows.append(
                                {
                                    "model_group": model_group,
                                    "run_name": run_dir.name,
                                    "route_combo": route_combo,
                                    "output_mode": output_mode,
                                    "source_combo": source_combo,
                                    "target_combo": route_combo,
                                    "patch_spec": patch_spec.name,
                                    "layer": patch_spec.layer,
                                    "component": patch_spec.component,
                                    "site": patch_spec.site,
                                    "basis_label": basis_spec.label,
                                    "basis_family": basis_spec.family,
                                    "basis_role": basis_spec.role,
                                    "basis_target": basis_spec.target,
                                    "rank": basis_spec.rank,
                                    "patch_mode": basis_spec.patch_mode,
                                    "basis_error": err,
                                    "source_accuracy": source_acc,
                                    "target_baseline_accuracy": target_acc,
                                    "patched_target_accuracy": "",
                                    "patch_delta_vs_target": "",
                                    "n_examples": len(pair_indices),
                                    "source_trained_input_combo": source_combo in trained_combos,
                                    "target_trained_input_combo": route_combo in trained_combos,
                                }
                            )
                            continue
                        basis = basis_bank[bank_key]
                        stats = basis_meta.get(bank_key, {})
                    outputs = forward_with_subspace_patch(
                        model,
                        target_batch,
                        patch={
                            "layer": patch_spec.layer,
                            "component": patch_spec.component,
                            "site": patch_spec.site,
                            "source_values": source_values,
                            "basis": basis,
                            "patch_mode": basis_spec.patch_mode,
                        },
                    )
                    patched_acc = float(correct_against(outputs, target_batch, target_batch).float().mean().detach().cpu())
                    rows.append(
                        {
                            "model_group": model_group,
                            "run_name": run_dir.name,
                            "route_combo": route_combo,
                            "output_mode": output_mode,
                            "source_combo": source_combo,
                            "target_combo": route_combo,
                            "patch_spec": patch_spec.name,
                            "layer": patch_spec.layer,
                            "component": patch_spec.component,
                            "site": patch_spec.site,
                            "basis_label": basis_spec.label,
                            "basis_family": basis_spec.family,
                            "basis_role": basis_spec.role,
                            "basis_target": basis_spec.target,
                            "rank": basis_spec.rank,
                            "patch_mode": basis_spec.patch_mode,
                            "source_accuracy": source_acc,
                            "target_baseline_accuracy": target_acc,
                            "patched_target_accuracy": patched_acc,
                            "patch_delta_vs_target": patched_acc - target_acc,
                            "n_examples": len(pair_indices),
                            "source_trained_input_combo": source_combo in trained_combos,
                            "target_trained_input_combo": route_combo in trained_combos,
                            **stats,
                        }
                    )
        del target_batch
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def summarize_rows(rows: list[dict[str, Any]], *, elapsed_sec: float, patch_pairs: int, basis_sample_pairs: int) -> dict[str, Any]:
    numeric_rows = [row for row in rows if row.get("patched_target_accuracy") != ""]
    summary_routes: list[dict[str, Any]] = []
    for route_combo in sorted({str(row["route_combo"]) for row in numeric_rows}):
        for model_group in sorted({str(row["model_group"]) for row in numeric_rows if row["route_combo"] == route_combo}):
            subset = [row for row in numeric_rows if row["route_combo"] == route_combo and row["model_group"] == model_group]
            if not subset:
                continue
            best = max(subset, key=lambda row: float(row["patched_target_accuracy"]))
            best_projected = max(
                [row for row in subset if row["patch_mode"] == "projected"] or subset,
                key=lambda row: float(row["patched_target_accuracy"]),
            )
            full_rows = [row for row in subset if row["basis_family"] == "full"]
            best_full = max(full_rows, key=lambda row: float(row["patched_target_accuracy"])) if full_rows else best
            projected = [row for row in subset if row["patch_mode"] == "projected" and row["basis_family"] != "random"]
            rand = [row for row in subset if row["basis_family"] == "random" and row["patch_mode"] == "projected"]
            summary_routes.append(
                {
                    "model_group": model_group,
                    "route_combo": route_combo,
                    "mean_target_baseline_accuracy": float(np.mean([float(row["target_baseline_accuracy"]) for row in subset])),
                    "best_full_patch_accuracy": float(best_full["patched_target_accuracy"]),
                    "best_full_patch_label": best_full["basis_label"],
                    "best_projected_accuracy": float(best_projected["patched_target_accuracy"]),
                    "best_projected_label": best_projected["basis_label"],
                    "best_projected_site": f"L{best_projected['layer']}:{best_projected['component']}:{best_projected['site']}",
                    "mean_projected_accuracy": float(np.mean([float(row["patched_target_accuracy"]) for row in projected])) if projected else float("nan"),
                    "mean_random_accuracy": float(np.mean([float(row["patched_target_accuracy"]) for row in rand])) if rand else float("nan"),
                    "best_accuracy": float(best["patched_target_accuracy"]),
                }
            )
    return {
        "elapsed_sec": elapsed_sec,
        "patch_pairs": patch_pairs,
        "basis_sample_pairs": basis_sample_pairs,
        "n_rows": len(rows),
        "n_numeric_rows": len(numeric_rows),
        "n_failed_basis_rows": len(rows) - len(numeric_rows),
        "route_summaries": summary_routes,
    }


def save_figures(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    numeric_rows = [row for row in rows if row.get("patched_target_accuracy") != ""]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for model_group in sorted({str(row["model_group"]) for row in numeric_rows}):
        for route_combo in sorted({str(row["route_combo"]) for row in numeric_rows if row["model_group"] == model_group}):
            route_rows = [row for row in numeric_rows if row["model_group"] == model_group and row["route_combo"] == route_combo]
            for patch_name in sorted({str(row["patch_spec"]) for row in route_rows}):
                subset = [row for row in route_rows if row["patch_spec"] == patch_name]
                basis_labels = sorted({str(row["basis_label"]) for row in subset})
                out_modes = list(OUTPUT_MODES)
                means = mean_by(subset, ("basis_label", "output_mode"), "patched_target_accuracy")
                matrix = np.full((len(basis_labels), len(out_modes)), np.nan, dtype=np.float64)
                for i, label in enumerate(basis_labels):
                    for j, out_mode in enumerate(out_modes):
                        value = means.get((label, out_mode))
                        if value is not None:
                            matrix[i, j] = value
                safe_route = route_combo.replace("+", "_")
                path = fig_dir / f"{model_group}_{safe_route}_{patch_name}_accuracy.png"
                save_heatmap(path, matrix, out_modes, basis_labels, f"{model_group} {route_combo} {patch_name}")


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Route Subspace Causal Patching",
        "",
        "This experiment patches only selected activation subspaces at route-localized carriers. For each phase-6 leaveout route, the same route is also evaluated in the earlier fully grokked `phase4_full_crossmodal` baseline as a trained-route control.",
        "",
        "## Headline",
        "",
        "| measurement | value |",
        "| --- | ---: |",
        f"| rows | {summary['n_rows']} |",
        f"| numeric rows | {summary['n_numeric_rows']} |",
        f"| failed basis rows | {summary['n_failed_basis_rows']} |",
        f"| patch pairs | {summary['patch_pairs']} |",
        f"| basis sample pairs | {summary['basis_sample_pairs']} |",
        f"| elapsed seconds | {summary['elapsed_sec']:.2f} |",
        "",
        "## Route Summary",
        "",
        "| model group | route | target baseline | best full patch | best projected patch | projected site | mean projected | random control |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for route in summary["route_summaries"]:
        lines.append(
            f"| `{route['model_group']}` | `{route['route_combo']}` | "
            f"{route['mean_target_baseline_accuracy']:.6f} | "
            f"{route['best_full_patch_accuracy']:.6f} | "
            f"{route['best_projected_accuracy']:.6f} `{route['best_projected_label']}` | "
            f"`{route['best_projected_site']}` | "
            f"{route['mean_projected_accuracy']:.6f} | "
            f"{route['mean_random_accuracy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- `full` is the positive-control intervention: the complete source vector at the specified site replaces the target vector.",
            "- `probe_*` uses the raw activation directions implied by linear classifiers for `a`, `b`, or `s`; `pca_*` uses high-variance local geometry; `random_*` is the dimension-matched control.",
            "- `orth` rows patch the complement of a probe subspace. A strong projected patch with a weak complement is evidence that the chosen decoded variable is close to the causal carrier; a strong complement means the probe is predictive but not causally sufficient.",
            "- The fully grokked baseline should have high target baseline accuracy because the route is trained. It is included to reveal whether the same projected subspaces preserve or carry the computation in a solved route, rather than only rescuing failed leaveout routes.",
            "",
            "## Artifacts",
            "",
            "```text",
            "route_subspace_patching_rows.csv",
            "summary.json",
            "figures/<model_group>_<route>_<patch_spec>_accuracy.png",
            "```",
            "",
        ]
    )
    (out_dir / "ROUTE_SUBSPACE_PATCHING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


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
    basis_sample_pairs: int,
    basis_sample_seed: int,
    val_fraction: float,
    probe_seed: int,
    ranks: list[int],
    targets: tuple[str, ...],
    lambdas: list[float],
    batch_size: int,
    include_pca: bool,
    include_random: bool,
    include_complements: bool,
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
    log(
        "started route subspace patching "
        f"jobs={','.join(job.omitted_combo for job in jobs)} baseline={include_baseline} "
        f"patch_pairs={patch_pairs} basis_sample_pairs={basis_sample_pairs} ranks={','.join(str(r) for r in ranks)}"
    )
    rows: list[dict[str, Any]] = []
    for job in jobs:
        patch_specs = patch_specs_for_route(job.omitted_combo, patch_spec_text)
        route_start = time.time()
        log(f"leaveout {job.run_name} route={job.omitted_combo} specs={','.join(spec.name for spec in patch_specs)}")
        rows.extend(
            run_model_route(
                model_group="leaveout",
                run_dir=runs_root / job.run_name,
                route_combo=job.omitted_combo,
                out_dir=out_dir,
                device=device,
                source_combos=source_combos,
                good_combos=good_combos,
                patch_specs=patch_specs,
                patch_pairs=patch_pairs,
                patch_split=patch_split,
                patch_seed=patch_seed,
                basis_sample_pairs=basis_sample_pairs,
                basis_sample_seed=basis_sample_seed,
                val_fraction=val_fraction,
                probe_seed=probe_seed,
                ranks=ranks,
                targets=targets,
                lambdas=lambdas,
                batch_size=batch_size,
                include_pca=include_pca,
                include_random=include_random,
                include_complements=include_complements,
            )
        )
        log(f"finished leaveout {job.run_name} elapsed_sec={time.time() - route_start:.2f}")
        if include_baseline:
            route_start = time.time()
            log(f"baseline {baseline_run_name} pseudo_route={job.omitted_combo}")
            rows.extend(
                run_model_route(
                    model_group="baseline_full",
                    run_dir=runs_root / baseline_run_name,
                    route_combo=job.omitted_combo,
                    out_dir=out_dir,
                    device=device,
                    source_combos=source_combos,
                    good_combos=good_combos,
                    patch_specs=patch_specs,
                    patch_pairs=patch_pairs,
                    patch_split=patch_split,
                    patch_seed=patch_seed,
                    basis_sample_pairs=basis_sample_pairs,
                    basis_sample_seed=basis_sample_seed,
                    val_fraction=val_fraction,
                    probe_seed=probe_seed,
                    ranks=ranks,
                    targets=targets,
                    lambdas=lambdas,
                    batch_size=batch_size,
                    include_pca=include_pca,
                    include_random=include_random,
                    include_complements=include_complements,
                )
            )
            log(f"finished baseline {baseline_run_name} pseudo_route={job.omitted_combo} elapsed_sec={time.time() - route_start:.2f}")

    write_csv(out_dir / "route_subspace_patching_rows.csv", rows)
    summary = summarize_rows(rows, elapsed_sec=time.time() - start, patch_pairs=patch_pairs, basis_sample_pairs=basis_sample_pairs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    save_figures(out_dir, rows)
    write_report(out_dir, summary)
    log(f"complete rows={len(rows)} elapsed_sec={time.time() - start:.2f}")
    return out_dir / "ROUTE_SUBSPACE_PATCHING_REPORT.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Subspace causal patching at directed route carriers, with fully grokked baseline comparison.")
    parser.add_argument("--runs-root", type=Path, default=Path("tri_modal_modular_grokking/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("tri_modal_modular_grokking/analysis/phase6_route_subspace_patching"))
    parser.add_argument("--jobs", default="all")
    parser.add_argument("--baseline-run-name", default=BASELINE_RUN_NAME)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-combos", default="good")
    parser.add_argument("--good-combos", default="good")
    parser.add_argument("--patch-specs", default="hotspots")
    parser.add_argument("--patch-pairs", type=int, default=256)
    parser.add_argument("--patch-split", default="heldout_pair")
    parser.add_argument("--patch-seed", type=int, default=731009)
    parser.add_argument("--basis-sample-pairs", type=int, default=927)
    parser.add_argument("--basis-sample-seed", type=int, default=714203)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--ranks", default="16,32")
    parser.add_argument("--targets", default="a,b,s")
    parser.add_argument("--lambdas", default="0.0001,0.001,0.01,0.1,1,10,100")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--no-complements", action="store_true")
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
            basis_sample_pairs=args.basis_sample_pairs,
            basis_sample_seed=args.basis_sample_seed,
            val_fraction=args.val_fraction,
            probe_seed=args.probe_seed,
            ranks=parse_int_list(args.ranks),
            targets=parse_targets(args.targets),
            lambdas=parse_float_list(args.lambdas),
            batch_size=args.batch_size,
            include_pca=not args.no_pca,
            include_random=not args.no_random,
            include_complements=not args.no_complements,
        )
    )


if __name__ == "__main__":
    main()
