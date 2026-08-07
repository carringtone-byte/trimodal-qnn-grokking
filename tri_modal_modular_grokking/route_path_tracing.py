from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
except Exception:  # pragma: no cover - optional plotting should not block analysis.
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - fallback plotting dependency.
    Image = ImageDraw = ImageFont = None

from .checkpoint_dynamics import load_model_from_checkpoint
from .data import MultiModalModularConfig, MultiModalModularDataset, OUTPUT_MODES, split_pairs
from .leave_combo_mech_interp import trained_combo_set
from .models import move_batch
from .patching import correct_against
from .train import resolve_device


SITES = (
    "bos",
    "a_mode",
    "operand_a_pool",
    "plus",
    "b_mode",
    "operand_b_pool",
    "mod",
    "output_mode",
    "answer_query",
)
COMPONENTS_BY_LAYER = {
    -1: ("embed",),
    0: ("resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"),
    1: ("resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"),
    2: ("resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"),
    3: ("resid_pre", "attn_out", "resid_mid", "mlp_out", "resid_post"),
}
DEFAULT_GOOD_COMBOS = ("number+number", "number+text", "text+number", "text+text")


@dataclass(frozen=True)
class SpanInfo:
    operand_a: tuple[int, int]
    operand_b: tuple[int, int]
    answer: int


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


def cell_id(dataset: MultiModalModularDataset, key: str) -> int:
    for idx, cell in enumerate(dataset.cells):
        if cell.key == key:
            return idx
    raise ValueError(f"unknown cell {key}")


def batch_for_cell(dataset: MultiModalModularDataset, pair_indices: list[int], cell_idx: int) -> dict[str, torch.Tensor]:
    per_pair = len(dataset.cells) * dataset.cfg.examples_per_pair_per_cell
    examples = [dataset[pair_idx * per_pair + cell_idx * dataset.cfg.examples_per_pair_per_cell] for pair_idx in pair_indices]
    return default_collate(examples)


def pair_indices_for_split(cfg: MultiModalModularConfig, *, split: str, n: int, seed: int) -> list[int]:
    train_pairs, heldout_pairs = split_pairs(cfg.modulus, cfg.train_fraction, cfg.seed)
    if split == "train_pair":
        pairs = train_pairs
    elif split == "heldout_pair":
        pairs = heldout_pairs
    else:
        raise ValueError(f"unknown split {split}")
    keys = np.asarray([a * cfg.modulus + b for a, b in pairs], dtype=np.int64)
    if len(keys) > n:
        rng = np.random.default_rng(seed)
        keys = rng.choice(keys, size=n, replace=False)
    return sorted(int(key) for key in keys)


def operand_lengths(model: torch.nn.Module, batch: dict[str, torch.Tensor], prefix: str) -> list[int]:
    lengths: list[int] = []
    for mode_id, text_mask in zip(batch[f"{prefix}_mode_id"].detach().cpu().tolist(), batch[f"{prefix}_text_mask"].detach().cpu()):
        mode = int(mode_id)
        if mode == 0:  # number
            lengths.append(1)
        elif mode == 1:  # text
            lengths.append(int(text_mask.bool().sum().item()))
        elif mode == 2:  # image
            lengths.append(int(model.cfg.image_tokens))
        else:
            raise ValueError(f"unknown mode id {mode}")
    return lengths


def sequence_spans(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> list[SpanInfo]:
    len_a = operand_lengths(model, batch, "operand_a")
    len_b = operand_lengths(model, batch, "operand_b")
    spans: list[SpanInfo] = []
    for a_len, b_len in zip(len_a, len_b):
        a_start = 2
        a_end = a_start + a_len
        b_start = a_end + 2
        b_end = b_start + b_len
        answer = b_end + 2
        spans.append(SpanInfo(operand_a=(a_start, a_end), operand_b=(b_start, b_end), answer=answer))
    return spans


def site_index(span: SpanInfo, site: str) -> int:
    if site == "bos":
        return 0
    if site == "a_mode":
        return 1
    if site == "plus":
        return span.operand_a[1]
    if site == "b_mode":
        return span.operand_a[1] + 1
    if site == "mod":
        return span.operand_b[1]
    if site == "output_mode":
        return span.operand_b[1] + 1
    if site == "answer_query":
        return span.answer
    raise ValueError(f"site {site} is not a singleton site")


def extract_site(x: torch.Tensor, spans: list[SpanInfo], site: str) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for row, span in enumerate(spans):
        if site == "operand_a_pool":
            start, end = span.operand_a
            rows.append(x[row, start:end].mean(dim=0))
        elif site == "operand_b_pool":
            start, end = span.operand_b
            rows.append(x[row, start:end].mean(dim=0))
        else:
            rows.append(x[row, site_index(span, site)])
    return torch.stack(rows, dim=0)


def apply_site_patch(x: torch.Tensor, spans: list[SpanInfo], site: str, values: torch.Tensor) -> None:
    for row, span in enumerate(spans):
        if site == "operand_a_pool":
            start, end = span.operand_a
            x[row, start:end] = values[row].view(1, -1).expand(end - start, -1)
        elif site == "operand_b_pool":
            start, end = span.operand_b
            x[row, start:end] = values[row].view(1, -1).expand(end - start, -1)
        else:
            x[row, site_index(span, site)] = values[row]


@torch.no_grad()
def collect_source_sites(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[tuple[int, str, str], torch.Tensor]:
    backbone = model.backbone
    x, key_padding, _answer_positions = backbone.build_sequence(batch)
    spans = sequence_spans(model, batch)
    out: dict[tuple[int, str, str], torch.Tensor] = {}
    for site in SITES:
        out[(-1, "embed", site)] = extract_site(x, spans, site).detach()
    for layer_idx, block in enumerate(backbone.layers):
        for site in SITES:
            out[(layer_idx, "resid_pre", site)] = extract_site(x, spans, site).detach()
        attn_out = block._sa_block(block.norm1(x), None, key_padding, is_causal=False)
        for site in SITES:
            out[(layer_idx, "attn_out", site)] = extract_site(attn_out, spans, site).detach()
        x_mid = x + attn_out
        for site in SITES:
            out[(layer_idx, "resid_mid", site)] = extract_site(x_mid, spans, site).detach()
        mlp_out = block._ff_block(block.norm2(x_mid))
        for site in SITES:
            out[(layer_idx, "mlp_out", site)] = extract_site(mlp_out, spans, site).detach()
        x = x_mid + mlp_out
        for site in SITES:
            out[(layer_idx, "resid_post", site)] = extract_site(x, spans, site).detach()
    return out


@torch.no_grad()
def forward_with_path_patch(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    patch: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    backbone = model.backbone
    x, key_padding, answer_positions = backbone.build_sequence(batch)
    spans = sequence_spans(model, batch)
    rows = torch.arange(x.shape[0], device=x.device)
    if patch is not None and int(patch["layer"]) == -1 and str(patch["component"]) == "embed":
        apply_site_patch(x, spans, str(patch["site"]), patch["values"])
    for layer_idx, block in enumerate(backbone.layers):
        if patch is not None and int(patch["layer"]) == layer_idx and str(patch["component"]) == "resid_pre":
            apply_site_patch(x, spans, str(patch["site"]), patch["values"])
        attn_out = block._sa_block(block.norm1(x), None, key_padding, is_causal=False)
        if patch is not None and int(patch["layer"]) == layer_idx and str(patch["component"]) == "attn_out":
            apply_site_patch(attn_out, spans, str(patch["site"]), patch["values"])
        x_mid = x + attn_out
        if patch is not None and int(patch["layer"]) == layer_idx and str(patch["component"]) == "resid_mid":
            apply_site_patch(x_mid, spans, str(patch["site"]), patch["values"])
        mlp_out = block._ff_block(block.norm2(x_mid))
        if patch is not None and int(patch["layer"]) == layer_idx and str(patch["component"]) == "mlp_out":
            apply_site_patch(mlp_out, spans, str(patch["site"]), patch["values"])
        x = x_mid + mlp_out
        if patch is not None and int(patch["layer"]) == layer_idx and str(patch["component"]) == "resid_post":
            apply_site_patch(x, spans, str(patch["site"]), patch["values"])
    x = backbone.final_norm(x)
    answer_slot = x[rows, answer_positions]
    return {
        "answer_slot": answer_slot,
        "answer_positions": answer_positions,
        "number_logits": model.number_head(answer_slot),
        "text_logits": model.text_head(answer_slot).view(answer_slot.shape[0], model.cfg.max_answer_len, model.cfg.text_vocab_size),
        "image_class_logits": model.image_class_head(answer_slot),
    }


def ordered_components() -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for layer, components in COMPONENTS_BY_LAYER.items():
        for component in components:
            out.append((layer, component))
    return out


@torch.no_grad()
def run_path_tracing_for_run(
    run_dir: Path,
    out_dir: Path,
    *,
    omitted_combo: str,
    device_name: str,
    patch_pairs: int,
    pair_split: str,
    source_combos: tuple[str, ...],
    target_combos: tuple[str, ...],
) -> Path:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    model, config, _metadata = load_model_from_checkpoint(run_dir / "checkpoint_final.pt", device)
    model.eval()
    cfg = MultiModalModularConfig.from_dict(config.get("dataset", {}))
    dataset = MultiModalModularDataset(cfg, split="all")
    trained_combos = trained_combo_set(cfg)
    pairs = pair_indices_for_split(cfg, split=pair_split, n=patch_pairs, seed=714001 + len(omitted_combo))
    rows: list[dict[str, Any]] = []
    components = ordered_components()
    for output_mode in OUTPUT_MODES:
        for direction, sources, targets in (
            ("good_to_omitted", source_combos, (omitted_combo,)),
            ("omitted_to_good", (omitted_combo,), target_combos),
        ):
            for source_combo in sources:
                source_key = f"{source_combo}->{output_mode}"
                source_idx = cell_id(dataset, source_key)
                source_batch = move_batch(batch_for_cell(dataset, pairs, source_idx), device)
                source_outputs = forward_with_path_patch(model, source_batch)
                source_acc = float(correct_against(source_outputs, source_batch, source_batch).float().mean().detach().cpu())
                source_sites = collect_source_sites(model, source_batch)
                for target_combo in targets:
                    target_key = f"{target_combo}->{output_mode}"
                    target_idx = cell_id(dataset, target_key)
                    target_batch = move_batch(batch_for_cell(dataset, pairs, target_idx), device)
                    target_outputs = forward_with_path_patch(model, target_batch)
                    target_acc = float(correct_against(target_outputs, target_batch, target_batch).float().mean().detach().cpu())
                    for layer, component in components:
                        for site in SITES:
                            values = source_sites[(layer, component, site)]
                            patched_outputs = forward_with_path_patch(
                                model,
                                target_batch,
                                patch={"layer": layer, "component": component, "site": site, "values": values},
                            )
                            patched_acc = float(correct_against(patched_outputs, target_batch, target_batch).float().mean().detach().cpu())
                            rows.append(
                                {
                                    "direction": direction,
                                    "pair_split": pair_split,
                                    "layer": layer,
                                    "component": component,
                                    "site": site,
                                    "source_combo": source_combo,
                                    "target_combo": target_combo,
                                    "source_cell": source_key,
                                    "target_cell": target_key,
                                    "output_mode": output_mode,
                                    "source_trained_input_combo": source_combo in trained_combos,
                                    "target_trained_input_combo": target_combo in trained_combos,
                                    "source_accuracy": source_acc,
                                    "target_baseline_accuracy": target_acc,
                                    "patched_target_accuracy": patched_acc,
                                    "patch_delta_vs_target": patched_acc - target_acc,
                                    "patch_ratio_to_source": patched_acc / source_acc if source_acc > 0 else "",
                                    "n_examples": len(pairs),
                                }
                            )
    write_csv(out_dir / "path_tracing_rows.csv", rows)
    summary = summarize_rows(rows, omitted_combo=omitted_combo, elapsed_sec=time.time() - start)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    save_heatmaps(out_dir / "figures", rows, omitted_combo=omitted_combo)
    write_report(out_dir, rows, summary)
    return out_dir / "PATH_TRACING_REPORT.md"


def summarize_rows(rows: list[dict[str, Any]], *, omitted_combo: str, elapsed_sec: float) -> dict[str, Any]:
    def top(direction: str, k: int = 12) -> list[dict[str, Any]]:
        subset = [row for row in rows if row["direction"] == direction]
        subset = sorted(subset, key=lambda row: float(row["patched_target_accuracy"]), reverse=True)[:k]
        return [
            {
                "direction": row["direction"],
                "layer": row["layer"],
                "component": row["component"],
                "site": row["site"],
                "source_combo": row["source_combo"],
                "target_combo": row["target_combo"],
                "output_mode": row["output_mode"],
                "patched_target_accuracy": row["patched_target_accuracy"],
                "target_baseline_accuracy": row["target_baseline_accuracy"],
                "source_accuracy": row["source_accuracy"],
            }
            for row in subset
        ]

    def strongest_drop(direction: str, k: int = 12) -> list[dict[str, Any]]:
        subset = [row for row in rows if row["direction"] == direction]
        subset = sorted(subset, key=lambda row: float(row["target_baseline_accuracy"]) - float(row["patched_target_accuracy"]), reverse=True)[:k]
        return [
            {
                "direction": row["direction"],
                "layer": row["layer"],
                "component": row["component"],
                "site": row["site"],
                "source_combo": row["source_combo"],
                "target_combo": row["target_combo"],
                "output_mode": row["output_mode"],
                "patched_target_accuracy": row["patched_target_accuracy"],
                "target_baseline_accuracy": row["target_baseline_accuracy"],
                "source_accuracy": row["source_accuracy"],
                "target_drop": float(row["target_baseline_accuracy"]) - float(row["patched_target_accuracy"]),
            }
            for row in subset
        ]

    def best_by_direction(direction: str) -> float:
        vals = [float(row["patched_target_accuracy"]) for row in rows if row["direction"] == direction]
        return float(max(vals)) if vals else float("nan")

    return {
        "elapsed_sec": elapsed_sec,
        "omitted_combo": omitted_combo,
        "n_rows": len(rows),
        "best_good_to_omitted": best_by_direction("good_to_omitted"),
        "best_omitted_to_good": best_by_direction("omitted_to_good"),
        "top_good_to_omitted": top("good_to_omitted"),
        "top_omitted_to_good": top("omitted_to_good"),
        "strongest_omitted_to_good_drops": strongest_drop("omitted_to_good"),
    }


def aggregate_matrix(rows: list[dict[str, Any]], *, direction: str) -> tuple[list[str], list[str], np.ndarray]:
    subset = [row for row in rows if row["direction"] == direction]
    component_order = [f"{layer}:{component}" for layer, component in ordered_components()]
    site_order = list(SITES)
    mat = np.full((len(component_order), len(site_order)), np.nan, dtype=np.float64)
    for i, key in enumerate(component_order):
        layer_text, component = key.split(":", 1)
        vals_by_site: dict[str, list[float]] = {site: [] for site in site_order}
        for row in subset:
            if int(row["layer"]) == int(layer_text) and row["component"] == component:
                vals_by_site[row["site"]].append(float(row["patched_target_accuracy"]))
        for j, site in enumerate(site_order):
            vals = vals_by_site[site]
            if vals:
                mat[i, j] = float(np.mean(vals))
    return component_order, site_order, mat


def _lerp_rgb(left: tuple[int, int, int], right: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(left, right))


def _heatmap_color(value: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (235, 235, 235)
    value = max(0.0, min(1.0, float(value)))
    if value < 0.5:
        return _lerp_rgb((68, 1, 84), (33, 145, 140), value / 0.5)
    return _lerp_rgb((33, 145, 140), (253, 231, 37), (value - 0.5) / 0.5)


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _contrast_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (0, 0, 0) if luminance > 145 else (255, 255, 255)


def _site_label(label: str) -> str:
    replacements = {
        "operand_a_pool": "operand_a\npool",
        "operand_b_pool": "operand_b\npool",
        "output_mode": "output\nmode",
        "answer_query": "answer\nquery",
    }
    return replacements.get(label, label)


def _save_heatmap_pil(path: Path, ylabels: list[str], xlabels: list[str], mat: np.ndarray, title: str) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("No PNG plotting backend is available: both matplotlib and PIL failed to import.")

    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    left_w = 132
    top_h = 42
    cell_w = 94
    cell_h = 28
    bottom_h = 74
    right_w = 20
    width = left_w + cell_w * len(xlabels) + right_w
    height = top_h + cell_h * len(ylabels) + bottom_h
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((10, 12), title, fill=(20, 20, 20), font=title_font)

    grid_left = left_w
    grid_top = top_h
    for i, ylabel in enumerate(ylabels):
        y0 = grid_top + i * cell_h
        draw.text((8, y0 + 8), ylabel, fill=(20, 20, 20), font=font)
        for j, _xlabel in enumerate(xlabels):
            value = float(mat[i, j]) if np.isfinite(mat[i, j]) else float("nan")
            color = _heatmap_color(value)
            x0 = grid_left + j * cell_w
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=color, outline=(255, 255, 255))
            if np.isfinite(value):
                text = f"{value:.2f}"
                tw, th = _text_size(draw, text, font)
                draw.text(
                    (x0 + (cell_w - tw) / 2, y0 + (cell_h - th) / 2 - 1),
                    text,
                    fill=_contrast_color(color),
                    font=font,
                )

    label_y = grid_top + cell_h * len(ylabels) + 7
    for j, xlabel in enumerate(xlabels):
        x0 = grid_left + j * cell_w
        label = _site_label(xlabel)
        lines = label.split("\n")
        for k, line in enumerate(lines):
            tw, _th = _text_size(draw, line, font)
            draw.text((x0 + (cell_w - tw) / 2, label_y + 13 * k), line, fill=(20, 20, 20), font=font)

    draw.rectangle(
        (grid_left, grid_top, grid_left + cell_w * len(xlabels), grid_top + cell_h * len(ylabels)),
        outline=(40, 40, 40),
        width=1,
    )
    image.save(path)


def save_heatmaps(fig_dir: Path, rows: list[dict[str, Any]], *, omitted_combo: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for direction in ("good_to_omitted", "omitted_to_good"):
        ylabels, xlabels, mat = aggregate_matrix(rows, direction=direction)
        path = fig_dir / f"{direction}_heatmap.png"
        title = f"{omitted_combo} path tracing: {direction}"
        if plt is None:
            _save_heatmap_pil(path, ylabels, xlabels, mat, title)
            continue
        fig_h = max(5.0, 0.32 * len(ylabels))
        fig, ax = plt.subplots(figsize=(10.5, fig_h), dpi=150)
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
        ax.set_yticks(range(len(ylabels)), ylabels)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if mat[i, j] < 0.45 else "black")
        fig.colorbar(im, ax=ax, label="patched target accuracy")
        fig.tight_layout()
        try:
            fig.savefig(path)
        except Exception:
            _save_heatmap_pil(path, ylabels, xlabels, mat, title)
        finally:
            plt.close(fig)


def write_report(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        f"# Route Path Tracing: `{summary['omitted_combo']}`",
        "",
        "This report traces causal rescue and causal failure across semantic token sites, transformer layers, and residual/attention/MLP components.",
        "",
        "## Headline",
        "",
        "| measurement | value |",
        "| --- | ---: |",
        f"| rows | {summary['n_rows']} |",
        f"| best mature-source patch into omitted target | {summary['best_good_to_omitted']:.6f} |",
        f"| best omitted-source patch into mature targets | {summary['best_omitted_to_good']:.6f} |",
        f"| elapsed seconds | {summary['elapsed_sec']:.2f} |",
        "",
        "## Top Mature-Source Into Omitted Target Sites",
        "",
        "| layer | component | site | source | target | output | patched acc | target base | source acc |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["top_good_to_omitted"]:
        lines.append(
            f"| {row['layer']} | `{row['component']}` | `{row['site']}` | `{row['source_combo']}` | `{row['target_combo']}` | `{row['output_mode']}` | {float(row['patched_target_accuracy']):.6f} | {float(row['target_baseline_accuracy']):.6f} | {float(row['source_accuracy']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Top Omitted-Source Into Mature Target Sites",
            "",
            "| layer | component | site | source | target | output | patched acc | target base | source acc |",
            "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["top_omitted_to_good"]:
        lines.append(
            f"| {row['layer']} | `{row['component']}` | `{row['site']}` | `{row['source_combo']}` | `{row['target_combo']}` | `{row['output_mode']}` | {float(row['patched_target_accuracy']):.6f} | {float(row['target_baseline_accuracy']):.6f} | {float(row['source_accuracy']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Omitted-Source Damage To Mature Targets",
            "",
            "| layer | component | site | source | target | output | patched acc | target base | drop | source acc |",
            "| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["strongest_omitted_to_good_drops"]:
        lines.append(
            f"| {row['layer']} | `{row['component']}` | `{row['site']}` | `{row['source_combo']}` | `{row['target_combo']}` | `{row['output_mode']}` | {float(row['patched_target_accuracy']):.6f} | {float(row['target_baseline_accuracy']):.6f} | {float(row['target_drop']):.6f} | {float(row['source_accuracy']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "```text",
            "path_tracing_rows.csv",
            "summary.json",
            "figures/good_to_omitted_heatmap.png",
            "figures/omitted_to_good_heatmap.png",
            "```",
            "",
        ]
    )
    (out_dir / "PATH_TRACING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def parse_tuple(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Layerwise causal path tracing for tri-modal directed route leave-outs.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--omitted-combo", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-pairs", type=int, default=256)
    parser.add_argument("--pair-split", default="heldout_pair")
    parser.add_argument("--source-combos", default=",".join(DEFAULT_GOOD_COMBOS))
    parser.add_argument("--target-combos", default=",".join(DEFAULT_GOOD_COMBOS))
    args = parser.parse_args()
    print(
        run_path_tracing_for_run(
            args.run_dir,
            args.out_dir,
            omitted_combo=args.omitted_combo,
            device_name=args.device,
            patch_pairs=args.patch_pairs,
            pair_split=args.pair_split,
            source_combos=parse_tuple(args.source_combos),
            target_combos=parse_tuple(args.target_combos),
        )
    )


if __name__ == "__main__":
    main()
