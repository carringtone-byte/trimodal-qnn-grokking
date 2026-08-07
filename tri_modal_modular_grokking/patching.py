from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .analyze import load_checkpoint
from .data import DECODER_TARGET_KINDS, MultiModalModularConfig, MultiModalModularDataset, OUTPUT_MODES
from .models import move_batch
from .train import resolve_device


def cell_id(dataset: MultiModalModularDataset, key: str) -> int:
    for idx, cell in enumerate(dataset.cells):
        if cell.key == key:
            return idx
    raise ValueError(f"unknown cell {key}; available: {[cell.key for cell in dataset.cells]}")


def batch_for_cell(dataset: MultiModalModularDataset, pairs: list[int], cell_idx: int) -> dict[str, torch.Tensor]:
    per_pair = len(dataset.cells) * dataset.cfg.examples_per_pair_per_cell
    examples = [dataset[pair_idx * per_pair + cell_idx * dataset.cfg.examples_per_pair_per_cell] for pair_idx in pairs]
    return default_collate(examples)


def correct_against(
    outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    reference: dict[str, torch.Tensor],
    target_batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    output_mode = target_batch["output_mode_id"]
    target_kind = target_batch["decoder_target_kind_id"]
    correct = torch.zeros(reference["s"].shape[0], dtype=torch.bool, device=reference["s"].device)
    number_logits = outputs["number_logits"]
    text_logits = outputs["text_logits"]
    image_logits = outputs["image_class_logits"]
    assert isinstance(number_logits, torch.Tensor)
    assert isinstance(text_logits, torch.Tensor)
    assert isinstance(image_logits, torch.Tensor)
    number_mask = output_mode == OUTPUT_MODES.index("number")
    text_mask = output_mode == OUTPUT_MODES.index("text")
    image_mask = target_kind == DECODER_TARGET_KINDS.index("image_class_proxy")
    if number_mask.any():
        correct[number_mask] = number_logits[number_mask].argmax(dim=-1) == reference["target_number"][number_mask]
    if text_mask.any():
        preds = text_logits[text_mask].argmax(dim=-1)
        labels = reference["target_text_ids"][text_mask]
        mask = reference["target_text_mask"][text_mask].bool()
        correct[text_mask] = ((preds == labels) | ~mask).all(dim=1)
    if image_mask.any():
        correct[image_mask] = image_logits[image_mask].argmax(dim=-1) == reference["target_number"][image_mask]
    return correct


def pair_indices_with_different_answers(dataset: MultiModalModularDataset, n: int) -> tuple[list[int], list[int]]:
    clean = []
    corrupt = []
    modulus = dataset.cfg.modulus
    for idx, (a, b) in enumerate(dataset.pairs):
        s = (a + b) % modulus
        for jdx in range(idx + 1, len(dataset.pairs)):
            aa, bb = dataset.pairs[jdx]
            if (aa + bb) % modulus != s:
                clean.append(idx)
                corrupt.append(jdx)
                break
        if len(clean) >= n:
            break
    if len(clean) < n:
        raise ValueError("not enough clean/corrupt pairs with different answers")
    return clean, corrupt


@torch.no_grad()
def run_patching(
    run_dir: Path,
    out_dir: Path,
    *,
    device_name: str,
    source_cell: str,
    target_cell: str,
    pairs: int,
    layers: list[int],
) -> Path:
    device = resolve_device(device_name)
    model, config, _metadata = load_checkpoint(run_dir, device)
    cfg = MultiModalModularConfig.from_dict(config.get("dataset", {}))
    dataset = MultiModalModularDataset(cfg, split="heldout")
    source_id = cell_id(dataset, source_cell)
    target_id = cell_id(dataset, target_cell)
    clean_idx, corrupt_idx = pair_indices_with_different_answers(dataset, pairs)
    clean = move_batch(batch_for_cell(dataset, clean_idx, source_id), device)
    corrupt = move_batch(batch_for_cell(dataset, corrupt_idx, target_id), device)
    clean_outputs = model(clean, return_hidden=True)
    baseline_outputs = model(corrupt)
    baseline_clean = correct_against(baseline_outputs, clean, corrupt).float().mean()
    baseline_corrupt = correct_against(baseline_outputs, corrupt, corrupt).float().mean()
    rows = []
    for layer in layers:
        if layer == -1:
            values = clean_outputs["answer_slot"]
        else:
            by_layer = clean_outputs["answer_slots_by_layer"]
            if not isinstance(by_layer, list):
                raise TypeError("answer_slots_by_layer missing")
            values = by_layer[layer]
        if not isinstance(values, torch.Tensor):
            raise TypeError("patch values must be a tensor")
        patched_outputs = model(corrupt, patch={"layer": layer, "values": values})
        clean_acc = correct_against(patched_outputs, clean, corrupt).float().mean()
        corrupt_acc = correct_against(patched_outputs, corrupt, corrupt).float().mean()
        source = dataset.cells[source_id]
        target = dataset.cells[target_id]
        output_relation = "fixed" if source.output_mode == target.output_mode else "cross"
        rows.append(
            {
                "layer": layer,
                "source_mode_a": source.mode_a,
                "source_mode_b": source.mode_b,
                "source_output_mode": source.output_mode,
                "target_mode_a": target.mode_a,
                "target_mode_b": target.mode_b,
                "target_output_mode": target.output_mode,
                "patch_type": "answer_slot",
                "patch_basis": "full",
                "output_relation": output_relation,
                "clean_answer_acc": float(clean_acc.detach().cpu()),
                "corrupt_answer_acc": float(corrupt_acc.detach().cpu()),
                "other_answer_acc": "",
                "baseline_clean_answer_acc": float(baseline_clean.detach().cpu()),
                "baseline_corrupt_answer_acc": float(baseline_corrupt.detach().cpu()),
                "n_examples": pairs,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "patching_matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "layer",
            "source_mode_a",
            "source_mode_b",
            "source_output_mode",
            "target_mode_a",
            "target_mode_b",
            "target_output_mode",
            "patch_type",
            "patch_basis",
            "output_relation",
            "clean_answer_acc",
            "corrupt_answer_acc",
            "other_answer_acc",
            "baseline_clean_answer_acc",
            "baseline_corrupt_answer_acc",
            "n_examples",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_layers(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tri-modal answer-slot activation patching.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-cell", default="text+image->number")
    parser.add_argument("--target-cell", default="number+number->number")
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--layers", default="-1")
    args = parser.parse_args()
    print(
        run_patching(
            args.run_dir,
            args.out_dir,
            device_name=args.device,
            source_cell=args.source_cell,
            target_cell=args.target_cell,
            pairs=args.pairs,
            layers=parse_layers(args.layers),
        )
    )


if __name__ == "__main__":
    main()
