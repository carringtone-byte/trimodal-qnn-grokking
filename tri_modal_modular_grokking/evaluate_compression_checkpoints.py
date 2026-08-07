from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import MultiModalModularConfig, make_datasets
from .models import build_model, parameter_count
from .train import evaluate, resolve_device


def aggregate_cells(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, float]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in keys)
        group = groups.setdefault(key, {"correct": 0.0, "loss": 0.0, "n": 0.0})
        n = float(row["n_examples"])
        group["correct"] += float(row["accuracy"]) * n
        group["loss"] += float(row["loss"]) * n
        group["n"] += n
    result = []
    for key, group in sorted(groups.items()):
        n = max(1.0, group["n"])
        result.append(
            {
                **dict(zip(keys, key)),
                "accuracy": group["correct"] / n,
                "loss": group["loss"] / n,
                "n_examples": int(group["n"]),
            }
        )
    return result


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    metadata = checkpoint["metadata"]
    dataset_cfg = MultiModalModularConfig.from_dict(config.get("dataset", {}))
    train_ds, heldout_ds = make_datasets(dataset_cfg)
    model = build_model(config.get("model", {}), metadata).to(device)
    model.load_state_dict(checkpoint["model"])
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "step": int(checkpoint["step"]),
        "parameter_count": parameter_count(model),
        "batch_size": batch_size,
        "device": str(device),
    }
    for split, dataset in (("train", train_ds), ("heldout", heldout_ds)):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        summary, cells = evaluate(
            model,
            dataset,
            loader,
            device,
            step=int(checkpoint["step"]),
            split=split,
            max_batches=None,
        )
        result[split] = {
            "summary": summary,
            "by_output_mode": aggregate_cells(cells, ("output_mode",)),
            "by_input_route": aggregate_cells(cells, ("mode_a", "mode_b")),
            "cells": cells,
        }
    return result


def write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for result in results:
        for split in ("train", "heldout"):
            summary = result[split]["summary"]
            output = {
                row["output_mode"]: row
                for row in result[split]["by_output_mode"]
            }
            rows.append(
                {
                    "checkpoint": result["checkpoint"],
                    "step": result["step"],
                    "parameter_count": result["parameter_count"],
                    "split": split,
                    "accuracy": summary["accuracy"],
                    "loss": summary["loss"],
                    "number_accuracy": output["number"]["accuracy"],
                    "text_exact_accuracy": output["text"]["accuracy"],
                    "image_template_accuracy": summary[
                        "image_template_accuracy"
                    ],
                    "image_foreground_iou": summary["image_foreground_iou"],
                    "image_pixel_mae": summary["image_pixel_mae"],
                    "n_examples": int(summary["n_examples"]),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaustively evaluate compressed trimodal checkpoints."
    )
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for raw_path in args.checkpoint:
        checkpoint_path = Path(raw_path)
        print(f"evaluating {checkpoint_path}", flush=True)
        result = evaluate_checkpoint(
            checkpoint_path,
            device=device,
            batch_size=args.batch_size,
        )
        results.append(result)
        (output_dir / f"{checkpoint_path.stem}_step_{result['step']}_exhaustive.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    write_summary_csv(results, output_dir / "summary.csv")
    print(output_dir, flush=True)


if __name__ == "__main__":
    main()
