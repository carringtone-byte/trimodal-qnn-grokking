from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from .data import DECODER_TARGET_KINDS, MultiModalModularConfig, MultiModalModularDataset, OUTPUT_MODES, make_datasets
from .losses import multimodal_loss, per_example_loss_and_correct, weights_from_config
from .models import build_model, move_batch, parameter_count
from .render import save_tensor_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infinite_loader(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def append_per_cell(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "step",
        "split",
        "mode_a",
        "mode_b",
        "output_mode",
        "decoder_target_kind",
        "accuracy",
        "loss",
        "n_examples",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def truncate_jsonl_after_step(path: Path, step: int) -> None:
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("step", -1)) <= step:
            kept.append(json.dumps(record, sort_keys=True))
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def truncate_csv_after_step(path: Path, step: int) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [row for row in reader if int(row.get("step", -1)) <= step]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def balanced_output_accuracy(rows: list[dict[str, Any]]) -> float:
    """Worst output-modality accuracy, aggregated by example count."""
    totals: dict[str, list[float]] = {}
    for row in rows:
        key = str(row["output_mode"])
        n = float(row["n_examples"])
        stats = totals.setdefault(key, [0.0, 0.0])
        stats[0] += float(row["accuracy"]) * n
        stats[1] += n
    return min(correct / max(1.0, n) for correct, n in totals.values())


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer.detach())


@torch.no_grad()
def save_image_predictions(
    model: torch.nn.Module,
    dataset: MultiModalModularDataset,
    device: torch.device,
    out_dir: Path,
    *,
    step: int,
    n_images: int = 8,
) -> None:
    """Save generated image answers and their targets for visual inspection."""
    if dataset.cfg.decoder_target_kind != "image_pixels":
        return
    selected: list[dict[str, torch.Tensor]] = []
    seen_sums: set[int] = set()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if int(sample["output_mode_id"]) != OUTPUT_MODES.index("image"):
            continue
        residue = int(sample["s"])
        if residue in seen_sums:
            continue
        selected.append(sample)
        seen_sums.add(residue)
        if len(selected) >= n_images:
            break
    if not selected:
        return
    batch = move_batch(default_collate(selected), device)
    was_training = model.training
    model.eval()
    outputs = model(batch)
    pixels = outputs.get("image_pixels")
    template_logits = outputs.get("image_template_logits")
    if not isinstance(pixels, torch.Tensor) or not isinstance(template_logits, torch.Tensor):
        raise ValueError("image pixel previews require generated pixels and template logits")
    preview_dir = out_dir / "image_previews" / f"step_{step:06d}"
    rows = []
    predicted = template_logits.argmax(dim=-1)
    for row in range(pixels.shape[0]):
        target_value = int(batch["s"][row].detach().cpu())
        predicted_value = int(predicted[row].detach().cpu())
        prediction_path = preview_dir / f"{row:02d}_target_{target_value:03d}_pred_{predicted_value:03d}.png"
        target_path = preview_dir / f"{row:02d}_target_{target_value:03d}_reference.png"
        save_tensor_image(pixels[row], prediction_path)
        save_tensor_image(batch["target_image"][row], target_path)
        rows.append(
            {
                "row": row,
                "a": int(batch["a"][row].detach().cpu()),
                "b": int(batch["b"][row].detach().cpu()),
                "target": target_value,
                "template_prediction": predicted_value,
                "prediction_path": str(prediction_path),
                "reference_path": str(target_path),
            }
        )
    (preview_dir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if was_training:
        model.train()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: MultiModalModularDataset,
    loader: DataLoader,
    device: torch.device,
    *,
    step: int,
    split: str,
    max_batches: int | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    n_total = 0
    image_mae_sum = 0.0
    image_iou_sum = 0.0
    image_template_correct = 0.0
    image_total = 0
    cell_stats: dict[int, dict[str, float]] = {}
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        outputs = model(batch)
        losses, correct = per_example_loss_and_correct(outputs, batch)
        loss_sum += float(losses.sum().detach().cpu())
        correct_sum += float(correct.float().sum().detach().cpu())
        n_total += int(correct.numel())
        image_pixels_mask = batch["decoder_target_kind_id"] == DECODER_TARGET_KINDS.index("image_pixels")
        if image_pixels_mask.any():
            pixels = outputs.get("image_pixels")
            template_logits = outputs.get("image_template_logits")
            if not isinstance(pixels, torch.Tensor) or not isinstance(template_logits, torch.Tensor):
                raise ValueError("image_pixels evaluation requires generated pixels and template logits")
            targets = batch["target_image"][image_pixels_mask]
            selected_pixels = pixels[image_pixels_mask]
            per_image_mae = (selected_pixels - targets).abs().flatten(1).mean(dim=1)
            pred_foreground = selected_pixels < 0.5
            target_foreground = targets < 0.5
            intersection = (pred_foreground & target_foreground).flatten(1).sum(dim=1).float()
            union = (pred_foreground | target_foreground).flatten(1).sum(dim=1).float()
            per_image_iou = intersection / union.clamp_min(1.0)
            count = int(image_pixels_mask.sum().detach().cpu())
            image_mae_sum += float(per_image_mae.sum().detach().cpu())
            image_iou_sum += float(per_image_iou.sum().detach().cpu())
            image_template_correct += float(
                (
                    template_logits[image_pixels_mask].argmax(dim=-1)
                    == batch["target_number"][image_pixels_mask]
                )
                .float()
                .sum()
                .detach()
                .cpu()
            )
            image_total += count
        for cell_id in batch["cell_id"].detach().cpu().unique().tolist():
            mask = batch["cell_id"] == int(cell_id)
            stats = cell_stats.setdefault(int(cell_id), {"loss": 0.0, "correct": 0.0, "n": 0.0})
            stats["loss"] += float(losses[mask].sum().detach().cpu())
            stats["correct"] += float(correct[mask].float().sum().detach().cpu())
            stats["n"] += float(mask.sum().detach().cpu())
    rows: list[dict[str, Any]] = []
    for cell_id, stats in sorted(cell_stats.items()):
        cell = dataset.cells[cell_id]
        kind = dataset.target_kind(cell.output_mode)
        rows.append(
            {
                "step": step,
                "split": split,
                "mode_a": cell.mode_a,
                "mode_b": cell.mode_b,
                "output_mode": cell.output_mode,
                "decoder_target_kind": kind,
                "accuracy": stats["correct"] / max(1.0, stats["n"]),
                "loss": stats["loss"] / max(1.0, stats["n"]),
                "n_examples": int(stats["n"]),
            }
        )
    model.train()
    summary = {
        "loss": loss_sum / max(1, n_total),
        "accuracy": correct_sum / max(1, n_total),
        "n_examples": float(n_total),
    }
    if image_total:
        summary.update(
            {
                "image_pixel_mae": image_mae_sum / image_total,
                "image_foreground_iou": image_iou_sum / image_total,
                "image_template_accuracy": image_template_correct / image_total,
                "image_examples": float(image_total),
            }
        )
    return summary, rows


def train_from_config(
    config: dict[str, Any],
    *,
    resume_checkpoint: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
) -> Path:
    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    out_dir = Path(config.get("output_dir", "tri_modal_modular_grokking/runs/smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    dataset_cfg = MultiModalModularConfig.from_dict(config.get("dataset", {}))
    train_ds, heldout_ds = make_datasets(dataset_cfg)
    metadata = train_ds.metadata()
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    model = build_model(config.get("model", {}), metadata).to(device)
    train_cfg = config.get("training", {})
    if initial_checkpoint is None:
        initial_checkpoint = train_cfg.get("initial_checkpoint")
    if resume_checkpoint is not None and initial_checkpoint is not None:
        raise ValueError("resume_checkpoint and initial_checkpoint are mutually exclusive")
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint, map_location=device)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        disallowed_missing = [
            key for key in incompatible.missing_keys if not key.startswith("image_pixel_decoder.")
        ]
        if unexpected or disallowed_missing:
            raise ValueError(
                f"incompatible initialization checkpoint; missing={disallowed_missing}, unexpected={unexpected}"
            )
        (out_dir / "initialization.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(initial_checkpoint),
                    "missing_new_parameters": list(incompatible.missing_keys),
                    "unexpected_parameters": unexpected,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    n_params = parameter_count(model)
    loss_weights = weights_from_config(config.get("loss", {}))
    adaptive_kind = config.get("loss", {}).get("adaptive_task_weighting", "none")
    adaptive_enabled = adaptive_kind is True or str(adaptive_kind).lower() in {
        "uncertainty",
        "homoscedastic_uncertainty",
    }
    uncertainty_log_vars = (
        torch.nn.Parameter(torch.zeros(3, device=device)) if adaptive_enabled else None
    )
    batch_size = int(train_cfg.get("batch_size", 128))
    steps = int(train_cfg.get("steps", 1000))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    eval_every = int(train_cfg.get("eval_every", max(1, steps // 10)))
    save_every = int(train_cfg.get("save_every", 0))
    max_eval_batches = train_cfg.get("max_eval_batches", None)
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    disable_progress = bool(train_cfg.get("disable_progress", False))
    image_preview_every = int(train_cfg.get("image_preview_every", 0))
    image_preview_count = int(train_cfg.get("image_preview_count", 8))
    ema_decay = float(train_cfg.get("ema_decay", 0.0))
    freeze_backbone_steps = int(train_cfg.get("freeze_backbone_steps", 0))
    early_cfg = train_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_min_step = int(early_cfg.get("min_step", 5_000))
    early_train_accuracy = float(early_cfg.get("train_accuracy", 0.999))
    early_heldout_accuracy = float(early_cfg.get("heldout_accuracy", 0.999))
    early_image_template_accuracy = float(
        early_cfg.get("heldout_image_template_accuracy", 0.999)
    )
    early_image_iou = float(
        early_cfg.get("heldout_image_foreground_iou", 0.95)
    )
    early_patience = int(early_cfg.get("patience_evals", 4))
    success_streak = 0
    stopped_early = False
    if freeze_backbone_steps:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)))
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    heldout_loader = DataLoader(heldout_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    train_iter = infinite_loader(train_loader)
    optimizer_groups: list[dict[str, Any]] = [
        {"params": list(model.parameters()), "lr": lr, "weight_decay": weight_decay}
    ]
    if uncertainty_log_vars is not None:
        optimizer_groups.append(
            {
                "params": [uncertainty_log_vars],
                "lr": lr * float(config.get("loss", {}).get("adaptive_lr_scale", 0.1)),
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups)
    metrics_path = out_dir / "metrics.jsonl"
    per_cell_path = out_dir / "per_cell_accuracy.csv"
    start_step = 0
    resume_state: dict[str, Any] | None = None
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        resume_state = checkpoint
        model.load_state_dict(checkpoint["model"])
        start_step = int(checkpoint.get("step", 0))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if uncertainty_log_vars is not None and "uncertainty_log_vars" in checkpoint:
            uncertainty_log_vars.data.copy_(checkpoint["uncertainty_log_vars"].to(device))
        truncate_jsonl_after_step(metrics_path, start_step)
        truncate_csv_after_step(per_cell_path, start_step)
    else:
        for path in (metrics_path, per_cell_path):
            if path.exists():
                path.unlink()

    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_kind = str(scheduler_cfg.get("type", "none")).lower()
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if scheduler_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, steps - start_step),
            eta_min=float(scheduler_cfg.get("min_lr", 0.0)),
        )
        if resume_state is not None and "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
    elif scheduler_kind not in {"none", ""}:
        raise ValueError(f"unsupported scheduler type: {scheduler_kind}")

    ema_model: torch.nn.Module | None = None
    if ema_decay > 0.0:
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must be between 0 and 1")
        ema_model = copy.deepcopy(model).to(device)
        ema_model.requires_grad_(False)
        if resume_state is not None and "ema_model" in resume_state:
            ema_model.load_state_dict(resume_state["ema_model"])

    def checkpoint_payload(step_value: int, *, use_ema_as_model: bool = False) -> dict[str, Any]:
        selected_model = ema_model if use_ema_as_model else model
        assert selected_model is not None
        payload: dict[str, Any] = {
            "model": selected_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "metadata": metadata,
            "step": step_value,
        }
        if ema_model is not None:
            payload["ema_model"] = ema_model.state_dict()
        if uncertainty_log_vars is not None:
            payload["uncertainty_log_vars"] = uncertainty_log_vars.detach().cpu()
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        return payload

    start = time.time()
    best_raw_score = -1.0
    best_ema_score = -1.0
    progress = tqdm(range(start_step + 1, steps + 1), desc=out_dir.name, disable=disable_progress)
    for step in progress:
        if freeze_backbone_steps and step == freeze_backbone_steps + 1:
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(True)
        model.train()
        batch = move_batch(next(train_iter), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, train_metrics = multimodal_loss(
            outputs,
            batch,
            weights=loss_weights,
            uncertainty_log_vars=uncertainty_log_vars,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
        optimizer.step()
        if uncertainty_log_vars is not None:
            uncertainty_log_vars.data.clamp_(-5.0, 5.0)
        if scheduler is not None:
            scheduler.step()
        if ema_model is not None:
            update_ema(ema_model, model, ema_decay)

        if step == 1 or step % eval_every == 0 or step == steps:
            train_eval, train_rows = evaluate(model, train_ds, train_eval_loader, device, step=step, split="train", max_batches=max_eval_batches)
            heldout_eval, heldout_rows = evaluate(model, heldout_ds, heldout_loader, device, step=step, split="heldout", max_batches=max_eval_batches)
            append_per_cell(per_cell_path, train_rows + heldout_rows)
            record: dict[str, Any] = {
                "step": step,
                "elapsed_sec": time.time() - start,
                "parameter_count": n_params,
                "train_eval_loss": train_eval["loss"],
                "train_eval_accuracy": train_eval["accuracy"],
                "heldout_loss": heldout_eval["loss"],
                "heldout_accuracy": heldout_eval["accuracy"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "heldout_balanced_output_accuracy": balanced_output_accuracy(heldout_rows),
            }
            for key, value in train_eval.items():
                if key not in {"loss", "accuracy", "n_examples"}:
                    record[f"train_{key}"] = value
            for key, value in heldout_eval.items():
                if key not in {"loss", "accuracy", "n_examples"}:
                    record[f"heldout_{key}"] = value
            for key, value in train_metrics.items():
                record[f"last_batch_{key}"] = value
            raw_score = balanced_output_accuracy(heldout_rows)
            if raw_score > best_raw_score:
                best_raw_score = raw_score
                torch.save(checkpoint_payload(step), out_dir / "checkpoint_best.pt")
            if ema_model is not None:
                ema_train_eval, ema_train_rows = evaluate(
                    ema_model, train_ds, train_eval_loader, device,
                    step=step, split="train_ema", max_batches=max_eval_batches,
                )
                ema_heldout_eval, ema_heldout_rows = evaluate(
                    ema_model, heldout_ds, heldout_loader, device,
                    step=step, split="heldout_ema", max_batches=max_eval_batches,
                )
                append_per_cell(per_cell_path, ema_train_rows + ema_heldout_rows)
                for key, value in ema_train_eval.items():
                    record[f"ema_train_{key}"] = value
                for key, value in ema_heldout_eval.items():
                    record[f"ema_heldout_{key}"] = value
                ema_score = balanced_output_accuracy(ema_heldout_rows)
                record["ema_heldout_balanced_output_accuracy"] = ema_score
                if ema_score > best_ema_score:
                    best_ema_score = ema_score
                    torch.save(
                        checkpoint_payload(step, use_ema_as_model=True),
                        out_dir / "checkpoint_best_ema.pt",
                    )
            write_jsonl(metrics_path, record)
            progress.set_postfix({"heldout_acc": f"{heldout_eval['accuracy']:.4f}", "loss": f"{heldout_eval['loss']:.4f}"})
            if image_preview_every and (step == 1 or step % image_preview_every == 0 or step == steps):
                save_image_predictions(
                    model,
                    heldout_ds,
                    device,
                    out_dir,
                    step=step,
                    n_images=image_preview_count,
                )
            qualifies = (
                step >= early_min_step
                and train_eval["accuracy"] >= early_train_accuracy
                and heldout_eval["accuracy"] >= early_heldout_accuracy
                and heldout_eval.get("image_template_accuracy", 0.0)
                >= early_image_template_accuracy
                and heldout_eval.get("image_foreground_iou", 0.0)
                >= early_image_iou
            )
            success_streak = success_streak + 1 if qualifies else 0
            if early_enabled and success_streak >= early_patience:
                stopped_early = True

        if save_every and step % save_every == 0:
            torch.save(checkpoint_payload(step), out_dir / f"checkpoint_{step}.pt")
        if stopped_early:
            break

    final_step = step
    torch.save(checkpoint_payload(final_step), out_dir / "checkpoint_final.pt")
    if ema_model is not None:
        torch.save(
            checkpoint_payload(final_step, use_ema_as_model=True),
            out_dir / "checkpoint_ema_final.pt",
        )
    final_summary = {
        "run_name": out_dir.name,
        "status": "early_stopped_success" if stopped_early else "completed",
        "final_step": final_step,
        "elapsed_sec": time.time() - start,
        "modulus": dataset_cfg.modulus,
        "train_fraction": dataset_cfg.train_fraction,
        "seed": seed,
        "model": {**config.get("model", {}), "parameters": n_params},
        "losses": config.get("loss", {}),
        "best_balanced_output_accuracy": {
            "raw": best_raw_score,
            "ema": best_ema_score if ema_model is not None else None,
        },
        "outputs": {
            "metrics": str(metrics_path),
            "per_cell_accuracy": str(per_cell_path),
            "checkpoint_final": str(out_dir / "checkpoint_final.pt"),
        },
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tri-modal modular-addition model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--initial-checkpoint", default=None)
    args = parser.parse_args()
    out_dir = train_from_config(
        load_config(args.config),
        resume_checkpoint=args.resume_checkpoint,
        initial_checkpoint=args.initial_checkpoint,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
