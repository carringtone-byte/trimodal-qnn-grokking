from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DatasetBundle, build_datasets
from .models import SequenceJEPA, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def infinite_loader(loader: DataLoader) -> Iterable[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def compute_loss_and_metrics(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    objective: str,
    *,
    jepa_mask_ratio: float = 0.4,
    auxiliary_residue_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if objective == "lm":
        hidden = None
        if auxiliary_residue_weight > 0.0:
            if not hasattr(model, "hidden") or not hasattr(model, "class_head") or not hasattr(model, "lm_head"):
                raise TypeError("auxiliary_residue_weight requires a model with hidden(), lm_head, and class_head")
            hidden = model.hidden(batch["input_ids"], causal=True)
            logits = model.lm_head(hidden)
        else:
            logits = model(batch["input_ids"], objective=objective)
        labels = batch["labels"]
        flat_labels = labels.reshape(-1)
        valid = flat_labels != -100
        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            flat_labels,
            ignore_index=-100,
            reduction="none",
        )
        loss_weights = batch.get("loss_weights")
        if loss_weights is not None:
            flat_weights = loss_weights.reshape(-1).to(logits.device)
            flat_weights = torch.where(valid, flat_weights, torch.zeros_like(flat_weights))
            weight_total = flat_weights.sum().clamp_min(1.0)
            loss = (flat_loss * flat_weights).sum() / weight_total
        else:
            flat_weights = valid.float()
            weight_total = valid.float().sum().clamp_min(1.0)
            loss = flat_loss[valid].mean()
        preds = logits.argmax(dim=-1)
        correct = (preds.reshape(-1) == flat_labels).float()
        acc = (correct * flat_weights).sum() / weight_total
        lm_loss = loss
        metrics = {
            "loss": float(lm_loss.detach().cpu()),
            "lm_loss": float(lm_loss.detach().cpu()),
            "accuracy": float(acc.detach().cpu()),
            "lm_accuracy": float(acc.detach().cpu()),
            "bits_per_token": float(loss.detach().cpu() / math.log(2)),
            "tokens": float(weight_total.detach().cpu()),
        }
        if auxiliary_residue_weight > 0.0:
            if hidden is None:
                hidden = model.hidden(batch["input_ids"], causal=True)
            if "residue_labels" not in batch or "aux_positions" not in batch:
                raise KeyError("auxiliary residue loss requires residue_labels and aux_positions in the batch")
            aux_positions = batch["aux_positions"].long()
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            aux_hidden = hidden[batch_idx, aux_positions]
            aux_logits = model.class_head(aux_hidden)
            residue_labels = batch["residue_labels"].long()
            aux_loss = F.cross_entropy(aux_logits, residue_labels)
            aux_preds = aux_logits.argmax(dim=-1)
            aux_acc = (aux_preds == residue_labels).float().mean()
            loss = lm_loss + auxiliary_residue_weight * aux_loss
            metrics.update(
                {
                    "loss": float(loss.detach().cpu()),
                    "aux_residue_loss": float(aux_loss.detach().cpu()),
                    "aux_residue_accuracy": float(aux_acc.detach().cpu()),
                }
            )
        return loss, metrics
    if objective == "classification":
        logits = model(batch["input_ids"], objective=objective)
        labels = batch["labels"]
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        return loss, {"loss": float(loss.detach().cpu()), "accuracy": float(acc.detach().cpu())}
    if objective == "regression":
        preds = model(batch["features"], objective=objective)
        labels = batch["labels"]
        loss = F.mse_loss(preds, labels)
        mae = F.l1_loss(preds, labels)
        return loss, {"loss": float(loss.detach().cpu()), "mse": float(loss.detach().cpu()), "mae": float(mae.detach().cpu())}
    if objective == "jepa":
        if not isinstance(model, SequenceJEPA):
            raise TypeError("objective=jepa requires SequenceJEPA")
        loss, metrics = model.loss(batch["input_ids"], mask_ratio=jepa_mask_ratio)
        metrics["loss"] = float(loss.detach().cpu())
        return loss, metrics
    raise ValueError(f"Unsupported objective: {objective}")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    objective: str,
    device: torch.device,
    *,
    max_batches: int = 50,
    jepa_mask_ratio: float = 0.4,
    auxiliary_residue_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        _, metrics = compute_loss_and_metrics(
            model,
            batch,
            objective,
            jepa_mask_ratio=jepa_mask_ratio,
            auxiliary_residue_weight=auxiliary_residue_weight,
        )
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    model.train()
    return {key: value / max(1, count) for key, value in totals.items()}


def apply_grokfast(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    alpha: float,
    lamb: float,
) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if name not in state:
            state[name] = torch.zeros_like(grad)
        state[name].mul_(alpha).add_(grad, alpha=1.0 - alpha)
        param.grad.add_(state[name], alpha=lamb)


def parameter_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        total += float(param.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def gradient_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def train_from_config(config: dict[str, Any]) -> Path:
    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    output_dir = Path(config.get("output_dir", "runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    bundle = build_datasets(config["dataset"])
    model = build_model(config["model"], bundle.spec).to(device)
    n_params = parameter_count(model)
    training = config.get("training", {})
    objective = training.get("objective", bundle.spec.objective)
    batch_size = int(training.get("batch_size", 64))
    steps = int(training.get("steps", 1000))
    lr = float(training.get("lr", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))
    eval_every = int(training.get("eval_every", max(1, steps // 10)))
    save_every = int(training.get("save_every", 0))
    grad_clip = training.get("grad_clip")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    max_eval_batches = int(training.get("max_eval_batches", 50))
    jepa_mask_ratio = float(training.get("jepa_mask_ratio", 0.4))
    auxiliary_residue_weight = float(training.get("auxiliary_residue_weight", 0.0))
    num_workers = int(training.get("num_workers", 0))
    amp_enabled = bool(training.get("amp", False)) and device.type == "cuda"
    disable_progress = bool(training.get("disable_progress", False))
    drop_last = bool(training.get("drop_last", False))

    train_loader = DataLoader(
        bundle.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    val_loader = DataLoader(bundle.val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = (
        DataLoader(bundle.test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if bundle.test is not None
        else None
    )
    train_iter = infinite_loader(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    grokfast_cfg = training.get("grokfast", {})
    grokfast_enabled = bool(grokfast_cfg.get("enabled", False))
    grokfast_state: dict[str, torch.Tensor] = {}
    grokfast_alpha = float(grokfast_cfg.get("alpha", 0.98))
    grokfast_lambda = float(grokfast_cfg.get("lambda", 1.0))

    start = time.time()
    tokens_seen = 0
    progress = tqdm(range(1, steps + 1), desc=output_dir.name, disable=disable_progress)
    for step in progress:
        model.train()
        batch = move_batch(next(train_iter), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss, train_metrics = compute_loss_and_metrics(
                model,
                batch,
                objective,
                jepa_mask_ratio=jepa_mask_ratio,
                auxiliary_residue_weight=auxiliary_residue_weight,
            )
        tokens_seen += int(train_metrics.get("tokens", 0.0))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grokfast_enabled:
            apply_grokfast(model, grokfast_state, alpha=grokfast_alpha, lamb=grokfast_lambda)
        grad_norm_value = gradient_norm(model)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if isinstance(model, SequenceJEPA):
            model.update_ema()

        if step == 1 or step % eval_every == 0 or step == steps:
            val_metrics = evaluate(
                model,
                val_loader,
                objective,
                device,
                max_batches=max_eval_batches,
                jepa_mask_ratio=jepa_mask_ratio,
                auxiliary_residue_weight=auxiliary_residue_weight,
            )
            record: dict[str, Any] = {
                "step": step,
                "elapsed_sec": time.time() - start,
                "objective": objective,
                "split": "val",
                "parameter_count": n_params,
                "tokens_seen": tokens_seen,
                "tokens_per_parameter": tokens_seen / max(1, n_params),
                "parameter_norm": parameter_norm(model),
                "gradient_norm": grad_norm_value,
            }
            for key, value in train_metrics.items():
                record[f"train_{key}"] = value
            for key, value in val_metrics.items():
                record[f"val_{key}"] = value
            write_jsonl(metrics_path, record)
            progress.set_postfix({k: f"{v:.4g}" for k, v in record.items() if k.endswith("loss") or k.endswith("accuracy")})

        if save_every and step % save_every == 0:
            torch.save({"model": model.state_dict(), "step": step, "config": config}, output_dir / f"checkpoint_{step}.pt")

    if test_loader is not None:
        test_metrics = evaluate(
            model,
            test_loader,
            objective,
            device,
            max_batches=max_eval_batches,
            jepa_mask_ratio=jepa_mask_ratio,
            auxiliary_residue_weight=auxiliary_residue_weight,
        )
        record = {"step": steps, "elapsed_sec": time.time() - start, "objective": objective, "split": "test"}
        for key, value in test_metrics.items():
            record[f"test_{key}"] = value
        write_jsonl(metrics_path, record)

    torch.save({"model": model.state_dict(), "step": steps, "config": config}, output_dir / "checkpoint_final.pt")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    args = parser.parse_args()
    output_dir = train_from_config(load_config(args.config))
    print(output_dir)


if __name__ == "__main__":
    main()
