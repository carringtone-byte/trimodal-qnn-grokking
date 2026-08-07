from __future__ import annotations

import argparse
import json
import random
import time
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import load_config, write_config
from .data import make_datasets
from .diagnostics import cross_ablation_accuracy, evaluate
from .heads import fourier_targets
from .models import TrimodalQNNModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _flatten_same_sum_target(out: dict[str, torch.Tensor], *, target: str) -> torch.Tensor:
    if target == "features":
        value = out["features"]
    elif target == "q_hat":
        value = out["q_hat"]
    elif target == "logits":
        value = out["logits"]
    else:
        raise ValueError("same_sum_target must be one of: features, q_hat, logits")
    return value.reshape(value.shape[0], -1)


def same_sum_loss(
    model: TrimodalQNNModel,
    batch: dict[str, torch.Tensor],
    *,
    train_cfg: dict[str, Any] | None = None,
    base_out: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    train_cfg = train_cfg or {}
    loss_kind = str(train_cfg.get("same_sum_loss_kind", "logit_kl"))
    modulus = model.modulus
    d = torch.randint(1, modulus, batch["a"].shape, device=batch["a"].device)
    shifted = dict(batch)
    shifted["a"] = (batch["a"] + d) % modulus
    shifted["b"] = (batch["b"] - d) % modulus
    out_1 = base_out if base_out is not None else model(batch)
    out_2 = model(shifted)
    if loss_kind in {"representation_mse", "feature_mse"}:
        target_name = str(train_cfg.get("same_sum_target", "features"))
        z1 = _flatten_same_sum_target(out_1, target=target_name)
        z2 = _flatten_same_sum_target(out_2, target=target_name)
        if bool(train_cfg.get("same_sum_normalize", True)):
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
        def squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return (left - right).pow(2).sum(dim=-1).mean()
        if bool(train_cfg.get("same_sum_stop_gradient", True)):
            return 0.5 * (squared_distance(z1, z2.detach()) + squared_distance(z1.detach(), z2))
        return squared_distance(z1, z2)
    if loss_kind != "logit_kl":
        raise ValueError("same_sum_loss_kind must be one of: logit_kl, representation_mse, feature_mse")
    logits_1 = out_1["logits"]
    logits_2 = out_2["logits"]
    p1 = F.log_softmax(logits_1, dim=-1)
    p2 = F.log_softmax(logits_2, dim=-1)
    q1 = p1.exp()
    q2 = p2.exp()
    return 0.5 * (F.kl_div(p1, q2, reduction="batchmean") + F.kl_div(p2, q1, reduction="batchmean"))


def scheduled_same_sum_weight(train_cfg: dict[str, Any], step: int) -> float:
    base_weight = float(train_cfg.get("same_sum_loss_weight", 0.0))
    if base_weight == 0.0:
        return 0.0
    start_step = int(train_cfg.get("same_sum_loss_start_step", 0) or 0)
    warmup_steps = int(train_cfg.get("same_sum_loss_warmup_steps", 0) or 0)
    if step < start_step:
        return 0.0
    if warmup_steps <= 0:
        return base_weight
    progress = min(1.0, max(0.0, float(step - start_step + 1) / float(warmup_steps)))
    return base_weight * progress


def fourier_auxiliary_loss(q_hat: torch.Tensor, labels: torch.Tensor, *, model: TrimodalQNNModel, train_cfg: dict[str, Any]) -> torch.Tensor:
    max_available = int(q_hat.shape[1])
    min_frequency = max(1, int(train_cfg.get("fourier_auxiliary_min_frequency", 1) or 1))
    max_frequency_cfg = train_cfg.get("fourier_auxiliary_max_frequency", None)
    max_frequency = max_available if max_frequency_cfg in {None, "none", "None"} else int(max_frequency_cfg)
    max_frequency = min(max_frequency, max_available)
    if min_frequency > max_frequency:
        return q_hat.new_zeros(())

    pred = q_hat[:, min_frequency - 1 : max_frequency, :]
    if bool(train_cfg.get("fourier_auxiliary_normalize_pairs", True)):
        pred = F.normalize(pred, dim=-1)
    target = fourier_targets(labels, modulus=model.modulus, max_frequency=max_frequency).to(dtype=pred.dtype)
    target = target.reshape(labels.shape[0], max_frequency, 2)[:, min_frequency - 1 : max_frequency, :]
    per_pair = (pred - target).pow(2).mean(dim=-1)

    power = float(train_cfg.get("fourier_auxiliary_frequency_power", 0.0) or 0.0)
    if power:
        freq_weights = torch.arange(min_frequency, max_frequency + 1, dtype=pred.dtype, device=pred.device).pow(power)
        freq_weights = freq_weights / freq_weights.mean().clamp_min(1e-8)
        per_pair = per_pair * freq_weights.unsqueeze(0)
    return per_pair.mean()


def layerwise_head_loss(model: TrimodalQNNModel, labels: torch.Tensor, *, train_cfg: dict[str, Any]) -> torch.Tensor:
    logits_by_layer = getattr(model, "_last_layerwise_logits", [])
    if not logits_by_layer:
        return labels.new_zeros((), dtype=torch.float32)
    losses = [F.cross_entropy(logits, labels) for logits in logits_by_layer]
    if bool(train_cfg.get("layerwise_dirac_depth_weighted", False)):
        weights = torch.arange(1, len(losses) + 1, dtype=losses[0].dtype, device=losses[0].device)
        weights = weights / weights.mean().clamp_min(1e-8)
        return torch.stack([weight * loss for weight, loss in zip(weights, losses)]).mean()
    return torch.stack(losses).mean()


def hard_offset_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    offsets: Any,
    margin: float,
    focus_gamma: float = 0.0,
) -> torch.Tensor:
    if isinstance(offsets, int):
        offsets = [offsets]
    offsets = [int(offset) for offset in offsets if int(offset) > 0]
    if not offsets:
        return logits.new_zeros(())
    idx = torch.arange(labels.shape[0], device=labels.device)
    true_logit = logits[idx, labels]
    parts = []
    for offset in offsets:
        for candidate in (logits[idx, (labels + offset) % logits.shape[1]], logits[idx, (labels - offset) % logits.shape[1]]):
            hinge = F.relu(float(margin) + candidate - true_logit)
            if focus_gamma > 0.0:
                denom = max(abs(float(margin)), 1e-6)
                weights = (hinge.detach() / denom).clamp_min(0.0).pow(float(focus_gamma))
                hinge = hinge * weights
            parts.append(hinge.mean())
    return torch.stack(parts).mean()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve_path(path_text: str | None) -> Path | None:
    if path_text in {None, "", "none", "None"}:
        return None
    return Path(str(path_text))


def finite_gradients(model: TrimodalQNNModel) -> bool:
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


def apply_stabilizers(model: TrimodalQNNModel, train_cfg: dict[str, Any]) -> None:
    parameter_clip = float(train_cfg.get("parameter_clip_abs", 0.0) or 0.0)
    sector_clip = float(train_cfg.get("sector_mixer_clip_abs", 0.0) or 0.0)
    scale_min = float(train_cfg.get("head_scale_log_min", -4.0))
    scale_max = float(train_cfg.get("head_scale_log_max", 4.0))
    with torch.no_grad():
        if parameter_clip > 0.0:
            for param in model.parameters():
                param.clamp_(min=-parameter_clip, max=parameter_clip)
        if sector_clip > 0.0:
            for name, param in model.named_parameters():
                if "sector_mixer" in name:
                    param.clamp_(min=-sector_clip, max=sector_clip)
        for module in model.modules():
            scale = getattr(module, "scale_log", None)
            if isinstance(scale, torch.nn.Parameter):
                scale.clamp_(min=scale_min, max=scale_max)


def nonfinite_row(step: int, start: float, event: str, parts: dict[str, torch.Tensor]) -> dict[str, Any]:
    row: dict[str, Any] = {"step": step, "elapsed_sec": time.time() - start, "event": event}
    for key, value in parts.items():
        value_detached = value.detach()
        row[key] = float(value_detached.cpu()) if value_detached.numel() == 1 and torch.isfinite(value_detached).all() else None
    return row


def train(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 0)))
    device = choose_device(str(config["training"].get("device", "auto")))
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, out_dir / "config.yaml")

    train_ds, heldout_ds = make_datasets(config["data"])
    model = TrimodalQNNModel(config["model"], modulus=int(config["data"]["modulus"])).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 0.003)),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    batch_size = int(config["training"].get("batch_size", 32))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    stream = cycle(loader)
    steps = int(config["training"].get("steps", 50))
    eval_every = int(config["training"].get("eval_every", 10))
    checkpoint_every = int(config["training"].get("checkpoint_every", steps))
    balance_weight = float(config["training"].get("amplitude_balance_weight", 0.0))
    fourier_weight = float(config["training"].get("fourier_auxiliary_weight", 0.0))
    layerwise_weight = float(config["training"].get("layerwise_dirac_loss_weight", 0.0))
    neighbor_weight = float(config["training"].get("hard_neighbor_margin_weight", 0.0))
    max_grad_norm = float(config["training"].get("max_grad_norm", 0.0) or 0.0)
    stop_on_nonfinite = bool(config["training"].get("stop_on_nonfinite", True))
    resume_checkpoint = resolve_path(config["training"].get("resume_checkpoint"))
    resume_optimizer = bool(config["training"].get("resume_optimizer", True))
    start_step = 0
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if resume_optimizer and "optimizer" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        apply_stabilizers(model, config["training"])
    metrics_path = out_dir / "metrics.jsonl"
    if resume_checkpoint is None or not metrics_path.exists():
        metrics_path.write_text("", encoding="utf-8")
    else:
        append_jsonl(
            metrics_path,
            {
                "step": start_step,
                "elapsed_sec": 0.0,
                "event": "resume",
                "resume_checkpoint": str(resume_checkpoint),
                "resume_optimizer": resume_optimizer,
                "lr": float(config["training"].get("lr", 0.003)),
            },
        )
    start = time.time()

    final_eval: dict[str, Any] = {}
    for step in range(start_step + 1, steps + 1):
        model.train()
        batch = batch_to_device(next(stream), device)
        out = model(batch)
        ce = F.cross_entropy(out["logits"], batch["y"])
        layerwise = layerwise_head_loss(model, batch["y"], train_cfg=config["training"]) if layerwise_weight else torch.tensor(0.0, device=device)
        same_sum_weight = scheduled_same_sum_weight(config["training"], step)
        ssl = same_sum_loss(model, batch, train_cfg=config["training"], base_out=out) if same_sum_weight else torch.tensor(0.0, device=device)
        amp = model.amplitudes.balance_loss() if balance_weight else torch.tensor(0.0, device=device)
        fourier_aux = (
            fourier_auxiliary_loss(out["q_hat"], batch["y"], model=model, train_cfg=config["training"])
            if fourier_weight
            else torch.tensor(0.0, device=device)
        )
        neighbor = (
            hard_offset_margin_loss(
                out["logits"],
                batch["y"],
                offsets=config["training"].get("hard_neighbor_offsets", [1]),
                margin=float(config["training"].get("hard_neighbor_margin", 1.0)),
                focus_gamma=float(config["training"].get("hard_neighbor_focus_gamma", 0.0) or 0.0),
            )
            if neighbor_weight
            else torch.tensor(0.0, device=device)
        )
        loss = (
            ce
            + same_sum_weight * ssl
            + balance_weight * amp
            + fourier_weight * fourier_aux
            + layerwise_weight * layerwise
            + neighbor_weight * neighbor
        )
        if not torch.isfinite(loss):
            row = nonfinite_row(
                step,
                start,
                "nonfinite_loss",
                {
                    "loss": loss,
                    "ce_loss": ce,
                    "same_sum_loss": ssl,
                    "amplitude_balance_loss": amp,
                    "fourier_auxiliary_loss": fourier_aux,
                    "layerwise_dirac_loss": layerwise,
                    "hard_neighbor_margin_loss": neighbor,
                },
            )
            append_jsonl(metrics_path, row)
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": config,
                    "nonfinite_row": row,
                },
                out_dir / f"checkpoint_nonfinite_{step}.pt",
            )
            if stop_on_nonfinite:
                raise FloatingPointError(f"non-finite QNN loss at step {step}")
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not finite_gradients(model):
            row = nonfinite_row(
                step,
                start,
                "nonfinite_gradient",
                {
                    "loss": loss,
                    "ce_loss": ce,
                    "same_sum_loss": ssl,
                    "amplitude_balance_loss": amp,
                    "fourier_auxiliary_loss": fourier_aux,
                    "layerwise_dirac_loss": layerwise,
                    "hard_neighbor_margin_loss": neighbor,
                },
            )
            append_jsonl(metrics_path, row)
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": config,
                    "nonfinite_row": row,
                },
                out_dir / f"checkpoint_nonfinite_{step}.pt",
            )
            if stop_on_nonfinite:
                raise FloatingPointError(f"non-finite QNN gradient at step {step}")
            break
        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        apply_stabilizers(model, config["training"])

        if step % eval_every == 0 or step == 1 or step == steps:
            train_eval = evaluate(model, train_ds, batch_size=batch_size, device=device)
            heldout_eval = evaluate(model, heldout_ds, batch_size=batch_size, device=device)
            cross_acc = cross_ablation_accuracy(model, heldout_ds, batch_size=batch_size, device=device)
            row = {
                "step": step,
                "elapsed_sec": time.time() - start,
                "loss": float(loss.detach().cpu()),
                "ce_loss": float(ce.detach().cpu()),
                "same_sum_loss": float(ssl.detach().cpu()),
                "same_sum_loss_weight_effective": same_sum_weight,
                "amplitude_balance_loss": float(amp.detach().cpu()),
                "fourier_auxiliary_loss": float(fourier_aux.detach().cpu()),
                "layerwise_dirac_loss": float(layerwise.detach().cpu()),
                "hard_neighbor_margin_loss": float(neighbor.detach().cpu()),
                "train_accuracy": train_eval["accuracy"],
                "train_loss": train_eval["loss"],
                "heldout_accuracy": heldout_eval["accuracy"],
                "heldout_loss": heldout_eval["loss"],
                "heldout_cross_ablation_accuracy": cross_acc,
                "heldout_fourier_addition_energy": heldout_eval["fourier_addition_energy"],
                "heldout_same_sum_feature_ratio": heldout_eval["same_sum_feature_ratio"],
                "heldout_phase_mae": heldout_eval["phase_mae"],
                "state_norm_mean": float(out["state_norm"].detach().mean().cpu()),
                **model.amplitude_metrics(),
            }
            append_jsonl(metrics_path, row)
            final_eval = {"train": train_eval, "heldout": heldout_eval, "cross_ablation_accuracy": cross_acc, "last_row": row}
            print(json.dumps(row, sort_keys=True))

        if step % checkpoint_every == 0 or step == steps:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": config,
                },
                out_dir / f"checkpoint_{step}.pt",
            )

    summary = {
        "output_dir": str(out_dir),
        "train_metadata": train_ds.metadata(),
        "heldout_metadata": heldout_ds.metadata(),
        "final": final_eval,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train trimodal QNN Codex experiment.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = train(load_config(args.config))
    print(json.dumps({"summary": summary["output_dir"], "parameter_count": summary["parameter_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
