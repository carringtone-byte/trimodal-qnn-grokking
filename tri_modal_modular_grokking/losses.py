from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .data import DECODER_TARGET_KINDS, OUTPUT_MODES


@dataclass(frozen=True)
class LossWeights:
    number_ce_weight: float = 1.0
    text_ce_weight: float = 1.0
    image_ce_or_vq_weight: float = 1.0
    image_pixel_foreground_weight: float = 8.0
    image_pixel_l1_weight: float = 0.25
    residue_supcon_weight: float = 0.0
    residue_supcon_tau: float = 0.2


def supervised_residue_contrastive(answer_slot: torch.Tensor, labels: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    if answer_slot.shape[0] < 2:
        return answer_slot.new_zeros(())
    z = F.normalize(answer_slot, dim=-1)
    logits = z @ z.T / tau
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    if not same.any():
        return answer_slot.new_zeros(())
    logits = logits.masked_fill(eye, -1e9)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_row = -(log_probs * same.float()).sum(dim=1) / same.float().sum(dim=1).clamp_min(1.0)
    valid = same.any(dim=1)
    return per_row[valid].mean()


def multimodal_loss(
    outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    batch: dict[str, torch.Tensor],
    *,
    weights: LossWeights,
    uncertainty_log_vars: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    output_mode = batch["output_mode_id"]
    target_kind = batch["decoder_target_kind_id"]
    device = output_mode.device
    total = torch.zeros((), device=device)
    metrics: dict[str, float] = {}
    active_terms = 0

    if uncertainty_log_vars is not None and uncertainty_log_vars.shape != (3,):
        raise ValueError("uncertainty_log_vars must have shape (3,) for number/text/image")

    def add_task_loss(loss: torch.Tensor, base_weight: float, task_idx: int, task_name: str) -> None:
        nonlocal total, active_terms
        if uncertainty_log_vars is None:
            total = total + base_weight * loss
        else:
            log_var = uncertainty_log_vars[task_idx]
            effective_weight = float(base_weight) * torch.exp(-log_var)
            total = total + effective_weight * loss + log_var
            metrics[f"adaptive_{task_name}_weight"] = float(effective_weight.detach().cpu())
            metrics[f"adaptive_{task_name}_log_var"] = float(log_var.detach().cpu())
        active_terms += 1

    number_mask = output_mode == OUTPUT_MODES.index("number")
    if number_mask.any():
        logits = outputs["number_logits"]
        assert isinstance(logits, torch.Tensor)
        loss = F.cross_entropy(logits[number_mask], batch["target_number"][number_mask])
        add_task_loss(loss, weights.number_ce_weight, 0, "number")
        acc = (logits[number_mask].argmax(dim=-1) == batch["target_number"][number_mask]).float().mean()
        metrics.update({"number_loss": float(loss.detach().cpu()), "number_accuracy": float(acc.detach().cpu())})

    text_mask = output_mode == OUTPUT_MODES.index("text")
    if text_mask.any():
        logits = outputs["text_logits"]
        assert isinstance(logits, torch.Tensor)
        labels = batch["target_text_ids"][text_mask]
        mask = batch["target_text_mask"][text_mask].bool()
        flat_loss = F.cross_entropy(logits[text_mask].reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none")
        denom = mask.reshape(-1).float().sum().clamp_min(1.0)
        loss = (flat_loss * mask.reshape(-1).float()).sum() / denom
        add_task_loss(loss, weights.text_ce_weight, 1, "text")
        preds = logits[text_mask].argmax(dim=-1)
        token_acc = ((preds == labels).float() * mask.float()).sum() / denom
        exact = (((preds == labels) | ~mask).all(dim=1)).float().mean()
        metrics.update(
            {
                "text_loss": float(loss.detach().cpu()),
                "text_token_accuracy": float(token_acc.detach().cpu()),
                "text_exact_accuracy": float(exact.detach().cpu()),
            }
        )

    image_proxy_id = DECODER_TARGET_KINDS.index("image_class_proxy")
    image_mask = target_kind == image_proxy_id
    if image_mask.any():
        logits = outputs["image_class_logits"]
        assert isinstance(logits, torch.Tensor)
        loss = F.cross_entropy(logits[image_mask], batch["target_number"][image_mask])
        add_task_loss(loss, weights.image_ce_or_vq_weight, 2, "image")
        acc = (logits[image_mask].argmax(dim=-1) == batch["target_number"][image_mask]).float().mean()
        metrics.update({"image_class_proxy_loss": float(loss.detach().cpu()), "image_class_proxy_accuracy": float(acc.detach().cpu())})

    image_pixels_id = DECODER_TARGET_KINDS.index("image_pixels")
    image_pixels_mask = target_kind == image_pixels_id
    if image_pixels_mask.any():
        logits = outputs.get("image_pixel_logits")
        pixels = outputs.get("image_pixels")
        template_logits = outputs.get("image_template_logits")
        if not isinstance(logits, torch.Tensor) or not isinstance(pixels, torch.Tensor):
            raise ValueError("image_pixels targets require an image-producing pixel decoder")
        targets = batch["target_image"][image_pixels_mask]
        foreground_weights = 1.0 + weights.image_pixel_foreground_weight * (1.0 - targets)
        bce = F.binary_cross_entropy_with_logits(logits[image_pixels_mask], targets, reduction="none")
        weighted_bce = (bce * foreground_weights).sum() / foreground_weights.sum().clamp_min(1.0)
        l1 = F.l1_loss(pixels[image_pixels_mask], targets)
        loss = weighted_bce + weights.image_pixel_l1_weight * l1
        add_task_loss(loss, weights.image_ce_or_vq_weight, 2, "image")
        pred_foreground = pixels[image_pixels_mask] < 0.5
        target_foreground = targets < 0.5
        intersection = (pred_foreground & target_foreground).flatten(1).sum(dim=1).float()
        union = (pred_foreground | target_foreground).flatten(1).sum(dim=1).float()
        foreground_iou = (intersection / union.clamp_min(1.0)).mean()
        metrics.update(
            {
                "image_pixel_loss": float(loss.detach().cpu()),
                "image_pixel_mae": float(l1.detach().cpu()),
                "image_foreground_iou": float(foreground_iou.detach().cpu()),
            }
        )
        if isinstance(template_logits, torch.Tensor):
            template_acc = (
                template_logits[image_pixels_mask].argmax(dim=-1) == batch["target_number"][image_pixels_mask]
            ).float().mean()
            metrics["image_template_accuracy"] = float(template_acc.detach().cpu())

    answer_slot = outputs["answer_slot"]
    if weights.residue_supcon_weight > 0.0 and isinstance(answer_slot, torch.Tensor):
        supcon = supervised_residue_contrastive(answer_slot, batch["s"], tau=weights.residue_supcon_tau)
        total = total + weights.residue_supcon_weight * supcon
        metrics["residue_supcon_loss"] = float(supcon.detach().cpu())

    if active_terms == 0:
        raise ValueError("batch had no active supervised output terms")
    exact_acc = exact_correct_mask(outputs, batch).float().mean()
    metrics["loss"] = float(total.detach().cpu())
    metrics["exact_accuracy"] = float(exact_acc.detach().cpu())
    return total, metrics


def primary_predictions(outputs: dict[str, torch.Tensor | list[torch.Tensor]], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    output_mode = batch["output_mode_id"]
    target_kind = batch["decoder_target_kind_id"]
    preds = torch.full_like(batch["s"], -1)
    number_logits = outputs["number_logits"]
    text_logits = outputs["text_logits"]
    image_logits = outputs["image_class_logits"]
    assert isinstance(number_logits, torch.Tensor)
    assert isinstance(text_logits, torch.Tensor)
    assert isinstance(image_logits, torch.Tensor)
    number_mask = output_mode == OUTPUT_MODES.index("number")
    text_mask = output_mode == OUTPUT_MODES.index("text")
    image_mask = target_kind == DECODER_TARGET_KINDS.index("image_class_proxy")
    image_pixels_mask = target_kind == DECODER_TARGET_KINDS.index("image_pixels")
    preds[number_mask] = number_logits[number_mask].argmax(dim=-1)
    preds[image_mask] = image_logits[image_mask].argmax(dim=-1)
    if image_pixels_mask.any():
        template_logits = outputs.get("image_template_logits")
        if not isinstance(template_logits, torch.Tensor):
            raise ValueError("image_pixels predictions require image_template_logits")
        preds[image_pixels_mask] = template_logits[image_pixels_mask].argmax(dim=-1)
    return preds


def exact_correct_mask(outputs: dict[str, torch.Tensor | list[torch.Tensor]], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    output_mode = batch["output_mode_id"]
    target_kind = batch["decoder_target_kind_id"]
    correct = torch.zeros_like(batch["s"], dtype=torch.bool)
    number_logits = outputs["number_logits"]
    text_logits = outputs["text_logits"]
    image_logits = outputs["image_class_logits"]
    assert isinstance(number_logits, torch.Tensor)
    assert isinstance(text_logits, torch.Tensor)
    assert isinstance(image_logits, torch.Tensor)
    number_mask = output_mode == OUTPUT_MODES.index("number")
    text_mask = output_mode == OUTPUT_MODES.index("text")
    image_mask = target_kind == DECODER_TARGET_KINDS.index("image_class_proxy")
    image_pixels_mask = target_kind == DECODER_TARGET_KINDS.index("image_pixels")
    if number_mask.any():
        correct[number_mask] = number_logits[number_mask].argmax(dim=-1) == batch["target_number"][number_mask]
    if text_mask.any():
        preds = text_logits[text_mask].argmax(dim=-1)
        labels = batch["target_text_ids"][text_mask]
        mask = batch["target_text_mask"][text_mask].bool()
        correct[text_mask] = ((preds == labels) | ~mask).all(dim=1)
    if image_mask.any():
        correct[image_mask] = image_logits[image_mask].argmax(dim=-1) == batch["target_number"][image_mask]
    if image_pixels_mask.any():
        template_logits = outputs.get("image_template_logits")
        if not isinstance(template_logits, torch.Tensor):
            raise ValueError("image_pixels correctness requires image_template_logits")
        correct[image_pixels_mask] = (
            template_logits[image_pixels_mask].argmax(dim=-1) == batch["target_number"][image_pixels_mask]
        )
    return correct


def per_example_loss_and_correct(
    outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    output_mode = batch["output_mode_id"]
    target_kind = batch["decoder_target_kind_id"]
    losses = torch.zeros(batch["s"].shape[0], device=batch["s"].device)
    correct = torch.zeros(batch["s"].shape[0], dtype=torch.bool, device=batch["s"].device)
    number_logits = outputs["number_logits"]
    text_logits = outputs["text_logits"]
    image_logits = outputs["image_class_logits"]
    assert isinstance(number_logits, torch.Tensor)
    assert isinstance(text_logits, torch.Tensor)
    assert isinstance(image_logits, torch.Tensor)

    number_mask = output_mode == OUTPUT_MODES.index("number")
    if number_mask.any():
        losses[number_mask] = F.cross_entropy(number_logits[number_mask], batch["target_number"][number_mask], reduction="none")
        correct[number_mask] = number_logits[number_mask].argmax(dim=-1) == batch["target_number"][number_mask]

    text_mask = output_mode == OUTPUT_MODES.index("text")
    if text_mask.any():
        labels = batch["target_text_ids"][text_mask]
        mask = batch["target_text_mask"][text_mask].bool()
        token_loss = F.cross_entropy(text_logits[text_mask].reshape(-1, text_logits.shape[-1]), labels.reshape(-1), reduction="none").view(labels.shape)
        losses[text_mask] = (token_loss * mask.float()).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
        preds = text_logits[text_mask].argmax(dim=-1)
        correct[text_mask] = ((preds == labels) | ~mask).all(dim=1)

    image_mask = target_kind == DECODER_TARGET_KINDS.index("image_class_proxy")
    if image_mask.any():
        losses[image_mask] = F.cross_entropy(image_logits[image_mask], batch["target_number"][image_mask], reduction="none")
        correct[image_mask] = image_logits[image_mask].argmax(dim=-1) == batch["target_number"][image_mask]

    image_pixels_mask = target_kind == DECODER_TARGET_KINDS.index("image_pixels")
    if image_pixels_mask.any():
        logits = outputs.get("image_pixel_logits")
        pixels = outputs.get("image_pixels")
        template_logits = outputs.get("image_template_logits")
        if not isinstance(logits, torch.Tensor) or not isinstance(pixels, torch.Tensor) or not isinstance(template_logits, torch.Tensor):
            raise ValueError("image_pixels evaluation requires pixel and template outputs")
        targets = batch["target_image"][image_pixels_mask]
        pixel_loss = F.binary_cross_entropy_with_logits(logits[image_pixels_mask], targets, reduction="none").flatten(1).mean(dim=1)
        losses[image_pixels_mask] = pixel_loss
        correct[image_pixels_mask] = (
            template_logits[image_pixels_mask].argmax(dim=-1) == batch["target_number"][image_pixels_mask]
        )
    return losses, correct


def weights_from_config(cfg: dict) -> LossWeights:
    return LossWeights(
        number_ce_weight=float(cfg.get("number_ce_weight", 1.0)),
        text_ce_weight=float(cfg.get("text_ce_weight", 1.0)),
        image_ce_or_vq_weight=float(cfg.get("image_ce_or_vq_weight", 1.0)),
        image_pixel_foreground_weight=float(cfg.get("image_pixel_foreground_weight", 8.0)),
        image_pixel_l1_weight=float(cfg.get("image_pixel_l1_weight", 0.25)),
        residue_supcon_weight=float(cfg.get("residue_supcon_weight", 0.0)),
        residue_supcon_tau=float(cfg.get("residue_supcon_tau", 0.2)),
    )
