from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .data import DECODER_TARGET_KINDS, INPUT_MODES, OUTPUT_MODES
from .render import RenderConfig, render_answer_image


SPECIAL_TOKENS = ("bos", "plus", "mod", "answer_query")


@dataclass(frozen=True)
class MultiModalModelConfig:
    modulus: int = 97
    text_vocab_size: int = 64
    text_len: int = 4
    image_height: int = 64
    image_width: int = 128
    image_tokens: int = 8
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    max_answer_len: int = 4
    image_output_kind: str = "image_class_proxy"
    image_decoder_channels: int = 32
    image_template_temperature: float = 0.02
    tie_number_embeddings: bool = False
    tie_text_embeddings: bool = False
    tie_image_class_to_number: bool = False

    @classmethod
    def from_dict(cls, cfg: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> "MultiModalModelConfig":
        data = dict(cfg)
        if metadata is not None:
            data.setdefault("modulus", metadata["config"]["modulus"])
            data.setdefault("text_vocab_size", len(metadata["text_vocab"]))
            data.setdefault("text_len", metadata["text_len"])
            render = metadata["config"].get("render", {})
            data.setdefault("image_height", render.get("height", 64))
            data.setdefault("image_width", render.get("width", 128))
            data.setdefault("max_answer_len", metadata["text_len"])
            data.setdefault("image_output_kind", metadata["config"].get("decoder_target_kind", "image_class_proxy"))
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in known})


class ImageEncoder(nn.Module):
    def __init__(self, *, d_model: int, image_tokens: int):
        super().__init__()
        hidden = max(8, d_model // 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hidden, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.image_tokens = image_tokens
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images)
        x = F.adaptive_avg_pool2d(x, (1, self.image_tokens)).squeeze(2).transpose(1, 2)
        return self.norm(x)


class ImagePixelDecoder(nn.Module):
    """Decode the shared answer slot into an actual grayscale answer image."""

    def __init__(self, *, d_model: int, height: int, width: int, channels: int = 32):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.seed_height = max(1, (self.height + 7) // 8)
        self.seed_width = max(1, (self.width + 7) // 8)
        c0 = max(8, int(channels))
        c1 = max(8, c0 // 2)
        c2 = max(4, c1 // 2)
        self.seed_channels = c0
        self.project = nn.Sequential(
            nn.Linear(d_model, c0 * self.seed_height * self.seed_width),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c2, 1, kernel_size=4, stride=2, padding=1),
        )
        final = self.decoder[-1]
        assert isinstance(final, nn.ConvTranspose2d)
        if final.bias is not None:
            nn.init.constant_(final.bias, 2.0)

    def forward(self, answer_slot: torch.Tensor) -> torch.Tensor:
        x = self.project(answer_slot)
        x = x.view(answer_slot.shape[0], self.seed_channels, self.seed_height, self.seed_width)
        logits = self.decoder(x)
        if logits.shape[-2:] != (self.height, self.width):
            logits = F.interpolate(logits, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return logits


class TiedTextOutput(nn.Module):
    """Decode each answer position with the input token embedding matrix."""

    def __init__(self, weight: nn.Parameter, *, max_answer_len: int, d_model: int, vocab_size: int):
        super().__init__()
        self.weight = weight
        self.position = nn.Parameter(torch.zeros(max_answer_len, d_model))
        self.bias = nn.Parameter(torch.zeros(max_answer_len, vocab_size))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, answer_slot: torch.Tensor) -> torch.Tensor:
        positioned = answer_slot[:, None, :] + self.position[None, :, :]
        logits = torch.einsum("bld,vd->blv", positioned, self.weight)
        return logits + self.bias[None, :, :]


class TriModalBackbone(nn.Module):
    def __init__(self, cfg: MultiModalModelConfig):
        super().__init__()
        self.cfg = cfg
        self.number = nn.Embedding(cfg.modulus, cfg.d_model)
        self.text = nn.Embedding(cfg.text_vocab_size, cfg.d_model)
        self.image = ImageEncoder(d_model=cfg.d_model, image_tokens=cfg.image_tokens)
        self.special = nn.Embedding(len(SPECIAL_TOKENS), cfg.d_model)
        self.input_mode = nn.Embedding(len(INPUT_MODES), cfg.d_model)
        self.output_mode = nn.Embedding(len(OUTPUT_MODES), cfg.d_model)
        self.pos = nn.Embedding(256, cfg.d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=cfg.d_model,
                    nhead=cfg.n_heads,
                    dim_feedforward=cfg.d_ff,
                    dropout=cfg.dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def _operand_tokens(self, batch: dict[str, torch.Tensor], prefix: str) -> list[torch.Tensor]:
        mode_ids = batch[f"{prefix}_mode_id"]
        number_tokens = self.number(batch[f"{prefix}_residue_id"]).unsqueeze(1)
        text_tokens = self.text(batch[f"{prefix}_text_ids"])
        image_tokens = self.image(batch[f"{prefix}_image"])
        out: list[torch.Tensor] = []
        for idx in range(mode_ids.shape[0]):
            mode = int(mode_ids[idx].detach().cpu())
            if INPUT_MODES[mode] == "number":
                out.append(number_tokens[idx])
            elif INPUT_MODES[mode] == "text":
                mask = batch[f"{prefix}_text_mask"][idx].bool()
                out.append(text_tokens[idx, mask])
            elif INPUT_MODES[mode] == "image":
                out.append(image_tokens[idx])
            else:
                raise ValueError(f"unknown input mode id: {mode}")
        return out

    def build_sequence(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = batch["a"].device
        batch_size = batch["a"].shape[0]
        special = self.special(torch.arange(len(SPECIAL_TOKENS), device=device))
        a_tokens = self._operand_tokens(batch, "operand_a")
        b_tokens = self._operand_tokens(batch, "operand_b")
        sequences: list[torch.Tensor] = []
        answer_positions: list[int] = []
        for row in range(batch_size):
            pieces = [
                special[0].unsqueeze(0),
                self.input_mode(batch["operand_a_mode_id"][row]).unsqueeze(0),
                a_tokens[row],
                special[1].unsqueeze(0),
                self.input_mode(batch["operand_b_mode_id"][row]).unsqueeze(0),
                b_tokens[row],
                special[2].unsqueeze(0),
                self.output_mode(batch["output_mode_id"][row]).unsqueeze(0),
                special[3].unsqueeze(0),
            ]
            seq = torch.cat(pieces, dim=0)
            answer_positions.append(seq.shape[0] - 1)
            sequences.append(seq)
        padded = pad_sequence(sequences, batch_first=True)
        key_padding = torch.ones(batch_size, padded.shape[1], dtype=torch.bool, device=device)
        for row, seq in enumerate(sequences):
            key_padding[row, : seq.shape[0]] = False
        pos = torch.arange(padded.shape[1], device=device).unsqueeze(0)
        padded = padded + self.pos(pos)
        return padded, key_padding, torch.tensor(answer_positions, device=device, dtype=torch.long)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_hidden: bool = False,
        patch: dict[str, torch.Tensor | int] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        x, key_padding, answer_positions = self.build_sequence(batch)
        rows = torch.arange(x.shape[0], device=x.device)
        hidden_by_layer: list[torch.Tensor] = []
        answer_slots_by_layer: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, src_key_padding_mask=key_padding)
            if patch is not None and int(patch.get("layer", -999)) == layer_idx:
                values = patch["values"]
                if not isinstance(values, torch.Tensor):
                    raise TypeError("patch values must be a tensor")
                x[rows, answer_positions] = values
            hidden_by_layer.append(x)
            answer_slots_by_layer.append(x[rows, answer_positions])
        x = self.final_norm(x)
        answer_slot = x[rows, answer_positions]
        if patch is not None and int(patch.get("layer", -999)) == -1:
            values = patch["values"]
            if not isinstance(values, torch.Tensor):
                raise TypeError("patch values must be a tensor")
            answer_slot = values
        out: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "answer_slot": answer_slot,
            "answer_positions": answer_positions,
        }
        if return_hidden:
            out["hidden_by_layer"] = hidden_by_layer
            out["answer_slots_by_layer"] = answer_slots_by_layer
        return out


class MultiModalModularModel(nn.Module):
    def __init__(self, cfg: MultiModalModelConfig, *, image_templates: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = TriModalBackbone(cfg)
        self.number_head = nn.Linear(cfg.d_model, cfg.modulus)
        if cfg.tie_number_embeddings:
            self.number_head.weight = self.backbone.number.weight
        self.text_head: nn.Module
        if cfg.tie_text_embeddings:
            self.text_head = TiedTextOutput(
                self.backbone.text.weight,
                max_answer_len=cfg.max_answer_len,
                d_model=cfg.d_model,
                vocab_size=cfg.text_vocab_size,
            )
        else:
            self.text_head = nn.Linear(cfg.d_model, cfg.max_answer_len * cfg.text_vocab_size)
        self.image_class_head = nn.Linear(cfg.d_model, cfg.modulus)
        if cfg.tie_image_class_to_number:
            self.image_class_head.weight = self.backbone.number.weight
        self.image_pixel_decoder = (
            ImagePixelDecoder(
                d_model=cfg.d_model,
                height=cfg.image_height,
                width=cfg.image_width,
                channels=cfg.image_decoder_channels,
            )
            if cfg.image_output_kind == "image_pixels"
            else None
        )
        if image_templates is None:
            image_templates = torch.empty(0, 1, cfg.image_height, cfg.image_width)
        self.register_buffer("image_templates", image_templates.float(), persistent=False)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_hidden: bool = False,
        patch: dict[str, torch.Tensor | int] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        out = self.backbone(batch, return_hidden=return_hidden, patch=patch)
        answer_slot = out["answer_slot"]
        if not isinstance(answer_slot, torch.Tensor):
            raise TypeError("answer_slot must be a tensor")
        out["number_logits"] = self.number_head(answer_slot)
        text_logits = self.text_head(answer_slot)
        out["text_logits"] = text_logits.view(
            answer_slot.shape[0], self.cfg.max_answer_len, self.cfg.text_vocab_size
        )
        out["image_class_logits"] = self.image_class_head(answer_slot)
        if self.image_pixel_decoder is not None:
            image_pixel_logits = self.image_pixel_decoder(answer_slot)
            image_pixels = image_pixel_logits.sigmoid()
            out["image_pixel_logits"] = image_pixel_logits
            out["image_pixels"] = image_pixels
            if self.image_templates.numel():
                squared_error = (image_pixels[:, None] - self.image_templates[None]).square().mean(dim=(-1, -2, -3))
                out["image_template_logits"] = -squared_error / max(self.cfg.image_template_temperature, 1e-8)
        return out


def build_model(model_cfg: dict[str, Any], metadata: dict[str, Any]) -> MultiModalModularModel:
    cfg = MultiModalModelConfig.from_dict(model_cfg, metadata=metadata)
    templates: torch.Tensor | None = None
    if cfg.image_output_kind == "image_pixels":
        render_cfg = RenderConfig(**metadata["config"].get("render", {}))
        templates = torch.stack([render_answer_image(value, render_cfg) for value in range(cfg.modulus)])
    return MultiModalModularModel(cfg, image_templates=templates)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def target_kind_names(ids: torch.Tensor) -> list[str]:
    return [DECODER_TARGET_KINDS[int(idx)] for idx in ids.detach().cpu().tolist()]
