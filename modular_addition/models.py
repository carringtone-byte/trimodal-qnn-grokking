from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .data import TaskSpec


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


class TransformerModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int | None,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.0,
        max_len: int = 4096,
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.class_head = nn.Linear(d_model, num_classes or vocab_size)

    def hidden(self, input_ids: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        batch, length = input_ids.shape
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        x = self.token(input_ids) + self.pos(pos)
        mask = _causal_mask(length, input_ids.device) if causal else None
        return self.encoder(x, mask=mask)

    def forward(self, input_ids: torch.Tensor, *, objective: str) -> torch.Tensor:
        hidden = self.hidden(input_ids, causal=objective == "lm")
        if objective == "lm":
            return self.lm_head(hidden)
        return self.class_head(hidden[:, -1])


class RecurrentModel(nn.Module):
    def __init__(
        self,
        *,
        kind: str,
        vocab_size: int,
        num_classes: int | None,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.class_head = nn.Linear(d_model, num_classes or vocab_size)

    def hidden(self, input_ids: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        hidden, _ = self.rnn(self.token(input_ids))
        return hidden

    def forward(self, input_ids: torch.Tensor, *, objective: str) -> torch.Tensor:
        hidden = self.hidden(input_ids)
        if objective == "lm":
            return self.lm_head(hidden)
        return self.class_head(hidden[:, -1])


class MicroRWKVBlock(nn.Module):
    """A compact RWKV-inspired block for controlled experiments.

    It uses token-mixing with a learned recurrent decay. This is not a drop-in
    replacement for production RWKV kernels; it is a transparent local baseline
    for streaming-state comparisons.
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mix_k = nn.Parameter(torch.full((d_model,), 0.5))
        self.mix_v = nn.Parameter(torch.full((d_model,), 0.5))
        self.mix_r = nn.Parameter(torch.full((d_model,), 0.5))
        self.decay_logit = nn.Parameter(torch.zeros(d_model))
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        self.channel = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        shifted = torch.cat([torch.zeros_like(y[:, :1]), y[:, :-1]], dim=1)
        xk = y * self.mix_k + shifted * (1.0 - self.mix_k)
        xv = y * self.mix_v + shifted * (1.0 - self.mix_v)
        xr = y * self.mix_r + shifted * (1.0 - self.mix_r)
        k = torch.sigmoid(self.key(xk))
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))
        decay = torch.sigmoid(self.decay_logit).view(1, -1)

        state = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        outs = []
        for t in range(x.shape[1]):
            state = decay * state + (1.0 - decay) * (k[:, t] * v[:, t])
            outs.append(r[:, t] * state)
        y = torch.stack(outs, dim=1)
        x = x + self.dropout(self.output(y))
        x = x + self.dropout(self.channel(self.norm2(x)))
        return x


class MicroRWKVModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        num_classes: int | None,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MicroRWKVBlock(d_model, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.class_head = nn.Linear(d_model, num_classes or vocab_size)

    def hidden(self, input_ids: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        x = self.token(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, input_ids: torch.Tensor, *, objective: str) -> torch.Tensor:
        x = self.hidden(input_ids)
        if objective == "lm":
            return self.lm_head(x)
        return self.class_head(x[:, -1])


class StateMLP(nn.Module):
    def __init__(self, *, input_dim: int, output_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, *, objective: str = "regression") -> torch.Tensor:
        return self.net(features)


class SequenceJEPA(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.0,
        max_len: int = 4096,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.mask_token_id = vocab_size - 1
        self.ema_decay = ema_decay
        self.online_token = nn.Embedding(vocab_size, d_model)
        self.target_token = copy.deepcopy(self.online_token)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.online_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        for param in self.target_token.parameters():
            param.requires_grad = False
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def _encode(self, token_emb: nn.Embedding, encoder: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        return encoder(token_emb(input_ids) + self.pos(pos))

    def loss(self, input_ids: torch.Tensor, *, mask_ratio: float = 0.4) -> tuple[torch.Tensor, dict[str, float]]:
        mask = torch.rand(input_ids.shape, device=input_ids.device) < mask_ratio
        if not mask.any():
            mask[:, -1] = True
        context_ids = input_ids.masked_fill(mask, self.mask_token_id)
        online = self.predictor(self._encode(self.online_token, self.online_encoder, context_ids))
        with torch.no_grad():
            target = self._encode(self.target_token, self.target_encoder, input_ids)
        pred_masked = online[mask]
        target_masked = target[mask]
        mse = F.mse_loss(pred_masked, target_masked)
        std = pred_masked.float().std(dim=0)
        variance_penalty = torch.mean(F.relu(1.0 - std))
        loss = mse + 0.01 * variance_penalty
        metrics = {
            "mse": float(mse.detach().cpu()),
            "variance_penalty": float(variance_penalty.detach().cpu()),
            "masked_fraction": float(mask.float().mean().detach().cpu()),
        }
        return loss, metrics

    @torch.no_grad()
    def update_ema(self) -> None:
        decay = self.ema_decay
        online_modules = [self.online_token, self.online_encoder]
        target_modules = [self.target_token, self.target_encoder]
        for online_module, target_module in zip(online_modules, target_modules):
            for online_param, target_param in zip(online_module.parameters(), target_module.parameters()):
                target_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)


def build_model(cfg: dict[str, Any], spec: TaskSpec) -> nn.Module:
    name = cfg.get("name", "transformer")
    if name in {"transformer", "transformer_lm", "transformer_classifier"}:
        if spec.vocab_size is None:
            raise ValueError("Transformer requires vocab_size")
        return TransformerModel(
            vocab_size=spec.vocab_size,
            num_classes=spec.num_classes,
            d_model=int(cfg.get("d_model", 128)),
            n_heads=int(cfg.get("n_heads", 4)),
            n_layers=int(cfg.get("n_layers", 2)),
            d_ff=int(cfg.get("d_ff", 256)),
            dropout=float(cfg.get("dropout", 0.0)),
            max_len=int(cfg.get("max_len", 4096)),
        )
    if name in {"lstm", "gru"}:
        if spec.vocab_size is None:
            raise ValueError(f"{name} requires vocab_size")
        return RecurrentModel(
            kind=name,
            vocab_size=spec.vocab_size,
            num_classes=spec.num_classes,
            d_model=int(cfg.get("d_model", 128)),
            n_layers=int(cfg.get("n_layers", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if name in {"microrwkv", "rwkv"}:
        if spec.vocab_size is None:
            raise ValueError("MicroRWKV requires vocab_size")
        return MicroRWKVModel(
            vocab_size=spec.vocab_size,
            num_classes=spec.num_classes,
            d_model=int(cfg.get("d_model", 128)),
            n_layers=int(cfg.get("n_layers", 2)),
            dropout=float(cfg.get("dropout", 0.0)),
        )
    if name in {"state_mlp", "mlp_regressor"}:
        if spec.input_dim is None or spec.output_dim is None:
            raise ValueError("StateMLP requires input_dim and output_dim")
        return StateMLP(
            input_dim=spec.input_dim,
            output_dim=spec.output_dim,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            n_layers=int(cfg.get("n_layers", 3)),
        )
    if name in {"sequence_jepa", "jepa"}:
        if spec.vocab_size is None:
            raise ValueError("SequenceJEPA requires vocab_size")
        return SequenceJEPA(
            vocab_size=spec.vocab_size,
            d_model=int(cfg.get("d_model", 128)),
            n_heads=int(cfg.get("n_heads", 4)),
            n_layers=int(cfg.get("n_layers", 2)),
            d_ff=int(cfg.get("d_ff", 256)),
            dropout=float(cfg.get("dropout", 0.0)),
            max_len=int(cfg.get("max_len", 4096)),
            ema_decay=float(cfg.get("ema_decay", 0.99)),
        )
    raise ValueError(f"Unsupported model: {name}")
