from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fourier_targets(labels: torch.Tensor, *, modulus: int, max_frequency: int) -> torch.Tensor:
    labels = labels.to(torch.float32)
    ks = torch.arange(1, max_frequency + 1, device=labels.device, dtype=torch.float32)
    angles = 2.0 * math.pi * labels[:, None] * ks[None, :] / float(modulus)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).reshape(labels.shape[0], 2 * max_frequency)


class FourierDeltaHead(nn.Module):
    """Predicts finite Fourier sum phases and synthesizes residue logits."""

    def __init__(self, feature_dim: int, modulus: int, max_frequency: int, *, residual: bool = False):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.modulus = int(modulus)
        self.max_frequency = int(max_frequency)
        self.proj = nn.Linear(self.feature_dim, 2 * self.max_frequency)
        self.frequency_weight_logits = nn.Parameter(torch.zeros(self.max_frequency))
        self.bias = nn.Parameter(torch.zeros(self.modulus))
        self.scale_log = nn.Parameter(torch.tensor(0.0))
        self.residual = bool(residual)
        self.residual_head = nn.Linear(self.feature_dim, self.modulus) if self.residual else None
        classes = torch.arange(self.modulus, dtype=torch.float32)
        ks = torch.arange(1, self.max_frequency + 1, dtype=torch.float32)
        basis_angles = 2.0 * math.pi * classes[:, None] * ks[None, :] / float(self.modulus)
        self.register_buffer("class_cos", torch.cos(basis_angles), persistent=False)
        self.register_buffer("class_sin", torch.sin(basis_angles), persistent=False)

    def logits_from_coefficients(self, coeffs: torch.Tensor) -> torch.Tensor:
        q = coeffs.reshape(coeffs.shape[0], self.max_frequency, 2)
        weights = torch.softmax(self.frequency_weight_logits, dim=0)
        matched = q[:, :, 0] @ (self.class_cos * weights).T + q[:, :, 1] @ (self.class_sin * weights).T
        return torch.exp(self.scale_log) * matched + self.bias

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        coeffs = self.proj(features)
        logits = self.logits_from_coefficients(coeffs)
        if self.residual_head is not None:
            logits = logits + self.residual_head(features)
        return {"logits": logits, "q_hat": coeffs.reshape(features.shape[0], self.max_frequency, 2)}


class DiracDeltaHead(nn.Module):
    """Finite modular delta readout over learned cyclic coefficients."""

    def __init__(
        self,
        feature_dim: int,
        modulus: int,
        max_frequency: int,
        *,
        init_kernel: str = "fejer",
        trainable_kernel: bool = True,
        coefficient_mode: str = "none",
        coefficient_eps: float = 1e-6,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.modulus = int(modulus)
        self.max_frequency = int(max_frequency)
        if self.max_frequency < 1 or self.max_frequency > self.modulus // 2:
            raise ValueError("max_frequency must be in [1, modulus // 2]")
        if coefficient_mode not in {"none", "unit", "soft_unit", "global_soft_unit"}:
            raise ValueError("coefficient_mode must be one of: none, unit, soft_unit, global_soft_unit")
        self.coefficient_mode = str(coefficient_mode)
        self.coefficient_eps = float(coefficient_eps)

        self.norm = nn.LayerNorm(self.feature_dim)
        self.proj = nn.Linear(self.feature_dim, 2 * self.max_frequency)
        self.bias = nn.Parameter(torch.zeros(self.modulus))
        self.scale_log = nn.Parameter(torch.tensor(0.0))

        if init_kernel == "fejer":
            weights = 1.0 - torch.arange(1, self.max_frequency + 1, dtype=torch.float32) / (self.max_frequency + 1)
        elif init_kernel == "dirichlet":
            weights = torch.ones(self.max_frequency, dtype=torch.float32)
        else:
            raise ValueError("init_kernel must be one of: fejer, dirichlet")
        self.frequency_weight_logits = nn.Parameter(torch.log(torch.expm1(weights.clamp_min(1e-4))))
        if not trainable_kernel:
            self.frequency_weight_logits.requires_grad_(False)

        classes = torch.arange(self.modulus, dtype=torch.float32)
        ks = torch.arange(1, self.max_frequency + 1, dtype=torch.float32)
        basis_angles = 2.0 * math.pi * classes[:, None] * ks[None, :] / float(self.modulus)
        self.register_buffer("class_cos", torch.cos(basis_angles), persistent=False)
        self.register_buffer("class_sin", torch.sin(basis_angles), persistent=False)

    def kernel_weights(self) -> torch.Tensor:
        return F.softplus(self.frequency_weight_logits)

    def coefficients_from_features(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.proj(self.norm(features))
        if self.coefficient_mode == "none":
            return raw
        pairs = raw.reshape(raw.shape[0], self.max_frequency, 2)
        pair_norm = pairs.norm(dim=-1, keepdim=True).clamp_min(self.coefficient_eps)
        unit = pairs / pair_norm
        if self.coefficient_mode == "unit":
            return unit.reshape(raw.shape)
        if self.coefficient_mode == "global_soft_unit":
            confidence = torch.tanh(pair_norm.mean(dim=1, keepdim=True))
            return (confidence * unit).reshape(raw.shape)
        confidence = torch.tanh(pair_norm)
        return (confidence * unit).reshape(raw.shape)

    def logits_from_coefficients(self, coeffs: torch.Tensor) -> torch.Tensor:
        q = coeffs.reshape(coeffs.shape[0], self.max_frequency, 2)
        weights = self.kernel_weights()
        matched = q[:, :, 0] @ (self.class_cos * weights).T + q[:, :, 1] @ (self.class_sin * weights).T
        return torch.exp(self.scale_log) * matched + self.bias

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        coeffs = self.coefficients_from_features(features)
        return {
            "logits": self.logits_from_coefficients(coeffs),
            "q_hat": coeffs.reshape(features.shape[0], self.max_frequency, 2),
        }
