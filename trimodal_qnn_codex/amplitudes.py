from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmplitudeCoefficients(nn.Module):
    """Normalized branch/sector coefficients.

    The returned ``lambda`` vector always satisfies ``sum |lambda|^2 = 1``.
    """

    def __init__(self, sector_count: int, mode: str = "fixed_equal"):
        super().__init__()
        self.sector_count = int(sector_count)
        self.mode = str(mode)
        if self.mode in {"learnable_real", "learnable_complex"}:
            self.logits = nn.Parameter(torch.zeros(self.sector_count))
        else:
            self.register_buffer("logits", torch.zeros(self.sector_count), persistent=False)
        if self.mode == "learnable_complex":
            self.phases = nn.Parameter(torch.zeros(self.sector_count))
        else:
            self.register_buffer("phases", torch.zeros(self.sector_count), persistent=False)
        if self.mode not in {"fixed_equal", "learnable_real", "learnable_complex"}:
            raise ValueError(f"unknown amplitude mode: {self.mode}")

    def forward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "fixed_equal":
            weights = torch.full_like(self.logits, 1.0 / self.sector_count)
        else:
            weights = F.softmax(self.logits, dim=0)
        magnitudes = torch.sqrt(weights.clamp_min(1e-12))
        if self.mode == "learnable_complex":
            phases = self.phases - self.phases[0]
            lambdas = magnitudes.to(torch.complex64) * torch.exp(1j * phases.to(torch.complex64))
        else:
            phases = torch.zeros_like(weights)
            lambdas = magnitudes.to(torch.complex64)
        return lambdas, weights, phases

    def balance_loss(self) -> torch.Tensor:
        _, weights, _ = self()
        target = torch.full_like(weights, 1.0 / self.sector_count)
        return (weights - target).pow(2).mean()
