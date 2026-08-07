from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from .train import resolve_device, set_seed


class ModularPairDataset(Dataset):
    def __init__(self, *, modulus: int, train_fraction: float, seed: int, split: str):
        pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
        rng = random.Random(seed)
        rng.shuffle(pairs)
        cut = int(len(pairs) * train_fraction)
        if split == "train":
            self.pairs = pairs[:cut]
        elif split in {"test", "val"}:
            self.pairs = pairs[cut:]
        else:
            raise ValueError(split)
        self.modulus = modulus

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, b = self.pairs[idx]
        return {
            "a": torch.tensor(a, dtype=torch.long),
            "b": torch.tensor(b, dtype=torch.long),
            "labels": torch.tensor((a + b) % self.modulus, dtype=torch.long),
            "wrap": torch.tensor(int(a + b >= self.modulus), dtype=torch.long),
        }


class DataReuploadingCircuit(nn.Module):
    """Batched statevector simulator for a small data-reuploading QNN.

    The circuit uses 7 qubits by default, so the output Hilbert space has 128
    basis states and can directly represent the 97 modular residues.
    """

    def __init__(self, *, n_qubits: int, n_layers: int, modulus: int, input_frequencies: list[float] | None = None):
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.modulus = int(modulus)
        self.dim = 2**self.n_qubits
        if self.dim < modulus:
            raise ValueError("need at least modulus computational basis states")
        self.input_frequencies = [float(freq) for freq in input_frequencies] if input_frequencies else []
        input_dim = 4 * len(self.input_frequencies) if self.input_frequencies else 4
        self.input_dim = input_dim
        self.input_weights = nn.Parameter(0.4 * torch.randn(n_layers, n_qubits, input_dim))
        self.input_bias = nn.Parameter(0.05 * torch.randn(n_layers, n_qubits, 2))
        self.rot = nn.Parameter(0.05 * torch.randn(n_layers, n_qubits, 3))
        self.register_buffer("basis", torch.arange(self.dim, dtype=torch.long), persistent=False)
        self._cnot_perms: dict[tuple[int, int], torch.Tensor] = {}

    def _bit_mask(self, qubit: int) -> torch.Tensor:
        return ((self.basis >> qubit) & 1).bool()

    def initial_state(self, batch: int, device: torch.device) -> torch.Tensor:
        state = torch.zeros(batch, self.dim, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0.0j
        return state

    def apply_ry(self, state: torch.Tensor, qubit: int, theta: torch.Tensor) -> torch.Tensor:
        bit = self._bit_mask(qubit).to(state.device)
        basis = self.basis.to(state.device)
        idx0 = basis[~bit]
        idx1 = (idx0 | (1 << qubit)).to(state.device)
        out = state.clone()
        amp0 = state[:, idx0]
        amp1 = state[:, idx1]
        c = torch.cos(theta / 2).to(state.dtype).unsqueeze(1)
        s = torch.sin(theta / 2).to(state.dtype).unsqueeze(1)
        out[:, idx0] = c * amp0 - s * amp1
        out[:, idx1] = s * amp0 + c * amp1
        return out

    def apply_rz(self, state: torch.Tensor, qubit: int, theta: torch.Tensor) -> torch.Tensor:
        bit = self._bit_mask(qubit).to(state.device)
        phase0 = torch.exp((-0.5j * theta).to(torch.complex64)).unsqueeze(1)
        phase1 = torch.exp((0.5j * theta).to(torch.complex64)).unsqueeze(1)
        out = state.clone()
        out[:, ~bit] = out[:, ~bit] * phase0
        out[:, bit] = out[:, bit] * phase1
        return out

    def apply_rx(self, state: torch.Tensor, qubit: int, theta: torch.Tensor) -> torch.Tensor:
        bit = self._bit_mask(qubit).to(state.device)
        basis = self.basis.to(state.device)
        idx0 = basis[~bit]
        idx1 = (idx0 | (1 << qubit)).to(state.device)
        out = state.clone()
        amp0 = state[:, idx0]
        amp1 = state[:, idx1]
        c = torch.cos(theta / 2).to(state.dtype).unsqueeze(1)
        s = (-1j * torch.sin(theta / 2)).to(state.dtype).unsqueeze(1)
        out[:, idx0] = c * amp0 + s * amp1
        out[:, idx1] = s * amp0 + c * amp1
        return out

    def cnot_perm(self, control: int, target: int, device: torch.device) -> torch.Tensor:
        key = (control, target)
        if key not in self._cnot_perms:
            idx = torch.arange(self.dim, dtype=torch.long)
            control_on = ((idx >> control) & 1).bool()
            flipped = idx ^ (1 << target)
            self._cnot_perms[key] = torch.where(control_on, flipped, idx)
        return self._cnot_perms[key].to(device)

    def apply_cnot(self, state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        return state[:, self.cnot_perm(control, target, state.device)]

    def input_features(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_angle = math.tau * a.float() / self.modulus
        b_angle = math.tau * b.float() / self.modulus
        if not self.input_frequencies:
            return torch.stack([a_angle, b_angle, torch.sin(a_angle), torch.sin(b_angle)], dim=1)
        features = []
        for freq in self.input_frequencies:
            fa = freq * a_angle
            fb = freq * b_angle
            features.extend([torch.sin(fa), torch.cos(fa), torch.sin(fb), torch.cos(fb)])
        return torch.stack(features, dim=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        batch = a.shape[0]
        device = a.device
        state = self.initial_state(batch, device)
        inputs = self.input_features(a, b)
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                weights = self.input_weights[layer, qubit]
                y_theta = inputs @ weights + self.input_bias[layer, qubit, 0]
                z_theta = inputs @ torch.roll(weights, shifts=1) + self.input_bias[layer, qubit, 1]
                state = self.apply_ry(state, qubit, y_theta)
                state = self.apply_rz(state, qubit, z_theta)
                state = self.apply_ry(state, qubit, self.rot[layer, qubit, 0].expand(batch))
                state = self.apply_rz(state, qubit, self.rot[layer, qubit, 1].expand(batch))
                state = self.apply_rx(state, qubit, self.rot[layer, qubit, 2].expand(batch))
            for qubit in range(self.n_qubits):
                state = self.apply_cnot(state, qubit, (qubit + 1) % self.n_qubits)
            if layer % 2 == 1:
                for qubit in range(self.n_qubits - 1, -1, -1):
                    state = self.apply_cnot(state, qubit, (qubit - 1) % self.n_qubits)
        return state

    def forward_layerwise(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        feature_kind: str,
        layer_heads: nn.ModuleList | None = None,
        adapter_projections: nn.ModuleList | None = None,
        adapter_scale: float = 0.1,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        batch = a.shape[0]
        device = a.device
        state = self.initial_state(batch, device)
        base_inputs = self.input_features(a, b)
        inputs = base_inputs
        layer_features: list[torch.Tensor] = []
        layer_logits: list[torch.Tensor] = []
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                weights = self.input_weights[layer, qubit]
                y_theta = inputs @ weights + self.input_bias[layer, qubit, 0]
                z_theta = inputs @ torch.roll(weights, shifts=1) + self.input_bias[layer, qubit, 1]
                state = self.apply_ry(state, qubit, y_theta)
                state = self.apply_rz(state, qubit, z_theta)
                state = self.apply_ry(state, qubit, self.rot[layer, qubit, 0].expand(batch))
                state = self.apply_rz(state, qubit, self.rot[layer, qubit, 1].expand(batch))
                state = self.apply_rx(state, qubit, self.rot[layer, qubit, 2].expand(batch))
            for qubit in range(self.n_qubits):
                state = self.apply_cnot(state, qubit, (qubit + 1) % self.n_qubits)
            if layer % 2 == 1:
                for qubit in range(self.n_qubits - 1, -1, -1):
                    state = self.apply_cnot(state, qubit, (qubit - 1) % self.n_qubits)

            probs = state.abs().pow(2)
            if feature_kind == "prob":
                features = probs
            elif feature_kind == "expval":
                features = self.expval_features(probs)
            else:
                raise ValueError(f"unsupported layerwise feature kind: {feature_kind}")
            layer_features.append(features)

            if layer_heads is not None:
                logits = layer_heads[layer](features)
                layer_logits.append(logits)
                if adapter_projections is not None and layer < self.n_layers - 1:
                    coeffs = layer_heads[layer].fourier_features(features)
                    delta_inputs = adapter_projections[layer](coeffs)
                    inputs = base_inputs + float(adapter_scale) * torch.tanh(delta_inputs)
        return state, layer_features, layer_logits

    def probabilities(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.forward(a, b).abs().pow(2)

    def expval_features(self, probs: torch.Tensor) -> torch.Tensor:
        features = []
        basis = self.basis.to(probs.device)
        for q in range(self.n_qubits):
            signs = torch.where(((basis >> q) & 1).bool(), -1.0, 1.0).to(probs.device)
            features.append(probs @ signs)
        for q in range(self.n_qubits):
            r = (q + 1) % self.n_qubits
            signs_q = torch.where(((basis >> q) & 1).bool(), -1.0, 1.0).to(probs.device)
            signs_r = torch.where(((basis >> r) & 1).bool(), -1.0, 1.0).to(probs.device)
            features.append(probs @ (signs_q * signs_r))
        return torch.stack(features, dim=1)


class FourierDeltaHead(nn.Module):
    """Predicts Fourier coefficients of the modular sum and synthesizes logits.

    If the predicted pair for frequency k is approximately
    [cos(2*pi*k*s/p), sin(2*pi*k*s/p)], its dot product with class c's Fourier
    pair gives cos(2*pi*k*(s-c)/p). Summing over k yields a finite modular
    delta-like kernel over candidate residues.
    """

    def __init__(
        self,
        feature_dim: int,
        modulus: int,
        *,
        max_frequency: int | None = None,
        init_kernel: str = "fejer",
        residual_linear: bool = False,
        residual_scale: float = 1.0,
        trainable_kernel: bool = True,
    ):
        super().__init__()
        self.modulus = int(modulus)
        self.max_frequency = int(max_frequency or (modulus // 2))
        if self.max_frequency < 1 or self.max_frequency > modulus // 2:
            raise ValueError("max_frequency must be in [1, modulus // 2]")
        self.norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, 2 * self.max_frequency)
        self.residual = nn.Linear(feature_dim, modulus) if residual_linear else None
        self.residual_scale = float(residual_scale)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.zeros(modulus))
        classes = torch.arange(modulus, dtype=torch.float32)
        basis_cols = []
        for k in range(1, self.max_frequency + 1):
            angle = math.tau * k * classes / modulus
            basis_cols.extend([torch.cos(angle), torch.sin(angle)])
        self.register_buffer("class_basis", torch.stack(basis_cols, dim=1), persistent=False)
        if init_kernel == "fejer":
            weights = 1.0 - torch.arange(1, self.max_frequency + 1, dtype=torch.float32) / (self.max_frequency + 1)
        elif init_kernel == "dirichlet":
            weights = torch.ones(self.max_frequency, dtype=torch.float32)
        else:
            weights = torch.ones(self.max_frequency, dtype=torch.float32)
        inv_softplus = torch.log(torch.expm1(weights.clamp_min(1e-4)))
        self.frequency_weight_logits = nn.Parameter(inv_softplus)
        if not trainable_kernel:
            self.frequency_weight_logits.requires_grad_(False)

    def kernel_weights(self) -> torch.Tensor:
        return F.softplus(self.frequency_weight_logits).repeat_interleave(2)

    def coefficients_from_normed(self, normed: torch.Tensor) -> torch.Tensor:
        return self.proj(normed)

    def fourier_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.coefficients_from_normed(self.norm(features))

    def raw_fourier_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(features))

    def delta_logits_from_coefficients(self, coeffs: torch.Tensor) -> torch.Tensor:
        logits = coeffs @ (self.class_basis * self.kernel_weights()).T
        logits = self.scale.exp().clamp(max=100.0) * logits + self.bias
        return logits

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normed = self.norm(features)
        coeffs = self.coefficients_from_normed(normed)
        logits = self.delta_logits_from_coefficients(coeffs)
        if self.residual is not None:
            logits = logits + self.residual_scale * self.residual(normed)
        return logits


class DiracDeltaHead(FourierDeltaHead):
    """Strict finite-modular delta readout over learned phase features.

    The head predicts one 2D phase vector per frequency. In the default `unit`
    mode each pair is normalized to the unit circle before class scoring, so
    class logits are exactly a learned finite circular delta kernel over
    candidate residues.
    """

    def __init__(
        self,
        feature_dim: int,
        modulus: int,
        *,
        max_frequency: int | None = None,
        init_kernel: str = "dirichlet",
        residual_linear: bool = False,
        residual_scale: float = 1.0,
        coefficient_mode: str = "unit",
        coefficient_eps: float = 1e-6,
        trainable_kernel: bool = False,
    ):
        super().__init__(
            feature_dim,
            modulus,
            max_frequency=max_frequency,
            init_kernel=init_kernel,
            residual_linear=residual_linear,
            residual_scale=residual_scale,
            trainable_kernel=trainable_kernel,
        )
        if coefficient_mode not in {"unit", "soft_unit", "global_soft_unit", "none"}:
            raise ValueError("dirac coefficient_mode must be one of: unit, soft_unit, global_soft_unit, none")
        self.coefficient_mode = coefficient_mode
        self.coefficient_eps = float(coefficient_eps)

    def coefficients_from_normed(self, normed: torch.Tensor) -> torch.Tensor:
        raw = self.proj(normed)
        if self.coefficient_mode == "none":
            return raw
        pairs = raw.view(raw.shape[0], self.max_frequency, 2)
        pair_norm = pairs.norm(dim=-1, keepdim=True).clamp_min(self.coefficient_eps)
        unit = pairs / pair_norm
        if self.coefficient_mode == "unit":
            return unit.reshape(raw.shape)
        if self.coefficient_mode == "global_soft_unit":
            confidence = torch.tanh(pair_norm.mean(dim=1, keepdim=True))
            return (confidence * unit).reshape(raw.shape)
        confidence = torch.tanh(pair_norm)
        return (confidence * unit).reshape(raw.shape)


class DiracResidualSharpeningHead(FourierDeltaHead):
    """Magnitude-carrying cyclic readout plus a small Dirac sharpening residual.

    The base path is exactly the existing Fourier-delta readout, so old
    checkpoints can be copied into the same top-level parameter names. The
    added `sharpen_*` parameters start with zero effect and are intended to
    move local decision margins without replacing the learned cyclic rule.
    """

    def __init__(
        self,
        feature_dim: int,
        modulus: int,
        *,
        max_frequency: int | None = None,
        init_kernel: str = "fejer",
        residual_linear: bool = False,
        residual_scale: float = 1.0,
        trainable_kernel: bool = True,
        sharpen_kernel_init: str = "dirichlet",
        sharpen_kernel_trainable: bool = True,
        sharpen_strength_init: float = 0.0,
        sharpen_strength_max: float = 0.25,
        coefficient_eps: float = 1e-6,
    ):
        super().__init__(
            feature_dim,
            modulus,
            max_frequency=max_frequency,
            init_kernel=init_kernel,
            residual_linear=residual_linear,
            residual_scale=residual_scale,
            trainable_kernel=trainable_kernel,
        )
        self.sharpen_strength_max = float(sharpen_strength_max)
        self.coefficient_eps = float(coefficient_eps)
        if sharpen_kernel_init == "fejer":
            weights = 1.0 - torch.arange(1, self.max_frequency + 1, dtype=torch.float32) / (self.max_frequency + 1)
        elif sharpen_kernel_init == "dirichlet":
            weights = torch.ones(self.max_frequency, dtype=torch.float32)
        else:
            weights = torch.ones(self.max_frequency, dtype=torch.float32)
        inv_softplus = torch.log(torch.expm1(weights.clamp_min(1e-4)))
        self.sharpen_frequency_weight_logits = nn.Parameter(inv_softplus)
        if not sharpen_kernel_trainable:
            self.sharpen_frequency_weight_logits.requires_grad_(False)
        init = float(sharpen_strength_init)
        if self.sharpen_strength_max > 0:
            init = max(-0.999, min(0.999, init / self.sharpen_strength_max))
            init = math.atanh(init)
        self.sharpen_strength_raw = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def sharpen_strength(self) -> torch.Tensor:
        return self.sharpen_strength_max * torch.tanh(self.sharpen_strength_raw)

    def sharpen_kernel_weights(self) -> torch.Tensor:
        return F.softplus(self.sharpen_frequency_weight_logits).repeat_interleave(2)

    def dirac_unit_coefficients_from_normed(self, normed: torch.Tensor) -> torch.Tensor:
        raw = self.proj(normed)
        pairs = raw.view(raw.shape[0], self.max_frequency, 2)
        pair_norm = pairs.norm(dim=-1, keepdim=True).clamp_min(self.coefficient_eps)
        return (pairs / pair_norm).reshape(raw.shape)

    def base_logits_from_normed(self, normed: torch.Tensor) -> torch.Tensor:
        coeffs = self.coefficients_from_normed(normed)
        logits = self.delta_logits_from_coefficients(coeffs)
        if self.residual is not None:
            logits = logits + self.residual_scale * self.residual(normed)
        return logits

    def base_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.base_logits_from_normed(self.norm(features))

    def correction_logits_from_normed(self, normed: torch.Tensor) -> torch.Tensor:
        unit_coeffs = self.dirac_unit_coefficients_from_normed(normed)
        correction = unit_coeffs @ (self.class_basis * self.sharpen_kernel_weights()).T
        correction = correction - correction.mean(dim=-1, keepdim=True)
        return self.sharpen_strength() * correction

    def correction_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.correction_logits_from_normed(self.norm(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normed = self.norm(features)
        return self.base_logits_from_normed(normed) + self.correction_logits_from_normed(normed)

    def set_sharpen_strength_override(self, value: float) -> None:
        if not hasattr(self, "sharpen_strength_raw"):
            return
        max_strength = max(float(self.sharpen_strength_max), 1e-8)
        clipped = max(-0.999, min(0.999, float(value) / max_strength))
        with torch.no_grad():
            self.sharpen_strength_raw.copy_(torch.tensor(math.atanh(clipped), dtype=self.sharpen_strength_raw.dtype, device=self.sharpen_strength_raw.device))


class DiracPrimarySharpeningHead(DiracResidualSharpeningHead):
    """A Dirac readout where sharpening is the dominant output path.

    The base Fourier-delta path is retained as a coarse scaffold, but it is
    multiplicatively downweighted so the correction path can become the main
    source of exact boundary selection.
    """

    def __init__(
        self,
        feature_dim: int,
        modulus: int,
        *,
        max_frequency: int | None = None,
        init_kernel: str = "fejer",
        residual_linear: bool = False,
        residual_scale: float = 1.0,
        trainable_kernel: bool = True,
        sharpen_kernel_init: str = "dirichlet",
        sharpen_kernel_trainable: bool = True,
        sharpen_strength_init: float = 1.0,
        sharpen_strength_max: float = 2.0,
        base_scale_init: float = 0.2,
        base_scale_trainable: bool = True,
        coefficient_eps: float = 1e-6,
    ):
        super().__init__(
            feature_dim,
            modulus,
            max_frequency=max_frequency,
            init_kernel=init_kernel,
            residual_linear=residual_linear,
            residual_scale=residual_scale,
            trainable_kernel=trainable_kernel,
            sharpen_kernel_init=sharpen_kernel_init,
            sharpen_kernel_trainable=sharpen_kernel_trainable,
            sharpen_strength_init=sharpen_strength_init,
            sharpen_strength_max=sharpen_strength_max,
            coefficient_eps=coefficient_eps,
        )
        init = math.log(max(float(base_scale_init), 1e-4))
        self.base_scale_raw = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        if not base_scale_trainable:
            self.base_scale_raw.requires_grad_(False)

    def base_scale(self) -> torch.Tensor:
        return torch.exp(self.base_scale_raw).clamp(max=10.0)

    def set_base_scale_override(self, value: float) -> None:
        clipped = max(float(value), 1e-8)
        with torch.no_grad():
            self.base_scale_raw.copy_(torch.tensor(math.log(clipped), dtype=self.base_scale_raw.dtype, device=self.base_scale_raw.device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normed = self.norm(features)
        return self.base_scale() * self.base_logits_from_normed(normed) + self.correction_logits_from_normed(normed)


class QNNClassifier(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        n_qubits: int,
        n_layers: int,
        modulus: int,
        input_frequencies: list[float] | None = None,
        auxiliary_head_moduli: list[int] | None = None,
        readout_type: str = "linear",
        fourier_max_frequency: int | None = None,
        fourier_kernel_init: str = "fejer",
        fourier_residual_linear: bool = False,
        fourier_residual_scale: float = 1.0,
        fourier_kernel_trainable: bool = True,
        dirac_coefficient_mode: str = "unit",
        dirac_coefficient_eps: float = 1e-6,
        dirac_kernel_trainable: bool = False,
        dirac_sharpen_kernel_init: str = "dirichlet",
        dirac_sharpen_kernel_trainable: bool = True,
        dirac_sharpen_strength_init: float = 0.0,
        dirac_sharpen_strength_max: float = 0.25,
        dirac_primary_base_scale_init: float = 0.2,
        dirac_primary_base_scale_trainable: bool = True,
        layerwise_dirac_adapter_scale: float = 0.1,
    ):
        super().__init__()
        if variant not in {"born", "prob_head", "expval_head"}:
            raise ValueError(variant)
        self.variant = variant
        self.modulus = int(modulus)
        self.readout_type = readout_type
        self._last_layerwise_logits: list[torch.Tensor] = []
        self._last_layerwise_features: list[torch.Tensor] = []
        self.circuit = DataReuploadingCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            modulus=modulus,
            input_frequencies=input_frequencies,
        )
        if variant == "prob_head":
            feature_dim = 2**n_qubits
        elif variant == "expval_head":
            feature_dim = 2 * n_qubits
        else:
            feature_dim = 2**n_qubits
        self.feature_dim = feature_dim
        self.layerwise_feature_kind = "prob" if variant in {"prob_head", "born"} else "expval"
        self.layerwise_dirac_adapter_scale = float(layerwise_dirac_adapter_scale)
        layerwise_readouts = {
            "layerwise_dirac_aux",
            "layerwise_dirac_adapter",
            "layerwise_dirac_mean",
            "layerwise_dirac_residual",
            "layerwise_dirac_ensemble",
        }
        if variant == "born":
            self.head = None
        elif readout_type in layerwise_readouts:
            self.layerwise_heads = nn.ModuleList(
                [
                    DiracDeltaHead(
                        feature_dim,
                        modulus,
                        max_frequency=fourier_max_frequency,
                        init_kernel=fourier_kernel_init,
                        residual_linear=False,
                        residual_scale=0.0,
                        coefficient_mode=dirac_coefficient_mode,
                        coefficient_eps=dirac_coefficient_eps,
                        trainable_kernel=dirac_kernel_trainable,
                    )
                    for _ in range(n_layers)
                ]
            )
            if readout_type == "layerwise_dirac_aux":
                self.head = DiracDeltaHead(
                    feature_dim,
                    modulus,
                    max_frequency=fourier_max_frequency,
                    init_kernel=fourier_kernel_init,
                    residual_linear=fourier_residual_linear,
                    residual_scale=fourier_residual_scale,
                    coefficient_mode=dirac_coefficient_mode,
                    coefficient_eps=dirac_coefficient_eps,
                    trainable_kernel=dirac_kernel_trainable,
                )
            elif readout_type == "layerwise_dirac_adapter":
                self.head = DiracDeltaHead(
                    feature_dim,
                    modulus,
                    max_frequency=fourier_max_frequency,
                    init_kernel=fourier_kernel_init,
                    residual_linear=fourier_residual_linear,
                    residual_scale=fourier_residual_scale,
                    coefficient_mode=dirac_coefficient_mode,
                    coefficient_eps=dirac_coefficient_eps,
                    trainable_kernel=dirac_kernel_trainable,
                )
                coeff_dim = 2 * int(fourier_max_frequency or (modulus // 2))
                self.layerwise_adapters = nn.ModuleList(
                    [nn.Linear(coeff_dim, self.circuit.input_dim) for _ in range(n_layers)]
                )
            elif readout_type in {"layerwise_dirac_residual", "layerwise_dirac_ensemble"}:
                self.head = None
                self.layerwise_logit_weights = nn.Parameter(torch.zeros(n_layers))
            else:
                self.head = None
        elif readout_type == "fourier_delta":
            self.head = FourierDeltaHead(
                feature_dim,
                modulus,
                max_frequency=fourier_max_frequency,
                init_kernel=fourier_kernel_init,
                residual_linear=fourier_residual_linear,
                residual_scale=fourier_residual_scale,
                trainable_kernel=fourier_kernel_trainable,
            )
        elif readout_type == "dirac_delta":
            self.head = DiracDeltaHead(
                feature_dim,
                modulus,
                max_frequency=fourier_max_frequency,
                init_kernel=fourier_kernel_init,
                residual_linear=fourier_residual_linear,
                residual_scale=fourier_residual_scale,
                coefficient_mode=dirac_coefficient_mode,
                coefficient_eps=dirac_coefficient_eps,
                trainable_kernel=dirac_kernel_trainable,
            )
        elif readout_type == "dirac_residual_sharpen":
            self.head = DiracResidualSharpeningHead(
                feature_dim,
                modulus,
                max_frequency=fourier_max_frequency,
                init_kernel=fourier_kernel_init,
                residual_linear=fourier_residual_linear,
                residual_scale=fourier_residual_scale,
                trainable_kernel=fourier_kernel_trainable,
                sharpen_kernel_init=dirac_sharpen_kernel_init,
                sharpen_kernel_trainable=dirac_sharpen_kernel_trainable,
                sharpen_strength_init=dirac_sharpen_strength_init,
                sharpen_strength_max=dirac_sharpen_strength_max,
                coefficient_eps=dirac_coefficient_eps,
            )
        elif readout_type == "dirac_primary_sharpen":
            self.head = DiracPrimarySharpeningHead(
                feature_dim,
                modulus,
                max_frequency=fourier_max_frequency,
                init_kernel=fourier_kernel_init,
                residual_linear=fourier_residual_linear,
                residual_scale=fourier_residual_scale,
                trainable_kernel=fourier_kernel_trainable,
                sharpen_kernel_init=dirac_sharpen_kernel_init,
                sharpen_kernel_trainable=dirac_sharpen_kernel_trainable,
                sharpen_strength_init=dirac_sharpen_strength_init,
                sharpen_strength_max=dirac_sharpen_strength_max,
                base_scale_init=dirac_primary_base_scale_init,
                base_scale_trainable=dirac_primary_base_scale_trainable,
                coefficient_eps=dirac_coefficient_eps,
            )
        elif variant == "prob_head":
            self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, modulus))
        elif variant == "expval_head":
            self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, modulus))
        else:
            raise ValueError(f"unsupported readout_type: {readout_type}")
        self.auxiliary_heads = nn.ModuleDict()
        for aux_modulus in auxiliary_head_moduli or []:
            aux_modulus = int(aux_modulus)
            if aux_modulus > 1:
                self.auxiliary_heads[str(aux_modulus)] = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, aux_modulus),
                )

    def forward_with_features(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._last_layerwise_logits = []
        self._last_layerwise_features = []
        if self.readout_type in {
            "layerwise_dirac_aux",
            "layerwise_dirac_adapter",
            "layerwise_dirac_mean",
            "layerwise_dirac_residual",
            "layerwise_dirac_ensemble",
        }:
            adapter_projections = getattr(self, "layerwise_adapters", None)
            _, layer_features, layer_logits = self.circuit.forward_layerwise(
                a,
                b,
                feature_kind=self.layerwise_feature_kind,
                layer_heads=self.layerwise_heads,
                adapter_projections=adapter_projections,
                adapter_scale=self.layerwise_dirac_adapter_scale,
            )
            self._last_layerwise_logits = layer_logits
            self._last_layerwise_features = layer_features
            final_features = layer_features[-1]
            if self.readout_type in {"layerwise_dirac_residual", "layerwise_dirac_ensemble"}:
                weights = F.softmax(self.layerwise_logit_weights, dim=0)
                logits = torch.stack(layer_logits, dim=0)
                return (weights.view(-1, 1, 1) * logits).sum(dim=0), final_features
            if self.readout_type == "layerwise_dirac_mean":
                return torch.stack(layer_logits, dim=0).mean(dim=0), final_features
            return self.head(final_features), final_features
        probs = self.circuit.probabilities(a, b)
        if self.variant == "born":
            class_probs = probs[:, : self.modulus]
            class_probs = class_probs / class_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return class_probs.clamp_min(1e-9).log(), probs
        if self.variant == "prob_head":
            return self.head(probs), probs
        features = self.circuit.expval_features(probs)
        return self.head(features), features

    def fourier_coefficients(self, features: torch.Tensor) -> torch.Tensor | None:
        if isinstance(self.head, FourierDeltaHead):
            return self.head.fourier_features(features)
        return None

    def base_logits_from_features(self, features: torch.Tensor) -> torch.Tensor | None:
        if hasattr(self.head, "base_logits"):
            return self.head.base_logits(features)
        return None

    def correction_logits_from_features(self, features: torch.Tensor) -> torch.Tensor | None:
        if hasattr(self.head, "correction_logits"):
            return self.head.correction_logits(features)
        return None

    def auxiliary_logits(self, features: torch.Tensor) -> dict[int, torch.Tensor]:
        return {int(modulus): head(features) for modulus, head in self.auxiliary_heads.items()}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_features(a, b)
        return logits


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: QNNClassifier, loader: DataLoader, device: torch.device, *, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    wrap_total = 0
    wrap_correct = 0
    nowrap_total = 0
    nowrap_correct = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        logits = model(batch["a"], batch["b"])
        loss = F.nll_loss(logits, batch["labels"]) if model.variant == "born" else F.cross_entropy(logits, batch["labels"])
        pred = logits.argmax(dim=-1)
        ok = pred.eq(batch["labels"])
        total += int(ok.numel())
        correct += int(ok.sum())
        loss_total += float(loss.detach().cpu()) * int(ok.numel())
        wrap = batch["wrap"].bool()
        wrap_total += int(wrap.sum())
        nowrap_total += int((~wrap).sum())
        wrap_correct += int((ok & wrap).sum())
        nowrap_correct += int((ok & ~wrap).sum())
    return {
        "accuracy": correct / max(total, 1),
        "loss": loss_total / max(total, 1),
        "wrap_accuracy": wrap_correct / max(wrap_total, 1),
        "nowrap_accuracy": nowrap_correct / max(nowrap_total, 1),
        "examples": float(total),
    }


def load_previous_best(path: Path, variant: str) -> tuple[float, dict[str, Any]]:
    if not path.exists():
        return 0.0, {}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0, {}
    for row in summary.get("variants", []):
        if row.get("variant") == variant:
            return float(row.get("best_test_accuracy", 0.0)), dict(row.get("best_record", {}))
    return 0.0, {}


def residue_logits_from_class_logits(logits: torch.Tensor, *, class_modulus: int, residue_modulus: int) -> torch.Tensor:
    residues = torch.arange(class_modulus, device=logits.device) % residue_modulus
    grouped = []
    for residue in range(residue_modulus):
        grouped.append(torch.logsumexp(logits[:, residues == residue], dim=1))
    return torch.stack(grouped, dim=1)


def auxiliary_residue_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_modulus: int,
    train_cfg: dict[str, Any],
) -> torch.Tensor:
    specs = train_cfg.get("auxiliary_residue_losses") or []
    if isinstance(specs, dict):
        specs = [{"modulus": key, "weight": value} for key, value in specs.items()]
    aux = logits.new_zeros(())
    for spec in specs:
        residue_modulus = int(spec.get("modulus", 0))
        weight = float(spec.get("weight", 1.0))
        if residue_modulus <= 1 or weight == 0.0:
            continue
        aux_logits = residue_logits_from_class_logits(
            logits,
            class_modulus=class_modulus,
            residue_modulus=residue_modulus,
        )
        aux = aux + weight * F.cross_entropy(aux_logits, labels % residue_modulus)
    return aux


def auxiliary_head_loss(
    aux_logits: dict[int, torch.Tensor],
    labels: torch.Tensor,
    *,
    train_cfg: dict[str, Any],
) -> torch.Tensor:
    specs = train_cfg.get("auxiliary_residue_losses") or []
    if isinstance(specs, dict):
        specs = [{"modulus": key, "weight": value} for key, value in specs.items()]
    first = next(iter(aux_logits.values()), None)
    if first is None:
        return labels.new_zeros((), dtype=torch.float32).to(labels.device)
    aux = first.new_zeros(())
    for spec in specs:
        residue_modulus = int(spec.get("modulus", 0))
        weight = float(spec.get("weight", 1.0))
        if residue_modulus <= 1 or weight == 0.0 or residue_modulus not in aux_logits:
            continue
        aux = aux + weight * F.cross_entropy(aux_logits[residue_modulus], labels % residue_modulus)
    return aux


def fourier_targets(labels: torch.Tensor, *, modulus: int, max_frequency: int) -> torch.Tensor:
    values = labels.float()
    parts = []
    for k in range(1, max_frequency + 1):
        angle = math.tau * k * values / modulus
        parts.extend([torch.cos(angle), torch.sin(angle)])
    return torch.stack(parts, dim=1)


def fourier_auxiliary_loss(
    model: QNNClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_cfg: dict[str, Any],
) -> torch.Tensor:
    weight = float(train_cfg.get("fourier_auxiliary_weight", 0.0))
    coeffs = model.fourier_coefficients(features)
    if weight == 0.0 or coeffs is None:
        return features.new_zeros(())
    max_available = coeffs.shape[1] // 2
    min_frequency = max(1, int(train_cfg.get("fourier_auxiliary_min_frequency", 1)))
    max_frequency = min(int(train_cfg.get("fourier_auxiliary_max_frequency", max_available)), max_available)
    if min_frequency > max_frequency:
        return features.new_zeros(())
    pred = coeffs[:, 2 * (min_frequency - 1) : 2 * max_frequency]
    num_frequencies = max_frequency - min_frequency + 1
    if bool(train_cfg.get("fourier_auxiliary_normalize_pairs", True)):
        pred_pairs = pred.reshape(pred.shape[0], num_frequencies, 2)
        pred = F.normalize(pred_pairs, dim=2).reshape(pred.shape[0], 2 * num_frequencies)
    target = fourier_targets(labels, modulus=model.modulus, max_frequency=max_frequency).to(pred.dtype)
    target = target[:, 2 * (min_frequency - 1) : 2 * max_frequency]
    per_dim = (pred - target).pow(2)
    power = float(train_cfg.get("fourier_auxiliary_frequency_power", 0.0))
    if power != 0.0:
        freq_weights = torch.arange(min_frequency, max_frequency + 1, device=pred.device, dtype=pred.dtype).pow(power)
        freq_weights = freq_weights / freq_weights.mean().clamp_min(1e-8)
        per_dim = per_dim * freq_weights.repeat_interleave(2).unsqueeze(0)
    return weight * per_dim.mean()


def layerwise_dirac_loss(
    model: QNNClassifier,
    labels: torch.Tensor,
    *,
    train_cfg: dict[str, Any],
    current_step: int | None = None,
) -> torch.Tensor:
    weight = scheduled_scalar(train_cfg, "layerwise_dirac_loss_weight", 0.0, current_step)
    if weight == 0.0 or not getattr(model, "_last_layerwise_logits", None):
        first = next(model.parameters())
        return first.new_zeros(())
    losses = [F.cross_entropy(logits, labels) for logits in model._last_layerwise_logits]
    if bool(train_cfg.get("layerwise_dirac_depth_weighted", False)):
        depth_weights = torch.arange(1, len(losses) + 1, device=losses[0].device, dtype=losses[0].dtype)
        depth_weights = depth_weights / depth_weights.mean().clamp_min(1e-8)
        return weight * torch.stack([w * loss for w, loss in zip(depth_weights, losses)]).mean()
    return weight * torch.stack(losses).mean()


def hard_neighbor_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_cfg: dict[str, Any],
    current_step: int | None = None,
) -> torch.Tensor:
    weight = scheduled_scalar(train_cfg, "hard_neighbor_margin_weight", 0.0, current_step)
    if weight == 0.0:
        return logits.new_zeros(())
    margin = scheduled_scalar(train_cfg, "hard_neighbor_margin", 1.0, current_step)
    offsets = train_cfg.get("hard_neighbor_offsets", [1])
    focus_gamma = float(train_cfg.get("hard_neighbor_focus_gamma", 0.0) or 0.0)
    return weight * hard_offset_margin_loss(logits, labels, offsets=offsets, margin=margin, focus_gamma=focus_gamma)


def hard_alias_margin_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    train_cfg: dict[str, Any],
    current_step: int | None = None,
) -> torch.Tensor:
    weight = scheduled_scalar(train_cfg, "hard_alias_margin_weight", 0.0, current_step)
    if weight == 0.0:
        return logits.new_zeros(())
    offsets = train_cfg.get("hard_alias_offsets", [])
    margin = scheduled_scalar(train_cfg, "hard_alias_margin", 1.0, current_step)
    mask = None
    if bool(train_cfg.get("hard_alias_boundary_only", True)):
        mask = batch_boundary_mask(batch, modulus=logits.shape[1], train_cfg=train_cfg, prefix="hard_alias")
    focus_gamma = float(train_cfg.get("hard_alias_focus_gamma", 0.0) or 0.0)
    return weight * hard_offset_margin_loss(logits, batch["labels"], offsets=offsets, margin=margin, mask=mask, focus_gamma=focus_gamma)


def same_sum_consistency_loss(
    model: QNNClassifier,
    batch: dict[str, torch.Tensor],
    logits: torch.Tensor,
    *,
    train_cfg: dict[str, Any],
    current_step: int | None = None,
) -> torch.Tensor:
    weight = scheduled_scalar(train_cfg, "same_sum_consistency_weight", 0.0, current_step)
    ce_weight = scheduled_scalar(train_cfg, "same_sum_consistency_ce_weight", 0.0, current_step)
    if weight == 0.0 and ce_weight == 0.0:
        return logits.new_zeros(())
    offsets = train_cfg.get("same_sum_consistency_offsets", [])
    if isinstance(offsets, int):
        offsets = [offsets]
    offsets = [int(offset) for offset in offsets if int(offset) != 0]
    include_swap = bool(train_cfg.get("same_sum_consistency_include_swap", False))
    if not offsets and not include_swap:
        return logits.new_zeros(())
    mask = None
    if bool(train_cfg.get("same_sum_consistency_boundary_only", False)):
        mask = batch_boundary_mask(batch, modulus=model.modulus, train_cfg=train_cfg, prefix="same_sum_consistency")
        if not bool(mask.any()):
            return logits.new_zeros(())
    a = batch["a"] if mask is None else batch["a"][mask]
    b = batch["b"] if mask is None else batch["b"][mask]
    labels = batch["labels"] if mask is None else batch["labels"][mask]
    base_logits = logits if mask is None else logits[mask]
    temperature = float(train_cfg.get("same_sum_consistency_temperature", 2.0) or 2.0)
    base_logp = F.log_softmax(base_logits / temperature, dim=-1)
    base_p = base_logp.detach().exp()
    losses = []
    for offset in offsets:
        shifted_a = (a + offset) % model.modulus
        shifted_b = (b - offset) % model.modulus
        shifted_logits = model(shifted_a, shifted_b)
        shifted_logp = F.log_softmax(shifted_logits / temperature, dim=-1)
        shifted_p = shifted_logp.detach().exp()
        if weight != 0.0:
            kl_ab = F.kl_div(shifted_logp, base_p, reduction="batchmean")
            kl_ba = F.kl_div(base_logp, shifted_p, reduction="batchmean")
            losses.append(weight * 0.5 * (kl_ab + kl_ba) * (temperature**2))
        if ce_weight != 0.0:
            losses.append(ce_weight * F.cross_entropy(shifted_logits, labels))
    if include_swap:
        swapped_logits = model(b, a)
        swapped_logp = F.log_softmax(swapped_logits / temperature, dim=-1)
        swapped_p = swapped_logp.detach().exp()
        if weight != 0.0:
            kl_ab = F.kl_div(swapped_logp, base_p, reduction="batchmean")
            kl_ba = F.kl_div(base_logp, swapped_p, reduction="batchmean")
            losses.append(weight * 0.5 * (kl_ab + kl_ba) * (temperature**2))
        if ce_weight != 0.0:
            losses.append(ce_weight * F.cross_entropy(swapped_logits, labels))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def fourier_weight_penalty_loss(model: QNNClassifier, *, train_cfg: dict[str, Any]) -> torch.Tensor:
    weight = float(train_cfg.get("fourier_weight_penalty_weight", 0.0))
    if weight == 0.0 or not hasattr(model.head, "frequency_weight_logits"):
        first = next(model.parameters())
        return first.new_zeros(())
    start_frequency = int(train_cfg.get("fourier_weight_penalty_start_frequency", model.modulus + 1))
    freq_weights = F.softplus(model.head.frequency_weight_logits)
    start_idx = max(0, min(freq_weights.shape[0], start_frequency - 1))
    if start_idx >= freq_weights.shape[0]:
        return freq_weights.new_zeros(())
    penalty_type = str(train_cfg.get("fourier_weight_penalty_type", "l2")).lower()
    selected = freq_weights[start_idx:]
    if penalty_type == "l1":
        penalty = selected.abs().mean()
    else:
        penalty = selected.pow(2).mean()
    return weight * penalty


def dirac_coefficient_norm_loss(model: QNNClassifier, features: torch.Tensor, *, train_cfg: dict[str, Any]) -> torch.Tensor:
    weight = float(train_cfg.get("dirac_coefficient_norm_weight", 0.0) or 0.0)
    if weight == 0.0 or not isinstance(model.head, DiracDeltaHead):
        return features.new_zeros(())
    raw = model.head.raw_fourier_features(features).reshape(features.shape[0], model.head.max_frequency, 2)
    norms = raw.norm(dim=-1)
    target = float(train_cfg.get("dirac_coefficient_norm_target", 1.0) or 1.0)
    return weight * (norms - target).pow(2).mean()


def logit_anchor_loss(
    model: QNNClassifier,
    logits: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_cfg: dict[str, Any],
    current_step: int | None = None,
) -> torch.Tensor:
    weight = scheduled_scalar(train_cfg, "logit_anchor_weight", 0.0, current_step)
    if weight == 0.0:
        return logits.new_zeros(())
    base_logits = model.base_logits_from_features(features)
    if base_logits is None:
        return logits.new_zeros(())
    base_logits = base_logits.detach()
    mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
    min_margin = float(train_cfg.get("logit_anchor_min_base_margin", 0.0) or 0.0)
    if min_margin > 0.0:
        top2 = base_logits.topk(k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        pred = base_logits.argmax(dim=-1)
        mask = mask & pred.eq(labels) & (margin >= min_margin)
    if not bool(mask.any()):
        return logits.new_zeros(())
    temperature = float(train_cfg.get("logit_anchor_temperature", 2.0) or 2.0)
    base_prob = F.softmax(base_logits[mask] / temperature, dim=-1)
    current_logp = F.log_softmax(logits[mask] / temperature, dim=-1)
    return weight * F.kl_div(current_logp, base_prob, reduction="batchmean") * (temperature**2)


def correction_nonlocal_loss(
    model: QNNClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_cfg: dict[str, Any],
    current_step: int | None = None,
) -> torch.Tensor:
    weight = scheduled_scalar(train_cfg, "correction_nonlocal_weight", 0.0, current_step)
    if weight == 0.0:
        return features.new_zeros(())
    correction = model.correction_logits_from_features(features)
    if correction is None:
        return features.new_zeros(())
    radius = int(train_cfg.get("correction_local_radius", 2) or 2)
    classes = torch.arange(correction.shape[1], device=correction.device).unsqueeze(0)
    labels_col = labels.unsqueeze(1)
    dist = (classes - labels_col).remainder(correction.shape[1])
    dist = torch.minimum(dist, correction.shape[1] - dist)
    nonlocal_mask = dist > radius
    selected = correction[nonlocal_mask]
    if selected.numel() == 0:
        return correction.new_zeros(())
    if bool(train_cfg.get("correction_nonlocal_positive_only", True)):
        selected = F.relu(selected)
    return weight * selected.pow(2).mean()


def correction_direct_ce_loss(
    model: QNNClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_cfg: dict[str, Any],
    current_step: int | None = None,
) -> torch.Tensor:
    weight = scheduled_scalar(train_cfg, "correction_direct_ce_weight", 0.0, current_step)
    if weight == 0.0:
        return features.new_zeros(())
    correction = model.correction_logits_from_features(features)
    if correction is None:
        return features.new_zeros(())
    if bool(train_cfg.get("correction_direct_ce_boundary_only", False)):
        batch = {"a": labels.new_zeros(labels.shape), "b": labels.new_zeros(labels.shape), "labels": labels}
        mask = batch_boundary_mask(batch, modulus=correction.shape[1], train_cfg=train_cfg, prefix="correction_direct_ce")
        if not bool(mask.any()):
            return features.new_zeros(())
        correction = correction[mask]
        labels = labels[mask]
    return weight * F.cross_entropy(correction, labels)


def scheduled_scalar(train_cfg: dict[str, Any], name: str, default: float, current_step: int | None) -> float:
    if f"{name}_start" not in train_cfg and f"{name}_end" not in train_cfg:
        return float(train_cfg.get(name, default))
    start = float(train_cfg.get(f"{name}_start", train_cfg.get(name, default)))
    end = float(train_cfg.get(f"{name}_end", train_cfg.get(name, default)))
    anneal_steps = int(train_cfg.get(f"{name}_anneal_steps", train_cfg.get("margin_anneal_steps", 1)) or 1)
    start_step = int(train_cfg.get(f"{name}_anneal_start_step", train_cfg.get("margin_anneal_start_step", 0)) or 0)
    step = max(0, int(current_step or 0) - start_step)
    mix = min(1.0, max(0.0, step / max(1, anneal_steps)))
    return start + mix * (end - start)


def batch_boundary_mask(
    batch: dict[str, torch.Tensor],
    *,
    modulus: int,
    train_cfg: dict[str, Any],
    prefix: str,
) -> torch.Tensor:
    a = batch["a"]
    b = batch["b"]
    labels = batch["labels"]
    operand_width = int(train_cfg.get(f"{prefix}_boundary_width", train_cfg.get("boundary_operand_width", 1)) or 1)
    residue_width = int(train_cfg.get(f"{prefix}_residue_width", 0) or 0)
    wrap_width = int(train_cfg.get(f"{prefix}_wrap_width", 0) or 0)
    mask = (a < operand_width) | (b < operand_width) | (a >= modulus - operand_width) | (b >= modulus - operand_width)
    if residue_width > 0:
        mask = mask | (labels < residue_width) | (labels >= modulus - residue_width)
    if wrap_width > 0:
        integer_sum = a + b
        mask = mask | ((integer_sum - modulus).abs() <= wrap_width)
    return mask


def hard_offset_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    offsets: Any,
    margin: float,
    mask: torch.Tensor | None = None,
    focus_gamma: float = 0.0,
) -> torch.Tensor:
    if isinstance(offsets, int):
        offsets = [offsets]
    offsets = [int(offset) for offset in offsets if int(offset) > 0]
    if not offsets:
        return logits.new_zeros(())
    if mask is not None and not bool(mask.any()):
        return logits.new_zeros(())
    idx = torch.arange(labels.shape[0], device=labels.device)
    true_logit = logits[idx, labels]
    losses = []
    for offset in offsets:
        candidates = [logits[idx, (labels + offset) % logits.shape[1]], logits[idx, (labels - offset) % logits.shape[1]]]
        for candidate in candidates:
            hinge = F.relu(margin + candidate - true_logit)
            if mask is not None:
                hinge = hinge[mask]
            if focus_gamma > 0.0:
                denom = max(abs(margin), 1e-6)
                weights = (hinge.detach() / denom).clamp_min(0.0).pow(focus_gamma)
                hinge = hinge * weights
            losses.append(hinge.mean())
    return torch.stack(losses).mean()


def compute_qnn_loss(
    model: QNNClassifier,
    batch: dict[str, torch.Tensor],
    *,
    variant: str,
    train_cfg: dict[str, Any],
    current_step: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, features = model.forward_with_features(batch["a"], batch["b"])
    main_loss = F.nll_loss(logits, batch["labels"]) if variant == "born" else F.cross_entropy(logits, batch["labels"])
    if bool(train_cfg.get("direct_auxiliary_heads", False)):
        aux_loss = auxiliary_head_loss(model.auxiliary_logits(features), batch["labels"], train_cfg=train_cfg)
    else:
        aux_loss = auxiliary_residue_loss(
            logits,
            batch["labels"],
            class_modulus=model.modulus,
            train_cfg=train_cfg,
        )
    fourier_loss = fourier_auxiliary_loss(model, features, batch["labels"], train_cfg=train_cfg)
    layerwise_loss = layerwise_dirac_loss(model, batch["labels"], train_cfg=train_cfg, current_step=current_step)
    neighbor_loss = hard_neighbor_margin_loss(logits, batch["labels"], train_cfg=train_cfg, current_step=current_step)
    alias_loss = hard_alias_margin_loss(logits, batch, train_cfg=train_cfg, current_step=current_step)
    consistency_loss = same_sum_consistency_loss(model, batch, logits, train_cfg=train_cfg, current_step=current_step)
    freq_weight_loss = fourier_weight_penalty_loss(model, train_cfg=train_cfg)
    dirac_norm_loss = dirac_coefficient_norm_loss(model, features, train_cfg=train_cfg)
    anchor_loss = logit_anchor_loss(
        model,
        logits,
        features,
        batch["labels"],
        train_cfg=train_cfg,
        current_step=current_step,
    )
    correction_loss = correction_nonlocal_loss(
        model,
        features,
        batch["labels"],
        train_cfg=train_cfg,
        current_step=current_step,
    )
    correction_direct_loss = correction_direct_ce_loss(
        model,
        features,
        batch["labels"],
        train_cfg=train_cfg,
        current_step=current_step,
    )
    loss = (
        main_loss
        + aux_loss
        + fourier_loss
        + layerwise_loss
        + neighbor_loss
        + alias_loss
        + consistency_loss
        + freq_weight_loss
        + dirac_norm_loss
        + anchor_loss
        + correction_loss
        + correction_direct_loss
    )
    return loss, {
        "logits": logits,
        "main_loss": main_loss,
        "aux_loss": aux_loss,
        "fourier_loss": fourier_loss,
        "layerwise_loss": layerwise_loss,
        "neighbor_loss": neighbor_loss,
        "alias_loss": alias_loss,
        "consistency_loss": consistency_loss,
        "freq_weight_loss": freq_weight_loss,
        "dirac_norm_loss": dirac_norm_loss,
        "anchor_loss": anchor_loss,
        "correction_loss": correction_loss,
        "correction_direct_loss": correction_direct_loss,
    }


def make_train_loader(train_ds: ModularPairDataset, train_cfg: dict[str, Any]) -> DataLoader:
    batch_size = int(train_cfg.get("batch_size", 256))
    boundary_weight = float(train_cfg.get("boundary_oversample_weight", 0.0) or 0.0)
    sampler = None
    shuffle = True
    if boundary_weight > 0.0:
        modulus = train_ds.modulus
        operand_width = int(train_cfg.get("boundary_operand_width", 1))
        residue_width = int(train_cfg.get("boundary_residue_width", 2))
        wrap_width = int(train_cfg.get("wrap_boundary_width", 3))
        weights = []
        for a, b in train_ds.pairs:
            label = (a + b) % modulus
            integer_sum = a + b
            near_operand_edge = a < operand_width or b < operand_width or a >= modulus - operand_width or b >= modulus - operand_width
            near_residue_edge = label < residue_width or label >= modulus - residue_width
            near_wrap = abs(integer_sum - modulus) <= wrap_width
            weights.append(1.0 + boundary_weight * float(near_operand_edge or near_residue_edge or near_wrap))
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=bool(train_cfg.get("drop_last", False)),
    )


def run_readout_refresh(
    model: QNNClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    variant: str,
    train_cfg: dict[str, Any],
    steps: int,
    current_step: int | None = None,
) -> dict[str, float]:
    if steps <= 0 or model.head is None:
        return {}
    old_requires_grad = {param: param.requires_grad for param in model.parameters()}
    for param in model.parameters():
        param.requires_grad_(False)
    head_params = list(model.head.parameters()) + list(model.auxiliary_heads.parameters())
    for param in head_params:
        param.requires_grad_(True)
    opt = torch.optim.AdamW(
        head_params,
        lr=float(train_cfg.get("readout_refresh_lr", train_cfg.get("lr", 0.01))),
        weight_decay=float(train_cfg.get("readout_refresh_weight_decay", 0.0)),
    )
    iterator = iter(loader)
    last: dict[str, float] = {}
    model.train()
    try:
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            loss, parts = compute_qnn_loss(model, batch, variant=variant, train_cfg=train_cfg, current_step=current_step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = train_cfg.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(head_params, float(grad_clip))
            opt.step()
            last = {
                "refresh_loss": float(loss.detach().cpu()),
                "refresh_main_loss": float(parts["main_loss"].detach().cpu()),
                "refresh_aux_loss": float(parts["aux_loss"].detach().cpu()),
                "refresh_fourier_loss": float(parts["fourier_loss"].detach().cpu()),
                "refresh_layerwise_loss": float(parts["layerwise_loss"].detach().cpu()),
                "refresh_neighbor_loss": float(parts["neighbor_loss"].detach().cpu()),
                "refresh_alias_loss": float(parts["alias_loss"].detach().cpu()),
                "refresh_consistency_loss": float(parts["consistency_loss"].detach().cpu()),
                "refresh_dirac_norm_loss": float(parts["dirac_norm_loss"].detach().cpu()),
                "refresh_anchor_loss": float(parts["anchor_loss"].detach().cpu()),
                "refresh_correction_loss": float(parts["correction_loss"].detach().cpu()),
                "refresh_correction_direct_loss": float(parts["correction_direct_loss"].detach().cpu()),
            }
    finally:
        for param, requires_grad in old_requires_grad.items():
            param.requires_grad_(requires_grad)
    return last


def load_matching_state(
    model: QNNClassifier,
    checkpoint_path: Path,
    *,
    device: torch.device,
    skip_prefixes: tuple[str, ...] = ("head.",),
    allow_partial_tensors: bool = False,
    partial_prefixes: tuple[str, ...] = ("head.",),
) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = checkpoint["model"]
    target = model.state_dict()
    copied: dict[str, torch.Tensor] = {}
    partial_copied: dict[str, torch.Tensor] = {}
    skipped = 0
    partial_elements = 0
    for key, value in source.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped += 1
            continue
        if key in target and tuple(target[key].shape) == tuple(value.shape):
            copied[key] = value
        elif (
            allow_partial_tensors
            and key in target
            and any(key.startswith(prefix) for prefix in partial_prefixes)
            and target[key].ndim == value.ndim
            and target[key].numel() > 0
            and value.numel() > 0
        ):
            updated = target[key].clone()
            slices = tuple(slice(0, min(target_dim, source_dim)) for target_dim, source_dim in zip(target[key].shape, value.shape))
            updated[slices] = value[slices].to(device=updated.device, dtype=updated.dtype)
            partial_copied[key] = updated
            partial_elements += int(updated[slices].numel())
        else:
            skipped += 1
    target.update(copied)
    target.update(partial_copied)
    model.load_state_dict(target)
    return {
        "copied": len(copied),
        "partial_copied": len(partial_copied),
        "partial_elements": partial_elements,
        "skipped": skipped,
        "source_step": int(checkpoint.get("step", -1)),
    }


def string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def apply_parameter_trainability(model: nn.Module, train_cfg: dict[str, Any]) -> dict[str, int]:
    trainable_prefixes = string_tuple(train_cfg.get("trainable_parameter_prefixes"), ())
    freeze_prefixes = string_tuple(train_cfg.get("freeze_parameter_prefixes"), ())
    if trainable_prefixes:
        for name, param in model.named_parameters():
            param.requires_grad_(any(name.startswith(prefix) for prefix in trainable_prefixes))
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in freeze_prefixes):
            param.requires_grad_(False)
    total = 0
    trainable = 0
    frozen = 0
    for param in model.parameters():
        total += int(param.numel())
        if param.requires_grad:
            trainable += int(param.numel())
        else:
            frozen += int(param.numel())
    return {"parameter_count": total, "trainable_parameter_count": trainable, "frozen_parameter_count": frozen}


def metric_improved(value: float, best: float, *, mode: str, min_delta: float) -> bool:
    if mode == "max":
        return value > best + min_delta
    if mode == "min":
        return value < best - min_delta
    raise ValueError(f"unsupported early_stopping_mode: {mode}")


def train_variant(
    config: dict[str, Any],
    *,
    variant: str,
    out_dir: Path,
    resume_run_dir: Path | None = None,
    resume_kind: str = "final",
    append_metrics: bool = False,
) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    data_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    train_ds = ModularPairDataset(
        modulus=int(data_cfg.get("modulus", 97)),
        train_fraction=float(data_cfg.get("train_fraction", 0.3)),
        seed=int(data_cfg.get("seed", 0)),
        split="train",
    )
    test_ds = ModularPairDataset(
        modulus=int(data_cfg.get("modulus", 97)),
        train_fraction=float(data_cfg.get("train_fraction", 0.3)),
        seed=int(data_cfg.get("seed", 0)),
        split="test",
    )
    model = QNNClassifier(
        variant=variant,
        n_qubits=int(model_cfg.get("n_qubits", 7)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        modulus=int(data_cfg.get("modulus", 97)),
        input_frequencies=model_cfg.get("input_frequencies"),
        auxiliary_head_moduli=model_cfg.get("auxiliary_head_moduli"),
        readout_type=str(model_cfg.get("readout_type", "linear")),
        fourier_max_frequency=model_cfg.get("fourier_max_frequency"),
        fourier_kernel_init=str(model_cfg.get("fourier_kernel_init", "fejer")),
        fourier_residual_linear=bool(model_cfg.get("fourier_residual_linear", False)),
        fourier_residual_scale=float(model_cfg.get("fourier_residual_scale", 1.0)),
        fourier_kernel_trainable=bool(model_cfg.get("fourier_kernel_trainable", True)),
        dirac_coefficient_mode=str(model_cfg.get("dirac_coefficient_mode", "unit")),
        dirac_coefficient_eps=float(model_cfg.get("dirac_coefficient_eps", 1e-6)),
        dirac_kernel_trainable=bool(model_cfg.get("dirac_kernel_trainable", False)),
        dirac_sharpen_kernel_init=str(model_cfg.get("dirac_sharpen_kernel_init", "dirichlet")),
        dirac_sharpen_kernel_trainable=bool(model_cfg.get("dirac_sharpen_kernel_trainable", True)),
        dirac_sharpen_strength_init=float(model_cfg.get("dirac_sharpen_strength_init", 0.0)),
        dirac_sharpen_strength_max=float(model_cfg.get("dirac_sharpen_strength_max", 0.25)),
        dirac_primary_base_scale_init=float(model_cfg.get("dirac_primary_base_scale_init", 0.2)),
        dirac_primary_base_scale_trainable=bool(model_cfg.get("dirac_primary_base_scale_trainable", True)),
        layerwise_dirac_adapter_scale=float(model_cfg.get("layerwise_dirac_adapter_scale", 0.1)),
    ).to(device)
    init_info: dict[str, Any] | None = None
    initialize_from = model_cfg.get("initialize_circuit_from")
    if initialize_from:
        init_path = Path(str(initialize_from))
        init_info = {
            "path": str(init_path),
            **load_matching_state(
                model,
                init_path,
                device=device,
                skip_prefixes=string_tuple(model_cfg.get("initialize_skip_prefixes"), ("head.",)),
                allow_partial_tensors=bool(model_cfg.get("initialize_partial_tensors", False)),
                partial_prefixes=string_tuple(model_cfg.get("initialize_partial_prefixes"), ("head.",)),
            ),
        }
    if hasattr(model.head, "set_sharpen_strength_override") and "dirac_sharpen_strength_override" in model_cfg:
        model.head.set_sharpen_strength_override(float(model_cfg["dirac_sharpen_strength_override"]))
    if hasattr(model.head, "set_base_scale_override") and "dirac_primary_base_scale_override" in model_cfg:
        model.head.set_base_scale_override(float(model_cfg["dirac_primary_base_scale_override"]))
    batch_size = int(train_cfg.get("batch_size", 256))
    loader = make_train_loader(train_ds, train_cfg)
    train_eval_loader = DataLoader(train_ds, batch_size=int(train_cfg.get("eval_batch_size", batch_size)), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(train_cfg.get("eval_batch_size", batch_size)), shuffle=False)
    steps = int(train_cfg.get("steps", 1000))
    eval_every = int(train_cfg.get("eval_every", 100))
    max_eval_batches = train_cfg.get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    grad_clip = train_cfg.get("grad_clip")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    metrics_path = out_dir / f"metrics_{variant}.jsonl"
    if metrics_path.exists() and not append_metrics:
        metrics_path.unlink()
    start_step = 0
    resume_path: Path | None = None
    if resume_run_dir is not None:
        candidate = resume_run_dir / f"checkpoint_{variant}_{resume_kind}.pt"
        if candidate.exists():
            checkpoint = torch.load(candidate, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            start_step = int(checkpoint.get("step", 0))
            resume_path = candidate
        else:
            raise FileNotFoundError(candidate)
    param_info = apply_parameter_trainability(model, train_cfg)
    opt_params = [param for param in model.parameters() if param.requires_grad]
    if not opt_params:
        raise ValueError("no trainable parameters remain after applying trainable/freeze prefixes")
    opt = torch.optim.AdamW(opt_params, lr=float(train_cfg.get("lr", 0.01)), weight_decay=float(train_cfg.get("weight_decay", 0.0)))
    iterator = iter(loader)
    previous_summary_path = (resume_run_dir or out_dir) / "summary.json"
    best_acc, best_record = load_previous_best(previous_summary_path, variant)
    best_checkpoint_path = out_dir / f"checkpoint_{variant}_best.pt"
    if resume_path is not None and not best_checkpoint_path.exists():
        torch.save(
            {
                "model": model.state_dict(),
                "config": config,
                "variant": variant,
                "step": start_step,
                "resumed_from": str(resume_path),
            },
            best_checkpoint_path,
        )
    start = time.time()
    if start_step >= steps:
        train_metrics = evaluate(model, train_eval_loader, device, max_batches=max_eval_batches)
        test_metrics = evaluate(model, test_loader, device, max_batches=max_eval_batches)
        return {
            "variant": variant,
            "best_test_accuracy": max(best_acc, test_metrics["accuracy"]),
            "best_record": best_record,
            "final_train_accuracy": train_metrics["accuracy"],
            "final_test_accuracy": test_metrics["accuracy"],
            **param_info,
            "checkpoint_best": str(out_dir / f"checkpoint_{variant}_best.pt"),
            "checkpoint_final": str(out_dir / f"checkpoint_{variant}_final.pt"),
            "metrics": str(metrics_path),
            "resumed_from": str(resume_path) if resume_path is not None else None,
            "initialized_from": init_info,
            "start_step": start_step,
        }
    initial_refresh_steps = int(train_cfg.get("initial_readout_refresh_steps", 0) or 0)
    if start_step == 0 and initial_refresh_steps > 0:
        run_readout_refresh(
            model,
            loader,
            device=device,
            variant=variant,
            train_cfg=train_cfg,
            steps=initial_refresh_steps,
            current_step=start_step,
        )
    progress = tqdm(range(start_step + 1, steps + 1), desc=f"qnn_{variant}", disable=bool(train_cfg.get("disable_progress", False)))
    early_metric = str(train_cfg.get("early_stopping_metric", "test_accuracy"))
    early_mode = str(train_cfg.get("early_stopping_mode", "max"))
    early_patience = int(train_cfg.get("early_stopping_patience_evals", 0) or 0)
    early_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0) or 0.0)
    early_min_step = int(train_cfg.get("early_stopping_min_step", 0) or 0)
    early_best = float("inf") if early_mode == "min" else -float("inf")
    early_best_step: int | None = None
    if isinstance(best_record, dict) and early_metric in best_record:
        early_best = float(best_record[early_metric])
        early_best_step = int(best_record.get("step", start_step))
    early_bad_evals = 0
    stopped_early = False
    early_stop_reason: str | None = None
    completed_step = start_step
    for step in progress:
        completed_step = step
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        batch = move_batch(batch, device)
        loss, parts = compute_qnn_loss(model, batch, variant=variant, train_cfg=train_cfg, current_step=step)
        logits = parts["logits"]
        main_loss = parts["main_loss"]
        aux_loss = parts["aux_loss"]
        fourier_loss = parts["fourier_loss"]
        neighbor_loss = parts["neighbor_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(opt_params, grad_clip)
        opt.step()
        refresh_record: dict[str, float] = {}
        refresh_every = int(train_cfg.get("readout_refresh_every", 0) or 0)
        refresh_steps = int(train_cfg.get("readout_refresh_steps", 0) or 0)
        if refresh_every > 0 and refresh_steps > 0 and step % refresh_every == 0:
            refresh_record = run_readout_refresh(
                model,
                loader,
                device=device,
                variant=variant,
                train_cfg=train_cfg,
                steps=refresh_steps,
                current_step=step,
            )
        if step == 1 or step % eval_every == 0 or step == steps:
            train_pred = logits.argmax(dim=-1)
            train_acc = float(train_pred.eq(batch["labels"]).float().mean().detach().cpu())
            train_full_metrics = evaluate(model, train_eval_loader, device, max_batches=max_eval_batches)
            test_metrics = evaluate(model, test_loader, device, max_batches=max_eval_batches)
            record = {
                "variant": variant,
                "step": step,
                "elapsed_sec": time.time() - start,
                "train_loss": float(loss.detach().cpu()),
                "train_main_loss": float(main_loss.detach().cpu()),
                "train_aux_loss": float(aux_loss.detach().cpu()),
                "train_fourier_loss": float(fourier_loss.detach().cpu()),
                "train_layerwise_loss": float(parts["layerwise_loss"].detach().cpu()),
                "train_neighbor_loss": float(neighbor_loss.detach().cpu()),
                "train_alias_loss": float(parts["alias_loss"].detach().cpu()),
                "train_consistency_loss": float(parts["consistency_loss"].detach().cpu()),
                "train_freq_weight_loss": float(parts["freq_weight_loss"].detach().cpu()),
                "train_dirac_norm_loss": float(parts["dirac_norm_loss"].detach().cpu()),
                "train_anchor_loss": float(parts["anchor_loss"].detach().cpu()),
                "train_correction_loss": float(parts["correction_loss"].detach().cpu()),
                "train_correction_direct_loss": float(parts["correction_direct_loss"].detach().cpu()),
                "train_batch_accuracy": train_acc,
                **refresh_record,
                **{f"train_full_{key}": value for key, value in train_full_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_record = record
                torch.save({"model": model.state_dict(), "config": config, "variant": variant, "step": step}, out_dir / f"checkpoint_{variant}_best.pt")
            if early_patience > 0:
                if early_metric not in record:
                    raise ValueError(f"early stopping metric {early_metric!r} was not found in eval record")
                metric_value = float(record[early_metric])
                if metric_improved(metric_value, early_best, mode=early_mode, min_delta=early_min_delta):
                    early_best = metric_value
                    early_best_step = step
                    early_bad_evals = 0
                elif step >= early_min_step:
                    early_bad_evals += 1
                if step >= early_min_step and early_bad_evals >= early_patience:
                    stopped_early = True
                    early_stop_reason = (
                        f"{early_metric} did not improve for {early_bad_evals} evals "
                        f"after step {early_best_step}; best={early_best:.6g}"
                    )
                    break
            progress.set_postfix({"loss": f"{record['train_loss']:.3f}", "test_acc": f"{test_metrics['accuracy']:.3f}"})
    final_refresh_steps = int(train_cfg.get("final_readout_refresh_steps", 0) or 0)
    if stopped_early and not bool(train_cfg.get("run_final_refresh_on_early_stop", False)):
        final_refresh_steps = 0
    if final_refresh_steps > 0:
        run_readout_refresh(
            model,
            loader,
            device=device,
            variant=variant,
            train_cfg=train_cfg,
            steps=final_refresh_steps,
            current_step=completed_step,
        )
    restored_best = False
    restored_best_step: int | None = None
    if bool(train_cfg.get("early_stopping_restore_best", False)) and best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        restored_best = True
        restored_best_step = int(best_checkpoint.get("step", completed_step))
    final_checkpoint_step = restored_best_step if restored_best_step is not None else completed_step
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "variant": variant,
            "step": final_checkpoint_step,
            "completed_step": completed_step,
            "stopped_early": stopped_early,
            "early_stop_reason": early_stop_reason,
            "restored_best": restored_best,
        },
        out_dir / f"checkpoint_{variant}_final.pt",
    )
    train_metrics = evaluate(model, train_eval_loader, device, max_batches=max_eval_batches)
    test_metrics = evaluate(model, test_loader, device, max_batches=max_eval_batches)
    if test_metrics["accuracy"] > best_acc:
        best_acc = test_metrics["accuracy"]
        best_record = {
            "variant": variant,
            "step": final_checkpoint_step,
            "completed_step": completed_step,
            "elapsed_sec": time.time() - start,
            **{f"train_full_{key}": value for key, value in train_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "final_readout_refresh": bool(final_refresh_steps > 0),
            "stopped_early": stopped_early,
            "restored_best": restored_best,
        }
        torch.save({"model": model.state_dict(), "config": config, "variant": variant, "step": final_checkpoint_step}, out_dir / f"checkpoint_{variant}_best.pt")
    return {
        "variant": variant,
        "best_test_accuracy": best_acc,
        "best_record": best_record,
        "final_train_accuracy": train_metrics["accuracy"],
        "final_test_accuracy": test_metrics["accuracy"],
        **param_info,
        "checkpoint_best": str(out_dir / f"checkpoint_{variant}_best.pt"),
        "checkpoint_final": str(out_dir / f"checkpoint_{variant}_final.pt"),
        "metrics": str(metrics_path),
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "initialized_from": init_info,
        "start_step": start_step,
        "completed_step": completed_step,
        "stopped_early": stopped_early,
        "early_stop_reason": early_stop_reason,
        "early_stop_best_metric": early_best if early_patience > 0 else None,
        "early_stop_best_step": early_best_step,
        "restored_best": restored_best,
    }


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "seed": 9201,
            "device": "cuda",
            "output_dir": "runs/modular_addition_qnn_mod97",
            "dataset": {"modulus": 97, "train_fraction": 0.3, "seed": 0},
            "model": {"n_qubits": 7, "n_layers": 4},
            "training": {
                "batch_size": 256,
                "eval_batch_size": 512,
                "steps": 1000,
                "lr": 0.01,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "eval_every": 100,
                "max_eval_batches": None,
                "disable_progress": False,
            },
        }
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_report(out_dir: Path, config: dict[str, Any], summary: dict[str, Any]) -> None:
    data_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    modulus = int(data_cfg.get("modulus", 97))
    n_qubits = int(model_cfg.get("n_qubits", 7))
    dim = 2**n_qubits
    lines = [
        "# Quantum Neural Networks On Modular Addition",
        "",
        f"This experiment trains simulated quantum neural networks on the numeric `(a+b) mod {modulus}` split.",
        "",
        f"The implemented circuit is a data-reuploading variational quantum classifier: inputs are encoded as repeated rotations on a {n_qubits}-qubit statevector, entangling CNOT rings mix the register, and labels are read from either Born probabilities or quantum-state features.",
        "",
        "## Architecture",
        "",
        f"- modulus: `{modulus}`",
        f"- qubits: `{n_qubits}`",
        f"- Hilbert dimension: `{dim}`",
        f"- circuit layers: `{int(model_cfg.get('n_layers', 4))}`",
        f"- readout type: `{model_cfg.get('readout_type', 'linear')}`",
        f"- input frequencies: `{model_cfg.get('input_frequencies', 'default low-frequency angles')}`",
        f"- Fourier max frequency: `{model_cfg.get('fourier_max_frequency', 'n/a')}`",
        f"- initialized from: `{model_cfg.get('initialize_circuit_from', 'scratch')}`",
        f"- direct auxiliary head moduli: `{model_cfg.get('auxiliary_head_moduli', [])}`",
        f"- auxiliary residue losses: `{config.get('training', {}).get('auxiliary_residue_losses', [])}`",
        "",
        "## Variants",
        "",
        "| variant | readout | quantum dependence |",
        "| --- | --- | --- |",
        f"| `born` | normalized probabilities of computational basis states 0..{modulus - 1} | strictest; no learned classical head after measurement. |",
        f"| `prob_head` | linear head over all {dim} basis-state probabilities | quantum feature map plus classical linear readout. |",
        "| `expval_head` | linear head over single-qubit Z and nearest-neighbor ZZ expectations | compact expectation-value readout. |",
        "",
        "## Results",
        "",
        "| variant | params | start step | best held-out accuracy | final train accuracy | final held-out accuracy | best step |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["variants"]:
        best = row.get("best_record", {})
        lines.append(
            f"| `{row['variant']}` | {row['parameter_count']} | {row.get('start_step', 0)} | "
            f"{row['best_test_accuracy']:.6f} | {row.get('final_train_accuracy', 0.0):.6f} | "
            f"{row.get('final_test_accuracy', 0.0):.6f} | {best.get('step', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These are near-term-style simulated QNN baselines, not claims of quantum advantage. The useful comparison for this branch is whether compact quantum-state representations naturally discover the cyclic rule under the same train/test split.",
            "",
            "A strict Born classifier is the most quantum-native readout. Hybrid probability or expectation heads test whether a quantum feature representation is usable once a small classical readout is allowed.",
            "",
            "## Artifacts",
            "",
            f"- Run directory: `{out_dir}`",
            f"- Config: `{out_dir / 'config.yaml'}`",
            f"- Summary: `{out_dir / 'summary.json'}`",
        ]
    )
    report_text = "\n".join(lines) + "\n"
    (out_dir / f"QNN_MOD{modulus}_REPORT.md").write_text(report_text, encoding="utf-8")
    generic_path = Path(f"RESULTS_QNN_MOD{modulus}.md")
    safe_run_name = re.sub(r"[^A-Za-z0-9]+", "_", out_dir.name).strip("_").upper()
    run_specific_path = Path(f"RESULTS_QNN_MOD{modulus}_{safe_run_name}.md")
    run_specific_path.write_text(report_text, encoding="utf-8")
    if not generic_path.exists():
        generic_path.write_text(report_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--variants", default="born,prob_head,expval_head")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--resume-run-dir", default=None)
    parser.add_argument("--resume-kind", default="final", choices=["final", "best"])
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.out_dir is not None:
        config["output_dir"] = args.out_dir
    if args.device is not None:
        config["device"] = args.device
    if args.steps is not None:
        config.setdefault("training", {})["steps"] = args.steps
    if args.eval_every is not None:
        config.setdefault("training", {})["eval_every"] = args.eval_every
    if args.disable_progress:
        config.setdefault("training", {})["disable_progress"] = True
    out_dir = Path(config.get("output_dir", "runs/modular_addition_qnn_mod97"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    summary = {
        "config": config,
        "variants": [
            train_variant(
                config,
                variant=variant,
                out_dir=out_dir,
                resume_run_dir=Path(args.resume_run_dir) if args.resume_run_dir else None,
                resume_kind=args.resume_kind,
                append_metrics=args.append_metrics,
            )
            for variant in variants
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir, config, summary)
    print(out_dir / "summary.json")


if __name__ == "__main__":
    main()
