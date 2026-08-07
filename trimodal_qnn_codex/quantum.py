from __future__ import annotations

import itertools
import math

import torch
import torch.nn as nn


def apply_single_qubit_gate(state: torch.Tensor, gate: torch.Tensor, *, qubit: int, n_qubits: int) -> torch.Tensor:
    """Apply example-specific single-qubit gates to a batched statevector."""

    batch = state.shape[0]
    left = 2**qubit
    right = 2 ** (n_qubits - qubit - 1)
    x = state.reshape(batch, left, 2, right)
    v0 = x[:, :, 0, :]
    v1 = x[:, :, 1, :]
    g00 = gate[:, 0, 0].reshape(batch, 1, 1)
    g01 = gate[:, 0, 1].reshape(batch, 1, 1)
    g10 = gate[:, 1, 0].reshape(batch, 1, 1)
    g11 = gate[:, 1, 1].reshape(batch, 1, 1)
    out0 = g00 * v0 + g01 * v1
    out1 = g10 * v0 + g11 * v1
    out = torch.stack([out0, out1], dim=2)
    return out.reshape(batch, 2**n_qubits)


def rotation_gates(theta: torch.Tensor, axis: str) -> torch.Tensor:
    theta = theta.to(torch.float32)
    half = theta / 2.0
    cos = torch.cos(half).to(torch.complex64)
    sin = torch.sin(half).to(torch.complex64)
    batch = theta.shape[0]
    gate = torch.zeros(batch, 2, 2, dtype=torch.complex64, device=theta.device)
    if axis == "x":
        gate[:, 0, 0] = cos
        gate[:, 1, 1] = cos
        gate[:, 0, 1] = -1j * sin
        gate[:, 1, 0] = -1j * sin
    elif axis == "y":
        gate[:, 0, 0] = cos
        gate[:, 1, 1] = cos
        gate[:, 0, 1] = -sin
        gate[:, 1, 0] = sin
    elif axis == "z":
        gate[:, 0, 0] = torch.exp(-0.5j * theta.to(torch.complex64))
        gate[:, 1, 1] = torch.exp(0.5j * theta.to(torch.complex64))
    else:
        raise ValueError(f"unknown axis {axis!r}")
    return gate


def cnot_permutation(n_qubits: int, control: int, target: int, device: torch.device) -> torch.Tensor:
    perm = []
    for idx in range(2**n_qubits):
        bits = [(idx >> bit) & 1 for bit in reversed(range(n_qubits))]
        if bits[control] == 1:
            bits[target] ^= 1
        out = 0
        for bit in bits:
            out = (out << 1) | bit
        perm.append(out)
    return torch.tensor(perm, dtype=torch.long, device=device)


class FeatureToState(nn.Module):
    def __init__(self, feature_dim: int, state_dim: int):
        super().__init__()
        self.proj = nn.Linear(feature_dim, 2 * state_dim)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.state_dim = int(state_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.proj(features)
        real, imag = raw.chunk(2, dim=-1)
        state = torch.complex(real, imag)
        norm = state.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return state / norm


class SectorMixingUnitary(nn.Module):
    def __init__(self, sector_count: int, init_scale: float = 0.02):
        super().__init__()
        self.sector_count = int(sector_count)
        real = torch.randn(sector_count, sector_count) * init_scale
        imag = torch.randn(sector_count, sector_count) * init_scale
        self.raw_real = nn.Parameter(real)
        self.raw_imag = nn.Parameter(imag)

    def hermitian(self) -> torch.Tensor:
        raw = torch.complex(self.raw_real, self.raw_imag)
        return 0.5 * (raw + raw.conj().T)

    def unitary(self) -> torch.Tensor:
        h = self.hermitian()
        return torch.linalg.matrix_exp((-1j * h).to(torch.complex64))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        u = self.unitary()
        return torch.einsum("st,btd->bsd", u, state)


class FactorizedRouteMixingUnitary(nn.Module):
    """Route mixer with operand-position factors plus optional low-rank interactions."""

    def __init__(
        self,
        *,
        modality_count: int = 3,
        interaction_rank: int = 0,
        init_scale: float = 0.02,
        interaction_scale: float = 0.02,
    ):
        super().__init__()
        self.modality_count = int(modality_count)
        self.sector_count = self.modality_count * self.modality_count
        self.interaction_rank = int(interaction_rank)
        self.interaction_scale = float(interaction_scale)
        self.left = SectorMixingUnitary(self.modality_count, init_scale=init_scale)
        self.right = SectorMixingUnitary(self.modality_count, init_scale=init_scale)
        if self.interaction_rank > 0:
            real = torch.randn(self.interaction_rank, self.sector_count) * init_scale
            imag = torch.randn(self.interaction_rank, self.sector_count) * init_scale
            self.interaction_real = nn.Parameter(real)
            self.interaction_imag = nn.Parameter(imag)
        else:
            self.register_parameter("interaction_real", None)
            self.register_parameter("interaction_imag", None)

    def hermitian(self) -> torch.Tensor:
        eye = torch.eye(self.modality_count, dtype=torch.complex64, device=self.left.raw_real.device)
        left_h = self.left.hermitian()
        right_h = self.right.hermitian()
        h = torch.kron(left_h, eye) + torch.kron(eye, right_h)
        if self.interaction_rank > 0:
            vectors = torch.complex(self.interaction_real, self.interaction_imag).to(torch.complex64)
            interaction = vectors.conj().T @ vectors
            h = h + (self.interaction_scale / math.sqrt(float(self.interaction_rank))) * interaction
        return 0.5 * (h + h.conj().T)

    def unitary(self) -> torch.Tensor:
        h = self.hermitian()
        return torch.linalg.matrix_exp((-1j * h).to(torch.complex64))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        u = self.unitary()
        return torch.einsum("st,btd->bsd", u, state)


class HybridFactorizedDenseRouteMixingUnitary(nn.Module):
    """Factorized route dynamics plus a small dense Hermitian residual."""

    def __init__(
        self,
        *,
        modality_count: int = 3,
        interaction_rank: int = 0,
        init_scale: float = 0.02,
        interaction_scale: float = 0.02,
        dense_residual_scale: float = 0.02,
    ):
        super().__init__()
        self.modality_count = int(modality_count)
        self.sector_count = self.modality_count * self.modality_count
        self.dense_residual_scale = float(dense_residual_scale)
        self.factorized = FactorizedRouteMixingUnitary(
            modality_count=self.modality_count,
            interaction_rank=int(interaction_rank),
            init_scale=init_scale,
            interaction_scale=float(interaction_scale),
        )
        self.dense_residual = SectorMixingUnitary(self.sector_count, init_scale=init_scale)

    def hermitian(self) -> torch.Tensor:
        h = self.factorized.hermitian()
        if self.dense_residual_scale:
            h = h + self.dense_residual_scale * self.dense_residual.hermitian()
        return 0.5 * (h + h.conj().T)

    def unitary(self) -> torch.Tensor:
        h = self.hermitian()
        return torch.linalg.matrix_exp((-1j * h).to(torch.complex64))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        u = self.unitary()
        return torch.einsum("st,btd->bsd", u, state)


class DataReuploadingLayer(nn.Module):
    def __init__(
        self,
        *,
        sector_count: int,
        feature_dim: int,
        n_qubits: int,
        cross_mixing: bool = True,
        entangle: bool = True,
        route_mixer_type: str = "dense",
        route_interaction_rank: int = 0,
        route_interaction_scale: float = 0.02,
        route_dense_residual_scale: float = 0.02,
    ):
        super().__init__()
        self.sector_count = int(sector_count)
        self.n_qubits = int(n_qubits)
        self.state_dim = 2**self.n_qubits
        self.angle_proj = nn.Linear(feature_dim, self.n_qubits * 3)
        self.cross_mixing = bool(cross_mixing)
        self.entangle = bool(entangle)
        self.route_mixer_type = str(route_mixer_type)
        if not self.cross_mixing:
            self.sector_mixer = None
        elif self.route_mixer_type == "dense":
            self.sector_mixer = SectorMixingUnitary(sector_count)
        elif self.route_mixer_type == "factorized":
            if self.sector_count != 9:
                raise ValueError("factorized route mixing expects 9 ordered-route sectors")
            self.sector_mixer = FactorizedRouteMixingUnitary(
                interaction_rank=int(route_interaction_rank),
                interaction_scale=float(route_interaction_scale),
            )
        elif self.route_mixer_type == "hybrid_factorized_dense":
            if self.sector_count != 9:
                raise ValueError("hybrid factorized+dense route mixing expects 9 ordered-route sectors")
            self.sector_mixer = HybridFactorizedDenseRouteMixingUnitary(
                interaction_rank=int(route_interaction_rank),
                interaction_scale=float(route_interaction_scale),
                dense_residual_scale=float(route_dense_residual_scale),
            )
        else:
            raise ValueError("route_mixer_type must be one of: dense, factorized, hybrid_factorized_dense")

    def apply_rotations(self, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch, sectors, dim = state.shape
        angles = self.angle_proj(features).reshape(batch * sectors, self.n_qubits, 3)
        flat = state.reshape(batch * sectors, dim)
        for q in range(self.n_qubits):
            for axis_idx, axis in enumerate(("z", "y", "x")):
                gate = rotation_gates(angles[:, q, axis_idx], axis)
                flat = apply_single_qubit_gate(flat, gate, qubit=q, n_qubits=self.n_qubits)
        return flat.reshape(batch, sectors, dim)

    def apply_entanglement(self, state: torch.Tensor) -> torch.Tensor:
        if not self.entangle or self.n_qubits < 2:
            return state
        flat = state.reshape(state.shape[0] * state.shape[1], state.shape[2])
        for control, target in itertools.pairwise(range(self.n_qubits)):
            perm = cnot_permutation(self.n_qubits, control, target, flat.device)
            flat = flat[:, perm]
        perm = cnot_permutation(self.n_qubits, self.n_qubits - 1, 0, flat.device)
        flat = flat[:, perm]
        return flat.reshape_as(state)

    def forward(self, state: torch.Tensor, features: torch.Tensor, *, ablate_cross: bool = False) -> torch.Tensor:
        state = self.apply_rotations(state, features)
        state = self.apply_entanglement(state)
        if self.sector_mixer is not None and not ablate_cross:
            state = self.sector_mixer(state)
        return state


def state_norm(state: torch.Tensor) -> torch.Tensor:
    return state.abs().pow(2).sum(dim=(-1, -2))
