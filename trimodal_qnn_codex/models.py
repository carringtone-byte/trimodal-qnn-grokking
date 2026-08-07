from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .amplitudes import AmplitudeCoefficients
from .data import MODALITIES, ORDERED_ROUTES
from .heads import DiracDeltaHead, FourierDeltaHead
from .quantum import DataReuploadingLayer, FeatureToState, state_norm
from .render import build_vocab, render_pair_image, render_value_image, tokenize_pair, tokenize_value, tokens_to_ids


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int, channels: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear(channels * 8, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrimodalEncoders(nn.Module):
    def __init__(self, *, modulus: int, feature_dim: int, operand_feature_dim: int, text_embed_dim: int, image_channels: int):
        super().__init__()
        self.modulus = int(modulus)
        self.feature_dim = int(feature_dim)
        self.operand_feature_dim = int(operand_feature_dim)
        self.vocab = build_vocab(self.modulus)

        max_value_len = max(len(tokenize_value(v)) for v in range(self.modulus))
        max_pair_len = max(len(tokenize_pair(a, b)) for a in range(self.modulus) for b in range(self.modulus))
        value_tokens = [tokens_to_ids(tokenize_value(v), self.vocab, max_value_len) for v in range(self.modulus)]
        pair_tokens = [
            tokens_to_ids(tokenize_pair(a, b), self.vocab, max_pair_len)
            for a in range(self.modulus)
            for b in range(self.modulus)
        ]
        value_images = [render_value_image(v) for v in range(self.modulus)]
        pair_images = [render_pair_image(a, b) for a in range(self.modulus) for b in range(self.modulus)]

        self.register_buffer("value_tokens", torch.tensor(value_tokens, dtype=torch.long), persistent=False)
        self.register_buffer("pair_tokens", torch.tensor(pair_tokens, dtype=torch.long), persistent=False)
        self.register_buffer("value_images", torch.tensor(np.stack(value_images), dtype=torch.float32), persistent=False)
        self.register_buffer("pair_images", torch.tensor(np.stack(pair_images), dtype=torch.float32), persistent=False)

        self.number_value = nn.Embedding(self.modulus, operand_feature_dim)
        self.number_pair = nn.Sequential(nn.Linear(2 * operand_feature_dim + 4, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.text_embedding = nn.Embedding(len(self.vocab), text_embed_dim, padding_idx=self.vocab["<pad>"])
        self.text_value = nn.Sequential(nn.Linear(text_embed_dim, operand_feature_dim), nn.GELU(), nn.LayerNorm(operand_feature_dim))
        self.text_pair = nn.Sequential(nn.Linear(text_embed_dim, feature_dim), nn.GELU(), nn.LayerNorm(feature_dim))
        self.image_value = ImageEncoder(operand_feature_dim, channels=image_channels)
        self.image_pair = ImageEncoder(feature_dim, channels=image_channels)
        self.route_embedding = nn.Embedding(len(ORDERED_ROUTES), operand_feature_dim)
        self.ordered_pair = nn.Sequential(
            nn.Linear(3 * operand_feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.value_to_feature = nn.Sequential(
            nn.Linear(operand_feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
        )
        self.operand_modality = nn.Embedding(len(MODALITIES), feature_dim)
        self.operand_position = nn.Embedding(3, feature_dim)
        self.answer_query = nn.Parameter(torch.zeros(feature_dim))
        nn.init.normal_(self.answer_query, std=0.02)

    def _pair_index(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * self.modulus + b

    def _mean_text(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.text_embedding(ids)
        mask = ids.ne(self.vocab["<pad>"]).to(emb.dtype)
        return (emb * mask[..., None]).sum(dim=-2) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def _number_angles(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        theta_a = 2.0 * torch.pi * a.to(torch.float32) / float(self.modulus)
        theta_b = 2.0 * torch.pi * b.to(torch.float32) / float(self.modulus)
        return torch.stack([torch.cos(theta_a), torch.sin(theta_a), torch.cos(theta_b), torch.sin(theta_b)], dim=-1)

    def pair_features(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        idx = self._pair_index(a, b)
        num = self.number_pair(torch.cat([self.number_value(a), self.number_value(b), self._number_angles(a, b)], dim=-1))
        text = self.text_pair(self._mean_text(self.pair_tokens[idx].to(a.device)))
        image = self.image_pair(self.pair_images[idx].to(a.device))
        return torch.stack([text, num, image], dim=1)

    def value_feature(self, values: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "N":
            return self.number_value(values)
        if modality == "T":
            return self.text_value(self._mean_text(self.value_tokens[values].to(values.device)))
        if modality == "I":
            return self.image_value(self.value_images[values].to(values.device))
        raise ValueError(f"unknown modality {modality!r}")

    def ordered_route_features(self, a: torch.Tensor, b: torch.Tensor, route_id: torch.Tensor) -> torch.Tensor:
        rows = []
        for idx, route in enumerate(ORDERED_ROUTES):
            mask = route_id.eq(idx)
            if not mask.any():
                continue
            fa = self.value_feature(a[mask], route[0])
            fb = self.value_feature(b[mask], route[1])
            route_emb = self.route_embedding(route_id[mask])
            rows.append((mask, self.ordered_pair(torch.cat([fa, fb, route_emb], dim=-1))))
        out = torch.zeros(a.shape[0], self.feature_dim, dtype=torch.float32, device=a.device)
        for mask, value in rows:
            out[mask] = value
        return out

    def all_ordered_route_features(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        route_features = []
        for idx, route in enumerate(ORDERED_ROUTES):
            fa = self.value_feature(a, route[0])
            fb = self.value_feature(b, route[1])
            route_id = torch.full_like(a, idx)
            route_emb = self.route_embedding(route_id)
            route_features.append(self.ordered_pair(torch.cat([fa, fb, route_emb], dim=-1)))
        return torch.stack(route_features, dim=1)

    def operand_query_features(self, a: torch.Tensor, b: torch.Tensor, route_id: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(a.shape[0], 3, self.feature_dim, dtype=torch.float32, device=a.device)
        for idx, route in enumerate(ORDERED_ROUTES):
            mask = route_id.eq(idx)
            if not mask.any():
                continue
            left_idx = MODALITIES.index(route[0])
            right_idx = MODALITIES.index(route[1])
            left_modality = torch.full((int(mask.sum().item()),), left_idx, dtype=torch.long, device=a.device)
            right_modality = torch.full((int(mask.sum().item()),), right_idx, dtype=torch.long, device=a.device)
            left_pos = torch.zeros_like(left_modality)
            right_pos = torch.ones_like(right_modality)
            fa = self.value_to_feature(self.value_feature(a[mask], route[0]))
            fb = self.value_to_feature(self.value_feature(b[mask], route[1]))
            out[mask, 0] = fa + self.operand_modality(left_modality) + self.operand_position(left_pos)
            out[mask, 1] = fb + self.operand_modality(right_modality) + self.operand_position(right_pos)
        query_pos = torch.full((a.shape[0],), 2, dtype=torch.long, device=a.device)
        out[:, 2] = self.answer_query[None, :] + self.operand_position(query_pos)
        return out


class TrimodalQNNModel(nn.Module):
    def __init__(self, cfg: dict[str, Any], *, modulus: int):
        super().__init__()
        self.cfg = dict(cfg)
        self.modulus = int(modulus)
        self.problem_mode = str(cfg.get("problem_mode", "three_sector"))
        if self.problem_mode == "three_sector":
            self.sectors = MODALITIES
        elif self.problem_mode == "ordered_route":
            self.sectors = ORDERED_ROUTES
        elif self.problem_mode == "operand_query":
            self.sectors = ("operand_a", "operand_b", "answer_query")
        else:
            raise ValueError(f"unknown problem_mode {self.problem_mode!r}")
        self.sector_count = len(self.sectors)
        self.n_qubits = int(cfg.get("n_qubits", 3))
        self.state_dim = 2**self.n_qubits
        self.n_layers = int(cfg.get("n_layers", 2))
        self.feature_dim = int(cfg.get("feature_dim", 24))
        self.ordered_initialization = str(cfg.get("ordered_initialization", "active_route"))
        self.readout_mode = str(cfg.get("readout_mode", "query_sector" if self.problem_mode == "operand_query" else "all_sectors"))
        if self.readout_mode not in {"all_sectors", "query_sector"}:
            raise ValueError("readout_mode must be one of: all_sectors, query_sector")
        if self.readout_mode == "query_sector" and "answer_query" not in self.sectors:
            raise ValueError("query_sector readout requires an answer_query sector")
        self.readout_sector_index = self.sectors.index("answer_query") if "answer_query" in self.sectors else None
        self.head_type = str(cfg.get("head_type", "fourier_delta"))
        self._last_layerwise_logits: list[torch.Tensor] = []
        self._last_layerwise_q_hats: list[torch.Tensor] = []

        self.encoders = TrimodalEncoders(
            modulus=self.modulus,
            feature_dim=self.feature_dim,
            operand_feature_dim=int(cfg.get("operand_feature_dim", 16)),
            text_embed_dim=int(cfg.get("text_embed_dim", 16)),
            image_channels=int(cfg.get("image_channels", 8)),
        )
        self.state_projector = FeatureToState(self.feature_dim, self.state_dim)
        self.amplitudes = AmplitudeCoefficients(self.sector_count, mode=str(cfg.get("amplitude_mode", "fixed_equal")))
        self.layers = nn.ModuleList(
            [
                DataReuploadingLayer(
                    sector_count=self.sector_count,
                    feature_dim=self.feature_dim,
                    n_qubits=self.n_qubits,
                    cross_mixing=bool(cfg.get("cross_mixing", True)),
                    entangle=bool(cfg.get("entangle_content", True)),
                    route_mixer_type=str(cfg.get("route_mixer_type", "dense")),
                    route_interaction_rank=int(cfg.get("route_interaction_rank", 0)),
                    route_interaction_scale=float(cfg.get("route_interaction_scale", 0.02)),
                    route_dense_residual_scale=float(cfg.get("route_dense_residual_scale", 0.02)),
                )
                for _ in range(self.n_layers)
            ]
        )
        measured_dim = self.state_dim + 1 if self.readout_mode == "query_sector" else self.sector_count * self.state_dim + self.sector_count
        max_frequency = int(cfg.get("fourier_max_frequency", min(8, self.modulus // 2)))
        if self.head_type == "fourier_delta":
            self.head = FourierDeltaHead(
                feature_dim=measured_dim,
                modulus=self.modulus,
                max_frequency=max_frequency,
                residual=bool(cfg.get("fourier_residual", False)),
            )
        elif self.head_type == "dirac_delta":
            self.head = DiracDeltaHead(
                feature_dim=measured_dim,
                modulus=self.modulus,
                max_frequency=max_frequency,
                init_kernel=str(cfg.get("dirac_kernel_init", cfg.get("fourier_kernel_init", "fejer"))),
                trainable_kernel=bool(cfg.get("dirac_kernel_trainable", True)),
                coefficient_mode=str(cfg.get("dirac_coefficient_mode", "none")),
                coefficient_eps=float(cfg.get("dirac_coefficient_eps", 1e-6)),
            )
        elif self.head_type == "layerwise_dirac_mean":
            if self.n_layers < 1:
                raise ValueError("layerwise_dirac_mean requires at least one QNN layer")
            self.head = None
            self.layerwise_heads = nn.ModuleList(
                [
                    DiracDeltaHead(
                        feature_dim=measured_dim,
                        modulus=self.modulus,
                        max_frequency=max_frequency,
                        init_kernel=str(cfg.get("dirac_kernel_init", cfg.get("fourier_kernel_init", "fejer"))),
                        trainable_kernel=bool(cfg.get("dirac_kernel_trainable", True)),
                        coefficient_mode=str(cfg.get("dirac_coefficient_mode", "none")),
                        coefficient_eps=float(cfg.get("dirac_coefficient_eps", 1e-6)),
                    )
                    for _ in range(self.n_layers)
                ]
            )
        else:
            raise ValueError(f"unknown head_type {self.head_type!r}")

    def initial_state_and_features(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        a = batch["a"]
        b = batch["b"]
        lambdas, _, _ = self.amplitudes()
        if self.problem_mode == "three_sector":
            features = self.encoders.pair_features(a, b)
            state = self.state_projector(features)
            state = state * lambdas[None, :, None].to(state.device)
            return state, features

        if self.problem_mode == "operand_query":
            features = self.encoders.operand_query_features(a, b, batch["route_id"])
            state = self.state_projector(features)
            state = state * lambdas[None, :, None].to(state.device)
            return state, features

        if self.ordered_initialization == "all_route_superposition":
            features = self.encoders.all_ordered_route_features(a, b)
            state = self.state_projector(features)
            state = state * lambdas[None, :, None].to(state.device)
            return state, features

        route_id = batch["route_id"]
        active_features = self.encoders.ordered_route_features(a, b, route_id)
        active_state = self.state_projector(active_features)
        state = torch.zeros(a.shape[0], self.sector_count, self.state_dim, dtype=torch.complex64, device=a.device)
        features = torch.zeros(a.shape[0], self.sector_count, self.feature_dim, dtype=torch.float32, device=a.device)
        rows = torch.arange(a.shape[0], device=a.device)
        state[rows, route_id] = active_state
        features[rows, route_id] = active_features
        return state, features

    def measure(self, state: torch.Tensor) -> torch.Tensor:
        probs = state.abs().pow(2)
        sector_masses = probs.sum(dim=-1)
        if self.readout_mode == "query_sector":
            assert self.readout_sector_index is not None
            idx = int(self.readout_sector_index)
            return torch.cat([probs[:, idx], sector_masses[:, idx : idx + 1]], dim=-1)
        return torch.cat([probs.reshape(probs.shape[0], -1), sector_masses], dim=-1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        ablate_cross: bool = False,
        sector_mask: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._last_layerwise_logits = []
        self._last_layerwise_q_hats = []
        state, features = self.initial_state_and_features(batch)
        if sector_mask is not None:
            mask = sector_mask.to(state.device, dtype=state.real.dtype).reshape(1, self.sector_count, 1)
            state = state * mask
            norm = state.abs().pow(2).sum(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-8)
            state = state / norm
        layer_states = []
        layer_logits = []
        layer_q_hats = []
        for layer_idx, layer in enumerate(self.layers):
            state = layer(state, features, ablate_cross=ablate_cross)
            if return_state:
                layer_states.append(state)
            if self.head_type == "layerwise_dirac_mean":
                layer_out = self.layerwise_heads[layer_idx](self.measure(state))
                layer_logits.append(layer_out["logits"])
                layer_q_hats.append(layer_out["q_hat"])
        measured = self.measure(state)
        if self.head_type == "layerwise_dirac_mean":
            self._last_layerwise_logits = layer_logits
            self._last_layerwise_q_hats = layer_q_hats
            out = {
                "logits": torch.stack(layer_logits, dim=0).mean(dim=0),
                "q_hat": torch.stack(layer_q_hats, dim=0).mean(dim=0),
            }
        else:
            out = self.head(measured)
        out["features"] = measured
        out["state_norm"] = state_norm(state)
        if return_state:
            out["state"] = state
            out["layer_states"] = layer_states
        return out

    def amplitude_metrics(self) -> dict[str, float]:
        _, weights, phases = self.amplitudes()
        out: dict[str, float] = {}
        for idx, sector in enumerate(self.sectors):
            out[f"weight_{sector}"] = float(weights[idx].detach().cpu())
            out[f"phase_{sector}"] = float(phases[idx].detach().cpu())
        return out
