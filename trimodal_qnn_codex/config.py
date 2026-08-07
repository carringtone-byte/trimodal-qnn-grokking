from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 0,
    "output_dir": "trimodal_qnn_codex/outputs/default",
    "data": {
        "modulus": 7,
        "train_fraction": 0.6,
        "split_seed": 0,
        "problem_mode": "three_sector",
        "train_routes": None,
        "eval_routes": None,
    },
    "model": {
        "problem_mode": "three_sector",
        "n_qubits": 3,
        "n_layers": 2,
        "feature_dim": 24,
        "operand_feature_dim": 16,
        "text_embed_dim": 16,
        "image_channels": 8,
        "amplitude_mode": "fixed_equal",
        "ordered_initialization": "active_route",
        "cross_mixing": True,
        "route_mixer_type": "dense",
        "route_interaction_rank": 0,
        "route_interaction_scale": 0.02,
        "route_dense_residual_scale": 0.02,
        "entangle_content": True,
        "head_type": "fourier_delta",
        "fourier_max_frequency": 3,
        "dirac_kernel_init": "fejer",
        "dirac_kernel_trainable": True,
        "dirac_coefficient_mode": "none",
        "dirac_coefficient_eps": 1e-6,
    },
    "training": {
        "steps": 50,
        "batch_size": 32,
        "lr": 0.003,
        "weight_decay": 0.0,
        "eval_every": 10,
        "checkpoint_every": 50,
        "same_sum_loss_weight": 0.0,
        "same_sum_loss_start_step": 0,
        "same_sum_loss_warmup_steps": 0,
        "same_sum_loss_kind": "logit_kl",
        "same_sum_target": "features",
        "same_sum_stop_gradient": True,
        "same_sum_normalize": True,
        "amplitude_balance_weight": 0.0,
        "fourier_auxiliary_weight": 0.0,
        "fourier_auxiliary_min_frequency": 1,
        "fourier_auxiliary_max_frequency": None,
        "fourier_auxiliary_normalize_pairs": True,
        "fourier_auxiliary_frequency_power": 0.0,
        "layerwise_dirac_loss_weight": 0.0,
        "layerwise_dirac_depth_weighted": False,
        "hard_neighbor_margin_weight": 0.0,
        "hard_neighbor_margin": 1.0,
        "hard_neighbor_offsets": [1],
        "hard_neighbor_focus_gamma": 0.0,
        "max_grad_norm": 1.0,
        "parameter_clip_abs": 0.0,
        "sector_mixer_clip_abs": 2.0,
        "head_scale_log_min": -4.0,
        "head_scale_log_max": 4.0,
        "stop_on_nonfinite": True,
        "resume_checkpoint": None,
        "resume_optimizer": True,
        "device": "auto",
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = deep_update(DEFAULT_CONFIG, data)
    data_mode = cfg["data"].get("problem_mode")
    model_mode = cfg["model"].get("problem_mode")
    if data_mode != model_mode:
        raise ValueError(f"data.problem_mode={data_mode!r} must match model.problem_mode={model_mode!r}")
    return cfg


def write_config(config: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
