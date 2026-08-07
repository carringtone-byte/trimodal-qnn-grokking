from __future__ import annotations

import json

import torch

from trimodal_qnn_codex.amplitudes import AmplitudeCoefficients
from trimodal_qnn_codex.data import ORDERED_ROUTES, ModularPairsDataset
from trimodal_qnn_codex.heads import DiracDeltaHead, FourierDeltaHead, fourier_targets
from trimodal_qnn_codex.models import TrimodalQNNModel
from trimodal_qnn_codex.quantum import FactorizedRouteMixingUnitary, HybridFactorizedDenseRouteMixingUnitary, SectorMixingUnitary
from trimodal_qnn_codex.run_seed_pipeline import choose_best_checkpoint, make_seed_config
from trimodal_qnn_codex.train import same_sum_loss, scheduled_same_sum_weight


def test_amplitudes_are_normalized():
    for mode in ("fixed_equal", "learnable_real", "learnable_complex"):
        amp = AmplitudeCoefficients(3, mode=mode)
        lambdas, weights, _ = amp()
        assert torch.allclose(lambdas.abs().pow(2).sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_ordered_dataset_routes_and_labels():
    cfg = {
        "modulus": 5,
        "train_fraction": 0.6,
        "split_seed": 0,
        "problem_mode": "ordered_route",
        "train_routes": ["TI", "NT"],
        "eval_routes": list(ORDERED_ROUTES),
    }
    ds = ModularPairsDataset(cfg, split="train")
    assert ds.metadata()["routes"] == ["TI", "NT"]
    row = ds[0]
    assert row["y"].item() == (row["a"].item() + row["b"].item()) % 5
    assert row["route_id"].item() in {ORDERED_ROUTES.index("TI"), ORDERED_ROUTES.index("NT")}


def test_operand_query_dataset_uses_ordered_routes():
    cfg = {
        "modulus": 5,
        "train_fraction": 0.6,
        "split_seed": 0,
        "problem_mode": "operand_query",
        "train_routes": ["TI", "NT"],
        "eval_routes": list(ORDERED_ROUTES),
    }
    ds = ModularPairsDataset(cfg, split="train")
    assert ds.metadata()["routes"] == ["TI", "NT"]
    row = ds[0]
    assert row["y"].item() == (row["a"].item() + row["b"].item()) % 5
    assert row["route_id"].item() in {ORDERED_ROUTES.index("TI"), ORDERED_ROUTES.index("NT")}


def test_sector_mixer_is_unitary():
    mixer = SectorMixingUnitary(4)
    u = mixer.unitary()
    ident = torch.eye(4, dtype=torch.complex64)
    assert torch.allclose(u.conj().T @ u, ident, atol=1e-5)


def test_factorized_route_mixer_is_unitary():
    mixer = FactorizedRouteMixingUnitary(interaction_rank=2)
    u = mixer.unitary()
    ident = torch.eye(9, dtype=torch.complex64)
    assert torch.allclose(u.conj().T @ u, ident, atol=1e-5)


def test_fourier_head_selects_synthetic_true_residue():
    modulus = 7
    head = FourierDeltaHead(feature_dim=2, modulus=modulus, max_frequency=modulus // 2)
    labels = torch.arange(modulus)
    coeffs = fourier_targets(labels, modulus=modulus, max_frequency=modulus // 2)
    logits = head.logits_from_coefficients(coeffs)
    assert torch.equal(logits.argmax(dim=-1), labels)


def test_dirac_head_selects_synthetic_true_residue():
    modulus = 7
    head = DiracDeltaHead(feature_dim=2, modulus=modulus, max_frequency=modulus // 2, init_kernel="dirichlet")
    labels = torch.arange(modulus)
    coeffs = fourier_targets(labels, modulus=modulus, max_frequency=modulus // 2)
    logits = head.logits_from_coefficients(coeffs)
    assert torch.equal(logits.argmax(dim=-1), labels)


def test_three_sector_forward_shapes_and_norm():
    cfg = {
        "problem_mode": "three_sector",
        "n_qubits": 2,
        "n_layers": 1,
        "feature_dim": 12,
        "operand_feature_dim": 8,
        "text_embed_dim": 8,
        "image_channels": 4,
        "amplitude_mode": "fixed_equal",
        "fourier_max_frequency": 2,
    }
    model = TrimodalQNNModel(cfg, modulus=5)
    batch = {
        "a": torch.tensor([0, 1, 2]),
        "b": torch.tensor([1, 2, 3]),
        "y": torch.tensor([1, 3, 0]),
        "route_id": torch.tensor([-1, -1, -1]),
    }
    out = model(batch)
    assert out["logits"].shape == (3, 5)
    assert out["q_hat"].shape == (3, 2, 2)
    assert torch.allclose(out["state_norm"], torch.ones(3), atol=1e-5)


def test_three_sector_layerwise_dirac_mean_forward_shapes_and_norm():
    cfg = {
        "problem_mode": "three_sector",
        "n_qubits": 2,
        "n_layers": 2,
        "feature_dim": 12,
        "operand_feature_dim": 8,
        "text_embed_dim": 8,
        "image_channels": 4,
        "amplitude_mode": "fixed_equal",
        "head_type": "layerwise_dirac_mean",
        "fourier_max_frequency": 2,
        "dirac_coefficient_mode": "none",
    }
    model = TrimodalQNNModel(cfg, modulus=5)
    batch = {
        "a": torch.tensor([0, 1, 2]),
        "b": torch.tensor([1, 2, 3]),
        "y": torch.tensor([1, 3, 0]),
        "route_id": torch.tensor([-1, -1, -1]),
    }
    out = model(batch)
    assert out["logits"].shape == (3, 5)
    assert out["q_hat"].shape == (3, 2, 2)
    assert len(model._last_layerwise_logits) == 2
    assert torch.allclose(out["state_norm"], torch.ones(3), atol=1e-5)


def test_ordered_route_forward_shapes_and_norm():
    cfg = {
        "problem_mode": "ordered_route",
        "n_qubits": 2,
        "n_layers": 1,
        "feature_dim": 12,
        "operand_feature_dim": 8,
        "text_embed_dim": 8,
        "image_channels": 4,
        "amplitude_mode": "fixed_equal",
        "ordered_initialization": "active_route",
        "fourier_max_frequency": 2,
    }
    model = TrimodalQNNModel(cfg, modulus=5)
    batch = {
        "a": torch.tensor([0, 1, 2]),
        "b": torch.tensor([1, 2, 3]),
        "y": torch.tensor([1, 3, 0]),
        "route_id": torch.tensor([ORDERED_ROUTES.index("TI"), ORDERED_ROUTES.index("NT"), ORDERED_ROUTES.index("II")]),
    }
    out = model(batch)
    assert out["logits"].shape == (3, 5)
    assert torch.allclose(out["state_norm"], torch.ones(3), atol=1e-5)


def test_ordered_route_factorized_superposition_forward_shapes_and_norm():
    cfg = {
        "problem_mode": "ordered_route",
        "n_qubits": 2,
        "n_layers": 1,
        "feature_dim": 12,
        "operand_feature_dim": 8,
        "text_embed_dim": 8,
        "image_channels": 4,
        "amplitude_mode": "fixed_equal",
        "ordered_initialization": "all_route_superposition",
        "route_mixer_type": "factorized",
        "route_interaction_rank": 2,
        "head_type": "layerwise_dirac_mean",
        "fourier_max_frequency": 2,
    }
    model = TrimodalQNNModel(cfg, modulus=5)
    batch = {
        "a": torch.tensor([0, 1, 2]),
        "b": torch.tensor([1, 2, 3]),
        "y": torch.tensor([1, 3, 0]),
        "route_id": torch.tensor([ORDERED_ROUTES.index("TI"), ORDERED_ROUTES.index("NT"), ORDERED_ROUTES.index("II")]),
    }
    out = model(batch)
    assert out["logits"].shape == (3, 5)
    assert out["q_hat"].shape == (3, 2, 2)
    assert torch.allclose(out["state_norm"], torch.ones(3), atol=1e-5)


def test_hybrid_factorized_dense_mixer_is_unitary():
    mixer = HybridFactorizedDenseRouteMixingUnitary(interaction_rank=1, dense_residual_scale=0.03)
    u = mixer.unitary()
    eye = torch.eye(9, dtype=torch.complex64)
    assert torch.allclose(u.conj().T @ u, eye, atol=1e-5)


def test_ordered_route_hybrid_superposition_forward_shapes_and_norm():
    cfg = {
        "problem_mode": "ordered_route",
        "n_qubits": 2,
        "n_layers": 1,
        "feature_dim": 12,
        "operand_feature_dim": 8,
        "text_embed_dim": 8,
        "image_channels": 4,
        "amplitude_mode": "fixed_equal",
        "ordered_initialization": "all_route_superposition",
        "route_mixer_type": "hybrid_factorized_dense",
        "route_interaction_rank": 1,
        "route_dense_residual_scale": 0.03,
        "head_type": "layerwise_dirac_mean",
        "fourier_max_frequency": 2,
    }
    model = TrimodalQNNModel(cfg, modulus=5)
    batch = {
        "a": torch.tensor([0, 1, 2]),
        "b": torch.tensor([1, 2, 3]),
        "y": torch.tensor([1, 3, 0]),
        "route_id": torch.tensor([ORDERED_ROUTES.index("TI"), ORDERED_ROUTES.index("NT"), ORDERED_ROUTES.index("II")]),
    }
    out = model(batch)
    assert out["logits"].shape == (3, 5)
    assert torch.allclose(out["state_norm"], torch.ones(3), atol=1e-5)


def test_operand_query_forward_uses_query_readout():
    cfg = {
        "problem_mode": "operand_query",
        "n_qubits": 2,
        "n_layers": 1,
        "feature_dim": 12,
        "operand_feature_dim": 8,
        "text_embed_dim": 8,
        "image_channels": 4,
        "amplitude_mode": "fixed_equal",
        "cross_mixing": True,
        "route_mixer_type": "dense",
        "readout_mode": "query_sector",
        "head_type": "dirac_delta",
        "fourier_max_frequency": 2,
    }
    model = TrimodalQNNModel(cfg, modulus=5)
    batch = {
        "a": torch.tensor([0, 1, 2]),
        "b": torch.tensor([1, 2, 3]),
        "y": torch.tensor([1, 3, 0]),
        "route_id": torch.tensor([ORDERED_ROUTES.index("TI"), ORDERED_ROUTES.index("NT"), ORDERED_ROUTES.index("II")]),
    }
    state, features = model.initial_state_and_features(batch)
    assert state.shape == (3, 3, 4)
    assert features.shape == (3, 3, 12)
    assert model.measure(state).shape == (3, 5)
    out = model(batch)
    assert out["logits"].shape == (3, 5)
    assert torch.allclose(out["state_norm"], torch.ones(3), atol=1e-5)


def test_scheduled_same_sum_weight_ramps_after_start():
    cfg = {"same_sum_loss_weight": 0.02, "same_sum_loss_start_step": 10, "same_sum_loss_warmup_steps": 5}
    assert scheduled_same_sum_weight(cfg, 9) == 0.0
    assert 0.0 < scheduled_same_sum_weight(cfg, 10) < 0.02
    assert scheduled_same_sum_weight(cfg, 14) == 0.02
    assert scheduled_same_sum_weight(cfg, 20) == 0.02


def test_representation_same_sum_loss_has_finite_gradients():
    cfg = {
        "problem_mode": "ordered_route",
        "n_qubits": 2,
        "n_layers": 1,
        "feature_dim": 12,
        "operand_feature_dim": 8,
        "text_embed_dim": 8,
        "image_channels": 4,
        "amplitude_mode": "fixed_equal",
        "ordered_initialization": "all_route_superposition",
        "route_mixer_type": "factorized",
        "route_interaction_rank": 1,
        "head_type": "layerwise_dirac_mean",
        "fourier_max_frequency": 2,
    }
    model = TrimodalQNNModel(cfg, modulus=5)
    batch = {
        "a": torch.tensor([0, 1, 2, 3]),
        "b": torch.tensor([1, 2, 3, 4]),
        "y": torch.tensor([1, 3, 0, 2]),
        "route_id": torch.tensor([
            ORDERED_ROUTES.index("TI"),
            ORDERED_ROUTES.index("NT"),
            ORDERED_ROUTES.index("II"),
            ORDERED_ROUTES.index("TN"),
        ]),
    }
    loss = same_sum_loss(
        model,
        batch,
        train_cfg={
            "same_sum_loss_kind": "representation_mse",
            "same_sum_target": "features",
            "same_sum_stop_gradient": True,
            "same_sum_normalize": True,
        },
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters())


def test_seed_pipeline_config_is_deterministic_and_non_mutating():
    base = {
        "seed": 9301,
        "output_dir": "trimodal_qnn_codex/outputs/base",
        "training": {"steps": 30000, "device": "auto", "resume_checkpoint": None},
        "notes": {"goal": "base"},
    }
    cfg = make_seed_config(
        base,
        seed=9302,
        run_root="trimodal_qnn_codex/outputs/sweep",
        device="cpu",
        training_steps=1234,
        resume_checkpoint="trimodal_qnn_codex/outputs/sweep/seed_9302/checkpoint_1000.pt",
    )
    assert cfg["seed"] == 9302
    assert cfg["output_dir"] == "trimodal_qnn_codex/outputs/sweep/seed_9302"
    assert cfg["training"]["device"] == "cpu"
    assert cfg["training"]["steps"] == 1234
    assert cfg["training"]["resume_checkpoint"].endswith("checkpoint_1000.pt")
    assert base["seed"] == 9301
    assert base["training"]["device"] == "auto"


def test_seed_pipeline_selects_best_metric_checkpoint(tmp_path):
    for step in (1000, 2000, 3000):
        (tmp_path / f"checkpoint_{step}.pt").write_text("placeholder", encoding="utf-8")
    rows = [
        {"step": 1000, "heldout_accuracy": 0.70, "train_accuracy": 0.91},
        {"step": 2000, "heldout_accuracy": 0.85, "train_accuracy": 0.95},
        {"step": 3000, "heldout_accuracy": 0.80, "train_accuracy": 0.99},
    ]
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    best = choose_best_checkpoint(tmp_path)
    assert best is not None
    assert best["step"] == 2000
    assert best["source"] == "best_heldout_accuracy"
    assert best["heldout_accuracy"] == 0.85
    preferred = choose_best_checkpoint(tmp_path, preferred_step=3000)
    assert preferred is not None
    assert preferred["step"] == 3000
    assert preferred["source"] == "preferred"


def test_seed_pipeline_checkpoint_selection_falls_back_to_latest(tmp_path):
    for step in (1000, 2000):
        (tmp_path / f"checkpoint_{step}.pt").write_text("placeholder", encoding="utf-8")
    best = choose_best_checkpoint(tmp_path)
    assert best is not None
    assert best["step"] == 2000
    assert best["source"] == "latest_regular_checkpoint"
