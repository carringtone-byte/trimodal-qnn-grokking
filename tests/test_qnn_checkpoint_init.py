from __future__ import annotations

import pytest
import torch

from modular_addition.qnn_mod97 import (
    DiracDeltaHead,
    DiracResidualSharpeningHead,
    QNNClassifier,
    apply_parameter_trainability,
    fourier_targets,
    hard_offset_margin_loss,
    load_matching_state,
    metric_improved,
    scheduled_scalar,
)


def test_partial_fourier_delta_head_initialization(tmp_path):
    source = QNNClassifier(
        variant="expval_head",
        n_qubits=4,
        n_layers=1,
        modulus=11,
        readout_type="fourier_delta",
        fourier_max_frequency=5,
        fourier_residual_linear=True,
    )
    target = QNNClassifier(
        variant="expval_head",
        n_qubits=4,
        n_layers=1,
        modulus=11,
        readout_type="fourier_delta",
        fourier_max_frequency=3,
        fourier_residual_linear=True,
    )
    with torch.no_grad():
        source.head.proj.weight.copy_(torch.arange(source.head.proj.weight.numel()).view_as(source.head.proj.weight) / 100.0)
        source.head.proj.bias.copy_(torch.arange(source.head.proj.bias.numel()) / 100.0)
        source.head.frequency_weight_logits.copy_(torch.arange(source.head.frequency_weight_logits.numel()) / 10.0)
        source.head.bias.copy_(torch.arange(source.head.bias.numel()) / 10.0)

    checkpoint_path = tmp_path / "source.pt"
    torch.save({"model": source.state_dict(), "variant": "expval_head", "step": 123}, checkpoint_path)

    info = load_matching_state(
        target,
        checkpoint_path,
        device=torch.device("cpu"),
        skip_prefixes=(),
        allow_partial_tensors=True,
        partial_prefixes=("head.",),
    )

    assert info["source_step"] == 123
    assert info["partial_copied"] >= 3
    assert torch.allclose(target.head.proj.weight, source.head.proj.weight[: target.head.proj.weight.shape[0]])
    assert torch.allclose(target.head.proj.bias, source.head.proj.bias[: target.head.proj.bias.shape[0]])
    assert torch.allclose(target.head.frequency_weight_logits, source.head.frequency_weight_logits[: target.head.max_frequency])
    assert torch.allclose(target.head.bias, source.head.bias)


def test_metric_improved_respects_mode_and_delta():
    assert metric_improved(0.955, 0.954, mode="max", min_delta=0.0005)
    assert not metric_improved(0.9542, 0.954, mode="max", min_delta=0.0005)
    assert metric_improved(0.12, 0.13, mode="min", min_delta=0.005)
    assert not metric_improved(0.127, 0.13, mode="min", min_delta=0.005)
    with pytest.raises(ValueError):
        metric_improved(1.0, 0.0, mode="median", min_delta=0.0)


def test_scheduled_scalar_linear_anneal():
    cfg = {"x_start": 0.2, "x_end": 1.2, "x_anneal_steps": 10, "x_anneal_start_step": 5}
    assert scheduled_scalar(cfg, "x", 0.0, 0) == pytest.approx(0.2)
    assert scheduled_scalar(cfg, "x", 0.0, 5) == pytest.approx(0.2)
    assert scheduled_scalar(cfg, "x", 0.0, 10) == pytest.approx(0.7)
    assert scheduled_scalar(cfg, "x", 0.0, 20) == pytest.approx(1.2)


def test_hard_offset_margin_loss_supports_masks():
    logits = torch.tensor(
        [
            [4.0, 2.0, 0.0, 0.0, 0.0],
            [1.0, 1.2, 1.1, 0.0, 0.0],
        ]
    )
    labels = torch.tensor([0, 1])
    unmasked = hard_offset_margin_loss(logits, labels, offsets=[1], margin=1.0)
    masked = hard_offset_margin_loss(logits, labels, offsets=[1], margin=1.0, mask=torch.tensor([True, False]))
    assert unmasked > 0.0
    assert masked == pytest.approx(torch.tensor(0.0))


def test_dirac_delta_head_full_basis_selects_true_residue():
    modulus = 11
    labels = torch.arange(modulus)
    head = DiracDeltaHead(
        feature_dim=3,
        modulus=modulus,
        max_frequency=modulus // 2,
        init_kernel="dirichlet",
        coefficient_mode="none",
        trainable_kernel=False,
    )
    coeffs = fourier_targets(labels, modulus=modulus, max_frequency=modulus // 2)
    logits = head.delta_logits_from_coefficients(coeffs)
    assert torch.equal(logits.argmax(dim=-1), labels)


def test_dirac_delta_head_unit_normalizes_frequency_pairs():
    head = DiracDeltaHead(feature_dim=4, modulus=11, max_frequency=5, coefficient_mode="unit")
    normed = torch.randn(7, 4)
    coeffs = head.coefficients_from_normed(normed).reshape(7, 5, 2)
    assert torch.allclose(coeffs.norm(dim=-1), torch.ones(7, 5), atol=1e-5)


def test_dirac_delta_head_global_soft_unit_uses_one_confidence_per_example():
    head = DiracDeltaHead(feature_dim=4, modulus=11, max_frequency=5, coefficient_mode="global_soft_unit")
    normed = torch.randn(7, 4)
    coeffs = head.coefficients_from_normed(normed).reshape(7, 5, 2)
    pair_norms = coeffs.norm(dim=-1)
    assert torch.all(pair_norms <= 1.0 + 1e-5)
    assert torch.allclose(pair_norms, pair_norms[:, :1].expand_as(pair_norms), atol=1e-5)


def test_qnn_dirac_delta_readout_forward_shape():
    model = QNNClassifier(
        variant="expval_head",
        n_qubits=4,
        n_layers=1,
        modulus=11,
        readout_type="dirac_delta",
        fourier_max_frequency=5,
    )
    logits = model(torch.tensor([0, 3]), torch.tensor([1, 4]))
    assert logits.shape == (2, 11)


def test_dirac_residual_sharpening_head_starts_as_identity():
    head = DiracResidualSharpeningHead(
        feature_dim=4,
        modulus=11,
        max_frequency=5,
        sharpen_strength_init=0.0,
        sharpen_strength_max=0.5,
    )
    features = torch.randn(6, 4)
    assert torch.allclose(head(features), head.base_logits(features), atol=1e-6)
    assert torch.allclose(head.correction_logits(features), torch.zeros(6, 11), atol=1e-6)


def test_dirac_residual_sharpening_head_correction_turns_on():
    head = DiracResidualSharpeningHead(
        feature_dim=4,
        modulus=11,
        max_frequency=5,
        sharpen_strength_init=0.25,
        sharpen_strength_max=0.5,
    )
    features = torch.randn(6, 4)
    correction = head.correction_logits(features)
    assert correction.shape == (6, 11)
    assert correction.abs().sum() > 0.0


def test_qnn_dirac_residual_sharpening_readout_forward_shape():
    model = QNNClassifier(
        variant="expval_head",
        n_qubits=4,
        n_layers=1,
        modulus=11,
        readout_type="dirac_residual_sharpen",
        fourier_max_frequency=5,
    )
    logits = model(torch.tensor([0, 3]), torch.tensor([1, 4]))
    assert logits.shape == (2, 11)


@pytest.mark.parametrize("readout_type", ["layerwise_dirac_mean", "layerwise_dirac_ensemble"])
def test_qnn_layerwise_average_readouts_forward_shape(readout_type):
    model = QNNClassifier(
        variant="prob_head",
        n_qubits=4,
        n_layers=2,
        modulus=11,
        readout_type=readout_type,
        fourier_max_frequency=5,
    )
    logits = model(torch.tensor([0, 3]), torch.tensor([1, 4]))
    assert logits.shape == (2, 11)
    assert len(model._last_layerwise_logits) == 2


def test_trainable_parameter_prefixes_freeze_base_path():
    model = QNNClassifier(
        variant="expval_head",
        n_qubits=4,
        n_layers=1,
        modulus=11,
        readout_type="dirac_residual_sharpen",
        fourier_max_frequency=5,
    )
    info = apply_parameter_trainability(model, {"trainable_parameter_prefixes": ["head.sharpen_"]})
    assert info["trainable_parameter_count"] > 0
    assert info["frozen_parameter_count"] > 0
    for name, param in model.named_parameters():
        assert param.requires_grad == name.startswith("head.sharpen_")
