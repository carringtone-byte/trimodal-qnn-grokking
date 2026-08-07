import torch
from torch.utils.data._utils.collate import default_collate

from tri_modal_modular_grokking.data import MultiModalModularConfig, MultiModalModularDataset
from tri_modal_modular_grokking.evaluate_compression_checkpoints import aggregate_cells
from tri_modal_modular_grokking.losses import LossWeights, multimodal_loss
from tri_modal_modular_grokking.models import build_model, parameter_count


def _batch_and_model():
    cfg = MultiModalModularConfig(
        modulus=7,
        train_fraction=0.5,
        seed=0,
        max_examples=2,
        render={"height": 16, "width": 32, "font_size": 12},
    )
    dataset = MultiModalModularDataset(cfg, split="train")
    batch = default_collate([dataset[i] for i in range(min(12, len(dataset)))])
    metadata = dataset.metadata()
    model = build_model({"d_model": 24, "n_layers": 1, "n_heads": 3, "d_ff": 48, "image_tokens": 2}, metadata)
    return batch, model, cfg


def test_tri_modal_model_forward_and_loss():
    batch, model, cfg = _batch_and_model()
    outputs = model(batch, return_hidden=True)
    assert outputs["number_logits"].shape == (batch["s"].shape[0], cfg.modulus)
    assert outputs["image_class_logits"].shape == (batch["s"].shape[0], cfg.modulus)
    assert outputs["answer_slot"].shape[0] == batch["s"].shape[0]
    assert len(outputs["answer_slots_by_layer"]) == 1
    loss, metrics = multimodal_loss(outputs, batch, weights=LossWeights())
    assert torch.isfinite(loss)
    assert "exact_accuracy" in metrics


def test_tri_modal_model_answer_slot_patch_shape():
    batch, model, _cfg = _batch_and_model()
    clean = model(batch, return_hidden=True)
    values = clean["answer_slot"].detach()
    patched = model(batch, patch={"layer": -1, "values": values})
    assert patched["number_logits"].shape == clean["number_logits"].shape


def test_tri_modal_pixel_decoder_produces_images_and_trains():
    cfg = MultiModalModularConfig(
        modulus=7,
        train_fraction=0.5,
        seed=0,
        max_examples=2,
        decoder_target_kind="image_pixels",
        render={"height": 16, "width": 32, "font_size": 12},
    )
    dataset = MultiModalModularDataset(cfg, split="train")
    batch = default_collate([dataset[i] for i in range(min(12, len(dataset)))])
    model = build_model(
        {
            "d_model": 24,
            "n_layers": 1,
            "n_heads": 3,
            "d_ff": 48,
            "image_tokens": 2,
            "image_decoder_channels": 8,
        },
        dataset.metadata(),
    )
    outputs = model(batch)
    assert outputs["image_pixel_logits"].shape == (batch["s"].shape[0], 1, 16, 32)
    assert outputs["image_pixels"].shape == batch["target_image"].shape
    assert outputs["image_template_logits"].shape == (batch["s"].shape[0], cfg.modulus)
    loss, metrics = multimodal_loss(outputs, batch, weights=LossWeights())
    assert torch.isfinite(loss)
    assert "image_pixel_loss" in metrics
    loss.backward()
    assert model.image_pixel_decoder is not None
    assert any(parameter.grad is not None for parameter in model.image_pixel_decoder.parameters())


def test_phase10_native_trimodal_compression_budget() -> None:
    cfg = MultiModalModularConfig(
        modulus=97,
        train_fraction=0.3,
        seed=0,
        max_examples=1,
        decoder_target_kind="image_pixels",
        render={"height": 64, "width": 128, "font_size": 36},
    )
    dataset = MultiModalModularDataset(cfg, split="train")
    model = build_model(
        {
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 64,
            "image_tokens": 4,
            "image_decoder_channels": 8,
            "image_output_kind": "image_pixels",
        },
        dataset.metadata(),
    )
    assert parameter_count(model) == 81_927
    assert model.image_pixel_decoder is not None
    assert model.backbone.number.embedding_dim == 32
    assert model.backbone.text.embedding_dim == 32
    assert model.backbone.image.image_tokens == 4


def test_compression_evaluator_aggregates_cells_by_example_count() -> None:
    rows = [
        {"output_mode": "number", "accuracy": 1.0, "loss": 0.2, "n_examples": 3},
        {"output_mode": "number", "accuracy": 0.0, "loss": 0.8, "n_examples": 1},
        {"output_mode": "text", "accuracy": 0.5, "loss": 0.4, "n_examples": 2},
    ]
    result = {
        row["output_mode"]: row
        for row in aggregate_cells(rows, ("output_mode",))
    }
    assert result["number"]["accuracy"] == 0.75
    assert abs(result["number"]["loss"] - 0.35) < 1e-12
    assert result["number"]["n_examples"] == 4
    assert result["text"]["accuracy"] == 0.5


def test_tied_trimodal_readouts_share_input_parameters() -> None:
    batch, _model, _cfg = _batch_and_model()
    dataset_cfg = MultiModalModularConfig(
        modulus=7,
        train_fraction=0.5,
        seed=0,
        max_examples=2,
        render={"height": 16, "width": 32, "font_size": 12},
    )
    dataset = MultiModalModularDataset(dataset_cfg, split="train")
    model = build_model(
        {
            "d_model": 24,
            "n_layers": 1,
            "n_heads": 3,
            "d_ff": 48,
            "image_tokens": 2,
            "tie_number_embeddings": True,
            "tie_text_embeddings": True,
            "tie_image_class_to_number": True,
        },
        dataset.metadata(),
    )
    assert model.number_head.weight is model.backbone.number.weight
    assert model.image_class_head.weight is model.backbone.number.weight
    assert model.text_head.weight is model.backbone.text.weight
    outputs = model(batch)
    assert outputs["text_logits"].shape[-1] == len(dataset.metadata()["text_vocab"])


def test_uncertainty_weighted_multimodal_loss_trains_log_vars() -> None:
    batch, model, _cfg = _batch_and_model()
    log_vars = torch.nn.Parameter(torch.zeros(3))
    outputs = model(batch)
    loss, metrics = multimodal_loss(
        outputs,
        batch,
        weights=LossWeights(),
        uncertainty_log_vars=log_vars,
    )
    loss.backward()
    assert log_vars.grad is not None
    assert torch.isfinite(log_vars.grad).all()
    assert "adaptive_number_weight" in metrics
