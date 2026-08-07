from torch.utils.data._utils.collate import default_collate

from tri_modal_modular_grokking.data import MultiModalModularConfig, MultiModalModularDataset, split_pairs
from tri_modal_modular_grokking.render import RenderConfig, render_operand_image


def test_tri_modal_split_counts_and_disjointness():
    train, heldout = split_pairs(97, 0.3, 0)
    assert len(train) == 2822
    assert len(heldout) == 6587
    assert set(train).isdisjoint(heldout)


def test_tri_modal_dataset_all_cells_and_payloads():
    cfg = MultiModalModularConfig(modulus=7, train_fraction=0.4, seed=0, render={"height": 16, "width": 32, "font_size": 12})
    dataset = MultiModalModularDataset(cfg, split="train")
    assert len(dataset.cells) == 27
    assert len(dataset) == len(dataset.pairs) * 27
    sample = dataset[0]
    assert sample["operand_a_text_ids"].shape == sample["operand_a_text_mask"].shape
    assert sample["operand_a_image"].shape == (1, 16, 32)
    assert sample["target_image"].shape == (1, 16, 32)
    assert 0 <= int(sample["cell_id"]) < 27
    batch = default_collate([dataset[i] for i in range(4)])
    assert batch["operand_b_image"].shape == (4, 1, 16, 32)


def test_limited_train_combo_adds_only_selected_train_pairs():
    omitted = ["image", "text"]
    train_combos = [[a, b] for a in ["number", "text", "image"] for b in ["number", "text", "image"] if [a, b] != omitted]
    cfg = MultiModalModularConfig(
        modulus=7,
        train_fraction=0.4,
        seed=0,
        train_input_combos=train_combos,
        heldout_input_combos=[omitted],
        limited_train_input_combos=[{"combo": omitted, "n_pairs": 2, "seed": 123}],
        render={"height": 16, "width": 32, "font_size": 12},
    )
    train = MultiModalModularDataset(cfg, split="train")
    assert len(train.pairs) == 19
    assert len(train) == 19 * 8 * 3 + 2 * 3

    limited_examples = []
    for idx in range(len(train)):
        pair_idx, _pair, cell = train.pair_and_cell(idx)
        if [cell.mode_a, cell.mode_b] == omitted:
            limited_examples.append((pair_idx, cell.output_mode))
    assert len(limited_examples) == 2 * 3
    assert {output for _pair_idx, output in limited_examples} == {"number", "text", "image"}
    assert len({pair_idx for pair_idx, _output in limited_examples}) == 2

    heldout = MultiModalModularDataset(cfg, split="heldout")
    assert len(heldout.pairs) == 30
    assert len(heldout) == 30 * 3


def test_tri_modal_render_deterministic():
    cfg = RenderConfig(height=16, width=32, font_size=12)
    first = render_operand_image(6, cfg)
    second = render_operand_image(6, cfg)
    assert first.shape == (1, 16, 32)
    assert (first == second).all()
