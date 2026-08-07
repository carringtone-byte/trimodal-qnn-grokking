from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import Dataset

from modular_addition.data import WordTokenizer, build_modular_word_tokenizer, modular_word_answer_tokens, number_to_words

from .render import RenderConfig, render_answer_image, render_operand_image, save_tensor_image


INPUT_MODES = ("number", "text", "image")
OUTPUT_MODES = ("number", "text", "image")
DECODER_TARGET_KINDS = ("number", "text", "image_class_proxy", "image_tokens", "image_pixels")


@dataclass(frozen=True)
class TaskCell:
    mode_a: str
    mode_b: str
    output_mode: str

    @property
    def key(self) -> str:
        return f"{self.mode_a}+{self.mode_b}->{self.output_mode}"


@dataclass
class MultiModalModularConfig:
    modulus: int = 97
    train_fraction: float = 0.30
    seed: int = 0
    input_modes: list[str] = field(default_factory=lambda: list(INPUT_MODES))
    output_modes: list[str] = field(default_factory=lambda: list(OUTPUT_MODES))
    train_input_combos: list[list[str]] | None = None
    heldout_input_combos: list[list[str]] | None = None
    limited_train_input_combos: list[dict[str, Any]] | None = None
    strict_pair_holdout_across_modalities: bool = True
    examples_per_pair_per_cell: int = 1
    decoder_target_kind: str = "image_class_proxy"
    render: dict[str, Any] = field(default_factory=dict)
    max_examples: int | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "MultiModalModularConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in cfg.items() if key in known})


def split_pairs(modulus: int, train_fraction: float, seed: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    cut = int(len(pairs) * train_fraction)
    return pairs[:cut], pairs[cut:]


def input_combos(modes: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for a in modes for b in modes]


def _combo_tuple(combo: list[str] | tuple[str, str]) -> tuple[str, str]:
    if len(combo) != 2:
        raise ValueError(f"input combo must have length 2, got {combo}")
    return str(combo[0]), str(combo[1])


def limited_train_specs(cfg: MultiModalModularConfig) -> list[dict[str, Any]]:
    return list(cfg.limited_train_input_combos or [])


def limited_train_combos(cfg: MultiModalModularConfig) -> list[tuple[str, str]]:
    combos: list[tuple[str, str]] = []
    for spec in limited_train_specs(cfg):
        n_pairs = int(spec.get("n_pairs", spec.get("count", 0)))
        if n_pairs <= 0:
            continue
        combos.append(_combo_tuple(spec["combo"]))
    return combos


def make_cells(cfg: MultiModalModularConfig, split: str) -> list[TaskCell]:
    combos = input_combos(cfg.input_modes)
    if split == "train" and cfg.train_input_combos is not None:
        combos = [_combo_tuple(combo) for combo in cfg.train_input_combos]
        for combo in limited_train_combos(cfg):
            if combo not in combos:
                combos.append(combo)
    if split in {"val", "test", "heldout"} and cfg.heldout_input_combos is not None:
        combos = [_combo_tuple(combo) for combo in cfg.heldout_input_combos]
    return [TaskCell(a, b, out) for a, b in combos for out in cfg.output_modes]


def residue(pair: tuple[int, int], modulus: int) -> int:
    return (int(pair[0]) + int(pair[1])) % modulus


def stratified_pair_indices(
    pairs: list[tuple[int, int]],
    *,
    n: int,
    modulus: int,
    seed: int,
) -> list[int]:
    if n < 0:
        raise ValueError(f"n must be nonnegative, got {n}")
    if n > len(pairs):
        raise ValueError(f"cannot sample {n} pairs from {len(pairs)} available pairs")
    groups: dict[int, list[int]] = {idx: [] for idx in range(modulus)}
    for idx, pair in enumerate(pairs):
        groups[residue(pair, modulus)].append(idx)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    residue_order = list(range(modulus))
    selected: list[int] = []
    while len(selected) < n:
        rng.shuffle(residue_order)
        made_progress = False
        for r in residue_order:
            group = groups[r]
            if group:
                selected.append(group.pop())
                made_progress = True
                if len(selected) >= n:
                    break
        if not made_progress:
            break
    if len(selected) != n:
        raise ValueError(f"could only sample {len(selected)} pairs, requested {n}")
    return sorted(selected)


def max_number_word_len(modulus: int) -> int:
    return max(len(number_to_words(value).split()) for value in range(modulus)) + 2


def pad_ids(ids: list[int], length: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(ids) > length:
        raise ValueError(f"token sequence length {len(ids)} exceeds max length {length}")
    mask = [1] * len(ids) + [0] * (length - len(ids))
    padded = ids + [pad_id] * (length - len(ids))
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


class MultiModalModularDataset(Dataset):
    def __init__(self, cfg: MultiModalModularConfig, *, split: str):
        if cfg.decoder_target_kind not in DECODER_TARGET_KINDS:
            raise ValueError(f"unknown decoder_target_kind: {cfg.decoder_target_kind}")
        if any(mode not in INPUT_MODES for mode in cfg.input_modes):
            raise ValueError(f"input_modes must be drawn from {INPUT_MODES}")
        if any(mode not in OUTPUT_MODES for mode in cfg.output_modes):
            raise ValueError(f"output_modes must be drawn from {OUTPUT_MODES}")
        self.cfg = cfg
        self.split = "heldout" if split in {"val", "test"} else split
        self.tokenizer = build_modular_word_tokenizer(cfg.modulus)
        self.text_len = max_number_word_len(cfg.modulus)
        self.render_cfg = RenderConfig(**cfg.render)
        self.mode_to_id = {mode: idx for idx, mode in enumerate(INPUT_MODES)}
        self.output_mode_to_id = {mode: idx for idx, mode in enumerate(OUTPUT_MODES)}
        self.target_kind_to_id = {kind: idx for idx, kind in enumerate(DECODER_TARGET_KINDS)}
        self._text_cache = [self._encode_text_value_uncached(value) for value in range(cfg.modulus)]
        self._image_cache = [render_operand_image(value, self.render_cfg) for value in range(cfg.modulus)]
        train_pairs, heldout_pairs = split_pairs(cfg.modulus, cfg.train_fraction, cfg.seed)
        if self.split == "train":
            pairs = train_pairs
        elif self.split == "heldout":
            pairs = heldout_pairs
        elif self.split == "all":
            pairs = [(a, b) for a in range(cfg.modulus) for b in range(cfg.modulus)]
        else:
            raise ValueError(f"unknown split: {split}")
        if cfg.max_examples is not None:
            pairs = pairs[: int(cfg.max_examples)]
        self.pairs = pairs
        self.cells = make_cells(cfg, self.split)
        self.cell_to_id = {cell.key: idx for idx, cell in enumerate(self.cells)}
        self._examples: list[tuple[int, int]] | None = None
        if self.split == "train" and limited_train_specs(cfg):
            self._examples = self._build_limited_train_examples()

    def _build_limited_train_examples(self) -> list[tuple[int, int]]:
        base_combos = {_combo_tuple(combo) for combo in (self.cfg.train_input_combos or input_combos(self.cfg.input_modes))}
        limited_by_combo: dict[tuple[str, str], set[int]] = {}
        for order, spec in enumerate(limited_train_specs(self.cfg)):
            combo = _combo_tuple(spec["combo"])
            n_pairs = int(spec.get("n_pairs", spec.get("count", 0)))
            if n_pairs <= 0:
                continue
            seed = int(spec.get("seed", self.cfg.seed + 100_003 + order))
            selected = set(stratified_pair_indices(self.pairs, n=n_pairs, modulus=self.cfg.modulus, seed=seed))
            limited_by_combo.setdefault(combo, set()).update(selected)
        examples: list[tuple[int, int]] = []
        for pair_idx, _pair in enumerate(self.pairs):
            for cell_idx, cell in enumerate(self.cells):
                combo = (cell.mode_a, cell.mode_b)
                if combo in base_combos or pair_idx in limited_by_combo.get(combo, set()):
                    for _ in range(self.cfg.examples_per_pair_per_cell):
                        examples.append((pair_idx, cell_idx))
        return examples

    def __len__(self) -> int:
        if self._examples is not None:
            return len(self._examples)
        return len(self.pairs) * len(self.cells) * self.cfg.examples_per_pair_per_cell

    def pair_and_cell(self, idx: int) -> tuple[int, tuple[int, int], TaskCell]:
        if self._examples is not None:
            pair_idx, cell_idx = self._examples[idx]
            return pair_idx, self.pairs[pair_idx], self.cells[cell_idx]
        per_pair = len(self.cells) * self.cfg.examples_per_pair_per_cell
        pair_idx = idx // per_pair
        rem = idx % per_pair
        cell_idx = rem // self.cfg.examples_per_pair_per_cell
        return pair_idx, self.pairs[pair_idx], self.cells[cell_idx]

    def _encode_text_value_uncached(self, value: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = ["<bos>"] + number_to_words(value).split() + ["<eos>"]
        return pad_ids(self.tokenizer.encode(tokens), self.text_len, self.tokenizer.pad_token_id)

    def encode_text_value(self, value: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self._text_cache[value]
        return ids.clone(), mask.clone()

    def operand_payload(self, value: int, mode: str, prefix: str) -> dict[str, torch.Tensor]:
        token_ids, token_mask = self.encode_text_value(value)
        image = self._image_cache[value].clone()
        active_text = mode == "text"
        active_image = mode == "image"
        return {
            f"{prefix}_mode_id": torch.tensor(self.mode_to_id[mode], dtype=torch.long),
            f"{prefix}_residue_id": torch.tensor(value, dtype=torch.long),
            f"{prefix}_text_ids": token_ids,
            f"{prefix}_text_mask": token_mask if active_text else torch.zeros_like(token_mask),
            f"{prefix}_image": image if active_image else torch.zeros_like(image),
            f"{prefix}_is_text": torch.tensor(active_text, dtype=torch.bool),
            f"{prefix}_is_image": torch.tensor(active_image, dtype=torch.bool),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair_idx, (a, b), cell = self.pair_and_cell(idx)
        s = (a + b) % self.cfg.modulus
        target_text_ids, target_text_mask = self.encode_text_value(s)
        target_image = self._image_cache[s].clone()
        out: dict[str, torch.Tensor] = {
            "a": torch.tensor(a, dtype=torch.long),
            "b": torch.tensor(b, dtype=torch.long),
            "s": torch.tensor(s, dtype=torch.long),
            "wrap": torch.tensor(a + b >= self.cfg.modulus, dtype=torch.long),
            "pair_id": torch.tensor(pair_idx, dtype=torch.long),
            "cell_id": torch.tensor(self.cell_to_id[cell.key], dtype=torch.long),
            "output_mode_id": torch.tensor(self.output_mode_to_id[cell.output_mode], dtype=torch.long),
            "decoder_target_kind_id": torch.tensor(self.target_kind_to_id[self.target_kind(cell.output_mode)], dtype=torch.long),
            "target_number": torch.tensor(s, dtype=torch.long),
            "target_text_ids": target_text_ids,
            "target_text_mask": target_text_mask,
            "target_image": target_image,
            "labels": torch.tensor(s, dtype=torch.long),
        }
        out.update(self.operand_payload(a, cell.mode_a, "operand_a"))
        out.update(self.operand_payload(b, cell.mode_b, "operand_b"))
        return out

    def target_kind(self, output_mode: str) -> str:
        if output_mode == "number":
            return "number"
        if output_mode == "text":
            return "text"
        return self.cfg.decoder_target_kind

    def metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.cfg),
            "split": self.split,
            "num_pairs": len(self.pairs),
            "num_cells": len(self.cells),
            "cells": [{"id": idx, "mode_a": c.mode_a, "mode_b": c.mode_b, "output_mode": c.output_mode, "key": c.key} for idx, c in enumerate(self.cells)],
            "input_modes": list(INPUT_MODES),
            "output_modes": list(OUTPUT_MODES),
            "decoder_target_kinds": list(DECODER_TARGET_KINDS),
            "text_vocab": self.tokenizer.tokens,
            "text_len": self.text_len,
        }


def make_datasets(cfg: MultiModalModularConfig) -> tuple[MultiModalModularDataset, MultiModalModularDataset]:
    return MultiModalModularDataset(cfg, split="train"), MultiModalModularDataset(cfg, split="heldout")


def write_corpus(cfg: MultiModalModularConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = MultiModalModularDataset(cfg, split="train")
    heldout = MultiModalModularDataset(cfg, split="heldout")
    (out_dir / "metadata.json").write_text(json.dumps({"train": train.metadata(), "heldout": heldout.metadata()}, indent=2), encoding="utf-8")
    for name, dataset in [("train", train), ("heldout", heldout)]:
        with (out_dir / f"{name}_manifest.jsonl").open("w", encoding="utf-8") as f:
            for pair_idx, (a, b) in enumerate(dataset.pairs):
                f.write(json.dumps({"pair_id": pair_idx, "a": a, "b": b, "s": (a + b) % cfg.modulus}) + "\n")
    sample_dir = out_dir / "render_samples"
    for value in range(min(cfg.modulus, 12)):
        save_tensor_image(render_operand_image(value, train.render_cfg), sample_dir / f"operand_{value:03d}.png")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tri-modal modular-addition data generator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="tri_modal_modular_grokking/corpora/mod97_seed0_train30")
    parser.add_argument("--write-corpus", action="store_true")
    args = parser.parse_args()
    cfg = MultiModalModularConfig.from_dict(load_config(args.config).get("dataset", load_config(args.config)))
    if args.write_corpus:
        write_corpus(cfg, Path(args.out_dir))
        print(args.out_dir)
    else:
        train, heldout = make_datasets(cfg)
        print(json.dumps({"train": train.metadata(), "heldout": heldout.metadata()}, indent=2))


if __name__ == "__main__":
    main()
