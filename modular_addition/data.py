from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TaskSpec:
    objective: str
    vocab_size: int | None = None
    num_classes: int | None = None
    input_dim: int | None = None
    output_dim: int | None = None
    pad_token_id: int | None = None


class DatasetBundle(NamedTuple):
    train: Dataset
    val: Dataset
    test: Dataset | None
    spec: TaskSpec


ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
TENS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def number_to_words(value: int) -> str:
    if value < 0 or value >= 1000:
        raise ValueError("number_to_words supports integers from 0 to 999")
    if value < 20:
        return ONES[value]
    if value >= 100:
        hundreds = value // 100
        remainder = value % 100
        if remainder == 0:
            return f"{ONES[hundreds]} hundred"
        return f"{ONES[hundreds]} hundred {number_to_words(remainder)}"
    tens = value // 10 * 10
    ones = value % 10
    if ones == 0:
        return TENS[tens]
    return f"{TENS[tens]} {ONES[ones]}"


class WordTokenizer:
    def __init__(self, tokens: list[str]):
        self.tokens = ["<pad>", "<bos>", "<eos>"] + sorted(set(tokens))
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(idx)] for idx in ids]


def build_modular_word_tokenizer(modulus: int, *, extra_tokens: list[str] | None = None) -> WordTokenizer:
    tokens = ["what", "is", "plus", "modulo", "?", "answer", "."]
    for value in range(modulus + 1):
        tokens.extend(number_to_words(value).split())
    if extra_tokens is not None:
        tokens.extend(extra_tokens)
    return WordTokenizer(tokens)


def modular_word_prompt_tokens(a: int, b: int, modulus: int) -> list[str]:
    return (
        ["what", "is"]
        + number_to_words(a).split()
        + ["plus"]
        + number_to_words(b).split()
        + ["modulo"]
        + number_to_words(modulus).split()
        + ["?", "answer", "is"]
    )


def modular_word_answer_tokens(answer: int) -> list[str]:
    return number_to_words(answer).split()


def modular_word_sentence(a: int, b: int, modulus: int) -> str:
    answer = (a + b) % modulus
    return " ".join(modular_word_prompt_tokens(a, b, modulus) + modular_word_answer_tokens(answer) + ["."])


class ModularArithmeticDataset(Dataset):
    def __init__(
        self,
        *,
        modulus: int,
        operation: str,
        split: str,
        train_fraction: float,
        seed: int,
    ):
        if operation not in {"add", "sub", "mul"}:
            raise ValueError("operation must be add, sub, or mul")
        pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
        rng = random.Random(seed)
        rng.shuffle(pairs)
        cut = int(len(pairs) * train_fraction)
        if split == "train":
            self.pairs = pairs[:cut]
        elif split in {"val", "test"}:
            self.pairs = pairs[cut:]
        else:
            raise ValueError(f"unknown split: {split}")
        self.modulus = modulus
        self.operation = operation
        self.op_token = modulus
        self.eq_token = modulus + 1

    def __len__(self) -> int:
        return len(self.pairs)

    def _answer(self, a: int, b: int) -> int:
        if self.operation == "add":
            return (a + b) % self.modulus
        if self.operation == "sub":
            return (a - b) % self.modulus
        return (a * b) % self.modulus

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, b = self.pairs[idx]
        input_ids = torch.tensor([a, self.op_token, b, self.eq_token], dtype=torch.long)
        label = torch.tensor(self._answer(a, b), dtype=torch.long)
        return {"input_ids": input_ids, "labels": label}


class ModularExpressionDataset(ModularArithmeticDataset):
    """Full modular expressions for JEPA-style answer-latent prediction."""

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, b = self.pairs[idx]
        answer = self._answer(a, b)
        input_ids = torch.tensor([a, self.op_token, b, self.eq_token, answer], dtype=torch.long)
        label = torch.tensor(answer, dtype=torch.long)
        return {"input_ids": input_ids, "labels": label}


class ModularWordSumLMDataset(Dataset):
    """Word-form modular-addition sentences for decoder-only LM pretraining."""

    def __init__(
        self,
        *,
        modulus: int,
        split: str,
        train_fraction: float,
        seed: int,
        tokenizer: WordTokenizer,
        sequence_length: int,
        loss_profile: str = "all_tokens",
        prompt_loss_weight: float = 1.0,
        answer_loss_weight: float = 1.0,
        period_loss_weight: float = 1.0,
        eos_loss_weight: float = 1.0,
    ):
        pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
        rng = random.Random(seed)
        rng.shuffle(pairs)
        cut = int(len(pairs) * train_fraction)
        if split == "train":
            self.pairs = pairs[:cut]
        elif split in {"val", "test"}:
            self.pairs = pairs[cut:]
        else:
            raise ValueError(f"unknown split: {split}")
        self.modulus = modulus
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.loss_profile = loss_profile
        self.prompt_loss_weight = float(prompt_loss_weight)
        self.answer_loss_weight = float(answer_loss_weight)
        self.period_loss_weight = float(period_loss_weight)
        self.eos_loss_weight = float(eos_loss_weight)

    def __len__(self) -> int:
        return len(self.pairs)

    def _answer(self, a: int, b: int) -> int:
        return (a + b) % self.modulus

    def prompt_tokens(self, idx: int) -> list[str]:
        a, b = self.pairs[idx]
        return modular_word_prompt_tokens(a, b, self.modulus)

    def answer_tokens(self, idx: int) -> list[str]:
        a, b = self.pairs[idx]
        return modular_word_answer_tokens(self._answer(a, b))

    def sentence(self, idx: int) -> str:
        a, b = self.pairs[idx]
        return modular_word_sentence(a, b, self.modulus)

    def prompt_ids(self, idx: int) -> list[int]:
        return [self.tokenizer.bos_token_id] + self.tokenizer.encode(self.prompt_tokens(idx))

    def answer_ids(self, idx: int) -> list[int]:
        return self.tokenizer.encode(self.answer_tokens(idx))

    def _full_ids(self, idx: int) -> list[int]:
        return (
            [self.tokenizer.bos_token_id]
            + self.tokenizer.encode(self.prompt_tokens(idx))
            + self.tokenizer.encode(self.answer_tokens(idx) + ["."])
            + [self.tokenizer.eos_token_id]
        )

    def _loss_weights(self, idx: int) -> list[float]:
        prompt_len = len(self.prompt_tokens(idx))
        answer_len = len(self.answer_tokens(idx))
        target_len = prompt_len + answer_len + 2
        if self.loss_profile == "all_tokens":
            return [1.0] * target_len
        if self.loss_profile == "answer_only":
            return [0.0] * prompt_len + [1.0] * answer_len + [0.0, 0.0]
        if self.loss_profile == "answer_and_period":
            return [0.0] * prompt_len + [1.0] * answer_len + [1.0, 0.0]
        if self.loss_profile == "weighted_answer":
            return (
                [self.prompt_loss_weight] * prompt_len
                + [self.answer_loss_weight] * answer_len
                + [self.period_loss_weight, self.eos_loss_weight]
            )
        raise ValueError(f"unknown modular word LM loss_profile: {self.loss_profile}")

    def aux_position(self, idx: int) -> int:
        return len(self.prompt_tokens(idx))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, b = self.pairs[idx]
        ids = self._full_ids(idx)
        input_ids = ids[:-1]
        labels = ids[1:]
        loss_weights = self._loss_weights(idx)
        pad_count = self.sequence_length - len(input_ids)
        if pad_count < 0:
            raise ValueError("sequence_length is too small for modular word sentence")
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_count
        labels = labels + [-100] * pad_count
        loss_weights = loss_weights + [0.0] * pad_count
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(loss_weights, dtype=torch.float32),
            "residue_labels": torch.tensor(self._answer(a, b), dtype=torch.long),
            "aux_positions": torch.tensor(self.aux_position(idx), dtype=torch.long),
        }


class ModularWordSumJEPADataset(Dataset):
    """Word-form modular-addition prompts for JEPA-style residue latents."""

    query_token = "<jepa_query>"

    def __init__(
        self,
        *,
        modulus: int,
        split: str,
        train_fraction: float,
        seed: int,
        tokenizer: WordTokenizer,
        sequence_length: int,
        commute_prob: float = 0.5,
    ):
        pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
        rng = random.Random(seed)
        rng.shuffle(pairs)
        cut = int(len(pairs) * train_fraction)
        if split == "train":
            self.pairs = pairs[:cut]
        elif split in {"val", "test"}:
            self.pairs = pairs[cut:]
        else:
            raise ValueError(f"unknown split: {split}")
        self.modulus = modulus
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.commute_prob = float(commute_prob)
        self.query_token_id = tokenizer.token_to_id[self.query_token]

    def __len__(self) -> int:
        return len(self.pairs)

    def _answer(self, a: int, b: int) -> int:
        return (a + b) % self.modulus

    def _encode_pair(self, a: int, b: int) -> tuple[list[int], list[bool]]:
        ids = (
            [self.tokenizer.bos_token_id]
            + self.tokenizer.encode(modular_word_prompt_tokens(a, b, self.modulus))
            + [self.query_token_id]
        )
        target_mask = [False] * (len(ids) - 1) + [True]
        pad_count = self.sequence_length - len(ids)
        if pad_count < 0:
            raise ValueError("sequence_length is too small for modular word JEPA prompt")
        ids = ids + [self.tokenizer.pad_token_id] * pad_count
        target_mask = target_mask + [False] * pad_count
        return ids, target_mask

    def _equivalent_pair(self, a: int, b: int) -> tuple[int, int]:
        shift = random.randrange(self.modulus)
        new_a = (a + shift) % self.modulus
        new_b = (b - shift) % self.modulus
        if random.random() < self.commute_prob:
            return new_b, new_a
        return new_a, new_b

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, b = self.pairs[idx]
        equiv_a, equiv_b = self._equivalent_pair(a, b)
        input_ids, target_mask = self._encode_pair(a, b)
        equiv_input_ids, equiv_target_mask = self._encode_pair(equiv_a, equiv_b)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(self._answer(a, b), dtype=torch.long),
            "target_mask": torch.tensor(target_mask, dtype=torch.bool),
            "equiv_input_ids": torch.tensor(equiv_input_ids, dtype=torch.long),
            "equiv_target_mask": torch.tensor(equiv_target_mask, dtype=torch.bool),
        }


def modular_word_sequence_length(modulus: int, tokenizer: WordTokenizer) -> int:
    max_len = 0
    for a in range(modulus):
        for b in range(modulus):
            answer = (a + b) % modulus
            tokens = (
                [tokenizer.bos_token_id]
                + tokenizer.encode(modular_word_prompt_tokens(a, b, modulus))
                + tokenizer.encode(modular_word_answer_tokens(answer) + ["."])
                + [tokenizer.eos_token_id]
            )
            max_len = max(max_len, len(tokens) - 1)
    return max_len


def modular_word_jepa_sequence_length(modulus: int, tokenizer: WordTokenizer) -> int:
    query_token_id = tokenizer.token_to_id[ModularWordSumJEPADataset.query_token]
    max_len = 0
    for a in range(modulus):
        for b in range(modulus):
            tokens = (
                [tokenizer.bos_token_id]
                + tokenizer.encode(modular_word_prompt_tokens(a, b, modulus))
                + [query_token_id]
            )
            max_len = max(max_len, len(tokens))
    return max_len


def _modular_bundle(cfg: dict[str, Any]) -> DatasetBundle:
    modulus = int(cfg.get("modulus", 67))
    operation = cfg.get("operation", "add")
    train_fraction = float(cfg.get("train_fraction", 0.4))
    seed = int(cfg.get("seed", 0))
    train = ModularArithmeticDataset(
        modulus=modulus,
        operation=operation,
        split="train",
        train_fraction=train_fraction,
        seed=seed,
    )
    val = ModularArithmeticDataset(
        modulus=modulus,
        operation=operation,
        split="val",
        train_fraction=train_fraction,
        seed=seed,
    )
    return DatasetBundle(
        train,
        val,
        val,
        TaskSpec(objective="classification", vocab_size=modulus + 2, num_classes=modulus),
    )


def _modular_jepa_bundle(cfg: dict[str, Any]) -> DatasetBundle:
    modulus = int(cfg.get("modulus", 67))
    operation = cfg.get("operation", "add")
    train_fraction = float(cfg.get("train_fraction", 0.4))
    seed = int(cfg.get("seed", 0))
    train = ModularExpressionDataset(
        modulus=modulus,
        operation=operation,
        split="train",
        train_fraction=train_fraction,
        seed=seed,
    )
    val = ModularExpressionDataset(
        modulus=modulus,
        operation=operation,
        split="val",
        train_fraction=train_fraction,
        seed=seed,
    )
    return DatasetBundle(
        train,
        val,
        val,
        TaskSpec(objective="jepa", vocab_size=modulus + 3, num_classes=modulus),
    )


def _modular_word_jepa_bundle(cfg: dict[str, Any]) -> DatasetBundle:
    modulus = int(cfg.get("modulus", 97))
    operation = cfg.get("operation", "add")
    if operation != "add":
        raise ValueError("modular_word_sum_jepa currently supports operation=add")
    train_fraction = float(cfg.get("train_fraction", 0.3))
    seed = int(cfg.get("seed", 0))
    tokenizer = build_modular_word_tokenizer(modulus, extra_tokens=[ModularWordSumJEPADataset.query_token])
    sequence_length = int(cfg.get("sequence_length", modular_word_jepa_sequence_length(modulus, tokenizer)))
    commute_prob = float(cfg.get("commute_prob", 0.5))
    train = ModularWordSumJEPADataset(
        modulus=modulus,
        split="train",
        train_fraction=train_fraction,
        seed=seed,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        commute_prob=commute_prob,
    )
    val = ModularWordSumJEPADataset(
        modulus=modulus,
        split="val",
        train_fraction=train_fraction,
        seed=seed,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        commute_prob=commute_prob,
    )
    return DatasetBundle(
        train,
        val,
        val,
        TaskSpec(
            objective="jepa",
            vocab_size=tokenizer.vocab_size + 1,
            num_classes=modulus,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )


def _modular_word_lm_bundle(cfg: dict[str, Any]) -> DatasetBundle:
    modulus = int(cfg.get("modulus", 97))
    operation = cfg.get("operation", "add")
    if operation != "add":
        raise ValueError("modular_word_sum_lm currently supports operation=add")
    train_fraction = float(cfg.get("train_fraction", 0.3))
    seed = int(cfg.get("seed", 0))
    tokenizer = build_modular_word_tokenizer(modulus)
    sequence_length = int(cfg.get("sequence_length", modular_word_sequence_length(modulus, tokenizer)))
    loss_profile = cfg.get("loss_profile", "all_tokens")
    prompt_loss_weight = float(cfg.get("prompt_loss_weight", 1.0))
    answer_loss_weight = float(cfg.get("answer_loss_weight", 1.0))
    period_loss_weight = float(cfg.get("period_loss_weight", 1.0))
    eos_loss_weight = float(cfg.get("eos_loss_weight", 1.0))
    auxiliary_residue = bool(cfg.get("auxiliary_residue", False))
    train = ModularWordSumLMDataset(
        modulus=modulus,
        split="train",
        train_fraction=train_fraction,
        seed=seed,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        loss_profile=loss_profile,
        prompt_loss_weight=prompt_loss_weight,
        answer_loss_weight=answer_loss_weight,
        period_loss_weight=period_loss_weight,
        eos_loss_weight=eos_loss_weight,
    )
    val = ModularWordSumLMDataset(
        modulus=modulus,
        split="val",
        train_fraction=train_fraction,
        seed=seed,
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        loss_profile=loss_profile,
        prompt_loss_weight=prompt_loss_weight,
        answer_loss_weight=answer_loss_weight,
        period_loss_weight=period_loss_weight,
        eos_loss_weight=eos_loss_weight,
    )
    return DatasetBundle(
        train,
        val,
        val,
        TaskSpec(
            objective="lm",
            vocab_size=tokenizer.vocab_size,
            num_classes=modulus if auxiliary_residue else None,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )


def build_datasets(cfg: dict[str, Any]) -> DatasetBundle:
    name = cfg.get("name", "modular_arithmetic")
    if name == "modular_arithmetic":
        return _modular_bundle(cfg)
    if name == "modular_arithmetic_jepa":
        return _modular_jepa_bundle(cfg)
    if name == "modular_word_sum_jepa":
        return _modular_word_jepa_bundle(cfg)
    if name == "modular_word_sum_lm":
        return _modular_word_lm_bundle(cfg)
    raise ValueError(f"Unsupported dataset for modular-only branch: {name}")
