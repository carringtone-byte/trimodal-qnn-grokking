from __future__ import annotations

import numpy as np


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
    value = int(value)
    if value < 0 or value >= 1000:
        raise ValueError("number_to_words supports values in [0, 999]")
    if value < 20:
        return ONES[value]
    if value < 100:
        ten = value // 10 * 10
        rem = value % 10
        return TENS[ten] if rem == 0 else f"{TENS[ten]} {ONES[rem]}"
    hundred = value // 100
    rem = value % 100
    return f"{ONES[hundred]} hundred" if rem == 0 else f"{ONES[hundred]} hundred {number_to_words(rem)}"


def tokenize_pair(a: int, b: int) -> list[str]:
    return ["<bos>", *number_to_words(a).split(), "plus", *number_to_words(b).split(), "<eos>"]


def tokenize_value(value: int) -> list[str]:
    return ["<bos>", *number_to_words(value).split(), "<eos>"]


def build_vocab(modulus: int) -> dict[str, int]:
    tokens = {"<pad>", "<bos>", "<eos>", "plus"}
    for value in range(modulus):
        tokens.update(number_to_words(value).split())
    return {token: idx for idx, token in enumerate(sorted(tokens))}


def tokens_to_ids(tokens: list[str], vocab: dict[str, int], width: int) -> list[int]:
    pad = vocab["<pad>"]
    ids = [vocab[token] for token in tokens]
    if len(ids) > width:
        raise ValueError(f"token sequence length {len(ids)} exceeds width {width}")
    return ids + [pad] * (width - len(ids))


SEGMENTS: dict[str, tuple[str, ...]] = {
    "0": ("a", "b", "c", "d", "e", "f"),
    "1": ("b", "c"),
    "2": ("a", "b", "g", "e", "d"),
    "3": ("a", "b", "c", "d", "g"),
    "4": ("f", "g", "b", "c"),
    "5": ("a", "f", "g", "c", "d"),
    "6": ("a", "f", "e", "d", "c", "g"),
    "7": ("a", "b", "c"),
    "8": ("a", "b", "c", "d", "e", "f", "g"),
    "9": ("a", "b", "c", "d", "f", "g"),
}


def _draw_digit(canvas: np.ndarray, x0: int, y0: int, digit: str) -> None:
    segs = SEGMENTS[digit]
    # Seven-segment glyph in a 7x5 box.
    if "a" in segs:
        canvas[y0, x0 + 1 : x0 + 4] = 1.0
    if "g" in segs:
        canvas[y0 + 3, x0 + 1 : x0 + 4] = 1.0
    if "d" in segs:
        canvas[y0 + 6, x0 + 1 : x0 + 4] = 1.0
    if "f" in segs:
        canvas[y0 + 1 : y0 + 3, x0] = 1.0
    if "b" in segs:
        canvas[y0 + 1 : y0 + 3, x0 + 4] = 1.0
    if "e" in segs:
        canvas[y0 + 4 : y0 + 6, x0] = 1.0
    if "c" in segs:
        canvas[y0 + 4 : y0 + 6, x0 + 4] = 1.0


def _draw_number(canvas: np.ndarray, x0: int, y0: int, value: int) -> int:
    cursor = x0
    for char in str(int(value)):
        _draw_digit(canvas, cursor, y0, char)
        cursor += 6
    return cursor


def render_value_image(value: int, *, height: int = 9, width: int = 18) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.float32)
    _draw_number(canvas, 1, 1, value)
    return canvas[None, :, :]


def render_pair_image(a: int, b: int, *, height: int = 9, width: int = 42) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.float32)
    cursor = _draw_number(canvas, 1, 1, a) + 2
    # Plus sign.
    canvas[4, cursor : cursor + 5] = 1.0
    canvas[2:7, cursor + 2] = 1.0
    _draw_number(canvas, cursor + 7, 1, b)
    return canvas[None, :, :]
