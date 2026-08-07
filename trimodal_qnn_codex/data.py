from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("T", "N", "I")
ORDERED_ROUTES = tuple(f"{a}{b}" for a in MODALITIES for b in MODALITIES)
ROUTE_PROBLEM_MODES = {"ordered_route", "operand_query"}


@dataclass(frozen=True)
class PairRecord:
    a: int
    b: int
    y: int
    route: str | None = None


def split_pairs(modulus: int, train_fraction: float, seed: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rng = np.random.default_rng(int(seed))
    pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng.shuffle(pairs)
    cut = int(round(len(pairs) * float(train_fraction)))
    return pairs[:cut], pairs[cut:]


def normalize_routes(routes: list[str] | None, *, problem_mode: str) -> list[str] | None:
    if routes is None:
        return None
    allowed = set(ORDERED_ROUTES)
    out = []
    for route in routes:
        route = str(route).upper()
        if problem_mode not in ROUTE_PROBLEM_MODES:
            raise ValueError("routes are only meaningful for route-conditioned datasets")
        if route not in allowed:
            raise ValueError(f"unknown route {route!r}; expected one of {sorted(allowed)}")
        out.append(route)
    return out


class ModularPairsDataset(Dataset):
    """Strict pair-split modular-addition dataset.

    ``three_sector`` examples contain one pair and no active route. The model
    receives coherent text, number, and image whole-expression branches.

    ``ordered_route`` examples contain one active ordered route such as ``TI``.
    The active route says which modality presents operand ``a`` and which
    modality presents operand ``b``.
    """

    def __init__(self, cfg: dict[str, Any], *, split: str):
        self.modulus = int(cfg.get("modulus", 7))
        self.problem_mode = str(cfg.get("problem_mode", "three_sector"))
        if self.problem_mode not in {"three_sector", *ROUTE_PROBLEM_MODES}:
            raise ValueError(f"unknown problem_mode: {self.problem_mode}")
        train_pairs, heldout_pairs = split_pairs(
            self.modulus,
            float(cfg.get("train_fraction", 0.6)),
            int(cfg.get("split_seed", cfg.get("seed", 0))),
        )
        if split == "train":
            pairs = train_pairs
            routes = normalize_routes(cfg.get("train_routes"), problem_mode=self.problem_mode)
        elif split in {"heldout", "val", "test"}:
            pairs = heldout_pairs
            routes = normalize_routes(cfg.get("eval_routes"), problem_mode=self.problem_mode)
        elif split == "all":
            pairs = [(a, b) for a in range(self.modulus) for b in range(self.modulus)]
            routes = normalize_routes(cfg.get("eval_routes"), problem_mode=self.problem_mode)
        else:
            raise ValueError(f"unknown split {split!r}")

        if self.problem_mode in ROUTE_PROBLEM_MODES and routes is None:
            routes = list(ORDERED_ROUTES)

        self.split = "heldout" if split in {"val", "test"} else split
        self.pairs = pairs
        self.routes = routes
        self.records: list[PairRecord] = []
        for a, b in pairs:
            y = (a + b) % self.modulus
            if self.problem_mode in ROUTE_PROBLEM_MODES:
                assert routes is not None
                for route in routes:
                    self.records.append(PairRecord(a=a, b=b, y=y, route=route))
            else:
                self.records.append(PairRecord(a=a, b=b, y=y, route=None))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        route_id = -1 if record.route is None else ORDERED_ROUTES.index(record.route)
        return {
            "a": torch.tensor(record.a, dtype=torch.long),
            "b": torch.tensor(record.b, dtype=torch.long),
            "y": torch.tensor(record.y, dtype=torch.long),
            "route_id": torch.tensor(route_id, dtype=torch.long),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "problem_mode": self.problem_mode,
            "modulus": self.modulus,
            "num_pairs": len(self.pairs),
            "num_records": len(self.records),
            "routes": self.routes,
            "strict_pair_split": True,
        }


def make_datasets(cfg: dict[str, Any]) -> tuple[ModularPairsDataset, ModularPairsDataset]:
    return ModularPairsDataset(cfg, split="train"), ModularPairsDataset(cfg, split="heldout")
