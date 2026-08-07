"""Trimodal QNN modular-addition research harness.

This package implements two related problem formulations:

- ``three_sector``: a coherent superposition over text, number, and image
  whole-expression branches.
- ``ordered_route``: one active ordered operand-route sector among the nine
  input routes ``TT, TN, ..., II``.
"""

from .data import MODALITIES, ORDERED_ROUTES, ModularPairsDataset, make_datasets
from .models import TrimodalQNNModel

__all__ = [
    "MODALITIES",
    "ORDERED_ROUTES",
    "ModularPairsDataset",
    "TrimodalQNNModel",
    "make_datasets",
]
