# Smoke-Test Record

Date: 2026-08-06

Environment: Python 3.10, PyTorch, Windows, automatic device selection.

## Unit Tests

| Scope | Result |
| --- | ---: |
| Tri-Modal QNN plus classical Tri-Modal data/Fourier/model tests | 32 passed |
| Modular QNN checkpoint and initialization tests | 14 passed |

Total: **46 passed**. PyTorch emitted a non-blocking deprecation warning for
the installed `pynvml` package.

## End-to-End Checks

| Check | Steps | Result |
| --- | ---: | --- |
| Tri-Modal QNN `three_sector` | 20 | completed; finite losses and unit state norm |
| Tri-Modal QNN `ordered_route` | 20 | completed; finite losses and unit state norm |
| Classical Tri-Modal training | 3 | completed; run artifacts written |

The smoke accuracies are intentionally not treated as research results. Their
purpose is to verify that the selected scripts execute outside the original
large monorepo.
