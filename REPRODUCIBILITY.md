# Reproducibility

## Environment

Python 3.10 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Focused Tests

```powershell
python -m pytest -q tests/test_trimodal_qnn_codex.py `
  tests/test_tri_modal_data.py `
  tests/test_tri_modal_fourier.py `
  tests/test_tri_modal_model.py `
  tests/test_qnn_checkpoint_init.py
```

## End-to-End Smoke Tests

The Tri-Modal QNN smoke runs two formulations at modulus 7 for 20 steps:

```powershell
python -m trimodal_qnn_codex.smoke
```

The classical Tri-Modal smoke runs three optimization steps at modulus 17:

```powershell
python -m tri_modal_modular_grokking.train `
  --config tri_modal_modular_grokking/configs/smoke_train.yaml
```

These commands verify data construction, forward and backward passes, unitary
mixing, heads, metrics, evaluation, and checkpoint/summary writing. They are
execution checks, not attempts to reproduce the headline accuracies.

## Full Experiments

The full configurations are included for inspection, but reproducing the
headline runs requires substantially more compute and storage. Start with:

```text
trimodal_qnn_codex/configs/phase1_three_sector_mod97_dirac_mean.yaml
trimodal_qnn_codex/configs/phase1_operand_query_mod97_bitter_lesson_5k.yaml
tri_modal_modular_grokking/configs/phase4_full_crossmodal.yaml
configs/modular_addition_qnn_mod97_same_sum_mean_seed0.yaml
```

Historical checkpoints are not redistributed. The reports identify the exact
checkpoints and follow-up analyses used for each claim.

## Determinism and Limits

Configurations pin data and model seeds where applicable. GPU kernels and
library versions can still introduce small numerical differences. Some
long-running source experiments terminated with nonfinite gradients after a
useful earlier checkpoint; the reports state which checkpoint was selected and
retain that stability limitation.
