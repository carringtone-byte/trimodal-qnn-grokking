# Tri-Modal Rigorous Probe Sweep

Run date: 2026-06-17

This report records the stricter sampled probe audit requested after the
checkpoint-dynamics pass. The run analyzes every numeric checkpoint from:

```text
tri_modal_modular_grokking/runs/phase4_full_crossmodal
```

Outputs are under:

```text
tri_modal_modular_grokking/analysis/phase4_rigorous_probes
```

## Protocol

The goal was to probe the answer-slot representation without letting pair
identity leak across train, validation, and test sets.

| item | value |
| --- | ---: |
| sampled operand pairs | 927 |
| sampled answer-slot states | 25,029 |
| task cells per pair | 27 |
| split seeds | 5 |
| ridge penalty grid | 7 values |
| numeric checkpoints | 20 |
| analyzed layer-units | 40 |
| global probe rows | 1,600 |
| cross-cell transfer rows | 145,800 |
| elapsed wall time | 683.21 sec |

For every split seed, the sampled training-side pairs were split into:

| split | pairs |
| --- | ---: |
| ridge train | 181 |
| ridge validation | 97 |
| held-out test | 649 |

The validation split keeps at least one training pair per residue bucket when
possible, then uses the validation set only to choose the ridge penalty. The
test set is made only of originally held-out operand pairs and is disjoint from
both ridge train and validation pairs.

Analysis schedule:

- final answer slot for all 20 numeric checkpoints;
- all backbone layers plus final answer slot for transition checkpoints
  `10000`, `11000`, `12000`, `13000`, and `20000`;
- global ridge probes for `s`, `a`, `b`, `wrap`, `mode_a`, `mode_b`,
  `output_mode`, and `cell_id`;
- permutation controls for the `s` probe;
- full `27 x 27` source-cell to target-cell residue-probe transfer for every
  analyzed layer-unit.

## Main Result

The rigorous probe sweep confirms the checkpoint-dynamics story with stricter
pair-disjoint train/validation/test splits.

| event | first final-slot checkpoint | value |
| --- | ---: | ---: |
| final-slot `s` probe mean >= 0.90 | 12,000 | 0.946105 |
| cross-cell `s` transfer mean >= 0.90 | 12,000 | 0.906745 |

At the final checkpoint, the answer slot is almost purely a sum-residue state:

| target | final test mean |
| --- | ---: |
| `s` | 0.998277 |
| `a` | 0.001301 |
| `b` | 0.002214 |
| `wrap` | 0.837334 |
| `mode_a` | 0.694664 |
| `mode_b` | 0.716065 |
| `output_mode` | 0.679427 |
| `cell_id` | 0.455721 |

The `s` permutation control remains at chance:

| item | value |
| --- | ---: |
| chance `1 / 97` | 0.010309 |
| final `s` permutation control | 0.010272 |

This strongly argues that the `s` probe is reading real sum structure rather
than an artifact of class imbalance or the probe fitting procedure.

## Multi-Seed Follow-Up

The strict probe protocol was repeated after the seed-`1706` robustness anomaly.
Seed `1705` was run with the same schedule as seed `1704`; seed `1706` was run
with an expanded all-layer schedule for every saved late checkpoint from
`10000` through `20000`.

Artifacts:

```text
RESULTS_TRI_MODAL_SEED1706_25K_PROBES.md
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1705/
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1706_expanded/
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_multiseed/
```

Key comparison:

| seed | first final `s` >= 0.9 | first transfer >= 0.9 | best final `s` | best transfer | final `s` | final transfer | final layer-2 `s` | final layer-3 `s` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1704 | 12000 | 12000 | 0.998277 | 0.971721 | 0.998277 | 0.958581 | 0.233305 | 0.998174 |
| 1705 | 12000 | 12000 | 0.997740 | 0.983691 | 0.995811 | 0.983691 | 0.233750 | 0.995971 |
| 1706 | none | none | 0.834469 | 0.843850 | 0.815431 | 0.840452 | 0.130651 | 0.815306 |

Seed `1705` is a clean replication of the strict seed-`1704` result. It crosses
both strict final-slot thresholds at step `12000`, and the final transfer mean
is `0.983691`.

Seed `1706` does not merely lose the layer-2 operand-value precursor. Under the
same strict pair-disjoint protocol, its final answer slot also remains below
the strong-seed shared-representation regime. It peaks at final-slot `s`
accuracy `0.834469` at step `13000`, peaks at final-slot cross-cell transfer
`0.843850` at step `18000`, and finishes at `0.815431` / `0.840452`.

Expanded seed-`1706` late-layer focus:

| step | layer-2 `s` | layer-3 `s` | final `s` | final transfer | sampled held-out |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 0.129270 | 0.772505 | 0.772334 | 0.741093 | 0.970581 |
| 12000 | 0.133322 | 0.815294 | 0.812555 | 0.774191 | 0.980957 |
| 13000 | 0.125823 | 0.836512 | 0.834469 | 0.790755 | 0.980469 |
| 15000 | 0.126885 | 0.824562 | 0.824722 | 0.824614 | 0.986450 |
| 16000 | 0.127900 | 0.817702 | 0.818947 | 0.828826 | 0.972290 |
| 18000 | 0.129259 | 0.823946 | 0.822964 | 0.843850 | 0.992188 |
| 20000 | 0.130651 | 0.815306 | 0.815431 | 0.840452 | 0.985596 |

This makes seed `1706` mechanistically important: high task accuracy does not
guarantee the clean strict linear answer-slot geometry observed in the fully
grokked seeds. The likely interpretation is a less linearly organized or more
cell-entangled solution.

## Checkpoint Trajectory

| step | held-out acc | held-out loss | final `s` probe | perm control | cross-cell transfer | transfer min |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.005493 | 3.543972 | 0.005878 | 0.010033 | 0.009718 | 0.000000 |
| 5,000 | 0.238525 | 2.787332 | 0.168316 | 0.010695 | 0.086986 | 0.000000 |
| 9,000 | 0.336792 | 2.834251 | 0.299663 | 0.012178 | 0.136397 | 0.000000 |
| 10,000 | 0.341064 | 2.732560 | 0.308669 | 0.010238 | 0.137520 | 0.000000 |
| 11,000 | 0.886353 | 0.280869 | 0.845426 | 0.010763 | 0.800614 | 0.528505 |
| 12,000 | 0.957886 | 0.107901 | 0.946105 | 0.010706 | 0.906745 | 0.705701 |
| 13,000 | 0.984985 | 0.036757 | 0.970941 | 0.008092 | 0.932001 | 0.744222 |
| 18,000 | 1.000000 | 0.000956 | 0.997398 | 0.011516 | 0.966167 | 0.748844 |
| 20,000 | 1.000000 | 0.000242 | 0.998277 | 0.010272 | 0.958581 | 0.567026 |

The representation does not drift gradually from checkpoint 1000 to 20000. It
has a clear transition: weak pre-grokking probes through step 10000, a large
jump at step 11000, and reliable cross-cell transfer by step 12000.

The final minimum transfer is lower than the final mean because a few strict
single-source-cell probes transfer poorly into text-output target cells. The
mean and off-diagonal transfer are still high:

| transfer statistic | value |
| --- | ---: |
| all source-target mean | 0.958581 |
| all source-target min | 0.567026 |
| same-cell mean | 0.997204 |
| off-diagonal mean | 0.957096 |

## Output-Mode Transfer Structure

The strict transfer audit is harsher than the earlier final-checkpoint
cross-modal probe matrix: each source-cell classifier is trained on only 181
sampled train pairs, selected by 97 validation pairs, then tested on 649
held-out pairs in every target cell.

At the final checkpoint, number and image output states transfer almost
perfectly to each other. Text-target states are the main residual weakness,
especially when the source output mode is not text.

| source output | target output | mean transfer | min transfer |
| --- | --- | ---: | ---: |
| `image` | `image` | 0.990531 | 0.944530 |
| `image` | `number` | 0.990413 | 0.947612 |
| `image` | `text` | 0.870433 | 0.567026 |
| `number` | `image` | 0.990785 | 0.939908 |
| `number` | `number` | 0.990569 | 0.941448 |
| `number` | `text` | 0.872491 | 0.577812 |
| `text` | `image` | 0.975472 | 0.864407 |
| `text` | `number` | 0.975666 | 0.862866 |
| `text` | `text` | 0.970869 | 0.821263 |

This is an important refinement of the earlier "near-perfect sharing" claim.
There is a shared residue coordinate, but it is not literally identical across
all requested output formats. Text outputs carry a stronger output-format
component and are the hardest transfer target.

## Layer Localization

The all-layer transition checkpoints show where the sum code becomes readable
and transferable.

| step | layer | `s` probe | cross-cell transfer | output-mode probe |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0 | 0.017371 | 0.012371 | 0.439400 |
| 10,000 | 1 | 0.020567 | 0.016873 | 0.466689 |
| 10,000 | 2 | 0.071015 | 0.046559 | 0.525138 |
| 10,000 | 3 | 0.308429 | 0.134834 | 0.708840 |
| 10,000 | final | 0.308669 | 0.137520 | 0.627313 |
| 11,000 | 0 | 0.044079 | 0.021529 | 0.374936 |
| 11,000 | 1 | 0.054272 | 0.047720 | 0.383108 |
| 11,000 | 2 | 0.152976 | 0.267982 | 0.408960 |
| 11,000 | 3 | 0.845084 | 0.791847 | 0.657696 |
| 11,000 | final | 0.845426 | 0.800614 | 0.657912 |
| 12,000 | 0 | 0.047241 | 0.024682 | 0.371261 |
| 12,000 | 1 | 0.065354 | 0.059486 | 0.380403 |
| 12,000 | 2 | 0.174034 | 0.342237 | 0.404588 |
| 12,000 | 3 | 0.945980 | 0.897463 | 0.656965 |
| 12,000 | final | 0.946105 | 0.906745 | 0.653027 |
| 13,000 | 0 | 0.049752 | 0.027253 | 0.369434 |
| 13,000 | 1 | 0.070319 | 0.068811 | 0.383176 |
| 13,000 | 2 | 0.187502 | 0.360606 | 0.397466 |
| 13,000 | 3 | 0.969343 | 0.920955 | 0.659259 |
| 13,000 | final | 0.970941 | 0.932001 | 0.654431 |
| 20,000 | 0 | 0.056006 | 0.019217 | 0.505313 |
| 20,000 | 1 | 0.079153 | 0.071398 | 0.457490 |
| 20,000 | 2 | 0.233305 | 0.553696 | 0.427130 |
| 20,000 | 3 | 0.998174 | 0.932531 | 0.677327 |
| 20,000 | final | 0.998277 | 0.958581 | 0.679427 |

The answer code is a late-layer phenomenon. Layers 0 and 1 remain weak even at
the final checkpoint. Layer 2 contains a growing but incomplete shared sum
signal. Layer 3 is where the representation becomes nearly final.

This agrees with the activation-patching dynamics: by step 20000, layers 2, 3,
and the final answer slot can causally transfer the answer under full-state
patching, while early layers remain weak.

## Hypothesis Adjudication

### H1: One Shared Cyclic/Fourier Rule

Status: strongly supported for seeds `1704` and `1705`; weaker for seed `1706`.

The strict probes agree with the previous Fourier, centroid, and patching
evidence. A final source-cell residue classifier transfers across all target
cells with mean accuracy `0.958581`, and the off-diagonal mean is `0.957096`
despite the small source-cell training set.

The seed-`1705` repeat confirms this result. Seed `1706` does not: it has
substantial above-chance final-slot `s` structure, but it does not reach the
strict `0.9` threshold and should be treated as a separate, less clean
mechanistic regime.

### H2: Three Modality-Specific Cyclic Rules

Status: disfavored for seeds `1704` and `1705`; still plausible as a partial
description of seed `1706`'s weaker/cell-entangled solution.

Three unrelated cyclic rules would predict poor source-cell to target-cell
transfer under strict pair splits. Instead, cross-cell transfer crosses `0.90`
by step 12000 and remains high at the final checkpoint.

The remaining nuance is that text-target states are not identical to number and
image target states. The evidence supports a shared late residue basis with
format-specific components, not a perfectly modality-blind scalar code.

### H3: Translation-To-Number Strategy

Status: answered by later operand-path and robustness tests.

The probes show that the final answer slot is shared and residue-dominant, but
they do not identify the operand-level algorithm. A model could still translate
text and image operands into a number-like internal code before applying a
shared cyclic rule. Dedicated operand-state alignment and replacement tests are
reported in `RESULTS_TRI_MODAL_OPERAND_PATH.md` and
`RESULTS_TRI_MODAL_OPERAND_VALUE_ROBUSTNESS.md`.

### H4: No True Shared Rule

Status: ruled out for seed `1704`; also ruled out for seed `1705` by the repeat
probe. Seed `1706` is not a clean no-rule case, but it does not meet the
strong-seed shared-linear-geometry standard.

The final slot has near-perfect held-out sum decodability, chance-level
permutation controls, high cross-cell transfer, high Fourier addition-diagonal
energy in the separate analysis, and causal full-state answer-slot patching.

## Audit Artifacts

```text
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/probe_manifest.json
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/probe_summary.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/probe_summary.json
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/global_probe_results.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/cross_cell_transfer.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/RIGOROUS_PROBE_REPORT.md
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1705/probe_summary.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1706_expanded/probe_summary.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_multiseed/LINEAR_PROBE_SEED_COMPARISON.md
```

## Reproduction

```powershell
cd modular_addition_mech_interp
python -m tri_modal_modular_grokking.rigorous_probes --run-dir tri_modal_modular_grokking/runs/phase4_full_crossmodal --out-dir tri_modal_modular_grokking/analysis/phase4_rigorous_probes --sample-pairs 927 --seeds 0,1,2,3,4 --lambdas 0.0001,0.001,0.01,0.1,1,10,100 --transition-steps 10000,11000,12000,13000,20000 --batch-size 4096
python -m tri_modal_modular_grokking.rigorous_probes --run-dir tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1705 --out-dir tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1705 --sample-pairs 927 --seeds 0,1,2,3,4 --lambdas 0.0001,0.001,0.01,0.1,1,10,100 --transition-steps 10000,11000,12000,13000,20000 --batch-size 4096
python -m tri_modal_modular_grokking.rigorous_probes --run-dir tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1706 --out-dir tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1706_expanded --sample-pairs 927 --seeds 0,1,2,3,4 --lambdas 0.0001,0.001,0.01,0.1,1,10,100 --transition-steps 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000 --batch-size 4096
```
