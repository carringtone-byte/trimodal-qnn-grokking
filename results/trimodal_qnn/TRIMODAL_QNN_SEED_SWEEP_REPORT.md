# Strict Dirac-Mean Seed Sweep Report

Run date: 2026-07-07

This report summarizes the strict trimodal QNN seed robustness run for the
three-sector `T/N/I` architecture. Seeds `9302-9305` were trained for exactly
`2000` steps and compared against the original strict Dirac-mean seed `9301`
at its step-`2000` checkpoint.

## Scope

All seeds use the same data split, architecture, and loss family:

| item | value |
| --- | --- |
| task | `(a + b) mod 97` |
| modalities/sectors | text `T`, number `N`, image `I` |
| split | fixed 30% pair train split, 70% pair held-out split |
| problem mode | `three_sector` |
| qubits | `7` |
| circuit layers | `4` |
| state dimension per sector | `128` |
| sector count | `3` |
| parameter count | `121944` |
| head | `layerwise_dirac_mean` |
| Fourier/Dirac frequencies | `k <= 21` |
| training steps | `2000` per new seed |
| eval/checkpoint cadence | `1000` steps |

The training objective is:

```text
L = CE
  + 0.001 * same_sum_KL
  + 0.50  * layerwise_Dirac_CE
  + 0.10  * hard_neighbor_margin
```

The same mechanistic analysis suite was run for every new seed:

- all-pair checkpoint diagnostics and layerwise Dirac logit lens;
- Fourier cutoff readouts;
- sector-mask ablations;
- sector-path ablations;
- frequency-band activation patching;
- sector/subspace CCA;
- single-sector residue probes.

## Artifacts

Pipeline summary:

```text
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_pipeline_summary.csv
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_pipeline_summary.json
```

Per-seed analysis directories:

```text
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9302_step_2000/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9302_step_2000_causal/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9302_step_2000_sector_cca/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9303_step_2000/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9303_step_2000_causal/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9303_step_2000_sector_cca/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9304_step_2000/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9304_step_2000_causal/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9304_step_2000_sector_cca/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9305_step_2000/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9305_step_2000_causal/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_9305_step_2000_sector_cca/
```

The original seed comparison artifacts are:

```text
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_causal/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_sector_cca/
```

## Behavioral Result

All four new seeds reproduced the high-accuracy step-`2000` solution. The
original seed `9301` was not an outlier or lucky run.

| seed | held-out acc | cross-ablate acc | drop | train acc | held-out loss | same-sum ratio | Fourier energy | top freq |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9301 | 0.912542 | 0.794716 | 0.117826 | 1.000000 | 0.506894 | 0.570536 | 0.789818 | 4 |
| 9302 | 0.941543 | 0.781203 | 0.160340 | 1.000000 | 0.501153 | 0.632449 | 0.792311 | 5 |
| 9303 | 0.938961 | 0.820984 | 0.117978 | 0.999646 | 0.481306 | 0.516911 | 0.818322 | 6 |
| 9304 | 0.910112 | 0.730489 | 0.179623 | 1.000000 | 0.554539 | 0.614916 | 0.778413 | 5 |
| 9305 | 0.914212 | 0.682964 | 0.231248 | 1.000000 | 0.591758 | 0.549469 | 0.823932 | 5 |

Aggregate:

| population | mean held-out | sd | min | max |
| --- | ---: | ---: | ---: | ---: |
| new seeds `9302-9305` | 0.926207 | 0.014149 | 0.910112 | 0.941543 |
| all seeds `9301-9305` | 0.923474 | 0.013785 | 0.910112 | 0.941543 |

Cross-sector ablation is more variable:

| population | mean cross-ablate | sd | min | max |
| --- | ---: | ---: | ---: | ---: |
| new seeds `9302-9305` | 0.753910 | 0.052023 | 0.682964 | 0.820984 |

Interpretation: the high-accuracy solution is seed-stable, while the degree of
reliance on cross-sector interaction is seed-variable.

## Training Dynamics

All seeds start at chance, partially grok by step `1000`, and become strong by
step `2000`.

| seed | held-out step 1 | held-out step 1000 | held-out step 2000 | cross step 1000 | cross step 2000 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9301 | 0.009566 | 0.644094 | 0.912542 | 0.563012 | 0.794716 |
| 9302 | 0.011084 | 0.672791 | 0.941543 | 0.522776 | 0.781203 |
| 9303 | 0.010325 | 0.702855 | 0.938961 | 0.611449 | 0.820984 |
| 9304 | 0.010325 | 0.571819 | 0.910112 | 0.429092 | 0.730489 |
| 9305 | 0.011236 | 0.636654 | 0.914212 | 0.448983 | 0.682964 |

Seed `9303` is the strongest cross-ablation seed; seed `9305` is the most
cross-sector dependent.

## Layerwise Emergence

The layerwise Dirac logit lens shows the cyclic answer becoming readable inside
the circuit rather than appearing only in the final readout.

| seed | initial | layer 1 | layer 2 | layer 3 | layer 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9301 | 0.010932 | 0.488005 | 0.749013 | 0.898725 | 0.925600 |
| 9302 | 0.012754 | 0.408442 | 0.800182 | 0.901761 | 0.940480 |
| 9303 | 0.009718 | 0.496963 | 0.815366 | 0.910416 | 0.937899 |
| 9304 | 0.008047 | 0.521257 | 0.760705 | 0.863954 | 0.907987 |
| 9305 | 0.009262 | 0.457182 | 0.690859 | 0.866535 | 0.917704 |

Interpretation: layer 2 is already strongly cyclic, and layers 3-4 sharpen the
Dirac/Fourier decision.

## Fourier Cutoff Diagnostics

Low/mid frequencies carry most of the answer, but the full `k<=21` basis is
needed for the final accuracy.

| seed | k1 | k2 | k3 | k5 | k8 | k13 | k21 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9301 | 0.072730 | 0.160947 | 0.270878 | 0.463559 | 0.595809 | 0.837990 | 0.912542 |
| 9302 | 0.070149 | 0.162466 | 0.281051 | 0.477073 | 0.721379 | 0.876860 | 0.941543 |
| 9303 | 0.076374 | 0.156696 | 0.269207 | 0.515943 | 0.677650 | 0.854236 | 0.938961 |
| 9304 | 0.079411 | 0.163073 | 0.241269 | 0.431825 | 0.613726 | 0.843759 | 0.910112 |
| 9305 | 0.075615 | 0.102794 | 0.264804 | 0.494078 | 0.656089 | 0.861828 | 0.914212 |

All-five mean `k<=13` held-out accuracy is `0.854935`, with range
`0.837990-0.876860`.

## Frequency-Causal Patching

Frequency-band activation patching confirms that the answer is not just
correlationally Fourier-like; the low/mid bands are causally involved.

| seed | restore k1-5 | restore k6-13 | restore k1-13 | restore k14-21 | ablate k1-13 drop |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9301 | 0.082144 | 0.204904 | 0.786403 | 0.011084 | 0.901458 |
| 9302 | 0.105565 | 0.233640 | 0.813658 | 0.015411 | 0.926131 |
| 9303 | 0.109361 | 0.263096 | 0.837420 | 0.012602 | 0.926359 |
| 9304 | 0.096720 | 0.229464 | 0.756263 | 0.019359 | 0.890753 |
| 9305 | 0.146599 | 0.185621 | 0.790882 | 0.019891 | 0.894321 |

All-five means:

| metric | mean |
| --- | ---: |
| restore `k=1..13` | 0.796925 |
| ablate `k=1..13` accuracy drop | 0.907804 |
| restore `k=14..21` | approximately 0.016 |
| ablate `k=14..21` accuracy drop | 0.127885 |

Interpretation: high frequencies help finite-residue sharpening, but the
causal core is low/mid-frequency.

## Sector-Mask Causality

Text is the dominant causally sufficient sector, but not the whole computation.
Number and image alone are near chance; adding either to text improves
performance.

| seed | T | N | I | T+N | T+I | N+I | all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9301 | 0.700577 | 0.031886 | 0.011388 | 0.737625 | 0.806408 | 0.027786 | 0.912542 |
| 9302 | 0.637716 | 0.030216 | 0.012299 | 0.737473 | 0.802612 | 0.024142 | 0.941543 |
| 9303 | 0.686760 | 0.032341 | 0.015032 | 0.756453 | 0.828272 | 0.019739 | 0.938961 |
| 9304 | 0.546918 | 0.038263 | 0.012754 | 0.675068 | 0.787580 | 0.022776 | 0.910112 |
| 9305 | 0.612056 | 0.022472 | 0.011995 | 0.656089 | 0.733222 | 0.042363 | 0.914212 |

All-five means:

| mask | mean accuracy |
| --- | ---: |
| `T` | 0.636805 |
| `T+N` | 0.712542 |
| `T+I` | 0.791619 |
| `all` | 0.923474 |

Interpretation: text is a high-capacity causal route, image is most useful when
paired with text, and number is synergistic rather than standalone in this raw
sector basis.

## Sector-Path Ablations

Path ablations show large early dependence on the text-to-text path and
smaller but nonzero number/image self-path contributions. The final layer is
less fragile than early/middle layers because the answer has already been
formed.

| seed | max T->T drop | layer4 T->T drop | max N->N drop | layer4 N->N drop | max I->I drop | layer4 I->I drop |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9301 | 0.879289 | 0.116763 | 0.106286 | 0.047070 | 0.152141 | 0.033101 |
| 9302 | 0.914060 | 0.143031 | 0.138627 | 0.036745 | 0.153659 | 0.026723 |
| 9303 | 0.916793 | 0.185697 | 0.110082 | 0.043122 | 0.141512 | 0.021257 |
| 9304 | 0.885515 | 0.154570 | 0.122381 | 0.048436 | 0.216520 | 0.043881 |
| 9305 | 0.849681 | 0.185545 | 0.180990 | 0.071060 | 0.209687 | 0.033252 |

## Sector CCA

The text-image CCA artifact is highly seed-stable. Final complex-state CCA is
large for `T-I` and small for `T-N` and `N-I`.

| seed | complex T-N | complex T-I | complex N-I |
| ---: | ---: | ---: | ---: |
| 9301 | 0.024516 | 0.948381 | 0.068821 |
| 9302 | 0.029661 | 0.965314 | 0.060161 |
| 9303 | 0.026994 | 0.956154 | 0.082964 |
| 9304 | 0.034954 | 0.964227 | 0.069143 |
| 9305 | 0.026978 | 0.982927 | 0.077356 |

All-five final complex-state means:

| pair | mean top-10 CCA |
| --- | ---: |
| `T-I` | 0.963401 |
| `T-N` | 0.028621 |
| `N-I` | 0.071689 |

Final probability-state CCA is also text-image dominated:

| seed | prob T-N | prob T-I | prob N-I |
| ---: | ---: | ---: | ---: |
| 9301 | 0.042392 | 0.774359 | 0.015618 |
| 9302 | 0.033005 | 0.823776 | 0.013686 |
| 9303 | 0.036668 | 0.794924 | 0.013310 |
| 9304 | 0.044968 | 0.829492 | 0.022113 |
| 9305 | 0.032035 | 0.890578 | 0.019331 |

All-five final probability-state means:

| pair | mean top-10 CCA |
| --- | ---: |
| `T-I` | 0.822626 |
| `T-N` | 0.037814 |
| `N-I` | 0.016812 |

Interpretation: the text-image binding pattern is not a seed accident. It is a
stable attractor of this strict Dirac-mean three-sector setup.

## Sector Probes

Single-sector residue probes remain modest, especially for number. This is
important because it separates causal sector sufficiency from a simple linear
"answer stored in one sector" story.

Final complex-state single-sector probes:

| seed | T | N | I |
| ---: | ---: | ---: | ---: |
| 9301 | 0.274218 | 0.010629 | 0.188278 |
| 9302 | 0.253416 | 0.008351 | 0.164136 |
| 9303 | 0.265867 | 0.009566 | 0.181142 |
| 9304 | 0.262527 | 0.009262 | 0.212876 |
| 9305 | 0.237625 | 0.008351 | 0.173550 |

Final probability-state single-sector probes:

| seed | T | N | I |
| ---: | ---: | ---: | ---: |
| 9301 | 0.362891 | 0.025964 | 0.145005 |
| 9302 | 0.320377 | 0.025053 | 0.137261 |
| 9303 | 0.319921 | 0.029001 | 0.139538 |
| 9304 | 0.331916 | 0.028545 | 0.187215 |
| 9305 | 0.297297 | 0.027483 | 0.163225 |

## Interpretation

The seed sweep gives strong evidence for a real cyclic/Fourier rule. It is not
just a benchmark win and not just a one-seed artifact. The decisive points are:

- all four new seeds reach high held-out accuracy at step `2000`;
- all seeds show a monotonic layerwise emergence of answer decodability;
- Fourier cutoff readouts show a stable low/mid-frequency scaffold;
- frequency patching shows `k=1..13` is causally important;
- sector masks show text is causally sufficient to a substantial degree but
  not sufficient for full performance;
- CCA shows a stable text-image shared subspace and weak final number
  alignment.

The best current mechanistic description is:

```text
The strict Dirac-mean trimodal QNN learns a seed-stable cyclic/Fourier answer
mechanism, implemented through an asymmetric sector-binding strategy. Text acts
as the dominant binding/readout sector, image aligns strongly to text, and
number contributes causally but is not linearly aligned with the final
text-image code.
```

This refines the original hypothesis space:

| hypothesis | status |
| --- | --- |
| one balanced shared cyclic/Fourier rule | partly supported for the cyclic rule, not supported for balanced sector geometry |
| three independent modality-specific cyclic rules | disfavored by strong `T-I` CCA and cross-sector causal effects |
| translation-to-number strategy | not supported in a simple final-linear-number-coordinate sense |
| no true shared rule | strongly disfavored |
| text-anchored shared cyclic rule | best current description |

## Caveat

Sector-mask accuracy and single-sector probes measure different things. A
single-sector masked forward pass still lets the remaining sector pass through
the circuit and readout. A sector probe asks for linear residue decodability
from a raw sector representation. The fact that `T` mask accuracy is high while
single-sector `T` probes are modest means "text-anchored computation", not
"the answer is linearly stored in text".

## Next Experiments

The highest-value next tests are interventions on the text hub:

1. Train a sector-dropout version where text is randomly removed during
   training.
2. Train a number-anchor regularized version that explicitly encourages
   `T-N` and `N-I` alignment.
3. Train an image/number hub forcing control by penalizing excessive `T-I`
   CCA or text-only sufficiency.
4. Port the strict Dirac-mean objective to ordered-route and all-route
   superposition variants.

The scientific question is now no longer whether the model can learn a cyclic
rule. It can. The question is whether the same architecture can be pushed from
a text-anchored cyclic rule into a genuinely balanced modality-shared cyclic
representation.
