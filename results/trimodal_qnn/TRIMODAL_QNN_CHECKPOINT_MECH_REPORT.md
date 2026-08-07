# Trimodal QNN Checkpoint Mechanistic Analysis

Generated: 2026-07-10 12:50:13

## Scope

This focused pass evaluates one requested QNN checkpoint (`phase1_operand_query_mod97_bitter_lesson_20k_from_5k` step `20000`) with exact all-pair diagnostics, layerwise logit lenses, Fourier-energy grids, readout ablations, cross-sector ablations, route-local metrics, and sector-mass dynamics.

## Best Checkpoints

| run | best step | held-out acc | Fourier-only held-out | cross-ablate held-out | Fourier energy | same-sum ratio | top frequency | head scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase1_operand_query_mod97_bitter_lesson_20k_from_5k | 20000 | 0.794497 | 0.794497 | 0.010325 | 0.967862 | 0.576598 | 3 | 7.389056 |

## Interpretation

- The exact all-pair metrics test whether the learned QNN state generalizes beyond the strict training-pair split.
- Fourier-only/readout ablations show how much of the solution is carried by the explicit cyclic delta readout rather than any residual class head. For layerwise Dirac-mean checkpoints this equals the full readout because there is no residual class head.
- Cross-sector ablations distinguish genuinely sector-interactive solutions from route-local or single-sector solutions.
- The layerwise logit lens tests when the trained cyclic readouts can decode the sum from intermediate QNN states; this is a stricter formation diagnostic than final accuracy alone.

## Figures

- `figures\checkpoint_heldout_accuracy.png`
- `figures\checkpoint_fourier_energy.png`
- `figures\phase1_operand_query_mo_0d30c0ec_layer_logit_lens.png`

## Tables

- `checkpoint_summary.csv`: one row per checkpoint with exact behavior, ablations, Fourier energy, same-sum ratio, head scale, and frequency concentration.
- `layer_logit_lens.csv`: one row per checkpoint and layer with frozen final-head decode accuracy and Fourier-energy diagnostics.
- `route_summary.csv`: route-local exact behavior and Fourier-energy summaries.
- `frequency_cutoffs.csv`: final-state Fourier cutoff diagnostics.
- `sector_masses.csv`: mean sector mass per checkpoint and layer.
- `failure_events.csv`: nonfinite events parsed from training metrics.

