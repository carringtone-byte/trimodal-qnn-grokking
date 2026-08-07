# Rigorous Tri-Modal Probe Report

Probe protocol: sampled ~25k answer states, pair-disjoint train/validation/test splits, five split seeds, ridge-classifier lambda sweep, permutation controls, and full 27x27 cross-cell residue transfer.

## Final Checkpoint

| metric | value |
| --- | ---: |
| final-slot `s` probe test mean | 0.998277 |
| final-slot `s` probe test std | 0.000076 |
| final-slot `s` permutation control | 0.010272 |
| cross-cell `s` transfer mean | 0.958581 |
| cross-cell `s` transfer min | 0.567026 |
| output-mode probe test mean | 0.679427 |

## Timing Markers

| event | first final-slot checkpoint | value |
| --- | ---: | ---: |
| `s` probe test mean >= 0.90 | 12000 | 0.9461051106452942 |
| cross-cell `s` transfer mean >= 0.90 | 12000 | 0.9067447856293457 |

## Outputs

```text
probe_manifest.json
probe_summary.csv
global_probe_results.csv
cross_cell_transfer.csv
```
