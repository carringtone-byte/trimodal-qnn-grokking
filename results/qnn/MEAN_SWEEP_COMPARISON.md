# Layerwise Mean Sweep Comparison

The `layerwise_dirac_mean` readout trains a fixed uniform average of all
layerwise Dirac/Fourier logits from the start. It is the baked-in version of
the post-hoc uniform layer averaging that improved the original auxiliary
checkpoint.

## Results

| family | p=31 held-out mean | p=97 held-out mean | p=127 held-out mean |
| --- | ---: | ---: | ---: |
| QNN auxiliary | 0.369490 | 0.941248 | 0.957252 |
| QNN adapter | 0.272412 | 0.921411 | 0.972397 |
| QNN residual learned average | 0.375433 | 0.945549 | 0.964751 |
| QNN fixed mean | 0.394750 | 0.975305 | 0.983438 |
| matched Fourier-delta baseline | 0.022784 | 0.863165 | 0.990966 |
| product-Fourier upper bound | 1.000000 | 1.000000 | 1.000000 |

Per-seed fixed-mean results:

| modulus | seed | held-out exact | train exact | best step |
| ---: | ---: | ---: | ---: | ---: |
| 31 | 0 | 0.383358 | 1.000000 | 2000 |
| 31 | 1 | 0.355126 | 0.996528 | 1750 |
| 31 | 2 | 0.445765 | 1.000000 | 2000 |
| 97 | 0 | 0.973736 | 0.990432 | 2250 |
| 97 | 1 | 0.968271 | 0.986534 | 2250 |
| 97 | 2 | 0.983908 | 0.992913 | 2500 |
| 127 | 0 | 0.989903 | 0.993799 | 2500 |
| 127 | 1 | 0.971039 | 0.984291 | 2750 |
| 127 | 2 | 0.989372 | 0.991319 | 2750 |

## Interpretation

The baked-in uniform average is the strongest QNN variant in the seed/modulus
sweep for `p=97` and `p=127`. It also slightly improves the weak `p=31` result,
but does not remove the small-modulus generalization failure.

This supports the original post-hoc observation: useful arithmetic signal is
distributed across layerwise Dirac/Fourier heads, and forcing the deployed
answer to average those heads during training improves robustness. It does not
support a quantum-advantage claim, because the matched classical Fourier-delta
baseline is near-perfect at `p=127`.
