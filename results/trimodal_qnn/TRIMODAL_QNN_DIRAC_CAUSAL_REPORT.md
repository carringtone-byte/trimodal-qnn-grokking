# Dirac-Mean Causal Mechanistic Analysis

Generated: 2026-07-07 10:32:59
Run: `phase1_three_sector_mod97_dirac_mean`
Checkpoint: `2000`
Checkpoint: `phase1_three_sector_mod97_dirac_mean/checkpoint_2000.pt` (not redistributed)
Elapsed seconds: `56.856`

## Theory And Artifact Context

This causal pass reads the step-2000 strict Dirac-mean checkpoint against the
repository-root equation source `qnn_derivations_numeric_trimodal.pdf`
(`../../../../qnn_derivations_numeric_trimodal.pdf` from this report
directory). The PDF predicts that a successful data-reuploading QNN should
expose a finite cyclic kernel:

```text
ell_c ~ sum_{k=1}^K rho_k cos(2*pi*k*((a+b)-c)/p)
```

and that the trimodal version is mechanistically meaningful only if the
coherent sector state:

```text
sum_{m in {T,N,I}} lambda_m |m>|chi_m(a,b)>
```

uses off-diagonal modality/sector interactions to align text, number, and
image into addition-diagonal phase features. The checkpoint analyzed here was
trained with `head_type: layerwise_dirac_mean`, `K=21`, four layerwise
Dirac/Fourier heads averaged uniformly, no residual class head, fixed equal
sector amplitudes, same-sum KL weight `0.001`, depth-weighted layerwise
Dirac CE weight `0.50`, and hard-neighbor margin weight `0.10`.

This report therefore asks two causal questions:

- Which Fourier coordinates are sufficient and necessary for the finite cyclic
  kernel?
- Which sector paths and sector masks are necessary for that kernel to be
  produced?

## Frequency Patching

Top all-layer single-frequency restorations by margin recovery:

| delta | frequency | clean acc | margin recovery | acc gain |
| ---: | ---: | ---: | ---: | ---: |
| 13 | 4 | 0.000759 | 0.243808 | 0.000759 |
| 13 | 3 | 0.001063 | 0.225020 | 0.001063 |
| 13 | 5 | 0.000152 | 0.175275 | 0.000152 |
| 5 | 5 | 0.000152 | 0.167438 | 0.000152 |
| 5 | 6 | 0.000304 | 0.133576 | 0.000304 |
| 5 | 4 | 0.000000 | 0.120999 | 0.000000 |
| 13 | 2 | 0.000152 | 0.118265 | 0.000152 |
| 1 | 10 | 0.035530 | 0.117569 | 0.017765 |
| 2 | 10 | 0.001974 | 0.116811 | 0.001367 |
| 2 | 5 | 0.001063 | 0.113482 | 0.000456 |

Top single-layer frequency necessities by clean-margin drop:

| delta | layer | frequency | acc drop | margin drop |
| ---: | --- | ---: | ---: | ---: |
| 13 | layer_4 | 4 | 0.016854 | 4.521548 |
| 13 | layer_3 | 4 | 0.015032 | 4.337320 |
| 13 | layer_4 | 3 | 0.021713 | 3.725931 |
| 13 | layer_3 | 3 | 0.020346 | 3.676196 |
| 13 | layer_2 | 4 | 0.011843 | 3.485427 |
| 13 | layer_4 | 5 | 0.020498 | 3.344216 |
| 13 | layer_2 | 3 | 0.019283 | 3.152624 |
| 13 | layer_3 | 5 | 0.016854 | 3.019978 |
| 13 | layer_2 | 5 | 0.010932 | 2.398223 |
| 5 | layer_4 | 5 | 0.090495 | 2.370838 |

Frequency-band all-layer restorations:

| delta | band | clean acc | margin recovery | acc gain |
| ---: | --- | ---: | ---: | ---: |
| 1 | full_1_21 | 0.912542 | 1.000000 | 0.894777 |
| 2 | full_1_21 | 0.912542 | 1.000000 | 0.911934 |
| 5 | full_1_21 | 0.912542 | 1.000000 | 0.912542 |
| 13 | full_1_21 | 0.912542 | 1.000000 | 0.912542 |
| 5 | lowmid_1_13 | 0.833890 | 0.986036 | 0.833890 |
| 1 | lowmid_1_13 | 0.815062 | 0.876702 | 0.797297 |
| 13 | lowmid_1_13 | 0.789250 | 0.991324 | 0.789250 |
| 2 | lowmid_1_13 | 0.707410 | 0.898275 | 0.706802 |
| 1 | mid_6_13 | 0.509110 | 0.641478 | 0.491345 |
| 13 | low_1_5 | 0.232463 | 0.795997 | 0.232463 |
| 13 | core_3_6 | 0.181445 | 0.692251 | 0.181445 |
| 2 | mid_6_13 | 0.170817 | 0.644067 | 0.170210 |

Frequency-band all-layer necessities:

| delta | band | acc drop | margin drop |
| ---: | --- | ---: | ---: |
| 5 | full_1_21 | 0.912542 | 42.295456 |
| 5 | lowmid_1_13 | 0.912542 | 41.704838 |
| 13 | full_1_21 | 0.912542 | 57.058631 |
| 13 | lowmid_1_13 | 0.912542 | 56.563602 |
| 2 | full_1_21 | 0.911934 | 12.040099 |
| 2 | lowmid_1_13 | 0.910416 | 10.815317 |
| 13 | low_1_5 | 0.907987 | 45.418526 |
| 2 | mid_6_13 | 0.895688 | 7.754628 |
| 1 | full_1_21 | 0.894777 | 3.306669 |
| 13 | core_3_6 | 0.883845 | 39.498908 |
| 5 | mid_6_13 | 0.877771 | 24.919548 |
| 1 | lowmid_1_13 | 0.870331 | 2.898962 |

## Sector Scattering

Largest sector-path ablation effects:

| layer | source | target | acc drop | ablated acc | logprob drop | removed power |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| layer_1 | T | T | 0.879289 | 0.033252 | 6.311792 | 0.303310 |
| layer_2 | T | T | 0.789402 | 0.123140 | 3.626593 | 0.363951 |
| layer_3 | T | T | 0.487549 | 0.424992 | 1.080590 | 0.366511 |
| layer_1 | I | I | 0.152141 | 0.760401 | 0.196454 | 0.302846 |
| layer_2 | I | I | 0.145005 | 0.767537 | 0.159808 | 0.295399 |
| layer_4 | T | T | 0.116763 | 0.795779 | 0.243026 | 0.374541 |
| layer_1 | N | N | 0.106286 | 0.806256 | 0.097793 | 0.332836 |
| layer_2 | N | N | 0.098391 | 0.814151 | 0.090225 | 0.333477 |
| layer_3 | I | I | 0.094746 | 0.817795 | 0.084464 | 0.289475 |
| layer_3 | N | N | 0.079715 | 0.832827 | 0.061198 | 0.333463 |
| layer_1 | I | T | 0.058913 | 0.853629 | 0.099710 | 0.030020 |
| layer_4 | N | N | 0.047070 | 0.865472 | 0.030874 | 0.333419 |

Sector-mask readout baselines:

| mask | accuracy | drop |
| --- | ---: | ---: |
| I | 0.011388 | 0.901154 |
| N | 0.031886 | 0.880656 |
| T | 0.700577 | 0.211965 |
| NI | 0.027786 | 0.884756 |
| TI | 0.806408 | 0.106134 |
| TN | 0.737625 | 0.174916 |
| all | 0.912542 | 0.000000 |

## Interpretation

- Frequency patching tests causal sufficiency of individual cyclic readout coordinates by moving one clean Fourier pair into a corrupted example.
- Frequency ablation tests necessity by replacing one clean Fourier pair with the corresponding corrupted pair.
- Sector scattering ablates a single source-to-target contribution at one learned sector mixer, then continues the QNN and measures the final layerwise Dirac-mean readout.
- Sector masks test whether the final behavior can be carried by one input modality sector alone.
- The frequency results support the PDF's finite-kernel story: restoring all `k=1..21` recovers the clean held-out accuracy, and `k=1..13` recovers most of the answer; ablating `k=1..13` destroys it for the larger corruptions.
- The sector results reject a balanced shared-sector story: text alone retains `0.700577`, text+image reaches `0.806408`, text+number reaches `0.737625`, but number-only and image-only are near chance.
- The largest path effects are diagonal text-to-text paths, especially layers 1-3, so the learned cyclic kernel is causally real but rides primarily on text-sector continuity with supporting image/number contributions.

## Baselines

```json
{
  "delta_13_clean": {
    "clean_accuracy": 0.9125417552383844,
    "clean_logprob": -0.5068942364537169,
    "clean_loss": 0.5068942295031601,
    "clean_minus_corrupt_margin": 27.737557502419907,
    "corrupt_accuracy": 0.0,
    "corrupt_logprob": -28.244451861666793,
    "n": 6586.0
  },
  "delta_13_corrupt": {
    "clean_accuracy": 0.0,
    "clean_logprob": -29.73936605583814,
    "clean_loss": 29.73936605583814,
    "clean_minus_corrupt_margin": -29.32107376205208,
    "corrupt_accuracy": 0.9416945034922563,
    "corrupt_logprob": -0.41829159641352753,
    "n": 6586.0
  },
  "delta_1_clean": {
    "clean_accuracy": 0.9125417552383844,
    "clean_logprob": -0.5068942364537169,
    "clean_loss": 0.5068942295031601,
    "clean_minus_corrupt_margin": 1.2420188794916538,
    "corrupt_accuracy": 0.06164591557849985,
    "corrupt_logprob": -1.7489131089948138,
    "n": 6586.0
  },
  "delta_1_corrupt": {
    "clean_accuracy": 0.017764955967203157,
    "clean_logprob": -2.4938843440690954,
    "clean_loss": 2.493884306999459,
    "clean_minus_corrupt_margin": -2.0646504466742903,
    "corrupt_accuracy": 0.9339508047373216,
    "corrupt_logprob": -0.42923394604870296,
    "n": 6586.0
  },
  "delta_2_clean": {
    "clean_accuracy": 0.9125417552383844,
    "clean_logprob": -0.5068942364537169,
    "clean_loss": 0.5068942295031601,
    "clean_minus_corrupt_margin": 5.163959668828766,
    "corrupt_accuracy": 0.0,
    "corrupt_logprob": -5.6708540651452894,
    "n": 6586.0
  },
  "delta_2_corrupt": {
    "clean_accuracy": 0.0006073489219556636,
    "clean_logprob": -7.295427245352357,
    "clean_loss": 7.295427467770176,
    "clean_minus_corrupt_margin": -6.87613914992503,
    "corrupt_accuracy": 0.9398724567263893,
    "corrupt_logprob": -0.41928814176437235,
    "n": 6586.0
  },
  "delta_5_clean": {
    "clean_accuracy": 0.9125417552383844,
    "clean_logprob": -0.5068942364537169,
    "clean_loss": 0.5068942295031601,
    "clean_minus_corrupt_margin": 20.294428255722366,
    "corrupt_accuracy": 0.0001518372304889159,
    "corrupt_logprob": -20.801322466690706,
    "n": 6586.0
  },
  "delta_5_corrupt": {
    "clean_accuracy": 0.0,
    "clean_logprob": -22.424734700026573,
    "clean_loss": 22.424734700026573,
    "clean_minus_corrupt_margin": -22.001027422041073,
    "corrupt_accuracy": 0.9369875493470999,
    "corrupt_logprob": -0.42370742626404495,
    "n": 6586.0
  }
}
```

## Figures

- `figures\frequency_patch_all_layers.png`
- `figures\frequency_patch_layer_heatmap_delta1.png`
- `figures\frequency_ablation_layer_heatmap_delta1.png`
- `figures\frequency_band_patch_accuracy.png`
- `figures\sector_mixer_power.png`
- `figures\sector_path_accuracy_drop.png`
- `figures\sector_path_frequency_effect.png`
- `figures\sector_mask_accuracy.png`

## Tables

- `frequency_patch.csv`
- `frequency_band_patch.csv`
- `sector_path_ablation.csv`
- `sector_path_frequency_effect.csv`
- `sector_mixer_unitaries.csv`
- `sector_mask_ablation.csv`
- `manifest.json`
