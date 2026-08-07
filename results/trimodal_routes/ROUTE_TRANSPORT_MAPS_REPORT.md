# Route Transport Maps

This experiment replaces PCA-only subspace patching with explicit low-dimensional transport maps from an omitted route's local activation manifold into a mature no-image source route manifold. The same tests are run against the fully grokked `phase4_full_crossmodal` baseline as trained-route controls.

## Headline

| measurement | value |
| --- | ---: |
| rows | 4488 |
| numeric rows | 4488 |
| failed rows | 0 |
| patch pairs | 256 |
| map sample pairs | 927 |
| elapsed seconds | 1269.81 |

## Route Summary

| model group | route | target baseline | best full | best learned transport | learned site | mean learned | random control |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `baseline_full` | `image+number` | 1.000000 | 1.000000 | 1.000000 `pca_ridge_r16` | `L0:resid_mid:operand_b_pool` | 0.734077 | 0.440409 |
| `leaveout` | `image+number` | 0.117188 | 1.000000 | 1.000000 `pca_ridge_r16` | `L0:resid_mid:operand_b_pool` | 0.210036 | 0.029989 |
| `baseline_full` | `image+text` | 1.000000 | 1.000000 | 1.000000 `procrustes_r16` | `L0:mlp_out:operand_b_pool` | 0.850382 | 0.535482 |
| `leaveout` | `image+text` | 0.450521 | 0.996094 | 0.855469 `pca_ridge_r64` | `L0:mlp_out:operand_b_pool` | 0.308539 | 0.077759 |
| `baseline_full` | `number+image` | 1.000000 | 1.000000 | 1.000000 `pca_ridge_r8` | `L0:attn_out:plus` | 0.927174 | 0.760715 |
| `leaveout` | `number+image` | 0.550781 | 0.984375 | 0.769531 `pca_ridge_r64` | `L0:attn_out:plus` | 0.359999 | 0.155029 |
| `baseline_full` | `text+image` | 1.000000 | 1.000000 | 1.000000 `pca_ridge_r32` | `L3:mlp_out:answer_query` | 0.619941 | 0.294881 |
| `leaveout` | `text+image` | 0.102865 | 0.777344 | 0.488281 `pca_ridge_r64` | `L0:mlp_out:operand_b_pool` | 0.083876 | 0.038167 |

## Leaveout Rescue Detail

| route | leaveout target | best full | best learned | best learned site | best random | baseline best learned | baseline min learned |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `image+text` | 0.450521 | 0.996094 `early_b_image_text` | 0.855469 `pca_ridge_r64` | `early_b_image_text` `image` `number+number` | 0.214844 | 1.000000 `procrustes_r16` | 0.085938 |
| `text+image` | 0.102865 | 0.777344 `early_b_text_image` | 0.488281 `pca_ridge_r64` | `early_b_text_image` `number` `number+number` | 0.078125 | 1.000000 `pca_ridge_r32` | 0.109375 |
| `number+image` | 0.550781 | 0.984375 `late_answer_query` | 0.769531 `pca_ridge_r64` | `early_plus_attn` `image` `number+text` | 0.300781 | 1.000000 `pca_ridge_r8` | 0.062500 |
| `image+number` | 0.117188 | 1.000000 `early_b_image_number_mid` | 1.000000 `pca_ridge_r16` | `early_b_image_number_mid` `number` `number+number` | 0.085938 | 1.000000 `pca_ridge_r16` | 0.015625 |

Mean leaveout patch accuracy by family:

| route | full | `pca_ridge` | `procrustes` | `identity_coord` | `random_coord` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `image+text` | 0.715820 | 0.443359 | 0.422363 | 0.059896 | 0.077759 |
| `text+image` | 0.477865 | 0.130900 | 0.090007 | 0.030721 | 0.038167 |
| `number+image` | 0.683268 | 0.490479 | 0.491699 | 0.097819 | 0.155029 |
| `image+number` | 0.472331 | 0.296326 | 0.292094 | 0.041687 | 0.029989 |

Mean learned leaveout accuracy by rank:

| route | r8 | r16 | r32 | r64 |
| --- | ---: | ---: | ---: | ---: |
| `image+text` | 0.250217 | 0.319173 | 0.330349 | 0.334418 |
| `text+image` | 0.067708 | 0.076660 | 0.088921 | 0.102214 |
| `number+image` | 0.320855 | 0.366826 | 0.373300 | 0.379015 |
| `image+number` | 0.201986 | 0.210503 | 0.213108 | 0.214545 |

## Interpretation

Learned transport maps beat rank-matched random controls on every leaveout route, so they are not empty. But they do not generally match the full-vector/PCA-subspace rescue ceiling from the previous test.

The distinction matters: source-vector and source-delta PCA patches can inject mature source-state information. A target-to-source transport map can only repair information already present in the target activation. The result therefore separates coordinate misalignment from missing or unstable route-local information.

`image+number` is the strongest coordinate-misalignment case: early `operand_b_pool` maps from compatible no-image sources fully repair the number-output route. `image+text` and `number+image` are partial coordinate-repair cases. `text+image` remains the strongest evidence that the target state is too weak or unstable for low-dimensional map repair.

The fully grokked baseline confirms that best learned maps can preserve solved routes, but also that bad learned maps can damage solved routes badly. Baseline preservation alone is not evidence for a causal transport map; matched leaveout rescue over target baseline is the key readout.

## Map Families

| family | intervention |
| --- | --- |
| `pca_ridge` | PCA coordinates for target and source, ridge map from target coordinates to source coordinates |
| `procrustes` | PCA coordinates with an orthogonal Procrustes map, preserving geometry as much as possible |
| `identity_coord` | target PCA coordinates copied into source PCA axes; alignment control |
| `random_coord` | random orthogonal coordinate map; rank-matched negative control |
| `source_full` | complete mature source vector copied at the site; positive control |

## Artifacts

```text
route_transport_rows.csv
route_transport_graph_rows.csv
summary.json
figures/*transport_accuracy.png
figures/*learned_transport_route_graph.png
```
