# Tri-Modal Route Transport Maps

Run date: 2026-06-20

Analysis directory:

```text
tri_modal_modular_grokking/analysis/phase6_route_transport_maps
```

Primary artifacts:

```text
route_transport_rows.csv
route_transport_graph_rows.csv
summary.json
ROUTE_TRANSPORT_MAPS_REPORT.md
figures/*transport_accuracy.png
figures/*learned_transport_route_graph.png
```

## Question

The previous subspace-patching test showed that full vectors and PCA subspaces at route-localized carriers can rescue omitted routes, while linear-probe `a/b/s` directions usually cannot. This experiment asks a sharper question:

Can an explicit learned low-dimensional coordinate map transform an omitted route's local activation state into a mature no-image route state strongly enough to causally restore the computation?

If yes, the failure is primarily coordinate misalignment. If no, the full/PCA rescue likely depends on information supplied by the source state that is not recoverable from the omitted target state alone.

## Design

Rows: `4,488`

Failed rows: `0`

Patch pairs: `256` held-out pairs

Map sample pairs: `927`

Compared models:

| group | model |
| --- | --- |
| `leaveout` | corresponding phase-6 single-route leaveout model |
| `baseline_full` | `phase4_full_crossmodal`, where the same route is trained and solved |

For each route hotspot, source combo, and output mode, maps were fit from target-route states to mature no-image source-route states on pair-disjoint train pairs and evaluated causally on held-out patch pairs.

Transport families:

| family | meaning |
| --- | --- |
| `pca_ridge` | target PCA coordinates mapped to source PCA coordinates with ridge regression |
| `procrustes` | orthogonal target-PCA to source-PCA coordinate map |
| `identity_coord` | target PCA coordinates copied into source PCA axes; alignment control |
| `random_coord` | random orthogonal coordinate map; rank-matched negative control |
| `source_full` | complete source vector copied into target; positive control |

Ranks: `8`, `16`, `32`, `64`.

## Results

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

The learned transport test gives a more restrictive answer than PCA subspace patching.

Learned maps are clearly meaningful: they beat random controls by a wide margin on every leaveout route, and the best learned maps substantially improve all four omitted routes. But they usually do not reach the full-vector or PCA-subspace causal ceiling.

This is decisive because the transport maps use only the target-route activation state at patch time. A full source patch or source-delta PCA patch can inject information from a mature source state. A learned target-to-source map can only repair information already present in the target state. Therefore:

| route | interpretation |
| --- | --- |
| `image+number` | strongest coordinate-misalignment case; rank-16/32 maps at early `operand_b_pool` can fully repair the route for number output from compatible mature sources |
| `image+text` | substantial but incomplete coordinate repair; route contains enough local information for `0.855469`, but not enough for perfect learned-map rescue |
| `number+image` | partial coordinate repair; learned maps help, especially at early `plus`, but full rescue still requires source-state information |
| `text+image` | hardest case; learned maps lift the route but remain far below full-state rescue, indicating missing/unstable target-local information rather than mere linear coordinate rotation |

The baseline controls are also informative. In the fully grokked model, best learned maps preserve solved routes at `1.000000`, but the minimum learned-map accuracy can be very low for some routes. This means learned transport maps can be off-manifold even in a solved model, so only matched leaveout rescue over target baseline should be interpreted as mechanistic evidence.

The route graph is asymmetric. For `image+number`, mature sources containing a numeric second-position route (`number+number`, `text+number`) are much more compatible than sources whose second operand is text. For `image+text`, the strongest learned rescue comes from `number+number` and `text+number` into the early second-operand carrier. This reinforces the directed-route grammar: compatibility depends on ordered operand modality roles, not just whether a route is globally "good".

## Conclusion

The mechanism is not simply "the omitted route has the right information in a rotated coordinate system." That is true for parts of `image+number`, but not generally.

The best current model is:

1. full/PCA source-state patching works because mature states supply both coordinate alignment and missing/stabilizing route context;
2. learned target-to-source transport works when the omitted target state already contains recoverable route-local information;
3. `text+image` remains the clearest case where the target state itself is too weak or too unstable for low-dimensional coordinate repair;
4. route compatibility is directed and operand-position-specific.

The next high-information test is now tiny omitted-route supervision rescue curves: add small numbers of direct examples for each omitted ordered route and test whether the learned transport maps, operand probes, and patching hotspots move toward the fully grokked baseline geometry.
