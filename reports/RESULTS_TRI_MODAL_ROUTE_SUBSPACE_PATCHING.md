# Tri-Modal Route Subspace Causal Patching

Run date: 2026-06-20

Analysis directory:

```text
tri_modal_modular_grokking/analysis/phase6_route_subspace_patching
```

Primary artifacts:

```text
route_subspace_patching_rows.csv
summary.json
ROUTE_SUBSPACE_PATCHING_REPORT.md
figures/<model_group>_<route>_<patch_spec>_accuracy.png
```

## Question

The directed path-tracing and route-probe experiments showed that omitted ordered routes are not blank: they often preserve operands, and full source activations from mature no-image routes can rescue failed targets. This test asks a sharper causal question:

Can the rescue be localized to compact subspaces that correspond to decoded `a`, `b`, or `s`, or does the causal transport require a broader route/state-manifold alignment?

The experiment also compares every omitted route against the earlier fully grokked baseline, `phase4_full_crossmodal`, where the same input route is trained and solved.

## Design

Rows: `5,880`

Failed basis rows: `0`

Patch pairs: `256` held-out pairs

Basis sample pairs: `927`

Source routes:

```text
number+number
number+text
text+number
text+text
```

Target routes:

```text
image+text
text+image
number+image
image+number
```

Models:

| group | model |
| --- | --- |
| `leaveout` | corresponding phase-6 single-route leaveout model |
| `baseline_full` | `phase4_full_crossmodal`, evaluated on the same route as a trained pseudo-leaveout |

The intervention uses same-pair source-to-target patching. Source and target examples have the same `(a, b)` and output mode, so successful rescue is not answer replacement. It is activation-state transport.

Projected patch:

```text
h_target <- h_target + P(h_source - h_target)
```

Orthogonal complement patch:

```text
h_target <- h_target + (I - P)(h_source - h_target)
```

Subspaces:

| family | meaning |
| --- | --- |
| `full` | complete source vector, positive-control causal transport |
| `pca_mature_no_image` | high-variance manifold of mature no-image source states |
| `pca_local_target` | high-variance manifold of target-route states |
| `probe_mature_no_image_{a,b,s}` | raw activation directions implied by mature no-image linear probes |
| `probe_local_target_{a,b,s}` | raw activation directions implied by target-route linear probes |
| `random_control` | rank-matched random subspace |
| `*_orth` | orthogonal complement of a probe subspace |

Ranks tested: `16`, `32`.

## Hotspots

The sites were selected from the prior directed path-tracing result:

| route | tested causal carriers |
| --- | --- |
| `image+text` | L0 `mlp_out` `operand_b_pool`; L3 `mlp_out` `answer_query` |
| `text+image` | L0 `mlp_out` `operand_b_pool`; L3 `mlp_out` `answer_query` |
| `number+image` | L0 `attn_out` `plus`; L0 `resid_mid` `plus`; L3 `mlp_out` `answer_query` |
| `image+number` | L0 `mlp_out` `operand_a_pool`; L0 `resid_mid` `operand_b_pool`; L0 `mlp_out` `operand_b_pool`; L3 `mlp_out` `answer_query` |

## Results

| route | leaveout target | best full | best PCA | best probe projected | best probe orthogonal | random mean | baseline target | baseline min patch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `image+text` | 0.503906 | 1.000000 `early_b_image_text` | 1.000000 `pca_mature_no_image_r16` `early_b_image_text` | 0.535156 `probe_local_target_s_r32` `late_answer_query` | 1.000000 `probe_mature_no_image_b_r16_orth` `early_b_image_text` | 0.299886 | 1.000000 | 0.906250 |
| `text+image` | 0.123698 | 0.816406 `early_b_text_image` | 0.835938 `pca_local_target_r16` `early_b_text_image` | 0.218750 `probe_local_target_s_r32` `late_answer_query` | 0.816406 `probe_mature_no_image_b_r16_orth` `early_b_text_image` | 0.105713 | 1.000000 | 0.273438 |
| `number+image` | 0.498698 | 0.960938 `late_answer_query` | 0.964844 `pca_mature_no_image_r32` `late_answer_query` | 0.570312 `probe_local_target_s_r32` `late_answer_query` | 0.964844 `probe_mature_no_image_s_r16_orth` `late_answer_query` | 0.533963 | 1.000000 | 1.000000 |
| `image+number` | 0.119792 | 1.000000 `early_b_image_number_mid` | 1.000000 `pca_mature_no_image_r16` `early_b_image_number_mid` | 0.937500 `probe_local_target_s_r32` `early_b_image_number_mid` | 1.000000 `probe_mature_no_image_b_r16_orth` `early_b_image_number_mid` | 0.143962 | 1.000000 | 0.371094 |

Mean leaveout patch accuracy by family:

| route | full mean | PCA mean | probe projected mean | probe orthogonal mean | random mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `image+text` | 0.710938 | 0.695394 | 0.315674 | 0.752930 | 0.299886 |
| `text+image` | 0.516602 | 0.463216 | 0.087565 | 0.529785 | 0.105713 |
| `number+image` | 0.648872 | 0.643663 | 0.500515 | 0.648383 | 0.533963 |
| `image+number` | 0.470296 | 0.392721 | 0.129829 | 0.506058 | 0.143962 |

Baseline stability:

| route | baseline target | mean patched | min patched |
| --- | ---: | ---: | ---: |
| `image+text` | 1.000000 | 0.983754 | 0.906250 |
| `text+image` | 1.000000 | 0.797918 | 0.273438 |
| `number+image` | 1.000000 | 1.000000 | 1.000000 |
| `image+number` | 1.000000 | 0.886989 | 0.371094 |

## Interpretation

The main result is not that a linear probe direction is the causal route. It is almost the opposite.

Full-vector patches and PCA subspace patches recover most of the causal effect at the localized route hotspots. `image+text` and `image+number` can be rescued to `1.000000`; `text+image` rises from `0.123698` to `0.835938`; and `number+image` rises from `0.498698` to `0.964844`.

By contrast, projected `a/b/s` probe subspaces are weak on average. Their orthogonal complements often recover the full-vector effect. This means the variable is decodable from the state, but the specific low-rank decoding directions are usually not the causal carrier. The causal object is closer to a broader route/state-manifold alignment than to a clean low-dimensional `a`, `b`, or `s` axis.

The strongest exception is `image+number`: a local-target `s` probe subspace at the early `operand_b_pool` carrier reaches `0.937500` from a `0.119792` route baseline. This is the most plausible compact target-local sum-like causal carrier found in the run, but it is not the general mechanism across routes.

The fully grokked baseline is essential for interpreting random and preservation effects. Because all baseline pseudo-routes are already solved at `1.000000`, baseline rows are not rescue tests. They show whether a partial source-target transport preserves a solved computation. `number+image` is perfectly stable under every tested patch; `image+text` is highly stable; `text+image` and `image+number` can be damaged by some low-rank patches. Thus a random subspace preserving baseline accuracy is not evidence for a causal variable subspace. The meaningful leaveout signal is the matched rescue over the failed target baseline, especially when it is localized to the path-tracing hotspot and beats random controls.

## Conclusion

This experiment strengthens the directed-route-grammar interpretation:

1. The shared cyclic answer state is real, because full route-hotspot patches can causally rescue failed targets.
2. Omitted routes often contain usable local information, because PCA subspaces at localized carriers can recover much of the effect.
3. The causal transport is usually not a clean low-rank linear-probe `a/b/s` direction.
4. The relevant object is best described as a route-compatible activation manifold or coordinate chart that must be learned for each ordered modality pair.

The next most valuable follow-up is to replace PCA with explicitly learned low-dimensional transport maps between source and target route manifolds, then test those maps causally against the same baseline-full controls.
