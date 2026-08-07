# Tri-Modal Directed Route Path Tracing Results

Run date: 2026-06-20

Primary outputs:

```text
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/
```

## Executive Result

Layerwise directed route path tracing was run on the four completed
single-combination leave-out models:

```text
phase6_leave_image_text
phase6_leave_text_image
phase6_leave_number_image
phase6_leave_image_number
```

The result is not simply "image works" or "image fails". The model learns a
directed route topology. Mature no-image source states can causally rescue all
omitted targets, including routes whose own final behavior is near chance. The
more diagnostic question is where rescue appears before the final answer query
and where omitted-source patches damage mature trained computation.

The answer-query result is the positive control: for every omitted route, a
mature source answer-query state can drive the omitted target to `1.000000`.
This confirms that the shallow output heads and answer-slot readout can use a
good shared answer state. The route-specific failure is upstream: some omitted
routes do not construct that state themselves.

The non-answer sites give the sharper mechanistic picture:

| omitted route | route class | best non-answer mature-source rescue | best site | answer-query mean rescue | main read |
| --- | --- | ---: | --- | ---: | --- |
| `image+text` | partial | 1.000000 | layer-0 `mlp_out`, `operand_b_pool` | 0.6402 | early second-operand/accumulator state can be rescued, but the route's own final answer state remains partial |
| `text+image` | failure | 0.785156 | layer-0 `mlp_out`, `operand_b_pool` | 0.3642 | early rescue exists but is not carried into a mature final cyclic state |
| `number+image` | partial | 0.785156 | layer-0 `attn_out`, `plus` | 0.8237 | strongest answer-query pathway among the four; image-as-second can partially enter the sum route when the first operand is number |
| `image+number` | failure | 1.000000 | layer-0 `resid_mid`, `operand_b_pool` | 0.4364 | mature no-image second-operand accumulator can rescue, but the omitted route does not build the needed image-first information itself |

This refines the phase-6 conclusion. The model has a reusable shared cyclic
answer representation, but it does not have a fully modality-invariant
composition rule. It has learned ordered routes that can be partially entered
or bypassed by causal state replacement. The missing computation is best
described as route construction/routing into the shared answer state.

## Method

The analysis patches same-pair activations between a source cell and a target
cell. Source and target share the same held-out operand pair `(a, b)`, so a
successful patch means that the source activation contains a state sufficient
for the target context to answer the same modular-addition problem.

Each run used:

| setting | value |
| --- | --- |
| checkpoint | `checkpoint_final.pt` |
| pair split | `heldout_pair` |
| patch pairs | 256 |
| output modes | `number`, `text`, `image` |
| mature source combos | `number+number`, `number+text`, `text+number`, `text+text` |
| omitted target combos | one omitted route per run |
| directions | `good_to_omitted`, `omitted_to_good` |
| rows per run | 4,536 |
| total patch rows | 18,144 |

Patched semantic sites:

```text
bos
a_mode
operand_a_pool
plus
b_mode
operand_b_pool
mod
output_mode
answer_query
```

Patched components:

```text
-1: embed
0..3: resid_pre, attn_out, resid_mid, mlp_out, resid_post
```

Important interpretive caveats:

1. A late `answer_query` rescue is close to a final answer-state overwrite. It
   is essential as a positive control, but it is not by itself evidence that an
   omitted route knows how to construct the answer.
2. `best_omitted_to_good = 1.0` is often a neutral patch into an irrelevant
   site. For omitted-source patches into mature targets, the meaningful
   statistic is damage: how much the patch lowers a mature target baseline.
3. Sites such as `operand_b_pool` after layer-0 attention are not pure lexical
   operand states. They can already contain globally mixed information. A
   successful layer-0 `operand_b_pool` patch should be read as an early routed
   accumulator or carrier, not necessarily as the literal value `b`.
4. The path-tracing baselines use the 256 held-out patch pairs and therefore
   differ slightly from exhaustive heldout-pair behavior.

## Behavioral Context

These are the final behavior/probe/Fourier results motivating the path-tracing
experiment:

| omitted route | final behavior | final `s` probe | Fourier addition energy | route class |
| --- | ---: | ---: | ---: | --- |
| `image+text` | 0.471079 | 0.555623 | 0.536068 | partial |
| `text+image` | 0.109256 | 0.058617 | 0.098098 | failure |
| `number+image` | 0.536511 | 0.351703 | 0.567461 | partial |
| `image+number` | 0.108800 | 0.103014 | 0.079796 | failure |

The route classes are empirical. The two partial routes have substantial
omitted-combination behavior and Fourier/addition energy; the two failure
routes are near chance at the final answer slot.

## Run Summary

| run | omitted route | rows | best mature-source into omitted | best omitted-source into mature | elapsed seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| `phase6_leave_image_text` | `image+text` | 4,536 | 1.000000 | 1.000000 | 824.90 |
| `phase6_leave_text_image` | `text+image` | 4,536 | 1.000000 | 1.000000 | 814.52 |
| `phase6_leave_number_image` | `number+image` | 4,536 | 1.000000 | 1.000000 | 724.09 |
| `phase6_leave_image_number` | `image+number` | 4,536 | 1.000000 | 1.000000 | 727.56 |

The `1.000000` maxima are expected because same-pair answer-query patches can
overwrite a bad or partial route with a mature source answer state. The rest of
this report focuses on non-answer rescue and omitted-source damage.

## Mature-Source Rescue Into Omitted Routes

Path-tracing target baselines are computed over the 256 held-out patch pairs:

| omitted route | number baseline | text baseline | image baseline | best non-answer rescue | best non-answer site | answer-query mean | operand-pool max |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `image+text` | 0.460938 | 0.453125 | 0.468750 | 1.000000 | L0 `mlp_out` `operand_b_pool` | 0.6402 | 1.000000 |
| `text+image` | 0.082031 | 0.082031 | 0.085938 | 0.785156 | L0 `mlp_out` `operand_b_pool` | 0.3642 | 0.785156 |
| `number+image` | 0.574219 | 0.507812 | 0.566406 | 0.785156 | L0 `attn_out` `plus` | 0.8237 | 0.652344 |
| `image+number` | 0.089844 | 0.097656 | 0.093750 | 1.000000 | L0 `resid_mid` `operand_b_pool` | 0.4364 | 1.000000 |

Site-wise mean mature-source rescue:

| omitted route | `operand_a_pool` | `operand_b_pool` | `plus` | `answer_query` |
| --- | ---: | ---: | ---: | ---: |
| `image+text` | 0.3448 | 0.4971 | 0.4595 | 0.6402 |
| `text+image` | 0.0742 | 0.2274 | 0.0835 | 0.3642 |
| `number+image` | 0.5442 | 0.5226 | 0.5382 | 0.8237 |
| `image+number` | 0.0837 | 0.2405 | 0.0932 | 0.4364 |

Interpretation:

- `answer_query` rescue is universal and strongest for `number+image`.
- `image+text` and `text+image` both have their best non-answer rescue at
  layer-0 `operand_b_pool`, but `image+text` can be fully rescued there while
  `text+image` only reaches about `0.785`.
- `number+image` has its strongest non-answer rescue at the `plus` token after
  layer-0 attention/residual propagation, suggesting that the route has a
  different early carrier than the mixed text/image routes.
- `image+number` is near chance behaviorally, but a mature no-image patch into
  layer-0 `operand_b_pool` can fully rescue it. This means the downstream
  target context can use a good routed accumulator, not that the omitted route
  constructs one unaided.

## Omitted-Source Damage To Mature Routes

For omitted-source patches into mature targets, the best-patch value is not
diagnostic because many sites are neutral. The important statistic is how much
an omitted-source state damages a mature target.

Mean target-accuracy drop by site:

| omitted source | `operand_a_pool` mean drop | `operand_b_pool` mean drop | `answer_query` mean drop | strongest drop |
| --- | ---: | ---: | ---: | ---: |
| `image+text` | 0.4704 | 0.4642 | 0.1436 | 0.992188 |
| `text+image` | 0.4537 | 0.4888 | 0.3960 | 0.992188 |
| `number+image` | 0.0541 | 0.0925 | 0.2004 | 0.980469 |
| `image+number` | 0.3959 | 0.2542 | 0.4357 | 0.992188 |

Strongest-damage examples:

| omitted source | mature target | output | site | component | patched acc | drop |
| --- | --- | --- | --- | --- | ---: | ---: |
| `image+text` | `number+number` | number | L-1/L0 `operand_a_pool` or `operand_b_pool` | `embed`/`resid_pre`/L0 states | 0.007812 | 0.992188 |
| `text+image` | `number+number` or `number+text` | number/text | L-1/L0 `operand_b_pool` and sometimes `operand_a_pool` | `embed`/`resid_pre` | 0.007812 | 0.992188 |
| `number+image` | `number+number` | number | L-1/L0 `operand_b_pool` | `embed`/`resid_pre` | 0.019531 | 0.980469 |
| `image+number` | `text+text` | number/text/image | L-1/L0 `operand_a_pool` | `embed`/`resid_pre` | 0.007812 | 0.992188 |

Interpretation:

- Omitted-source states are not just uninformative. At operand pools and late
  answer-query states they can actively corrupt mature no-image computation.
- Damage localizes to the operand whose modality is missing from the trained
  ordered route. For `number+image`, the damaging site is mainly
  `operand_b_pool`, matching image as the second operand. For `image+number`,
  the damaging site is mainly `operand_a_pool`, matching image as the first
  operand.
- `text+image` has high answer-query damage (`0.3960` mean drop), consistent
  with its final answer slot being a poor or wrong answer state rather than a
  merely absent one.

## Figures

### `image+text`

Mature no-image sources into omitted `image+text`:

![image+text good-to-omitted heatmap](tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/figures/good_to_omitted_heatmap.png)

Omitted `image+text` source into mature no-image targets:

![image+text omitted-to-good heatmap](tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/figures/omitted_to_good_heatmap.png)

Reading: `image+text` is the clearest partial route. The layer-0
`operand_b_pool` carrier can be fully rescued from mature no-image sources, and
late `answer_query` patches also rescue. Omitted-source patches strongly damage
mature operand-pool computation but have less mean answer-query damage than the
near-failure routes.

### `text+image`

Mature no-image sources into omitted `text+image`:

![text+image good-to-omitted heatmap](tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/figures/good_to_omitted_heatmap.png)

Omitted `text+image` source into mature no-image targets:

![text+image omitted-to-good heatmap](tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/figures/omitted_to_good_heatmap.png)

Reading: `text+image` has early recoverable structure, but not enough to form
a mature answer state. The best non-answer rescue reaches `0.785156`, while
answer-query mean rescue is only `0.3642`. Omitted-source patches also cause
large operand-pool and answer-query damage, matching the earlier final-slot
probe and Fourier failure.

### `number+image`

Mature no-image sources into omitted `number+image`:

![number+image good-to-omitted heatmap](tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/figures/good_to_omitted_heatmap.png)

Omitted `number+image` source into mature no-image targets:

![number+image omitted-to-good heatmap](tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/figures/omitted_to_good_heatmap.png)

Reading: `number+image` is the strongest image-as-second route. The
answer-query mean rescue is `0.8237`, and the best non-answer rescue appears
at the layer-0 `plus` state rather than the operand pools. This suggests that
the number-first context gives the model a better early route into shared
composition, even though the final omitted answer state remains only partial.

### `image+number`

Mature no-image sources into omitted `image+number`:

![image+number good-to-omitted heatmap](tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/figures/good_to_omitted_heatmap.png)

Omitted `image+number` source into mature no-image targets:

![image+number omitted-to-good heatmap](tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/figures/omitted_to_good_heatmap.png)

Reading: `image+number` is behaviorally a failure, but not because downstream
number-second computation is impossible. A mature no-image state at layer-0
`operand_b_pool` can fully rescue the target. The omitted source itself badly
damages mature computation at `operand_a_pool`, consistent with the missing
image-first translation/routing step.

## Mechanistic Interpretation

The phase-6 controls separate three facts that were previously entangled:

1. The shared cyclic answer state exists and is readable by all three small
   output heads.
2. Trained image-containing routes can learn to construct that state.
3. An unseen ordered route may still fail to construct it, even when both
   modalities are individually trained elsewhere.

The path-tracing experiment adds the location of this failure. The failed or
partial routes are not blank at all sites. Mature-source patches into early
layer-0 carriers can sometimes rescue them strongly. Conversely, omitted-source
patches into mature targets can catastrophically damage the same computation.
That combination implies that the omitted routes enter nearby representational
territory but do not complete the right route into the shared answer-query
state.

The most useful next mechanistic hypothesis is:

```text
The model has a shared cyclic answer representation but a learned directed
route grammar for constructing it. Some unseen ordered routes activate enough
of the grammar to become partial, while others miss or corrupt the early
carrier that transports operand information into the answer query.
```

This is stronger than "no shared rule" and weaker than "one completely
modality-invariant rule". It also refines the translation-to-value-coordinate
story: translation into a shared value coordinate can exist for trained
contexts, but the model still needs a route-specific carrier that places those
values into the composition pathway.

## Consequences For The Original Hypotheses

| hypothesis | update from directed path tracing |
| --- | --- |
| one shared cyclic/Fourier rule | Supported for the late answer state, but not as an automatically invoked route-independent computation |
| three modality-specific cyclic rules | Still disfavored as the final explanation because mature no-image states rescue all omitted targets; however, early route carriers are modality/order-specific |
| translation-to-value-coordinate strategy | Supported as a representational component, but insufficient without route construction into the answer query |
| no true shared rule | Too pessimistic; the shared answer state is causally real, but access to it is directed and incomplete |

## Audit Artifacts

Per-run reports:

```text
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/PATH_TRACING_REPORT.md
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/PATH_TRACING_REPORT.md
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/PATH_TRACING_REPORT.md
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/PATH_TRACING_REPORT.md
```

Structured data:

```text
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/*/path_tracing_rows.csv
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/*/summary.json
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/pipeline.log
```

Figures:

```text
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/figures/good_to_omitted_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/figures/omitted_to_good_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/figures/good_to_omitted_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/figures/omitted_to_good_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/figures/good_to_omitted_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/figures/omitted_to_good_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/figures/good_to_omitted_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/figures/omitted_to_good_heatmap.png
```

## Reproduction

```powershell
cd modular_addition_mech_interp
python -m tri_modal_modular_grokking.run_directed_route_path_tracing_pipeline --patch-pairs 256 --device auto
```

Single-run form:

```powershell
cd modular_addition_mech_interp
python -m tri_modal_modular_grokking.route_path_tracing --run-dir tri_modal_modular_grokking/runs/phase6_leave_image_text --out-dir tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text --omitted-combo image+text --patch-pairs 256 --device auto
```
