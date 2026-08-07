# Tri-Modal Modular Grokking Results

Run date: 2026-06-17

Last updated: 2026-08-04

Branch: `tri-modal-modular-grokking`

Primary implementation:

```text
tri_modal_modular_grokking/
```

Related trimodal QNN implementation:

```text
trimodal_qnn_codex/
RESULTS_TRIMODAL_QNN_CODEX.md
```

Primary run:

```text
tri_modal_modular_grokking/runs/phase4_full_crossmodal
```

Primary analysis:

```text
tri_modal_modular_grokking/analysis/phase4_full_crossmodal
```

Real image-decoder extension:

```text
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels
tri_modal_modular_grokking/analysis/phase9_image_head_mech_ckpt15000
tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000
```

Native trimodal compression extension:

```text
tri_modal_modular_grokking/runs/phase10_native_trimodal_82k_seed1704
tri_modal_modular_grokking/analysis/phase10_native_trimodal_82k
```

Repeat-seed and robustness analyses:

```text
tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1705
tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1706
tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1707
tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1708
tri_modal_modular_grokking/analysis/phase4_operand_value_robustness_multiseed
tri_modal_modular_grokking/analysis/phase8_seed_program_queue
```

Architecture and training theory:

```text
TRI_MODAL_ARCHITECTURE_TRAINING_THEORY.md
```

## Executive Conclusion

The phase-4 full cross-modal experiment succeeded as a capability baseline and
produced strong mechanistic evidence for a shared modular-addition answer
representation across number, text, and image modalities.

The final checkpoint reached exact train accuracy `1.000000` over all `76,194`
train pair-cell examples and exact held-out accuracy `0.999871` over all
`177,849` held-out pair-cell examples. The sampled training telemetry reported
`1.000000` held-out accuracy at step `20,000`; the exhaustive final evaluation
found `23` residual held-out errors.

The answer-slot representation is almost perfectly residue-decodable and
cross-modal-transferable:

| diagnostic | value |
| --- | ---: |
| all-pair answer-slot states | 254,043 |
| final `s` probe train accuracy | 1.000000 |
| final `s` probe test accuracy | 0.999764 |
| mean cross-modal `s` probe transfer | 0.999632 |
| worst cross-modal `s` probe transfer | 0.997267 |
| strict 25k-state final `s` probe test mean | 0.998277 |
| strict 25k-state cross-cell `s` transfer mean | 0.958581 |
| full held-out exact by seed | 1704: 0.999871; 1705: 0.999854; 1706: 0.989255; 1707: 0.997464; 1708: 0.989328 |
| strict final `s` probe by seed | 1704: 0.998277; 1705: 0.995811; 1706: 0.815431; 1707: 0.934315; 1708: 0.657958 |
| strict final cross-cell `s` transfer by seed | 1704: 0.958581; 1705: 0.983691; 1706: 0.840452; 1707: 0.918476; 1708: 0.713975 |
| phase-8 directed leave-out mean across 20 seed-route cells | 0.299337 |
| zero-shot omitted routes above 0.5 | 4 / 20 |
| reduced rescue mean at 250 direct pairs | 0.880835 over 16 weak cells |
| weak routes above 0.9 after 250 direct pairs | 13 / 16 |
| test-6 mean best late-answer repairability | 1.000000 |
| test-6 mean phase-4 similarity | 0.949687 |
| final layer-2 text/image->number-modality operand-map NN, seed 1704 | 0.986235 / 0.964920 sampled; 0.999990 / 0.997866 exhaustive |
| three-seed layer-2 operand-map robustness | 1704 strong, 1705 strong, 1706 weak at final checkpoint |
| mean Fourier addition-diagonal energy | 0.937730 |
| full answer-slot patch clean-answer accuracy | 1.000000 |
| full answer-slot patch corrupt-answer accuracy | 0.000000 |
| phase-9 real image-pixel held-out image-template accuracy | 0.999899 exhaustive at checkpoint 15000 |
| phase-9 real image-pixel held-out foreground IoU | 0.975426 exhaustive at checkpoint 15000 |
| phase-9 real image-pixel final-slot `s` probe | 0.998949 held-out, permutation control 0.007613 |
| phase-9 non-image-output probe -> image-output `s` transfer | 0.998853 |
| phase-9 image-output probe -> non-image-output `s` transfer | 0.997352 |
| phase-9 image-decoder answer-slot patch | clean-template 1.000000; corrupt-template 0.000000 |
| phase-9 rank-32 residue subspace only -> image decode | 1.000000 |
| phase-9 rank-32 residue subspace removed -> image decode | 0.003222 |
| phase-10 parameter count / phase-9 compression | 81,927 / 18.356x |
| phase-10 step-33k exhaustive held-out exact | 0.918729 over 177,849 examples |
| phase-10 exhaustive held-out output accuracy | number 0.962030; text 0.905082; image 0.889074 |
| phase-10 exhaustive held-out image+image route | 0.904813 across outputs |
| phase-10 held-out image IoU / pixel MAE | 0.773364 / 0.012606 |

![full seed diagnostics](tri_modal_modular_grokking/analysis/phase8_seed_program_queue/scientific_figures/fig01_full_seed_diagnostic_matrix.png)


This strongly disfavors the "no shared rule" hypothesis and makes three
independent modality-specific rules unlikely as the primary explanation. The
full five-seed ensemble is most consistent with a shared cyclic/Fourier answer
substrate at the late answer slot. Seeds `1704` and `1705` are clean shared
linear-chart seeds; seed `1707` is strong but behaviorally imperfect; seeds
`1706` and `1708` show real all-state cyclic geometry and causal answer-slot
repair but weaker strict pair-disjoint linear charts. The phase-8 leave-out
program adds the key caveat: the shared answer substrate does not by itself
guarantee zero-shot access by every directed modality route. Only `4/20`
omitted routes exceed `0.5` without direct supervision, but `13/16` weak routes
cross `0.9` after only 250 direct examples. The best current theory is shared
cyclic answer machinery plus seed- and route-specific access grammar.

The phase-9 image-pixel extension resolves the main image-output caveat in the
fully supervised 27-cell setting. The original phase-4 run used
`image_class_proxy`: the image output was a 97-way class head whose prediction
could be rendered offline. Phase 9 replaces that stand-in with a real
`ImagePixelDecoder` from the final answer slot to a grayscale `1 x 64 x 128`
answer patch, trained from scratch on the same strict pair split. The durable
checkpoint at step `15000` reaches exhaustive held-out image-template accuracy
`0.999899`, foreground IoU `0.975426`, and pixel MAE `0.001124`. Mechanistic
checks show that this is not an independent memorizing image head: the decoder
receives only the final `answer_slot`; a final-slot `s` probe reaches
`0.998949` on held-out pairs with permuted-label control `0.007613`; probes
transfer between non-image-output and image-output slots at `0.998853`; and
patching a clean `number+number->number` answer slot into an
`image+image->image` target makes the generated image switch to the donor
answer with `1.000000` clean-template accuracy and `0.000000` corrupt-template
accuracy.

The phase-9 geometry battery strengthens the readout story. Raw Euclidean slot
arithmetic, `slot(r1) + slot(r2) - slot(0)`, fails near chance, so the final
answer space is not a simple additive vector code. But a rank-32 residue
subspace is sufficient for perfect image decoding, and removing that subspace
drops image decoding to `0.003222`. Interpolation between adjacent residue
centroids stays in broad endpoint basins (`0.961575` nearest-endpoint match),
while nearest-manifold repair of failed raw arithmetic does not recover the
modular sum. The correct interpretation is therefore not "linear vector
arithmetic in activation space"; it is a shared cyclic residue manifold that
the image-pixel decoder reads causally.

The phase-10 compression extension establishes the first sub-100k native
trimodal point. It reduces the phase-9 model from `1,503,859` to `81,927`
parameters (`18.356x`) while retaining the same three input mechanisms, all
nine ordered input routes, one shared backbone, conditioned number/text/image
outputs, and a real pixel-producing decoder. Exhaustive evaluation of the
selected durable step-`33,000` checkpoint reaches held-out exact `0.918729`,
with number output `0.962030`, text output `0.905082`, image-template output
`0.889074`, and image-image input-route accuracy `0.904813`. Every input route
is above `0.90`. The limiting component is the compressed image output decoder,
not the ability to consume image operands. This is a strong compression
capability result but not grokking: train and held-out accuracy rise together,
training never stably saturates, and the late trajectory oscillates.

The "translation-to-number" hypothesis should be read as shorthand for
translation into a shared residue-value representation or coordinate chart, not
as a literal activation becoming a scalar number. In that precise sense, the
layer-2 operand-value precursor is strongly supported in the two fully grokked
seeds `1704` and `1705`, including causal answer-query subspace patching. It is
not seed-universal: seed `1706` lacks the same direct layer-2 number-modality
coordinate chart, lacks the same layer-2 causal carrier, and does not reach the
strong-seed strict final-slot sharing regime. Test 6 further shows that the
most robust cross-seed causal carrier is the late `answer_query`, not a single
literal number-like early operand vector.

The QNN follow-up now implements the same trimodal problem family with a
data-reuploading quantum-style model, and the latest result directly imports
one of the main classical lessons: the answer-query carrier matters. The
strict coherent three-sector QNN remains the highest-accuracy QNN formulation:
the layerwise Dirac-mean seed sweep reached mean held-out exact `0.923474` over
seeds `9301-9305`, with best seed `9302` at `0.941543` by step `2000`. Causal
frequency patching and CCA show a real cyclic/Fourier code, but not a balanced
sector-common number manifold: it is text-dominant, text-image aligned, and
only weakly linearly number-aligned in the final raw sector state.

The route-conditioned QNN result is now much stronger than the old
ordered-route baselines. The bitter-lesson `operand_query` model uses three
sectors: `operand_a`, `operand_b`, and `answer_query`. Operand sectors carry
value-plus-modality features, the query sector is learned, and the strict
Dirac/Fourier head reads only the query sector. At step `20000`, it reaches
train exact `0.905144`, held-out exact `0.794497`, held-out loss `0.584886`,
exact query Fourier energy `0.967862`, and cross-sector ablation chance
`0.010325`. All nine ordered routes are above held-out `0.731`, with `TT`
strongest at `0.841634` and `NN` weakest at `0.731248`. This is the clearest
QNN evidence so far that route-conditioned composition can work when the
architecture forces operand information into an answer-query carrier.

This does not mean the QNN has learned a literal scalar "number." The stronger
claim is representation-level: it learns a cyclic/Fourier representation of
the modular sum in a query coordinate chart. The older residual-head
three-sector run (`0.627543` held-out) and legacy ordered-route run
(`0.357273` held-out) remain useful controls. Across the original `80`
checkpoint mech-interp pass, exact all-pair Fourier energy was high at the
best states (`0.945081` to `0.979104`), but final-head logit lenses stayed
near chance until the final circuit layer. The new operand-query checkpoint
shows the same late-formation signature: chance through layer 3, then
`0.794497` at layer 4. The dedicated QNN report is stored in
`RESULTS_TRIMODAL_QNN_CODEX.md`, with the design note in
`trimodal_qnn_codex/BITTER_LESSON_TRANSLATION.md`.

The phase-5 leave-combination-out follow-up is a sharper zero-shot composition
test. It held out `image+text`, `text+image`, `image+number`, and
`number+image` input combinations while still training all output modes for the
remaining combinations. The model learned a near-perfect no-image cyclic rule:
no-image cells reached `0.999987` mean heldout-pair accuracy, final-slot `s`
probe `0.995041`, and Fourier addition energy `0.959461`. Image-input cells,
including trained `image+image`, stayed at chance: heldout-pair accuracy
`0.009544`, final-slot `s` probe `0.002679`, and Fourier addition energy
`0.000098`. Same-pair final answer-slot patching from a good no-image source
rescued image targets to `1.000000`, while reverse `image+image` patches into
good no-image targets stayed at `0.009766`. The phase-5 failure therefore
localizes upstream of the answer-slot readout: the output heads can read a good
shared answer state, but image operands are not translated/routed into it in
this regime.

The phase-6 single-combination leave-out control removes the main phase-5
confound. It held out only `image+text` while training all other ordered input
combinations, including `image+image`, `image+number`, `number+image`, and
`text+image`. All trained combinations reached `1.000000` final heldout-pair
accuracy, so image inputs were learned. The omitted `image+text` combination
reached only `0.471079` mean heldout-pair accuracy, with final-slot `s` probe
`0.555623` and Fourier addition energy `0.536068`, versus trained image-input
`s` probe `0.999996` and Fourier energy `0.956771`. Mature no-image
answer-slot patches rescue the omitted target to `1.000000`, but final
omitted-source patches into trained targets average only `0.442057`, matching
the omitted source's own partial correctness. This is evidence for partial
zero-shot sharing and early compatible information, not for a fully automatic
ordered-modality composition rule.

The matched reverse-order phase-6 control held out only `text+image`. This
route was much weaker: final omitted behavior `0.109256`, final-slot `s` probe
`0.058617`, and Fourier addition energy `0.098098`. Mature no-image answer
states still rescue the omitted target to `0.994466`, so the target output
heads can read a good answer state. But final omitted-source patches into
trained targets average only `0.097656`, while layer-0 omitted-source patches
reach `0.905273`. The asymmetry between `image+text` and `text+image` is
therefore a real ordered-route effect: early information is present, but the
untrained `text+image` route does not carry it through later layers into the
shared cyclic answer state.

The directed route follow-up has now been extended to all four single-route
omissions: `image+text`, `text+image`, `number+image`, and `image+number`.
The two partial routes are `image+text` and `number+image`; the two near-failure
routes are `text+image` and `image+number`. Full layerwise path tracing over
`18,144` patch measurements shows that mature no-image answer-query states can
rescue every omitted target to `1.000000`, but non-answer rescue and
omitted-source damage are strongly route-specific. The result is a directed
route grammar: the shared cyclic answer state is causally real, but unseen
ordered modality pairs do not automatically construct the carrier that routes
operands into it.

The follow-up all-layer operand-probe test then separated local information
from shared-coordinate information. All omitted routes preserve substantial
operand information, and even the failed `image+number` route has locally
decodable late `s` at `0.522`. But mature no-image `s` probes transfer into
`image+number` at only `0.104`, and into `text+image` at only `0.099`.
Therefore the failures are not pure sensory/operand absence; they are route
alignment failures in which locally decodable information is not placed into
the causal/readout-compatible shared cyclic coordinate.

The third directed-route follow-up added subspace causal patching at the
localized carriers, with the earlier fully grokked `phase4_full_crossmodal`
model as a matched trained-route baseline. Across `5,880` causal patch rows,
full-vector and PCA subspace patches recovered much of the route-hotspot causal
effect: `image+text` and `image+number` reached `1.000000`, `text+image`
rose from `0.123698` to `0.835938`, and `number+image` rose from `0.498698`
to `0.964844`. However, projected linear-probe `a/b/s` subspaces were weak on
average, while their orthogonal complements often recovered the full-vector
effect. This shows that the relevant causal object is usually a broader
route-compatible activation manifold or coordinate chart, not a clean
low-dimensional `a`, `b`, or `s` probe direction.

The next transport-map test made this more precise. Across `4,488` causal rows,
explicit low-dimensional maps from omitted-route states into mature no-image
route coordinates beat random controls on every route, but only fully solved
one route. `image+number` is the clearest coordinate-misalignment case:
rank-16/32 maps at early `operand_b_pool` rescue it to `1.000000`.
`image+text` reaches `0.855469`, `number+image` reaches `0.769531`, and
`text+image` reaches only `0.488281`. The source-state/PCA rescue from the
previous test therefore cannot generally be reduced to a simple target-state
coordinate transform. For the hardest routes, the mature source state appears
to supply missing or stabilizing route context, not merely a rotated coordinate
chart.

The tiny supervision rescue test then intervened on the training data itself.
Each phase-6 leaveout model was fine-tuned for only `500` steps with `0`, `10`,
`25`, `50`, `100`, or `250` deterministic direct train pairs from the omitted
ordered route, while evaluation stayed on the full strict held-out omitted
route. Every route was strongly inducible. `image+number` is the clearest case:
it starts near chance at `0.108800`, reaches `0.827843` with only `10` direct
pairs, `0.940236` with `25`, and `0.986640` with `250`. `text+image`, the
hardest route, rises from `0.109256` to `0.932443` with `100` direct pairs.
This supports a missing ordered-route trigger/alignment interpretation rather
than absence of a reusable arithmetic circuit.

The rescued-checkpoint mechanistic follow-up confirms that this behavioral
rescue is usually a real move into the shared cyclic answer geometry. Across
the selected `0`, `10`, `25`, best-count, and phase-4 baseline checkpoints,
`image+number` moves from mature-to-omitted `s` transfer `0.103942` and
Fourier energy `0.079796` at zero direct to transfer `0.802540` and Fourier
energy `0.695584` with only `10` direct pairs; by `25` direct pairs transfer is
`0.927989`. `image+text` shows the same pattern, reaching transfer `0.947827`
and Fourier energy `0.842697` at `25` pairs, and matching phase-4 by `250`.
`text+image` is slower but rises to transfer `0.833915`, Fourier energy
`0.769423`, and omitted-to-mature final-slot patch accuracy `0.927734` at its
best `100`-pair rescue. `number+image` is the caveat: behavior and causal
patching become strong, but linear transfer remains moderate because even its
same-model mature no-image chart is less linearly organized. The refined
conclusion is that tiny supervision often enrolls an omitted route into the
shared cyclic state, but not every rescued model has the fully phase-4-like
linear coordinate chart.

The rescued route-local follow-up now covers all four omitted image-containing
routes. The result strengthens the shared-rule interpretation: every route has
a repairable late answer-query carrier, and every route approaches or reaches
phase-4-like local causal compatibility at its best rescue count. `image+text`
and `image+number` become locally strong with only `10` direct pairs;
`text+image` is the slow route, reaching late answer-query patching `0.996094`
and early-carrier patching `0.953125` at its best `100`-pair checkpoint;
`number+image` remains the geometric caveat, with strong causal compatibility
but less clean linear coordinatization. The refined conclusion is that tiny
direct supervision installs or stabilizes ordered-route carriers that feed the
pre-existing shared cyclic answer machinery; it does not merely tune output
decoders.

The first long-continuation test extends the most performant unsupervised
leave-out, `number+image`, from `20,000` to `40,000` steps. This falsifies a
simple stable-plateau reading: sampled omitted-route accuracy rises from
`0.533325` at 20k to `0.894531` at step `31,750`. But it also falsifies a clean
monotone delayed-grokking story. The same run collapses to `0.007324` at step
`35,000`, recovers to `0.852173` at step `37,250`, then finishes at exact full
heldout-route accuracy `0.681089` at 40k. The new combined train/loss curve
clarifies the mechanism: the `35,000` crash is global, with train accuracy
falling to `0.507568`, whereas the later decline from `37,250` to `40,000`
occurs after train accuracy has returned to `1.000000`. The current
interpretation is unstable route formation plus optimizer-induced basin
switching: the omitted route can transiently enter a high-performing
shared-rule-compatible regime, but that regime is not stable under the
continuing optimizer trajectory.

The phase-8 seed-robustness program ran as a resumable queue under
`tri_modal_modular_grokking/analysis/phase8_seed_program_queue` and is now
complete for tests 1-6. Tests 1 and 2 covered the seed-`1706` deep audit and
the two added full seeds `1707` and `1708`, combined with the existing full
seeds `1704`, `1705`, and `1706`. Test 3 completed the full five-seed directed
leave-out matrix. Test 4 completed the reduced tiny-supervision rescue grid
for all weak seed-route cells below `0.5`, using direct-pair counts `0`, `25`,
and `250`. Test 5 completed one rescue-selected route-local mechanistic audit
per ensemble seed. Test 6 completed the cross-seed mechanism synthesis:
carrier recurrence, phase-4-baseline similarity, rescue-inducibility taxonomy,
and seed/route mechanism classes.

The completed directed leave-out scope is:

```text
seeds: 1704, 1705, 1706, 1707, 1708
routes: image+text, text+image, number+image, image+number
matrix: 20 leave-out trainings to 20,000 steps, saved every 1,000 steps
```

Primary phase-8 artifacts:

```text
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/test3_directed_leaveout_summary.csv
tri_modal_modular_grokking/analysis/phase8_seed*_route_supervision_rescue_500step/route_supervision_rescue_summary.csv
tri_modal_modular_grokking/analysis/phase8_seed*_route_local_*/RESCUED_ROUTE_LOCAL_ANALYSIS_REPORT.md
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/test6_cross_seed_mechanism/TEST6_CROSS_SEED_MECHANISM_REPORT.md
tri_modal_modular_grokking/analysis/phase8_seed*_leave_*_mech/LEAVE_COMBO_MECH_INTERP_REPORT.md
tri_modal_modular_grokking/analysis/phase8_seed*_leave_*_path/PATH_TRACING_REPORT.md
tri_modal_modular_grokking/runs/phase8_seed*_leave_*/checkpoint_final.pt
```

The phase-4 full-seed table has now been combined across the original three
seeds and the two added seeds:

```text
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/FIVE_SEED_PHASE4_SUMMARY.md
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/five_seed_phase4_summary.csv
```

| seed | exact heldout | errors | all-state `s` probe | all-state transfer | Fourier | patch clean | strict `s` probe | strict transfer | interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `1704` | 0.999871 | 23 | 0.999764 | 0.999632 | 0.937730 | 1.000000 | 0.998277 | 0.958581 | clean shared linear cyclic geometry |
| `1705` | 0.999854 | 26 | 0.999775 | 0.999804 | 0.942719 | 1.000000 | 0.995811 | 0.983691 | clean shared linear cyclic geometry |
| `1706` | 0.989255 | 1,911 | 0.994940 | 0.992579 | 0.899903 | 0.988281 | 0.815431 | 0.840452 | shared cyclic geometry with strict-chart/behavior caveat |
| `1707` | 0.997464 | 451 | 0.998802 | 0.998035 | 0.896950 | 0.992188 | 0.934315 | 0.918476 | shared cyclic geometry with behavior caveat |
| `1708` | 0.989328 | 1,898 | 0.993376 | 0.991477 | 0.861053 | 0.992188 | 0.657958 | 0.713975 | shared cyclic geometry with strict-chart/behavior caveat |

The five-seed combined result strengthens the distinction between "shared
cyclic geometry exists" and "the full model is perfectly behaviorally
stabilized." All five full seeds have high all-state `s` decodability,
cross-cell transfer, Fourier addition energy, and causal answer-slot patching.
Only `1704` and `1705` are fully clean by both exhaustive behavior and strict
pair-disjoint linear-chart criteria. Seed `1707` clears the strict linear
sharing threshold but still has nontrivial residual held-out errors. Seeds
`1706` and `1708` show in these trained seeds that a real shared cyclic answer state can coexist with
weak strict linear transfer, so the shared rule is not always expressed as one
globally clean pair-disjoint linear chart.

## Phase-8 Test 3: Five-Seed Directed Leave-Out Matrix

Test 3 asks whether the fully trained shared backbone's cyclic rule is
available to omitted ordered modality routes without direct route supervision.
For each seed, we trained four leave-out models to `20,000` steps:

```text
omit image+text
omit text+image
omit number+image
omit image+number
```

Each run trained on the remaining ordered input combinations, evaluated the
omitted route exactly over `19,761` held-out examples, and then ran the same
mechanistic audit: behavior by cell, local probes, cross-cell transfer,
Fourier analysis, same-pair final-slot activation patching, and route path
tracing. All `20/20` seed-route cells have final checkpoints, exact eval rows,
mech-interp reports, and path-tracing reports.

The exact omitted-route accuracy matrix is:

![phase8 zero shot](tri_modal_modular_grokking/analysis/phase8_seed_program_queue/scientific_figures/fig03_zero_shot_leaveout_heatmap.png)

| seed | image+text | text+image | number+image | image+number | mean | best | worst |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `1704` | 0.728708 | 0.240878 | 0.394869 | 0.008805 | 0.343315 | `image+text` 0.728708 | `image+number` 0.008805 |
| `1705` | 0.549669 | 0.110470 | 0.053489 | 0.274429 | 0.247014 | `image+text` 0.549669 | `number+image` 0.053489 |
| `1706` | 0.480998 | 0.374323 | 0.622843 | 0.336167 | 0.453583 | `number+image` 0.622843 | `image+number` 0.336167 |
| `1707` | 0.369870 | 0.224634 | 0.315875 | 0.681241 | 0.397905 | `image+number` 0.681241 | `text+image` 0.224634 |
| `1708` | 0.092151 | 0.069784 | 0.030262 | 0.027276 | 0.054868 | `image+text` 0.092151 | `image+number` 0.027276 |

Route-wise aggregates:


| route | mean | median | population sd | weakest seed | strongest seed | interpretation |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `image+text` | 0.444279 | 0.480998 | 0.211133 | `1708` 0.092151 | `1704` 0.728708 | Usually the easiest omitted route, but not seed-stable. |
| `text+image` | 0.204018 | 0.224634 | 0.107292 | `1708` 0.069784 | `1706` 0.374323 | Consistently weaker than `image+text`; ordered-route direction matters. |
| `number+image` | 0.283467 | 0.315875 | 0.221648 | `1708` 0.030262 | `1706` 0.622843 | Highly seed-dependent; can become a strong zero-shot route. |
| `image+number` | 0.265584 | 0.274429 | 0.245176 | `1704` 0.008805 | `1707` 0.681241 | The most seed-volatile route; flips from chance to strong transfer. |

Across all 20 cells, mean omitted-route accuracy is `0.299337`, median is
`0.295152`, minimum is seed-`1704` `image+number` at `0.008805`, and maximum
is seed-`1704` `image+text` at `0.728708`. Only `4/20` omitted routes exceed
`0.5`; `16/20` are below `0.5`, `9/20` are below `0.25`, and `6/20` are below
`0.1`.

The train/loss curves for every seed show that most failed omitted routes are
not simple undertraining of the observed routes. By the final checkpoint,
trained-route sampled accuracy is near saturated in `18/20` cells. The two
notable exceptions are seed-`1704` `image+number`, which is a true
undertrained/unstable run at step `20,000` with final train accuracy
`0.514526`, and the weaker late-training seed-`1707`/`1708` cells whose train
accuracy remains high but not exactly saturated. The seed-`1705`
`number+image` rerun is particularly important: it has the same nominal seed
and route as the earlier phase-6 long-continuation source, but finishes at
exact omitted-route accuracy `0.053489`, showing that the original 20k partial
success was not reproduced by the phase-8 rerun.

![seed1704 train loss](tri_modal_modular_grokking/analysis/phase8_seed_program_queue/figures/seed1704_train_loss_curves.png)

![seed1705 train loss](tri_modal_modular_grokking/analysis/phase8_seed_program_queue/figures/seed1705_train_loss_curves.png)

![seed1706 train loss](tri_modal_modular_grokking/analysis/phase8_seed_program_queue/figures/seed1706_train_loss_curves.png)

![seed1707 train loss](tri_modal_modular_grokking/analysis/phase8_seed_program_queue/figures/seed1707_train_loss_curves.png)

![seed1708 train loss](tri_modal_modular_grokking/analysis/phase8_seed_program_queue/figures/seed1708_train_loss_curves.png)

| seed | route | exact heldout 20k | final train acc | final train loss | best sampled step | best sampled heldout |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `1704` | `image+text` | 0.728708 | 1.000000 | 0.000370 | 20000 | 0.723999 |
| `1704` | `text+image` | 0.240878 | 1.000000 | 0.000121 | 18750 | 0.276978 |
| `1704` | `number+image` | 0.394869 | 0.999268 | 0.003794 | 19250 | 0.431030 |
| `1704` | `image+number` | 0.008805 | 0.514526 | 1.488802 | 5500 | 0.010742 |
| `1705` | `image+text` | 0.549669 | 1.000000 | 0.000160 | 20000 | 0.553467 |
| `1705` | `text+image` | 0.110470 | 1.000000 | 0.000155 | 11500 | 0.220337 |
| `1705` | `number+image` | 0.053489 | 0.999512 | 0.002041 | 18500 | 0.053955 |
| `1705` | `image+number` | 0.274429 | 0.999878 | 0.000675 | 10250 | 0.316284 |
| `1706` | `image+text` | 0.480998 | 0.999023 | 0.002851 | 19750 | 0.697754 |
| `1706` | `text+image` | 0.374323 | 1.000000 | 0.000151 | 19500 | 0.388916 |
| `1706` | `number+image` | 0.622843 | 0.999023 | 0.003698 | 19250 | 0.628906 |
| `1706` | `image+number` | 0.336167 | 1.000000 | 0.000119 | 17750 | 0.346069 |
| `1707` | `image+text` | 0.369870 | 1.000000 | 0.000133 | 17000 | 0.389648 |
| `1707` | `text+image` | 0.224634 | 0.989502 | 0.019299 | 16500 | 0.363159 |
| `1707` | `number+image` | 0.315875 | 0.991577 | 0.016715 | 17500 | 0.359619 |
| `1707` | `image+number` | 0.681241 | 1.000000 | 0.000124 | 20000 | 0.684448 |
| `1708` | `image+text` | 0.092151 | 1.000000 | 0.000134 | 19250 | 0.180176 |
| `1708` | `text+image` | 0.069784 | 0.984985 | 0.033711 | 19500 | 0.071045 |
| `1708` | `number+image` | 0.030262 | 0.986816 | 0.028744 | 11000 | 0.079224 |
| `1708` | `image+number` | 0.027276 | 0.993896 | 0.013481 | 8000 | 0.103027 |

The main result is therefore not "one route-invariant compositional rule is
always used." The full models can learn a shared cyclic answer state, but the
leave-out models often fail to route omitted ordered modality pairs into that
state. The failures are not uniformly tied to a single modality. Seed `1708`
is a global weak/failure seed across every omitted route despite its full-model
shared cyclic geometry; seed `1706` is strong on `number+image`, while seed
`1707` is strong on the reverse `image+number`. This is stronger evidence for
a shared cyclic arithmetic substrate plus seed- and route-specific access
grammar than for either three isolated modality-specific rules or a trivial
translation-to-number strategy.

Mechanistically, the completed test-3 reports support the same decomposition
seen in the phase-6 route audits. In high-transfer cells, the omitted route
enters an answer-slot representation that is close enough to the shared cyclic
state for cross-cell probes and final-slot patching to work. In low-transfer
cells, local operand information can still be present, but it is not aligned to
the causal/readout-compatible cyclic coordinate. Thus the object we should call
"the number" is more precisely a learned representation of residue value: a
coordinate chart or activation manifold that can support modular addition when
it is aligned, but that can be locally encoded in incompatible forms when an
ordered route is omitted.


## Phase-8 Test 4: Reduced Tiny-Supervision Rescue Grid

Test 4 asks whether the weak directed leave-out routes are absent arithmetic
circuits or merely missing route-access triggers. The reduced completed grid
selected every test-3 seed-route cell below `0.5` zero-direct omitted-route
accuracy, then fine-tuned for `500` steps using `0`, `25`, or `250` direct
pairs from the omitted route. This gives a baseline, a tiny-supervision point,
and a high-supervision point for all weak cells without spending time on the
four already-strong zero-shot routes.

Coverage:

| item | value |
| --- | ---: |
| selected weak cells | 16 |
| direct-pair counts | `0`, `25`, `250` |
| completed rows | 48 |
| queue elapsed for test 4 | `16822.408` seconds |

Aggregate rescue:


| direct pairs | n | mean omitted-route accuracy | median | min | max | cells >= 0.9 | cells >= 0.98 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | 16 | 0.212768 | 0.232756 | 0.008805 | 0.480998 | 0 | 0 |
| `25` | 16 | 0.554027 | 0.672334 | 0.008704 | 0.872628 | 0 | 0 |
| `250` | 16 | 0.880835 | 0.952887 | 0.005718 | 0.994940 | 13 | 4 |

Per-cell rescue matrix:


| seed | route | count 0 | count 25 | count 250 | interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| `1704` | `image+number` | 0.008805 | 0.008704 | 0.005718 | severe non-rescue outlier |
| `1704` | `number+image` | 0.394869 | 0.837103 | 0.967360 | strongly inducible |
| `1704` | `text+image` | 0.240878 | 0.401903 | 0.906735 | strongly inducible by 250 pairs |
| `1705` | `image+number` | 0.274429 | 0.787865 | 0.994889 | strongly inducible |
| `1705` | `number+image` | 0.053489 | 0.027984 | 0.937807 | delayed but strong rescue |
| `1705` | `text+image` | 0.110470 | 0.658823 | 0.980163 | strongly inducible |
| `1706` | `image+number` | 0.336167 | 0.842113 | 0.945448 | strongly inducible |
| `1706` | `image+text` | 0.480998 | 0.825515 | 0.994940 | strongly inducible |
| `1706` | `text+image` | 0.374323 | 0.685846 | 0.939730 | strongly inducible |
| `1707` | `image+text` | 0.369870 | 0.859369 | 0.973382 | strongly inducible |
| `1707` | `number+image` | 0.315875 | 0.821062 | 0.975558 | strongly inducible |
| `1707` | `text+image` | 0.224634 | 0.654623 | 0.956278 | strongly inducible |
| `1708` | `image+number` | 0.027276 | 0.040534 | 0.949496 | delayed but strong rescue |
| `1708` | `image+text` | 0.092151 | 0.872628 | 0.980568 | strongly inducible |
| `1708` | `number+image` | 0.030262 | 0.073225 | 0.754820 | partial rescue |
| `1708` | `text+image` | 0.069784 | 0.467132 | 0.830474 | partial rescue |

Route aggregates at `250` direct pairs:

| route | selected cells | mean at 250 | interpretation |
| --- | ---: | ---: | --- |
| `image+text` | 3 | 0.982963 | all selected weak `image+text` routes become strong |
| `text+image` | 5 | 0.922676 | generally rescued, with seed `1708` still partial |
| `number+image` | 4 | 0.908886 | generally rescued, with seed `1708` still partial |
| `image+number` | 4 | 0.723888 | dragged down by seed `1704` non-rescue |

The scientific point is strong. A small number of direct examples usually
teaches the omitted ordered route how to access the already-learned arithmetic
machinery. That strongly favors a missing or unstable route-access grammar over
absence of the shared cyclic rule. The caveat is seed-`1704` `image+number`,
which is not rescued at all by this reduced `500`-step protocol.

## Phase-8 Test 5: Rescue-Selected Route-Local Mechanistic Audits

Test 5 selected one high-information route per seed and ran the route-local
path tracing and subspace patching battery on the zero-direct checkpoint,
rescued checkpoints, and the fully trained phase-4 baseline. The selected
cases were:

| seed | selected route | selection role |
| ---: | --- | --- |
| `1704` | `image+number` | hardest non-rescued outlier |
| `1705` | `number+image` | weak zero-shot route rescued at 250 pairs |
| `1706` | `image+number` | anomalous seed with strong rescue |
| `1707` | `text+image` | representative reverse-order rescued route |
| `1708` | `image+number` | globally weak zero-shot seed with strong route rescue |

Coverage:

![phase8 mechanism scores](tri_modal_modular_grokking/analysis/phase8_seed_program_queue/scientific_figures/fig08_selected_mechanism_score_matrix.png)

| seed | route | path rows | path summary rows | subspace rows | subspace summary rows |
| ---: | --- | ---: | ---: | ---: | ---: |
| `1704` | `image+number` | 13,608 | 6 | 2,952 | 12 |
| `1705` | `number+image` | 18,144 | 8 | 3,600 | 12 |
| `1706` | `image+number` | 18,144 | 8 | 3,936 | 16 |
| `1707` | `text+image` | 18,144 | 8 | 2,112 | 8 |
| `1708` | `image+number` | 18,144 | 8 | 3,936 | 16 |

Late answer-query repairability was the most consistent causal signature. In
all selected cases, copying a mature good-route answer-query state into the
omitted route could reach `1.000000` best patch accuracy. The late answer slot
is therefore a reusable causal carrier across seeds. The difference between
successful and unsuccessful behavioral rescue is whether the route learns to
produce or enter that mature answer state without patching.

Representative route-local findings:

| seed | route | zero-direct late-answer patch | best rescued late-answer patch | phase-4 baseline late-answer patch | interpretation |
| ---: | --- | ---: | ---: | ---: | --- |
| `1704` | `image+number` | 1.000000 | 0.906250 at 25 pairs | 1.000000 | answer state exists causally, but route does not behaviorally rescue |
| `1705` | `number+image` | 0.750000 | 1.000000 at 250 pairs | 1.000000 | rescue becomes phase-4-like |
| `1706` | `image+number` | 1.000000 | 1.000000 at 250 pairs | 1.000000 | anomalous seed still converges to shared answer carrier |
| `1707` | `text+image` | 0.984375 | 1.000000 at 25/250 pairs | 1.000000 | reverse-order route converges cleanly after rescue |
| `1708` | `image+number` | 0.906250 | 1.000000 at 250 pairs | 1.000000 | weak seed can be routed into mature answer state |

The subspace results match the earlier phase-7 pattern: full-vector and
orthogonal-complement patches often carry more causal rescue than projected
low-rank `a/b/s` probe subspaces. The route access mechanism is therefore not
just a single low-rank value direction. It is closer to an activation manifold
or coordinate-chart alignment feeding the late answer-query carrier.

## Phase-8 Test 6: Cross-Seed Mechanism Synthesis

Test 6 is an analysis-only synthesis over tests 3-5. It compares zero-shot
behavior, rescue efficiency, route-local causal carriers, and similarity to the
fully trained phase-4 baseline.

Artifacts:

```text
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/test6_cross_seed_mechanism/test6_case_index.csv
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/test6_cross_seed_mechanism/test6_mechanism_scores.csv
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/test6_cross_seed_mechanism/test6_carrier_recurrence.csv
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/test6_cross_seed_mechanism/test6_phase4_similarity.csv
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/test6_cross_seed_mechanism/test6_taxonomy_assignments.csv
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/test6_cross_seed_mechanism/TEST6_CROSS_SEED_MECHANISM_REPORT.md
```

Summary:

| metric | value |
| --- | ---: |
| selected seed-route cases | 5 |
| route-local reports present | 5 |
| mean best late-answer repairability | 1.000000 |
| mean phase-4 similarity | 0.949687 |
| canonical phase-4-like shared-chart cases | 4 |
| weak or unstable seed-route cases | 1 |

Taxonomy:


| seed | route | zero-direct accuracy | best rescue accuracy | phase-4 similarity | best late-answer repairability | class |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `1704` | `image+number` | 0.008805 | 0.008805 | 0.806250 | 1.000000 | weak or unstable seed-route |
| `1705` | `number+image` | 0.053489 | 0.937807 | 0.993750 | 1.000000 | canonical phase-4-like shared chart |
| `1706` | `image+number` | 0.336167 | 0.945448 | 0.989063 | 1.000000 | canonical phase-4-like shared chart |
| `1707` | `text+image` | 0.224634 | 0.956278 | 0.993750 | 1.000000 | canonical phase-4-like shared chart |
| `1708` | `image+number` | 0.027276 | 0.949496 | 0.965625 | 1.000000 | canonical phase-4-like shared chart |

Recurrent high-effect carriers:


| carrier category | high-effect rows |
| --- | ---: |
| `answer_query` | 34 |
| `operand_b_pool` | 14 |
| `operand_a_pool` | 4 |
| `other` | 4 |
| `bos` | 2 |

The cross-seed synthesis sharpens the final interpretation. The route-local
audits are not an unbiased sample of all 20 leave-out cells, but within the
selected high-information cases the causal pattern is highly consistent:
late `answer_query` is the dominant reusable carrier, `operand_b_pool` is the
most recurrent upstream operand carrier, and four of five selected cases become
phase-4-like after rescue. Seed-`1704` `image+number` remains the principled
exception: it has causally repairable answer states but the route itself is not
behaviorally rescued by the reduced supervision schedule.

## Figure Assets And Quality

The core curated figure set still contains ten selected figures across phases
4-8, chosen for coverage of training dynamics, leave-out behavior, path
tracing, rescue, seed robustness, and mechanism synthesis. This report now
also embeds the 40k continuation train/loss curve and five all-seed
train/loss telemetry figures. The complete generated phase-8 scientific figure
suite remains under:

```text
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/scientific_figures/
```

The curated ten-figure set is stored separately under:

```text
tri_modal_modular_grokking/analysis/selected_figures_10/
```

The manifest and automated quality report are:

```text
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/scientific_figures/FIGURE_MANIFEST.md
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/scientific_figures/figure_quality_report.csv
```

All `10/10` generated phase-8 figures passed the automated quality gates:
minimum dimensions, file size, grayscale dynamic range, entropy, and non-white
pixel density. The generation script is:

```text
tri_modal_modular_grokking/make_phase8_scientific_figures.py
```

The additional train/loss telemetry figures and summaries are stored under:

```text
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/figures/train_loss_curve_combined.png
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/key_step_train_loss_summary.csv
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/figures/seed*_train_loss_curves.png
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/all_seed_training_curve_summary.csv
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/figures/TRAIN_LOSS_CURVE_MANIFEST.md
```

All six train/loss PNGs passed the local image sanity check: nonzero file
size, large rendered dimensions, nonblank grayscale entropy, and nonwhite
pixel density. The generation script is:

```text
tri_modal_modular_grokking/make_leaveout_training_curve_figures.py
```

## Experimental Question

The experiment asks whether one shared backbone trained on modular addition
across numbers, English text, and rendered images learns:

1. one shared cyclic/Fourier rule;
2. three modality-specific cyclic rules;
3. a translation-to-number strategy, meaning translation into a shared
   numeric/value-coordinate operand representation;
4. or no true shared rule.

The task is modular addition:

```text
s = (a + b) mod 97
```

The model sees operands through any ordered pair of input modalities:

```text
number, text, image
```

and must answer through any output modality:

```text
number, text, image
```

This creates `3 x 3 x 3 = 27` task cells.

## Implementation Summary

The implementation is isolated from the older `modular_addition/` package:

| file | role |
| --- | --- |
| `tri_modal_modular_grokking/data.py` | strict pair splits, 27-cell dataset, text/image caches, corpus writer |
| `tri_modal_modular_grokking/render.py` | deterministic digit image rendering |
| `tri_modal_modular_grokking/models.py` | shared-backbone tri-modal model and answer-slot extraction |
| `tri_modal_modular_grokking/losses.py` | number/text/image losses and per-example correctness |
| `tri_modal_modular_grokking/train.py` | training, per-cell metrics, checkpointing, resume support |
| `tri_modal_modular_grokking/analyze.py` | answer-slot extraction and linear probes |
| `tri_modal_modular_grokking/rigorous_probes.py` | sampled pair-disjoint ridge probes across checkpoints and layers |
| `tri_modal_modular_grokking/operand_path_analysis.py` | operand-state probes, text/image-to-number alignment, and operand replacement patching |
| `tri_modal_modular_grokking/operand_value_robustness.py` | exhaustive all-pair operand-value robustness, null controls, rank sweeps, context transfer, seed repeats, and layer-2 answer-query subspace patching |
| `tri_modal_modular_grokking/fourier.py` | 2D Fourier energy diagnostics by cell |
| `tri_modal_modular_grokking/patching.py` | final answer-slot activation patching |
| `tri_modal_modular_grokking/leave_combo_mech_interp.py` | leave-combination behavior, probes, Fourier, alignment, and same-pair answer-slot patching |
| `tri_modal_modular_grokking/route_path_tracing.py` | semantic-site and component-level directed route activation patching |
| `tri_modal_modular_grokking/run_directed_route_path_tracing_pipeline.py` | sequential runner for the four phase-6 directed route path-tracing jobs |
| `tri_modal_modular_grokking/route_operand_probes.py` | all-layer semantic-site probes for `a`, `b`, and `s` on directed route leave-outs |
| `tri_modal_modular_grokking/route_subspace_patching.py` | projected/full/orthogonal subspace causal patching at localized route carriers, with fully grokked baseline comparison |
| `tri_modal_modular_grokking/route_transport_maps.py` | learned low-dimensional target-to-mature route transport maps with causal patching and route-graph summaries |
| `tri_modal_modular_grokking/route_supervision_rescue.py` | tiny omitted-route direct-supervision rescue curves, generated configs, full heldout evaluation, and audit figures |
| `tri_modal_modular_grokking/rescued_checkpoint_mech.py` | mechanistic analysis of rescued checkpoints with Fourier energy, probe transfer, and final-slot causal patching against phase-4 baseline |
| `tri_modal_modular_grokking/rescued_route_local_analysis.py` | route-local path tracing and subspace patching for selected rescued checkpoints against phase-4 baseline |
| `tri_modal_modular_grokking/image_head_mech.py` | phase-9 real image-pixel decoder audit: exhaustive behavior, final-slot probes, output-type transfer probes, and image-decoder answer-slot patching |
| `tri_modal_modular_grokking/slot_arithmetic.py` | phase-9 counterfactual raw answer-slot arithmetic tests on copied checkpoints |
| `tri_modal_modular_grokking/slot_geometry_experiments.py` | phase-9 slot-geometry battery: Fourier/phase reconstruction, residue-subspace sufficiency/ablation, interpolation, manifold repair, and checkpoint dynamics |

Focused tests were added:

```text
tests/test_tri_modal_data.py
tests/test_tri_modal_model.py
tests/test_tri_modal_fourier.py
```

The focused tests passed:

```powershell
python -m pytest tests/test_tri_modal_data.py tests/test_tri_modal_model.py tests/test_tri_modal_fourier.py
```

Result:

```text
9 passed, 1 warning
```

## Data And Splits

The run uses a strict shared pair split across all modalities. A pair held out
from training is held out for every input/output modality cell.

| item | value |
| --- | ---: |
| modulus | 97 |
| all operand pairs | 9,409 |
| train fraction | 0.30 |
| train pairs | 2,822 |
| held-out pairs | 6,587 |
| task cells | 27 |
| train pair-cell examples | 76,194 |
| held-out pair-cell examples | 177,849 |
| all pair-cell states for analysis | 254,043 |

The text channel uses English number words for residues `0..96`, with a fixed
maximum answer length of 4 tokens including BOS/EOS. The recorded vocabulary has
38 tokens.

The image channel uses deterministic grayscale digit renderings with shape
`1 x 64 x 128`. The first full phase-4 run used an `image_class_proxy`, not
pixel reconstruction. This deliberately kept the decoder small: image answers
were scored by a 97-way class head from the shared answer slot, rather than by
a large image generator that could hide arithmetic inside the decoder. The
phase-9 extension keeps the same strict data split and shared-backbone
structure but changes the image target to `image_pixels`: a compact
transposed-convolution pixel decoder maps only the final answer slot to a
rendered grayscale answer image.

## Model

The phase-4 model is a small shared transformer backbone:

| component | value |
| --- | ---: |
| parameters | 965,082 |
| `d_model` | 128 |
| transformer layers | 4 |
| attention heads | 4 |
| feed-forward width | 512 |
| image tokens per operand | 8 |

Each example is converted into a variable-length shared sequence with:

```text
BOS, operand-A mode, operand-A tokens, plus,
operand-B mode, operand-B tokens, mod,
requested output mode, answer query
```

The answer query position is the common mechanistic target. The number, text,
and image-class heads are shallow linear heads from this answer slot.

This architecture intentionally gives the arithmetic computation to the shared
backbone. The output heads are too small to plausibly implement separate full
addition algorithms on their own.

The phase-9 image-pixel variant keeps the same backbone width and depth but
adds an `ImagePixelDecoder` and template-based image evaluation:

| component | value |
| --- | ---: |
| parameters | 1,503,859 |
| `d_model` | 128 |
| transformer layers | 4 |
| attention heads | 4 |
| feed-forward width | 512 |
| image tokens per operand | 8 |
| image decoder channels | 32 |
| image template temperature | 0.02 |

The pixel decoder receives only `answer_slot`. It does not receive `a`, `b`,
input modality IDs, output modality IDs, pair IDs, or target templates. During
evaluation, generated pixels are compared to all rendered residue templates to
compute image-template accuracy, foreground IoU, and pixel MAE.

## Architecture, Loss, And Training Theory

A dedicated theory note now records the architectural and optimization
reasoning behind the phase-4 design:

```text
TRI_MODAL_ARCHITECTURE_TRAINING_THEORY.md
```

The core factorization is:

```text
modality-specific input encoders
-> shared transformer sequence model
-> single answer-query bottleneck
-> shallow output heads
```

The final model has `965,082` parameters. Of these, `920,448` are in the shared
backbone and only `44,634` are in all three output heads combined. The heads are
linear maps from the same `d_model = 128` answer-slot vector:

| head | map | parameters |
| --- | --- | ---: |
| number | `R^128 -> R^97` | 12,513 |
| text | `R^128 -> R^(4 * 38)` | 19,608 |
| image class proxy | `R^128 -> R^97` | 12,513 |

This is the anti-cheating constraint. The heads can decode a residue
representation, but they cannot inspect operand tokens or run separate
multi-step arithmetic algorithms. Any arithmetic needed by all three output
formats must be made available at the shared answer-query state by the shared
backbone.

The phase-4 loss is exactly the sum of active supervised output terms with
equal weights:

| loss term | weight |
| --- | ---: |
| number cross entropy | 1.0 |
| text masked token cross entropy | 1.0 |
| image-class proxy cross entropy | 1.0 |
| supervised residue contrastive loss | 0.0 |

The absence of a residue contrastive or Fourier auxiliary term matters. The
observed cyclic/Fourier structure is not injected directly by the loss; it
emerges from supervised modular addition under the shared answer-slot
bottleneck, shallow heads, strict pair split, long training, and high weight
decay.

The training regime is grokking-relevant: AdamW with learning rate `0.001`,
weight decay `0.1`, batch size `256`, and `20,000` steps over `76,194` training
pair-cell examples. This is about `5.12M` sampled training examples, or `67.2`
passes over the train pair-cell dataset. The theory note argues that this setup
first permits brittle train/cell features, then favors a lower-complexity cyclic
rule that explains many held-out pairs once optimization finds it.

The theory note also preserves the important caveat that the step-10,000 resume
used a fresh optimizer because the checkpoint did not yet store optimizer
state. The run demonstrates that this architecture can reach a grokked shared
solution; it does not isolate whether the optimizer reset was necessary.

## Training

Configuration:

```text
tri_modal_modular_grokking/configs/phase4_full_crossmodal.yaml
```

Key settings:

| item | value |
| --- | ---: |
| steps | 20,000 |
| batch size | 256 |
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.1 |
| eval interval | 250 |
| checkpoint interval | 1,000 |
| telemetry eval batches | 32 |

Training was suspended after a durable checkpoint at step `10,000` and then
resumed. The step-10,000 checkpoint was created before optimizer-state saving
was added, so the continuation resumed model weights with a fresh optimizer.
From step `11,000` onward checkpoints include optimizer state.

Final checkpoints:

```text
tri_modal_modular_grokking/runs/phase4_full_crossmodal/checkpoint_20000.pt
tri_modal_modular_grokking/runs/phase4_full_crossmodal/checkpoint_final.pt
```

## Training Dynamics

The run showed a sharp grokking transition after the resume. Up to step `10,000`
the held-out metric remained around one third exact accuracy. Between steps
`10,500` and `11,250`, held-out accuracy jumped from below `0.5` to above
`0.9`.

The table below is sampled telemetry. It evaluates only the configured first 32
batches per split, so it is useful for dynamics but not the final exhaustive
accuracy claim.

| step | held-out acc | train acc | held-out loss |
| ---: | ---: | ---: | ---: |
| 1 | 0.007812 | 0.006958 | 4.200700 |
| 1,000 | 0.005493 | 0.012939 | 3.543970 |
| 5,000 | 0.238525 | 0.453125 | 2.787330 |
| 9,000 | 0.336792 | 0.460205 | 2.834250 |
| 10,000 | 0.341064 | 0.457520 | 2.732560 |
| 10,250 | 0.332520 | 0.463867 | 2.764090 |
| 10,500 | 0.476196 | 0.716675 | 1.518050 |
| 10,750 | 0.808472 | 0.967407 | 0.480589 |
| 11,000 | 0.886353 | 0.985352 | 0.280869 |
| 12,000 | 0.957886 | 0.995972 | 0.107901 |
| 13,000 | 0.984985 | 0.997925 | 0.036757 |
| 14,000 | 0.973145 | 0.995239 | 0.070274 |
| 15,000 | 0.978516 | 0.998169 | 0.059532 |
| 16,000 | 0.991211 | 0.997559 | 0.024578 |
| 17,000 | 0.986206 | 0.995972 | 0.039653 |
| 18,000 | 1.000000 | 1.000000 | 0.000956 |
| 19,000 | 1.000000 | 1.000000 | 0.000429 |
| 20,000 | 1.000000 | 1.000000 | 0.000242 |

Threshold crossing in sampled telemetry:

| held-out threshold | first step |
| ---: | ---: |
| 0.50 | 10,750 |
| 0.80 | 10,750 |
| 0.90 | 11,250 |
| 0.99 | 14,500 |
| 0.999 | 17,500 |
| 1.000 | 18,000 |

The transient dip around steps `14,000` and `15,000` is visible in sampled
telemetry, but the run later stabilizes.

## Exhaustive Final Evaluation

Because telemetry was batch-limited, a full final-checkpoint evaluation was run
after training:

```text
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/full_final_eval.json
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/full_final_per_cell_accuracy.csv
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/full_final_errors.csv
```

Final exhaustive accuracy:

| split | examples | exact accuracy | mean loss |
| --- | ---: | ---: | ---: |
| train | 76,194 | 1.000000 | 0.000111 |
| held-out | 177,849 | 0.999871 | 0.000647 |

Held-out per-cell accuracy:

| statistic | value |
| --- | ---: |
| cells | 27 |
| min cell accuracy | 0.999241 |
| mean cell accuracy | 0.999871 |
| max cell accuracy | 1.000000 |
| total held-out errors | 23 |

The worst held-out cells were:

| cell | accuracy | errors / 6,587 |
| --- | ---: | ---: |
| `text+text->text` | 0.999241 | 5 |
| `text+text->number` | 0.999545 | 3 |
| `text+text->image` | 0.999545 | 3 |
| `text+image->number` | 0.999545 | 3 |
| `text+image->text` | 0.999696 | 2 |
| `text+image->image` | 0.999696 | 2 |
| `image+text->text` | 0.999696 | 2 |

The residual errors concentrate in cells with text operands. Examples include:

| pair | true `s` | wrong cell/output | prediction |
| --- | ---: | --- | ---: |
| `(30, 30)` | 60 | `text+text->number` | 70 |
| `(30, 30)` | 60 | `text+image->number` | 13 |
| `(9, 1)` | 10 | `text+image->number` | 11 |
| `(59, 59)` | 21 | `text+text->number` | 11 |
| `(60, 60)` | 23 | `text+text->number` | 33 |
| `(0, 1)` | 1 | `text+image->number` | 0 |

Some text-output failures were invalid answer-token sequences rather than a
different exact residue string. These are recorded with blank `pred` values in
`full_final_errors.csv`.

## Linear Probe Evidence

Answer-slot activations were extracted for all `9,409` operand pairs and all
27 cells:

```text
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/answer_slots.pt
```

The final-slot probe summary:

| target | train accuracy | test accuracy |
| --- | ---: | ---: |
| `s` | 1.000000 | 0.999764 |
| `wrap` | 0.916581 | 0.900466 |
| `mode_a` | 0.672140 | 0.672447 |
| `mode_b` | 0.717038 | 0.714977 |
| `output_mode` | 0.688611 | 0.688573 |
| `a` | 0.667244 | 0.042519 |
| `b` | 0.615652 | 0.033686 |

Interpretation:

- The answer slot linearly contains the answer residue almost perfectly.
- Operand identity is not linearly stable on held-out examples from the final
  answer slot, which argues against a simple operand-copying representation at
  the answer site.
- Modality and requested output mode remain moderately decodable. The final
  state is not modality-blind; it is residue-dominant with residual task-cell
  information.

The cross-modal `s` probe transfer matrix contains `729` train-cell to test-cell
entries. A residue probe trained on one cell transfers almost perfectly to other
cells:

| statistic | accuracy |
| --- | ---: |
| minimum transfer | 0.997267 |
| mean transfer | 0.999632 |
| maximum transfer | 1.000000 |

Worst transfers:

| train cell | test cell | accuracy |
| --- | --- | ---: |
| `image+number->text` | `text+image->text` | 0.997267 |
| `image+number->text` | `text+text->text` | 0.998178 |
| `number+text->text` | `text+image->text` | 0.998330 |
| `text+text->text` | `text+image->text` | 0.998330 |
| `text+image->text` | `text+image->text` | 0.998330 |

This is hard to reconcile with three unrelated modality-specific cyclic rules.
It is consistent with a common residue coordinate system at the final answer
slot.

## Rigorous Pair-Disjoint Probe Follow-Up

After the checkpoint-dynamics pass, a stricter probe sweep was run. See:

```text
RESULTS_TRI_MODAL_RIGOROUS_PROBES.md
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/
```

This audit samples `927` operand pairs, giving `25,029` answer-slot states
across the 27 cells. For each of five split seeds, the sampled train-side pairs
are split into `181` ridge-train pairs and `97` ridge-validation pairs; the
test split uses `649` originally held-out pairs. All splits are pair-disjoint
across every modality cell.

The strict run analyzes the final answer slot for all 20 numeric checkpoints
and all layers for checkpoints `10,000`, `11,000`, `12,000`, `13,000`, and
`20,000`. It uses ridge classifiers with seven L2 penalties, validation
selection, permutation controls for `s`, and full `27 x 27` cross-cell residue
transfer for every analyzed layer-unit.

Final strict-probe readout:

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

The `s` permutation control remains at chance: `0.010272` vs `1 / 97 =
0.010309`. This is a useful tightening over the broader final-checkpoint
probe: the final slot linearly contains the sum residue, while held-out operand
identity is at or below chance.

Strict final-slot checkpoint trajectory:

| step | held-out acc | final `s` probe | perm control | cross-cell transfer |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.005493 | 0.005878 | 0.010033 | 0.009718 |
| 5,000 | 0.238525 | 0.168316 | 0.010695 | 0.086986 |
| 10,000 | 0.341064 | 0.308669 | 0.010238 | 0.137520 |
| 11,000 | 0.886353 | 0.845426 | 0.010763 | 0.800614 |
| 12,000 | 0.957886 | 0.946105 | 0.010706 | 0.906745 |
| 13,000 | 0.984985 | 0.970941 | 0.008092 | 0.932001 |
| 18,000 | 1.000000 | 0.997398 | 0.011516 | 0.966167 |
| 20,000 | 1.000000 | 0.998277 | 0.010272 | 0.958581 |

The strict probes independently confirm the temporal story: final-slot sum
decodability and source-cell to target-cell residue transfer both become strong
at the grokking transition and cross `0.90` by checkpoint `12,000`.

The final `27 x 27` strict transfer mean is `0.958581`; the off-diagonal mean is
`0.957096`; the same-cell mean is `0.997204`. The worst transfer is `0.567026`,
which is much lower than the earlier all-state final transfer matrix. This is
not a contradiction. The strict transfer classifiers are trained from only one
source cell and only `181` sampled train pairs per split seed, then evaluated on
held-out pairs in every target cell. The residual weakness is concentrated in
text-output target cells:

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

The layerwise transition probes localize sharing in depth. Layers 0 and 1 stay
weak even at the final checkpoint. Layer 2 contains a growing but incomplete
shared sum signal. Layer 3 and the final answer slot carry the mature residue
code:

| step | layer | `s` probe | cross-cell transfer |
| ---: | ---: | ---: | ---: |
| 10,000 | 2 | 0.071015 | 0.046559 |
| 10,000 | 3 | 0.308429 | 0.134834 |
| 11,000 | 2 | 0.152976 | 0.267982 |
| 11,000 | 3 | 0.845084 | 0.791847 |
| 12,000 | 2 | 0.174034 | 0.342237 |
| 12,000 | 3 | 0.945980 | 0.897463 |
| 20,000 | 2 | 0.233305 | 0.553696 |
| 20,000 | 3 | 0.998174 | 0.932531 |
| 20,000 | final | 0.998277 | 0.958581 |

The refined conclusion is therefore: the model learns a shared late cyclic
answer representation, but the state is not perfectly modality-blind. Text
output demands leave a measurable format component in the answer-slot geometry.

## Multi-Seed Linear Probe Follow-Up

After the seed-`1706` robustness anomaly, the strict pair-disjoint probe sweep
was repeated for seed `1705`, and an expanded late-checkpoint probe sweep was
run for seed `1706`. See:

```text
RESULTS_TRI_MODAL_SEED1706_25K_PROBES.md
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1705/
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1706_expanded/
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_multiseed/
```

Seed `1705` used the same schedule as seed `1704`: final slot for every
checkpoint, and all layers at `10000`, `11000`, `12000`, `13000`, and `20000`.
Seed `1706` used an expanded schedule with all layers at every late saved
checkpoint from `10000` through `20000`, bracketing the unsaved sampled
telemetry peak at step `15500` with saved checkpoints `15000` and `16000`.

Key strict-probe comparison:

| seed | first final `s` >= 0.9 | first transfer >= 0.9 | best final `s` | best transfer | final `s` | final transfer | final layer-2 `s` | final layer-3 `s` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1704 | 12000 | 12000 | 0.998277 | 0.971721 | 0.998277 | 0.958581 | 0.233305 | 0.998174 |
| 1705 | 12000 | 12000 | 0.997740 | 0.983691 | 0.995811 | 0.983691 | 0.233750 | 0.995971 |
| 1706 | none | none | 0.834469 | 0.843850 | 0.815431 | 0.840452 | 0.130651 | 0.815306 |

The seed-`1705` replication is strong: it crosses the strict final-slot `s`
and cross-cell-transfer thresholds at step `12000`, like seed `1704`, and has
final cross-cell transfer `0.983691`.

The seed-`1706` result is more revealing. It never crosses `0.9` for either
strict final-slot `s` decodability or strict cross-cell transfer. The best
final-slot `s` probe is `0.834469` at step `13000`, while the best final-slot
transfer is `0.843850` at step `18000`. At the final checkpoint it remains at
`0.815431` for final-slot `s` and `0.840452` for transfer.

Seed `1706` late-checkpoint layer focus:

| step | layer 2 `s` | layer 3 `s` | final `s` | final transfer | sampled held-out |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 0.129270 | 0.772505 | 0.772334 | 0.741093 | 0.970581 |
| 12000 | 0.133322 | 0.815294 | 0.812555 | 0.774191 | 0.980957 |
| 13000 | 0.125823 | 0.836512 | 0.834469 | 0.790755 | 0.980469 |
| 15000 | 0.126885 | 0.824562 | 0.824722 | 0.824614 | 0.986450 |
| 16000 | 0.127900 | 0.817702 | 0.818947 | 0.828826 | 0.972290 |
| 18000 | 0.129259 | 0.823946 | 0.822964 | 0.843850 | 0.992188 |
| 20000 | 0.130651 | 0.815306 | 0.815431 | 0.840452 | 0.985596 |

This rules out the simple story that seed `1706` transiently formed the same
strict shared answer geometry at saved checkpoints and then lost only the
layer-2 operand-value precursor. Under the same pair-disjoint probe protocol,
its layer-2 `s` signal stays near `0.13`, while layer 3/final plateau around
`0.81-0.84` instead of the `0.99+` strong-seed regime. Seed `1706` therefore
looks like a high-accuracy but less clean, more cell-entangled solution.

## Operand Path Shared Operand-Value Test

A focused operand-path pass was then run to test whether a shared
representation of operand residue values in `Z_97` exists immediately before
the cyclic answer representation forms. See:

```text
RESULTS_TRI_MODAL_OPERAND_PATH.md
tri_modal_modular_grokking/analysis/phase4_operand_path/
```

The pass analyzes checkpoints `10,000`, `11,000`, `12,000`, `13,000`, and
`20,000`, layers `0..3`, the same `25,029` sampled pair-cell states, five split
seeds, and `256` operand-replacement patch examples per diagnostic spec.

The key temporal fact is that the answer code appears between layer 2 and layer
3:

| final checkpoint marker | value |
| --- | ---: |
| layer-2 answer `s` probe | 0.233305 |
| layer-3 answer `s` probe | 0.998174 |
| layer-2 operand-A `a` probe | 0.539873 |
| layer-2 operand-B `b` probe | 0.528117 |

At that same layer-2 pre-cyclic point, text and image operand states are highly
linearly alignable to number-modality operand states with the same underlying
residue value:

| final layer-2 source | same-value nearest-neighbor acc in number-modality space | reconstruction R2 |
| --- | ---: | ---: |
| text operand states -> number-modality operand states | 0.986235 | 0.884350 |
| image operand states -> number-modality operand states | 0.964920 | 0.853711 |

For seed `1704`, this is strong evidence for a linearly aligned operand
residue-value representation before the mature cyclic answer code. It is not
evidence that modalities have collapsed to the same raw coordinate system: raw
nearest accuracy in the number-modality state space at final layer 2 is only
`0.381099` for text and `0.100668` for image. The relationship is mostly
affine/linear, and the activation vector is a representation of an operand
value, not a number itself.

The causal operand replacement results are informative but more limited.
Exact-token same-image patching at the embedding layer is a positive control:
it switches clean answers with accuracy `1.000000`. Pooled number-modality
state patches partially substitute into text/image operand targets early, but
text/image pooled states do not substitute well into number-modality targets,
and content-token patches after layer 2 have no effect.

Single-seed interpretation:

- geometry strongly supports a shared operand-value representation before
  cyclic answer formation;
- the number modality appears privileged relative to text/image in early pooled
  patching;
- layer-2 content-token patching is not a decisive causal negative, because by
  then operand information has likely already been copied into the answer query
  or other residual-stream states.

This motivated the follow-up carrier-state test: layer-2 answer-query
subspace patching, plus null controls and repeat seeds.

## Operand-Value Robustness And Seed Repeat

The operand-path finding was then stress-tested because the initial analysis
was a single-seed, sampled-state result. See:

```text
RESULTS_TRI_MODAL_OPERAND_VALUE_ROBUSTNESS.md
tri_modal_modular_grokking/analysis/phase4_operand_value_robustness_multiseed/
```

The follow-up implements the six requested robustness checks:

- exhaustive all-pair/all-cell layer-2 alignment;
- target-permuted ridge and random orthogonal null controls;
- train-on-one-context and test-on-held-out-context transfer;
- low-rank ridge and CCA rank sweeps;
- layer-2 answer-query subspace patching;
- repeat seeds `1704`, `1705`, and `1706`.

Training outcome across the three final checkpoints:

| seed | sampled final held-out acc | exhaustive train acc | exhaustive held-out acc | held-out errors |
| ---: | ---: | ---: | ---: | ---: |
| 1704 | 1.000000 | 1.000000 | 0.999871 | 23 |
| 1705 | 1.000000 | 1.000000 | 0.999854 | 26 |
| 1706 | 0.985596 | 0.992742 | 0.989255 | 1,911 |

Seed `1706` is not a total capability failure, but its final checkpoint is not
in the same regime as `1704` and `1705`. Its sampled telemetry peaked earlier:
step `15,500` reached `0.996826` sampled held-out accuracy, while the final
step `20,000` checkpoint evaluated exhaustively at `0.989255`.

Exhaustive layer-2 text/image-to-number-modality alignment:

| seed | text -> number-modality NN | image -> number-modality NN |
| ---: | ---: | ---: |
| 1704 | 0.999990 | 0.997866 |
| 1705 | 0.974809 | 0.960872 |
| 1706 | 0.125522 | 0.121310 |

Null controls stayed at chance. Chance is `1 / 97 = 0.010309`:

| control | source | mean NN | max NN |
| --- | --- | ---: | ---: |
| random orthogonal | image | 0.009031 | 0.018825 |
| random orthogonal | text | 0.007044 | 0.027934 |
| target-permuted ridge | image | 0.010836 | 0.018977 |
| target-permuted ridge | text | 0.010294 | 0.024138 |

Context-transfer is stricter than same-cell alignment because it trains the map
in one source context/output cell and tests it in held-out context/output
cells:

| seed | text transfer mean | text min | image transfer mean | image min |
| ---: | ---: | ---: | ---: | ---: |
| 1704 | 0.812425 | 0.222863 | 0.747643 | 0.205405 |
| 1705 | 0.874936 | 0.499317 | 0.719288 | 0.049947 |
| 1706 | 0.088560 | 0.016851 | 0.093357 | 0.020799 |

Low-rank ridge shows that the strong seeds use a moderately broad subspace. At
rank `64`, seeds `1704` and `1705` are already close to full-rank performance;
seed `1706` remains weak:

| seed | source | rank 8 | rank 16 | rank 32 | rank 64 | rank 128 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1704 | text | 0.210217 | 0.377135 | 0.737962 | 0.995852 | 0.999990 |
| 1704 | image | 0.281485 | 0.526977 | 0.852637 | 0.995668 | 0.997866 |
| 1705 | text | 0.501292 | 0.793867 | 0.923624 | 0.967503 | 0.974807 |
| 1705 | image | 0.596955 | 0.887270 | 0.935680 | 0.957161 | 0.960872 |
| 1706 | text | 0.051010 | 0.067832 | 0.084044 | 0.112152 | 0.125520 |
| 1706 | image | 0.053332 | 0.071066 | 0.091972 | 0.113486 | 0.121310 |

CCA gives a subtler readout. Seed `1706` reaches high CCA nearest-neighbor
accuracy at ranks `32` and `64`, so it likely contains shared operand
information in some latent sense. The failure is more precise: seed `1706`
does not expose that information in the same direct number-modality coordinate
chart, and it does not use the same layer-2 causal answer-query carrier.

| seed | source | CCA rank 8 | CCA rank 16 | CCA rank 32 | CCA rank 64 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1704 | text | 0.989309 | 1.000000 | 1.000000 | 0.994491 |
| 1704 | image | 0.996648 | 1.000000 | 1.000000 | 0.998752 |
| 1705 | text | 0.995707 | 0.998010 | 1.000000 | 0.993774 |
| 1705 | image | 0.956649 | 0.968873 | 0.995982 | 0.975573 |
| 1706 | text | 0.354872 | 0.889764 | 0.997763 | 0.982364 |
| 1706 | image | 0.152197 | 0.507603 | 0.956831 | 0.963045 |

Layer-2 answer-query patching resolves the earlier causal uncertainty for the
two strong seeds:

| seed | full-state clean-answer acc | corrupt-answer acc |
| ---: | ---: | ---: |
| 1704 | 1.000000 | 0.000000 |
| 1705 | 0.999512 | 0.000000 |
| 1706 | 0.106445 | 0.061523 |

Low-dimensional causal subspace patching shows that the layer-2 carrier is
compact in the strong seeds:

| seed | basis | rank 8 clean acc | rank 16 clean acc | high-rank clean acc |
| ---: | --- | ---: | ---: | ---: |
| 1704 | answer PCA | 0.989746 | 0.999512 | 1.000000 |
| 1704 | sum-centroid PCA | 0.991211 | 1.000000 | 1.000000 |
| 1705 | answer PCA | 0.965820 | 0.999512 | 0.999512 |
| 1705 | sum-centroid PCA | 0.971680 | 0.999512 | 0.999512 |
| 1706 | answer PCA | 0.108398 | 0.104980 | 0.106445 |
| 1706 | sum-centroid PCA | 0.076172 | 0.104980 | 0.106445 |

The revised conclusion is therefore:

> A shared operand residue-value representation immediately before cyclic
> answer formation is a stable feature of the fully grokked tri-modal solutions
> found by seeds `1704` and `1705`. It is not guaranteed by the architecture or
> objective alone. Seed `1706` reaches high final accuracy without the same
> direct layer-2 coordinate chart or causal answer-query carrier.

## Fourier Evidence

The Fourier diagnostic computes energy over the operand grid for each cell. The
key quantity is energy on the addition diagonal, which corresponds to functions
of `(a + b) mod p`.

Summary over all 27 cells:

| component | min | mean | max |
| --- | ---: | ---: | ---: |
| addition diagonal | 0.920992 | 0.937730 | 0.944334 |
| `a` only | 0.004541 | 0.007239 | 0.014055 |
| `b` only | 0.004147 | 0.006886 | 0.013602 |
| difference diagonal | 0.001783 | 0.002079 | 0.002874 |
| other energy | 0.040599 | 0.046065 | 0.055049 |

Worst addition-diagonal cells:

| cell | addition energy | other energy | top frequency |
| --- | ---: | ---: | ---: |
| `text+text->text` | 0.920992 | 0.054513 | 2 |
| `text+image->text` | 0.926054 | 0.052754 | 2 |
| `text+text->number` | 0.927008 | 0.055049 | 39 |
| `text+text->image` | 0.927499 | 0.054728 | 39 |
| `text+image->number` | 0.930258 | 0.054103 | 19 |

Best addition-diagonal cells:

| cell | addition energy | other energy | top frequency |
| --- | ---: | ---: | ---: |
| `number+number->image` | 0.944334 | 0.042226 | 39 |
| `number+number->number` | 0.944260 | 0.042217 | 39 |
| `image+number->image` | 0.943303 | 0.044869 | 19 |
| `image+number->number` | 0.943148 | 0.044972 | 19 |
| `number+image->number` | 0.943037 | 0.045245 | 3 |

Interpretation:

- All cells concentrate most answer-slot energy on the modular-addition
  diagonal.
- Text-heavy cells have slightly weaker addition-diagonal purity and account
  for most residual output errors.
- The top individual frequency varies by cell, but this does not undermine the
  shared-rule interpretation: the diagnostic is measuring total diagonal energy,
  not requiring one single scalar Fourier frequency to dominate every cell.

## Activation Patching Evidence

The first full answer-slot patching audit used:

| item | value |
| --- | --- |
| source cell | `text+image->number` |
| target cell | `number+number->number` |
| layer | final answer slot, `-1` |
| patch basis | full answer-slot vector |
| examples | 128 |

Result:

| condition | clean-answer acc | corrupt-answer acc |
| --- | ---: | ---: |
| baseline corrupt run | 0.000000 | 1.000000 |
| patched run | 1.000000 | 0.000000 |

In other words, replacing the final answer slot with the source-cell answer
slot fully switches the target output to the source answer over this audit set.
This is direct causal evidence that the final answer slot carries answer
information in a form usable by the number output head across modality cells.

The current patching evidence is deliberately limited:

- it uses full-state patching, not a low-rank or Fourier-subspace patch;
- it covers one source-target pair, not all 27 x 27 cell pairs;
- it patches the final answer slot, not earlier computation layers.

It is still an important positive control: the answer-slot representation is
not merely probe-visible; it is causally sufficient for the output head.

## Hypothesis Adjudication

### H1: One Shared Cyclic/Fourier Rule

Status: strongly supported for fully grokked seeds `1704` and `1705`; partially
supported but not clean for seed `1706`.

Evidence:

- near-perfect full held-out performance across all 27 modality cells;
- final answer slot is almost perfectly residue-decodable;
- residue probes transfer across cells with mean accuracy `0.999632`;
- strict pair-disjoint sampled probes reach final `s` accuracy `0.998277` and
  cross-cell transfer mean `0.958581`;
- all cells show high addition-diagonal Fourier energy, mean `0.937730`;
- full answer-slot patching transfers the answer across modality cells.
- the strict probe result replicates in seed `1705`, with final-slot `s`
  accuracy `0.995811` and cross-cell transfer `0.983691`.

Seed `1706` weakens the seed-universal version of this claim. Its strict
final-slot `s` probe is above chance and substantial, but remains below the
strong-seed regime: best `0.834469`, final `0.815431`, and final cross-cell
transfer `0.840452`. This is evidence for a less linearly organized or more
cell-entangled answer representation, not for a clean copy of the seed
`1704`/`1705` shared answer space.

The strongest statement justified by the current run is:

> Fully grokked phase-4 seeds learn a shared, residue-dominant cyclic answer
> space that is used causally by at least the number output head. Seed `1706`
> shows that high task accuracy can be achieved with a weaker strict linear
> shared-answer geometry.

### H2: Three Separate Cyclic/Fourier Rules

Status: disfavored, not mathematically impossible.

Separate rules would predict weak cross-modal probe transfer and poor
answer-slot alignment. Instead, all cell-to-cell `s` probes transfer at
`>= 0.997267`. This makes three unrelated modality-specific final answer spaces
unlikely.

The caveat is that the current run directly trains all 27 cells. A model could
still use partially separate early pathways that converge to a shared final
answer coordinate. The strict transition probes localize readable sharing to
late layer 3/final answer slots. The phase-6 directed route path-tracing
follow-up confirms that early route carriers are order/modality-specific even
when they can be causally rescued into the same late answer state.

### H3: Translation-To-Value-Coordinate Strategy

Status: supported for strong seeds, not seed-universal.

The older shorthand for this hypothesis was "translation-to-number." More
precisely, the model may translate every operand into a shared
numeric/value-coordinate internal representation of its residue class in
`Z_97` before applying a shared rule. This is a claim about representation,
coordinate charts, and causal carriers, not about an activation literally being
a mathematical number.

Evidence for a translation-to-value-coordinate story in the strong seeds:

- at final layer 2, seed `1704` text/image-to-number-modality operand maps
  reach exhaustive same-value nearest-neighbor accuracy `0.999990` and
  `0.997866`;
- the same diagnostic replicates in seed `1705` at `0.974809` and `0.960872`;
- target-permuted ridge and random orthogonal controls stay at chance;
- context-transfer maps remain far above chance in seeds `1704` and `1705`;
- layer-2 answer-query full-state patching switches clean answers with
  accuracy `1.000000` in seed `1704` and `0.999512` in seed `1705`;
- rank-8 answer-PCA and sum-centroid-PCA subspace patches already recover most
  of that causal effect in both strong seeds;
- the answer `s` is still weak at layer 2 and becomes mature in layer 3,
  placing the operand-value geometry just before cyclic answer formation.

Evidence against an overly simple raw-collapse story:

- the answer slot does not linearly preserve held-out operand identities well;
- raw text/image operand states are not already in the same coordinate basis as
  number-modality states, especially for image operands;
- output mode and input modality remain decodable;
- the seed-`1706` final checkpoint reaches high accuracy but has weak direct
  ridge nearest-neighbor alignment: `0.125522` for text and `0.121310` for
  image;
- seed `1706` has high-rank CCA evidence for shared operand information, so
  the negative result is not "no shared operand information"; it is "no shared
  direct layer-2 number-modality coordinate chart and no matching layer-2
  causal answer-query carrier at the final checkpoint."

Evidence still needed:

- rerun the robustness battery on seed `1706` best checkpoints around steps
  `15,000` and `15,500`;
- use the completed single-combination leave-outs to test localized route
  carriers with all-layer operand probes and low-rank subspace patching;
- compare against text-image alignment to determine whether number is uniquely
  privileged or simply one coordinate chart among several;
- extend from the four phase-6 directed route path-tracing runs to all-cell
  and low-rank subspace causal patching.

### H4: No True Shared Rule

Status: ruled out for strong seeds `1704` and `1705`; inadequate but not cleanly
ruled out by the same evidence for seed `1706`.

The model generalizes across held-out operand pairs, reaches `0.999871` full
held-out exact accuracy, has near-perfect residue probes on all pair-cell
states, and shows high addition-diagonal Fourier energy. This is not surface
agreement or train-pair memorization.

For seed `1706`, task generalization is still high (`0.989255` exhaustive
held-out accuracy), and strict final-slot transfer is far above chance
(`0.840452`). That makes "no shared rule" too pessimistic. However, seed
`1706` does not meet the strong-seed standard for a clean shared linear answer
space, so it should be treated as a separate mechanistic regime until further
patching/Fourier diagnostics are run.

## Relationship To Prior Branch Results

This run combines lessons from the earlier branch:

- The supervised transformer and MicroRWKV showed that modular addition can be
  learned as a cyclic/Fourier rule.
- The JEPA results showed that same-sum structure must be encouraged or read
  out carefully; high-level objectives can otherwise memorize target latents.
- The word decoder showed that English answer generation can ride on a cyclic
  scaffold but may mix arithmetic and language features.
- The image I-JEPA failures showed that image-space latent prediction alone is
  not enough to induce a modular residue rule.
- The QNN results showed that Fourier structure can be present without exact
  finite-residue selection.

The tri-modal experiment is the first branch result to train one shared
backbone jointly across number, word, and image operands and across number,
word, and image answer channels. Its final answer slot is cleaner and more
shared than the word decoder's late language state, and its image modality
works under direct answer-slot supervision unlike image I-JEPA.

## Checkpoint Dynamics Follow-Up

A checkpoint-by-checkpoint dynamics pass was run after the final-checkpoint
audit. See:

```text
RESULTS_TRI_MODAL_CHECKPOINT_DYNAMICS.md
tri_modal_modular_grokking/analysis/phase4_checkpoint_dynamics/
```

That pass studies three questions across all numeric checkpoints:

1. when the answer slot becomes a shared residue code;
2. when Fourier/addition-diagonal structure forms;
3. when cross-modal answer-slot patching becomes causally sufficient.

The main temporal result is that cross-cell residue alignment snaps into place
at step `11,000`, causal final-slot patching becomes strong by step `12,000`,
and selected full-grid Fourier addition energy continues sharpening through
step `20,000`. The pass generated 60 per-checkpoint PNGs, three for each
checkpoint, plus two GIF animations of cyclic-rule formation.

![phase4 transition geometry](tri_modal_modular_grokking/analysis/phase4_checkpoint_dynamics/figures/step_12000_geometry.png)


## Leave-Combination-Out Follow-Up

The phase-5 leave-combination-out run was trained after the phase-4 shared-rule
baseline. It used the same architecture and heads, but trained only these input
combinations:

```text
number+number
text+text
image+image
number+text
text+number
```

The held-out input combinations were:

```text
image+text
text+image
image+number
number+image
```

All three output modes were included for every trained or held-out input
combination. The run completed to step `20,000`, with checkpoints every `1,000`
steps.

Behaviorally, the final train/heldout split is misleading if read only as an
aggregate. The model learned every no-image input combination and generalized
those combinations to held-out operand pairs, but it did not learn image-input
combinations:

| final group | mean heldout-pair accuracy |
| --- | ---: |
| no-image input cells | 0.999987 |
| image-input cells | 0.009544 |
| trained input combos overall | 0.801923 |

Mechanistic diagnostics agree with that localization:

| diagnostic | no-image cells | image-input cells |
| --- | ---: | ---: |
| final-slot `s` probe | 0.995041 | 0.002679 |
| Fourier addition-diagonal energy | 0.959461 | 0.000098 |
| cross-cell transfer from no-image sources | 0.991 to no-image targets | 0.0097 to image targets |

The decisive causal test used same-pair answer-slot patching. Over `512`
heldout pairs, patching a good `number+text->number` final answer slot into
image-input targets gave `1.000000` target accuracy. The reverse patch from
`image+image` into successful no-image targets gave only `0.009766`, matching
the image source baseline, while the no-image target baseline was `0.999837`.

Layerwise no-image-to-image patching shows that the answer state becomes
causally sufficient by layer 2:

![phase5 behavior](tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/figures/final_behavior_heldout_pairs.png)

![phase5 patching](tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/figures/final_same_pair_patching.png)

| layer | number output | text output | image output |
| ---: | ---: | ---: | ---: |
| 0 | 0.1027 | 0.0703 | 0.1031 |
| 1 | 0.8516 | 0.6777 | 0.8445 |
| 2 | 1.0000 | 0.9992 | 1.0000 |
| 3 | 1.0000 | 1.0000 | 1.0000 |
| final | 1.0000 | 1.0000 | 1.0000 |

Interpretation: this phase-5 run is not a negative result for the existence of
a shared cyclic rule. It shows that the no-image path learned such a rule, but
the image-input encoder/routing path did not translate image operands into the
shared answer-state computation. The failure is upstream of the small decoders
and upstream of the final answer-slot readout.

Primary artifacts:

```text
RESULTS_TRI_MODAL_LEAVE_COMBO_OUT.md
tri_modal_modular_grokking/runs/phase5_leave_combo_out/
tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/
```

### Phase-6 Single-Combination Control

The next control held out only one ordered input combination, `image+text`, and
trained all other ordered input combinations. This directly tests zero-shot
composition after image operands have been learned in trained contexts.

Training completed to step `20,000`. The final sampled training telemetry
reported trained-cell accuracy `1.000000`, omitted-combination accuracy
`0.471924`, trained-cell loss `0.000145`, and omitted-combination loss
`2.945795`.

The exhaustive final heldout-pair analysis shows that image inputs were learned
when they appeared in trained combinations:



| input-combination group | final mean heldout-pair accuracy |
| --- | ---: |
| all trained input combinations | 1.000000 |
| trained image-input combinations | 1.000000 |
| omitted `image+text` combination | 0.471079 |

Final omitted-cell accuracies:

| cell | heldout-pair accuracy |
| --- | ---: |
| `image+text->number` | 0.475330 |
| `image+text->text` | 0.458479 |
| `image+text->image` | 0.479429 |

The omitted route becomes increasingly structured over checkpoints but remains
below the trained image-input regime:

| step | omitted behavior | trained image behavior | omitted `s` probe | trained image `s` probe | omitted Fourier | trained image Fourier |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 0.328981 | 0.992966 | 0.296949 | 0.985367 | 0.365861 | 0.891704 |
| 12000 | 0.379839 | 0.994105 | 0.365855 | 0.991676 | 0.396048 | 0.898598 |
| 13000 | 0.406407 | 0.992789 | 0.400806 | 0.991545 | 0.400300 | 0.888568 |
| 15000 | 0.441020 | 1.000000 | 0.468617 | 0.999713 | 0.487228 | 0.940924 |
| 17000 | 0.448408 | 1.000000 | 0.482196 | 1.000000 | 0.502065 | 0.971746 |
| 20000 | 0.471079 | 1.000000 | 0.555623 | 0.999996 | 0.536068 | 0.956771 |

Full answer-slot patching localizes the failure. Patching a mature
`number+text->number` answer state into the omitted `image+text->number`
target gives:

| patched layer | target baseline | patched target accuracy |
| ---: | ---: | ---: |
| 0 | 0.443359 | 0.447266 |
| 1 | 0.443359 | 0.607422 |
| 2 | 0.443359 | 0.865234 |
| 3 | 0.443359 | 1.000000 |
| final | 0.443359 | 1.000000 |

The reverse control patches the omitted `image+text` source into trained
targets:

| patched layer | omitted-source baseline | patched target accuracy |
| ---: | ---: | ---: |
| 0 | 0.442057 | 1.000000 |
| 1 | 0.442057 | 0.863839 |
| 2 | 0.442057 | 0.793341 |
| 3 | 0.442057 | 0.442057 |
| final | 0.442057 | 0.442057 |

Interpretation: phase 6 is partial zero-shot transfer, not a solved
compositional modality rule. The trained image paths and all output heads are
capable. The omitted `image+text` route has early compatible information, as
shown by layer-0 omitted-source patches into trained targets, but its own later
states do not become the mature shared cyclic answer representation.

Primary artifacts:

```text
RESULTS_TRI_MODAL_LEAVE_IMAGE_TEXT.md  # exhaustive phase-6 report
tri_modal_modular_grokking/configs/phase6_leave_image_text.yaml
tri_modal_modular_grokking/runs/phase6_leave_image_text/
tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp/
```

### Phase-6 Reverse-Order Control

The matched reverse-order run held out only `text+image` and trained every
other ordered input combination. This is the direct control for whether the
partial `image+text` zero-shot route is symmetric.

It is not symmetric. Training completed to step `20,000`. The final sampled
training telemetry reported trained-cell accuracy `0.996704`,
omitted-combination accuracy `0.109863`, trained-cell loss `0.006428`, and
omitted-combination loss `9.046402`. The best sampled omitted accuracy was
`0.136108` at step `15,000`.

The exhaustive final heldout-pair result:

| input-combination group | final mean heldout-pair accuracy |
| --- | ---: |
| all trained input combinations | 0.991258 |
| trained image-input combinations | 0.987324 |
| omitted `text+image` combination | 0.109256 |

Final omitted-cell accuracies:

| cell | heldout-pair accuracy |
| --- | ---: |
| `text+image->number` | 0.110217 |
| `text+image->text` | 0.103993 |
| `text+image->image` | 0.113557 |

The checkpoint dynamics show a weak route that peaks near step `15,000` and
does not become a mature answer state:

| step | omitted behavior | trained image behavior | omitted `s` probe | trained image `s` probe | omitted Fourier | trained image Fourier |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | 0.049694 | 0.603664 | 0.024864 | 0.382146 | 0.057878 | 0.569606 |
| 12000 | 0.080664 | 0.978923 | 0.044161 | 0.741123 | 0.112933 | 0.853097 |
| 13000 | 0.112545 | 0.994396 | 0.064049 | 0.842860 | 0.145046 | 0.871900 |
| 15000 | 0.129902 | 0.999583 | 0.074996 | 0.912070 | 0.140563 | 0.896531 |
| 17000 | 0.079500 | 0.999937 | 0.051398 | 0.954759 | 0.077097 | 0.918435 |
| 20000 | 0.109256 | 0.987324 | 0.058617 | 0.918813 | 0.098098 | 0.834497 |

The final-slot probes are diagnostic:

| target | omitted `text+image` | trained image-input combos | no-image combos |
| --- | ---: | ---: | ---: |
| `a` | 0.606498 | 0.447852 | 0.397374 |
| `b` | 0.340958 | 0.419319 | 0.409304 |
| `s` | 0.058617 | 0.918813 | 0.932978 |

The omitted state preserves substantial information about the first operand
`a`, which is text in this route, but it does not represent the sum.

Fourier analysis agrees:

| group | addition diag | `a` only | `b` only | other |
| --- | ---: | ---: | ---: | ---: |
| omitted `text+image` | 0.098098 | 0.161112 | 0.089640 | 0.640882 |
| trained image-input combos | 0.834497 | 0.026123 | 0.025151 | 0.108434 |
| no-image combos | 0.846999 | 0.024437 | 0.025026 | 0.097815 |

Full answer-slot patching again rules out decoder failure:

| causal patch | value |
| --- | ---: |
| no-image final patch into omitted `text+image` target | 0.994466 |
| omitted-source layer-0 patch into trained targets | 0.905273 |
| omitted-source final patch into trained targets | 0.097656 |

Interpretation: `text+image` is not merely a weaker copy of `image+text`; it is
a qualitatively poorer ordered route. The target heads can read a mature answer
state, and early omitted-route states are useful to trained downstream
computation, but the omitted route's own later layers do not produce the shared
cyclic answer representation.

Primary artifacts:

```text
RESULTS_TRI_MODAL_LEAVE_TEXT_IMAGE.md  # exhaustive reverse-order phase-6 report
tri_modal_modular_grokking/configs/phase6_leave_text_image.yaml
tri_modal_modular_grokking/runs/phase6_leave_text_image/
tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp/
```

### Phase-6 Long Continuation: `number+image` To 40k

The first long-continuation test resumed the original phase-6
`number+image` leave-out checkpoint from step `20,000` and trained to
`40,000`. This was the most performant unsupervised phase-6 omitted route at
20k, so it was the right case for testing whether partial zero-shot route
formation was a stable plateau, a delayed grokking precursor, or an unstable
route attractor.

The continuation used:

```text
tri_modal_modular_grokking/runs/phase6_leave_number_image/checkpoint_20000.pt
tri_modal_modular_grokking/configs/phase6_leave_number_image_to40000.yaml
tri_modal_modular_grokking/runs/phase6_leave_number_image_to40000/
```

The combined train/loss curve is the critical artifact:

![40k train loss](tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/figures/train_loss_curve_combined.png)

Key sampled telemetry:

| event | step | sampled heldout acc | heldout loss | train acc | train loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| original endpoint | 20,000 | 0.533325 | 2.031925 | 0.993042 | 0.013930 |
| continuation peak | 31,750 | 0.894531 | 0.301698 | 1.000000 | 0.000131 |
| collapse minimum | 35,000 | 0.007324 | 3.539201 | 0.507568 | 1.706982 |
| recovery peak | 37,250 | 0.852173 | 0.491232 | 1.000000 | 0.000682 |
| final sampled endpoint | 40,000 | 0.682129 | 1.215100 | 1.000000 | 0.000201 |

The final exact heldout-route evaluation at step `40,000` used all `19,761`
heldout `number+image` examples and gave route mean accuracy `0.681089`.
Per-head exact accuracies were `0.687263` for number output, `0.667831` for
text output, and `0.688174` for the image-class proxy output.

The key mechanistic point is that the continuation contains two different
failure modes. The collapse at `35,000` is not a pure omitted-route failure:
train accuracy also drops to `0.507568`, so that event is a global optimizer
or basin-switching instability. The later decline from `37,250` to `40,000`
is more specifically about omitted-route stability: train accuracy is back at
`1.000000`, while omitted-route accuracy falls from `0.852173` to exact
`0.681089`. This supports an unstable-route-formation reading rather than a
stable plateau or a monotone delayed-grokking story.

The all-seed phase-8 reruns bound the claim. The phase-8 seed-`1705`
`number+image` rerun, with the same nominal seed and route, finished at exact
omitted-route accuracy `0.053489` at 20k. Therefore the 40k continuation shows
that one partially successful omitted route can enter and leave a high-
accuracy shared-rule-compatible regime, but it does not establish deterministic
reproducibility of that route under the nominal seed label.

Primary artifacts:

```text
RESULTS_TRI_MODAL_LEAVE_NUMBER_IMAGE_40K_EXTENSION.md
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/key_step_train_loss_summary.csv
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/continuation_summary.json
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/final_exact_eval.json
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/figures/train_loss_curve_combined.png
tri_modal_modular_grokking/make_leaveout_training_curve_figures.py
```

### Phase-6 Directed Route Path Tracing

The next high-information follow-up was layerwise directed route path tracing
over all four completed single-combination omissions. A dedicated synthesis is
stored in:

```text
RESULTS_TRI_MODAL_DIRECTED_ROUTE_PATH_TRACING.md
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/
```

The analysis patches same-pair activations between a mature no-image source
and an omitted target, and in the reverse direction from an omitted source into
mature no-image targets. It covers all three output modes, `256` heldout
pairs, nine semantic sites, and the embedding plus all layer-0..3
residual/attention/MLP components. Each run produces `4,536` patch rows, so the
four-route sweep contains `18,144` causal patch measurements.

Patched sites:

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

Behavioral context for the four omitted routes:

| omitted route | final behavior | final `s` probe | Fourier addition energy | route class |
| --- | ---: | ---: | ---: | --- |
| `image+text` | 0.471079 | 0.555623 | 0.536068 | partial |
| `text+image` | 0.109256 | 0.058617 | 0.098098 | failure |
| `number+image` | 0.536511 | 0.351703 | 0.567461 | partial |
| `image+number` | 0.108800 | 0.103014 | 0.079796 | failure |

Headline path-tracing summary:

| omitted route | rows | best mature-source into omitted | best omitted-source into mature | elapsed seconds |
| --- | ---: | ---: | ---: | ---: |
| `image+text` | 4,536 | 1.000000 | 1.000000 | 824.90 |
| `text+image` | 4,536 | 1.000000 | 1.000000 | 814.52 |
| `number+image` | 4,536 | 1.000000 | 1.000000 | 724.09 |
| `image+number` | 4,536 | 1.000000 | 1.000000 | 727.56 |

The `1.000000` maxima should not be overread. Same-pair late
`answer_query` patches are close to answer-state overwrites, so they are a
positive control that the target heads and target context can read a mature
answer state. The diagnostic result is the non-answer rescue pattern and the
damage caused by omitted-source patches into mature targets.

Mature-source rescue into omitted targets:

| omitted route | path-tracing baselines, number/text/image | best non-answer rescue | best non-answer site | answer-query mean | operand-pool max |
| --- | --- | ---: | --- | ---: | ---: |
| `image+text` | 0.460938 / 0.453125 / 0.468750 | 1.000000 | L0 `mlp_out` `operand_b_pool` | 0.6402 | 1.000000 |
| `text+image` | 0.082031 / 0.082031 / 0.085938 | 0.785156 | L0 `mlp_out` `operand_b_pool` | 0.3642 | 0.785156 |
| `number+image` | 0.574219 / 0.507812 / 0.566406 | 0.785156 | L0 `attn_out` `plus` | 0.8237 | 0.652344 |
| `image+number` | 0.089844 / 0.097656 / 0.093750 | 1.000000 | L0 `resid_mid` `operand_b_pool` | 0.4364 | 1.000000 |

Omitted-source damage into mature targets:

| omitted source | `operand_a_pool` mean drop | `operand_b_pool` mean drop | `answer_query` mean drop | strongest drop |
| --- | ---: | ---: | ---: | ---: |
| `image+text` | 0.4704 | 0.4642 | 0.1436 | 0.992188 |
| `text+image` | 0.4537 | 0.4888 | 0.3960 | 0.992188 |
| `number+image` | 0.0541 | 0.0925 | 0.2004 | 0.980469 |
| `image+number` | 0.3959 | 0.2542 | 0.4357 | 0.992188 |

Interpretation:

- All four omitted targets can be rescued by mature late answer-query states,
  so the shallow heads and target answer readout are not the bottleneck.
- The partial routes are not symmetric with the failure routes. `image+text`
  and `number+image` have substantially stronger final behavior and cyclic
  evidence; `text+image` and `image+number` remain near chance.
- The strongest non-answer rescue often appears at layer-0 carriers. These
  are not necessarily pure operand values after attention has mixed the
  sequence; `operand_b_pool` can already behave as an early routed accumulator.
- Omitted-source states can actively corrupt mature routes. This rules out the
  interpretation that failed routes are merely blank. They enter nearby but
  wrong or incomplete carrier states.
- The best current mechanistic description is a directed route grammar: the
  shared cyclic answer representation exists, but the model must route operand
  information into it through learned ordered-modality carriers.

The eight generated heatmaps are part of the report artifact, not auxiliary
scratch outputs:

`image+text`, mature sources into omitted target:


`image+text`, omitted source into mature targets:


`text+image`, mature sources into omitted target:


`text+image`, omitted source into mature targets:


`number+image`, mature sources into omitted target:


`number+image`, omitted source into mature targets:


`image+number`, mature sources into omitted target:

![image+number good-to-omitted heatmap](tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/figures/good_to_omitted_heatmap.png)

`image+number`, omitted source into mature targets:


### Phase-6 Route Operand Probes

Priority 2 from the directed-route plan was then run to test whether failed
routes encode operands but fail to compose them, or whether the relevant
operand is missing before composition. The dedicated report is:

```text
RESULTS_TRI_MODAL_ROUTE_OPERAND_PROBES.md
tri_modal_modular_grokking/analysis/phase6_route_operand_probes/
```

The probe pass used the same four phase-6 single-combination leave-out models
at the final checkpoint. It fitted strict pair-disjoint ridge probes for `a`,
`b`, and `s` at all nine semantic sites and all embedding/layer-0..3
residual/attention/MLP components, using `927` sampled pairs, three split
seeds, and `27,216` total probe rows. It also generated `24` heatmap PNGs:
local omitted-route probes and mature-to-omitted transfer probes for `a`, `b`,
and `s` on all four omitted routes.

The key distinction is local decodability versus transfer decodability. A
local probe can decode information in a route-specific coordinate. A
mature-to-omitted transfer probe asks whether the omitted state uses the mature
no-image coordinate system.

Best post-embedding results:

| omitted route | best local `a` | best local `b` | best local `s` | best mature->omitted `s` | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `image+text` | 0.826 | 0.687 | 0.702 | 0.487 | strongest partial shared-coordinate route |
| `text+image` | 0.798 | 0.776 | 0.366 | 0.099 | operands and some sum are local, but mature sum coordinate is absent |
| `number+image` | 0.801 | 0.775 | 0.355 | 0.266 | weaker partial shared-coordinate route |
| `image+number` | 0.812 | 0.822 | 0.522 | 0.104 | local sum exists but is not in mature/readout-compatible coordinate |

Path-tracing hotspot probe results:

| route | hotspot | local omitted `a/b/s` | mature->omitted `a/b/s` | read |
| --- | --- | --- | --- | --- |
| `image+text` | L0 `mlp_out` `operand_b_pool` | 0.064 / 0.687 / 0.003 | 0.011 / 0.311 / 0.004 | second-operand carrier exists locally and partially transfers |
| `image+text` | L3 `mlp_out` `answer_query` | 0.009 / 0.009 / 0.702 | 0.009 / 0.007 / 0.487 | late sum is the best shared-coordinate positive case |
| `text+image` | L0 `mlp_out` `operand_b_pool` | 0.077 / 0.667 / 0.010 | 0.019 / 0.040 / 0.008 | image-as-second is local but not in mature `b` coordinate |
| `text+image` | L3 `mlp_out` `answer_query` | 0.108 / 0.041 / 0.366 | 0.070 / 0.017 / 0.099 | local sum does not transfer to mature coordinate |
| `number+image` | L0 `attn_out` `plus` | 0.429 / 0.462 / 0.011 | 0.030 / 0.041 / 0.012 | early plus carrier mixes operands but not mature sum |
| `number+image` | L3 `mlp_out` `answer_query` | 0.077 / 0.031 / 0.355 | 0.062 / 0.020 / 0.266 | weaker but real shared-coordinate sum |
| `image+number` | L0 `resid_mid` `operand_b_pool` | 0.087 / 0.775 / 0.004 | 0.014 / 0.277 / 0.009 | number second operand transfers better than image first |
| `image+number` | L3 `mlp_out` `answer_query` | 0.108 / 0.006 / 0.522 | 0.023 / 0.011 / 0.104 | local sum exists but is not mature/readout-compatible |

This changes the mechanistic interpretation of the near-failure routes. They
are not simply blank or sensory-dead. They contain substantial operand
information and can even contain a locally decodable sum. The failure is that
this information is not routed into the shared cyclic coordinate that mature
no-image probes and output heads use.

Representative sum-transfer heatmaps:





### Phase-6 Route Subspace Causal Patching

Priority 3 from the directed-route plan was then run as a causal subspace test
at the localized path-tracing/probe hotspots. The dedicated report is:

```text
RESULTS_TRI_MODAL_ROUTE_SUBSPACE_PATCHING.md
tri_modal_modular_grokking/analysis/phase6_route_subspace_patching/
```

The run produced `5,880` causal patch rows, no failed basis rows, `22` heatmap
PNGs, and a matched comparison to the earlier fully grokked
`phase4_full_crossmodal` baseline. The intervention was same-pair
source-to-target patching:

```text
h_target <- h_target + P(h_source - h_target)
```

where `P` was a rank-16 or rank-32 PCA, probe, random, or orthogonal-complement
projector. The source routes were the mature no-image routes
`number+number`, `number+text`, `text+number`, and `text+text`. The target
routes were the four omitted phase-6 routes. Because source and target examples
used the same `(a, b)` pair and output mode, successful patching reflects
activation-state transport rather than answer replacement.

Headline leaveout rescue:

| route | leaveout target | best full | best PCA | best probe projected | best probe orthogonal | random mean | baseline target | baseline min patch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `image+text` | 0.503906 | 1.000000 `early_b_image_text` | 1.000000 `pca_mature_no_image_r16` `early_b_image_text` | 0.535156 `probe_local_target_s_r32` `late_answer_query` | 1.000000 `probe_mature_no_image_b_r16_orth` `early_b_image_text` | 0.299886 | 1.000000 | 0.906250 |
| `text+image` | 0.123698 | 0.816406 `early_b_text_image` | 0.835938 `pca_local_target_r16` `early_b_text_image` | 0.218750 `probe_local_target_s_r32` `late_answer_query` | 0.816406 `probe_mature_no_image_b_r16_orth` `early_b_text_image` | 0.105713 | 1.000000 | 0.273438 |
| `number+image` | 0.498698 | 0.960938 `late_answer_query` | 0.964844 `pca_mature_no_image_r32` `late_answer_query` | 0.570312 `probe_local_target_s_r32` `late_answer_query` | 0.964844 `probe_mature_no_image_s_r16_orth` `late_answer_query` | 0.533963 | 1.000000 | 1.000000 |
| `image+number` | 0.119792 | 1.000000 `early_b_image_number_mid` | 1.000000 `pca_mature_no_image_r16` `early_b_image_number_mid` | 0.937500 `probe_local_target_s_r32` `early_b_image_number_mid` | 1.000000 `probe_mature_no_image_b_r16_orth` `early_b_image_number_mid` | 0.143962 | 1.000000 | 0.371094 |

The central read is that full-vector and PCA subspace patches recover most of
the causal effect at route-localized carriers, but variable-specific linear
probe subspaces usually do not. Their orthogonal complements often recover the
effect better than the projected probe subspace. So the route failures are not
best described as missing a single low-rank `a`, `b`, or `s` axis. They are
better described as failures to enter a route-compatible activation manifold or
coordinate chart.

The fully grokked baseline is not a rescue test because every pseudo-route
starts at `1.000000`. It is a stability/alignment control. `number+image` is
perfectly stable under every tested subspace patch, `image+text` is highly
stable, and `text+image`/`image+number` can be damaged by some partial
patches. This prevents over-reading random or baseline-preserving patches as
evidence for a clean causal variable subspace.

Representative subspace heatmaps:





### Phase-6 Route Transport Maps

The next high-information follow-up replaced PCA-only subspace patching with
explicit learned low-dimensional transport maps. The dedicated report is:

```text
RESULTS_TRI_MODAL_ROUTE_TRANSPORT_MAPS.md
tri_modal_modular_grokking/analysis/phase6_route_transport_maps/
```

The run produced `4,488` causal rows, `32` route-graph rows, no failed rows,
and `24` heatmap PNGs. For each hotspot, source combo, and output mode, maps
were fit from omitted-route target states to mature no-image source-route
states on pair-disjoint training pairs, then causally evaluated on held-out
patch pairs. Tested map families were `pca_ridge`, `procrustes`,
`identity_coord`, `random_coord`, and `source_full`, with ranks `8`, `16`,
`32`, and `64`.

Headline transport-map rescue:

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

The transport result is more restrictive than the preceding PCA-subspace
patching result. Learned target-to-source maps use only the target route's own
activation state at patch time. Therefore a successful learned map means the
omitted state already contains repairable information; a full/PCA source-state
patch can also inject missing source-state context. The data show that
`image+number` is mostly a coordinate-misalignment case, while `text+image`
is not: it improves over random but remains far below full-state rescue.

Representative transport-map heatmaps:




![leaveout learned transport route graph](tri_modal_modular_grokking/analysis/phase6_route_transport_maps/figures/leaveout_learned_transport_route_graph.png)

### Phase-7 Tiny Route Supervision Rescue

The next high-information follow-up directly tested whether the omitted ordered
routes lack an arithmetic rule or merely lack the route trigger/alignment that
connects them to the existing shared cyclic computation. The dedicated report
is:

```text
RESULTS_TRI_MODAL_ROUTE_SUPERVISION_RESCUE.md
tri_modal_modular_grokking/analysis/phase7_route_supervision_rescue_500step/
```

The data layer now supports `limited_train_input_combos`, a strict
pair-stratified way to add a tiny number of direct train pairs for a specified
ordered input combination without changing the global held-out pair split or
the normal 27-cell dataset behavior. Each nonzero rescue point resumes from the
corresponding phase-6 leaveout `checkpoint_final.pt`, inserts direct examples
for the omitted ordered route only, fine-tunes for `500` steps, and then
evaluates the full strict held-out omitted route across all three output modes.

The 5,000-step version was intentionally stopped after proving too slow for the
full matrix. The completed run is therefore a fast-induction test, not an
asymptotic direct-data threshold sweep.

Full held-out omitted-route accuracy:

| omitted route | 0 | 10 | 25 | 50 | 100 | 250 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `image+text` | 0.471079 | 0.839887 | 0.950205 | 0.962502 | 0.990689 | 1.000000 |
| `text+image` | 0.109256 | 0.375436 | 0.656900 | 0.709529 | 0.932443 | 0.920298 |
| `number+image` | 0.536511 | 0.760336 | 0.887961 | 0.909266 | 0.934771 | 0.938212 |
| `image+number` | 0.108800 | 0.827843 | 0.940236 | 0.944638 | 0.962249 | 0.986640 |

Best point by route:

| omitted route | best direct-pair count | best heldout accuracy |
| --- | ---: | ---: |
| `image+text` | 250 | 1.000000 |
| `text+image` | 100 | 0.932443 |
| `number+image` | 250 | 0.938212 |
| `image+number` | 250 | 0.986640 |

The decisive point is that the near-failure routes are not structurally unable
to use the shared arithmetic rule. `image+number` starts at `0.108800`, but
only `10` direct train pairs lift it to `0.827843`; `25` direct pairs lift it
to `0.940236`. `text+image` is harder, but still rises to `0.932443` with
`100` direct train pairs. The learned output heads are not the limiting factor:
per-head accuracies track the route mean closely, and the output-head spread
stays small.

This result refines the learned transport-map result. Static
target-to-mature maps fully repaired only `image+number`, but tiny direct
supervision strongly rescues every omitted route. Therefore the missing
mechanism is probably not just a fixed low-dimensional coordinate rotation of
the existing target state. Direct examples likely change ordered-route gating,
state stabilization, or broader route context as well as coordinate alignment.

Representative rescue figures:


![route supervision rescue curves](tri_modal_modular_grokking/analysis/phase7_route_supervision_rescue_500step/figures/route_supervision_rescue_curves.png)


### Phase-7 Rescued Checkpoint Mechanistic Analysis

The supervision rescue result raised a sharper mechanistic question: did tiny
direct supervision create the same shared cyclic/Fourier answer representation,
or only a route-local shortcut? The dedicated follow-up report is:

```text
RESULTS_TRI_MODAL_RESCUED_CHECKPOINT_MECH.md
tri_modal_modular_grokking/analysis/phase7_rescued_checkpoint_mech_selected/
```

The selected matrix analyzes `20` checkpoints: each route at `0`, `10`, `25`,
its best behavioral rescue count, and the trained-route `phase4_full_crossmodal`
baseline. For each checkpoint it computes final answer-slot Fourier addition
energy, local omitted-route `s` probes, mature no-image-to-omitted probe
transfer, omitted-to-mature probe transfer, and same-pair final-slot patching
from omitted routes into `number+number` targets.

Headline mechanistic rescue table:

| route | count | behavior | Fourier add | local omitted `s` | mature->omitted `s` | omitted->mature `s` | omitted->mature patch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `image+text` | `0` | 0.471079 | 0.536068 | 0.556601 | 0.463185 | 0.822529 | 0.469401 |
| `image+text` | `10` | 0.839887 | 0.736916 | 0.849805 | 0.829917 | 0.994699 | 0.868490 |
| `image+text` | `25` | 0.950205 | 0.842697 | 0.956277 | 0.947827 | 0.995496 | 0.934245 |
| `image+text` | `250` | 1.000000 | 0.941503 | 1.000000 | 0.998583 | 0.997710 | 1.000000 |
| `text+image` | `0` | 0.109256 | 0.098098 | 0.058448 | 0.091240 | 0.519394 | 0.118490 |
| `text+image` | `10` | 0.375436 | 0.249937 | 0.272304 | 0.313091 | 0.787207 | 0.376953 |
| `text+image` | `25` | 0.656900 | 0.517066 | 0.575426 | 0.584991 | 0.913023 | 0.658854 |
| `text+image` | `100` | 0.932443 | 0.769423 | 0.841152 | 0.833915 | 0.954595 | 0.927734 |
| `number+image` | `0` | 0.536511 | 0.567461 | 0.375588 | 0.345580 | 0.646349 | 0.516927 |
| `number+image` | `10` | 0.760336 | 0.675819 | 0.529882 | 0.505136 | 0.757097 | 0.760417 |
| `number+image` | `25` | 0.887961 | 0.732516 | 0.625171 | 0.587976 | 0.744459 | 0.893229 |
| `number+image` | `250` | 0.938212 | 0.788154 | 0.698244 | 0.673751 | 0.756389 | 0.930990 |
| `image+number` | `0` | 0.108800 | 0.079796 | 0.100501 | 0.103942 | 0.335193 | 0.103516 |
| `image+number` | `10` | 0.827843 | 0.695584 | 0.818531 | 0.802540 | 0.965070 | 0.837240 |
| `image+number` | `25` | 0.940236 | 0.815682 | 0.953646 | 0.927989 | 0.969144 | 0.946615 |
| `image+number` | `250` | 0.986640 | 0.867351 | 0.983958 | 0.975305 | 0.974748 | 0.986328 |

The phase-4 trained-route baselines sit near `0.999` probe transfer and
`0.929-0.942` Fourier energy for all four routes. The direct comparison shows
that `image+number` and `image+text` rapidly approach the phase-4 shared
geometry; `text+image` approaches it more slowly and remains below phase-4 at
its best count; `number+image` becomes behaviorally and causally compatible but
less linearly phase-4-like.

This resolves the first rescue question in favor of real mechanistic route
enrollment, with a caveat. Tiny supervision often moves the route into the
shared cyclic answer coordinate. It is not merely training a decoder shortcut.
However, a high behavioral rescue score does not guarantee the exact same
linear coordinate chart as the fully grokked baseline; `number+image` is the
main example.

Representative rescued-checkpoint figures:

![rescued mature no-image transfer](tri_modal_modular_grokking/analysis/phase7_rescued_checkpoint_mech_selected/figures/mature_to_omitted_s_transfer.png)



### Phase-7 Rescued `number+image` Route-Local Analysis

The selected rescued-checkpoint final-slot analysis left one important
ambiguity: `number+image` had high behavior and strong full answer-slot causal
patching, but only moderate mature no-image linear `s` transfer. The next test
therefore repeated route-local path tracing and subspace causal patching on
`number+image` at `0`, `10`, `25`, and `250` direct pairs, with the
`phase4_full_crossmodal` trained-route checkpoint as the baseline.

Dedicated report and artifacts:

```text
RESULTS_TRI_MODAL_RESCUED_ROUTE_LOCAL_NUMBER_IMAGE.md
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/
```

The run produced `22,680` path-tracing rows and `4,500` route-local subspace
patching rows. Path tracing used `64` patch pairs per spec. Subspace patching
used `256` patch pairs, `927` basis sample pairs, and ranks `16` and `32` at
`early_plus_attn`, `early_plus_resid_mid`, and `late_answer_query`.

Path-tracing summary:

| count | direction | baseline | best patch | early plus | late answer |
| --- | --- | ---: | ---: | ---: | ---: |
| `0` | good-to-omitted | 0.515625 | 0.984375 | 0.781250 | 0.937500 |
| `10` | good-to-omitted | 0.776042 | 1.000000 | 0.859375 | 0.984375 |
| `25` | good-to-omitted | 0.807292 | 0.984375 | 0.906250 | 0.968750 |
| `250` | good-to-omitted | 0.947917 | 0.984375 | 0.968750 | 0.984375 |
| `phase4_full` | good-to-omitted | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

Subspace summary:

| count | patch spec | baseline | best full | best projected | mean projected | random mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `early_plus_attn` | 0.544271 | 0.789062 | 0.789062 | 0.542460 | 0.525553 |
| `0` | `late_answer_query` | 0.544271 | 0.976562 | 0.980469 | 0.650187 | 0.676270 |
| `10` | `early_plus_attn` | 0.757812 | 0.863281 | 0.863281 | 0.739156 | 0.711263 |
| `10` | `late_answer_query` | 0.757812 | 0.980469 | 0.980469 | 0.811605 | 0.837240 |
| `25` | `early_plus_attn` | 0.889323 | 0.933594 | 0.941406 | 0.845805 | 0.829427 |
| `25` | `late_answer_query` | 0.889323 | 0.988281 | 0.984375 | 0.911153 | 0.927409 |
| `250` | `early_plus_attn` | 0.953125 | 0.964844 | 0.968750 | 0.944173 | 0.943685 |
| `250` | `late_answer_query` | 0.953125 | 0.988281 | 0.988281 | 0.958130 | 0.965983 |
| `phase4_full` | all three carriers | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

The mechanistic interpretation is sharper than the final-slot-only result.
Zero-direct `number+image` is already repairable by mature route states, so the
target head and late answer-state readout are compatible with the shared cyclic
state. Direct supervision then quickly stabilizes the route's own carrier:
with `25` direct pairs, early plus and late answer-query patches are already
strong, and with `250` direct pairs the sampled route is nearly saturated.

However, this is not the same signature as a fully clean low-dimensional linear
`s` chart. Mean projected subspace patches and random controls are often close,
and orthogonal complements can retain much of the causal effect. The result is
best described as enrollment into a causally compatible shared cyclic manifold,
not full convergence to the phase-4 linear coordinate chart.

Representative route-local figures:



### Phase-7 Rescued Route-Local All-Route Matrix

The route-local rescued-checkpoint battery has now been extended from
`number+image` to all four image-containing omitted routes:

```text
RESULTS_TRI_MODAL_RESCUED_ROUTE_LOCAL_ALL.md
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_image_text_balanced/
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_text_image_balanced/
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_image_number_balanced/
```

Each route used the same balanced design: `0`, `10`, `25`, best-count, and
phase-4 baseline checkpoints; `64` path patch pairs; `256` subspace patch
pairs; `927` basis pairs; ranks `16` and `32`. The four runs produced `90,720`
path-tracing rows and `14,700` subspace-patching rows.

Good-to-omitted route-local path tracing:

| route | count | clean target | early carrier | late answer | best patch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `image+text` | `0` | 0.416667 | 0.421875 | 0.984375 | 1.000000 |
| `image+text` | `10` | 0.822917 | 0.859375 | 1.000000 | 1.000000 |
| `image+text` | `25` | 0.937500 | 0.953125 | 1.000000 | 1.000000 |
| `image+text` | `250` | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `text+image` | `0` | 0.166667 | 0.187500 | 0.781250 | 1.000000 |
| `text+image` | `10` | 0.421875 | 0.421875 | 0.890625 | 1.000000 |
| `text+image` | `25` | 0.671875 | 0.671875 | 0.906250 | 1.000000 |
| `text+image` | `100` | 0.921875 | 0.953125 | 1.000000 | 1.000000 |
| `number+image` | `0` | 0.515625 | 0.781250 | 0.937500 | 0.984375 |
| `number+image` | `10` | 0.776042 | 0.859375 | 0.984375 | 1.000000 |
| `number+image` | `25` | 0.807292 | 0.906250 | 0.968750 | 0.984375 |
| `number+image` | `250` | 0.947917 | 0.968750 | 0.984375 | 0.984375 |
| `image+number` | `0` | 0.119792 | 0.125000 | 0.968750 | 1.000000 |
| `image+number` | `10` | 0.875000 | 0.906250 | 0.984375 | 1.000000 |
| `image+number` | `25` | 0.984375 | 1.000000 | 1.000000 | 1.000000 |
| `image+number` | `250` | 0.979167 | 0.984375 | 1.000000 | 1.000000 |

Late answer-query subspace patching:

| route | count | baseline | best full | best projected | mean projected | random mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `image+text` | `0` | 0.438802 | 0.996094 | 0.988281 | 0.551331 | 0.534668 |
| `image+text` | `10` | 0.817708 | 1.000000 | 1.000000 | 0.851400 | 0.876465 |
| `image+text` | `25` | 0.947917 | 0.984375 | 0.984375 | 0.955017 | 0.964681 |
| `image+text` | `250` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| `text+image` | `0` | 0.113281 | 0.660156 | 0.632812 | 0.206380 | 0.153483 |
| `text+image` | `10` | 0.338542 | 0.816406 | 0.789062 | 0.424459 | 0.388835 |
| `text+image` | `25` | 0.690104 | 0.937500 | 0.929688 | 0.730713 | 0.722656 |
| `text+image` | `100` | 0.925781 | 0.996094 | 0.996094 | 0.942403 | 0.945150 |
| `number+image` | `0` | 0.544271 | 0.976562 | 0.980469 | 0.650187 | 0.676270 |
| `number+image` | `10` | 0.757812 | 0.980469 | 0.980469 | 0.811605 | 0.837240 |
| `number+image` | `25` | 0.889323 | 0.988281 | 0.984375 | 0.911153 | 0.927409 |
| `number+image` | `250` | 0.953125 | 0.988281 | 0.988281 | 0.958130 | 0.965983 |
| `image+number` | `0` | 0.110677 | 0.968750 | 0.941406 | 0.248250 | 0.172526 |
| `image+number` | `10` | 0.811198 | 0.984375 | 0.980469 | 0.842509 | 0.858724 |
| `image+number` | `25` | 0.947917 | 0.996094 | 0.996094 | 0.958354 | 0.966960 |
| `image+number` | `250` | 0.984375 | 0.996094 | 0.996094 | 0.986430 | 0.986816 |

The all-route route-local matrix makes the route grammar more concrete. Late
answer-query is shared and broadly repairable even before the omitted route
behaves well. The hard part is building the ordered-route carrier that reaches
that shared answer-query machinery. `image+number` is the clearest
missing-trigger case: it starts near chance, but zero-direct mature patches can
already repair early `b` and late answer states, and `10` direct pairs are
enough to stabilize the route. `text+image` is the slow case: it has weak early
carrier evidence at zero direct and improves gradually until the `100`-pair
checkpoint. `number+image` remains the broad-manifold case: causal patches are
strong, but projected and random subspace controls often stay close.

## Phase-9 Real Image-Pixel Decoder Extension

The original phase-4 experiment deliberately used `image_class_proxy` for image
answers. That was the right first control: it kept the answer readout shallow
and made it hard for image-output arithmetic to hide inside a large generator.
But it also left an important limitation. The model had learned to select a
rendered answer class, not to generate the rendered answer image itself.

Phase 9 fixes that limitation in the fully supervised 27-cell setting. It
reuses the same core scientific setup:

```text
modulus: 97
train_fraction: 0.30
train pairs: 2,822
held-out pairs: 6,587
task cells: 27
input modes: number, text, image
output modes: number, text, image
strict held-out pair split across all modalities
```

The only substantive target change is:

```text
decoder_target_kind: image_pixels
```

Instead of treating image output as a class proxy, the model now has a compact
pixel-producing head:

```text
final answer_slot
-> linear projection
-> small transposed-convolution decoder
-> 1 x 64 x 128 grayscale answer patch
```

Training configuration:

| item | value |
| --- | ---: |
| run | `phase9_full_crossmodal_image_pixels` |
| seed | 1704 |
| parameters | 1,503,859 |
| batch size | 256 |
| nominal steps | 20,000 |
| stopped at durable checkpoint | 15,000 |
| latest logged metric step | 15,500 |
| learning rate | 0.001 |
| weight decay | 0.1 |
| image pixel foreground weight | 8.0 |
| image pixel L1 weight | 0.25 |
| render shape | `1 x 64 x 128` |
| render style | digit |

The run was stopped manually after saturation. `checkpoint_final.pt` is a copy
of the durable `checkpoint_15000.pt`; metrics had advanced to step `15500`, but
the step-15500 state was not checkpointed. All mechanistic analysis below uses
the durable step-15000 checkpoint, or an explicit copy of it.

### Phase-9 Training Result

Sampled training telemetry saturated before the manual stop:

| step | held-out accuracy | held-out image-template accuracy | held-out image IoU | held-out pixel MAE |
| ---: | ---: | ---: | ---: | ---: |
| 14,000 | 0.995850 | 0.993407 | 0.958571 | 0.001921 |
| 14,250 | 1.000000 | 1.000000 | 0.971704 | 0.001281 |
| 15,000 | 1.000000 | 1.000000 | 0.974699 | 0.001139 |
| 15,500 | 1.000000 | 1.000000 | 0.974949 | 0.001089 |

The exhaustive checkpoint-15000 evaluation is slightly lower than the sampled
telemetry, as expected:

| split | examples | overall accuracy | image examples | image-template accuracy | image IoU | pixel MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 76,194 | 1.000000 | 25,398 | 1.000000 | 0.977578 | 0.001033 |
| held-out | 177,849 | 0.999899 | 59,283 | 0.999899 | 0.975426 | 0.001124 |

This is the key behavioral result: from scratch, with strict held-out pair
generalization and all 27 input/output cells, the same shared backbone can
drive a real rendered image-producing answer head.

### Phase-9 Image-Head Mechanistic Check

The first mechanistic question was whether the new image head was doing the
work itself, for example by memorizing sums or learning a head-private shortcut.
The architecture already rules out the strongest version of that concern:
`ImagePixelDecoder.forward()` receives only the final `answer_slot`. It has no
direct access to operands, pair ID, modality IDs, output-mode ID, or labels.

The stronger empirical checks also support the shared-answer-slot
interpretation:

| check | value |
| --- | ---: |
| final-slot held-out `s` probe | 0.998949 |
| final-slot permuted-label `s` control | 0.007613 |
| final-slot held-out `a` probe | 0.229043 |
| final-slot held-out `b` probe | 0.169678 |
| non-image-output probe -> image-output held-out `s` | 0.998853 |
| image-output probe -> non-image-output held-out `s` | 0.997352 |
| image-output probe -> image-output held-out `s` | 0.998937 |

The operand probes are not chance, so the final slot is not a mathematically
pure residue-only scalar. But the asymmetry is decisive: the sum residue is
near-perfect and transfers across output types, while individual operands are
much less linearly recoverable on held-out pairs.

The causal patch is the cleanest head-interface test. Clean
`number+number->number` answer slots were patched into corrupt
`image+image->image` examples:

| patch result | value |
| --- | ---: |
| unpatched corrupt image follows corrupt answer | 1.000000 |
| unpatched corrupt image follows clean donor answer | 0.000000 |
| patched image follows clean donor answer | 1.000000 |
| patched image follows original corrupt answer | 0.000000 |
| patched pixel MAE to clean image | 0.002364 |
| patched pixel MAE to corrupt image | 0.045974 |

This is direct causal evidence that the generated image follows the transplanted
answer-slot content. The image decoder is a readout of the shared answer state,
not an independent arithmetic solver.

### Phase-9 Slot Geometry Battery

The second mechanistic question was what kind of answer geometry the image
decoder reads.

#### Raw counterfactual slot arithmetic

The first risky test tried literal vector arithmetic:

```text
slot(r1) + slot(r2) - slot(0)
```

Then the resulting vector was decoded with the image head and compared to
`(r1 + r2) mod 97`. This was run on a copied checkpoint:

```text
tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000/checkpoint_15000_working_copy.pt
```

The result is negative and informative:

| source slot context | original centroid decode | raw arithmetic top-1 | nontrivial top-1 |
| --- | ---: | ---: | ---: |
| `number+number->image` | 1.000000 | 0.030078 | 0.009766 |
| `image+image->image` | 1.000000 | 0.029652 | 0.009332 |
| `number+number->number` | 1.000000 | 0.028590 | 0.008247 |

The nontrivial score excludes identity cases where `r1=0` or `r2=0`. It is
essentially chance. Therefore the answer slot is not a simple Euclidean
additive code.

#### Fourier/phase reconstruction

The negative raw-vector result is compatible with cyclic/Fourier geometry. The
phase-9 battery fit Fourier reconstructions of the learned residue centroids
and decoded the reconstructed target centroid for `(r1+r2) mod 97`.

| harmonics | image decode accuracy | mean circular abs error | within +/-1 |
| ---: | ---: | ---: | ---: |
| 1 | 0.041237 | 9.453609 | 0.113402 |
| 2 | 0.082474 | 4.206185 | 0.237113 |
| 4 | 0.103093 | 4.123711 | 0.226804 |
| 8 | 0.134021 | 3.309278 | 0.309278 |
| 16 | 0.340206 | 1.907217 | 0.752577 |
| 32 | 1.000000 | 0.000000 | 1.000000 |
| 48 | 1.000000 | 0.000000 | 1.000000 |

The high-harmonic result should not be overread as "the model literally
computes by a 32-harmonic external phase decoder"; high harmonics can closely
reconstruct the empirical centroid manifold. The useful conclusion is narrower:
the answer manifold is cyclic and image-readable, but not linear-additive under
raw slot addition.

#### Residue subspace sufficiency and ablation

The strongest phase-9 geometry result is the residue-subspace causal test.
Centroids were used to fit PCA bases of increasing rank. The analysis decoded:

1. only the residue subspace projection;
2. the residual after removing that residue subspace.

| rank | only residue subspace decode | residue subspace removed decode |
| ---: | ---: | ---: |
| 2 | 0.070554 | 0.975838 |
| 4 | 0.207152 | 0.850838 |
| 8 | 0.575709 | 0.517075 |
| 16 | 0.958119 | 0.111791 |
| 32 | 1.000000 | 0.003222 |
| 64 | 1.000000 | 0.008376 |

This is the cleanest new mechanistic evidence. A rank-32 residue manifold is
sufficient for perfect image decoding, and removing it drops image decoding to
near chance. The image-pixel decoder is listening to the residue manifold.

#### Interpolation and manifold repair

Adjacent-residue interpolation shows broad local decoder basins:

| interpolation metric | value |
| --- | ---: |
| nearest endpoint match | 0.961575 |
| from-or-to endpoint match | 0.997188 |
| midpoint from-or-to endpoint match | 0.969072 |

Nearest-manifold repair of failed raw arithmetic does not rescue the arithmetic
composition:

| repair metric | value |
| --- | ---: |
| raw decode accuracy | 0.030078 |
| nearest centroid accuracy | 0.023913 |
| PCA projected nearest accuracy | 0.025826 |
| PCA repaired decode accuracy | 0.025826 |

So the raw arithmetic vectors are not merely slightly off the manifold in a way
that nearest-centroid projection fixes. The raw operation itself is the wrong
coordinate operation for this answer geometry.

#### Phase-9 checkpoint dynamics

The image-readable residue manifold emerges earlier than the final stopped
checkpoint. Using 8 examples per residue for dynamics:

| step | original centroid decode | raw nontrivial arithmetic | H16 phase arithmetic |
| ---: | ---: | ---: | ---: |
| 1,000 | 0.010309 | 0.010308 | 0.010309 |
| 2,000 | 0.061856 | 0.014106 | 0.061856 |
| 3,000 | 0.113402 | 0.012370 | 0.113402 |
| 4,000 | 0.103093 | 0.010525 | 0.113402 |
| 5,000 | 0.597938 | 0.009006 | 0.195876 |
| 6,000 | 0.917526 | 0.010308 | 0.164948 |
| 7,000 | 1.000000 | 0.011068 | 0.412371 |
| 8,000 | 1.000000 | 0.011285 | 0.371134 |
| 9,000 | 1.000000 | 0.008681 | 0.329897 |
| 10,000 | 1.000000 | 0.009766 | 0.360825 |
| 11,000 | 1.000000 | 0.011068 | 0.443299 |
| 12,000 | 1.000000 | 0.006727 | 0.360825 |
| 13,000 | 1.000000 | 0.007812 | 0.432990 |
| 14,000 | 1.000000 | 0.009332 | 0.381443 |
| 15,000 | 1.000000 | 0.009332 | 0.350515 |

Image-readable centroids become perfect by step `7000`, well before the final
manual stop. Raw vector arithmetic remains near chance throughout. This
separates "the decoder can read the learned residue slots" from "the slot
space supports naive Euclidean addition."

### Phase-9 Interpretation

Phase 9 upgrades the classical trimodal result from "one backbone can select an
image answer class" to "one backbone can generate rendered image answers from a
shared answer state." It also resolves a key mechanistic concern: the image
pixel decoder is not memorizing sums or computing arithmetic in a hidden
head-private route. It reads the same late answer-slot residue manifold that
drives the number and text heads.

The refined mechanistic claim is:

```text
number/text/image operands
-> shared late cyclic residue manifold at answer_query
-> modality-specific shallow readouts, including a real image-pixel decoder
```

The caveat is equally important. The answer state should not be described as a
simple additive vector embedding. Raw slot addition fails. The evidence favors
a curved/cyclic manifold with local image-decoder basins and a causally
necessary residue subspace.

## Phase-10 Native Trimodal Compression

Phase 10 tests whether the phase-9 capability depends on a million-parameter
model or survives a large end-to-end reduction. It preserves native
multimodality rather than returning to the separate modality-blind raw-stream
question. Number IDs, text tokens, and raster images still enter through small
learned interfaces; all routes then use one shared Transformer, and the three
conditioned output types include actual generated pixels.

The compressed configuration is:

| component | value |
| --- | ---: |
| total parameters | 81,927 |
| compression from phase 9 | 18.356x |
| `d_model` | 32 |
| Transformer layers | 2 |
| attention heads | 4 |
| feed-forward width | 64 |
| image tokens per operand | 4 |
| image decoder channels | 8 |

The shared backbone including all input encoders has `35,104` parameters, the
real pixel decoder has `35,405`, and all output heads together have `11,418`.
Thus the budget is already balanced between representation/computation and
image generation; the large phase-9 projection is no longer overwhelmingly
dominant.

Training used the unchanged phase-9 27-cell dataset, batch size `256`, AdamW
learning rate `0.001`, weight decay `0.1`, and seed `1704`. A user-requested
suspension preserved step `15,000` and its optimizer, after which training
resumed and replayed post-checkpoint steps. The run completed `40,000` steps.
It did not meet the sustained near-perfect early-stop criteria.

Sampled performance rose gradually from `0.050781` held-out at step `5,000` to
`0.894287` at step `30,000`. It peaked at `0.928955` at unsaved step `34,250`,
then oscillated and ended at `0.867554`. Because checkpoints were written every
`1,000` steps, step `33,000` was selected as the best durable checkpoint by the
pre-existing sampled evaluation.

Both step `33,000` and final step `40,000` were then evaluated exhaustively:

| checkpoint | split | overall | number | text | image | image IoU | pixel MAE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 33,000 | train | 0.943946 | 0.987204 | 0.942161 | 0.902473 | 0.772451 | 0.012428 |
| 33,000 | held-out | 0.918729 | 0.962030 | 0.905082 | 0.889074 | 0.773364 | 0.012606 |
| 40,000 | train | 0.923774 | 0.976691 | 0.923419 | 0.871210 | 0.772355 | 0.012311 |
| 40,000 | held-out | 0.894607 | 0.947675 | 0.882850 | 0.853297 | 0.773294 | 0.012465 |

The selected checkpoint's exhaustive held-out input routes are:

| input route | exact accuracy across outputs |
| --- | ---: |
| number + number | 0.935277 |
| number + text | 0.929153 |
| number + image | 0.911847 |
| text + number | 0.925358 |
| text + text | 0.927281 |
| text + image | 0.904053 |
| image + number | 0.922069 |
| image + text | 0.908709 |
| image + image | 0.904813 |

Every route exceeds `0.90`; the compressed model is behaviorally native
trimodal. The weakest cells are image outputs, led by
`text+image->image=0.869288` and `image+image->image=0.871717`. This localizes
the main compression cost to output rendering rather than visual operand
recognition.

Compared with phase 9, the selected model reduces parameters by `94.55%` while
held-out exact accuracy declines from `0.999899` to `0.918729`. Image-template
accuracy declines from `0.999899` to `0.889074`, and image IoU from `0.975426`
to `0.773364`. The result is therefore a strong first compression-frontier
point, not a cost-free replacement for phase 9 and not evidence for the full
three-order-of-magnitude target.

The complete audit is recorded in:

```text
tri_modal_modular_grokking/analysis/phase10_native_trimodal_82k/COMPRESSION_REPORT.md
tri_modal_modular_grokking/analysis/phase10_native_trimodal_82k/summary.csv
tri_modal_modular_grokking/analysis/phase10_native_trimodal_82k/checkpoint_33000_exhaustive.json
tri_modal_modular_grokking/analysis/phase10_native_trimodal_82k/checkpoint_40000_exhaustive.json
```

## Limitations

The result is strong but not final.

1. The phase-4 result is a fully supervised 27-cell capability and
   shared-representation baseline. The phase-5 leave-combination-out run
   localized an image-input translation/routing failure, but it did not learn
   image-input arithmetic even for trained `image+image`. The phase-6
   single-combination controls now cover `image+text`, `text+image`,
   `number+image`, and `image+number`; they show two partial routes and two
   near-failure routes. These controls have only been run for one seed, and
   they still do not cover single-combination omissions of the no-image mixed
   routes `number+text` and `text+number`.
2. The phase-4 image output was a class proxy, not a rendered image decoder.
   Phase 9 resolves this for the fully supervised 27-cell setting by training a
   compact real image-pixel decoder from scratch. However, the pixel-decoder
   result has not yet been repeated across seeds, has not yet been run under
   leave-combination-out or tiny-supervision route-rescue settings, and uses
   deterministic digit renderings rather than naturalistic images.
3. Three phase-4 seeds have now been trained and evaluated at their final
   checkpoints. Seeds `1704` and `1705` fully grokked; seed `1706` remained
   weaker at final evaluation and below the strong-seed strict linear-probe
   regime. Seed robustness is therefore mixed.
4. Full-state final answer-slot patching was run for one source-target cell
   pair in the phase-4 baseline. Layer-2 answer-query subspace patching was
   added for the operand-value robustness pass and is strongly positive for
   seeds `1704` and `1705`. Directed semantic-site path tracing is now complete
   for the four phase-6 image-containing route omissions, and all-layer
   semantic-site operand probes are now complete for those same routes.
   Low-rank subspace causal patching and learned low-dimensional transport-map
   patching are also complete at the localized carriers. Tiny omitted-route
   supervision rescue curves are also complete for the four image-containing
   phase-6 route omissions, but only with a `500`-step fast-rescue budget. A
   selected rescued-checkpoint mechanistic pass is complete for `0`, `10`,
   `25`, best-count, and phase-4 baseline checkpoints. Route-local
   path-tracing and subspace-patching follow-ups are now complete for all four
   image-containing rescued routes at `0`, `10`, `25`, best-count, and phase-4
   baseline checkpoints. Exhaustive analysis of every rescue count, every
   layer, every route, and every route-local semantic site remains open.
5. Fourier diagnostics are representation diagnostics. They do not by
   themselves prove causal sufficiency of a Fourier subspace.
6. The initial sampled telemetry slightly overstated final exactness. The
   exhaustive final evaluation is `0.999871` held-out accuracy, not exactly
   `1.000000`.
7. Translation into a shared numeric/value-coordinate operand representation is
   supported for strong seeds `1704` and `1705`, including causal layer-2
   answer-query subspace patching. It is not seed-universal at the final
   checkpoint, because seed `1706` lacks the same direct layer-2 coordinate
   chart and causal carrier.
8. The 82k compression result has one seed. Its late trajectory is strongly
   oscillatory, its selected checkpoint does not reach perfect train fit, and
   it loses substantial image fidelity relative to phase 9. It establishes a
   capability point, not a seed-robust compression boundary or grokking result.

## Recommended Next Experiments

The directed-route follow-up plan is stored in:

```text
TRI_MODAL_DIRECTED_ROUTE_FOLLOWUP_PLAN.md
```

Priorities 1 through 5 from that plan are now complete at the final checkpoint:
directed semantic-site path tracing, all-layer operand probes, subspace causal
patching, and learned transport-map route graphs over the localized carriers
with a fully grokked baseline comparison, followed by 500-step tiny
supervision rescue curves. The first rescued-checkpoint final-slot mechanistic
pass is also complete. The rescued route-local path-tracing and
subspace-patching matrix is now complete for all four image-containing routes
at `0`, `10`, `25`, best-count, and phase-4 baseline checkpoints. The next
experiments should stress-test and refine the all-route result:

1. Increase patch-pair counts for `number+image` and `text+image` late-answer
   subspace controls to separate true broad-manifold causality from
   high-baseline random-control effects.
2. Add route-context ablations that patch answer-like components while holding
   route tokens/sites fixed, then patch route context while holding inferred
   answer components fixed.
3. Continue selected rescue checkpoints beyond `500` fine-tuning steps:
   `text+image` at `25` and `100`, and `number+image` at `25` and `250`.
4. Extend the rescued-checkpoint mechanistic pass from selected counts to all
   rescue counts and all layers, not only final answer slots.
5. Build a directed route graph over checkpoints using probe transfer, CKA,
   ridge alignment, full-state patching, PCA-subspace patching, and learned
   transport-map matrices, to test whether the directed route grammar emerges
   gradually or snaps in at route grokking.
6. Rerun the rescue curves with longer fine-tuning for the nonmonotonic or
   hardest points, especially `text+image` at `100` and `250` direct pairs.
7. Continue the remaining failed or partial phase-6 runs beyond `20,000` steps
   to distinguish delayed route grokking from stable route absence. The first
   completed continuation, `number+image` to `40,000`, showed transient high
   accuracy, collapse, recovery, and late degradation rather than a stable
   monotone transition.
8. Rerun the four phase-6 route omissions on additional seeds after the
   analysis-only steps above, so seed variance is interpreted through the
   localized carrier hypotheses rather than only behavior.
9. Run seed-`1706` checkpoint-specific operand-value robustness on saved
   checkpoints `15,000` and `16,000`. The strict linear probe audit shows that
   the clean strong-seed answer geometry did not appear at saved late
   checkpoints, but the layer-2 operand-coordinate and causal-patching battery
   has not yet been rerun there.
10. Repeat the phase-9 real image-pixel decoder run across seeds, especially a
   clean seed (`1705`) and a strict-chart-weak seed (`1706` or `1708`), to test
   whether image generation inherits the same seed taxonomy as the class-proxy
   result.
11. Run the real image-pixel decoder under a leave-route or tiny-supervision
   rescue setting to test whether generated image output also becomes
   route-inducible with few direct examples.
12. Extend the phase-9 slot-geometry battery to Fourier-subspace causal
   patching, not only PCA residue-subspace projection and ablation.
13. Extend the seed repeat to more seeds and later to `p in {31, 127}`.
14. Repeat the 81,927-parameter phase-10 model for at least two more seeds and
    use validation-selected durable checkpoints to estimate seed success and
    checkpoint-instability rates.
15. After the 82k replication, continue the native compression ladder at
    `37,207`, `18,575`, and `10,755` parameters, retaining exact all-cell
    evaluation and honest image-pixel metrics at every point.

## Reproduction Commands

Focused tests:

```powershell
cd modular_addition_mech_interp
python -m pytest tests/test_tri_modal_data.py tests/test_tri_modal_model.py tests/test_tri_modal_fourier.py
```

Training:

```powershell
cd modular_addition_mech_interp
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase4_full_crossmodal.yaml
```

Phase-9 real image-pixel decoder training:

```powershell
cd modular_addition_mech_interp
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase9_full_crossmodal_image_pixels.yaml
```

Phase-10 81,927-parameter native trimodal training and exhaustive audit:

```powershell
cd modular_addition_mech_interp
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase10_native_trimodal_82k_seed1704.yaml
python -m tri_modal_modular_grokking.evaluate_compression_checkpoints --checkpoint tri_modal_modular_grokking/runs/phase10_native_trimodal_82k_seed1704/checkpoint_33000.pt --checkpoint tri_modal_modular_grokking/runs/phase10_native_trimodal_82k_seed1704/checkpoint_final.pt --output-dir tri_modal_modular_grokking/analysis/phase10_native_trimodal_82k --batch-size 256
```

The reported phase-9 model uses the durable step-15000 checkpoint:

```text
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/checkpoint_15000.pt
```

Resume from the durable step-10,000 checkpoint:

```powershell
cd modular_addition_mech_interp
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase4_full_crossmodal.yaml --resume-checkpoint tri_modal_modular_grokking/runs/phase4_full_crossmodal/checkpoint_10000.pt
```

Post-training analysis:

```powershell
cd modular_addition_mech_interp
python -m tri_modal_modular_grokking.analyze --run-dir tri_modal_modular_grokking/runs/phase4_full_crossmodal --out-dir tri_modal_modular_grokking/analysis/phase4_full_crossmodal
python -m tri_modal_modular_grokking.rigorous_probes --run-dir tri_modal_modular_grokking/runs/phase4_full_crossmodal --out-dir tri_modal_modular_grokking/analysis/phase4_rigorous_probes --sample-pairs 927 --seeds 0,1,2,3,4 --lambdas 0.0001,0.001,0.01,0.1,1,10,100 --transition-steps 10000,11000,12000,13000,20000 --batch-size 4096
python -m tri_modal_modular_grokking.rigorous_probes --run-dir tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1705 --out-dir tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1705 --sample-pairs 927 --seeds 0,1,2,3,4 --lambdas 0.0001,0.001,0.01,0.1,1,10,100 --transition-steps 10000,11000,12000,13000,20000 --batch-size 4096
python -m tri_modal_modular_grokking.rigorous_probes --run-dir tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1706 --out-dir tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1706_expanded --sample-pairs 927 --seeds 0,1,2,3,4 --lambdas 0.0001,0.001,0.01,0.1,1,10,100 --transition-steps 10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000 --batch-size 4096
python -m tri_modal_modular_grokking.fourier --answer-slots tri_modal_modular_grokking/analysis/phase4_full_crossmodal/answer_slots.pt --out-dir tri_modal_modular_grokking/analysis/phase4_full_crossmodal/fourier
python -m tri_modal_modular_grokking.patching --run-dir tri_modal_modular_grokking/runs/phase4_full_crossmodal --out-dir tri_modal_modular_grokking/analysis/phase4_full_crossmodal/patching --pairs 128 --layers -1
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase4_full_crossmodal_seed1705.yaml
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase4_full_crossmodal_seed1706.yaml
python -m tri_modal_modular_grokking.operand_value_robustness --run-dirs tri_modal_modular_grokking/runs/phase4_full_crossmodal,tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1705,tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1706 --out-dir tri_modal_modular_grokking/analysis/phase4_operand_value_robustness_multiseed --layers 2 --batch-size 8192 --split-seeds 0,1,2,3,4 --lambdas 0.0001,0.001,0.01,0.1,1,10,100 --ranks 2,4,8,16,32,64,128 --patch-pairs 512
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase5_leave_combo_out.yaml
python -m tri_modal_modular_grokking.leave_combo_mech_interp --run-dir tri_modal_modular_grokking/runs/phase5_leave_combo_out --out-dir tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final --checkpoint-steps 20000 --probe-layers -1 --extra-final-layers none --seeds 0,1,2 --lambdas 0.01,0.1,1,10 --patch-pairs 512 --patch-layers 0,1,2,3,-1 --cka-sample 2048 --batch-size 8192
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase6_leave_image_text.yaml
python -m tri_modal_modular_grokking.leave_combo_mech_interp --run-dir tri_modal_modular_grokking/runs/phase6_leave_image_text --out-dir tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp --checkpoint-steps 10000,12000,13000,15000,17000,20000 --probe-layers -1 --extra-final-layers none --seeds 0,1,2 --lambdas 0.01,0.1,1,10 --patch-pairs 512 --patch-layers 0,1,2,3,-1 --cka-sample 2048 --batch-size 8192
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase6_leave_text_image.yaml
python -m tri_modal_modular_grokking.leave_combo_mech_interp --run-dir tri_modal_modular_grokking/runs/phase6_leave_text_image --out-dir tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp --checkpoint-steps 10000,12000,13000,15000,17000,20000 --probe-layers -1 --extra-final-layers none --seeds 0,1,2 --lambdas 0.01,0.1,1,10 --patch-pairs 512 --patch-layers 0,1,2,3,-1 --cka-sample 2048 --batch-size 8192
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase6_leave_number_image.yaml
python -m tri_modal_modular_grokking.leave_combo_mech_interp --run-dir tri_modal_modular_grokking/runs/phase6_leave_number_image --out-dir tri_modal_modular_grokking/analysis/phase6_leave_number_image_mech_interp --checkpoint-steps 10000,12000,13000,15000,17000,20000 --probe-layers -1 --extra-final-layers none --seeds 0,1,2 --lambdas 0.01,0.1,1,10 --patch-pairs 512 --patch-layers 0,1,2,3,-1 --cka-sample 2048 --batch-size 8192
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase6_leave_number_image_to40000.yaml --resume-checkpoint tri_modal_modular_grokking/runs/phase6_leave_number_image/checkpoint_20000.pt
python -m tri_modal_modular_grokking.leaveout_continuation_summary --run-dir tri_modal_modular_grokking/runs/phase6_leave_number_image_to40000 --out-dir tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension --omitted-combo number+image --exact-final-json tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/final_exact_eval.json
python -m tri_modal_modular_grokking.make_leaveout_training_curve_figures
python -m tri_modal_modular_grokking.train --config tri_modal_modular_grokking/configs/phase6_leave_image_number.yaml
python -m tri_modal_modular_grokking.leave_combo_mech_interp --run-dir tri_modal_modular_grokking/runs/phase6_leave_image_number --out-dir tri_modal_modular_grokking/analysis/phase6_leave_image_number_mech_interp --checkpoint-steps 10000,12000,13000,15000,17000,20000 --probe-layers -1 --extra-final-layers none --seeds 0,1,2 --lambdas 0.01,0.1,1,10 --patch-pairs 512 --patch-layers 0,1,2,3,-1 --cka-sample 2048 --batch-size 8192
python -m tri_modal_modular_grokking.run_directed_route_path_tracing_pipeline --patch-pairs 256 --device auto
python -m tri_modal_modular_grokking.route_operand_probes --out-dir tri_modal_modular_grokking/analysis/phase6_route_operand_probes --jobs all --sample-pairs 927 --seeds 0,1,2 --lambdas 0.0001,0.001,0.01,0.1,1,10,100 --sites all --batch-size 512 --device auto
python -m tri_modal_modular_grokking.route_subspace_patching --out-dir tri_modal_modular_grokking/analysis/phase6_route_subspace_patching --jobs all --baseline-run-name phase4_full_crossmodal --source-combos good --good-combos good --patch-specs hotspots --patch-pairs 256 --patch-split heldout_pair --basis-sample-pairs 927 --ranks 16,32 --targets a,b,s --batch-size 512 --device auto
python -m tri_modal_modular_grokking.route_transport_maps --out-dir tri_modal_modular_grokking/analysis/phase6_route_transport_maps --jobs all --baseline-run-name phase4_full_crossmodal --source-combos good --good-combos good --patch-specs hotspots --patch-pairs 256 --patch-split heldout_pair --sample-pairs 927 --ranks 8,16,32,64 --map-kinds all --batch-size 512 --device auto
python -m tri_modal_modular_grokking.rescued_route_local_analysis --out-dir tri_modal_modular_grokking\analysis\phase7_rescued_route_local_image_text_balanced --route image+text --counts 0,10,25,best --include-phase4-baseline --path-patch-pairs 64 --subspace-patch-pairs 256 --basis-sample-pairs 927 --ranks 16,32 --batch-size 512 --device auto
python -m tri_modal_modular_grokking.rescued_route_local_analysis --out-dir tri_modal_modular_grokking\analysis\phase7_rescued_route_local_text_image_balanced --route text+image --counts 0,10,25,best --include-phase4-baseline --path-patch-pairs 64 --subspace-patch-pairs 256 --basis-sample-pairs 927 --ranks 16,32 --batch-size 512 --device auto
python -m tri_modal_modular_grokking.rescued_route_local_analysis --out-dir tri_modal_modular_grokking\analysis\phase7_rescued_route_local_number_image_balanced --route number+image --counts 0,10,25,best --include-phase4-baseline --path-patch-pairs 64 --subspace-patch-pairs 256 --basis-sample-pairs 927 --ranks 16,32 --batch-size 512 --device auto
python -m tri_modal_modular_grokking.rescued_route_local_analysis --out-dir tri_modal_modular_grokking\analysis\phase7_rescued_route_local_image_number_balanced --route image+number --counts 0,10,25,best --include-phase4-baseline --path-patch-pairs 64 --subspace-patch-pairs 256 --basis-sample-pairs 927 --ranks 16,32 --batch-size 512 --device auto
python -m tri_modal_modular_grokking.image_head_mech --checkpoint tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/checkpoint_15000.pt --out-dir tri_modal_modular_grokking/analysis/phase9_image_head_mech_ckpt15000 --device auto --batch-size 2048 --patch-pairs 512
New-Item -ItemType Directory -Force tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000
Copy-Item tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/checkpoint_15000.pt tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000/checkpoint_15000_working_copy.pt
python -m tri_modal_modular_grokking.slot_arithmetic --checkpoint tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000/checkpoint_15000_working_copy.pt --out-dir tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000 --device auto --source-cell number+number->image --examples-per-residue 32 --batch-size 2048 --max-saved-images 64
python -m tri_modal_modular_grokking.slot_geometry_experiments --checkpoint tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000/checkpoint_15000_working_copy.pt --run-dir tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels --out-dir tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000 --device auto --source-cell number+number->image --examples-per-residue 32 --dynamics-examples-per-residue 8 --batch-size 2048 --max-saved-images 64
```

## Audit Artifacts

Training:

```text
tri_modal_modular_grokking/runs/phase4_full_crossmodal/config.yaml
tri_modal_modular_grokking/runs/phase4_full_crossmodal/metadata.json
tri_modal_modular_grokking/runs/phase4_full_crossmodal/metrics.jsonl
tri_modal_modular_grokking/runs/phase4_full_crossmodal/per_cell_accuracy.csv
tri_modal_modular_grokking/runs/phase4_full_crossmodal/checkpoint_*.pt
tri_modal_modular_grokking/runs/phase4_full_crossmodal/checkpoint_final.pt
tri_modal_modular_grokking/runs/phase4_full_crossmodal/final_summary.json
tri_modal_modular_grokking/configs/phase4_full_crossmodal_seed1705.yaml
tri_modal_modular_grokking/configs/phase4_full_crossmodal_seed1706.yaml
tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1705/checkpoint_final.pt
tri_modal_modular_grokking/runs/phase4_full_crossmodal_seed1706/checkpoint_final.pt
tri_modal_modular_grokking/configs/phase9_full_crossmodal_image_pixels.yaml
tri_modal_modular_grokking/configs/phase9_attach_image_pixels.yaml
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/config.yaml
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/metadata.json
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/metrics.jsonl
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/per_cell_accuracy.csv
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/checkpoint_*.pt
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/checkpoint_15000.pt
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/checkpoint_final.pt
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/manual_stop_summary.json
tri_modal_modular_grokking/runs/phase9_full_crossmodal_image_pixels/image_previews/
```

Analysis:

```text
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/analysis_summary.json
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/answer_slots.pt
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/probe_transfer_matrix.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/probe_manifest.json
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/probe_summary.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/global_probe_results.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes/cross_cell_transfer.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1705/probe_summary.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1705/global_probe_results.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1705/cross_cell_transfer.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1706_expanded/probe_summary.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1706_expanded/global_probe_results.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_seed1706_expanded/cross_cell_transfer.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_multiseed/LINEAR_PROBE_SEED_COMPARISON.md
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_multiseed/seed_probe_key_summary.csv
tri_modal_modular_grokking/analysis/phase4_rigorous_probes_multiseed/seed1706_focus_table.csv
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/full_final_eval.json
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/full_final_per_cell_accuracy.csv
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/full_final_errors.csv
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/fourier/fourier_summary.json
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/fourier/fourier_energy_by_layer.csv
tri_modal_modular_grokking/analysis/phase4_full_crossmodal/patching/patching_matrix.csv
tri_modal_modular_grokking/analysis/phase4_operand_value_robustness_multiseed/manifest.json
tri_modal_modular_grokking/analysis/phase4_operand_value_robustness_multiseed/all_pair_alignment.csv
tri_modal_modular_grokking/analysis/phase4_operand_value_robustness_multiseed/context_transfer.csv
tri_modal_modular_grokking/analysis/phase4_operand_value_robustness_multiseed/rank_sweep.csv
tri_modal_modular_grokking/analysis/phase4_operand_value_robustness_multiseed/subspace_patching.csv
tri_modal_modular_grokking/analysis/phase4_operand_value_robustness_multiseed/OPERAND_VALUE_ROBUSTNESS_REPORT.md
tri_modal_modular_grokking/analysis/phase9_image_head_mech_ckpt15000/IMAGE_HEAD_MECH_REPORT.md
tri_modal_modular_grokking/analysis/phase9_image_head_mech_ckpt15000/summary.json
tri_modal_modular_grokking/analysis/phase9_image_head_mech_ckpt15000/probe_results.csv
tri_modal_modular_grokking/analysis/phase9_image_head_mech_ckpt15000/transfer_results.csv
tri_modal_modular_grokking/analysis/phase9_image_head_mech_ckpt15000/patch_result.json
tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000/SLOT_ARITHMETIC_REPORT.md
tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000/summary.json
tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000/slot_arithmetic_results.csv
tri_modal_modular_grokking/analysis/phase9_slot_arithmetic_ckpt15000/checkpoint_15000_working_copy.pt
tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000/SLOT_GEOMETRY_REPORT.md
tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000/summary.json
tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000/phase_arithmetic.csv
tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000/subspace_ablation.csv
tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000/interpolation.csv
tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000/repair_results.csv
tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000/checkpoint_dynamics.csv
tri_modal_modular_grokking/analysis/phase9_slot_geometry_battery_ckpt15000/raw_arithmetic_failure_images/
tri_modal_modular_grokking/runs/phase5_leave_combo_out/config.yaml
tri_modal_modular_grokking/runs/phase5_leave_combo_out/metrics.jsonl
tri_modal_modular_grokking/runs/phase5_leave_combo_out/per_cell_accuracy.csv
tri_modal_modular_grokking/runs/phase5_leave_combo_out/checkpoint_*.pt
tri_modal_modular_grokking/runs/phase5_leave_combo_out/checkpoint_final.pt
tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/LEAVE_COMBO_MECH_INTERP_REPORT.md
tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/behavior_by_cell_pair_split.csv
tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/cell_local_probe_results.csv
tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/cross_cell_transfer_results.csv
tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/fourier_by_cell.csv
tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/alignment_results.csv
tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/same_pair_patching_results.csv
tri_modal_modular_grokking/analysis/phase5_leave_combo_mech_interp_final/same_pair_patching_reverse_image_source.csv
tri_modal_modular_grokking/configs/phase6_leave_image_text.yaml
tri_modal_modular_grokking/runs/phase6_leave_image_text/config.yaml
tri_modal_modular_grokking/runs/phase6_leave_image_text/metrics.jsonl
tri_modal_modular_grokking/runs/phase6_leave_image_text/per_cell_accuracy.csv
tri_modal_modular_grokking/runs/phase6_leave_image_text/checkpoint_*.pt
tri_modal_modular_grokking/runs/phase6_leave_image_text/checkpoint_final.pt
tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp/LEAVE_COMBO_MECH_INTERP_REPORT.md
tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp/behavior_by_cell_pair_split.csv
tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp/cell_local_probe_results.csv
tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp/cross_cell_transfer_results.csv
tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp/fourier_by_cell.csv
tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp/alignment_results.csv
tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp/same_pair_patching_results.csv
tri_modal_modular_grokking/analysis/phase6_leave_image_text_mech_interp/same_pair_patching_omitted_source.csv
tri_modal_modular_grokking/configs/phase6_leave_text_image.yaml
tri_modal_modular_grokking/runs/phase6_leave_text_image/config.yaml
tri_modal_modular_grokking/runs/phase6_leave_text_image/metrics.jsonl
tri_modal_modular_grokking/runs/phase6_leave_text_image/per_cell_accuracy.csv
tri_modal_modular_grokking/runs/phase6_leave_text_image/checkpoint_*.pt
tri_modal_modular_grokking/runs/phase6_leave_text_image/checkpoint_final.pt
tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp/LEAVE_COMBO_MECH_INTERP_REPORT.md
tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp/behavior_by_cell_pair_split.csv
tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp/cell_local_probe_results.csv
tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp/cross_cell_transfer_results.csv
tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp/fourier_by_cell.csv
tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp/alignment_results.csv
tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp/same_pair_patching_results.csv
tri_modal_modular_grokking/analysis/phase6_leave_text_image_mech_interp/same_pair_patching_omitted_source.csv
tri_modal_modular_grokking/configs/phase6_leave_number_image.yaml
tri_modal_modular_grokking/runs/phase6_leave_number_image/config.yaml
tri_modal_modular_grokking/runs/phase6_leave_number_image/metrics.jsonl
tri_modal_modular_grokking/runs/phase6_leave_number_image/per_cell_accuracy.csv
tri_modal_modular_grokking/runs/phase6_leave_number_image/checkpoint_*.pt
tri_modal_modular_grokking/runs/phase6_leave_number_image/checkpoint_final.pt
tri_modal_modular_grokking/analysis/phase6_leave_number_image_mech_interp/LEAVE_COMBO_MECH_INTERP_REPORT.md
tri_modal_modular_grokking/configs/phase6_leave_number_image_to40000.yaml
tri_modal_modular_grokking/runs/phase6_leave_number_image_to40000/config.yaml
tri_modal_modular_grokking/runs/phase6_leave_number_image_to40000/metrics.jsonl
tri_modal_modular_grokking/runs/phase6_leave_number_image_to40000/per_cell_accuracy.csv
tri_modal_modular_grokking/runs/phase6_leave_number_image_to40000/checkpoint_*.pt
tri_modal_modular_grokking/runs/phase6_leave_number_image_to40000/checkpoint_final.pt
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/LEAVEOUT_LONG_CONTINUATION_REPORT.md
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/continuation_summary.json
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/final_exact_eval.json
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/key_step_train_loss_summary.csv
tri_modal_modular_grokking/analysis/phase6_leave_number_image_40k_extension/figures/
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/all_seed_training_curve_summary.csv
tri_modal_modular_grokking/analysis/phase8_seed_program_queue/figures/TRAIN_LOSS_CURVE_MANIFEST.md
tri_modal_modular_grokking/make_leaveout_training_curve_figures.py
tri_modal_modular_grokking/configs/phase6_leave_image_number.yaml
tri_modal_modular_grokking/runs/phase6_leave_image_number/config.yaml
tri_modal_modular_grokking/runs/phase6_leave_image_number/metrics.jsonl
tri_modal_modular_grokking/runs/phase6_leave_image_number/per_cell_accuracy.csv
tri_modal_modular_grokking/runs/phase6_leave_image_number/checkpoint_*.pt
tri_modal_modular_grokking/runs/phase6_leave_image_number/checkpoint_final.pt
tri_modal_modular_grokking/analysis/phase6_leave_image_number_mech_interp/LEAVE_COMBO_MECH_INTERP_REPORT.md
RESULTS_TRI_MODAL_DIRECTED_ROUTE_PATH_TRACING.md
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/pipeline.log
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/PATH_TRACING_REPORT.md
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/path_tracing_rows.csv
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/summary.json
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/figures/good_to_omitted_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_text/figures/omitted_to_good_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/PATH_TRACING_REPORT.md
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/path_tracing_rows.csv
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/summary.json
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/figures/good_to_omitted_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_text_image/figures/omitted_to_good_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/PATH_TRACING_REPORT.md
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/path_tracing_rows.csv
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/summary.json
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/figures/good_to_omitted_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_number_image/figures/omitted_to_good_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/PATH_TRACING_REPORT.md
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/path_tracing_rows.csv
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/summary.json
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/figures/good_to_omitted_heatmap.png
tri_modal_modular_grokking/analysis/phase6_directed_route_path_tracing/phase6_leave_image_number/figures/omitted_to_good_heatmap.png
RESULTS_TRI_MODAL_ROUTE_OPERAND_PROBES.md
tri_modal_modular_grokking/analysis/phase6_route_operand_probes/route_operand_probe_rows.csv
tri_modal_modular_grokking/analysis/phase6_route_operand_probes/summary.json
tri_modal_modular_grokking/analysis/phase6_route_operand_probes/ROUTE_OPERAND_PROBES_REPORT.md
tri_modal_modular_grokking/analysis/phase6_route_operand_probes/pipeline.log
tri_modal_modular_grokking/analysis/phase6_route_operand_probes/figures/
RESULTS_TRI_MODAL_ROUTE_SUBSPACE_PATCHING.md
tri_modal_modular_grokking/analysis/phase6_route_subspace_patching/route_subspace_patching_rows.csv
tri_modal_modular_grokking/analysis/phase6_route_subspace_patching/summary.json
tri_modal_modular_grokking/analysis/phase6_route_subspace_patching/ROUTE_SUBSPACE_PATCHING_REPORT.md
tri_modal_modular_grokking/analysis/phase6_route_subspace_patching/pipeline.log
tri_modal_modular_grokking/analysis/phase6_route_subspace_patching/figures/
RESULTS_TRI_MODAL_ROUTE_TRANSPORT_MAPS.md
tri_modal_modular_grokking/analysis/phase6_route_transport_maps/route_transport_rows.csv
tri_modal_modular_grokking/analysis/phase6_route_transport_maps/route_transport_graph_rows.csv
tri_modal_modular_grokking/analysis/phase6_route_transport_maps/summary.json
tri_modal_modular_grokking/analysis/phase6_route_transport_maps/ROUTE_TRANSPORT_MAPS_REPORT.md
tri_modal_modular_grokking/analysis/phase6_route_transport_maps/pipeline.log
tri_modal_modular_grokking/analysis/phase6_route_transport_maps/figures/
RESULTS_TRI_MODAL_RESCUED_ROUTE_LOCAL_NUMBER_IMAGE.md
RESULTS_TRI_MODAL_RESCUED_ROUTE_LOCAL_ALL.md
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_image_text_balanced/RESCUED_ROUTE_LOCAL_ANALYSIS_REPORT.md
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_image_text_balanced/rescued_route_local_path_rows.csv
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_image_text_balanced/rescued_route_local_subspace_rows.csv
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_text_image_balanced/RESCUED_ROUTE_LOCAL_ANALYSIS_REPORT.md
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_text_image_balanced/rescued_route_local_path_rows.csv
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_text_image_balanced/rescued_route_local_subspace_rows.csv
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/RESCUED_ROUTE_LOCAL_ANALYSIS_REPORT.md
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/rescued_route_local_path_rows.csv
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/rescued_route_local_path_summary.csv
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/rescued_route_local_subspace_rows.csv
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/rescued_route_local_subspace_summary.csv
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/summary.json
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/path_runs/
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_number_image_balanced/subspace_figures/figures/
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_image_number_balanced/RESCUED_ROUTE_LOCAL_ANALYSIS_REPORT.md
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_image_number_balanced/rescued_route_local_path_rows.csv
tri_modal_modular_grokking/analysis/phase7_rescued_route_local_image_number_balanced/rescued_route_local_subspace_rows.csv
tri_modal_modular_grokking/analysis/phase4_full_crossmodal_seed1705/full_final_eval.json
tri_modal_modular_grokking/analysis/phase4_full_crossmodal_seed1705/full_final_per_cell_accuracy.csv
tri_modal_modular_grokking/analysis/phase4_full_crossmodal_seed1705/full_final_errors.csv
tri_modal_modular_grokking/analysis/phase4_full_crossmodal_seed1706/full_final_eval.json
tri_modal_modular_grokking/analysis/phase4_full_crossmodal_seed1706/full_final_per_cell_accuracy.csv
tri_modal_modular_grokking/analysis/phase4_full_crossmodal_seed1706/full_final_errors.csv
```
