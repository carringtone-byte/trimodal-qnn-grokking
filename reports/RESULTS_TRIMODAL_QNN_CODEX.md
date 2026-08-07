# Trimodal QNN Codex Results

Run date: 2026-07-05; updated 2026-07-10

Branch: `trimodal-qnn-codex`

Primary implementation:

```text
trimodal_qnn_codex/
```

Primary queue:

```text
trimodal_qnn_codex/outputs/phase1_v1_v2_queue
```

Primary result artifacts:

```text
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons_v2
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons_v2
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_5k
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_20k_from_5k
trimodal_qnn_codex/analysis/phase1_operand_query_mod97_bitter_lesson_20k_checkpoint
trimodal_qnn_codex/BITTER_LESSON_TRANSLATION.md
```

## Executive Conclusion

The branch now contains a self-contained trimodal QNN implementation for
modular addition over number, English text, and rendered digit image inputs. It
now implements three problem formulations:

1. `three_sector`: a QNN-native coherent superposition over text, number, and
   image whole-expression sectors.
2. `ordered_route`: an explicit nine-route formulation over ordered operand
   modalities `TT`, `TN`, `TI`, `NT`, `NN`, `NI`, `IT`, `IN`, `II`.
3. `operand_query`: a classical-trimodal-inspired three-sector route problem
   with an operand-a sector, an operand-b sector, and an answer-query sector.
   The two operand sectors receive value-plus-modality features, while the
   answer-query sector is learned and initially contains no operand value.
   The head reads only the answer-query sector, so the dense sector mixer must
   move operand information into the query carrier.

There are now two different "strongest" results, depending on which scientific
question is being asked.

For the easier coherent-sector question, the strict layerwise Dirac-mean
three-sector model remains the strongest QNN family. Across five seeds
`9301-9305`, it reached mean strict held-out exact accuracy `0.923474` at step
`2000`, with best seed `9302` reaching `0.941543`. Causal frequency patching
and sector CCA show that this is a real cyclic/Fourier code, but not a balanced
sector-common number manifold: the computation is text-dominant, with strong
text-image alignment and weaker final raw number-sector alignment.

For the harder route-conditioned composition question, the new
`operand_query` "bitter-lesson" run is the strongest result. At step `20000`
it reached train exact `0.905144` and strict held-out exact `0.794497`, with
held-out loss `0.584886`, telemetry Fourier energy `0.790078`, exact all-pair
query Fourier energy `0.967862`, and cross-sector ablation at chance
`0.010325`. It is far stronger than the earlier explicit ordered-route QNNs
(`0.357273` best held-out) and it succeeds specifically by using a dense
cross-sector route into an answer-query carrier.

The original residual-head `three_sector` and `ordered_route` runs remain
important controls. The completed 6-qubit `three_sector` model reached
train exact `1.000000` and held-out exact `0.627543` at step `30000`. The best
legacy ordered-route v2 checkpoint reached held-out exact `0.357273` before
nonfinite gradients. These older route variants show why the new
`operand_query` design matters: active-route sector initialization and
residual/full-sector heads made route composition fragile, while a small
query-sector readout and dense sector mixer produce a much cleaner route
composition signal.

Current interpretation after the strict Dirac-mean, CCA, and operand-query
follow-ups:

- The `three_sector` formulation supports a real causal cyclic rule.
- The legacy explicit `ordered_route` formulation is harder and mostly
  route-local or route-fragile.
- The `operand_query` formulation shows that route-conditioned QNN
  composition becomes much more learnable when the architecture mirrors the
  classical answer-slot mechanism: operand carriers feed a shared query sector
  through dense cross-modal interactions, and the head is forced to read the
  query carrier only.
- The strongest strict checkpoint is not modality-balanced: text is the
  dominant usable sector, text-image alignment is very strong, and number
  support is causal but weakly linearly aligned in the final raw sector state.
- The QNN learns measurable Fourier structure before exact finite-residue
  selection. The strict Dirac-mean objective sharply improves selection in the
  three-sector family; the operand-query objective shows that removing
  auxiliary baggage and reading only the answer-query sector can make ordered
  composition learnable.
- Nonfinite-gradient failure remains a first-class result for the older
  sector-heavy variants, but the operand-query run completed the 20k target
  without becoming the bottleneck.

## Scientific Question

The classical trimodal experiments showed that a shared transformer backbone
can learn a late cyclic answer substrate across number, text, and image
modalities, but that directed route access can be seed- and route-specific.
The QNN experiment asks whether a data-reuploading quantum-style model can
learn the same kind of shared modular rule under a more structured inductive
bias:

```text
sum sector lambda_sector |sector>|chi_sector(a,b)>
```

The key issue is not benchmark accuracy alone. The experiment asks which
mechanistic regime is learned:

| hypothesis | meaning | current status |
| --- | --- | --- |
| H1 shared cyclic rule | one common residue/Fourier coordinate across modalities or sectors | supported at the readout/cyclic-code level for strict `three_sector` and strongly supported at the answer-query readout level for `operand_query`; still not balanced across raw sectors |
| H2 route-specific cyclic rules | each modality/route learns a separate cyclic algorithm | plausible for weak legacy ordered-route checkpoints; weakened by `operand_query`, where all nine routes share the same query-sector head and all routes are above `0.731` held-out |
| H3 translation-to-value coordinate | modalities are converted into a shared residue-value coordinate before cyclic addition | not supported as a literal scalar number or simple final number-sector coordinate; better framed as translation into a shared cyclic representation carried by text/query-dominant coordinates |
| H4 no true cyclic rule | held-out performance is shortcut/memorization/noise | strongly disfavored for strict `three_sector` by causal frequency patching; strongly disfavored for `operand_query` by strict pair generalization, chance cross-ablation, layerwise late readout, and exact query Fourier energy `0.967862` |

The QNN is therefore a mechanistic object, not just a classifier. Its relevant
internal objects are sector amplitudes, sector-mixing unitaries, layerwise
complex statevectors, measurement probabilities, predicted Fourier
coefficients, and the small Fourier-delta readout.

## Equation Source And Theory

The equation source for this experiment is:

```text
../qnn_derivations_numeric_trimodal.pdf
```

I also extracted the PDF text for local auditability at:

```text
analysis/qnn_derivations_numeric_trimodal_extracted.txt
```

The PDF is titled `Data-Reuploading QNNs, Fourier-Delta Heads, and Trimodal
Interaction` and dated June 17, 2026. Its central claim is the one tested
here: the final Fourier-delta score can have the same finite cyclic form in a
numeric QNN and a trimodal QNN, but in the trimodal case that kernel is
mechanistically meaningful only if the modalities interact and align into a
shared cyclic representation.

### Numeric Fourier-Delta Spine

For modulus `p`, operands are embedded as cyclic phases:

```text
theta_a = 2*pi*a/p
theta_b = 2*pi*b/p
exp(2*pi*i*(a+b)/p) = exp(i*theta_a) * exp(i*theta_b)
```

A data-reuploading QNN with input-conditioned rotations produces measured
features that are finite trigonometric polynomials over the operand torus:

```text
z_m(a,b) = sum_{(u,v) in Omega_m} c_{u,v,m} exp(i*(u*theta_a + v*theta_b))
```

Modular addition lives on diagonal modes `(u,v) = (k,k)` because:

```text
g(a+b) = sum_k g_hat_k exp(2*pi*i*k*(a+b)/p)
       = sum_k g_hat_k exp(i*k*theta_a) exp(i*k*theta_b)
```

The target class indicator is the finite cyclic delta:

```text
delta_p(a+b-c) = (1/p) * sum_{k=0}^{p-1} exp(2*pi*i*k*(a+b-c)/p)
```

For odd prime `p=97`, the real class-separating part is a cosine series:

```text
delta_97(t) = 1/97 + (2/97) * sum_{k=1}^{48} cos(2*pi*k*t/97)
```

The implemented readout truncates this to `K=21` learned frequencies. It
predicts sum-phase coefficient pairs:

```text
q_hat_k = (u_hat_k, v_hat_k)
        ~ (cos(2*pi*k*(a+b)/p), sin(2*pi*k*(a+b)/p))
```

and scores candidate class `c` by a circular matched filter:

```text
ell_c = beta_c + gamma * sum_{k=1}^K rho_k *
        [u_hat_k*cos(2*pi*k*c/p) + v_hat_k*sin(2*pi*k*c/p)]
```

If `q_hat_k` is the true sum phase, the bracket becomes:

```text
cos(2*pi*k*((a+b)-c)/p)
```

So all class logits are shifted copies of one finite cyclic kernel:

```text
kappa_K(t) = sum_{k=1}^K rho_k cos(2*pi*k*t/p)
t = (a+b-c) mod p
```

This is why our mech interp emphasizes Fourier cutoffs, frequency-band
patching, same-sum invariance, and phase error: the readout is mathematically
supposed to expose a cyclic phase code, not an arbitrary class head.

### Trimodal Extension

The trimodal derivation replaces a simple numeric initial state by a coherent
sector state:

```text
|psi_0(a,b)> = sum_{m in {T,N,I}} lambda_m |m>|chi_m(a,b)>
```

The PDF writes the equal-amplitude case as:

```text
|psi_0(a,b)> = (1/sqrt(3)) *
  ( |T>|chi_T(a,b)> + |N>|chi_N(a,b)> + |I>|chi_I(a,b)> )
```

The implemented strict three-sector run uses exactly this fixed-equal sector
amplitude regime: `lambda_T = lambda_N = lambda_I = 1/sqrt(3)` at
initialization, with no answer residue supplied to any branch.

The conceptual layer decomposition in the PDF is:

```text
U_l = C_cross,l C_intra,l R_data,l
```

where data reuploading injects modality-specific operand features, intra-state
operations process the content, and cross-sector interaction prevents the
model from being only a late average of three independent adders. In code this
is realized as:

```text
feature-conditioned Rx/Ry/Rz rotations
ring entangling permutations over the state basis
learned sector unitary U_sector,l = exp(-i H_l)
```

The learned sector unitary is the concrete implementation of the PDF's
off-diagonal interaction idea:

```text
|T><N| x A_TN + |N><T| x A_TN^dagger
|T><I| x A_TI + |I><T| x A_TI^dagger
|N><I| x A_NI + |I><N| x A_NI^dagger
```

Our implementation applies a learned Hermitian-generator sector unitary over
the sector register. It is therefore a compact sector-level analogue of these
off-diagonal terms rather than a full separate content operator for every
source-target pair.

The later `operand_query` variant is the most direct implementation of the
lesson we learned from the classical trimodal transformer. It does not present
three complete expression sectors. Instead it presents:

```text
|psi_0(a,b,r)> =
  lambda_A |operand_a>|chi_{r_a}(a)>
+ lambda_B |operand_b>|chi_{r_b}(b)>
+ lambda_Q |answer_query>|q>
```

where `r = r_a r_b` is one of the nine ordered modality routes. The answer
query state `|q>` is learned and route-position tagged, but it does not receive
the answer or operand values directly. The readout is restricted to the
answer-query sector:

```text
ell_c = DiracDeltaHead(measure_Q( U_L ... U_1 |psi_0(a,b,r)> ))
```

This makes the sector interaction terms unavoidable. If the dense mixer does
not route operand information from `operand_a` and `operand_b` into
`answer_query`, accuracy must remain at chance. The 20k result shows exactly
the desired separation: normal held-out accuracy is `0.794497`, while
cross-sector ablation collapses to chance `0.010325`.

The PDF makes the crucial distinction between a clean aligned trimodal kernel
and a more general pre-alignment kernel. If all modalities align their operand
phase coordinates, the trimodal system can collapse back to the numeric
operand torus. Before that collapse, the multimodal addition diagonal is
distributed:

```text
sum_{m in {T,N,I}} u_m = sum_{m in {T,N,I}} v_m = k
```

and the sum-phase estimator can decompose into unimodal and cross-modal
pathway terms:

```text
q_hat_tri,k =
  A_{k,T}  q_hat_{k,T}
+ A_{k,N}  q_hat_{k,N}
+ A_{k,I}  q_hat_{k,I}
+ A_{k,TN} q_hat_{k,TN}
+ A_{k,TI} q_hat_{k,TI}
+ A_{k,NI} q_hat_{k,NI}
+ A_{k,TNI}q_hat_{k,TNI}
```

The clean kernel:

```text
ell_c ~ sum_{k=1}^K rho_k cos(2*pi*k*((a+b)-c)/p)
```

is therefore the successful-alignment limit, not something guaranteed by a
three-sector input state alone. This is exactly why our strongest result is
subtle: the Dirac-mean checkpoint learns a real cyclic kernel, but CCA and
sector tomography show that the pathway decomposition is asymmetric rather
than balanced.

### Interpretability Tests From The Derivation

The PDF specifically motivates the tests we ran:

| PDF test idea | implemented audit |
| --- | --- |
| modality/sector ablation | sector masks, cross-sector ablation, sector path ablation |
| cross-modal phase alignment | CCA over sector states, anchor-overlap CCA, single-sector probes |
| distributed Fourier energy | exact all-pair Fourier energy, low-frequency cutoff sweeps |
| cross-modal patching | frequency-coordinate clean/corrupt patching and ablation |
| same-sum orbit consistency | same-sum feature ratio and same-sum KL training term |

The current evidence should therefore be read against the PDF's criterion for
real multimodal interaction: not merely high accuracy, but cyclic energy,
causal frequency necessity/sufficiency, sector-interaction dependence, and
cross-sector alignment structure.

## Data Preparation

The data layer is implemented in:

```text
trimodal_qnn_codex/data.py
trimodal_qnn_codex/render.py
```

The task is modular addition:

```text
y = (a + b) mod 97
```

The split is strict at the ordered operand-pair level:

| item | value |
| --- | ---: |
| modulus | 97 |
| all ordered pairs | 9409 |
| train fraction | 0.30 |
| train pairs | 2823 |
| held-out pairs | 6586 |
| strict pair split across modalities | true |

For `three_sector`, each pair creates one example. The model receives a
coherent set of three whole-expression branches:

```text
text:   "forty two plus nine"
number: learned numeric embeddings plus sin/cos operand angles
image:  seven-segment rendering of "42 + 9"
```

For `ordered_route`, each pair creates up to nine active-route examples. A
route specifies the modality of operand `a` and the modality of operand `b`.
The default phase-1 ordered-route configs train and evaluate all nine routes:

```text
TT TN TI
NT NN NI
IT IN II
```

This means the ordered-route datasets contain:

| split | pairs | routes | records |
| --- | ---: | ---: | ---: |
| train | 2823 | 9 | 25407 |
| held-out | 6586 | 9 | 59274 |

The `operand_query` data layer uses the same strict pair split and the same
nine ordered routes as `ordered_route`, but each record is decomposed into
three semantic sectors instead of one active route sector:

```text
sector 0: operand_a, carrying value a in modality route[0]
sector 1: operand_b, carrying value b in modality route[1]
sector 2: answer_query, carrying only a learned query/position feature
```

Thus the route is still ordered and modality-specific, but the model no longer
gets a monolithic route feature that can behave like a separate route-local
problem. The answer has to be assembled by interactions among the two operand
sectors and the query sector.

Rendering details:

- Text values use deterministic English number words, tokenized with
  `<bos>`, `<eos>`, and `plus`.
- Number values use learned embeddings, and whole-expression number features
  also receive circular operand features `cos(2*pi*a/p)`, `sin(2*pi*a/p)`,
  `cos(2*pi*b/p)`, `sin(2*pi*b/p)`.
- Image values and pairs use small deterministic seven-segment grayscale
  glyphs, not natural images.
- The rendered data is generated on the fly into tensors/buffers; no external
  corpus is required.

This makes the experiment reproducible and blocks ordinary train/test leakage:
held-out numeric pairs are unseen in every modality and route.

## Architecture

The main model is implemented in:

```text
trimodal_qnn_codex/models.py
trimodal_qnn_codex/quantum.py
trimodal_qnn_codex/amplitudes.py
trimodal_qnn_codex/heads.py
```

### Encoders

`TrimodalEncoders` maps each modality into real-valued feature vectors:

- number value encoder: learned `Embedding(p, operand_feature_dim)`
- number pair encoder: MLP over two value embeddings plus circular operand
  angles
- text value/pair encoder: token embedding followed by masked mean pooling and
  a small MLP/LayerNorm
- image value/pair encoder: two-layer CNN, adaptive pooling, linear projection,
  and LayerNorm
- ordered-route encoder: concatenates `feature(a in route[0])`,
  `feature(b in route[1])`, and a learned route embedding
- operand-query encoder: builds `value_feature(a, route[0])`,
  `value_feature(b, route[1])`, adds learned modality and sector-position
  embeddings, and supplies a learned answer-query feature with no direct
  operand value

The encoders are deliberately small. They provide modality translation but are
not intended to contain a full arithmetic algorithm.

### State Initialization

`FeatureToState` linearly projects each sector feature into real and imaginary
parts of a complex statevector and normalizes it:

```text
feature -> Linear(feature_dim, 2 * 2^n_qubits) -> complex state -> L2 normalize
```

The model state has shape:

```text
[batch, sector_count, 2^n_qubits]
```

For `three_sector`, `sector_count = 3` and sectors are `T`, `N`, `I`. The
initial state is:

```text
lambda_T |T>|chi_T(a,b)> + lambda_N |N>|chi_N(a,b)> + lambda_I |I>|chi_I(a,b)>
```

The current phase-1 configs use fixed equal amplitudes:

```text
|lambda_T|^2 = |lambda_N|^2 = |lambda_I|^2 = 1/3
```

For `ordered_route`, `sector_count = 9`. In the default `active_route`
initialization, only the active route sector is populated at input. The code
also supports `all_route_superposition`, which is an important next case but
was not part of the completed phase-1 queue.

For `operand_query`, `sector_count = 3` but the semantics are not `T`, `N`,
and `I`. They are:

```text
operand_a, operand_b, answer_query
```

The active route is represented inside the operand sector features, not as a
separate route sector. The initialized state is therefore:

```text
lambda_A |operand_a>|chi_{r_a}(a)>
+ lambda_B |operand_b>|chi_{r_b}(b)>
+ lambda_Q |answer_query>|q>
```

The phase-1 operand-query configs used fixed equal sector amplitudes
`1/sqrt(3)`. This is not a claim that the three sectors remain equally used
after the circuit; it is only the input normalization.

### Data-Reuploading Circuit

Each `DataReuploadingLayer` performs:

1. Feature-conditioned single-qubit rotations.
2. A ring of CNOT-like permutation entanglers.
3. Optional cross-sector mixing.

For each qubit, the layer projects sector features into three angles and
applies:

```text
Rz(theta_z) -> Ry(theta_y) -> Rx(theta_x)
```

The entangler applies CNOT permutations along adjacent qubits and closes the
ring from the last qubit to the first. This is a simulated statevector QNN, not
a hardware run.

### Sector Mixing

Cross-sector interaction is implemented as a learned unitary:

```text
U_sector = exp(-i H)
H = (raw + raw^dagger) / 2
```

This directly addresses the proposal's off-diagonal interaction terms. In
`three_sector`, the mixer can implement interactions analogous to:

```text
|T><N| x A_TN + |N><T| x A_TN^dagger
|T><I| x A_TI + |I><T| x A_TI^dagger
|N><I| x A_NI + |I><N| x A_NI^dagger
```

In `ordered_route`, the same mechanism mixes the nine ordered route sectors.
The current route-explicit runs are therefore not missing route interactions;
they test whether the active-route state plus learned sector unitary can build
a route-compositional cyclic rule.

In `operand_query`, the dense sector mixer is the core architectural object.
It is the only path by which operand values can reach the measured
answer-query sector. This is deliberately closer to the classical
answer-slot/answer-query mechanism than the older active-route QNN design. It
also removes the unnecessary inductive baggage of factorized all-route priors:
the model can learn commutativity or ordered asymmetry from data through the
same dense interaction matrix.

### Measurement And Readout

The final complex state is measured into:

```text
per-sector computational-basis probabilities
sector masses
```

These real features are passed to a small cyclic readout. The original
phase-1 runs used `FourierDeltaHead`, which predicts low-frequency Fourier
coefficients:

```text
q_hat[k] = (cos phase_k, sin phase_k), k = 1..K
```

and synthesizes class logits by matching those coefficients against residue
class Fourier bases. The phase-1 configs use `K = 21`.

The configs currently set `fourier_residual: true`, adding a small linear
residual head from measured features to residue logits. This is useful for
training, but it is a mechanistic caveat: a stricter follow-up should ablate
or freeze the residual readout to prove arithmetic is not hiding in the
linear classifier.

The follow-up three-sector Dirac run removed this caveat. It uses
`head_type: layerwise_dirac_mean`: every QNN layer has a strict
`DiracDeltaHead`, each head maps measured features to Fourier coefficient
pairs and residue logits, and the final logits are the uniform mean of the
four layer logits. There is no residual class head in this path.

For the strict step-2000 artifact, each layer `l` produces:

```text
f_l = measure(|psi_l>)
q_hat_{l,k} = W_l LayerNorm(f_l) + b_l
rho_{l,k} = softplus(alpha_{l,k})
ell_{l,c} = exp(s_l) * sum_{k=1}^K rho_{l,k} *
            [u_hat_{l,k} cos(2*pi*k*c/p)
             + v_hat_{l,k} sin(2*pi*k*c/p)]
            + beta_{l,c}
ell_c = (1/L) * sum_{l=1}^L ell_{l,c}
```

This is the direct finite-kernel form from the PDF, applied after every QNN
layer and averaged. The artifact used `K=21`, `p=97`, Fejer-initialized
trainable frequency weights, `dirac_coefficient_mode: none`, and no residual
class head.

The `operand_query` bitter-lesson run uses a stricter final-slot analogue:

```text
readout_mode = query_sector
head_type = dirac_delta
fourier_residual = false
layerwise_dirac_loss_weight = 0
fourier_auxiliary_weight = 0
same_sum_loss_weight = 0
hard_neighbor_margin_weight = 0
```

Only measured answer-query probabilities and the query-sector mass are visible
to the head. The final head is therefore small and cyclic: it maps query-sector
features to Fourier coefficient pairs and class logits through the
Dirac/Fourier kernel. At the 20k checkpoint, full accuracy and Fourier-only
accuracy are identical because there is no residual class path.

## Loss Function

Training is implemented in:

```text
trimodal_qnn_codex/train.py
```

The general implemented loss family is:

```text
L = CE(logits, y)
  + w_same * L_same_sum
  + w_amp * L_amplitude_balance
  + w_fourier * L_fourier_aux
  + w_layerwise * L_layerwise_dirac
  + w_margin * L_hard_neighbor_margin
```

Terms:

- `CE`: exact residue cross-entropy.
- `L_same_sum`: symmetric KL between predictions for `(a,b)` and
  `(a+d,b-d)`, encouraging same-sum invariance.
- `L_amplitude_balance`: optional equal-sector amplitude regularizer.
- `L_fourier_aux`: MSE between normalized predicted Fourier coefficient pairs
  and the true residue Fourier phases.
- `L_layerwise_dirac`: optional cross-entropy on each layer's Dirac/Fourier
  readout, depth-weighted when configured.
- `L_hard_neighbor_margin`: hinge margin against nearby modular offsets,
  currently offsets `+/-1` and `+/-2`.

The loss is intentionally shaped by lessons from the unimodal QNN work:
Fourier structure is easy to form, but exact finite-residue selection requires
margin/Dirac-like sharpening pressure.

For the strict Dirac-mean artifact that produced the text-binding result, the
active objective was:

```text
L_artifact =
  CE(mean_layer_logits, y)
  + 0.001 * L_same_sum
  + 0.50  * L_layerwise_dirac
  + 0.10  * L_hard_neighbor_margin
```

with:

```text
w_amp = 0
w_fourier = 0
```

The terms are concretely:

```text
L_same_sum = 0.5 * [ KL(p(a,b) || p(a+d,b-d))
                   + KL(p(a+d,b-d) || p(a,b)) ]
```

where `d` is sampled uniformly from `1..p-1`, preserving the target sum.
Because the code evaluates `same_sum_loss` before `layerwise_head_loss`, the
stored layerwise logits for this run come from the most recent same-sum-shifted
forward pass when `same_sum_loss_weight` is nonzero. Since `(a+d)+(b-d) = a+b`
mod `p`, the layerwise Dirac loss still uses the correct residue label. In
effect, the auxiliary layerwise readouts are trained on an orbit-augmented
same-sum view rather than only the original minibatch view.

The layerwise Dirac term is:

```text
L_layerwise_dirac =
  mean_l alpha_l * CE(ell_l, y)
```

with depth weighting enabled. For `L=4` layers the normalized weights are:

```text
alpha = [0.4, 0.8, 1.2, 1.6]
```

The hard-neighbor term enforces local finite-residue sharpening against the
nearest offsets:

```text
L_margin = mean_{o in {1,2}, sign in {-1,+1}}
           ReLU(margin + ell_{y+sign*o} - ell_y) * focus_weight
margin = 2.0
focus_gamma = 0.50
```

This matters scientifically: the objective is not just "learn a classifier".
It combines the PDF's finite cyclic kernel with explicit pressure for
same-sum invariance, per-layer cyclic readability, and local Dirac-like
answer sharpening.

The `operand_query` bitter-lesson objective intentionally removes that
auxiliary shaping to test the minimal cross-modal route mechanism:

```text
L_operand_query = CE(DiracDeltaHead(query_sector_features), y)
```

with:

```text
w_same = 0
w_amp = 0
w_fourier = 0
w_layerwise = 0
w_margin = 0
fourier_residual = false
readout_mode = query_sector
route_mixer_type = dense
```

This was a deliberate "bitter lesson" translation from the classical
trimodal experiments. The QNN is not helped by bespoke all-route
factorization, same-sum auxiliary constraints, or layerwise losses in this
variant. It receives the raw supervised residue target and must discover the
usable cross-sector query carrier through optimization.

## Training Regime

Phase-1 full runs used:

| setting | v1 three-sector | v1 ordered-route | v2 three-sector | v2 ordered-route |
| --- | ---: | ---: | ---: | ---: |
| modulus | 97 | 97 | 97 | 97 |
| n_qubits | 6 | 6 | 7 | 7 |
| n_layers | 4 | 4 | 4 | 4 |
| sectors | 3 | 9 | 3 | 9 |
| feature_dim | 64 | 64 | 64 | 64 |
| Fourier frequencies | 21 | 21 | 21 | 21 |
| batch size | 256 | 256 | 256 | 256 |
| target steps | 30000 | 30000 | 30000 | 30000 |
| checkpoint interval | 1000 | 1000 | 1000 | 1000 |
| parameter count | 71451 | 126237 | 107239 | 215401 |

Optimization details:

- optimizer: AdamW
- weight decay: `0.0001`
- gradient clipping enabled
- parameter clipping enabled
- sector-mixer clipping enabled
- head `scale_log` clamp enabled
- nonfinite loss/gradient detection enabled
- nonfinite checkpoints saved as `checkpoint_nonfinite_*.pt`

The completed v1 three-sector run resumed from `checkpoint_5000.pt` with a
fresh optimizer after an earlier NaN run. That earlier failed run is preserved
as:

```text
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons_failed_nan_v1
```

The strict three-sector layerwise Dirac-mean follow-up used:

| setting | value |
| --- | ---: |
| seed | 9301 |
| modulus | 97 |
| train fraction | 0.30 |
| train pairs | 2823 |
| held-out pairs | 6586 |
| problem mode | `three_sector` |
| n_qubits | 7 |
| state dimension per sector | 128 |
| sectors | 3 |
| QNN layers | 4 |
| feature dim | 64 |
| Fourier/Dirac frequencies | 21 |
| head | `layerwise_dirac_mean` |
| residual class head | false |
| sector amplitudes | fixed equal |
| cross mixing | true |
| batch size | 256 |
| learning rate | 0.0004 |
| weight decay | 0.0001 |
| checkpoint interval | 1000 |
| max grad norm | 0.5 |
| parameter count | 121944 |

Output directory:

```text
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean
```

It reached the best regular checkpoint at step `2000` and then hit a
nonfinite gradient at step `2292`, preserving both `checkpoint_2000.pt` and
`checkpoint_nonfinite_2292.pt`.

Training telemetry for the artifact:

| step | train acc | held-out acc | held-out loss | cross-ablate held-out | same-sum ratio | CE | layerwise loss | neighbor loss | total loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.013461 | 0.009566 | 4.811875 | 0.009718 | 0.975979 | 4.808010 | 5.753506 | 2.027386 | 7.887504 |
| 1000 | 0.957492 | 0.644094 | 1.296334 | 0.563012 | 0.530066 | 0.946897 | 1.197656 | 0.682700 | 1.614253 |
| 2000 | 1.000000 | 0.912542 | 0.506894 | 0.794716 | 0.570536 | 0.226778 | 0.426140 | 0.034472 | 0.443458 |
| 2292 | nonfinite gradient | n/a | n/a | n/a | n/a | 0.003279 | 0.247915 | 0.000000 | 0.127762 |

The operand-query bitter-lesson follow-up used:

| setting | value |
| --- | ---: |
| seed | inherited config default |
| modulus | 97 |
| train fraction | 0.30 |
| train pairs | 2823 |
| held-out pairs | 6586 |
| routes | 9 |
| train records | 25407 |
| held-out records | 59274 |
| problem mode | `operand_query` |
| n_qubits | 7 |
| state dimension per sector | 128 |
| sectors | 3: `operand_a`, `operand_b`, `answer_query` |
| QNN layers | 4 |
| feature dim | 64 |
| Fourier/Dirac frequencies | 21 |
| head | `dirac_delta` |
| readout | `query_sector` |
| residual class head | false |
| route mixer | dense |
| sector amplitudes | fixed equal |
| batch size | 256 |
| checkpoint interval | 1000 |
| parameter count | 61713 |

Training was staged only for operational convenience: the model was first run
to 5k and then continued to 20k from the saved 5k checkpoint while preserving
the original checkpoint series.

| stage | step | train acc | held-out acc | held-out loss | cross-ablate held-out | Fourier energy | same-sum ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| initial run | 5000 | 0.524265 | 0.276361 | 1.720179 | 0.010325 | 0.693910 | n/a |
| continuation | 20000 | 0.905144 | 0.794497 | 0.584886 | 0.010325 | 0.790078 | 0.576598 |

The continuation took `7986.66` seconds for the additional 15k steps and
completed without the nonfinite-gradient failures seen in the older
sector-heavy route variants.

## Code Logic Map

| file | role |
| --- | --- |
| `trimodal_qnn_codex/data.py` | strict pair split, route expansion, `three_sector`, `ordered_route`, and `operand_query` dataset metadata |
| `trimodal_qnn_codex/render.py` | English text rendering and deterministic seven-segment image rendering |
| `trimodal_qnn_codex/models.py` | trimodal encoders, state initialization, full QNN module, operand-query sector features, and query-sector readout |
| `trimodal_qnn_codex/quantum.py` | complex statevector simulation, feature-conditioned rotations, entanglers, sector unitaries |
| `trimodal_qnn_codex/amplitudes.py` | fixed or learned normalized sector amplitudes |
| `trimodal_qnn_codex/heads.py` | Fourier targets, Fourier-delta readout, and strict Dirac-delta cyclic readout |
| `trimodal_qnn_codex/diagnostics.py` | held-out evaluation, route stats for ordered-route and operand-query modes, Fourier energy, same-sum feature ratio, phase MAE, cross-mixing ablation |
| `trimodal_qnn_codex/train.py` | training loop, auxiliary losses, checkpointing, nonfinite guards, resume support |
| `trimodal_qnn_codex/analyze_checkpoints.py` | checkpoint-level exact QNN diagnostics and layerwise logit lenses |
| `trimodal_qnn_codex/analyze_dirac_causal.py` | frequency-resolved causal patching and sector scattering tomography for layerwise Dirac checkpoints |
| `trimodal_qnn_codex/analyze_sector_cca.py` | sector/subspace CCA, text-hub scoring, anchor-overlap analysis, and single-sector residue probes |
| `trimodal_qnn_codex/run_lessons_queue.py` | skip-aware queue runner with logs and status JSONL |
| `trimodal_qnn_codex/run_seed_pipeline.py` | four-seed strict Dirac-mean training plus checkpoint, causal, and CCA mech-interp pipeline |
| `tests/test_trimodal_qnn_codex.py` | unit tests for amplitudes, datasets, sector unitarity, Fourier head, forward norms, seed config generation, and checkpoint selection |

## Results

The full queue ran four phase-1 configurations with `--continue-on-error`.
Total queue wall time was `60390.63` seconds. The queue status is stored in:

```text
trimodal_qnn_codex/outputs/phase1_v1_v2_queue/status.jsonl
trimodal_qnn_codex/outputs/phase1_v1_v2_queue/queue_summary.json
```

An exhaustive checkpoint-level mechanistic pass was then run over all `80`
regular saved checkpoints from the four phase-1 runs:

```text
trimodal_qnn_codex/analysis/phase1_checkpoint_mech
```

This pass evaluates exact all-pair behavior, layerwise frozen-head logit
lenses, final-state Fourier-only ablations, low-frequency cutoff readouts,
cross-sector ablations, route-local behavior, same-sum feature compression,
sector-mass dynamics, and nonfinite failure events.

### Aggregate Table

| run | status | best step | train acc at best | held-out acc at best | held-out loss | Fourier energy | phase MAE | same-sum feature ratio | cross-ablation acc | failure |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v1 `three_sector`, 6 qubits | completed | 30000 | 1.000000 | 0.627543 | 1.053505 | 0.665197 | 0.958456 | 0.274636 | 0.407683 | none after resume |
| v1 `ordered_route`, 6 qubits | failed | 21000 | 0.347385 | 0.175170 | 2.971650 | 0.624647 | 1.453506 | 0.997060 | 0.175170 | nonfinite gradient at 21331 |
| v2 `three_sector`, 7 qubits | failed | 12000 | 0.867163 | 0.543729 | 1.335579 | 0.699410 | 1.136958 | 0.199142 | 0.065746 | nonfinite gradient at 12107 |
| v2 `ordered_route`, 7 qubits | failed | 16000 | 0.451057 | 0.357273 | 1.948037 | 0.689657 | 1.312021 | 0.984467 | 0.357273 | nonfinite gradient at 17464 |
| `operand_query` bitter-lesson, 7 qubits | completed | 20000 | 0.905144 | 0.794497 | 0.584886 | 0.790078 | 1.428773 | 0.576598 | 0.010325 | none |

The table above uses training-loop held-out telemetry for the Fourier column.
The exhaustive all-pair analysis gives higher exact full-grid Fourier-energy
values because the full grid no longer requires held-out-cell filling:

| run | best checkpoint | exact all-pair Fourier energy |
| --- | ---: | ---: |
| v1 `three_sector`, 6 qubits | 30000 | 0.945081 |
| v1 `ordered_route`, 6 qubits | 21000 | 0.889197 |
| v2 `three_sector`, 7 qubits | 12000 | 0.977639 |
| v2 `ordered_route`, 7 qubits | 16000 | 0.979104 |
| `operand_query` bitter-lesson, 7 qubits | 20000 | 0.967862 |

### Operand-Query Bitter-Lesson Route Model

Outputs:

```text
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_5k
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_20k_from_5k
```

Analysis:

```text
trimodal_qnn_codex/analysis/phase1_operand_query_mod97_bitter_lesson_20k_checkpoint
trimodal_qnn_codex/BITTER_LESSON_TRANSLATION.md
```

Final 20k metrics:

| metric | value |
| --- | ---: |
| step | 20000 |
| parameter count | 61713 |
| train records | 25407 |
| held-out records | 59274 |
| train accuracy | 0.905144 |
| train loss | 0.252270 |
| held-out accuracy | 0.794497 |
| held-out loss | 0.584886 |
| cross-mixing ablation accuracy | 0.010325 |
| held-out Fourier addition energy | 0.790078 |
| exact all-pair query Fourier energy | 0.967862 |
| held-out phase MAE | 1.428773 |
| held-out same-sum feature ratio | 0.576598 |
| state norm mean | 0.999995 |

Held-out route accuracies at 20k:

| route | held-out acc | held-out loss | exact route Fourier energy |
| --- | ---: | ---: | ---: |
| TT | 0.841634 | 0.449079 | 0.966307 |
| IT | 0.822502 | 0.488916 | 0.967633 |
| TI | 0.809292 | 0.495238 | 0.966652 |
| II | 0.806408 | 0.491940 | 0.967851 |
| NT | 0.798512 | 0.576096 | 0.966211 |
| NI | 0.791679 | 0.593109 | 0.965794 |
| TN | 0.781203 | 0.642268 | 0.965660 |
| IN | 0.767993 | 0.690435 | 0.966379 |
| NN | 0.731248 | 0.836899 | 0.963717 |

Layerwise frozen-head logit lens at the 20k checkpoint:

| state | held-out acc | exact query Fourier energy |
| --- | ---: | ---: |
| initial | 0.011084 | 1.000000 |
| layer 1 | 0.010376 | 0.998935 |
| layer 2 | 0.010511 | 0.875365 |
| layer 3 | 0.010696 | 0.885058 |
| layer 4 | 0.794497 | 0.967862 |

Interpretation:

- This is the strongest route-conditioned trimodal QNN result so far. It is
  not the strongest QNN result overall, because the strict coherent
  three-sector Dirac-mean seed sweep reaches higher held-out accuracy in an
  easier formulation.
- The result is mechanistically cleaner than the older ordered-route runs.
  The readout sees only the answer-query sector, the head is a strict
  Dirac/Fourier head, and there is no residual class head.
- Cross-sector ablation collapses to chance. This is the central causal sign:
  operand information must be routed into the query sector by the dense sector
  mixer.
- All nine ordered routes are learned well above chance, but not equally.
  `TT` is strongest and `NN` is weakest. That is not evidence for a literal
  number-manifold solution; it is evidence for a shared query-carrier rule
  whose route access remains modality dependent.
- The layerwise logit lens shows late answer formation. Earlier states have
  high Fourier-energy geometry under the final head but chance accuracy; only
  after layer 4 does the query sector contain the finite-residue information
  that the strict head can read.
- The best description is therefore not "the QNN learned a number." It learned
  a cyclic representation of the sum in an answer-query coordinate chart.
  That representation is number-like in the modular/Fourier sense, but it is
  not a literal scalar number or a balanced raw-sector number subspace.

### Completed Three-Sector Run

Output:

```text
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons
```

Final metrics:

| metric | value |
| --- | ---: |
| step | 30000 |
| parameter count | 71451 |
| train records | 2823 |
| held-out records | 6586 |
| train accuracy | 1.000000 |
| train loss | 0.012915 |
| held-out accuracy | 0.627543 |
| held-out loss | 1.053505 |
| cross-mixing ablation accuracy | 0.407683 |
| held-out Fourier addition energy | 0.665197 |
| held-out phase MAE | 0.958456 |
| held-out same-sum feature ratio | 0.274636 |
| state norm mean | 0.999995 |

Interpretation:

- The model fits the train split perfectly and generalizes substantially above
  chance under strict pair holdout.
- Cross-sector mixing matters: disabling cross mixing reduces held-out
  accuracy from `0.627543` to `0.407683`.
- The Fourier addition energy is high enough to indicate a cyclic scaffold,
  but not high enough to claim a clean finite-residue algorithm.
- The same-sum feature ratio `0.274636` indicates nontrivial compression by
  residue class, unlike the ordered-route runs where the ratio stays near `1`.

### Ordered-Route v1

Output:

```text
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons
```

Best metrics:

| metric | value |
| --- | ---: |
| best step | 21000 |
| parameter count | 126237 |
| train accuracy | 0.347385 |
| held-out accuracy | 0.175170 |
| held-out loss | 2.971650 |
| Fourier addition energy | 0.624647 |
| phase MAE | 1.453506 |
| same-sum feature ratio | 0.997060 |
| cross-ablation accuracy | 0.175170 |
| failure step | 21331 |

Interpretation:

- The model is learning above chance but remains weak.
- Cross-ablation is identical to normal accuracy because active-route
  initialization leaves little useful cross-sector superposition to ablate.
- The same-sum ratio near `1` indicates the measured representation has not
  collapsed onto residue classes.
- This is a route-explicit partial-learning failure, not a successful shared
  route-composition result.

### Three-Sector v2

Output:

```text
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons_v2
```

Best metrics:

| metric | value |
| --- | ---: |
| best step | 12000 |
| parameter count | 107239 |
| train accuracy | 0.867163 |
| held-out accuracy | 0.543729 |
| held-out loss | 1.335579 |
| Fourier addition energy | 0.699410 |
| phase MAE | 1.136958 |
| same-sum feature ratio | 0.199142 |
| cross-ablation accuracy | 0.065746 |
| failure step | 12107 |

Interpretation:

- The 7-qubit state space helps early learning and same-sum compression.
- The very low cross-ablation accuracy means the solution is heavily dependent
  on sector mixing.
- The run was promising but unstable; it failed before the long transition that
  allowed the 6-qubit three-sector model to reach `0.627543`.

### Ordered-Route v2

Output:

```text
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons_v2
```

Best metrics:

| metric | value |
| --- | ---: |
| best step | 16000 |
| parameter count | 215401 |
| train accuracy | 0.451057 |
| held-out accuracy | 0.357273 |
| held-out loss | 1.948037 |
| Fourier addition energy | 0.689657 |
| phase MAE | 1.312021 |
| same-sum feature ratio | 0.984467 |
| cross-ablation accuracy | 0.357273 |
| last good checkpoint | 17000 |
| failure step | 17464 |

Interpretation:

- This is the strongest legacy active-route checkpoint so far, but it has now
  been superseded for route-conditioned composition by the `operand_query`
  result.
- It is clearly better than the v1 ordered-route model, reaching roughly twice
  the held-out accuracy.
- It still does not show strong same-sum feature compression.
- It should be resumed from `checkpoint_16000.pt`, not the later
  `checkpoint_17000.pt`, because held-out accuracy fell by step `17000`.

## Mechanistic Reading

The checkpoint-level QNN evidence is best described as "late-forming cyclic
state, imperfect finite-residue readout." That refines the earlier phrasing of
"Fourier scaffold before exact algorithm." Trigonometric structure emerges
naturally from data reuploading, but exact modular residue selection and
readout calibration remain separate problems.

Strongest mechanistic signs:

- `three_sector` cross-ablation drop shows off-diagonal sector interactions
  are doing useful work.
- `operand_query` cross-ablation is decisive: normal held-out accuracy is
  `0.794497`, while ablated held-out accuracy is chance `0.010325`.
- `operand_query` is read from the answer-query sector only, so that collapse
  specifically identifies the operand-to-query sector interaction as causal.
- `three_sector` same-sum feature ratio is far below `1`, indicating residue
  class compression.
- Exact all-pair Fourier energy is high at the best checkpoints:
  `0.945081`, `0.889197`, `0.977639`, `0.979104`, and `0.967862` for the
  operand-query 20k checkpoint.
- The frozen final-head logit lens is near chance until the final circuit
  layer, then jumps sharply. For operand-query the held-out lens stays at
  `0.011084`, `0.010376`, `0.010511`, and `0.010696` through the initial
  state and first three layers, then jumps to `0.794497` at layer 4. This
  argues against a trivial encoder-only explanation.
- The v2 models have more exact Fourier energy but are less stable, suggesting
  the optimization landscape and readout calibration, not just
  representational capacity, are bottlenecks.

Important caveats:

- Fourier energy alone is not proof of a correct modular algorithm. Early
  near-chance checkpoints can already have high addition-diagonal energy, so
  the energy must be interpreted together with phase MAE, exact accuracy,
  same-sum compression, and low-frequency cutoff behavior.
- The residual readout often hurts, rather than helps, held-out behavior. The
  measured QNN state contains a better low-frequency cyclic readout than the
  full trained head exposes in three of the four best checkpoints.
- Ordered-route active initialization is a strict formulation. It may be too
  harsh for the first QNN route-composition test because only one route sector
  is populated before mixing.
- We have now run layerwise frozen-head probes, sector-mass diagnostics,
  frequency-resolved causal patching, sector scattering tomography, and
  sector/subspace CCA for the strongest strict Dirac-mean checkpoint.
  Route-to-route transport on explicit ordered-route QNN states remains open.

### Checkpoint Mech-Interp Findings

The strongest new finding is that the QNN states often support a better
Fourier-only or low-frequency readout than the trained full head:

| run | checkpoint | full held-out | Fourier-only held-out | best low-frequency cutoff |
| --- | ---: | ---: | ---: | ---: |
| v1 `three_sector` | 30000 | 0.627543 | 0.615093 | `k<=3`: 0.659885 |
| v2 `three_sector` | 12000 | 0.543729 | 0.633009 | `k<=3`: 0.636502 |
| v1 `ordered_route` | 21000 | 0.175170 | 0.227216 | `k<=3`: 0.227199 |
| v2 `ordered_route` | 16000 | 0.357273 | 0.429733 | `k<=1`: 0.429817 |

This changes the interpretation of the failed/partial runs. The state is not
merely failing to form cyclic structure. It often forms a usable low-frequency
cyclic representation, then the residual/full head degrades generalization.
The head scale is also saturated at the configured upper clamp (`exp(2.5) =
12.182494`) in all best checkpoints, which is a concrete calibration and
stability warning.

The layerwise frozen-head logit lens shows that usable answer information
appears almost entirely at the final QNN layer:

| run | checkpoint | initial held-out | layer 1 | layer 2 | layer 3 | layer 4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 `three_sector` | 30000 | 0.010629 | 0.014880 | 0.020346 | 0.016702 | 0.627543 |
| v2 `three_sector` | 12000 | 0.010021 | 0.030216 | 0.033860 | 0.037048 | 0.543729 |
| v1 `ordered_route` | 21000 | 0.009599 | 0.010021 | 0.011270 | 0.011236 | 0.175170 |
| v2 `ordered_route` | 16000 | 0.008756 | 0.008469 | 0.014340 | 0.010527 | 0.357273 |

The same-sum feature ratio distinguishes coherent-sector and ordered-route
solutions, with operand-query sitting between the two:

| run | best checkpoint | held-out same-sum ratio |
| --- | ---: | ---: |
| v2 `three_sector` | 12000 | 0.199142 |
| v1 `three_sector` | 30000 | 0.274636 |
| `operand_query` | 20000 | 0.576598 |
| v2 `ordered_route` | 16000 | 0.984467 |
| v1 `ordered_route` | 21000 | 0.997060 |

The coherent three-sector runs learn a real sum-compressed measured state. The
ordered-route runs learn above-chance cyclic readouts without a clean shared
same-sum manifold. Operand-query does not collapse as strongly as the
three-sector models, but it is no longer near `1`; the answer-query carrier
has meaningful residue-class compression while preserving route-specific
access differences.

Cross-sector ablation reinforces the distinction:

| run | full held-out | cross-ablate held-out | drop |
| --- | ---: | ---: | ---: |
| v1 `three_sector` | 0.627543 | 0.407683 | 0.219860 |
| v2 `three_sector` | 0.543729 | 0.065746 | 0.477984 |
| `operand_query` | 0.794497 | 0.010325 | 0.784172 |

For ordered-route active initialization, cross-ablation is effectively
unchanged. The active-route models are therefore not yet using a mature shared
route-composition unitary; they mostly decode active-route states. For
operand-query, the opposite is true: ablation destroys the computation because
the head can only read the answer-query sector after cross-sector transfer.

The best ordered-route v2 checkpoint is broad but shallow:

| route | full held-out | Fourier-only held-out | Fourier energy |
| --- | ---: | ---: | ---: |
| TT | 0.429244 | 0.477224 | 0.990024 |
| TN | 0.426207 | 0.542211 | 0.973978 |
| TI | 0.339812 | 0.437899 | 0.980147 |
| NT | 0.378378 | 0.500759 | 0.973979 |
| NN | 0.370938 | 0.444124 | 0.964825 |
| NI | 0.300182 | 0.359854 | 0.970964 |
| IT | 0.363954 | 0.421045 | 0.986828 |
| IN | 0.318555 | 0.361980 | 0.975305 |
| II | 0.288187 | 0.322502 | 0.979033 |

This is not a single-route artifact. Every ordered route is above chance, and
every route improves under Fourier-only readout, but no route is close to a
fully solved algorithm.

Final sector masses in the coherent models are also informative:

| run | checkpoint | T | N | I |
| --- | ---: | ---: | ---: | ---: |
| v1 `three_sector` | 30000 | 0.430 | 0.333 | 0.236 |
| v2 `three_sector` | 12000 | 0.446 | 0.333 | 0.221 |

The number sector remains near one third while text gains mass and image loses
mass. This suggests the coherent mixer privileges the more stable symbolic
channel while retaining image contribution.

### Layerwise Dirac-Mean Checkpoint

The strict three-sector Dirac-mean follow-up is the strongest trimodal QNN
checkpoint so far. It removes the residual class-head confound and directly
trains a Dirac/Fourier readout on every QNN layer. The key checkpoint is:

```text
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean/checkpoint_2000.pt
```

Exact all-pair and held-out analysis is stored in:

```text
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000
```

Best metrics:

| metric | value |
| --- | ---: |
| train accuracy | 1.000000 |
| held-out accuracy | 0.912542 |
| all-pair accuracy | 0.938782 |
| cross-sector ablation held-out | 0.794716 |
| addition-diagonal Fourier energy | 0.789818 |
| held-out same-sum ratio | 0.570536 |
| mean head scale | 1.963759 |
| top averaged frequency | 4 |
| parameter count | 121944 |
| failure step after checkpoint | 2292 |

Layerwise answer formation is progressive rather than final-layer-only:

| layer | held-out accuracy |
| --- | ---: |
| initial | 0.010932 |
| layer 1 | 0.488005 |
| layer 2 | 0.749013 |
| layer 3 | 0.898725 |
| layer 4 | 0.925600 |
| averaged Dirac-mean readout | 0.912542 |

This is a qualitatively different model from the earlier residual
Fourier-head runs. The cyclic residue code is already readable in intermediate
QNN states, and the final result is obtained without a residual linear class
head. The nonfinite gradient at step `2292` is still a real stability warning:
the useful checkpoint is step `2000`, not the failure checkpoint.

### Dirac-Mean Causal Mech-Interp

The novel causal pass is stored in:

```text
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_causal
```

It runs two tests on the held-out split (`6586` examples):

- frequency-resolved causal patching: patch clean Fourier coefficient pairs
  into corrupted examples, or replace clean coefficients with corrupted ones;
- sector scattering tomography: ablate one learned source-to-target sector
  mixer contribution and continue the QNN to the final layerwise Dirac-mean
  readout.

Frequency-band patching shows that the answer is a distributed cyclic code.
Single frequencies move margins but are not sufficient alone. Patching the
full `k=1..21` band exactly recovers the clean held-out accuracy, and patching
`k=1..13` recovers most of it:

| delta | band restored across all layers | patched clean accuracy | margin recovery |
| ---: | --- | ---: | ---: |
| 5 | full `1..21` | 0.912542 | 1.000000 |
| 2 | full `1..21` | 0.912542 | 1.000000 |
| 1 | full `1..21` | 0.912542 | 1.000000 |
| 13 | full `1..21` | 0.912542 | 1.000000 |
| 5 | low/mid `1..13` | 0.833890 | 0.986036 |
| 1 | low/mid `1..13` | 0.815062 | 0.876702 |
| 13 | low/mid `1..13` | 0.789250 | 0.991324 |
| 2 | low/mid `1..13` | 0.707410 | 0.898275 |

Band ablation gives the complementary necessity result: replacing clean
`k=1..13` with corrupted coefficients destroys the answer for the larger
offsets, while ablating only high frequencies is much less damaging:

| delta | band ablated across all layers | remaining clean accuracy | accuracy drop |
| ---: | --- | ---: | ---: |
| 13 | low/mid `1..13` | 0.000000 | 0.912542 |
| 5 | low/mid `1..13` | 0.000000 | 0.912542 |
| 2 | low/mid `1..13` | 0.002126 | 0.910416 |
| 1 | low/mid `1..13` | 0.042211 | 0.870331 |
| 13 | high `14..21` | 0.789250 | 0.123292 |
| 5 | high `14..21` | 0.833890 | 0.078652 |
| 1 | high `14..21` | 0.815062 | 0.097480 |

The sector tests show a text-dominant but not text-only computation:

| input sector mask | held-out accuracy |
| --- | ---: |
| all sectors | 0.912542 |
| text + image | 0.806408 |
| text + number | 0.737625 |
| text only | 0.700577 |
| number only | 0.031886 |
| image only | 0.011388 |

The learned sector unitaries remain close to identity, but the diagonal
text-to-text path is causally dominant. Ablating `T -> T` at early layers has
the largest effect:

| ablated sector path | ablated at layer | resulting accuracy | accuracy drop |
| --- | --- | ---: | ---: |
| `T -> T` | layer 1 | 0.033252 | 0.879289 |
| `T -> T` | layer 2 | 0.123140 | 0.789402 |
| `T -> T` | layer 3 | 0.424992 | 0.487549 |
| `I -> I` | layer 1 | 0.760401 | 0.152141 |
| `N -> N` | layer 1 | 0.806256 | 0.106286 |

Interpretation: the strict Dirac-mean model has a genuine causal cyclic code,
but not an evenly shared modality algorithm. The dominant computation rides on
the text sector, with number and image sectors providing measurable support.
This strengthens H1 relative to H4 for the three-sector problem, but it also
warns that "shared cyclic rule" should be qualified: the rule is shared at the
readout/state level more than it is balanced across modality sectors.

This directly matches the PDF's distinction between the algebraic kernel and
the mechanism that produces it. The model has learned a high-accuracy
truncated cyclic kernel, but the pathway decomposition is not a symmetric
`T/N/I` alignment. In the PDF's notation, the observed solution looks closer
to a large `T` and `T-I` pathway contribution plus smaller causal number
support than to equal `q_hat_{k,T} ~= q_hat_{k,N} ~= q_hat_{k,I}` phase
estimates.

### Sector CCA Binding Test

The text-dominant sector tomography raised a sharper question: did the model
choose text as a binding modality, or did text merely become the best
single-sector classifier? We ran a CCA audit on the step-2000 Dirac-mean
checkpoint:

```text
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_sector_cca
```

CCA maps are fit on strict train pairs and evaluated on strict held-out pairs.
The tested representations are encoder sector features, real/imaginary
complex sector statevectors at initialization and all four QNN layers, and
per-sector probability features plus sector mass. The main statistic is
held-out mean top-10 canonical correlation for `T-N`, `T-I`, and `N-I`.
The text-hub score is:

```text
0.5 * (CCA(T,N) + CCA(T,I)) - CCA(N,I)
```

A balanced shared modality manifold would have all three pairwise CCA values
high and similar. A text binding result would have `T-N` and `T-I` high enough
to exceed `N-I`, with the text-side CCA subspaces overlapping. A text-only
classifier would have high text probes but weak cross-sector CCA.

Final-layer held-out mean top-10 CCA:

| representation | T-N | T-I | N-I | text-hub score |
| --- | ---: | ---: | ---: | ---: |
| encoder `input` | 0.637456 | 0.913776 | 0.656514 | 0.119102 |
| complex_state `layer_4` | 0.024516 | 0.948381 | 0.068821 | 0.417628 |
| prob_state `layer_4` | 0.042392 | 0.774359 | 0.015618 | 0.392757 |

The important transition is from input to final state. At the encoder input,
all modalities still share substantial deterministic information, with
`T-I` strongest but `T-N` and `N-I` also moderate. By layer 4, the learned
state has collapsed into a very strong text-image alignment while number is
almost CCA-orthogonal to both text and image in the raw final state. This is
not a symmetric three-way common subspace.

The strongest held-out text-hub rows are all late QNN states:

| representation | layer | T-N | T-I | N-I | text-hub score |
| --- | --- | ---: | ---: | ---: | ---: |
| complex_state | layer 4 | 0.024516 | 0.948381 | 0.068821 | 0.417628 |
| complex_state | layer 3 | 0.059773 | 0.963170 | 0.097112 | 0.414360 |
| prob_state | layer 4 | 0.042392 | 0.774359 | 0.015618 | 0.392757 |
| prob_state | layer 3 | 0.038529 | 0.794975 | 0.027134 | 0.389618 |
| prob_state | layer 2 | 0.032290 | 0.841713 | 0.056187 | 0.380815 |

Single-sector sum probes agree with the causal sector masks: text has the
strongest standalone residue information, image is secondary, and number is
near chance when isolated at the final learned state.

| representation | layer | T | N | I |
| --- | --- | ---: | ---: | ---: |
| encoder | input | 0.058761 | 0.002733 | 0.008047 |
| complex_state | layer 4 | 0.274218 | 0.010629 | 0.188278 |
| prob_state | layer 4 | 0.362891 | 0.025964 | 0.145005 |

Anchor-subspace overlap gives the caveat. At the input, text-side CCA
subspaces for `T-N` and `T-I` overlap strongly (`0.612084` mean cosine at
rank 10). By layer 4 this text-anchor overlap falls to `0.210250` in complex
state and `0.218625` in probability state. Number-anchor overlap is higher
than text-anchor overlap at layer 4 (`0.389030` complex, `0.387229`
probability), but the corresponding pairwise CCA with number is weak. This
means the late state is not simply a clean text hub spanning number and image.

Interpretation: the CCA supports an asymmetric binding story. Text is the
dominant usable sector, and the learned final state contains a very strong
text-image shared subspace. Number still contributes causally in the full
model, especially when paired with text, but its final raw subspace is not
linearly CCA-aligned with the text-image code. The result therefore refines
the earlier statement: the model did not learn one balanced sector-common
manifold; it learned a text-dominant cyclic computation with a tight
text-image alignment and a weaker, less linearly aligned number contribution.

The most surprising derived facts are:

| observation | numbers | interpretation |
| --- | --- | --- |
| number CCA collapses while accuracy rises | complex `T-N`: `0.631 -> 0.025`; complex `N-I`: `0.671 -> 0.069`; held-out layer lens: `0.011 -> 0.926` | the model becomes more correct while becoming less linearly number-aligned |
| text-image CCA is preserved | complex `T-I`: `0.924 -> 0.948`, peaking at `0.997` after layer 1 | the circuit selects text-image as the stable shared geometry |
| number is synergistic rather than standalone | `N` only `0.031886`; `T` only `0.700577`; `T+N` `0.737625`; all `0.912542` | number helps the full computation but is not a clean solo sector code |
| image is anchored by text | `I` only `0.011388`; `T+I` `0.806408`; final image-sector probe `0.188278` complex | image becomes useful when paired with text |
| sector mixer off-diagonal power is almost all text-image | layer-1 offdiag `0.183023`, of which `T-I/I-T = 0.180038` | the learned cross-sector interaction is highly selective |
| CCA strength and answer causality differ | final prob `T-I` CCA `0.774359`, but CCA-pair probe `0.136350`; final prob `T-N` CCA `0.042392`, but CCA-pair probe `0.271181` | large CCA can reflect shared expression geometry, while weaker subspaces can carry more residue-relevant signal |

This is why the result is scientifically interesting rather than merely
"text wins". A naive translation-to-number story predicts number as the
binding coordinate. The artifact instead appears to use text as the dominant
symbolic carrier, locks image to text, and uses number as a weaker nonlinear
or correction-like contributor.

Saved CCA audit artifacts:

```text
pairwise_cca.csv
text_binding_scores.csv
anchor_subspace_overlap.csv
sector_sum_probe.csv
TRIMODAL_QNN_SECTOR_CCA_REPORT.md
figures/encoder_pairwise_cca.png
figures/complex_state_pairwise_cca.png
figures/prob_state_pairwise_cca.png
figures/text_hub_score_heatmap.png
figures/*_anchor_overlap.png
figures/*_sector_sum_probe.png
```

### Strict Dirac-Mean Seed Robustness

We then ran the strict `layerwise_dirac_mean` setup for four additional seeds,
`9302-9305`, with the same data split, architecture, loss weights, and
checkpoint cadence as seed `9301`. Each new seed was capped at exactly `2000`
training steps, then analyzed with the same checkpoint, causal, and sector CCA
suite. This turns the one-seed Dirac-mean artifact into a five-seed robustness
result.

Execution command:

```powershell
python -m trimodal_qnn_codex.run_seed_pipeline --seeds 9302 9303 9304 9305 --training-steps 2000 --force-config --resume-existing
```

All four new runs completed with exit code `0`, and all three analysis passes
completed with exit code `0` for every seed. Pipeline wall time was
`8602.43` seconds.

The comparison population is now:

| seed | status | checkpoint | held-out acc | cross-ablate acc | train acc |
| ---: | --- | ---: | ---: | ---: | ---: |
| 9301 | original strict Dirac-mean run | 2000 | 0.912542 | 0.794716 | 1.000000 |
| 9302 | new seed sweep | 2000 | 0.941543 | 0.781203 | 1.000000 |
| 9303 | new seed sweep | 2000 | 0.938961 | 0.820984 | 0.999646 |
| 9304 | new seed sweep | 2000 | 0.910112 | 0.730489 | 1.000000 |
| 9305 | new seed sweep | 2000 | 0.914212 | 0.682964 | 1.000000 |

Aggregate held-out accuracy:

| population | mean | sd | min | max |
| --- | ---: | ---: | ---: | ---: |
| new seeds `9302-9305` | 0.926207 | 0.014149 | 0.910112 | 0.941543 |
| all seeds `9301-9305` | 0.923474 | 0.013785 | 0.910112 | 0.941543 |

The strict Dirac-mean result is therefore seed-stable. The original seed was
not a lucky run.

Layerwise logit-lens accuracy shows the same internal formation pattern across
seeds: answer information appears by layer 1, becomes strong by layer 2, and is
sharpened by layers 3 and 4.

| seed | initial | layer 1 | layer 2 | layer 3 | layer 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9301 | 0.010932 | 0.488005 | 0.749013 | 0.898725 | 0.925600 |
| 9302 | 0.012754 | 0.408442 | 0.800182 | 0.901761 | 0.940480 |
| 9303 | 0.009718 | 0.496963 | 0.815366 | 0.910416 | 0.937899 |
| 9304 | 0.008047 | 0.521257 | 0.760705 | 0.863954 | 0.907987 |
| 9305 | 0.009262 | 0.457182 | 0.690859 | 0.866535 | 0.917704 |

Fourier cutoff diagnostics also replicate. Low/mid frequencies carry most of
the answer, while the full `k<=21` basis is needed for final finite-residue
sharpening.

| seed | k1 | k3 | k5 | k8 | k13 | k21 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9301 | 0.072730 | 0.270878 | 0.463559 | 0.595809 | 0.837990 | 0.912542 |
| 9302 | 0.070149 | 0.281051 | 0.477073 | 0.721379 | 0.876860 | 0.941543 |
| 9303 | 0.076374 | 0.269207 | 0.515943 | 0.677650 | 0.854236 | 0.938961 |
| 9304 | 0.079411 | 0.241269 | 0.431825 | 0.613726 | 0.843759 | 0.910112 |
| 9305 | 0.075615 | 0.264804 | 0.494078 | 0.656089 | 0.861828 | 0.914212 |

All-five mean `k<=13` accuracy is `0.854935`. Frequency-band causal patching
agrees: restoring `k=1..13` recovers most clean behavior, while high-only
`k=14..21` does not.

| seed | restore k1-13 | ablate k1-13 drop | restore k14-21 |
| ---: | ---: | ---: | ---: |
| 9301 | 0.786403 | 0.901458 | 0.011084 |
| 9302 | 0.813658 | 0.926131 | 0.015411 |
| 9303 | 0.837420 | 0.926359 | 0.012602 |
| 9304 | 0.756263 | 0.890753 | 0.019359 |
| 9305 | 0.790882 | 0.894321 | 0.019891 |

All-five mean `k=1..13` restore accuracy is `0.796925`; all-five mean
`k=1..13` ablation drop is `0.907804`.

Sector masks preserve the asymmetric text-anchored story. Text alone is
substantially above chance, number/image alone are near chance, and adding
number or image to text improves performance.

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

The CCA result is the most decisive robustness finding. The final complex
state repeatedly shows high `T-I` CCA and weak `T-N` / `N-I` CCA.

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

Final probability-state CCA gives the same conclusion:

| pair | mean top-10 CCA |
| --- | ---: |
| `T-I` | 0.822626 |
| `T-N` | 0.037814 |
| `N-I` | 0.016812 |

Single-sector probes remain modest, which prevents a simplistic "answer stored
linearly in text" interpretation. Final probability-state probe means are:

| sector | mean held-out probe acc |
| --- | ---: |
| `T` | 0.326480 |
| `N` | 0.027209 |
| `I` | 0.154449 |

The correct interpretation is therefore not text-only storage. It is a
text-anchored circuit route: text is the dominant binding/readout sector, image
aligns strongly to text, and number contributes causally without becoming a
clean final linear anchor.

The five-seed result updates the hypothesis table:

| hypothesis | current status |
| --- | --- |
| no true shared cyclic rule | strongly disfavored |
| three unrelated modality-specific rules | disfavored by stable `T-I` CCA and cross-sector causal effects |
| translation-to-number strategy | not supported in a simple final-linear-number-coordinate sense |
| one balanced sector-common cyclic manifold | only partly supported; cyclic rule is shared, sector geometry is not balanced |
| text-anchored shared cyclic/Fourier rule | best current description |

Dedicated seed-sweep report:

```text
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/TRIMODAL_QNN_SEED_SWEEP_REPORT.md
```

## Failure Modes

### Nonfinite Gradients

Three of the four full phase-1 queue runs ended with nonfinite gradients:

| run | failure step | saved failure checkpoint |
| --- | ---: | --- |
| v1 ordered-route | 21331 | `checkpoint_nonfinite_21331.pt` |
| v2 three-sector | 12107 | `checkpoint_nonfinite_12107.pt` |
| v2 ordered-route | 17464 | `checkpoint_nonfinite_17464.pt` |

Likely sources:

- matrix exponential sector mixers can create sharp gradients as Hermitian
  parameters grow;
- the Fourier head scale and residual readout can amplify logits;
- hard-neighbor margin pressure can become large near decision boundaries;
- 7-qubit runs have larger measured feature spaces and more parameters;
- active-route ordered initialization creates sparse sector states, which may
  make sector mixing harder to optimize smoothly.

Mitigations already implemented:

- gradient clipping;
- global parameter clipping;
- sector-mixer clipping;
- Fourier head scale clamp;
- nonfinite loss and gradient detection;
- durable nonfinite checkpoint saves;
- resume support with optional optimizer reset.

Recommended next mitigation:

- resume from the best checkpoint with lower LR;
- lower or schedule hard-neighbor margin weight;
- temporarily disable `fourier_residual`;
- add a sector-mixer spectral-norm/angle penalty;
- add exact layerwise diagnostics before resuming, so we know whether the
  failing point is useful mechanistically.

### Route Fragility

The ordered-route formulation has much lower accuracy than the three-sector
formulation. The most plausible reason is that active-route initialization
asks the model to infer a common route grammar from one populated sector per
example. In contrast, `three_sector` always presents all three modalities
coherently, giving the sector mixer a denser alignment signal.

The `operand_query` result changes this failure analysis. Route conditioning
itself is not the blocker: the 20k operand-query model reaches held-out
`0.794497` across nine ordered routes. The blocker was the older route
architecture. A one-hot active-route sector plus full-sector readout made it
too easy to learn route-local partial rules and too hard to share an
answer-carrier. Decomposing the route into two operand sectors plus one
answer-query sector gives the dense mixer a direct compositional problem that
matches the classical transformer's successful answer-slot structure.

Immediate fixes to test:

- continue `operand_query` beyond 20k to see whether the query carrier closes
  the gap to the strict three-sector Dirac-mean family;
- run an operand-query seed sweep before adding new regularizers;
- compare dense sector mixing against diagonal/no-cross and constrained
  pairwise mixers in the same operand-query architecture;
- train a commutative augmentation/control where `(a,b)` and `(b,a)` routes
  share labels explicitly, to test whether commutativity emerges or needs
  pressure;
- keep the old `ordered_initialization: all_route_superposition` as a control,
  but no longer treat it as the main route-composition path.

### Decoder/Head Leakage

The Fourier-delta head is small, but `fourier_residual: true` gives the model a
linear residual path from measured features to logits. That path should be
treated as a training aid, not as a final mechanistic proof.

Immediate controls:

- rerun the best configs with `fourier_residual: false`;
- freeze the QNN and train only a strict Fourier-delta head;
- evaluate with the residual head ablated after training;
- fit a post-hoc Dirac/Fourier readout to layerwise QNN states.

The operand-query run already satisfies the strictest version of this control:
`fourier_residual: false`, `head_type: dirac_delta`, and `readout_mode:
query_sector`. Its full accuracy equals its Fourier-only accuracy at the
20k checkpoint, so this result cannot be explained by a hidden residual
linear classifier over all sectors.

## Immediate Cases To Extend

Priority 1: replicate and extend the operand-query result.

```text
operand_query: continue from checkpoint_20000.pt toward 40k
operand_query seeds: run at least four additional seeds to 20k
operand_query controls: dense mixer vs no-cross/diagonal/pairwise mixers
```

The seed/control split is important. A higher 40k checkpoint tests whether the
current model is still improving; the seed sweep tests whether the result is
robust; the mixer controls test the causal claim that dense cross-modal sector
interaction is doing the work.

Priority 2: stabilize and continue older best checkpoints as controls.

```text
v2 three-sector: resume from checkpoint_12000.pt at lower LR
v2 ordered-route: resume from checkpoint_16000.pt at lower LR
v1 ordered-route: resume from checkpoint_21000.pt only as a control
```

Priority 3: run mechanistic analysis before changing the best checkpoints.

- all-pair exact evaluation, not only held-out telemetry;
- exact Fourier energy on full `p x p` grids;
- layerwise state extraction for every checkpoint;
- sector-mass dynamics;
- per-sector and route-local Fourier coefficient diagnostics;
- cross-sector ablation by layer and sector pair;
- state patching from good three-sector checkpoints into ordered-route
  checkpoints;
- frozen readout tests on measured QNN features and raw complex states.

Priority 4: remove head confounds.

- `fourier_residual: false`;
- fixed Fourier class basis with no residual;
- post-hoc frozen Dirac/Fourier heads;
- compare to a matched classical MLP over the same rendered/text/number
  encoders.

Priority 5: test route-composition variants.

- operand-query dense mixer;
- operand-query no-cross/diagonal mixer;
- operand-query pairwise constrained mixer;
- ordered-route active initialization as a historical control;
- ordered-route all-route superposition as a historical control;
- ordered-route with learned complex sector amplitudes as a historical control;
- route leave-one-out QNN training mirroring the transformer phase-5/6 tests;
- commutative vs ordered operands as separate configs.

Priority 6: seed/modulus robustness.

- completed strict Dirac-mean seed sweep for four additional seeds:
  `9302`, `9303`, `9304`, and `9305`;
- all four seeds were capped at step `2000`, reached held-out accuracy
  `0.910112-0.941543`, and completed checkpoint, causal, and CCA
  mech-interp analyses;
- the text-image CCA artifact is seed-stable: final complex-state `T-I`
  mean top-10 CCA is `0.963401`, while `T-N` is `0.028621` and `N-I` is
  `0.071689`;
- the next robustness controls are now sector-dropout, number-anchor
  alignment, operand-query seed sweep, moduli 31/97/127, train fractions
  0.30/0.50/0.70, and the ordered-route/all-route-superposition Dirac-mean
  variants as controls.

## Reproduction Commands

Smoke tests:

```powershell
cd modular_addition_mech_interp
python -m trimodal_qnn_codex.train --config trimodal_qnn_codex/configs/smoke_three_sector.yaml
python -m trimodal_qnn_codex.train --config trimodal_qnn_codex/configs/smoke_ordered_route.yaml
python -m trimodal_qnn_codex.smoke
```

Full phase-1 queue:

```powershell
cd modular_addition_mech_interp
python -m trimodal_qnn_codex.run_lessons_queue --continue-on-error --force --out-dir trimodal_qnn_codex/outputs/phase1_v1_v2_queue --configs trimodal_qnn_codex/configs/phase1_three_sector_mod97_lessons.yaml trimodal_qnn_codex/configs/phase1_ordered_route_mod97_lessons.yaml trimodal_qnn_codex/configs/phase1_three_sector_mod97_lessons_v2.yaml trimodal_qnn_codex/configs/phase1_ordered_route_mod97_lessons_v2.yaml
```

Checkpoint mechanistic analysis:

```powershell
cd modular_addition_mech_interp
python -m trimodal_qnn_codex.analyze_checkpoints --out-dir trimodal_qnn_codex/analysis/phase1_checkpoint_mech --batch-size 1024 --device auto
```

Unit tests:

```powershell
cd modular_addition_mech_interp
python -m pytest tests/test_trimodal_qnn_codex.py
```

Strict Dirac-mean follow-up and causal analysis:

```powershell
cd modular_addition_mech_interp
python -m trimodal_qnn_codex.train --config trimodal_qnn_codex/configs/phase1_three_sector_mod97_dirac_mean.yaml
python -m trimodal_qnn_codex.analyze_checkpoints --run-dir trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean --checkpoint-step 2000 --out-dir trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000 --batch-size 1024 --device auto --cutoffs 1,2,3,5,8,13,21
python -m trimodal_qnn_codex.analyze_dirac_causal --run-dir trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean --checkpoint-step 2000 --out-dir trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_causal --split heldout --batch-size 512 --device auto --deltas 1,2,5,13
python -m trimodal_qnn_codex.analyze_sector_cca --run-dir trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean --checkpoint-step 2000 --out-dir trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_sector_cca --batch-size 1024 --device auto --n-components 32 --cca-reg 1e-3 --probe-reg 1e-2
```

Strict Dirac-mean four-seed robustness pipeline:

```powershell
cd modular_addition_mech_interp
python -m trimodal_qnn_codex.run_seed_pipeline --seeds 9302 9303 9304 9305 --training-steps 2000 --force-config --resume-existing
```

This writes deterministic per-seed configs under
`trimodal_qnn_codex/configs/seed_sweeps`, trains each seed under
`trimodal_qnn_codex/outputs/phase1_three_sector_dirac_mean_seed_sweep`, selects
the best regular finite checkpoint by held-out accuracy, and writes the three
analysis directories per seed under
`trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep`.
The completed run wrote `seed_pipeline_summary.csv`,
`seed_pipeline_summary.json`, and `TRIMODAL_QNN_SEED_SWEEP_REPORT.md` in the
seed-sweep analysis directory. The dry-run used to validate config generation
before launch was:

```powershell
python -m trimodal_qnn_codex.run_seed_pipeline --dry-run --no-train --no-analyze --seeds 9302 9303 9304 9305
```

Operand-query bitter-lesson route model:

```powershell
cd modular_addition_mech_interp
python -m trimodal_qnn_codex.train --config trimodal_qnn_codex/configs/phase1_operand_query_mod97_bitter_lesson_5k.yaml
python -m trimodal_qnn_codex.train --config trimodal_qnn_codex/configs/phase1_operand_query_mod97_bitter_lesson_20k_from_5k.yaml
python -m trimodal_qnn_codex.analyze_checkpoints --run-dir trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_20k_from_5k --checkpoint-step 20000 --out-dir trimodal_qnn_codex/analysis/phase1_operand_query_mod97_bitter_lesson_20k_checkpoint --batch-size 1024 --device auto --cutoffs 1,2,3,5,8,13,21
python -m pytest tests/test_trimodal_qnn_codex.py -q
```

## Audit Artifacts

Core code:

```text
trimodal_qnn_codex/__init__.py
trimodal_qnn_codex/amplitudes.py
trimodal_qnn_codex/config.py
trimodal_qnn_codex/data.py
trimodal_qnn_codex/diagnostics.py
trimodal_qnn_codex/heads.py
trimodal_qnn_codex/models.py
trimodal_qnn_codex/quantum.py
trimodal_qnn_codex/render.py
trimodal_qnn_codex/analyze_checkpoints.py
trimodal_qnn_codex/analyze_dirac_causal.py
trimodal_qnn_codex/analyze_sector_cca.py
trimodal_qnn_codex/run_lessons_queue.py
trimodal_qnn_codex/run_seed_pipeline.py
trimodal_qnn_codex/smoke.py
trimodal_qnn_codex/train.py
```

Theory sources:

```text
../qnn_derivations_numeric_trimodal.pdf
analysis/qnn_derivations_numeric_trimodal_extracted.txt
```

Configs:

```text
trimodal_qnn_codex/configs/smoke_three_sector.yaml
trimodal_qnn_codex/configs/smoke_three_sector_dirac_mean.yaml
trimodal_qnn_codex/configs/smoke_ordered_route.yaml
trimodal_qnn_codex/configs/phase1_three_sector_mod97_lessons.yaml
trimodal_qnn_codex/configs/phase1_ordered_route_mod97_lessons.yaml
trimodal_qnn_codex/configs/phase1_three_sector_mod97_lessons_v2.yaml
trimodal_qnn_codex/configs/phase1_ordered_route_mod97_lessons_v2.yaml
trimodal_qnn_codex/configs/phase1_three_sector_mod97_dirac_mean.yaml
trimodal_qnn_codex/configs/phase1_three_sector_mod97_dirac_mean_stable_from_2000.yaml
trimodal_qnn_codex/configs/smoke_operand_query_bitter_lesson.yaml
trimodal_qnn_codex/configs/phase1_operand_query_mod97_bitter_lesson_5k.yaml
trimodal_qnn_codex/configs/phase1_operand_query_mod97_bitter_lesson_20k_from_5k.yaml
trimodal_qnn_codex/configs/seed_sweeps/phase1_three_sector_mod97_dirac_mean_seed9302.yaml
trimodal_qnn_codex/configs/seed_sweeps/phase1_three_sector_mod97_dirac_mean_seed9303.yaml
trimodal_qnn_codex/configs/seed_sweeps/phase1_three_sector_mod97_dirac_mean_seed9304.yaml
trimodal_qnn_codex/configs/seed_sweeps/phase1_three_sector_mod97_dirac_mean_seed9305.yaml
```

Outputs:

```text
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons/config.yaml
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons/metrics.jsonl
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons/summary.json
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons/checkpoint_*.pt
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons/metrics.jsonl
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons/checkpoint_*.pt
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons/checkpoint_nonfinite_21331.pt
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons_v2/metrics.jsonl
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons_v2/checkpoint_*.pt
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_lessons_v2/checkpoint_nonfinite_12107.pt
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons_v2/metrics.jsonl
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons_v2/checkpoint_*.pt
trimodal_qnn_codex/outputs/phase1_ordered_route_mod97_lessons_v2/checkpoint_nonfinite_17464.pt
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean/metrics.jsonl
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean/checkpoint_1000.pt
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean/checkpoint_2000.pt
trimodal_qnn_codex/outputs/phase1_three_sector_mod97_dirac_mean/checkpoint_nonfinite_2292.pt
trimodal_qnn_codex/outputs/phase1_three_sector_dirac_mean_seed_sweep/seed_9302/
trimodal_qnn_codex/outputs/phase1_three_sector_dirac_mean_seed_sweep/seed_9303/
trimodal_qnn_codex/outputs/phase1_three_sector_dirac_mean_seed_sweep/seed_9304/
trimodal_qnn_codex/outputs/phase1_three_sector_dirac_mean_seed_sweep/seed_9305/
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_5k/config.yaml
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_5k/metrics.jsonl
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_5k/summary.json
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_5k/checkpoint_*.pt
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_20k_from_5k/config.yaml
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_20k_from_5k/metrics.jsonl
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_20k_from_5k/summary.json
trimodal_qnn_codex/outputs/phase1_operand_query_mod97_bitter_lesson_20k_from_5k/checkpoint_*.pt
trimodal_qnn_codex/outputs/phase1_v1_v2_queue/status.jsonl
trimodal_qnn_codex/outputs/phase1_v1_v2_queue/queue_summary.json
trimodal_qnn_codex/outputs/phase1_v1_v2_queue/logs/
```

Checkpoint mechanistic analysis:

```text
trimodal_qnn_codex/analysis/phase1_checkpoint_mech/manifest.json
trimodal_qnn_codex/analysis/phase1_checkpoint_mech/checkpoint_summary.csv
trimodal_qnn_codex/analysis/phase1_checkpoint_mech/layer_logit_lens.csv
trimodal_qnn_codex/analysis/phase1_checkpoint_mech/route_summary.csv
trimodal_qnn_codex/analysis/phase1_checkpoint_mech/frequency_cutoffs.csv
trimodal_qnn_codex/analysis/phase1_checkpoint_mech/sector_masses.csv
trimodal_qnn_codex/analysis/phase1_checkpoint_mech/failure_events.csv
trimodal_qnn_codex/analysis/phase1_checkpoint_mech/TRIMODAL_QNN_CHECKPOINT_MECH_REPORT.md
trimodal_qnn_codex/analysis/phase1_checkpoint_mech/figures/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_causal/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_step2000_sector_cca/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_pipeline_summary.json
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_pipeline_summary.csv
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/TRIMODAL_QNN_SEED_SWEEP_REPORT.md
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_*_pipeline_summary.json
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_*_step_*/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_*_step_*_causal/
trimodal_qnn_codex/analysis/phase1_three_sector_dirac_mean_seed_sweep/seed_*_step_*_sector_cca/
trimodal_qnn_codex/analysis/phase1_operand_query_mod97_bitter_lesson_5k_checkpoint/
trimodal_qnn_codex/analysis/phase1_operand_query_mod97_bitter_lesson_20k_checkpoint/
trimodal_qnn_codex/BITTER_LESSON_TRANSLATION.md
```

## Bottom Line

The trimodal QNN codebase is now a real experimental scaffold. The original
residual-head three-sector runs learned meaningful but imperfect cyclic
structure. The strict layerwise Dirac-mean three-sector follow-up is much
stronger: across five seeds it reaches mean held-out exact `0.923474` at step
`2000` without a residual class head, and causal patching shows that the
learned answer depends on a distributed low/mid-frequency cyclic code. It
still does not prove balanced modality use. The five-seed sector tomography
and CCA show a seed-stable text-dominant computation, a strong text-image
shared subspace, and number support that is causal but not cleanly linearly
aligned in the final raw sector state.

The new operand-query result changes the next decisive step. The text-hub
interventions remain important for the strict three-sector family, but route
composition now has a concrete working path: two operand sectors, one
answer-query sector, dense cross-sector interaction, and a strict query-sector
Dirac/Fourier head. The checkpoint, causal, seed, and operand-query analyses
say the highest-value follow-ups separate:

```text
modality translation
sector/route mixing
cyclic Fourier phase formation
finite-residue Dirac sharpening
```

The highest-priority concrete variants are now: continue operand-query to 40k,
run an operand-query seed sweep, run dense-mixer ablations, and then test
sector-dropout/number-anchor controls in the strict three-sector family. The
decisive scientific question has split in two:

```text
Can the seed-stable 0.92 three-sector solution be made modality-balanced?
Can the 0.79 operand-query route solution become seed-stable and fully grokked?
```

The strongest current interpretation is that the QNN can learn a real cyclic
rule, but the representational carrier depends heavily on the problem
factorization. Whole-expression coherent sectors favor a text-anchored cyclic
rule; operand-query sectors favor a classical-style answer-query cyclic
carrier. Neither result should be described as a literal learned number.
