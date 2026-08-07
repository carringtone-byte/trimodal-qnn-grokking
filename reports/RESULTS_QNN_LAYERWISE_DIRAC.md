# Layerwise Dirac Fourier QNN Tests

This report records the main QNN intervention in the modular-addition branch:
placing a finite Dirac/Fourier modular-addition readout after every
data-reuploading circuit layer.

The central question is not merely whether a QNN can be made more accurate by
adding a better final head. The stronger question is:

> Can the circuit be trained so that modular-addition evidence appears
> progressively across depth, with each intermediate state approximating a
> finite cyclic delta over `(a+b-c) mod p`?

Earlier QNN runs showed that a final Fourier-delta readout can rescue a weak
low-frequency cyclic scaffold, but those runs left an ambiguity: perhaps the
entire arithmetic rule only appears at the final readout, with the circuit
acting as a generic feature map. The layerwise Dirac/Fourier intervention tests
the opposite hypothesis: every reuploading layer should be pressured to expose
the group law.

The first three runs used `prob_head` probability features and trained from
scratch. Follow-up sweeps then tested a fixed uniform layerwise mean and a
same-sum invariance objective.

## Motivation

Data-reuploading QNNs build trigonometric feature maps progressively. After
layer `l`, the state:

```latex
|\psi_\ell(a,b)\rangle =
U_\ell(a,b;\Theta_\ell)\cdots U_1(a,b;\Theta_1)|0\rangle^{\otimes n}
```

induces a measured feature vector:

```latex
z_\ell(a,b)=M(|\psi_\ell(a,b)\rangle).
```

Because each reuploading layer multiplies and mixes trigonometric terms, later
layers can express richer frequency interactions than earlier layers. A final
head alone only asks:

```latex
H_L(z_L(a,b)) \rightarrow (a+b)\bmod p.
```

That objective gives no direct constraint on whether intermediate states
represent the group law. It also allows the final layer/head pair to carry a
fragile, highly co-adapted boundary rule.

The layerwise intervention changes the training problem. For each intermediate
state, a finite modular delta head asks:

```latex
H_\ell(z_\ell(a,b)) \rightarrow (a+b)\bmod p.
```

Equivalently, for each layer `l`, the model is asked to approximate the cyclic
residual kernel:

```latex
\ell^{(\ell)}_c(a,b)
\approx
K_\ell((a+b-c)\bmod p).
```

The exact finite-group target has Fourier expansion:

```latex
\delta_p(a+b-c)
= {1\over p}\sum_{k=0}^{p-1}
\exp\left({2\pi i k(a+b-c)\over p}\right).
```

So a layerwise Dirac/Fourier head is not an arbitrary auxiliary classifier. It
is a local finite-group readout that asks whether the current quantum feature
state already contains enough addition-diagonal structure to support the
correct residue.

This has three motivations:

1. **Optimization:** every layer receives a direct residue-shaped gradient,
   rather than relying only on a final classifier loss.
2. **Mechanistic localization:** each layer can be evaluated as an arithmetic
   state, producing a causal trajectory from broad cyclic estimate to sharper
   residue selection.
3. **Distributed evidence:** if different layers capture different frequency
   bands or boundary cases, an ensemble of layerwise logits may outperform the
   final layer alone.

The third point is the motivation for averaging. It became central after the
exhaustive post-hoc analysis found that the auxiliary model's uniform
layerwise logit average reached `0.975558`, substantially above its deployed
final head at `0.951571`.

## Averaging Principle

Let the layerwise logits be:

```latex
\ell^{(1)},\ell^{(2)},\ldots,\ell^{(L)}\in\mathbb{R}^p.
```

The fixed layerwise mean deploys:

```latex
\ell_{\text{mean}}
= {1\over L}\sum_{\ell=1}^{L}\ell^{(\ell)}.
```

Prediction is then:

```latex
\hat y
= \arg\max_c \ell_{\text{mean},c}.
```

This is logit averaging, not probability averaging. If
`p_l(c|a,b)=softmax(l^{(l)})_c`, then:

```latex
softmax(\ell_{\text{mean}})_c
\propto
\left[
\prod_{\ell=1}^{L} \exp(\ell^{(\ell)}_c)
\right]^{1/L}.
```

So the fixed mean is a log-opinion pool: classes supported by multiple layers
are amplified, while layer-specific alias errors can cancel. This is exactly
the desired behavior if each layer contains a noisy finite-delta approximation
with partially independent boundary errors.

The margin view makes the same point. Define the layerwise correct-vs-class
margin:

```latex
m^{(\ell)}_c
= \ell^{(\ell)}_y-\ell^{(\ell)}_c.
```

The averaged margin is:

```latex
m^{\text{mean}}_c
= {1\over L}\sum_{\ell=1}^{L}m^{(\ell)}_c.
```

The mean fixes an error whenever the average margin is positive even if the
final-layer margin is negative:

```latex
m^{(L)}_c < 0
\quad\text{but}\quad
{1\over L}\sum_{\ell=1}^{L}m^{(\ell)}_c > 0.
```

This explains the auxiliary result: earlier heads were not perfect, but their
errors were not identical to the final head's errors. The uniform average used
that disagreement constructively.

The learned residual mixture is:

```latex
\alpha_\ell =
{e^{w_\ell}\over \sum_j e^{w_j}},
\qquad
\ell_{\text{residual}}
= \sum_{\ell=1}^{L}\alpha_\ell\ell^{(\ell)}.
```

This is more flexible, but it can collapse toward the final layer. The best
seed-0 learned-residual run placed about 80% of the weight on layer 3:

```text
[0.063517, 0.060813, 0.074647, 0.801023]
```

The fixed mean is therefore a stronger intervention: it forces the deployed
answer to use all intermediate Dirac/Fourier heads from the start.

The branch result supports this interpretation. In the seed/modulus sweep, the
fixed mean was the best QNN family at `p=97` and `p=127`, with held-out means
`0.975305` and `0.983438`. The same-sum follow-up improved the seed-0 fixed
mean further to `0.980871`.

The train-ratio and split-seed sweep also supports the fixed-mean intervention.
With model seed `0` and split seeds `0,1,2`, the fixed mean reached `0.990719`
at `p=97` train fraction `0.50` and `0.994332` at train fraction `0.70`.
At `p=31`, where the `0.30` split was weak, increasing train coverage raised
fixed-mean held-out accuracy to `0.651421` at train fraction `0.50` and
`0.880046` at train fraction `0.70`. The aggregate is saved in
`analysis/qnn_split_ratio_sweep/SWEEP_AGGREGATE.md`.

The remaining validation grid for this claim is implemented in
`QNN_REMAINING_TESTS_PLAN.md` and `analysis/qnn_remaining_tests`. It expands
the fixed-mean/residual comparison to the full model-seed, split-seed, and
train-ratio Cartesian product and adds the missing classical controls.

## Variants

| run | readout type | intervention | intended test |
| --- | --- | --- | --- |
| auxiliary | `layerwise_dirac_aux` | final Dirac/Fourier head plus per-layer auxiliary Dirac losses | Does residue-shaped deep supervision improve optimization without changing the circuit state? |
| adapter | `layerwise_dirac_adapter` | per-layer Dirac/Fourier coefficients projected back into next-layer input features | Does feeding residue-aligned coefficients into later quantum layers help the circuit compose the rule? |
| residual | `layerwise_dirac_residual` | final logits are a learned weighted sum of all per-layer Dirac/Fourier logits | Can the answer emerge progressively across layers rather than only at the final layer? |
| fixed mean | `layerwise_dirac_mean` | final logits are the uniform average of all per-layer Dirac/Fourier logits from the start | Can the post-hoc layer averaging benefit be made part of training? |

## Layerwise Architecture

Let `z_l(a,b)` be the measured probability feature vector after data-reuploading
layer `l`. A layerwise Dirac/Fourier head predicts a residue score vector:

```latex
\ell^{(\ell)}(a,b)=H_\ell(z_\ell(a,b)).
```

Each head uses a finite Fourier-delta parameterization. It predicts per-
frequency sum features:

```latex
(\hat{u}_{\ell k},\hat{v}_{\ell k})
\approx
\left[
\cos\left({2\pi k(a+b)\over p}\right),
\sin\left({2\pi k(a+b)\over p}\right)
\right],
```

then scores a candidate class `c` as:

```latex
\ell^{(\ell)}_c
= \beta^{(\ell)}_c
+ \gamma_\ell
\sum_{k=1}^{K}\rho_{\ell k}
\left[
\hat{u}_{\ell k}\cos\left({2\pi kc\over p}\right)
+
\hat{v}_{\ell k}\sin\left({2\pi kc\over p}\right)
\right].
```

If the learned phase equals the sum phase, the score is a finite kernel over
`(a+b-c) mod p`:

```latex
\ell^{(\ell)}_c \approx
\sum_{k=1}^{K}\rho_{\ell k}
\cos\left({2\pi k((a+b)-c)\over p}\right).
```

The auxiliary architecture trains all heads but deploys the final one:

```latex
\mathcal{L}_{\text{aux}}
= \operatorname{CE}(\ell^{(L)},y)
+\lambda_{\text{layer}}{1\over L}
\sum_{\ell=1}^{L}\operatorname{CE}(\ell^{(\ell)},y).
```

The learned residual architecture deploys a softmax-weighted average:

```latex
\alpha_\ell =
{e^{w_\ell}\over\sum_j e^{w_j}},
\qquad
\ell_{\text{residual}} =
\sum_{\ell=1}^{L}\alpha_\ell \ell^{(\ell)}.
```

The fixed mean architecture removes learned weighting entirely:

```latex
\ell_{\text{mean}} =
{1\over L}\sum_{\ell=1}^{L}\ell^{(\ell)}.
```

This is the baked-in version of the post-hoc discovery that uniform averaging
of the auxiliary model's layerwise logits beats its deployed final head. The
scientific question is whether the useful arithmetic state is distributed
across layer depths and whether training can force that distributed signal to
be decisive.

The adapter architecture is different. It uses the layerwise Fourier/Dirac
coefficients as feedback:

```latex
\phi_{\ell+1}(a,b)
= \phi(a,b) + \eta_\ell A_\ell c_\ell(a,b).
```

The adapter ablation showed this path is causally used, but the architecture is
comparatively worse than the readout-only layerwise interventions.

## Results

| run | params | best step | train exact | held-out exact | wrap | no-wrap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| layerwise auxiliary | 29217 | 2250 | 0.976612 | 0.951571 | 0.930305 | 0.971835 |
| layerwise adapter | 29905 | 2500 | 0.933735 | 0.874601 | 0.818295 | 0.928254 |
| layerwise residual | 23428 | 2500 | 0.983345 | 0.968119 | 0.949284 | 0.986066 |
| fixed layerwise mean seed-0 control | 23424 | 2250 | 0.990432 | 0.973736 | 0.961419 | 0.985473 |

The residual layerwise readout is the strongest of the three scratch tests. It
does not beat the best warm-start/no-residual QNN result of `0.982997`, but it
is much stronger than the previous scratch Dirac-residual and head-decisive
runs.

The fixed uniform mean was added after the exhaustive post-hoc analysis showed
that averaging the auxiliary model's four layerwise logits reached `0.975558`,
better than the deployed final auxiliary head at `0.951571`. Training the
uniform average from the start produced `0.973736` on seed 0, and the later
seed/modulus sweep showed it is the strongest QNN family at `p=97` and
`p=127`.

## Mechanistic Summary

| diagnostic | auxiliary | adapter | residual |
| --- | ---: | ---: | ---: |
| held-out exact | 0.951571 | 0.874601 | 0.968119 |
| held-out `a+b <= 50` | 0.992505 | 0.931478 | 0.980728 |
| mod-add residual R2 | 0.646932 | 0.619808 | 0.649418 |
| feature addition-diagonal energy | 0.102239 | 0.098588 | 0.098710 |
| k=1 sum phase exact | 0.212388 | 0.142554 | 0.211477 |
| k=1 sum phase within 5 | 0.973888 | 0.923486 | 0.981479 |

## Exhaustive Mechanistic Interp

The three checkpoints were then analyzed with the dedicated layerwise analyzer:

```text
modular_addition/analyze_qnn_layerwise_dirac.py
analysis/qnn_layerwise_dirac_exhaustive
```

The exhaustive pass writes component metrics, frequency cutoffs, residual fits,
coefficient diagnostics, layer-overlap tables, failure audits, calibration, and
plots for each architecture.

Architecture-level summary:

| architecture | held-out | within one | top-2 | mod-add R2 | logit addition energy | feature addition energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auxiliary | 0.951571 | 0.998026 | 0.994687 | 0.646932 | 0.648190 | 0.102239 |
| adapter | 0.874601 | 0.994990 | 0.977835 | 0.619808 | 0.618930 | 0.098588 |
| residual | 0.968119 | 0.999696 | 0.999696 | 0.649418 | 0.649462 | 0.098710 |

Per-layer held-out exact accuracy:

| architecture | layer 0 | layer 1 | layer 2 | layer 3 | final head | uniform layer mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auxiliary | 0.587673 | 0.864582 | 0.940185 | 0.964172 | 0.951571 | 0.975558 |
| adapter | 0.412783 | 0.628966 | 0.782602 | 0.868073 | 0.874601 | 0.900258 |
| residual | 0.449370 | 0.683771 | 0.843935 | 0.964627 | 0.968119 | 0.966297 |

This changes the interpretation slightly. The auxiliary run is better than its
deployed final head if the four layerwise logits are averaged uniformly
(`0.975558` vs `0.951571`). The adapter run also improves under a uniform
layerwise mean (`0.900258` vs `0.874601`), even though it remains the weakest
architecture. The residual run is already close to optimal under its learned
weights (`0.968119`), with the final layer carrying most of the decision.

Layer-overlap diagnostics show the computational progression:

| architecture | layer 3 held-out | final-wrong cases fixed by layer 3 | final-correct cases broken by layer 3 | agreement with final |
| --- | ---: | ---: | ---: | ---: |
| auxiliary | 0.964172 | 158 | 75 | 0.963413 |
| adapter | 0.868073 | 164 | 207 | 0.940792 |
| residual | 0.964627 | 6 | 29 | 0.994535 |

The residual model is the cleanest final-layer story: layer 3 and the final
weighted mixture nearly agree. In contrast, the auxiliary model contains useful
disagreement between layers; a simple ensemble recovers many final-head errors.

Frequency cutoff behavior is consistent across all three:

| architecture | k<=5 exact | k<=5 within one | k<=8 exact | k<=13 exact | k<=21 exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| auxiliary final | 0.618339 | 0.989069 | 0.930469 | 0.945043 | 0.951571 |
| adapter final | 0.561712 | 0.973433 | 0.802034 | 0.865948 | 0.874601 |
| residual layer 3 | 0.637620 | 0.991802 | 0.942311 | 0.967664 | 0.964627 |

Low frequencies produce the broad circular estimate; exact residue selection
requires mid-frequency sharpening. For the residual architecture, `k<=13`
slightly beats the full `k<=21` layer-3 head, again suggesting that some higher
terms are not cleanly helpful.

The adapter ablation is decisive in a narrow sense: setting adapter feedback to
zero at evaluation collapses the trained adapter model to `0.135115` held-out
accuracy. So the adapter path is causally used by that model. The negative
result is comparative: the feedback architecture learns a worse rule than
auxiliary or residual layerwise supervision, especially on wrap cases.

The limited-range hypothesis is not supported. All three models are better on
low ordinary sums than on the full held-out set, but the analyzers do not mark
them as limited-to-50 models. They learn global cyclic structure and mostly
fail by adjacent residue margins.

Top held-out error offsets:

| run | correct | -1 offset | +1 offset | other notable offsets |
| --- | ---: | ---: | ---: | --- |
| auxiliary | 6268 | 239 | 67 | small number of `+2`, `-2`, and nonlocal alias cases |
| adapter | 5761 | 347 | 446 | more local boundary confusion than the other two |
| residual | 6377 | 169 | 39 | only two nonlocal offset-34 errors |

## Layerwise Residual Weighting

The learned residual mixture for the best layerwise-residual checkpoint is:

| layerwise logit | softmax weight |
| --- | ---: |
| layer 0 | 0.063517 |
| layer 1 | 0.060813 |
| layer 2 | 0.074647 |
| layer 3 | 0.801023 |

The model is therefore still mostly final-layer driven, but the earlier
Dirac/Fourier layer logits contribute about 20% of the final answer. This is
not an equal-depth vote. It is a final-layer cyclic rule with useful earlier
regularization or residual support.

## Frozen Readout Diagnostic

The best layerwise-residual checkpoint is stronger than a frozen post-hoc
readout over its final features:

| checkpoint | deployed exact | frozen linear | frozen Fourier delta | best frozen kernel | cutoff |
| --- | ---: | ---: | ---: | --- | ---: |
| layerwise residual best | 0.968119 | 0.527402 | 0.533475 | Fejer | 3 |

This is the opposite of a simple "features already contain a clean Fourier
code" story. The deployed circuit/head pair has co-adapted into a strong
classifier, while a fresh frozen readout over the final probability vector
cannot recover that exact rule.

## Interpretation

Interspersing Dirac/Fourier heads makes research sense, and the results are
positive but bounded.

The auxiliary variant shows that residue-shaped deep supervision can train a
strong cyclic rule from scratch. The residual variant is better: letting the
model combine layerwise Dirac logits reaches `0.968119` held-out exact with
mostly adjacent errors. The adapter variant is the negative result: feeding
Dirac coefficients back into the next layer worsened optimization and created
a larger wrap/no-wrap gap.

The most defensible conclusion is:

> Layerwise Dirac/Fourier structure is useful as deep supervision and as a
> residual readout scaffold. The auxiliary run even contains a better
> layer-ensemble solution than its final head. The current QNN still wants a
> dominant late cyclic decision layer, and feeding the residue code back into
> the quantum feature map is not automatically helpful.

This supports the earlier conclusion that the Dirac head's role is sharpening
and shaping the cyclic rule, not replacing the learned rule.

## Same-Sum Invariance Follow-Up

The same-sum sweep tests whether explicitly teaching the group law improves the
layerwise QNNs. For any offset `d`, the view:

```latex
T_d(a,b)=((a+d)\bmod p,(b-d)\bmod p)
```

has the same label as `(a,b)`. Operand swap also preserves the label. The
objective adds a symmetric KL penalty between the original and same-sum view:

```latex
\mathcal{L}_{\text{same}}
= {1\over 2}
\left[
\operatorname{KL}(p_\tau(\ell(x))\Vert p_\tau(\ell(Tx)))
+
\operatorname{KL}(p_\tau(\ell(Tx))\Vert p_\tau(\ell(x)))
\right],
```

and a small CE term on the transformed view. This directly pressures the model
to represent the same-sum equivalence class rather than the ordered operand
pair.

The first pass used `p=97`, data split seed `0`, model seed `0`, and train
ratio `0.30`. Controls are the matching seed-0 publication-sweep runs.

| architecture | control | same-sum | gain | control errors | same-sum errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| auxiliary/control | 0.959769 | 0.975406 | +0.015637 | 265 | 162 |
| fixed mean | 0.973736 | 0.980871 | +0.007135 | 173 | 126 |
| learned residual | 0.931228 | 0.975254 | +0.044026 | 453 | 163 |

The result is positive across all three tested layerwise families. The
fixed-mean plus same-sum run is the best seed-0 from-scratch QNN in this set:
`0.980871` held-out exact, with wrap `0.983199` and no-wrap `0.978654`.

This supports the branch's broader interpretation. The QNN benefits when the
training objective makes the finite-group symmetry explicit. Same-sum pressure
does not remove the need for Dirac/Fourier sharpening, but it makes the
underlying cyclic representation more stable.

Artifacts:

```text
modular_addition/qnn_same_sum_sweep.py
configs/modular_addition_qnn_mod97_same_sum_*_seed0.yaml
runs/modular_addition_qnn_mod97_same_sum_*_seed0
analysis/qnn_same_sum_sweep/QNN_SAME_SUM_SWEEP_SUMMARY.md
```

## Seed and Modulus Sweep

The publication-readiness sweep repeated the layerwise QNN families over
moduli `31`, `97`, and `127`, with seeds `0`, `1`, and `2`, and compared them
against matched classical Fourier-delta controls.

| family | p=31 held-out mean | p=97 held-out mean | p=127 held-out mean |
| --- | ---: | ---: | ---: |
| QNN auxiliary | 0.369490 | 0.941248 | 0.957252 |
| QNN adapter | 0.272412 | 0.921411 | 0.972397 |
| QNN residual | 0.375433 | 0.945549 | 0.964751 |
| QNN fixed layerwise mean | 0.394750 | 0.975305 | 0.983438 |
| matched Fourier-delta baseline | 0.022784 | 0.863165 | 0.990966 |
| product-Fourier delta upper bound | 1.000000 | 1.000000 | 1.000000 |

The sweep changes the interpretation. The `p=31` QNN runs do not generalize
well despite high train accuracy, while `p=97` and `p=127` are strong across
seeds. The matched classical Fourier-delta baseline is weaker than the QNNs at
`p=97` but almost perfect at `p=127`, so the result should be discussed as an
inductive-bias comparison rather than quantum advantage.

The fixed layerwise mean is the strongest QNN family in this sweep. This
confirms that the post-hoc averaging improvement was not just an evaluation
artifact: forcing the model to train the uniform layerwise average directly
improves seed-averaged held-out accuracy at `p=97` and `p=127`.

## Artifacts

- Code: `modular_addition/qnn_mod97.py`
- Configs:
  - `configs/modular_addition_qnn_mod97_layerwise_dirac_aux_scratch.yaml`
  - `configs/modular_addition_qnn_mod97_layerwise_dirac_adapter_scratch.yaml`
  - `configs/modular_addition_qnn_mod97_layerwise_dirac_residual_scratch.yaml`
  - `configs/modular_addition_qnn_mod97_same_sum_aux_control_seed0.yaml`
  - `configs/modular_addition_qnn_mod97_same_sum_mean_seed0.yaml`
  - `configs/modular_addition_qnn_mod97_same_sum_residual_seed0.yaml`
- Runs:
  - `runs/modular_addition_qnn_mod97_layerwise_dirac_aux_scratch`
  - `runs/modular_addition_qnn_mod97_layerwise_dirac_adapter_scratch`
  - `runs/modular_addition_qnn_mod97_layerwise_dirac_residual_scratch`
  - `runs/modular_addition_qnn_mod97_same_sum_aux_control_seed0`
  - `runs/modular_addition_qnn_mod97_same_sum_mean_seed0`
  - `runs/modular_addition_qnn_mod97_same_sum_residual_seed0`
- Analysis:
  - `analysis/qnn_layerwise_dirac/aux_scratch_mech`
  - `analysis/qnn_layerwise_dirac/adapter_scratch_mech`
  - `analysis/qnn_layerwise_dirac/residual_scratch_mech`
  - `analysis/qnn_layerwise_dirac/residual_scratch_delta_head`
  - `analysis/qnn_layerwise_dirac_exhaustive`
  - `analysis/qnn_same_sum_sweep/QNN_SAME_SUM_SWEEP_SUMMARY.md`
  - `analysis/qnn_layerwise_mean_sweep/MEAN_SWEEP_COMPARISON.md`
  - `analysis/qnn_publication_sweep/SWEEP_AGGREGATE.md`
  - `figures/qnn_publication/qnn_seed_modulus_sweep.png`
