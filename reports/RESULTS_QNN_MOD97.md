# Quantum Neural Networks On Modular Addition

This experiment trains simulated quantum neural networks on the numeric
`(a+b) mod 97` split.

The implemented circuit is a data-reuploading variational quantum classifier:
inputs are encoded as repeated rotations on a 7-qubit statevector, entangling
CNOT rings mix the register, and labels are read from either Born probabilities
or quantum-state features.

## Variants

| variant | readout | quantum dependence |
| --- | --- | --- |
| `born` | normalized probabilities of computational basis states 0..96 | strictest; no learned classical head after measurement. |
| `prob_head` | linear head over all 128 basis-state probabilities | quantum feature map plus classical linear readout. |
| `expval_head` | linear head over single-qubit Z and nearest-neighbor ZZ expectations | compact expectation-value readout. |

## Methodology And Architecture

The QNN is a simulated data-reuploading variational circuit. It receives
residues `a,b in Z_p` through periodic angles:

```latex
\theta_a = {2\pi a\over p}, \qquad
\theta_b = {2\pi b\over p}.
```

With `n` qubits and `L` reuploading layers, the state is:

```latex
|\psi_L(a,b;\Theta)\rangle =
U_L(a,b;\Theta_L)\cdots U_2(a,b;\Theta_2)U_1(a,b;\Theta_1)
|0\rangle^{\otimes n}.
```

Each layer applies trainable input-dependent rotations and an entangling block:

```latex
U_\ell(a,b;\Theta_\ell)
= C_\ell
\prod_{q=1}^{n}
R_z(\alpha^z_{\ell q}\cdot\phi(a,b)+\rho^z_{\ell q})
R_y(\alpha^y_{\ell q}\cdot\phi(a,b)+\rho^y_{\ell q})
R_x(\alpha^x_{\ell q}\cdot\phi(a,b)+\rho^x_{\ell q}).
```

The reason this architecture is relevant to modular addition is that the
rotation entries are trigonometric functions of the inputs, and products of
rotation entries remain trigonometric polynomials. After reuploading and
entanglement, measured features can be written schematically as:

```latex
z_m(a,b) =
\sum_{(u,v)\in\Omega_m}
c_{uvm} e^{i(u\theta_a+v\theta_b)}.
```

Functions of the modular sum concentrate on the addition diagonal
`(u,v)=(k,k)`. The exact class indicator for candidate answer `c` is the finite
cyclic delta:

```latex
\delta_p(a+b-c)
= {1\over p}
\sum_{k=0}^{p-1}
\exp\left({2\pi i k(a+b-c)\over p}\right).
```

In real form, the answer can be represented by cosine kernels over the modular
residual `r=(a+b-c) mod p`:

```latex
K(r) = \beta_0 + \sum_{k=1}^{K}
\rho_k \cos\left({2\pi kr\over p}\right).
```

This motivates the Fourier-delta readout. The head predicts Fourier features
of the sum:

```latex
(\hat{u}_k,\hat{v}_k)
\approx
\left[
\cos\left({2\pi k(a+b)\over p}\right),
\sin\left({2\pi k(a+b)\over p}\right)
\right],
```

then scores class `c` by dotting those features with the class Fourier basis:

```latex
\ell_c =
\beta_c
+ \gamma\sum_{k=1}^{K}\rho_k
\left[
\hat{u}_k\cos\left({2\pi kc\over p}\right)
+
\hat{v}_k\sin\left({2\pi kc\over p}\right)
\right]
+ r_c.
```

When the predicted phase is correct, the summand becomes
`cos(2*pi*k*((a+b)-c)/p)`, i.e. a finite modular delta kernel. The main
mechanistic question is whether the QNN learns this cyclic scaffold and whether
the readout sharpens it into exact residue selection.

Layerwise Dirac/Fourier variants attach such a head after every reuploading
layer. If `z_l` is the measured feature state after layer `l`, the layerwise
head is:

```latex
\ell^{(\ell)} = H_\ell(z_\ell).
```

The auxiliary version deploys the final head while adding layerwise losses:

```latex
\mathcal{L}
= \operatorname{CE}(\ell_{\text{final}},y)
+\lambda_{\text{layer}} {1\over L}
\sum_{\ell=1}^{L}
\operatorname{CE}(\ell^{(\ell)},y).
```

The fixed mean version makes the deployed answer a uniform average:

```latex
\ell_{\text{mean}} = {1\over L}\sum_{\ell=1}^{L}\ell^{(\ell)}.
```

The learned residual version uses a softmax-weighted layer mixture:

```latex
\ell_{\text{residual}}
= \sum_{\ell=1}^{L}
{e^{w_\ell}\over \sum_j e^{w_j}}
\ell^{(\ell)}.
```

The same-sum objective uses equivalent views
`(a,b)` and `((a+d) mod p, (b-d) mod p)`, plus optional operand swap. It adds a
symmetric KL consistency penalty:

```latex
\mathcal{L}_{\text{same}}
= {1\over 2}
\left[
\operatorname{KL}(p_\tau(\ell(x))\Vert p_\tau(\ell(x')))
+
\operatorname{KL}(p_\tau(\ell(x'))\Vert p_\tau(\ell(x)))
\right],
```

where `x` and `x'` have the same modular sum. This is the QNN analogue of the
same-sum invariance pressure that made the contrastive JEPA branch work.

The full theory appendix is:

```text
QNN_DATA_REUPLOADING_FOURIER_THEORY.md
```

## Results

Chance accuracy is `1/97 = 0.010309`.

| run | variant | params | best held-out accuracy | train accuracy at best | wrap accuracy at best | no-wrap accuracy at best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| diagnostic 300 steps | `born` | 252 | 0.022013 | not measured | 0.027380 | 0.016899 |
| diagnostic 300 steps | `prob_head` | 13021 | 0.170184 | not measured | 0.177660 | 0.163060 |
| diagnostic 300 steps | `expval_head` | 1735 | 0.134811 | not measured | 0.133167 | 0.136377 |
| main 1000 steps | `prob_head` | 13021 | 0.242751 | not measured | 0.253578 | 0.232434 |
| main 1000 steps | `expval_head` | 1735 | 0.218005 | not measured | 0.219353 | 0.216721 |
| continuation 3000 steps | `prob_head` | 13021 | 0.311523 | 0.674699 | 0.320162 | 0.303291 |
| continuation 3000 steps | `expval_head` | 1735 | 0.368605 | 0.568037 | 0.390790 | 0.347465 |
| resumed 6000-step target | `prob_head` | 13021 | 0.354334 | 0.750886 | 0.355009 | 0.353691 |
| resumed 6000-step target | `expval_head` | 1735 | 0.395020 | 0.613749 | 0.387990 | 0.401720 |
| resumed 10000-step target | `prob_head` | 13021 | 0.389403 | 0.792346 | 0.407281 | 0.372369 |
| resumed 10000-step target | `expval_head` | 1735 | 0.422499 | 0.621191 | 0.442128 | 0.403795 |
| direct-aux 30000-step rescue | `prob_head` | 13021 | 0.469106 | 0.884479 | 0.504045 | 0.435814 |
| direct-aux 30000-step rescue | `expval_head` | 1735 | 0.496432 | 0.676116 | 0.497822 | 0.495108 |
| initialized Fourier-delta 10000-step rescue | `expval_head` | 4804 | 0.902535 | 0.994685 | 0.921282 | 0.884672 |
| 10k-to-30k precision continuation | `expval_head` | 4804 | 0.926066 | 1.000000 | 0.948040 | 0.905129 |
| native `k<=21` Fourier-delta rescue | `expval_head` | 3967 | 0.954304 | 1.000000 | 0.965775 | 0.943374 |
| no-residual boundary rescue | `expval_head` | 2512 | 0.979505 | 0.996456 | 0.980709 | 0.978358 |
| no-residual `k<=13` warm-start seed-sweep best | `expval_head` | 2264 | 0.982541 | 0.996456 | 0.981954 | 0.983101 |
| 4k continuation from boundary best | `expval_head` | 2512 | 0.981327 | 0.997165 | 0.981021 | 0.981619 |
| Dirac identity bridge smoke | `expval_head` | 2512 | 0.982693 | 0.997165 | 0.982887 | 0.982508 |
| Dirac identity bridge full run | `expval_head` | 2512 | 0.982693 | 0.997165 | 0.982887 | 0.982508 |
| Dirac `soft_unit` smoke | `expval_head` | 2512 | 0.206923 | 0.203047 | 0.211263 | 0.202787 |
| Dirac `global_soft_unit` smoke | `expval_head` | 2512 | 0.204190 | 0.200567 | 0.200373 | 0.207827 |
| Dirac residual sharpener | `expval_head` | 1051 | 0.982997 | 0.997165 | 0.982887 | 0.983101 |
| Dirac residual sharpener from scratch | `expval_head` | 1051 | 0.734325 | 0.793055 | 0.751089 | 0.718352 |
| Dirac correction-only from identity | `expval_head` | 1051 | 0.982845 | 0.997165 | 0.982845 | 0.982845 |
| Head-decisive scratch | `expval_head` | 2534 | 0.676029 | 0.776045 | 0.681394 | 0.670916 |
| Head-decisive warm-start | `expval_head` | 2534 | 0.948080 | 0.984054 | 0.948351 | 0.947821 |
| Primary-head ablate correction | `expval_head` | 2535 | 0.936086 | 0.968816 | 0.936086 | 0.936086 |
| Primary-head ablate base | `expval_head` | 2535 | 0.116138 | 0.163005 | 0.116138 | 0.116138 |
| Original expval Dirac bridge | `expval_head` | 1029 | 0.754820 | 0.856130 | 0.781269 | 0.729618 |
| Original prob Dirac bridge | `prob_head` | 6045 | 0.977076 | 0.999291 | 0.983199 | 0.971242 |
| Original expval Dirac `soft_unit` | `expval_head` | 1029 | 0.462122 | 0.567682 | 0.511512 | 0.415061 |
| Original expval Dirac `global_soft_unit` | `expval_head` | 1029 | 0.343555 | 0.481928 | 0.369322 | 0.319004 |
| Original expval residual sharpener | `expval_head` | 1051 | 0.754972 | 0.856485 | 0.781269 | 0.729914 |
| Original expval primary Dirac | `expval_head` | 1052 | 0.859116 | 0.918498 | 0.859676 | 0.858583 |
| Original expval primary ablate correction | `expval_head` | 1052 | 0.861697 | 0.918498 | 0.861697 | 0.861697 |
| Original expval primary ablate base | `expval_head` | 1052 | 0.096250 | 0.144224 | 0.096250 | 0.096250 |
| Original expval primary scratch | `expval_head` | 1052 | 0.460756 | 0.546067 | 0.481643 | 0.440854 |
| Layerwise Dirac auxiliary scratch | `prob_head` | 29217 | 0.951571 | 0.976612 | 0.930305 | 0.971835 |
| Layerwise Dirac adapter scratch | `prob_head` | 29905 | 0.874601 | 0.933735 | 0.818295 | 0.928254 |
| Layerwise Dirac residual scratch | `prob_head` | 23428 | 0.968119 | 0.983345 | 0.949284 | 0.986066 |
| Layerwise Dirac fixed mean seed-0 control | `prob_head` | 23424 | 0.973736 | 0.990432 | 0.961419 | 0.985473 |
| Same-sum layerwise auxiliary seed 0 | `prob_head` | 29217 | 0.975406 | 0.982636 | 0.976665 | 0.974207 |
| Same-sum layerwise fixed mean seed 0 | `prob_head` | 23424 | 0.980871 | 0.992204 | 0.983199 | 0.978654 |
| Same-sum layerwise residual seed 0 | `prob_head` | 23428 | 0.975254 | 0.984408 | 0.973864 | 0.976579 |

Best post-hoc guarded calibration over frozen Fourier-delta logits reaches
`0.984060` held-out exact with zero high-margin prediction changes. This is
reported separately because it is an evaluation-time correction diagnostic, not
a newly trained checkpoint.

Final checkpoints:

| variant | final step | final train accuracy | final held-out accuracy | best step |
| --- | ---: | ---: | ---: | ---: |
| `prob_head` | 10000 | 0.792346 | 0.389403 | 10000 |
| `expval_head` | 10000 | 0.517009 | 0.365872 | 8000 |

`expval_head` reached its best checkpoint at step 8000 and then regressed.
Comparisons should use `checkpoint_expval_head_best.pt`, not the final
checkpoint. `prob_head` improved through step 10000, so its best and final
checkpoints are the same.

30k direct-aux rescue checkpoints:

| variant | best step | best train accuracy | best held-out accuracy | final step | final held-out accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prob_head` | 26000 | 0.884479 | 0.469106 | 30000 | 0.431608 |
| `expval_head` | 22000 | 0.676116 | 0.496432 | 30000 | 0.447852 |

Fourier-delta rescue checkpoints:

| run | best step | best train accuracy | best held-out accuracy | final step | final held-out accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| initialized 10k delta rescue | 10000 | 0.994685 | 0.902535 | 10000 | 0.902535 |
| 10k-to-30k precision continuation | 15000 | 1.000000 | 0.926066 | 30000 | 0.906938 |

## Interpretation

These are near-term-style simulated QNN baselines, not claims of quantum
advantage. The useful comparison for this branch is whether compact
quantum-state representations naturally discover the cyclic rule under the
same train/test split.

A strict Born classifier is the most quantum-native readout. Hybrid probability
or expectation heads test whether a quantum feature representation is usable
once a small classical readout is allowed.

The 30000-step direct-aux rescue improved both hybrid models but still did not
produce grokking. `prob_head` reached `0.884479` train and `0.469106` held-out
accuracy at its best step 26000 checkpoint, then regressed by step 30000.
`expval_head` reached the best overall held-out checkpoint, `0.496432`, at step
22000 with train accuracy `0.676116`, then also regressed. Longer training and
auxiliary residue pressure therefore help, but do not recover an exact modular
algorithm under this architecture.

The readout comparison is now clearer. `prob_head` fits the training split more
strongly and carries more non-sum feature structure. `expval_head` gives the
best held-out result and the cleaner compact cyclic feature geometry, but it is
still only an approximate arithmetic readout.

The strict `born` classifier remains the clearest negative control. Simply
allocating one basis state per residue does not make the circuit discover a
residue classifier under this depth and training budget. Hybrid heads are
needed to extract the partial signal.

## 30k Mech Interp

The completed 30k pair was analyzed with all-pairs behavior stratification,
logit residual fits, Fourier spectra, feature-grid FFTs, Fourier feature
probes, shortcut tests, and nearest-train diagnostics:

```text
analysis/qnn_mod97_mech/QNN_MOD97_MECH_INTERP_REPORT.md
```

The small-range hypothesis is not supported. `prob_head_best` gets `0.457173`
held-out accuracy on examples with ordinary sum `a+b <= 50`, below its overall
held-out accuracy `0.469106`. `expval_head_best` gets `0.507495` on that slice,
only about one point above its overall held-out accuracy `0.496432`. Wrap and
no-wrap accuracies are similar, so there is no evidence for a cutoff around 50.

The stronger explanation is a coarse low-frequency cyclic estimator. Exact
held-out accuracy is only about 47-50%, but both best checkpoints are correct
or off by one on about 95% of held-out examples:

| checkpoint | exact | within 1 | within 2 | mod-add R2 | addition energy k <= 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prob_head_best` | 0.469106 | 0.946106 | 0.989221 | 0.521841 | 0.999442 |
| `expval_head_best` | 0.496432 | 0.950205 | 0.994687 | 0.564857 | 0.999997 |

This is a limited Fourier/cyclic circuit, but not one limited to ordinary sums
up to 50. It is limited by frequency/readout precision: the learned logit rule
puts substantial energy on the modular-addition residual manifold, yet almost
all of that energy is in the first few Fourier modes. Operand-copy,
class-prior, and small-modulus shortcuts are weak by comparison.

## Why This Happens

The best mechanistic interpretation is that the data-reuploading QNN behaves
like a trainable finite Fourier feature map. The inputs enter as angles, the
circuit repeatedly applies rotations controlled by those angle features, and
the measured probabilities or expectation values are trigonometric functions of
`a` and `b`. With only 7 qubits, 4 reuploading layers, and ring entanglement,
the easiest stable functions are low-frequency modes on the operand torus.

Exact mod-97 classification needs more than a smooth estimate of `(a+b) mod
97`. It needs a sharp residue-selective readout: the score for class `c` must
peak tightly when `(a+b-c) mod 97 = 0` and drop for the neighboring residues.
The QNNs instead learn a broad circular bump. That gives high within-one
accuracy, but only about 47-50% exact accuracy.

This explains why the failure is not a small ordinary-sum rule. Low-frequency
cyclic modes are periodic and global, so they do not stop working at ordinary
sum 50. Their failure mode is angular precision: the model knows roughly where
the answer sits on the residue circle but often cannot resolve the adjacent
residue.

The readout bottleneck also matters. `prob_head` exposes all 128 basis-state
probabilities to a linear head, so it can fit train examples more aggressively
and carry more operand-specific structure. `expval_head` compresses the quantum
state into only 14 Z/ZZ observables, which forces a cleaner addition-diagonal
geometry but loses fine detail. The strict Born readout is worst because the
computational basis is not naturally organized by mod-97 addition; the circuit
would have to learn both the cyclic arithmetic representation and an exact
allocation of residues to basis outcomes.

The direct auxiliary residue head helps by pressuring the measured features to
carry the modular sum, but it does not by itself create the missing
high-frequency components. The result is a real cyclic scaffold without the
sharp class-selection mechanism seen in the successful Transformer, RWKV
classifier, and contrastive JEPA.

## Frozen Delta-Head Test

We then tested whether a finite-modular "Dirac delta" head could sharpen the
existing frozen QNN features. The analyzer trains linear ridge probes from
frozen QNN features to Fourier features of `s=(a+b) mod 97`, then synthesizes
class logits with Dirichlet, Fejer, or train-R2-weighted circular kernels.

```text
modular_addition/qnn_delta_head.py
analysis/qnn_delta_head/QNN_DELTA_HEAD_REPORT.md
```

Result:

| checkpoint | original | frozen linear | best Fourier delta | delta kernel | cutoff |
| --- | ---: | ---: | ---: | --- | ---: |
| `prob_head_best` | 0.469106 | 0.474723 | 0.489145 | Fejer | 5 |
| `prob_head_final` | 0.431608 | 0.470320 | 0.494762 | Fejer | 5 |
| `expval_head_best` | 0.496432 | 0.729164 | 0.271747 | train-R2 | 8 |
| `expval_head_final` | 0.447852 | 0.666920 | 0.256566 | Fejer | 5 |

The post-hoc Fourier delta head does not solve the problem. It gives only a
small improvement for `prob_head` and is much worse than the checkpoint head for
`expval_head`. This means the current frozen features are not linearly
Fourier-clean enough for a hand-synthesized delta kernel.

The important positive result is the frozen generic linear readout:
`expval_head_best` jumps from `0.496432` to `0.729164` held-out exact accuracy
when the circuit is frozen and only a new linear softmax readout is trained.
That says the compact expectation-value features contain much more
class-separating information than the trained checkpoint head uses, but that
information is warped rather than arranged as a clean Fourier basis.

A delta head still makes sense as an end-to-end training target, not as a
post-hoc frozen readout. The next rescue should combine:

- an alternating or final readout-refresh phase, especially for `expval_head`;
- Fourier auxiliary losses over `k=1..48`;
- a hard-neighbor margin loss against `s+1` and `s-1`;
- then a trainable Fourier/delta synthesis head whose features co-adapt with
  the circuit.

## 10k Expval Delta Rescue

We implemented that combined rescue for `expval_head`:

```text
configs/modular_addition_qnn_mod97_expval_delta_10k.yaml
runs/modular_addition_qnn_mod97_expval_delta_10k
```

The run initializes from the completed 30k `expval_head_best` circuit, skips
the old softmax head, and trains a hybrid Fourier-delta readout:

- Fourier/delta synthesis over all `k=1..48`;
- residual linear readout path, because the frozen features were class-
  separable but not Fourier-clean;
- direct 97-way auxiliary residue head;
- Fourier auxiliary MSE over `k=1..48`;
- hard-neighbor margin against `s+1` and `s-1`;
- initial, periodic, and final head-refresh phases with the circuit frozen.

Result after 10,000 main steps:

| metric | value |
| --- | ---: |
| best held-out exact accuracy | 0.902535 |
| final held-out exact accuracy | 0.902535 |
| final train exact accuracy | 0.994685 |
| held-out no-wrap accuracy | 0.884672 |
| held-out wrap accuracy | 0.921282 |
| held-out `a+b <= 50` accuracy | 0.892934 |
| held-out within +/-1 accuracy | 0.996508 |
| held-out within +/-2 accuracy | 0.998026 |

This is a large rescue. It improves over the 30k direct-aux `expval_head_best`
checkpoint from `0.496432` to `0.902535` held-out exact accuracy.

Post-run mech diagnostics:

```text
analysis/qnn_expval_delta_10k_mech/QNN_MOD97_MECH_INTERP_REPORT.md
analysis/qnn_expval_delta_10k_delta_head/QNN_DELTA_HEAD_REPORT.md
analysis/qnn_expval_delta_10k_exhaustive/QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md
```

| diagnostic | value |
| --- | ---: |
| logit mod-add residual R2 | 0.546051 |
| feature addition-diagonal energy | 0.338184 |
| addition energy captured by `k <= 3` | 0.998864 |
| top held-out error offset `+1` | 0.055109 |
| top held-out error offset `-1` | 0.038864 |

The rescue mostly solved the adjacent-residue readout problem. Exact held-out
accuracy is now above 90%, and nearly all remaining errors are still local on
the residue circle. The small-range hypothesis remains unsupported:
`a+b <= 50` is slightly worse than the overall held-out split.

The post-run frozen readout diagnostic is also informative. A fresh generic
linear readout over the final features reaches only `0.803704`, and a post-hoc
Fourier delta readout reaches only `0.275543`. The trained head is doing real
work. The result is not just that the circuit features became cleanly
Fourier-linear; rather, the end-to-end Fourier-delta head co-adapted with the
circuit.

The exhaustive head decomposition clarifies where the 90% behavior lives:

| head variant | held-out exact | held-out within +/-1 | held-out top-2 |
| --- | ---: | ---: | ---: |
| full checkpoint head | 0.902535 | 0.996508 | 0.992106 |
| Fourier-delta only | 0.898588 | 0.997723 | 0.993927 |
| Fourier-delta plus bias | 0.902383 | 0.997723 | 0.993624 |
| residual linear only | 0.013056 | 0.041749 | 0.036284 |
| residual linear plus bias | 0.010475 | 0.034006 | 0.034462 |

So the residual linear channel is not the source of the rescue. It is near
chance when isolated. Almost all exact class selection is in the trained
Fourier-delta pathway, with bias and residual terms providing small local
adjustments.

The cutoff sweep shows why the earlier QNNs had high within-one accuracy but
poor exact accuracy:

| maximum frequency | held-out exact | held-out within +/-1 |
| ---: | ---: | ---: |
| 3 | 0.275239 | 0.712616 |
| 5 | 0.581145 | 0.961895 |
| 8 | 0.822529 | 0.997267 |
| 13 | 0.902991 | 0.997419 |
| 21 | 0.907697 | 0.996812 |
| 48 | 0.902535 | 0.996508 |

Frequencies `1-5` build the broad circular bump. Frequencies around `8-21`
sharpen the kernel enough for exact residue selection. Higher frequencies do
not help in this checkpoint and slightly reduce exact accuracy.

## 10k To 30k Precision Continuation

We then continued the 10k best checkpoint in a separate run directory, leaving
the original 10k artifacts untouched and preserving a copy at
`runs/modular_addition_qnn_mod97_expval_delta_10k_preserved_20260608`.

```text
configs/modular_addition_qnn_mod97_expval_delta_10k_to_30k_precision.yaml
runs/modular_addition_qnn_mod97_expval_delta_10k_to_30k_precision
analysis/qnn_expval_delta_10k_to_30k_precision_mech
analysis/qnn_expval_delta_10k_to_30k_precision_delta_head
analysis/qnn_expval_delta_10k_to_30k_precision_exhaustive_best
analysis/qnn_expval_delta_10k_to_30k_precision_exhaustive_final
```

This continuation used a lower learning rate, Fourier auxiliary pressure focused
through `k<=21`, a mild penalty above `k=21`, wider hard-neighbor ranking over
`s±1`, `s±2`, and `s±3`, and boundary oversampling.

| checkpoint | step | held-out exact | train exact | sum <= 50 | wrap | no-wrap | within +/-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| precision best | 15000 | 0.926066 | 1.000000 | 0.896146 | 0.948040 | 0.905129 | 0.998785 |
| precision final | 30000 | 0.906938 | 1.000000 | 0.872591 | 0.931861 | 0.883190 | 0.997419 |

The continuation improved the best exact held-out score from `0.902535` to
`0.926066`, but it did not approach `0.99`. Training was oscillatory: the final
checkpoint drifted back to `0.906938` despite perfect train accuracy.

Head decomposition for the precision best checkpoint:

| head variant | held-out exact | held-out within +/-1 | held-out top-2 |
| --- | ---: | ---: | ---: |
| full checkpoint head | 0.926066 | 0.998785 | 0.995294 |
| Fourier-delta only | 0.919083 | 0.998482 | 0.996508 |
| Fourier-delta plus bias | 0.920298 | 0.998634 | 0.996356 |
| residual linear only | 0.020950 | 0.064066 | 0.049643 |

The same basic mechanism persists: the trained Fourier-delta pathway carries the
answer, while the residual path is near chance by itself. The best checkpoint is
not a small-range rule: `a+b<=50` is worse than the full held-out split.

The cutoff sweep is the most actionable result:

| maximum frequency | held-out exact, precision best | held-out exact, precision final |
| ---: | ---: | ---: |
| 5 | 0.546076 | 0.487779 |
| 8 | 0.848034 | 0.847275 |
| 13 | 0.926522 | 0.908152 |
| 21 | 0.931684 | 0.918324 |
| 48 | 0.926066 | 0.906938 |

For both best and final checkpoints, evaluating the same trained head with a
`k<=21` cutoff beats the full `k<=48` head. The next run should therefore use a
native `fourier_max_frequency` around `21` rather than merely penalizing high
frequencies. The current penalty was too weak to remove harmful high-frequency
terms.

Frozen-readout diagnostics confirm that the trained head/circuit combination is
still doing the work. On the precision best checkpoint, a refreshed frozen
linear readout reaches `0.824958`, and the best frozen analytic delta readout
reaches only `0.276454`.

## Native Low-Frequency Rescue And Early Stopping

We implemented the next QNN recommendation directly:

```text
configs/modular_addition_qnn_mod97_expval_delta_k21_native.yaml
configs/modular_addition_qnn_mod97_expval_delta_k13_native.yaml
runs/modular_addition_qnn_mod97_expval_delta_k21_native
```

The key code change is that `modular_addition/qnn_mod97.py` can now initialize
compatible tensor slices when loading a checkpoint. This lets a native
`k<=21` Fourier-delta head inherit the low-frequency projection and frequency
weights from the old `k<=48` precision checkpoint while discarding the harmful
high-frequency channels.

The manually stopped native `k<=21` run reached a new QNN best before the
optimizer drifted:

| step | held-out exact | train exact | wrap | no-wrap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.927736 | 0.978030 | 0.930616 | 0.924993 |
| 500 | 0.937149 | 0.992913 | 0.951462 | 0.923510 |
| 1000 | 0.954304 | 1.000000 | 0.965775 | 0.943374 |
| 2000 | 0.952786 | 1.000000 | 0.963597 | 0.942484 |
| 6000 | 0.940641 | 1.000000 | 0.953640 | 0.928254 |

The best checkpoint is:

```text
runs/modular_addition_qnn_mod97_expval_delta_k21_native/checkpoint_expval_head_best.pt
```

with checkpoint step `1000` and held-out exact accuracy `0.954304`. This is a
clear improvement over both the previous full `k<=48` precision best
(`0.926066`) and the analysis-time `k<=21` cutoff of that checkpoint
(`0.931684`). The result supports the native-low-frequency-head hypothesis.

The run also confirmed that the remaining failure mode is optimization
stability, not simple undertraining. Accuracy oscillated after reaching perfect
train accuracy, so the trainer now supports validation early stopping:

```text
early_stopping_metric
early_stopping_mode
early_stopping_min_delta
early_stopping_min_step
early_stopping_patience_evals
early_stopping_restore_best
run_final_refresh_on_early_stop
```

Both native rescue configs enable early stopping on held-out exact accuracy and
restore the best checkpoint. A CPU smoke run verified that early stopping stops
after validation stagnation, preserves the first best checkpoint, and writes a
final checkpoint restored to that best state.

## Native K21 Exhaustive Mech Interp

We then ran the full QNN mech-interp stack on the successful native `k<=21`
checkpoint:

```text
runs/modular_addition_qnn_mod97_expval_delta_k21_native/checkpoint_expval_head_best.pt
analysis/qnn_expval_delta_k21_native_mech
analysis/qnn_expval_delta_k21_native_exhaustive_best
analysis/qnn_expval_delta_k21_native_delta_head
```

Behavioral evaluation:

| split | accuracy | loss |
| --- | ---: | ---: |
| train | 1.000000 | 0.001593 |
| held-out | 0.954304 | 0.692788 |
| held-out no-wrap | 0.943374 | 1.053228 |
| held-out wrap | 0.965775 | 0.314516 |
| held-out `a+b <= 50` | 0.927195 | 2.984158 |

The limited-range hypothesis is not supported. The small ordinary-sum slice is
worse than overall held-out accuracy, not better, and wrap cases are stronger
than no-wrap cases.

Signed-offset evaluation:

| split | errors | `+1` errors | `-1` errors | adjacent-error share | nonlocal `|offset|>2` |
| --- | ---: | ---: | ---: | ---: | ---: |
| held-out | 301 | 130 | 159 | 0.960133 | 12 |
| no-wrap | 191 | 87 | 92 | 0.937173 | 12 |
| wrap | 110 | 43 | 67 | 1.000000 | 0 |
| `a+b <= 50` | 68 | 36 | 20 | 0.823529 | 12 |

So almost all remaining failures are exact-residue boundary errors: the model
usually lands on the correct residue or an adjacent residue. The only nonlocal
held-out errors occur in no-wrap/low-sum boundary regions, especially `a=0`
and nearby operand-boundary cases. Accuracy on held-out pairs with `a=0` or
`b=0` is only `0.564885`; with `a=96` or `b=96` it is `0.674074`; and across a
broader operand-boundary slice it is `0.811794`.

Head decomposition:

| head variant | held-out exact | within +/-1 | top-2 |
| --- | ---: | ---: | ---: |
| full checkpoint head | 0.954304 | 0.998178 | 0.996508 |
| Fourier-delta only | 0.955974 | 0.998330 | 0.996812 |
| Fourier-delta plus bias | 0.956278 | 0.998330 | 0.996660 |
| residual linear only | 0.023228 | 0.070745 | 0.047670 |
| residual plus bias | 0.021406 | 0.066798 | 0.046607 |

The explicit Fourier-delta pathway is not merely sufficient; it is slightly
better than the full head. The residual linear channel hurts a little and is
near chance by itself.

Frequency cutoff sweep:

| maximum frequency | held-out exact | within +/-1 | margin |
| ---: | ---: | ---: | ---: |
| 3 | 0.251404 | 0.667223 | -43.957535 |
| 5 | 0.544709 | 0.953089 | -2.153417 |
| 8 | 0.854866 | 0.997116 | 10.340780 |
| 13 | 0.950053 | 0.998330 | 12.817835 |
| 21 | 0.954304 | 0.998178 | 12.847431 |

The result is not a tiny low-frequency circuit. Frequencies `1-5` form the
broad circular bump, but mid-frequency terms through about `13-21` are needed
for exact residue sharpening.

Mechanistic diagnostics:

| diagnostic | value |
| --- | ---: |
| full-logit modular residual R2 | 0.545855 |
| full-logit addition residual Fourier energy | 0.545766 |
| feature addition-diagonal FFT energy | 0.333235 |
| nearest-train same-sum rate from feature cosine | 0.220586 |
| held-out ECE | 0.031845 |

The logits have a strong add-circulant component, but the 14-dimensional
expectation-feature state is still warped rather than a clean same-sum latent
space. Frozen readout controls confirm that interpretation: a refreshed frozen
linear readout over the features reaches `0.820100`, while the best post-hoc
analytic Fourier-delta readout reaches only `0.279338`. The successful result
therefore depends on end-to-end co-adaptation between the circuit features and
the trained Fourier-delta head.

Updated interpretation: the native `k<=21` QNN has learned a strong global
cyclic/addition rule, not a small-range adder. It is still not fully grokked
because a small set of operand-boundary aliases and adjacent-residue margin
failures remain. The next rescue should target boundary consistency and margin
stability rather than simply adding more training steps.

## No-Residual Boundary Rescue

To address the remaining errors, we implemented the targeted rescue implied by
the signed-offset audit:

```text
configs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary.yaml
runs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_norefresh
analysis/qnn_expval_delta_k21_noresid_boundary_mech
analysis/qnn_expval_delta_k21_noresid_boundary_exhaustive_best
analysis/qnn_expval_delta_k21_noresid_boundary_delta_head
```

The intervention combined:

- native `k<=21` Fourier-delta head initialized from the successful checkpoint;
- no residual linear path, since `delta_plus_bias` beat the full head;
- annealed hard margins on `s±1` and `s±2`;
- boundary-only same-sum consistency for `(a+k,b-k)` views and swapped operands;
- same-label CE on those boundary-equivalent views;
- heavier boundary oversampling;
- validation early stopping with best-checkpoint restoration.

An initial version with periodic readout refresh was stopped because the
consistency-augmented refresh path was too slow. The final no-refresh run
completed 2500 configured steps and restored the best checkpoint from step
`2000`.

Training trajectory:

| step | held-out exact | no-wrap | wrap | train exact |
| ---: | ---: | ---: | ---: | ---: |
| native `k<=21` source | 0.954304 | 0.943374 | 0.965775 | 1.000000 |
| no-residual step 1 | 0.956733 | 0.948117 | 0.965775 | 0.988661 |
| step 500 | 0.969789 | 0.963534 | 0.976353 | 0.992204 |
| step 1000 | 0.973129 | 0.971242 | 0.975109 | 0.994685 |
| step 1500 | 0.978291 | 0.977172 | 0.979465 | 0.996811 |
| step 2000 best | 0.979505 | 0.978358 | 0.980709 | 0.996456 |
| step 2500 final eval | 0.978442 | 0.975986 | 0.981021 | 0.996102 |

This reduced held-out errors from `301` to `135`, a `55.1%` error reduction
relative to the native `k<=21` checkpoint.

Updated signed-offset audit:

| split | errors | `+1` errors | `-1` errors | adjacent-error share | nonlocal `|offset|>2` |
| --- | ---: | ---: | ---: | ---: | ---: |
| held-out | 135 | 62 | 71 | 0.985185 | 2 |
| no-wrap | 73 | 37 | 34 | 0.972603 | 2 |
| wrap | 62 | 25 | 37 | 1.000000 | 0 |
| `a+b <= 50` | 18 | 7 | 9 | 0.888889 | 2 |
| `a=0` or `b=0` | 12 | 6 | 6 | 1.000000 | 0 |
| `a=96` or `b=96` | 9 | 3 | 6 | 1.000000 | 0 |

The boundary rescue mostly solved the previous nonlocal alias problem. The old
native checkpoint had 12 held-out nonlocal errors; the no-residual boundary
checkpoint has only 2, both at `(15,34)` and `(15,35)` where the model predicts
`14` instead of `49`/`50`.

Component and frequency diagnostics:

| diagnostic | value |
| --- | ---: |
| full checkpoint held-out exact | 0.979505 |
| delta-only held-out exact | 0.981934 |
| residual-only held-out exact | 0.010172 |
| held-out within +/-1 | 0.999696 |
| held-out top-2 | 0.999696 |
| held-out ECE | 0.003264 |
| full-logit modular residual R2 | 0.541456 |
| feature addition-diagonal FFT energy | 0.329046 |
| frozen generic linear readout | 0.833915 |
| frozen post-hoc delta readout | 0.269167 |

The mechanism is the same but cleaner: a trained Fourier-delta readout over
smooth QNN features. The residual path is now removed; `delta_only` is still
slightly better than the deployed full head, so a future run can probably remove
or reduce bias as well. The remaining errors are almost entirely adjacent
margin mistakes. This checkpoint still has not reached `0.99`, but it is much
closer: reaching `0.99` now means fixing about 69 more held-out examples rather
than 235.

## 4k Continuation From No-Residual Boundary Best

We then continued from the preserved no-residual boundary best checkpoint in a
separate run directory, leaving the step-2000 object untouched:

```text
configs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_to4000.yaml
runs/modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_to4000
analysis/qnn_expval_delta_k21_noresid_boundary_to4000_mech
analysis/qnn_expval_delta_k21_noresid_boundary_to4000_exhaustive_best
analysis/qnn_expval_delta_k21_noresid_boundary_to4000_delta_head
```

This continuation lowered LR to `1e-4`, evaluated every `100` steps, increased
adjacent-margin pressure, kept alias pressure mild, oversampled boundary
regions more strongly, and restored the best checkpoint by validation accuracy.
It was configured for `4000` total steps but stopped early at completed step
`2900` because validation failed to improve for six evaluations after step
`2300`. The restored checkpoint is step `2300`.

Training trajectory:

| step | held-out exact | no-wrap | wrap | train exact |
| ---: | ---: | ---: | ---: | ---: |
| resumed source step 2000 | 0.979505 | 0.978358 | 0.980709 | 0.996456 |
| step 2100 | 0.979505 | 0.979247 | 0.979776 | 0.997874 |
| step 2200 | 0.979201 | 0.977468 | 0.981021 | 0.998228 |
| step 2300 best | 0.981327 | 0.981619 | 0.981021 | 0.997165 |
| step 2400 | 0.979809 | 0.983398 | 0.976042 | 0.995039 |
| step 2500 | 0.979961 | 0.981322 | 0.978531 | 0.996102 |
| step 2600 | 0.981327 | 0.980136 | 0.982576 | 0.997519 |
| step 2700 | 0.978594 | 0.978950 | 0.978220 | 0.997519 |
| step 2800 | 0.976317 | 0.977172 | 0.975420 | 0.996811 |
| step 2900 early stop | 0.979809 | 0.977468 | 0.982265 | 0.998937 |

The continuation improved the QNN best from `0.979505` to `0.981327`, reducing
held-out errors from `135` to `123`. This is a real but smaller gain than the
previous rescue stage. It is still not grokked to the `0.99` target; reaching
`0.99` now requires fixing about `58` more held-out examples.

Signed-offset audit:

| split | errors | `+1` errors | `-1` errors | adjacent-error share | nonlocal `|offset|>2` |
| --- | ---: | ---: | ---: | ---: | ---: |
| held-out | 123 | 67 | 52 | 0.967480 | 4 |
| no-wrap | 62 | 31 | 28 | 0.951613 | 3 |
| wrap | 61 | 36 | 24 | 0.983607 | 1 |
| `a+b <= 50` | 16 | 9 | 4 | 0.812500 | 3 |
| `a=0` or `b=0` | 9 | 7 | 2 | 1.000000 | 0 |
| `a=96` or `b=96` | 7 | 1 | 6 | 1.000000 | 0 |

The continuation improved both operand-boundary slices: `a=0` or `b=0` rose
from `0.908397` to `0.931298`, and `a=96` or `b=96` rose from `0.933333` to
`0.948148`. It also reduced adjacent errors overall. The tradeoff is that
nonlocal held-out errors increased from `2` to `4`, with confident failures at
`(15,34)->14`, `(19,84)->26`, `(45,3)->85`, and `(15,35)->14`.

Component and frequency diagnostics:

| diagnostic | value |
| --- | ---: |
| full checkpoint held-out exact | 0.981327 |
| delta-only held-out exact | 0.982693 |
| residual-only held-out exact | 0.010172 |
| held-out within +/-1 | 0.999393 |
| held-out top-2 | 0.999848 |
| held-out ECE | 0.002990 |
| full-logit modular residual R2 | 0.541316 |
| feature addition-diagonal FFT energy | 0.328956 |
| frozen generic linear readout | 0.832853 |
| frozen post-hoc delta readout | 0.268407 |

The mechanism did not qualitatively change. The deployed answer rule is still
the trained Fourier-delta path; `delta_only` again beats the full checkpoint
slightly, and the residual-only path is chance. Frequency cutoffs show that
`k<=13` already reaches `0.981023`, while `k<=21` reaches `0.981327`. The
remaining errors are therefore not a low-frequency-only failure; they are local
and a few persistent alias/margin failures in the co-adapted Fourier-delta
readout.

## Dirac Delta Branch

Detailed plan and first results:

```text
QNN_DIRAC_DELTA_HEAD_PLAN.md
RESULTS_QNN_DIRAC_DELTA.md
```

The new branch `qnn-dirac-delta-head` adds a `DiracDeltaHead` readout. In
strict mode, it normalizes every predicted Fourier pair to the unit circle and
scores candidate residues with a finite modular delta kernel.

The Dirac branch now has one successful preservation run and three failed
phase-only diagnostics:

| run | held-out exact | interpretation |
| --- | ---: | --- |
| strict unit `k<=21` Dirac smoke | 0.204342 | hard unit-phase projection is not a drop-in replacement; the grokked QNN still uses coefficient magnitude/calibration. |
| identity bridge smoke | 0.982693 | new head code path preserves the grokked Fourier-delta geometry and slightly improves the previous best. |
| full identity bridge | 0.982693 | early-stopped at completed step 700 and restored the step-1 checkpoint; this became the best saved QNN object at that stage. |
| `soft_unit` bridge from identity | 0.206923 | bounding each per-frequency magnitude by `tanh(raw_norm)` collapses the answer rule. |
| `global_soft_unit` bridge from identity | 0.204190 | allowing only one confidence scalar per example also collapses the answer rule. |

The full identity bridge has `114` held-out errors: `61` are `+1`, `49` are
`-1`, and only `4` are nonlocal. Its raw Fourier-pair norms have mean
`4.184538`, median `3.252236`, p95 `11.550005`, and zero near-zero pairs.
That explains the failure of strict phase projection: the learned QNN solution
is cyclic, but its residue selection depends on per-frequency coefficient
magnitudes and learned kernel calibration.

We then tested the corrected interpretation: the Dirac component should sharpen
boundaries, not replace the cyclic rule. The implemented `dirac_residual_sharpen`
readout freezes the copied Fourier-delta base and trains only `22`
`head.sharpen_*` parameters:

```text
runs/modular_addition_qnn_mod97_dirac_residual_sharpen_k21_from_identity
analysis/qnn_dirac_residual_sharpen_k21_mech
analysis/qnn_dirac_residual_sharpen_k21_exhaustive
```

It reached `0.982997` held-out exact, improving the identity bridge from `114`
held-out errors to `112`. The residual fixed two `+1` errors, `(39,15)->54`
and `(28,25)->53`, and introduced zero new held-out errors. The exhaustive
component audit confirms the intended role:

| component | held-out exact |
| --- | ---: |
| full | 0.982997 |
| base full | 0.982693 |
| correction only | 0.000152 |
| delta only | 0.983149 |
| full without correction | 0.982693 |

The correction is not an independent arithmetic mechanism. It is a small local
boundary adjustment over an already-grokked cyclic Fourier-delta rule.

## Guarded Post-Hoc Correction

Detailed reports:

```text
analysis/qnn_guarded_correction/QNN_GUARDED_CORRECTION_REPORT.md
analysis/qnn_guarded_correction_alias_candidates/QNN_GUARDED_CORRECTION_REPORT.md
```

We then tested the planned micro-correction/calibration pass over frozen
Fourier-delta logits. The analyzer compared:

- frozen base variants: `full`, `base_full`, `delta_only`, and `delta_plus_bias`;
- global temperature/scale plus class-bias calibration;
- low-margin or boundary-low-margin gated scale/bias calibration;
- a tiny offset head that could only add scores to a bounded candidate set around
  the frozen prediction.

The guardrail was explicit: corrections were gated by the frozen margin, and
high-margin predictions were tracked separately. The best clean results were:

| checkpoint | best frozen base | best correction | corrected held-out | gain vs best frozen | fixed | broken | high-margin changed | nonlocal errors |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4k boundary continuation | `delta_only` 0.982693 | guarded scale/bias on `full` | 0.983756 | 0.001063 | 20 | 4 | 0 | 4 |
| Dirac identity bridge | `delta_only` 0.983149 | scale/bias on `delta_only` | 0.984060 | 0.000911 | 14 | 8 | 0 | 4 |
| Dirac residual sharpener | `delta_only` 0.983149 | low-margin guarded scale/bias on `delta_only` | 0.984060 | 0.000911 | 14 | 8 | 0 | 4 |

The alias-candidate diagnostic additionally allowed offsets
`-36,-35,-2,-1,0,1,2,20,37`, chosen to cover the known nonlocal displacement
families. It did not improve over ordinary guarded calibration. This is the key
negative result: the easy remaining gain is a small calibration gain, not a
learned alias repair.

Interpretation: frozen Fourier-delta logits still contain useful margin
structure, and a guarded calibrator can recover six net held-out examples for
the best current checkpoint without changing high-confidence predictions. But
the four nonlocal aliases persist, and the corrected accuracy `0.984060` remains
well below `0.99`. The remaining errors likely require a representation-level
change or seed/modulus evidence that the aliases are not stable.

## No-Residual Boundary Seed Sweep

Detailed sweep report:

```text
analysis/qnn_noresid_seed_sweep/NORESID_SEED_SWEEP_SUMMARY.md
```

We then tested whether the no-residual boundary recipe's remaining failures
were stable across continuation seeds. This is a warm-start continuation sweep
from the existing native `k<=21` checkpoint, not a fully independent
source-checkpoint seed sweep. It therefore tests optimization sensitivity of
the late boundary-sharpening phase, not full training robustness from scratch.

The sweep compared two Fourier cutoffs:

- `k<=13`, because earlier cutoff diagnostics showed that most useful exact
  sharpening arrived by the mid-frequency band;
- `k<=21`, because the strongest no-residual checkpoint had used that cutoff.

Per-run results:

| cutoff | seed | best step | held-out exact | held-out errors | train exact | wrap | no-wrap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `k<=13` | 0 | 2500 | 0.977076 | 151 | 0.992913 | 0.980398 | 0.973910 |
| `k<=13` | 1 | 2500 | 0.978442 | 142 | 0.996456 | 0.978843 | 0.978061 |
| `k<=13` | 2 | 2000 | 0.982541 | 115 | 0.996456 | 0.981954 | 0.983101 |
| `k<=21` | 0 | 2000 | 0.979505 | 135 | 0.996456 | 0.980709 | 0.978358 |
| `k<=21` | 1 | 2250 | 0.976317 | 156 | 0.993622 | 0.969197 | 0.983101 |
| `k<=21` | 2 | 2500 | 0.975254 | 163 | 0.994330 | 0.972931 | 0.977468 |

Aggregate:

| cutoff | seeds | held-out mean | held-out std | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `k<=13` | 3 | 0.979353 | 0.002844 | 0.977076 | 0.982541 |
| `k<=21` | 3 | 0.977025 | 0.002212 | 0.975254 | 0.979505 |

The result shifts the interpretation slightly. The broader `k<=21` head is not
clearly better in the late no-residual boundary phase. `k<=13` has the better
mean and best seed, suggesting that the remaining errors are not solved by
simply adding more Fourier terms. The failure mode is more like calibration and
alias control within a mostly correct cyclic rule.

Because all six runs share the same upstream source checkpoint, this is not a
complete robustness result. It is still useful: the 115-163 error range shows
that the final margin/alias pattern is seed-sensitive enough that individual
nonlocal aliases should not yet be over-interpreted as fixed architectural
impossibilities.

## Layerwise Dirac Fourier Tests

Detailed report:

```text
RESULTS_QNN_LAYERWISE_DIRAC.md
analysis/qnn_layerwise_dirac_exhaustive/QNN_LAYERWISE_DIRAC_ARCHITECTURE_COMPARISON.md
```

The layerwise Dirac/Fourier tests are the main architectural intervention in
the QNN branch. Earlier Fourier-delta rescues proved that a final finite-delta
head can sharpen a low-frequency cyclic scaffold. The layerwise question is
stronger:

> Can every intermediate reuploading layer be trained to expose a usable
> finite cyclic delta over `(a+b-c) mod p`?

Let `z_l(a,b)` be the measured feature vector after circuit layer `l`. A
layerwise head produces:

```latex
\ell^{(\ell)}(a,b)=H_\ell(z_\ell(a,b)).
```

Each head has the same finite Fourier-delta target:

```latex
\ell^{(\ell)}_c(a,b)
\approx
\sum_{k=1}^{K}\rho_{\ell k}
\cos\left({2\pi k((a+b)-c)\over p}\right).
```

This matters for three reasons. First, it gives every reuploading layer a
direct residue-shaped gradient. Second, it makes the layerwise arithmetic
trajectory measurable. Third, it allows evidence from multiple partially
correct cyclic estimates to be combined.

The averaging intervention is the cleanest version of that third idea. Given
layerwise logits `l^(1),...,l^(L)`, the fixed mean deploys:

```latex
\ell_{\text{mean}}
= {1\over L}\sum_{\ell=1}^{L}\ell^{(\ell)}.
```

This is a logit average, equivalent to a log-opinion pool:

```latex
softmax(\ell_{\text{mean}})_c
\propto
\left[
\prod_{\ell=1}^{L}\exp(\ell^{(\ell)}_c)
\right]^{1/L}.
```

Thus classes supported across layers are reinforced, while layer-specific alias
errors can cancel. In margin terms:

```latex
m^{\text{mean}}_c
= {1\over L}\sum_{\ell=1}^{L}
\left(\ell^{(\ell)}_y-\ell^{(\ell)}_c\right).
```

The mean can fix a final-layer error when the final margin is negative but the
average layerwise margin is positive. This is exactly what the post-hoc
auxiliary analysis suggested: the final head made errors that earlier heads did
not always share.

Three from-scratch `prob_head` variants were first run for 2500 steps:

| run | readout type | held-out exact | train exact | wrap | no-wrap |
| --- | --- | ---: | ---: | ---: | ---: |
| layerwise auxiliary | `layerwise_dirac_aux` | 0.951571 | 0.976612 | 0.930305 | 0.971835 |
| layerwise adapter | `layerwise_dirac_adapter` | 0.874601 | 0.933735 | 0.818295 | 0.928254 |
| layerwise residual | `layerwise_dirac_residual` | 0.968119 | 0.983345 | 0.949284 | 0.986066 |

The layerwise residual run is the strongest scratch result in this family,
though it does not beat the warm-start no-residual/Dirac-sharpener results.
Its all-pairs mech interp shows held-out `0.968119`, mod-add residual R2
`0.649418`, feature addition-diagonal energy `0.098710`, and a k=1 sum phase
that is only `0.211477` exact but `0.981479` within five residues. Its held-out
errors are mostly local: `169` are signed `-1`, `39` are signed `+1`, and only
two nonlocal offset-34 errors remain.

The learned layerwise-residual softmax weights are:

```text
[0.063517, 0.060813, 0.074647, 0.801023]
```

So the model is still final-layer dominated, with earlier Dirac/Fourier logits
contributing about 20% of the final answer. A frozen post-hoc readout over the
best checkpoint's final features is much weaker: refreshed linear readout gets
`0.527402`, and the best frozen Fourier-delta readout gets `0.533475`. The
deployed result therefore comes from end-to-end co-adaptation of the circuit
and layerwise heads, not from a cleanly reusable frozen final feature basis.

The adapter feedback variant is the negative result. Feeding layerwise
Dirac/Fourier coefficients back into later input rotations degraded held-out
accuracy and widened the wrap/no-wrap gap. For this circuit, Dirac/Fourier
structure is useful as deep supervision and residual answer scaffolding, not
as an unqualified recurrent control signal.

The exhaustive layerwise pass adds two important details:

| architecture | layer 0 | layer 1 | layer 2 | layer 3 | final | uniform layer mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auxiliary | 0.587673 | 0.864582 | 0.940185 | 0.964172 | 0.951571 | 0.975558 |
| adapter | 0.412783 | 0.628966 | 0.782602 | 0.868073 | 0.874601 | 0.900258 |
| residual | 0.449370 | 0.683771 | 0.843935 | 0.964627 | 0.968119 | 0.966297 |

First, the auxiliary architecture contains a better layer-ensemble answer rule
than its deployed final head: uniform averaging of the four layerwise logits
gets `0.975558` held-out exact. Second, the adapter path is causally used but
comparatively harmful: evaluating the trained adapter model with adapter
feedback disabled collapses held-out exact accuracy to `0.135115`, yet the
architecture is still much worse than auxiliary or residual training.

Frequency cutoffs tell the same story as the earlier QNN rescues. `k<=5` gives
high near-answer accuracy but poor exact accuracy; exact class selection needs
mid-frequency sharpening through about `k<=8` to `k<=13`. For the residual
architecture, layer 3 reaches `0.637620` exact and `0.991802` within-one at
`k<=5`, then `0.942311` exact at `k<=8` and `0.967664` at `k<=13`.

## Same-Sum Invariance Objective

Detailed report:

```text
analysis/qnn_same_sum_sweep/QNN_SAME_SUM_SWEEP_SUMMARY.md
```

The next objective-level intervention was to add explicit same-sum invariance
to the layerwise QNN families. This mirrors the successful JEPA
`group_contrastive` fix: examples related by
`(a,b) -> ((a+d) mod p, (b-d) mod p)` and by operand swap should carry the same
answer. The model is therefore penalized when same-sum views produce different
answer distributions.

For logits `l(x)` and same-sum view `l(x')`, the added consistency term is:

```latex
\mathcal{L}_{\text{same}}
= {1\over 2}
\left[
\operatorname{KL}(p_\tau(\ell(x)) \Vert p_\tau(\ell(x')))
+
\operatorname{KL}(p_\tau(\ell(x')) \Vert p_\tau(\ell(x)))
\right],
```

plus a small CE term on the same-sum view. The training objective is:

```latex
\mathcal{L}
= \operatorname{CE}(\ell(x),y)
+ \lambda_{\text{view}}\operatorname{CE}(\ell(x'),y)
+ \lambda_{\text{same}}\mathcal{L}_{\text{same}}
+ \lambda_{\text{layer}}\mathcal{L}_{\text{layer}}
+ \lambda_{\text{margin}}\mathcal{L}_{\text{neighbor}}.
```

The first pass used `p=97`, data split seed `0`, model seed `0`, and the same
30% train split as the rest of the branch. Controls are the existing
publication-sweep seed-0 runs.

Comparison against controls:

| architecture | control held-out | same-sum held-out | gain | control errors | same-sum errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| auxiliary/control | 0.959769 | 0.975406 | +0.015637 | 265 | 162 |
| fixed mean | 0.973736 | 0.980871 | +0.007135 | 173 | 126 |
| learned residual | 0.931228 | 0.975254 | +0.044026 | 453 | 163 |

Full run status:

| architecture | kind | best step | held-out | train | wrap | no-wrap | consistency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| auxiliary/control | control | 2250 | 0.959769 | 0.977321 | 0.943684 | 0.975096 | 0.000000 |
| auxiliary/control | same-sum | 2500 | 0.975406 | 0.982636 | 0.976665 | 0.974207 | 0.015605 |
| fixed mean | control | 2250 | 0.973736 | 0.990432 | 0.961419 | 0.985473 | 0.000000 |
| fixed mean | same-sum | 2500 | 0.980871 | 0.992204 | 0.983199 | 0.978654 | 0.012734 |
| learned residual | control | 2500 | 0.931228 | 0.954288 | 0.931549 | 0.930922 | 0.000000 |
| learned residual | same-sum | 2250 | 0.975254 | 0.984408 | 0.973864 | 0.976579 | 0.012109 |

This is a strong positive result. Same-sum invariance improves every tested
layerwise architecture, with the largest gain in the learned-residual family.
The fixed-mean plus same-sum model is the best seed-0 from-scratch layerwise
QNN in this set at `0.980871`, just below the best warm-start Dirac/QNN
checkpoints and above the original layerwise residual scratch run.

Mechanistically, this supports the current theory: the QNN is not merely
learning an arbitrary classifier over ordered pairs. It benefits when the
objective explicitly identifies the group orbit:

```latex
\{(a+d,b-d): d\in Z_p\}.
```

That is the same equivalence class that produces addition-diagonal Fourier
energy. The QNN still needs finite-delta sharpening to separate adjacent
residues, but same-sum pressure makes the cyclic group-law representation more
stable.

## Publication Readiness Package

The branch now includes the planned publication-readiness infrastructure:

```text
QNN_PUBLICATION_READINESS_PLAN.md
QNN_DATA_REUPLOADING_FOURIER_THEORY.md
modular_addition/qnn_publication_sweeps.py
modular_addition/run_qnn_publication_subset.py
modular_addition/classical_fourier_baseline.py
modular_addition/make_qnn_publication_figures.py
modular_addition/qnn_noresid_seed_sweep.py
modular_addition/qnn_same_sum_sweep.py
analysis/qnn_publication_sweep/COMMAND_MANIFEST.md
analysis/qnn_publication_sweep/SWEEP_AGGREGATE.md
analysis/qnn_split_ratio_sweep/SWEEP_AGGREGATE.md
analysis/qnn_noresid_seed_sweep/NORESID_SEED_SWEEP_SUMMARY.md
analysis/qnn_same_sum_sweep/QNN_SAME_SUM_SWEEP_SUMMARY.md
analysis/qnn_layerwise_mean_sweep/MEAN_SWEEP_COMPARISON.md
figures/qnn_publication/QNN_PUBLICATION_STORY.md
```

The sweep manifest covers `p=31`, `p=97`, and `p=127`, seeds `0,1,2`, the
three layerwise QNN architectures, and five classical Fourier baselines. The
five most informative families were run across all three moduli and seeds,
completing `45/45` selected runs:

| family | p=31 mean | p=97 mean | p=127 mean | interpretation |
| --- | ---: | ---: | ---: | --- |
| QNN auxiliary | 0.369490 | 0.941248 | 0.957252 | strong at 97/127, weak at 31 despite high train accuracy. |
| QNN adapter | 0.272412 | 0.921411 | 0.972397 | weakest at 31; best QNN mean at 127. |
| QNN residual | 0.375433 | 0.945549 | 0.964751 | best QNN mean at 97; strong at 127. |
| QNN fixed layerwise mean | 0.394750 | 0.975305 | 0.983438 | best QNN family at 97/127; still weak at 31. |
| matched Fourier-delta baseline | 0.022784 | 0.863165 | 0.990966 | weak at 31, good at 97, near-perfect at 127. |
| product-Fourier delta upper bound | 1.000000 | 1.000000 | 1.000000 | explicit interaction-basis sanity check. |

The product-Fourier baseline is intentionally not treated as a fair learned
baseline; it verifies that the Fourier-delta readout can solve the finite group
law when supplied with the correct interaction basis. The matched
Fourier-delta baseline is the fairer classical comparison: it is weaker than
the QNNs at `p=97`, but nearly perfect at `p=127`. The result should therefore
be framed as a representational and inductive-bias comparison, not as evidence
for quantum advantage.

The fixed layerwise mean result is the clearest positive follow-up from the
post-hoc layer averaging observation. Training the uniform average directly
beats the auxiliary, adapter, and learned-residual QNN families at `p=97` and
`p=127`, with held-out means `0.975305` and `0.983438`.

### Train-Ratio And Split-Seed Robustness

We also ran the robustness test that varied train fraction and train/test split
membership. This sweep used model seed `0`, split seeds `0,1,2`, train
fractions `0.50` and `0.70`, and moduli `31`, `97`, and `127`. It compared
the fixed layerwise mean, learned residual, and matched Fourier-delta control.
All `54/54` runs completed.

| family | p=31 train 50 | p=31 train 70 | p=97 train 50 | p=97 train 70 | p=127 train 50 | p=127 train 70 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QNN fixed layerwise mean | 0.651421 | 0.880046 | 0.990719 | 0.994332 | 0.996280 | 0.987050 |
| QNN learned residual | 0.683992 | 0.860438 | 0.978250 | 0.981108 | 0.966770 | 0.967349 |
| matched Fourier-delta baseline | 0.022869 | 0.018454 | 0.995466 | 0.999646 | 0.999711 | 0.999931 |

This gives two important checks. First, `p=97` is not a split fluke: the fixed
layerwise mean improves from `0.975305` at train fraction `0.30` to `0.990719`
at `0.50` and `0.994332` at `0.70`. Second, the weak `p=31` result at train
fraction `0.30` is partly a coverage problem: with more observed table entries,
the QNN mean reaches `0.880046` and the residual reaches `0.860438`. The
remaining limitation is that this sweep fixes the model seed at `0`; it tests
split and train-ratio robustness, not the full Cartesian product of model
seeds, split seeds, and train ratios.

### Remaining Tests Implementation

The remaining robustness work is now converted into an executable plan:

```text
QNN_REMAINING_TESTS_PLAN.md
modular_addition/qnn_remaining_tests.py
modular_addition/run_qnn_remaining_tests.py
analysis/qnn_remaining_tests/QNN_REMAINING_TESTS_MANIFEST.md
analysis/qnn_remaining_tests/remaining_tests_status.csv
analysis/qnn_remaining_tests/pending_commands.ps1
```

The manifest currently has `333` rows, with `94` complete and `239` pending.
The high-priority pending work is the full Cartesian robustness grid over
moduli `31/97/127`, train fractions `0.30/0.50/0.70`, split seeds `0/1/2`,
model seeds `0/1/2`, fixed layerwise mean, learned residual, and matched
Fourier-delta controls. The tool also adds the missing raw numeric MLP baseline
and generates same-sum and optional-modulus follow-ups.

## Literature Context

The circuit family follows the data-reuploading idea from Perez-Salinas et al.,
where classical inputs are repeatedly encoded into a variational circuit to
increase expressivity:

```text
https://arxiv.org/abs/1907.02085
```

The hybrid feature-map framing is related to quantum feature-space and kernel
views of quantum machine learning:

```text
https://www.nature.com/articles/s41586-019-0980-2
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.040504
```

In branch terms, these references motivate treating the simulated quantum state
as a learned representation, then asking whether modular addition becomes
linearly or measurement-readout accessible. The current answer is format-
dependent: strict Born and frozen analytic readouts remain weak, generic linear
readouts reveal substantial hidden signal, and an end-to-end trained
Fourier-delta readout makes the representation strongly algorithmic but still
not perfectly grokked.

## Artifacts

- Run directory: `runs\modular_addition_qnn_mod97`
- 30k direct-aux run directory: `runs\modular_addition_qnn_mod97_direct_aux_30k`
- 30k mech-interp report: `analysis\qnn_mod97_mech\QNN_MOD97_MECH_INTERP_REPORT.md`
- frozen delta-head report: `analysis\qnn_delta_head\QNN_DELTA_HEAD_REPORT.md`
- 10k expval delta run directory: `runs\modular_addition_qnn_mod97_expval_delta_10k`
- 10k expval delta mech report: `analysis\qnn_expval_delta_10k_mech\QNN_MOD97_MECH_INTERP_REPORT.md`
- 10k expval delta frozen-readout report: `analysis\qnn_expval_delta_10k_delta_head\QNN_DELTA_HEAD_REPORT.md`
- 10k expval delta exhaustive report: `analysis\qnn_expval_delta_10k_exhaustive\QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md`
- 10k-to-30k precision run directory: `runs\modular_addition_qnn_mod97_expval_delta_10k_to_30k_precision`
- 10k-to-30k precision mech report: `analysis\qnn_expval_delta_10k_to_30k_precision_mech\QNN_MOD97_MECH_INTERP_REPORT.md`
- 10k-to-30k precision frozen-readout report: `analysis\qnn_expval_delta_10k_to_30k_precision_delta_head\QNN_DELTA_HEAD_REPORT.md`
- 10k-to-30k precision best exhaustive report: `analysis\qnn_expval_delta_10k_to_30k_precision_exhaustive_best\QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md`
- 10k-to-30k precision final exhaustive report: `analysis\qnn_expval_delta_10k_to_30k_precision_exhaustive_final\QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md`
- native k<=21 config: `configs\modular_addition_qnn_mod97_expval_delta_k21_native.yaml`
- native k<=13 control config: `configs\modular_addition_qnn_mod97_expval_delta_k13_native.yaml`
- native k<=21 run directory: `runs\modular_addition_qnn_mod97_expval_delta_k21_native`
- native k<=21 stop note: `runs\modular_addition_qnn_mod97_expval_delta_k21_native\STOPPED_FOR_EARLY_STOPPING.md`
- native k<=21 mech report: `analysis\qnn_expval_delta_k21_native_mech\QNN_MOD97_MECH_INTERP_REPORT.md`
- native k<=21 exhaustive report: `analysis\qnn_expval_delta_k21_native_exhaustive_best\QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md`
- native k<=21 signed-offset eval: `analysis\qnn_expval_delta_k21_native_exhaustive_best\signed_offset_eval.md`
- native k<=21 frozen-readout report: `analysis\qnn_expval_delta_k21_native_delta_head\QNN_DELTA_HEAD_REPORT.md`
- no-residual boundary config: `configs\modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary.yaml`
- no-residual boundary run: `runs\modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_norefresh`
- no-residual boundary mech report: `analysis\qnn_expval_delta_k21_noresid_boundary_mech\QNN_MOD97_MECH_INTERP_REPORT.md`
- no-residual boundary exhaustive report: `analysis\qnn_expval_delta_k21_noresid_boundary_exhaustive_best\QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md`
- no-residual boundary signed-offset eval: `analysis\qnn_expval_delta_k21_noresid_boundary_exhaustive_best\signed_offset_eval.md`
- no-residual boundary frozen-readout report: `analysis\qnn_expval_delta_k21_noresid_boundary_delta_head\QNN_DELTA_HEAD_REPORT.md`
- no-residual k<=13/k<=21 seed-sweep runner: `modular_addition\qnn_noresid_seed_sweep.py`
- no-residual k<=13/k<=21 seed-sweep report: `analysis\qnn_noresid_seed_sweep\NORESID_SEED_SWEEP_SUMMARY.md`
- no-residual k<=13 seed-sweep configs: `configs\modular_addition_qnn_mod97_expval_delta_k13_noresid_boundary_seed*.yaml`
- no-residual k<=21 seed-sweep configs: `configs\modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_seed*.yaml`
- 4k continuation config: `configs\modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_to4000.yaml`
- 4k continuation run: `runs\modular_addition_qnn_mod97_expval_delta_k21_noresid_boundary_to4000`
- 4k continuation mech report: `analysis\qnn_expval_delta_k21_noresid_boundary_to4000_mech\QNN_MOD97_MECH_INTERP_REPORT.md`
- 4k continuation exhaustive report: `analysis\qnn_expval_delta_k21_noresid_boundary_to4000_exhaustive_best\QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md`
- 4k continuation signed-offset eval: `analysis\qnn_expval_delta_k21_noresid_boundary_to4000_exhaustive_best\signed_offset_eval.md`
- 4k continuation frozen-readout report: `analysis\qnn_expval_delta_k21_noresid_boundary_to4000_delta_head\QNN_DELTA_HEAD_REPORT.md`
- Dirac-delta plan: `QNN_DIRAC_DELTA_HEAD_PLAN.md`
- Dirac-delta results: `RESULTS_QNN_DIRAC_DELTA.md`
- Dirac identity bridge smoke: `runs\_smoke_qnn_dirac_delta_identity_bridge`
- Dirac identity bridge exhaustive report: `analysis\qnn_dirac_delta_identity_bridge_smoke_exhaustive\QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md`
- Dirac identity bridge full run: `runs\modular_addition_qnn_mod97_dirac_delta_k21_identity_bridge_from_grokked`
- Dirac identity bridge full exhaustive report: `analysis\qnn_dirac_delta_k21_identity_bridge_exhaustive\QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md`
- Dirac identity bridge signed-offset eval: `analysis\qnn_dirac_delta_k21_identity_bridge_exhaustive\signed_offset_eval.md`
- Dirac soft bridge smoke: `runs\_smoke_qnn_dirac_delta_soft_bridge_from_identity`
- Dirac global-confidence smoke: `runs\_smoke_qnn_dirac_delta_global_soft_from_identity`
- Dirac residual sharpener config: `configs\modular_addition_qnn_mod97_dirac_residual_sharpen_k21_from_identity.yaml`
- Dirac residual sharpener run: `runs\modular_addition_qnn_mod97_dirac_residual_sharpen_k21_from_identity`
- Dirac residual sharpener mech report: `analysis\qnn_dirac_residual_sharpen_k21_mech\QNN_MOD97_MECH_INTERP_REPORT.md`
- Dirac residual sharpener exhaustive report: `analysis\qnn_dirac_residual_sharpen_k21_exhaustive\QNN_EXPVAL_DELTA_10K_EXHAUSTIVE_REPORT.md`
- Dirac residual sharpener changed cases: `analysis\qnn_dirac_residual_sharpen_k21_exhaustive\residual_sharpen_changed_cases.json`
- Guarded correction analyzer: `modular_addition\qnn_guarded_correction.py`
- Guarded correction report: `analysis\qnn_guarded_correction\QNN_GUARDED_CORRECTION_REPORT.md`
- Alias-candidate guarded correction report: `analysis\qnn_guarded_correction_alias_candidates\QNN_GUARDED_CORRECTION_REPORT.md`
- Layerwise Dirac/Fourier results: `RESULTS_QNN_LAYERWISE_DIRAC.md`
- Layerwise Dirac/Fourier configs: `configs\modular_addition_qnn_mod97_layerwise_dirac_*_scratch.yaml`
- Layerwise Dirac/Fourier runs: `runs\modular_addition_qnn_mod97_layerwise_dirac_*_scratch`
- Layerwise Dirac/Fourier analysis: `analysis\qnn_layerwise_dirac`
- Layerwise Dirac/Fourier exhaustive analysis: `analysis\qnn_layerwise_dirac_exhaustive`
- same-sum layerwise QNN runner: `modular_addition\qnn_same_sum_sweep.py`
- same-sum layerwise QNN configs: `configs\modular_addition_qnn_mod97_same_sum_*_seed0.yaml`
- same-sum layerwise QNN runs: `runs\modular_addition_qnn_mod97_same_sum_*_seed0`
- same-sum layerwise QNN report: `analysis\qnn_same_sum_sweep\QNN_SAME_SUM_SWEEP_SUMMARY.md`
- Publication-readiness plan: `QNN_PUBLICATION_READINESS_PLAN.md`
- Data-reuploading Fourier theory note: `QNN_DATA_REUPLOADING_FOURIER_THEORY.md`
- Publication sweep manifest: `analysis\qnn_publication_sweep\COMMAND_MANIFEST.md`
- Publication sweep aggregate: `analysis\qnn_publication_sweep\SWEEP_AGGREGATE.md`
- Train-ratio and split-seed aggregate: `analysis\qnn_split_ratio_sweep\SWEEP_AGGREGATE.md`
- Train-ratio and split-seed status: `analysis\qnn_split_ratio_sweep\sweep_status.csv`
- Remaining-test plan: `QNN_REMAINING_TESTS_PLAN.md`
- Remaining-test manifest: `analysis\qnn_remaining_tests\QNN_REMAINING_TESTS_MANIFEST.md`
- Remaining-test status: `analysis\qnn_remaining_tests\remaining_tests_status.csv`
- Layerwise mean comparison: `analysis\qnn_layerwise_mean_sweep\MEAN_SWEEP_COMPARISON.md`
- Publication figures: `figures\qnn_publication`
- Exhaustive analyzer: `modular_addition\analyze_qnn_delta_rescue.py`
- Config: `runs\modular_addition_qnn_mod97\config.yaml`
- Summary: `runs\modular_addition_qnn_mod97\summary.json`
- Suspension note: `runs\modular_addition_qnn_mod97\SUSPENDED.md`
- 6000-step resume note: `runs\modular_addition_qnn_mod97\RESUMED.md`
- 10000-step resume note: `runs\modular_addition_qnn_mod97\RESUME_10000.md`
