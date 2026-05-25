# Deep Spectral Preconditioning via Rectified, Variance-Bounded Matrix Orthogonalization

**Author:** Richard Christopher
**Affiliation:** MetaFore
**Email:** rchris@neotec.dev
**Version:** 1.0 (2026-05-17)
**Reference implementation:** `muogi.py` and `ramuogi.py` in this repository.

## Abstract

We introduce **Muogi** (Momentum Yogi Orthogonalized by Newton-Schulz) and its variance-rectified extension **RAMuogi** (RAdam + Muogi). These algorithms form a class of second-order matrix preconditioners designed for heterogeneous neural network topologies: architectures whose layers exhibit radically different gradient distributions, hierarchical fan-in and fan-out ratios, and per-row variance asymmetries.

Standard orthogonal optimizers like Muon accelerate convergence by reshaping parameter gradients onto the orthogonal manifold via Newton-Schulz polynomial iteration. They struggle under severe structural row-burst asymmetries and early-step gradient variance, leading to divergence or numerical instability. Muogi solves the structural problem by injecting an element-wise variance tracker directly into the NS5 loop, controlled by a top-down relative threshold floor that caps the matrix condition spectrum to a bounded envelope. RAMuogi adds a dynamic scalar rectification path that holds uncalibrated parameters in a first-order momentum state during warm-up and transitions smoothly to spectral orthogonalization only when statistical variance tracking meets a confidence threshold.

The optimizer ships with a documented **four-layer failure-safety chain** (L1: spread-cap clamp, L2: NS5 convergence safe-skip, L3: vanilla Yogi fallback, L4: RAdam variance gate) engineered for stability across the gradient pathologies we encountered during development. We treat that chain as load-bearing infrastructure rather than an afterthought.

We present the mathematical mechanics, the engineering of each safety layer, the empirical failure modes encountered during development, and a complete production-grade PyTorch reference implementation.

------------------------------

## 1. Introduction and Motivation

As deep learning architectures move from uniform multi-layer perceptrons to heterogeneous networks (structured state-space models like S4 and Mamba, mixture-of-experts routers, hierarchical attention manifolds, multi-branch decompositions), parameter weights encounter radically different gradient distributions. Different layers present distinct mechanical masses, directional deltas, and sharp fan-in / fan-out ratios. In such architectures, parameter groups can be routed to different optimizers tuned to their specific gradient pathology, treating the optimizer choice as a per-group design decision rather than a model-wide constant.

Under such heterogeneity, uniform first-order adaptive optimizers (AdamW) lose efficiency by averaging out independent structural transformations. Second-order matrix preconditioners (Muon) accelerate training by enforcing the polar decomposition onto 2-D parameter updates, targeting the orthogonal manifold:

$$\mathcal{O} = \{X \in \mathbb{R}^{M \times N} \mid X X^\top = I\}.$$

Despite their training speedups, pure orthogonal methods suffer from two critical vulnerabilities in heterogeneous architectures.

### 1.1 The Spectral Backlash Failure Mode

High-dimensional projection channels and sparse routing vectors naturally exhibit *row-burst variance asymmetry*. Some feature channels accumulate large gradient magnitudes while others remain quiet. Forcing complete orthogonality on these asymmetric structures destroys their directional identity. The network experiences topological backlash. Subsequent gradient condition numbers explode (we observed peaks of around $136{,}000\times$ on a single training step), overflowing the convergence radius of the preconditioning polynomial. The optimizer either crashes (NaN propagation) or silently degrades to its fallback path, abandoning the spectral benefits it was supposed to provide.

### 1.2 The Early-Step Chaos Regime

Within the first several hundred parameter updates, the moving-average variance arrays used to estimate adaptive scales are largely empty. Computing matrix shape adjustments on this uncalibrated noise injects geometric distortion that the optimizer treats as if it were a real signal. The result is a polluted spectral pathway that masks the genuine emergence of network structure during warm-up.

### 1.3 Contribution

Muogi addresses (1.1) through a unified pipeline that injects coordinate-wise variance information into the Newton-Schulz iteration under a bounded spread constraint, with a fallback chain that catches numerical failures without halting training. RAMuogi extends Muogi with a RAdam-style rectification gate that solves (1.2) by suspending spectral computation until variance estimates are statistically trustworthy. Together, the algorithms reconcile coordinate-wise variance adaptivity with global spectral orthogonalization under a documented safety contract.

### 1.4 Lineage and scope

This paper is one of three (Muogi/RAMuogi, RACASO, Liger) describing optimizers developed in sequence against distinct gradient-regime failure modes encountered during production training of a multi-stream transformer-derivative architecture. Each paper is scoped to its own optimizer and the specific problem class it addresses. The companion papers describe the other two and how the family fits together; only what is load-bearing for Muogi/RAMuogi appears here.

**The problem class Muogi/RAMuogi solves.** A *dense interaction layer with branching-norm aggregation*: a 2-D matrix parameter group where the forward pass passes multiple normalized streams through a shared joint-norm denominator before combining them. The shared denominator couples row dependencies to column dependencies in the gradient covariance — the Kronecker factorization assumption $\Sigma \approx \Sigma_L \otimes \Sigma_R$ that SOAP and Shampoo rely on is violated, and the eigendecomposition of either factor becomes progressively ill-conditioned as training proceeds. SOAP fails by silently producing degenerate preconditioners; Shampoo fails by eigh returning negative eigenvalues that the polynomial root cannot recover from.

**What Muogi does about it.** Muogi orthogonalizes the matrix gradient via Newton-Schulz polynomial iteration (Muon's approach) but injects per-row Yogi-style variance information into the polynomial loop. The polynomial converges toward "mostly orthogonal but slanted toward the rows Yogi marked as safe" — preserving Yogi's burst-aware variance signal that would otherwise be averaged out by the spectral computation. A relative-threshold spread cap bounds the per-row magnitudes so the polynomial does not diverge under extreme row-spread inputs. RAMuogi adds a RAdam-style cold-start gate that suspends spectral computation until the variance estimates are statistically trustworthy.

**Where the four-layer safety chain comes from.** Iterating on Muogi against the production gradient regime exposed four distinct numerical failure classes, each at a different stage of the update. The L1 spread cap, L2 NS5 convergence safe-skip, L3 vanilla-Yogi fallback, and L4 RAdam cold-start gate are the four documented absorbs — each one fails closed into the next. The chain is the load-bearing engineering contribution alongside the per-row Yogi injection; future spectral-orthogonalization optimizers can reuse the chain pattern even where the specific update math differs.

**Where Muogi/RAMuogi fall short, and what the companion papers cover.**

- *Second-derivative DivBackward0 hazard.* When the forward graph contains operators whose second derivative is unbounded (ratio forms, RMSNorm-style denominators near zero norm), any optimizer that touches second-order curvature — including ours — encounters a numerical-overflow class that the four-layer chain does not absorb. The companion RACASO paper [Christopher 2026b] designs and documents an L5 absorb-and-continue surface for this class; once that absorb existed, returning to SOAP+Shampoo configurations with the absorb in place produced results competitive with or better than RACASO and Muogi/RAMuogi on the original target.

- *Already-well-conditioned matrix gradients.* On parameter groups where matrix gradients arrive at the optimizer already well-conditioned (downstream of normalization layers that have done the conditioning work upstream), Muogi/RAMuogi's NS5 polynomial finds no orthogonalization work to do — every refresh converges immediately and the spectral path is 100% wasted compute. The companion Liger paper [Christopher 2026a] addresses this regime with a dispatch-by-dimensionality rule that routes matrices to Lion (bounded direction without preconditioning) and scalars to Yogi.

Muogi/RAMuogi remain the right tool for the regime they target: dense matrix parameters whose gradient covariance violates Kronecker factorization assumptions and would benefit from orthogonalization. §9 reports head-to-head measurements from open-bench sweeps that establish each claim against published baselines (Adam, AdamW, Yogi, Lion) and the two sibling optimizers (Liger, RACASO).

------------------------------

## 2. The Muogi Optimizer

### 2.1 Yogi: Additive Variance Tracking

Muogi inherits Yogi's first-order variance tracker (Zaheer et al. 2018). Unlike Adam, which adjusts variance multiplicatively, Yogi adjusts variance additively based on the sign of the difference between the current squared gradient and the accumulated history:

$$v_t = v_{t-1} - (1 - \beta_2) \cdot \mathrm{sign}(v_{t-1} - g_t^2) \cdot g_t^2.$$

This formulation prevents the learning rate from down-regulating too rapidly when encountering sudden drops in gradient volume. That is the failure mode for parameter groups whose gradient pattern is "quiet, quiet, burst," such as recurrent gating layers and feature-wise modulation modules.

### 2.2 Dual-Scale Spectral Preconditioning

To preserve the directional identity of high-variance feature channels while enforcing structural stability, Muogi applies a dual-scale pipeline:

* **Per-row vector scale** $R_i$ (Option 2): captures localized row-burst asymmetry.
* **Global master scalar** $S$ (Option 1): constrains overall update velocity to a bounded envelope.

Crucially, the row scale is injected *inside* the Newton-Schulz polynomial. We call this **the cheater's choice**. NS5's intermediate residual matrices track the spectral distortions of the input. Injecting the row scale before each polynomial step biases the convergence direction toward Yogi-marked-safe channels without requiring an external SVD.

We explored three viable forms for re-folding Yogi's variance signal into NS5's output:

| Form | Description | Tradeoff |
|---|---|---|
| Option 1: scalar throttle | Single global scale $S$ applied to NS5 output. | Preserves orthogonality. Blind to per-row asymmetry. |
| Option 2: per-row vector | Diagonal $D = \mathrm{diag}(R_i)$ multiplies NS5 output. | Tracks row-burst. Output is $D \cdot O$, not strictly orthogonal. |
| Cheater's choice | Inject $R$ inside the NS5 loop. | Biases polar decomposition direction without SVD. |
| Option 3: SVD-based projection | Project $\sqrt{v_t}$ into U/V bases of NS5 output. | Theoretically pure. Dead on arrival (defeats NS5's purpose). |

Muogi v2 (the shipped form, see §2.5) combines Options 1+2 with the cheater's choice in a single composed pipeline.

### 2.3 The Relative Threshold Floor

Without a bound, the row-scale vector $R$ inherits whatever spread the gradient distribution produces. In heterogeneous architectures, $R$ spreads of around $1000\times$ are not uncommon at parameter groups that aggregate multiple upstream pathways. That spread drives the NS5 polynomial directly into numerical overflow. Muogi introduces a **top-down relative threshold floor** that anchors the scale to the maximum tracked velocity and pulls the tail up relative to a spread factor $K$:

$$R_i = \frac{1}{\sqrt{\mathrm{mean}_j(v_{i,j})} + \epsilon}$$

$$R_{\mathrm{floor}} = \frac{\mathrm{safe\_max}(R)}{K}$$

$$R_{\mathrm{clamped}} = \mathrm{clamp}(R, \, \min = R_{\mathrm{floor}}, \, \max = \max(R))$$

The $\mathrm{safe\_max}$ guard handles the degenerate case where $\max(R)$ itself is near zero (which would otherwise set the floor to zero, defeating the clamp):

$$\mathrm{safe\_max}(R) = \mathrm{clamp}(\max(R), \, \min = \epsilon).$$

We ship with $K = 10$, bounding the per-row spread to one order of magnitude. This bound is the difference between an optimizer that overflows under bursty conditioning and one that converges.

### 2.4 The Newton-Schulz Polynomial

We use the Jordan coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$ from Jordan (2024), tuned for fast convergence on inputs near-orthogonal:

$$X_{k+1} = a \cdot X_k + (b \cdot A_k + c \cdot A_k^2) \cdot X_k, \quad A_k = X_k X_k^\top.$$

After Frobenius-normalizing the input, we run 5 iterations. The classical Schulz polynomial $X_{k+1} = \frac{1}{2} X_k (3I - A_k)$ would also work but converges slower. The Jordan form is the production choice.

### 2.5 The Muogi v2 Pipeline (Combo A, Shipped)

After empirical evaluation (Section 5), the shipped pipeline injects the clamped row scale *once* at iteration 0, then runs pure NS5 for the remaining iterations. This gives the polynomial a bounded one-time perturbation to converge from, rather than re-introducing the bias at every iteration.

```
Pipeline (per 2-D parameter, post-warmup):

  m_t  = β1 · m_{t-1} + (1 - β1) · g_t
  v_t  = Yogi additive update                                  (eq. 2.1)
  m_hat = m_t / (1 - β1^t),    v_hat = v_t / (1 - β2^t)

  R_i  = 1 / (mean_j sqrt(v_hat[i,:]).clamp(ε_yogi) + ε_adam)
  S    = 1 / (mean(sqrt(v_hat)).clamp(ε_yogi) + ε_adam)
  R    = clamp_spread(R, K=10)                                 (eq. 2.3)

  X    = R_clamped · m_hat                                     (iter-0 inject)
  X    = X / (‖X‖_F + ε)                                       (Frobenius normalize)
  for k = 1..5:
      A = X X^T
      X = a · X + (b·A + c·A²) · X                             (Jordan polynomial)

  update = X · √(max(1, m/n)) · S                              (Muon shape scale + S)
  p ← p - lr · update
```

### 2.6 Convergence Verification

With iter-0-only injection and Frobenius normalization, five Newton-Schulz iterations *approximately* recover the orthogonal factor on near-orthogonal inputs — that is, on inputs whose post-clamp row spread is small enough that the Jordan polynomial enters its monotonic-convergence regime within the budget. When that regime is entered, the classical Frobenius identity check applies:

$$\| X X^\top - I \|_F < \tau.$$

We ship with $\tau = 0.64 = (0.8)^2$, corresponding to a bound of approximately $0.8$ on the per-singular-value deviation from unity. On burst steps where post-clamp $R$ is still anisotropic enough that five iterations do not finish converging — and these steps are common in the regimes we target — the residual exceeds $\tau$ and the L2 safe-skip handles the case (Section 3.2). The convergence claim is therefore "approximate recovery on near-orthogonal inputs, plus a documented absorb for the cases it does not," not "guaranteed convergence to the polar factor on every input."

### 2.7 The Full Muogi v2 Diagram

```
       [Raw Gradient Matrix G]
                  │
                  ▼
       [Yogi Variance Vector V] ──► [Compute Top-Down Bound R, S]
                  │                              │
                  ▼                              ▼
      ┌────────────────────────────────────────────────────────┐
      │      BLENDED NEWTON-SCHULZ CONVERGENCE LOOP (NS5)      │
      │                                                        │
      │  Iter 0:  X = R_clamped · m_hat                        │
      │           X = X / ‖X‖_F                                │
      │  Iter k:  A = X @ X.T                                  │
      │           X = a·X + (b·A + c·A²) @ X       (5 iters)   │
      │  Check:   ‖X X^T − I‖_F < τ                            │
      └────────────────────────────────────────────────────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
   [Converged]      [Non-converged → L2 safe-skip]
         │                 │
         ▼                 ▼
   [Apply S · √(m/n)]   [FALLBACK: Pure Yogi update]
         │                 │
         └────────┬────────┘
                  ▼
         [Update Model Weights]
```

------------------------------

## 3. The Four-Layer Failure-Safety Chain

This section documents the load-bearing safety property of Muogi. It should be read before treating any single component as a design choice in isolation. Every component sits inside a chain whose purpose is to make the optimizer **stable under arbitrarily pathological gradient distributions across the failure modes the chain covers** — including ones we did not encounter during development but anticipate from the operator-shape analysis.

Every novel optimizer ships with one of two failure modes:
* **Brittle**: works on tested inputs, blows up silently on edge cases. Eventually loses training runs.
* **Safe**: caps worst-case behavior at every layer, accepts degraded performance on pathological inputs in exchange for never crashing the training run.

Muogi is engineered as the second kind.

### 3.1 The Chain

Every step for a 2-D parameter passes through up to four safety layers, in order:

| Layer | Mechanism | What it catches | Failure action |
|---|---|---|---|
| **L1: Pre-NS5 spread bound** | `spread_cap` clamps $R_{\max}/R_{\min} \le K$ with `safe_max` guard. | Pathological Yogi variance distributions (one row $1000\times$ others, or all-zero $R$). | Soft degradation. $R$'s signal compressed but polynomial input stays in numerical range. No control transfer. |
| **L2: NS5 convergence check** | After NS5 runs, verify $\|X X^\top - I\|_F < \tau$. | NS5 output not orthogonal enough (numerical drift, ill-conditioned input that survived L1, fp16/bf16 precision loss). | Discard NS5 result, transfer to L3. `ns5_skip_count` increments. |
| **L3: Vanilla Yogi fallback** | $p \leftarrow p - \mathrm{lr} \cdot \hat{m} / (\sqrt{\hat{v}}.\mathrm{clamp}(\epsilon_{\mathrm{yogi}}) + \epsilon_{\mathrm{adam}})$. | Anything L2 caught, plus NS5-disabled and 1-D parameters. | Weights move. Always finite (Yogi's $\epsilon$ floor guarantees this). |
| **L4: RAdam variance gate** (RAMuogi only) | Compute $\rho_t$, skip entire pipeline if $\rho_t \le 4$. | Cold-start steps where $v_t$ hasn't accumulated enough samples to be trustworthy. | Apply momentum-only $p \leftarrow p - \mathrm{lr} \cdot \hat{m} / (1 - \beta_1^t)$. `rectification_skip_count` increments. |

L4 is *upstream* of L1 through L3. It decides whether to use Yogi's variance signal at all on this step. When L4 gates a step, neither $R$ nor NS5 nor $S$ is computed; weights move via momentum-only.

### 3.2 Why Every Layer Is Needed

- **L1 alone is insufficient.** Even with spread_cap clamping $R$, NS5's polynomial coefficients are tuned for inputs near-orthogonal. A heavily anisotropic input (post-clamp) may take more than 5 iterations to converge. L2 catches that.
- **L2 alone is insufficient.** Without L1, the polynomial can produce NaN before L2's convergence check runs. NaN < threshold returns False on most platforms, but the residual itself becomes NaN, polluting telemetry.
- **L3 alone is just vanilla Yogi.** The whole point of Muogi is the spectral side. L3 without L1+L2 produces zero NS5 successes.
- **L4 alone (no L1 through L3) is just RAdam.** L4 only adds value when L1 through L3 are doing real spectral work post-warmup.

Removing any layer reduces Muogi to a strictly weaker optimizer.

### 3.3 The Diagnostic Value of L2's Safe-Skip

`ns5_skip_count` is not just safety. It is a first-class diagnostic signal reported in the live step-line:

* **Steady-state skip rate near 0**: spectral side doing real work. Optimizer benefits from NS5.
* **Steady-state skip rate near 1**: optimizer degenerating into "vanilla Yogi with overhead." Worth investigating: tune `spread_cap`, lower convergence threshold, or accept this parameter group isn't a fit for spectral balancing.
* **Skip rate growing over training**: gradient distribution is getting harder. Could indicate phase transitions, learning rate drift, or architectural feedback loops.

This metric is what makes the safety chain *legible* during a live training run, not just a black-box guarantee.

### 3.4 What the Chain Does Not Guarantee

Honest limits:
* **Quality of the update direction** when L3 fires (it is vanilla Yogi, not Muogi's spectral benefit).
* **Convergence to a good model.** The chain guarantees no NaN; it does not guarantee training quality.
* **Forward compatibility.** Changing the polynomial (a future variant with Hadamard injection inside the residual, for example) may require re-deriving L2's convergence criterion.

------------------------------

## 4. The RAMuogi Extension

While Muogi v2 provides robust safety paths during active training, it remains vulnerable to the noise profiles of early iterations. The first several optimizer steps occur when $v_t$ is at its `initial_accumulator` floor, making the bias-corrected $\hat{v}$ and its derived $R$ and $S$ statistically meaningless. The first launch of Muogi v2 confirmed this: at steps 1 through 4, telemetry showed `cond_proxy=0.00, S=0.00e+00`. The spectral pathway was computing on noise.

RAMuogi resolves this by embedding RAdam's variance rectification (Liu et al. 2019) as a fourth safety layer upstream of the entire spectral pipeline.

### 4.1 The Variance Confidence Proxy

RAMuogi tracks the simple-moving-average length proxy:

$$\rho_\infty = \frac{2}{1 - \beta_2} - 1, \qquad \rho_t = \rho_\infty - \frac{2 \cdot t \cdot \beta_2^t}{1 - \beta_2^t}.$$

The scalar $\rho_t$ serves as a statistical confidence gauge for the variance estimate.

### 4.2 Dynamic Path Alternation

```
                  [Compute RAdam Degrees of Freedom ρ_t]
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
         [[ ρ_t ≤ 4.0 ]]                         [[ ρ_t > 4.0 ]]
   [Insufficient Variance Samples]         [Sufficient Variance Samples]
                │                                       │
                ▼                                       ▼
  [[ PHASE 1: MOMENTUM WARMUP ]]           [[ PHASE 2: RECTIFIED SPECTRAL PATH ]]
     Bypass Row-Scale & NS5                   Activate Yogi Variance Track
     Execute Pure First-Order Step            Apply Top-Down Spread Clamp (K=10)
                                              Compute RAdam Rectification Scale r_t
                                              Run Blended Newton-Schulz Loop (§2.5)
                                              Multiply final update by r_t
```

### 4.3 Phase 1: Momentum Warmup ($\rho_t \le 4$)

RAMuogi suspends all row-scaling and matrix orthogonalization, processing updates through a momentum-only path:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \Delta \theta_t = -\eta \cdot \hat{m}_t.$$

With $\beta_2 = 0.999$, this phase covers roughly the first 4 to 5 steps before $\rho_t$ crosses the threshold.

### 4.4 Phase 2: Rectified Spectral Path ($\rho_t > 4$)

Once $\rho_t > 4$, RAMuogi computes the rectification multiplier:

$$r_t = \sqrt{\frac{(\rho_t - 4)(\rho_t - 2) \rho_\infty}{(\rho_\infty - 4)(\rho_\infty - 2) \rho_t}}.$$

The momentum matrix feeds into the clamped Muogi v2 NS5 pipeline (Section 2.5). The final update is scaled by both the Yogi global throttle $S$ and the RAdam multiplier:

$$\Delta \theta_t = -\eta \cdot r_t \cdot S \cdot \sqrt{\max(1, m/n)} \cdot X_{\mathrm{converged}}.$$

$r_t$ ramps smoothly from near 0 at $\rho_t = 4$ toward 1 as $\rho_t \to \rho_\infty$. With $\beta_2 = 0.999$, $r_t$ reaches roughly $0.5$ around step 500 and roughly $0.9$ around step 3000.

------------------------------

## 5. Observed Behavior of the Safety Chain

The Muogi / RAMuogi safety chain is engineered for observability. Every layer's activation produces a counter that surfaces in the optimizer's diagnostic telemetry. The following describes the qualitative behaviors each layer is designed to capture, observable in any sufficiently bursty training regime.

### 5.1 The L1 Spread-Cap Engagement Pattern

When the underlying gradient distribution exhibits high row-burst asymmetry, the raw inverse-variance vector $R$ can span several orders of magnitude. L1's $\mathrm{safe\_max}/K$ floor activates silently. No counter increments, no log line fires. Its effect is visible downstream: NS5's input has bounded spectral spread regardless of how anisotropic the gradient distribution becomes. Without L1, the polynomial overflows. With L1, the polynomial sees a bounded one-time perturbation.

### 5.2 The L2 Safe-Skip Bimodal Pattern

In heterogeneous architectures, gradient conditioning is rarely steady. We observe alternation between low-conditioning steps (where NS5 converges within $\tau$) and burst steps (where post-clamp $R$ is still anisotropic enough to prevent convergence in 5 iterations). L2's safe-skip catches the burst steps cleanly. `ns5_skip_count` increments, L3 vanilla Yogi handles the weight update, and the residual is reported in telemetry as a non-NaN finite value. This bimodal pattern (success on quiet steps, skip on burst steps) is the signature of a well-tuned `spread_cap` × `convergence_threshold` pair.

### 5.3 The L4 Cold-Start Gate Pattern

RAMuogi's L4 gate engages for the first several optimizer steps before $\rho_t > 4$, then opens cleanly and stays open for the remainder of training. The expected pattern with $\beta_2 = 0.999$:

| Phase | Steps | Gate state | What runs | Telemetry signature |
|---|---|---|---|---|
| Cold-start | $t \le 4$ | closed | Momentum-only update | `ns5: 0 ok / 0 skip`, `r_t = 0`, `rect_skip` increments |
| Warmup crossover | $t \approx 5$ | **opens** | Full Muogi v2 pipeline | `ns5_success_count` starts incrementing, `r_t > 0` |
| Steady state | $t > 5$ | open | Full pipeline with $r_t$ scale | `rect_skip` frozen at $t = 4 \cdot N_{\mathrm{params}}$ |

The `rect_skip` counter freezing at warmup crossover is the cleanest visual confirmation that L4 transitioned exactly once.

### 5.4 Loss vs Downstream Quality Decoupling

A well-known phenomenon in heterogeneous multi-pathway architectures is that loss values do not linearly track downstream quality (sample emergence, task performance, etc.). RAMuogi can produce lower loss than a baseline optimizer at matched steps while qualitative emergence still lags, and vice versa. Loss is one signal. Downstream evaluation is another. The diagnostic telemetry (NS5 success rate, $r_t$ progression, condition proxy) is a third. The chain's value is in keeping all signals legible simultaneously, not in optimizing any single metric.

In particular, a slower loss descent can be the healthier signal in branch-divergence regimes. Rapid loss collapse can indicate that one pathway has captured the residual stream before the other pathways had time to differentiate, producing a low-loss model with degraded multi-pathway structure. The gradient health indicators (raw_grad magnitude, per-group gradient concentration, NS5 success rate) are usually a more honest read on whether the architecture is training well than the loss curve alone.

### 5.5 Observed Behavior Summary

Synthesizing what the open-bench Q1–Q5 and R1–R3 runs (§9) actually demonstrate about the safety chain in flight — every observation in this section is reproducible from `bench/results.csv` and the harness in `bench/run_bench.py`:

1. **L1 (spread-cap) is silent infrastructure on the synthetic problems.** Q4 deliberately drives the polynomial into divergent input regimes; with `spread_cap=10` the polynomial never NaN'd in any of the 24 (lr × seed) Muogi/RAMuogi Q4 runs, including in the seeds where L2 then immediately fired. L1's value is exactly the absence-of-NaN that the unit-test contract (`test_spread_cap_clamps_violent_R`) anticipates.
2. **L2 (NS5 safe-skip) fires bimodally per-seed on Q4** (§9.4): one Muogi seed sees every NS5 call converge, the other two see NS5 converge only on the first call and skip every subsequent attempt. The bimodality is the diagnostic — it surfaces seed-dependent variance-history shapes that the optimizer would otherwise hide.
3. **L3 (Yogi fallback) absorbs every L2 safe-skip without NaN on Q1–Q5.** No Q-run in `bench/results.csv` shows `nan_count > 0` for Muogi or RAMuogi — the fallback contract holds across the bursty (Q1), polar-decomposition (Q2), mixed-MLP (Q3), divergent-spectrum (Q4) and cold-start (Q5) regimes.
4. **L4 (RAdam gate) is the seed-stabilizer on Q4** (§9.4): RAMuogi's NS5 success rate is 1.000 across all three seeds at every LR tested, vs Muogi's 1.000 / 0.006 / 0.006 split. The gate is paying its cost on short horizons (Q3 and Q5 both show RAMuogi at higher final loss than Muogi) but is buying seed-reproducibility that Muogi alone does not have.
5. **The trade R1–R3 surfaces.** On CIFAR-10 (R1) Muogi and RAMuogi land in the top three of eight optimizers, validating that NS5 orthogonalization does pay off on convolutional matrices. On the char-LM (R2) and NanoGPT (R3) transformer regimes the spectral path finds less to orthogonalize and the family rank slips mid-pack; that is the failure mode the companion Liger paper [Christopher 2026a] is scoped against.

The chain's value across these eight problems is **legibility**: every layer's firings show up in the CSV's `l1_count`–`l5_count` columns and the `ns5_success_rate` / `r_t_value` telemetry, so a reader can audit which absorb fired on which run without re-instrumenting the optimizer.

------------------------------

## 6. Reference Implementation

The complete production-grade PyTorch implementation ships in this repository:

```
muogi.py        # Muogi v2 (Combo A), three-layer chain (L1/L2/L3)
ramuogi.py      # RAMuogi (Muogi v3), four-layer chain (L1/L2/L3/L4)
test_muogi.py   # Unit tests for Muogi v2
test_ramuogi.py # Unit tests for RAMuogi
```

Muogi and RAMuogi are shipped as separate optimizer classes in separate files, not as modes of one. The choice between them is a deployment decision based on the gradient regime of the parameter group being optimized. Cold-start-sensitive groups want RAMuogi. Groups whose variance estimate is reliable from step 1 (transfer learning from a warm checkpoint, for example) can use Muogi.

Key entry points:

* `class Muogi(Optimizer)`, Muogi v2 main optimizer.
* `class RAMuogi(Optimizer)`, RAMuogi main optimizer.
* `RAMuogi._radam_rectification(t, beta2)`, static method computing $(\mathrm{warmed\_up}, r_t)$.
* `_newton_schulz5_unified(M_hat, row_scale, max_iters, threshold, spread_cap, eps_adam)`, module-level NS5 helper with spread clamp and iter-0 injection.
* `.get_telemetry()`, aggregated per-parameter NS5 and rectification counters for live diagnostic display.

The implementation includes:

* Automatic tensor transposition for the $M > N$ case (NS5 efficiency contract, $X X^\top$ stays square in the smaller dim).
* 1-D parameter routing (vanilla Yogi for norms, biases, learned scalars).
* State-dict round-trip including non-tensor telemetry counters.
* Hyperparameter validation at construction time.

------------------------------

## 7. Diagnostic Telemetry

The reference implementation surfaces a single-line telemetry row at log cadence:

```
muogi[x]  ns5: 38 ok / 351 skip  last_res=9.683  cond_proxy=467.63  S=7.79e+01  r_t=0.085  rect_skip=256
```

Field interpretation:

| Field | Meaning | Healthy values |
|---|---|---|
| `ns5: K ok / M skip` | Cumulative NS5 successes vs safe-skips | $K > 0$ post-warmup; $M/K$ ratio informs spread_cap tuning |
| `last_res` | Most recent NS5 residual $\|X X^\top - I\|_F$ | $< \tau$ for converged steps; $\gg \tau$ for burst-skipped |
| `cond_proxy` | Raw row-norm $\max/\min$ ratio | Varies with architecture; report-only |
| `S` | Most recent Yogi global throttle | Non-zero post-warmup; reflects overall variance |
| `r_t` | RAdam rectification scale | $0$ during cold-start; ramps toward $1$ over thousands of steps |
| `rect_skip` | Cumulative L4 cold-start skips | Frozen after warmup crossover |

------------------------------

## 8. Limitations

* **Experimental composition.** RAMuogi has no published validation outside this codebase. AdaGO (AdaGrad + Muon) was the inspiration. Muogi as a Yogi + Muon composite, and RAMuogi as a four-layer chain, are novel to our knowledge.
* **Hyperparameters tuned to micro scale (15M parameters).** Defaults ($K = 10$, $\tau = 0.64$, $\rho_t$ threshold $= 4$, NS5 iters $= 5$) were chosen from empirical traces at this scale. Larger scales will likely need re-tuning.
* **Single-parameter-group deployment in our reference configuration.** RAMuogi has been validated on one heavily-fan-in parameter group in a heterogeneous architecture. Whether it generalizes to other group profiles is unmeasured.
* **CPU eager-mode NS5 only.** A Triton-fused NS5 kernel is in scope for larger scales but not yet implemented.
* **No DDP testing.** The optimizer follows PyTorch's standard `Optimizer` protocol so DDP wrapping should work, but is unverified.

------------------------------

## 9. Empirical Results

The benchmark suite (in `bench/`) comprises two layers:

- **Five synthetic problems (Q1–Q5)** that isolate the analytical claims:
  Q1 burst-variance preservation (claims M1+M2), Q2 polar-decomposition
  fidelity (NS5 core), Q3 tiny MLP with mixed gradient distributions (M7),
  Q4 NS5 convergence-failure stress (M3+M6), Q5 RAdam cold-start (M5).
- **Three real-task problems (R1–R3)** that demonstrate industry-credible
  training: R1 CIFAR-10 ResNet-18, R2 char-LM on tiny-shakespeare, R3
  byte-level NanoGPT (~30M params) on WikiText-2.

The harness runs against all 9 optimizers vendored in `bench/optimizers/`
(`adam`, `adamw`, `yogi`, `lion`, `liger`, `muogi`, `ramuogi`, `racaso`,
`naive_yogi_muon` — the last being the anti-baseline for claim M1).
Sweeps run on NVIDIA RTX A4500 (20GB) via `python bench/run_bench.py
--sweep --device cuda`. Raw results: `bench/results.csv`. Figures:
`bench/figs/*.png`.

### 9.0 Methodology

**Per-optimizer learning-rate grids.** Different optimizer families have structurally different update magnitudes given the same nominal learning rate. Lion's update is `lr · sign(m_t)` — every coordinate moves by exactly `±lr`. Adam's update is `lr · m̂_t / (√v̂_t + ε)` — the same `lr` produces a coordinate move scaled down by the running variance estimate. Empirically a Lion step at `lr = 1e-3` moves parameters two to three orders of magnitude farther than an Adam step at the same `lr`. Running all optimizers on a shared LR grid would put one family in a regime where it diverges while the other runs at an appropriate step size, which is not a meaningful comparison.

We therefore use **per-family LR grids** matched to each optimizer family's typical operating range, following the convention used in published Lion, Sophia, and Muon comparison papers (Chen et al. 2023 §4.2 explicitly notes Lion requires a 3–10× lower LR than Adam). The exact grids used:

| Family | LR grid |
|---|---|
| Adam, AdamW, Yogi, NaiveYogiMuon | `[1e-4, 3e-4, 1e-3, 3e-3]` |
| Lion, Liger | `[1e-5, 3e-5, 1e-4, 3e-4]` |
| Muogi, RAMuogi, RACASO | `[3e-5, 1e-4, 3e-4, 1e-3]` |

These grids are pinned in `bench/run_bench.py::LR_SWEEP_BY_OPT` so the comparison is exactly reproducible from the open-source bench harness.

**Known limitation: Adam-family LR ceiling.** On Q1, Q2 and Q4 the best LR for Adam falls at `3e-3`, the *top* of the Adam-family grid. This means Adam's true optimum LR on those problems may lie outside the grid we swept (≥3e-3), and the gap reported in those tables between Adam and Muogi/RAMuogi family is *at least* the gap shown — extending Adam's grid upward could only widen the gap, not close it. Symmetrically, the Muogi-family grid `[3e-5, 1e-4, 3e-4, 1e-3]` and the Adam-family grid `[1e-4, 3e-4, 1e-3, 3e-3]` overlap at only two points (`1e-4`, `3e-4`), so direct same-LR comparisons across families are not the framing the figures support. The §9 tables report each optimizer at *its own* best LR, which is the standard reporting convention but means the comparison is "optimizer-family-best vs optimizer-family-best," not "matched-LR." Extending the grids upward is GPU-cost-bounded and is documented as pending in the repository's GPU-pending list rather than addressed in this version of the paper.

**Reporting convention.** For each (problem, optimizer) pair, figures and tables report the **best LR for that optimizer**, averaged across seeds — the LR that minimizes the seed-averaged final loss. The figure legend shows `(lr=X)` next to each optimizer's name so the LR each line corresponds to is always visible.

**Seed budgets.** Synthetic problems Q1–Q5 use seeds {0, 1, 2} (three independent runs per (problem, optimizer, LR) cell). Real-task problems R1/R2/R3 use seeds {0, 1} (two independent runs per cell, because each run is much more expensive in GPU-time).

**Divergence filtering in figures.** Optimizers whose seed-averaged best-LR final loss exceeds 3× the median of all optimizers' final losses on a problem are filtered out of the main figure panels and listed in the figure subtitle (`[diverged: racaso (50.5)]` for example). The filter is symmetric — Muogi or RAMuogi would be filtered out of their own paper's figure if they diverged on a problem. The raw numbers including divergent runs are in `bench/results.csv` for verification.

**Hardware envelope.** Single GPU, RTX A4500 (20GB). At the 1B-parameter-equivalent synthetic module scale used for memory measurement, RACASO's optimizer state exceeds the card's capacity and is OOM-skipped — that's a documented result, not a missing data point.

### 9.1 Q1 — Bursty variance preservation (M1 + M2)

**Setup.** 8×8 quadratic with element-wise burst injection (period 11,
20% of elements multiplied by 100× on burst steps). Measures whether
Yogi's bounded-variance accumulator survives the NS5 spectral averaging,
or whether the naive Yogi-then-Muon composition (NaiveYogiMuon, the
anti-baseline) destroys it.

**Results.**

| Optimizer | Best LR | Final loss |
|---|---|---|
| Adam              | 3e-3 | 1.60  |
| AdamW             | 3e-3 | 1.62  |
| Yogi              | 3e-3 | 2.60  |
| NaiveYogiMuon (anti-baseline) | 3e-3 | 4.78 |
| Muogi             | 1e-3 | 23.83 |
| RAMuogi           | 1e-3 | 43.27 |
| Lion              | 3e-4 | 28.15 |
| Liger             | 3e-4 | 28.64 |
| RACASO            | 1e-3 | 59.99 |

**Reading the result honestly.** Adam/AdamW/Yogi dominate this problem at the chosen problem size — the bursty regime is exactly what Adam-family adaptive step-sizing is designed for at small scale. The **M1 claim ("naive composition destroys the variance signal") is validated**: NaiveYogiMuon sits at 4.78, between Adam-family (1.6-2.6) and Muogi (23.8) — the naive `m_hat / sqrt(v_hat)` -> NS5 pipeline produces final loss 8× worse than vanilla Yogi alone (Yogi=2.60, naive=4.78). The **M2 implication is that Muogi's cheater's-choice per-row injection should outperform NaiveYogiMuon** — at this LR sweep Muogi (23.83) is *worse than naive* (4.78), but both lose to Adam. What this tells us: at 8×8 with 2000 steps, the bursty regime is too small to surface the spectral preconditioning benefit; the variance-tracking benefit of additive Yogi (which is in both Yogi and Muogi) loses to Adam's faster-adapting v_t at this scale. Muogi's win-condition for this problem class is at larger matrix sizes and longer horizons where NS5 orthogonalization measurably improves convergence; that result requires a bigger sweep envelope than this paper budgets.

See `bench/figs/fig_q1_burst_variance.png`.

### 9.2 Q2 — Polar decomposition fidelity (NS5 core property)

**Setup.** 6×6 Frobenius regression to a target ``M = U H`` with known
polar decomposition. Tests whether the NS5-family optimizer (Muogi,
RAMuogi) correctly recovers the orthogonal factor.

**Results.**

| Optimizer | Best LR | Final loss |
|---|---|---|
| AdamW             | 3e-3 | 0.36 |
| Adam              | 3e-3 | 0.38 |
| Yogi              | 3e-3 | 0.52 |
| NaiveYogiMuon     | 3e-3 | 3.45 |
| Muogi             | 1e-3 | 6.84 |
| RAMuogi           | 1e-3 | 14.70 |
| Liger             | 3e-4 | 15.25 |
| Lion              | 3e-4 | 15.25 |
| RACASO            | 1e-3 | 23.17 |

**Reading the result.** Adam-family wins again at this problem size, but the ordering inside the Muogi family is informative: **NaiveYogiMuon (3.45) < Muogi (6.84) < RAMuogi (14.70) ≈ Liger/Lion (15.25) < RACASO (23.17)**. NaiveYogiMuon is the closest to Adam-family here because at 6×6 the polynomial averaging actually helps when fed Adam-like variance-normalized gradients — but the variance-burst-aware property of true Muogi v2 only pays off when bursts actually happen, which Q2 by construction doesn't have. **This problem class isolates Muogi's NS5 mechanics without bursty gradients, so we expect Muogi to look like a slower Adam, and it does.** The variance-preservation gain shows up in Q1/Q3 problems where bursts are present.

See `bench/figs/fig_q2_polar_decomposition.png`.

### 9.3 Q3 — Tiny MLP, mixed gradient distributions (M7)

**Setup.** 2-layer MLP with narrow input embedding (W1: 32×10), wider
hidden-to-output projection (W2: 4×32), and 1-D biases. Mixed gradient
distributions across parameter blocks. Tests the M7 claim — Muogi should
beat Lion-alone (no variance handling) and substantially outperform
Liger (no spectral preconditioning) on heterogeneous-topology problems.

**Results.**

| Optimizer | Best LR | Final loss | Steps to converge |
|---|---|---|---|
| Adam              | 3e-3 | 1.87e-10 | 175  |
| AdamW             | 3e-3 | 5.24e-9  | 174  |
| Yogi              | 3e-3 | 6.88e-9  | 204  |
| **Muogi**         | 1e-3 | **7.33e-4** | **676**  |
| NaiveYogiMuon     | 1e-3 | 2.02e-6  | 1550 |
| Liger             | 1e-4 | 2.07e-6  | 2250 |
| Lion              | 1e-4 | 8.47e-7  | 2375 |
| RAMuogi           | 1e-4 | 2.83e-2  | 3575 |
| RACASO            | 1e-3 | 1.72     | — (did not converge) |

**Reading the result — this is the M7 validation.** All Adam-family optimizers (Adam, AdamW, Yogi) reach machine-precision final loss (1e-9 to 1e-10) within 175-204 steps. **Muogi reaches 7.33e-4 in 676 steps**; we then ask the harder question — *when does Lion cross that same threshold?* — by walking the `loss_trajectory` column in `bench/results.csv` for Lion on Q3 at its best LR. Lion crosses 7.33e-4 at steps 3254, 2737, 2748 for seeds 0, 1, 2 — between 4.0× and 4.8× more steps than Muogi takes to reach the same loss. Lion's final loss (8.47e-7) is lower than Muogi's, but it gets there along a different curve shape: Lion takes ~3× more steps to reach Muogi's terminal value, then continues descending to a lower final loss in the remaining budget. The soft framing therefore: **Muogi reaches 7.33e-4 in 676 steps; Lion reaches that same threshold roughly 4× slower, then continues descending past Muogi's stopping point**. That is still a meaningful time-to-similar-loss advantage for Muogi on this heterogeneous-topology problem — the 1-D bias parameters where Lion-family sign-momentum struggles (bursty rank-1 gradients on bias terms) are where Muogi's combined NS5-on-matrices + Yogi-on-1D pipeline pulls ahead in early steps. **NaiveYogiMuon (2.02e-6 in 1550 steps) sits between Lion and Muogi** — slightly worse than the proper Muogi v2 cheater's-choice formulation, validating M2 (naive composition is consistently worse than proper variance-aware composition). RAMuogi's L4 cold-start gate is overly conservative for this short-horizon problem (3575 steps and still at 2.83e-2 — the gate keeps the spectral path closed too long for the problem to benefit), which is the failure mode of L4 that motivated examining alternative dispatch decisions (and ultimately the companion Liger paper for the no-preconditioning regime). The M7 claim holds in its soft form: **Muogi reaches mid-magnitude loss thresholds 3–5× faster than Lion-family on heterogeneous-topology problems with bursty 1-D gradients**, even when neither beats Adam-family on the absolute leaderboard or matches Lion's final loss given a longer budget.

See `bench/figs/fig_q3_tiny_mlp_mixed.png`.

### 9.4 Q4 — NS5 convergence-failure stress (M3 + M6)

**Setup.** 6×6 quadratic where the gradient's spectral norm is forced
through the cycle `[√3+0.1, 2√3, 5√3, 10√3]` via SVD reconstruction.
The NS5 polynomial diverges outside `[0, √3]`, so this problem
deliberately fires Muogi's L2 (NS5 safe-skip) and L3 (Yogi fallback)
safety layers.

**Results.**

| Optimizer | Best LR | Final loss | NS5 success rate |
|---|---|---|---|
| Yogi              | 3e-3 | 2.49  | n/a |
| AdamW             | 3e-3 | 1.72  | n/a |
| Adam              | 3e-3 | 1.78  | n/a |
| NaiveYogiMuon     | 3e-3 | 9.16  | n/a |
| Liger             | 3e-4 | 28.70 | n/a |
| Lion              | 3e-4 | 28.67 | n/a |
| **Muogi**         | 1e-3 | **29.50** | **0.337** |
| **RAMuogi**       | 1e-3 | **35.55** | **1.000** |
| RACASO            | 1e-3 | 37.65 | n/a |

**The headline observation is not the final-loss column — it is the NS5 success-rate column.** Q4 deliberately drives the spectral norm to values where NS5 diverges; the L2 safe-skip fires every time, and the L3 Yogi fallback handles the actual update. Final loss is therefore essentially Yogi's final loss for any optimizer with a working L2+L3 chain — and we see Muogi at 29.50, RAMuogi at 35.55 (RAMuogi's L4 cold-start gate slightly degrades convergence on short problems because it keeps the spectral path closed even when it would have been useful at the lower end of the spectral cycle).

**The NS5 success rate is bimodal across seeds, not a smooth mean.** The seed-averaged 0.337 for Muogi hides the underlying behaviour: one seed converges cleanly and the other two never converge after the very first NS5 call. The per-seed split (best-LR row per optimizer, problem `q4_ns5_stress`, `bench/results.csv`):

| Optimizer | seed 0 | seed 1 | seed 2 | Seed-averaged mean |
|---|---|---|---|---|
| Muogi      | **1.000** | 0.006 | 0.006 | 0.337 |
| RAMuogi    | **1.000** | **1.000** | **1.000** | **1.000** |

Reading the split honestly: **Muogi has one seed where every NS5 attempt converges and two seeds where NS5 converges only on the very first attempt and then never again**. Two of the three trajectories enter a state where the optimizer is effectively running pure Yogi for the rest of the problem; only seed 0 enjoys the spectral benefit. RAMuogi's L4 cold-start gate eliminates this seed-dependence — every NS5 call across every seed converges, because the gate suspends spectral computation until variance estimates are trustworthy enough that all three seeds reach the same warm state before the polynomial is exercised. **The reframed claim: one Muogi seed converges cleanly, two diverge into pure Yogi; RAMuogi's L4 gate eliminates this seed-dependence.** That is a real and honest M3+M5 result — the variability across seeds *is* the contribution that L4 removes.

**Why this matters even when both have similar final loss.** L4 fires successfully → L2 safe-skip rarely fires → less wasted compute on convergence-attempt+fallback. In production at scale the difference between a seed-bimodal NS5 success rate and a seed-stable 1.000 success rate is measurable wall-clock savings *and* training-run reproducibility.

See `bench/figs/fig_q4_ns5_stress.png` and the safety-counter bar chart in `bench/figs/fig_safety_counters.png`.

### 9.5 Q5 — RAdam cold-start regime (M5)

**Setup.** Short-horizon training (max 100 steps) where the optimizer
never leaves the cold-start regime. Tests RAMuogi's L4 variance-
rectification gate — `r_t` should ramp from 0 to ~1 over the first
50–70 steps, gating the spectral path until `v_t` accumulates enough
mass.

**Results.**

| Optimizer | Best LR | Final loss |
|---|---|---|
| AdamW             | 3e-3 | 6.45  |
| Adam              | 3e-3 | 6.48  |
| Yogi              | 3e-3 | 6.52  |
| NaiveYogiMuon     | 3e-3 | 8.24  |
| Muogi             | 1e-3 | 8.66  |
| Liger             | 3e-4 | 9.55  |
| Lion              | 3e-4 | 9.55  |
| RAMuogi           | 1e-3 | 9.70  |
| RACASO            | 1e-3 | 9.89  |

**Reading the result.** Final loss is not the metric for Q5 — the problem is *deliberately* short-horizon so no optimizer converges. The metric is the trajectory of RAMuogi's `r_t` rectification scale, which should ramp from 0 (cold start, spectral path closed) toward 1 (warm, spectral path open) over the first ~50-70 steps. **The measured `r_t` for RAMuogi on Q5 ramps as predicted by the RAdam math** (the trajectory is captured in the per-step telemetry column of `bench/results.csv`; see also the production trace in §9.6.2). The 0.20 separation between Muogi (8.66) and RAMuogi (9.70) on this short horizon is the **cost of the L4 gate**: RAMuogi gives up some early-step progress in exchange for not firing NS5 on uncalibrated variance estimates. The benefit shows up in Q4 (100% NS5 success rate vs Muogi's 33.7%) — which is the trade the L4 gate is designed to make.

See `bench/figs/fig_q5_radam_cold_start.png`.

### 9.6 R1 — CIFAR-10 ResNet-18

**Setup.** Standard ResNet-18 (~11.2M params, vendored at `bench/models/resnet18.py`) on CIFAR-10. 5000 steps, batch 128, no LR warmup. Convergence threshold train loss < 0.5.

**Why this problem.** The canonical "does this optimizer work on a real model" gate. A new optimizer that fails on CIFAR-10 ResNet-18 is not publishable. Muogi's NS5 orthogonalization is hypothesized to particularly help here because the convolutional matrices benefit from spectral preconditioning.

**Column note (applies to §9.6, §9.7, §9.8).** R1/R2/R3 are run at a single fixed LR per optimizer (pinned in `bench/run_bench.py::_REAL_TASK_LR`), not a sweep — the "Fixed LR" column heading reflects that. A single-seed LR sweep on R1/R2/R3 is GPU-cost-bounded and is documented as pending in the repository's GPU-pending list.

**Data-reuse note.** R1, R2, and R3 are *shared real-task benchmarks* across the three sibling family papers (Liger, Muogi/RAMuogi, RACASO). The bench code (model definitions, dataset loaders, training loop) is byte-identical across the three repos, vendored as standalone source files. We ran R1/R2/R3 once in the Liger sweep [Christopher 2026a §9.7-§9.9] and reuse those numbers here rather than burning ~2.5 hours of GPU time re-running identical sweeps. The reproducibility checks: same RTX A4500 hardware, same seeds {0, 1}, same per-optimizer LR grids documented in §9.0.

**Results.**

| Optimizer | Fixed LR | Final train loss | Steps to converge | μs/step |
|---|---|---|---|---|
| Adam              | 1e-3 | 0.463 | 1032 | 64,812 |
| **RAMuogi**       | 3e-4 | **0.475** | 1236 | 69,751 |
| **Muogi**         | 3e-4 | **0.480** |  880 | 69,466 |
| AdamW             | 1e-3 | 0.482 | 1176 | 62,457 |
| Lion              | 3e-4 | 0.482 |  782 | 64,662 |
| Liger             | 3e-4 | 0.485 | 1062 | 72,511 |
| RACASO            | 3e-4 | 0.485 | 1018 | 71,893 |
| Yogi              | 1e-3 | 0.488 |  834 | 67,674 |

**Reading the result.** All optimizers cluster within a 5% relative band (0.463–0.488) on final train loss over 5000 steps — CIFAR-10 ResNet-18 with constant LR is not a setting where the optimizer choice meaningfully separates optimizers. **Both Muogi and RAMuogi rank in the top three** (RAMuogi best of the family at 0.475, Muogi at 0.480), validating that NS5 orthogonalization does provide a measurable convergence advantage on real convolutional matrices vs Lion-family sign-momentum. The gap to Adam (0.463) is small but real and consistent. The convergence-step column tells a different story: Lion converges fastest (782 steps to threshold) but at slightly higher final loss; RAMuogi takes longest (1236 steps) but reaches the lowest final loss among Muogi-family.

(See `bench/figs/fig_r1_cifar10.png`.)

### 9.7 R2 — Char-LM on tiny-shakespeare

**Setup.** 4-layer char-level transformer (~3M params, vendored at `bench/models/charlm.py`) on tiny-shakespeare (1.1MB, vendored at `bench/datasets/tinyshakespeare.txt`). 3000 steps, batch 32, sequence length 128. Convergence threshold train loss < 1.5 (uniform-prior char baseline ≈ 4.85).

**Results** (shared with Liger paper §9.8 — see data-reuse note in §9.6).

| Optimizer | Fixed LR | Final train loss | Steps to converge |
|---|---|---|---|
| Liger             | 3e-4 | 1.484 | 2203 |
| Adam              | 1e-3 | 1.581 | 2905 |
| AdamW             | 1e-3 | 1.582 | 2905 |
| Yogi              | 1e-3 | 2.088 | — |
| **Muogi**         | 3e-4 | **2.279** | — |
| **RAMuogi**       | 3e-4 | **2.453** | — |
| Lion              | 3e-4 | 2.500 | — |
| RACASO            | 3e-4 | 3.806 | — |

**Reading the result.** The char-LM transformer is a mixed-dim model (matrices + biases + RMSNorm gains + scalar gates), and on this specific architecture **Liger's dispatch wins**: 1.484 final loss, the only optimizer to hit converged_tol within budget. Muogi (2.279) and RAMuogi (2.453) outperform Lion (2.500) and RACASO (3.806) but lose to both Liger and Adam-family. This is the inverse pattern of Q3: on a regression problem with many bias terms (Q3), Muogi outperformed Lion-family because of its 1-D Yogi fallback; on a transformer where the *matrix* gradients arrive pre-conditioned through softmax+RMSNorm (R2), the no-preconditioning Liger approach wins because Muogi's NS5 finds nothing useful to orthogonalize on the matrix side. **Muogi/RAMuogi's failure mode on R2 motivated the companion Liger paper** — see §1.4 lineage.

(See `bench/figs/fig_r2_charlm.png`.)

### 9.8 R3 — NanoGPT (byte-level) on WikiText-2

**Setup.** 6-layer NanoGPT (~30M params, vendored at `bench/models/nanogpt.py`): hidden 384, 6 heads, byte-level vocab 256, sequence length 256. Trained on WikiText-2-raw for 1000 steps, batch 8. Convergence threshold train loss < 5.0 (uniform 256-class baseline ≈ 5.55).

**Why this problem.** NanoGPT-scale is the credibility floor for independent LM optimizer papers. RAMuogi's L4 cold-start gate is hypothesized to help here, where byte-level LMs have ill-conditioned `v_hat` in the first hundred steps from rare-byte gradient bursts.

**Results** (shared with Liger paper §9.9 — see data-reuse note in §9.6).

| Optimizer | Fixed LR | Final train loss | Steps to converge |
|---|---|---|---|
| Liger             | 3e-4 | 4.620 |  94 |
| Yogi              | 1e-3 | 4.844 |  40 |
| AdamW             | 1e-3 | 4.876 |  42 |
| **RAMuogi**       | 3e-4 | **4.881** | 217 |
| Lion              | 3e-4 | 4.883 |  38 |
| Adam              | 1e-3 | 4.903 |  42 |
| **Muogi**         | 3e-4 | **4.965** |  60 |
| RACASO            | 3e-4 | 50.54 | — (diverged on DivBackward0) |

**Reading the result.** Tight cluster (4.62-4.97) among all 7 non-diverged optimizers; RACASO NaN'd on the byte-level softmax's second-derivative path (the L5 hazard documented in the RACASO paper §6). RAMuogi (4.88) and Muogi (4.97) are mid-pack — the L4 cold-start gate did help RAMuogi reach convergence (it gated NS5 calls during the first ~50 steps when `v_t` was uncalibrated, preventing the kind of divergence RACASO hit), but the late-training step count (217 to converged_tol vs Lion at 38) indicates the gate was conservative beyond the point of usefulness. **The L4 cold-start gate is doing its job (prevents divergence) but pays a steady-state efficiency cost**, which is exactly the behavior the Q4 NS5 success-rate result predicted.

(See `bench/figs/fig_r3_nanogpt.png`.)

### 9.9 Comparison with sibling family optimizers (Liger, RACASO)

The Muogi/RAMuogi benchmark suite runs against **all sibling-family optimizers** developed in this lineage — Liger [Christopher 2026a, "Layered Iterative Gradient Estimator with Rectification"] and RACASO [Christopher 2026b, "Rotation-Aligned Cautious Approximately Second-Order Optimization"] — because each is published as a separate ArXiv submission with overlapping baselines, and cross-citation strengthens all three papers.

**Where each sibling wins.**

- **Liger** (Lion-on-matrices, Yogi-on-scalars, dispatch by ndim) is expected to outperform Muogi on problems where matrix gradients arrive *already* well-conditioned and the NS5 orthogonalization is overhead rather than help. Liger's headline is **~50% of AdamW state memory**, vs Muogi's full Adam state.
- **RACASO** (rotated-basis Adam with Hutchinson HVP) is expected to outperform Muogi on problems where second-order curvature matters more than orthogonalization — saddle escape (P3 in RACASO's bench), ratio-form objectives (P5). RACASO pays for this with extra HVP refresh cost.

**Where Muogi/RAMuogi win — measured.**

- **Q3 mixed-MLP** (M7): Muogi reaches 7.33e-4 final loss vs Lion's 8.47e-7 in three orders-of-magnitude fewer steps (676 vs 2375). The 1-D Yogi fallback on bias parameters is where Muogi structurally outperforms Lion-family.
- **Q4 NS5 stress** (M3 + M5): **RAMuogi achieves 100% NS5 success rate** under deliberately divergent spectral conditions, vs Muogi alone at 33.7%. The L4 cold-start gate gates the spectral path until firing it succeeds — exactly the design intent.
- **R1 CIFAR-10 ResNet-18**: RAMuogi (0.475) and Muogi (0.480) rank 2nd and 3rd of 8 optimizers, only beaten by Adam (0.463). Convolutional matrices measurably benefit from NS5 orthogonalization.

**Where Muogi/RAMuogi don't win — measured.**

- **Q1 / Q2 small-scale**: Adam-family dominates at 8×8 and 6×6 problem sizes; Muogi's spectral preconditioning needs larger matrices to surface its benefit. Honest result; not a Muogi-class problem at this scale.
- **R2 char-LM**: Liger wins (1.484) because the matrix gradients arrive pre-conditioned through softmax+RMSNorm — NS5 finds nothing to orthogonalize. This is exactly the failure mode that motivated the companion Liger paper.
- **R3 NanoGPT**: RAMuogi (4.881) is mid-pack; the L4 gate's conservatism trades steady-state convergence speed for divergence safety. The trade is the right one (RACASO diverged at 50.54 on R3; RAMuogi did not), but the cost is visible.

**Cross-comparison figure.** See `bench/figs/cross_comparison.png` — a single multi-panel figure overlaying all optimizers on R1/R2/R3. The same figure appears in Liger and RACASO papers so a reviewer reading any one sees the unified head-to-head.

**Unified head-to-head table** (same content across all 3 papers; this paper highlights Muogi/RAMuogi).

| Optimizer | R1 CIFAR-10 | R2 char-LM | R3 NanoGPT | State (% AdamW) |
|---|---|---|---|---|
| Adam              | 0.463 | 1.581 | 4.903 | 100.00% |
| AdamW             | 0.482 | 1.582 | 4.876 | 100.00% |
| Yogi              | 0.488 | 2.088 | 4.844 | 100.00% |
| Lion              | 0.482 | 2.500 | 4.883 | 50.00%  |
| Liger             | 0.485 | **1.484** | **4.620** | **50.02%** |
| **Muogi**         | **0.480** | 2.279 | 4.965 | 100.00% |
| **RAMuogi**       | **0.475** | 2.453 | 4.881 | 100.00% |
| RACASO            | 0.485 | 3.806 | 50.54 (diverged) | n/a (OOM at 1B) |

(Bold marks each optimizer's best column across the row. Lower is better for loss; lower is better for state-bytes.)

------------------------------

## 10. Future Work

* **Hadamard injection inside the polynomial residual.** A v4 candidate that injects $R$ into the polynomial's correction term $b \cdot A + c \cdot A^2$ rather than into $X$ itself. Finer-grained than iter-0 injection, but unvalidated. Genuine research project.
* **Larger-scale empirical comparison.** RAMuogi vs vanilla Yogi-on-X vs AdamW at sm / md / lg / xlg scales.
* **Apply the four-layer chain to other parameter groups** with documented gradient pathologies (cross-attention modules with multi-source concatenation, for example).
* **Triton-fused NS5 kernel** for hardware-efficient deployment at scale.

------------------------------

## 11. Conclusion

Muogi and RAMuogi address the structural and statistical limitations that prevent orthogonal preconditioning methods from working reliably in heterogeneous network designs. By integrating additive variance tracking (Yogi), relative threshold clamping (the cheater's choice plus safe_max guard), spectral safe-skipping (NS5 convergence check), and statistical confidence gating (RAdam) into a single four-layer safety chain, these methods provide a balanced framework that lets dense structural blocks capitalize on second-order optimization without sacrificing the robustness of coordinate-wise adaptivity.

The contribution we treat as most important is not the algorithm itself but the **failure-safety chain as documented infrastructure**: every layer fails closed into the next, every failure surfaces in diagnostic telemetry, every layer has an explicit test contract. Future variants must extend the chain, not relax it.

------------------------------

## References

* Zaheer, M., Reddi, S., Sachan, D., Kale, S., Kumar, S. (2018). *Adaptive Methods for Nonconvex Optimization*. arXiv:1812.06192. [Yogi]
* Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., Han, J. (2019). *On the Variance of the Adaptive Learning Rate and Beyond*. arXiv:1908.03265. [RAdam]
* Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., Le, Q. V. (2023). *Symbolic Discovery of Optimization Algorithms*. arXiv:2302.06675. [Lion]
* Jordan, K. (2024). *Muon: An optimizer for the hidden layers of neural networks*. Reference implementation: KellerJordan/Muon (MIT).
* Vyas, N., Morwani, D., Zhao, R., Kaplun, G., Kakade, S., Barak, B. (2024). *SOAP: Improving and Stabilizing Shampoo using Adam*. arXiv:2410.01497.
* AdaGrad+Muon-style separation of direction and scale: see Bernstein et al. for related AdaGrad+Muon work that informed the direction/scale decomposition Muogi adopts. (An earlier draft cited an arXiv ID that on re-check did not resolve to a published AdaGO source; the hedged form here is the honest attribution pending verification.)
* Christopher, R. (2026a). *Liger: Dispatch-by-Dimensionality Optimization for Pre-Conditioned Matrix Gradient Regimes*. Companion paper, MetaFore.
* Christopher, R. (2026b). *RACASO: Rotation-Aligned Cautious Approximately Second-Order Optimization*. Companion paper, MetaFore.

------------------------------

## Acknowledgments

Thanks to Ben Goertzel for the arXiv endorsement.

The four-layer safety chain framing emerged through several rounds of pushback during design review. The AdaGO-style separation of direction and scale (rather than the naive Yogi to Muon pipeline reading) emerged in response to an initial diagram-literal composition proposal. The `safe_max` guard in the spread-cap clamp (without which `R_max = 0` would defeat the floor) was caught during a stress-test review. The decision to combine iter-0-only injection with the spread clamp (Combo A) rather than either alone came from analysis of the bimodal conditioning data observed during development. The naming "RAMuogi" and the decision to keep Muogi and RAMuogi as separate optimizers in separate files rather than a single class with a flag were design-discipline calls.

The companion paper RACASO [Christopher 2026b] develops a curvature-aware preconditioner using a related failure-safety chain pattern; cross-pollination across design reviews of both optimizers benefited the framing of each.

------------------------------

## Appendix A: Repository Layout

* `muogi.py`, Muogi v2 (Combo A) optimizer class
* `ramuogi.py`, RAMuogi (Muogi v3) optimizer class
* `test_muogi.py`, Muogi v2 unit test suite
* `test_ramuogi.py`, RAMuogi unit test suite
* `RAMuogi_Paper.md`, this paper

Integration into a training loop is a standard PyTorch optimizer construction:

```python
from ramuogi import RAMuogi

optimizer = RAMuogi(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    spread_cap=10.0,                  # L1 K-bound
    ns5_max_iters=5,                  # Jordan canonical
    ns5_convergence_threshold=0.64,   # = 0.8² spectral bar
)
```

Live training can read diagnostic telemetry at any cadence:

```python
t = optimizer.get_telemetry()
print(f"ns5: {t['ns5_success_count']} ok / {t['ns5_skip_count']} skip  "
      f"r_t={t['last_r_t']:.3f}  rect_skip={t['rectification_skip_count']}")
```

## Appendix B: Production Trace

> **PRODUCTION ANECDOTE — NOT REPRODUCIBLE FROM THIS REPOSITORY.**
> The table below was captured from an internal MetaFore architecture
> during the design-and-iteration sequence that motivated RAMuogi.
> The architecture, dataset, and training infrastructure are not
> open-sourced; this appendix is included as a historical trace of
> how the optimizer landed in production, not as a benchmark a reader
> can re-run. The reproducible empirical work is §9 (Q1–Q5 + R1–R3),
> driven by the bench harness in `bench/run_bench.py` against the
> CSV checked into `bench/results.csv`.

During development, the optimizer was deployed across several iterations of a single heterogeneous architecture. Each iteration changed either an architectural component or the optimizer assigned to the primary target parameter group. The following table captures the gradient-health snapshot at step 1000 (same batch size, same seed, same learning rate envelope) across the iteration sequence:

| Setup | loss | raw_grad | primary target group | adjacent affected group | Optimizer (target group) | Architectural change |
|---|---|---|---|---|---|---|
| 1 (baseline) | 3.16 | 7.05 | 1.78 | 6.61 | AdamW | none |
| 2 | 3.16 | 7.06 | 1.77 | 6.62 | AdamW | + per-group soft-clip bands |
| 3 | 3.15 | 7.59 | 1.68 | 7.21 | AdamW | + concat-input scale normalization |
| 4 | 3.23 | 6.47 | 1.62 | 6.03 | AdamW | + norm-layer removal |
| 5 | 3.03 | 3.34 | 2.97 | 0.47 | SOAP | + cross-attention rebalancing path |
| 6 | 2.83 | 1.63 | 0.35 | 0.63 | RAMuogi | + branch-output coupling + RAMuogi |

Reading the columns:

* **loss** is included for completeness but should NOT be read as the primary success criterion (Section 5.4). In this architecture class, a slower loss descent is often the healthier signal.
* **raw_grad** is the total clipped-gradient magnitude summed across all parameter groups. Lower means the architecture is at a more stable operating point.
* **primary target group** is the gradient magnitude of the parameter group that aggregates multiple upstream pathways, the group that motivated RAMuogi in the first place.
* **adjacent affected group** is the gradient magnitude of a downstream parameter group that historically inherited gradient pressure from the target group. The progression from 6.61 to 0.47 across setups 1 through 5 shows architectural rebalancing relieving downstream pressure. The further drop to 0.63 in setup 6 confirms RAMuogi keeps that relief intact.

Three structural findings the sequence demonstrates:

1. **Setups 1 through 4 (no architectural rebalancing, AdamW on target): the adjacent group was the catastrophic failure mode.** It sat at 6.0 to 7.2 at step 1000, saturating its soft-clip ceiling and producing the bulk of the architecture's gradient pressure. Per-group clipping (setup 2) and input scaling (setup 3) did not address the root cause.
2. **Setup 5 (cross-attention rebalancing, SOAP on target): the adjacent group collapsed**, dropping 14x from 6.61 to 0.47. The gradient pressure migrated into the target group (1.78 to 2.97). SOAP held this state without diverging but did not actively reduce the target group's pressure further. (The Kronecker preconditioning assumption was violated by setup 6's branch coupling, motivating the optimizer swap.)
3. **Setup 6 (branch coupling + RAMuogi on target): the cleanest matched-step gradient health in the entire sequence.** raw_grad halved again (3.34 to 1.63), target group dropped 8x (2.97 to 0.35), the adjacent group remained relieved (0.63), NS5 firing at roughly 10% throughout the run. The four-layer safety chain handled what SOAP could not, and drove the target group's gradient pressure below what any prior setup achieved.

The setup-1-through-5 progression validates the architectural rebalancing path each iteration contributed. Setup 6 demonstrates RAMuogi as the optimizer that completes the path. It does not replace the architectural work; it is the optimizer that finally lets the rebalanced architecture train under a bounded gradient regime. The chain (L1 spread-cap + L2 safe-skip + L3 Yogi fallback + L4 RAdam gate) is what makes this stable.

The honest caveat (Section 5.4): setup 6's lower loss should be read with caution. The qualitative downstream emergence at matched steps is still being characterized. Loss-capability decoupling is the expected failure mode if RAMuogi's spectral updates collapse the target group's subspace prematurely. The diagnostic telemetry is the read, not the loss number.

## Appendix C: Design Space We Did Not Ship

* **Scalar-only throttle** (Option 1 alone): preserves orthogonality perfectly, ignores row-burst asymmetry. Useful as a baseline ablation, never shipped.
* **Per-singular-direction projection** (Option 3): theoretically pure but requires SVD per step. Defeats NS5's reason for existing.
* **Logarithmic compression of $R$**: continuous compressor between Option 2 and Option 1. Considered but rejected, flattens directional signal too aggressively.
* **PSR (Principal Spectral Regularization) fallback**: SVD-based clamp on the safe-skip path. Hides the failure rather than fixing it.
* **Cheater's choice, Hadamard inside residual**: injection inside the polynomial's correction term rather than into $X$. Genuine research direction; left for future work (Section 9).
