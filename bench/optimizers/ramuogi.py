"""RAMuogi optimizer — Muogi v3 = RAdam rectification + Muogi v2 (Combo A).

by Richard I Christopher, 2026

RAMuogi extends Muogi v2 with a fourth safety layer (L4) based on RAdam's
variance-confidence gate (Liu et al. 2019). At cold start, Yogi's v_t
hasn't accumulated enough samples for the bias-corrected v_hat to be
trustworthy — the per-row scale R and global scalar S derived from it
are meaningless noise. Muogi v2's first few steps showed cond_proxy=0.00,
S=0.00e+00 (the smoking gun).

RAdam solves this by computing a per-step rectification term r_t that
quantifies whether v_t has enough samples. When ``rho_t <= 4``, the
optimizer skips the entire adaptive pipeline and applies a momentum-only
SGD-style update. Once ``rho_t > 4`` (typically by step 5 with β2=0.999),
the optimizer transitions smoothly to full Muogi v2 math, with the final
update scaled by r_t.

The four-layer safety chain:
  L1 — spread_cap clamps R.max()/R.min() ≤ K=10
  L2 — NS5 convergence safe-skip on residual > threshold
  L3 — vanilla Yogi fallback (1-D params, NS5 disabled, NS5 skip)
  L4 — RAdam rectification gate (skip entire pipeline when v_t cold)

L4 is upstream of L1-L3: it decides whether to USE Yogi's variance signal
at all this step. When L4 gates a step out, neither R nor NS5 nor S is
computed; weights move via momentum-only update.

See ``docs/Muogi_Paper.md`` §11 for design rationale.

## What RAMuogi adds on top of Muogi

The full Muogi v2 composite pipeline (Yogi additive variance, per-row
cheater's-choice injection inside NS5, global throttle S, Muon shape
scale, three-layer L1/L2/L3 safety chain) is documented in
``muogi.py``'s module docstring and is reused here unchanged when the
L4 gate is open. **The only delta in RAMuogi is the L4 cold-start
gate**: RAdam's variance-confidence rectification (Liu et al. 2019)
applied *upstream* of the Muogi v2 pipeline. Read ``muogi.py`` for
the Yogi+NS5 mechanics; read this file for the L4 wrapping.

L4 in one sentence: compute RAdam's ``rho_t``; if ``rho_t <= 4`` the
variance estimate is too uncertain to drive the spectral path, so
skip R, NS5, and S entirely and apply a momentum-only SGD-style
update; otherwise run the full Muogi v2 step with the final update
multiplied by RAdam's rectification scalar ``r_t``.

See ``RAMuogi_Paper.md`` §4 for the design rationale and §9.4 for the
seed-stabilization result that motivates the gate (Muogi alone is
seed-bimodal on the Q4 NS5-stress problem at 1.000 / 0.006 / 0.006
NS5 success rate across three seeds; RAMuogi with L4 closed is
1.000 / 1.000 / 1.000 across the same seeds).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


def _newton_schulz5_unified(
    M_hat: torch.Tensor,
    row_scale: torch.Tensor,
    max_iters: int = 5,
    convergence_threshold: float = 0.5,
    spread_cap: float = 10.0,
    eps_adam: float = 1e-8,
) -> Tuple[torch.Tensor, bool, float]:
    """Muogi v2 (Combo A) — clamped row-scale + iter-0-only injection.

    Composes two mitigations that addressed Muogi v1's failure mode
    (every NS5 attempt safe-skipping under bursty conditioning, with
    row-norm condition ratios oscillating between 1 and 100000+):

      1. **Clamp R spread to ``spread_cap``** before any use. R's
         max-to-min ratio is bounded so the polynomial cannot be
         driven into overflow regardless of Yogi's variance estimate.
         Uses ``safe_max = R_max.clamp(min=eps_adam)`` to guard
         against the degenerate ``R_max = 0`` case (would otherwise
         set the floor to 0, defeating the clamp).

      2. **Inject R once before the loop**, then run pure NS5 for
         the remaining iterations. The polar-decomposition polynomial
         only feels R's perturbation once and has ``max_iters - 1``
         subsequent iters to absorb it. Frees us to use the classical
         ``||X X^T − I||_F < threshold`` convergence check because the
         converged matrix is now polar(R · m_hat), a true orthogonal.

    Returns ``(X, converged, residual)`` where:
      - ``residual = ||X X^T − I||_F`` (or ``||X^T X − I||_F`` for the
        tall-matrix internal orientation)
      - ``converged`` is True iff ``residual < convergence_threshold``

    Operates in fp32 for numerical stability; returns in input dtype.
    """
    assert M_hat.ndim == 2, f"NS5 expects 2-D, got {tuple(M_hat.shape)}"
    assert row_scale.ndim == 1 and row_scale.shape[0] == M_hat.shape[0], (
        f"row_scale shape {tuple(row_scale.shape)} mismatched with "
        f"M_hat shape {tuple(M_hat.shape)}"
    )
    a, b, c = (3.4445, -4.7750, 2.0315)
    orig_dtype = M_hat.dtype
    X = M_hat.to(torch.float32)
    R = row_scale.to(torch.float32)

    # ── Mitigation 1: clamp R spread to spread_cap ────────────────────
    # Bounds R.max() / R.min() at spread_cap. With safe_max guard, the
    # floor is well-defined even when R_max is zero or eps_adam-small.
    R_max = R.max()
    safe_max = torch.clamp(R_max, min=eps_adam)
    R = R.clamp(max=R_max, min=safe_max / spread_cap)

    # ── Mitigation 2: inject R once before the loop ───────────────────
    # X = R · m_hat (in the original orientation, before NS5's internal
    # transpose). The polynomial subsequently runs on this biased input
    # without re-injecting R each iteration, so the bias enters NS5 once
    # and the polynomial has all 5 iterations to converge to polar(R·m).
    X = R.unsqueeze(-1) * X      # (m, n) <- (m, 1) * (m, n), original orientation

    # Frobenius-normalize the biased input. After this point R is no
    # longer used — pure NS5 runs on the (normalized, biased) X.
    X = X / (X.norm() + 1e-7)

    # NS5 efficiency contract: ``X X^T`` must be square in the SMALLER
    # of (m, n). If the original matrix is wider (m <= n), X stays in
    # native orientation. If it's taller (m > n) — e.g. wide projection
    # layers — we operate on X.T internally so the polynomial cost is
    # O(min(m,n)^2 * max(m,n)) instead of O(max(m,n)^2 * min(m,n)).
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(max_iters):
        # Pure NS5 polynomial step (no re-injection).
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    # ── Convergence check ─────────────────────────────────────────────
    # Iter-0-only injection means the converged X is polar(R · m_hat),
    # a TRUE orthogonal matrix, so the classical ||X X^T − I||_F check
    # applies again. Use the side whose product is square in the smaller
    # dim (matches NS5's internal transpose convention).
    if X.shape[0] <= X.shape[1]:
        eye = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
        residual_t = (X @ X.T - eye).norm()
    else:
        eye = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
        residual_t = (X.T @ X - eye).norm()
    residual = float(residual_t.item())
    converged = residual < convergence_threshold
    return X.to(orig_dtype), converged, residual


class RAMuogi(Optimizer):
    """RAMuogi — Muogi v3 = RAdam rectification + Muogi v2 (Combo A).

    Adds a fourth safety layer (L4) on top of Muogi v2's three-layer
    chain. When v_t hasn't accumulated enough variance samples
    (rho_t <= 4 via RAdam math), skips the entire spectral pipeline
    and applies a momentum-only update. Once warmed up, runs the full
    Muogi v2 pipeline with the final update scaled by RAdam's r_t.

    Default-on rectification eliminates the cold-start telemetry noise
    (cond_proxy=0, S=0) that Muogi v2 showed at steps 1-4 and prevents
    early-step spectral garbage from polluting the run.

    Designed for parameter groups whose gradient covariance does not
    factor as a Kronecker product, where cold-start variance estimates
    cannot be trusted (e.g., training from scratch on novel layers).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps_yogi: float = 1e-3,
        eps_adam: float = 1e-8,
        weight_decay: float = 0.0,
        ns5_freq: int = 3,
        ns5_max_iters: int = 5,    # Jordan canonical (rolled back from v2 attempt 4's 9)
        ns5_convergence_threshold: float = 0.64,   # = 0.8² spectral guarantee
        ns5_adaptive_trigger: bool = True,
        ns5_cond_ratio_threshold: float = 2.0,
        ns5_enabled: bool = True,
        spread_cap: float = 10.0,
        ramuogi_enabled: bool = True,
        initial_accumulator: float = 1e-6,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid RAMuogi learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid RAMuogi beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid RAMuogi beta2: {betas[1]}")
        if eps_yogi <= 0.0:
            raise ValueError(f"Invalid eps_yogi: {eps_yogi}")
        if eps_adam < 0.0:
            raise ValueError(f"Invalid eps_adam: {eps_adam}")
        if ns5_freq < 1:
            raise ValueError(f"Invalid ns5_freq: {ns5_freq}")
        if ns5_max_iters < 1:
            raise ValueError(f"Invalid ns5_max_iters: {ns5_max_iters}")
        if ns5_convergence_threshold <= 0.0:
            raise ValueError(
                f"Invalid ns5_convergence_threshold: {ns5_convergence_threshold}"
            )
        if ns5_cond_ratio_threshold <= 1.0:
            raise ValueError(
                f"Invalid ns5_cond_ratio_threshold (must be > 1): "
                f"{ns5_cond_ratio_threshold}"
            )
        if spread_cap <= 1.0:
            raise ValueError(
                f"Invalid spread_cap (must be > 1): {spread_cap}"
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps_yogi=eps_yogi,
            eps_adam=eps_adam,
            weight_decay=weight_decay,
            ns5_freq=ns5_freq,
            ns5_max_iters=ns5_max_iters,
            ns5_convergence_threshold=ns5_convergence_threshold,
            ns5_adaptive_trigger=ns5_adaptive_trigger,
            ns5_cond_ratio_threshold=ns5_cond_ratio_threshold,
            ns5_enabled=ns5_enabled,
            spread_cap=spread_cap,
            ramuogi_enabled=ramuogi_enabled,
            initial_accumulator=initial_accumulator,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _radam_rectification(t: int, beta2: float) -> Tuple[bool, float]:
        """RAdam variance-confidence gate (L4).

        Returns ``(warmed_up, r_t)``:
          - ``warmed_up``: True iff rho_t > 4 (variance trustworthy)
          - ``r_t``: rectification scale to multiply the final update
            by when warmed_up. Smooth ramp from ~0 just after warmup
            crossover to ~1 in steady state. When not warmed_up,
            ``r_t = 0.0`` (sentinel; caller uses momentum-only path).

        With β2=0.999, rho_inf ≈ 1999 and rho_t crosses 4 at step ≈ 5.
        """
        rho_inf = 2.0 / (1.0 - beta2) - 1.0
        beta2_t = beta2 ** t
        rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t)
        if rho_t <= 4.0:
            return False, 0.0
        r_t = (
            ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
            / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
        ) ** 0.5
        return True, r_t

    def _should_run_ns5(
        self,
        t: int,
        adapt_for_cond: torch.Tensor,
        state: dict,
        ns5_freq: int,
        adaptive: bool,
        ratio_threshold: float,
    ) -> bool:
        """Decide whether to run NS5 this step.

        Two triggers OR'd: time-based (every ns5_freq steps) and
        condition-based (row-norm max/min ratio jumps > ratio_threshold ×
        last seen). Step 1 always triggers so the first step gets a
        baseline NS5 attempt.
        """
        if t == 1:
            return True
        if t % ns5_freq == 0:
            return True
        if not adaptive:
            return False
        row_norms = adapt_for_cond.norm(dim=-1)
        cond_proxy = float(
            (row_norms.max() / (row_norms.min() + 1e-7)).item()
        )
        last_cond = state.get("last_cond_proxy", cond_proxy)
        state["last_cond_proxy"] = cond_proxy
        return cond_proxy > ratio_threshold * last_cond

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps_yogi = group["eps_yogi"]
            eps_adam = group["eps_adam"]
            wd = group["weight_decay"]
            ns5_freq = group["ns5_freq"]
            ns5_max_iters = group["ns5_max_iters"]
            ns5_threshold = group["ns5_convergence_threshold"]
            ns5_adaptive = group["ns5_adaptive_trigger"]
            ns5_ratio = group["ns5_cond_ratio_threshold"]
            ns5_enabled = group["ns5_enabled"]
            spread_cap = group["spread_cap"]
            ramuogi_enabled = group["ramuogi_enabled"]
            init_acc = group["initial_accumulator"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muogi does not support sparse gradients")

                state = self.state[p]
                # Lazy init.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.full_like(
                        p, init_acc, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.full_like(
                        p, init_acc, memory_format=torch.preserve_format
                    )
                    state["ns5_success_count"] = 0
                    state["ns5_skip_count"] = 0
                    state["last_ns5_step"] = 0
                    state["last_ns5_residual"] = 0.0
                    state["last_cond_proxy"] = 0.0
                    state["last_global_scale"] = 0.0
                    state["rectification_skip_count"] = 0
                    state["last_r_t"] = 0.0

                state["step"] += 1
                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Yogi moments — m_t multiplicative, v_t additive.
                exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                grad_sq = g * g
                exp_avg_sq.addcmul_(
                    torch.sign(exp_avg_sq - grad_sq),
                    grad_sq,
                    value=-(1.0 - beta2),
                )

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                # Decoupled weight decay (AdamW-style).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # ── L4: RAdam variance-rectification gate ────────────
                # Cold-start protection. When v_hat hasn't seen enough
                # samples (rho_t <= 4), skip the entire spectral
                # pipeline (R, NS5, S) and apply momentum-only update.
                # Disable via ramuogi_enabled=False for A/B vs Muogi v2.
                if ramuogi_enabled:
                    warmed_up, r_t = self._radam_rectification(t, beta2)
                    state["last_r_t"] = r_t
                    if not warmed_up:
                        state["rectification_skip_count"] += 1
                        p.add_(exp_avg, alpha=-lr / bias_correction1)
                        continue
                else:
                    r_t = 1.0   # RAMuogi disabled: no rectification scale

                use_ns5 = ns5_enabled and g.ndim == 2

                if not use_ns5:
                    # 1-D fallback OR NS5 globally disabled: vanilla Yogi.
                    m_hat = exp_avg / bias_correction1
                    v_hat = exp_avg_sq / bias_correction2
                    denom = v_hat.sqrt().clamp_(min=eps_yogi).add_(eps_adam)
                    p.addcdiv_(m_hat, denom, value=-lr * r_t)
                    continue

                # ── Unified Muogi 2-D update ─────────────────────────
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Yogi-derived scales: per-row vector R and global S.
                sqrt_v = v_hat.sqrt()
                # Per-row scale (option 2 + cheater's choice — injected
                # inside the NS5 loop).
                row_sqrt = sqrt_v.mean(dim=-1)                # (m,)
                R = 1.0 / (row_sqrt.clamp_(min=eps_yogi) + eps_adam)
                # Global scale (option 1 — applied at output as a
                # master throttle).
                global_sqrt = sqrt_v.mean()
                S = 1.0 / (float(global_sqrt.clamp_(min=eps_yogi).item()) + eps_adam)
                # Surface S to telemetry regardless of which branch
                # this step takes (success, safe-skip, or cached-direction
                # reuse all compute S the same way). Eliminates the
                # display artifact where telemetry rolled up 0.0 when
                # a param had only ever safe-skipped.
                state["last_global_scale"] = S

                if self._should_run_ns5(
                    t, m_hat, state, ns5_freq, ns5_adaptive, ns5_ratio
                ):
                    direction, converged, residual = _newton_schulz5_unified(
                        m_hat,
                        row_scale=R,
                        max_iters=ns5_max_iters,
                        convergence_threshold=ns5_threshold,
                        spread_cap=spread_cap,
                        eps_adam=eps_adam,
                    )
                    state["last_ns5_residual"] = residual
                    state["last_ns5_step"] = t
                    if not converged:
                        # Safe-skip: fall back to vanilla Yogi for this step.
                        # Scale by r_t (RAMuogi rectification).
                        state["ns5_skip_count"] += 1
                        denom = sqrt_v.clamp_(min=eps_yogi).add_(eps_adam)
                        p.addcdiv_(m_hat, denom, value=-lr * r_t)
                        continue
                    state["ns5_success_count"] += 1
                    # Cache the converged direction so off-schedule
                    # steps reuse it. Memory footprint matches Muon's
                    # momentum_buffer.
                    state["last_direction"] = direction
                    state["last_global_scale"] = S
                else:
                    direction = state.get("last_direction")
                    if direction is None or direction.shape != m_hat.shape:
                        # Cold start before any NS5 success: vanilla Yogi.
                        # Scale by r_t (RAMuogi rectification).
                        denom = sqrt_v.clamp_(min=eps_yogi).add_(eps_adam)
                        p.addcdiv_(m_hat, denom, value=-lr * r_t)
                        continue
                    # Reuse cached direction; refresh global scale from
                    # current v_hat (cheap, keeps throttle responsive).
                    state["last_global_scale"] = S

                # Muon shape scale + global throttle.
                m, n = direction.shape[-2], direction.shape[-1]
                shape_scale = max(1.0, m / n) ** 0.5

                # Apply: p -= lr * r_t * S * direction * shape_scale.
                # (r_t is RAMuogi's L4 rectification scale; S is the
                # Yogi global volatility throttle.)
                p.add_(direction, alpha=-lr * r_t * S * shape_scale)

        return loss

    # ── Telemetry ────────────────────────────────────────────────────
    def get_telemetry(self) -> dict:
        """Aggregate per-organ NS5 counters across all parameters.

        Returns:
            dict with keys:
              - ``ns5_success_count``: total successful NS5 calls
              - ``ns5_skip_count``: total non-convergent NS5 attempts
              - ``last_ns5_step``: max step at which NS5 last fired
              - ``last_ns5_residual``: residual from the most recent NS5 call
              - ``last_cond_proxy``: most recent row-norm conditioning proxy
              - ``last_global_scale``: most recent Yogi global throttle S
              - ``num_2d_params``: number of 2-D params tracked
        """
        total_ok = 0
        total_skip = 0
        total_rect_skip = 0
        last_step = 0
        last_residual = 0.0
        last_cond = 0.0
        last_global_scale = 0.0
        last_r_t = 0.0
        num_2d = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, {})
                if not state:
                    continue
                if p.ndim != 2:
                    continue
                num_2d += 1
                total_ok += state.get("ns5_success_count", 0)
                total_skip += state.get("ns5_skip_count", 0)
                total_rect_skip += state.get("rectification_skip_count", 0)
                # Track latest r_t from any 2-D param (all see same t).
                _rt = state.get("last_r_t", 0.0)
                if _rt > last_r_t:
                    last_r_t = _rt
                if state.get("last_ns5_step", 0) > last_step:
                    last_step = state["last_ns5_step"]
                    last_residual = state.get("last_ns5_residual", 0.0)
                    last_cond = state.get("last_cond_proxy", 0.0)
                    last_global_scale = state.get("last_global_scale", 0.0)
        return {
            "ns5_success_count": total_ok,
            "ns5_skip_count": total_skip,
            "rectification_skip_count": total_rect_skip,
            "last_ns5_step": last_step,
            "last_ns5_residual": last_residual,
            "last_cond_proxy": last_cond,
            "last_global_scale": last_global_scale,
            "last_r_t": last_r_t,
            "num_2d_params": num_2d,
        }

    def get_safety_counts(self) -> dict:
        """Return the L1-L5 safety-chain counter dict consumed by the bench harness.

        RAMuogi extends Muogi's chain with L4 (RAdam cold-start gate).
        L1/L3/L5 remain silent paths (degradations, not increments).
        """
        tel = self.get_telemetry()
        return {
            "l1": 0,
            "l2": tel["ns5_skip_count"],
            "l3": 0,
            "l4": tel["rectification_skip_count"],
            "l5": 0,
        }
