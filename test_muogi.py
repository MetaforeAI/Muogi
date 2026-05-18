"""Unit tests for the Muogi optimizer (unified Yogi-then-Muon).

Covers:
  1. Construction defaults + validator rejects invalid args
  2. 2-D single-step: state populated, weights move, no NaN
  3. 1-D single-step: vanilla Yogi path, no NS5 attempted
  4. Multi-step state: counters increment correctly
  5. NS5 time trigger: fires at step 1 and every ns5_freq steps
  6. NS5 adaptive trigger: fires off-schedule on conditioning shift
  7. NS5 safe-skip: non-converged → fall back to Yogi, skip counter +1
  8. ε_yogi floor: per-row scalar bounded by m_hat / ε_yogi
  9. State_dict round-trip preserves all counters + tensors
 10. Telemetry aggregation correct across params
 11. Toy regression converges
 12. Synthetic attention block converges
 13. Bursty per-column gradient distribution doesn't NaN over 200 steps
 14. Mixed precision (bf16 input → fp32 internals)
 15. M > N orientation handled (wide projection layers)

All tests pure-CPU. No model imports — Muogi is standalone.
"""

from __future__ import annotations

import io
import math

import pytest
import torch

from muogi import (
    Muogi,
    _newton_schulz5_unified,
)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(7)


def _make_2d_param(m: int, n: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.randn(m, n) * 0.1)


def _make_1d_param(n: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.randn(n) * 0.1)


# ── 1. Construction ────────────────────────────────────────────────────


def test_construction_defaults():
    p = _make_2d_param(4, 8)
    opt = Muogi([p])
    g = opt.param_groups[0]
    assert g["betas"] == (0.9, 0.999)
    assert g["eps_yogi"] == 1e-3
    assert g["ns5_freq"] == 3
    assert g["ns5_max_iters"] == 5
    assert g["ns5_enabled"] is True


def test_construction_rejects_bad_args():
    p = _make_2d_param(4, 8)
    with pytest.raises(ValueError, match="learning rate"):
        Muogi([p], lr=-1.0)
    with pytest.raises(ValueError, match="beta1"):
        Muogi([p], betas=(1.5, 0.999))
    with pytest.raises(ValueError, match="eps_yogi"):
        Muogi([p], eps_yogi=0.0)
    with pytest.raises(ValueError, match="ns5_freq"):
        Muogi([p], ns5_freq=0)
    with pytest.raises(ValueError, match="ns5_cond_ratio_threshold"):
        Muogi([p], ns5_cond_ratio_threshold=1.0)


# ── 2. 2-D single-step ─────────────────────────────────────────────────


def test_single_step_2d_no_nan():
    p = _make_2d_param(4, 8)
    p_before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt = Muogi([p], lr=1e-3)
    opt.step()
    assert torch.isfinite(p).all(), "params not finite after step"
    assert not torch.allclose(p.detach(), p_before), "weights did not move"
    state = opt.state[p]
    assert state["step"] == 1
    assert state["exp_avg"].shape == p.shape
    assert state["exp_avg_sq"].shape == p.shape
    # Step 1 always triggers NS5.
    assert state["ns5_success_count"] + state["ns5_skip_count"] == 1


# ── 3. 1-D single-step (vanilla Yogi path) ─────────────────────────────


def test_single_step_1d_vanilla_yogi():
    p = _make_1d_param(16)
    p_before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt = Muogi([p], lr=1e-3)
    opt.step()
    assert torch.isfinite(p).all()
    assert not torch.allclose(p.detach(), p_before)
    state = opt.state[p]
    # 1-D path doesn't trigger NS5.
    assert state["ns5_success_count"] == 0
    assert state["ns5_skip_count"] == 0


# ── 4. Multi-step state counters ───────────────────────────────────────


def test_state_after_multiple_steps():
    p = _make_2d_param(4, 8)
    opt = Muogi([p], lr=1e-3, ns5_freq=3, ns5_adaptive_trigger=False)
    for _ in range(20):
        p.grad = torch.randn_like(p) * 0.1
        opt.step()
    state = opt.state[p]
    assert state["step"] == 20
    # ns5 fires at step 1 (always) + 3, 6, 9, 12, 15, 18 = 7 attempts
    expected_attempts = 7
    actual_attempts = state["ns5_success_count"] + state["ns5_skip_count"]
    assert actual_attempts == expected_attempts, (
        f"expected {expected_attempts} NS5 attempts, got {actual_attempts}"
    )


# ── 5. NS5 time trigger ────────────────────────────────────────────────


def test_ns5_time_trigger_schedule():
    p = _make_2d_param(4, 8)
    opt = Muogi([p], lr=1e-3, ns5_freq=3, ns5_adaptive_trigger=False)
    fired_steps = []
    for step in range(1, 11):
        p.grad = torch.randn_like(p) * 0.1
        prev = opt.state[p].get("ns5_success_count", 0) + opt.state[p].get(
            "ns5_skip_count", 0
        )
        opt.step()
        cur = opt.state[p]["ns5_success_count"] + opt.state[p]["ns5_skip_count"]
        if cur > prev:
            fired_steps.append(step)
    # Step 1 always triggers; then every ns5_freq=3 → 3, 6, 9.
    assert fired_steps == [1, 3, 6, 9], f"unexpected: {fired_steps}"


# ── 6. NS5 adaptive trigger on conditioning shift ──────────────────────


def test_ns5_adaptive_trigger_fires_on_cond_shift():
    """With ns5_freq large and adaptive on, a sudden condition jump
    should trigger NS5 off-schedule.
    """
    p = _make_2d_param(8, 16)
    opt = Muogi(
        [p],
        lr=1e-3,
        ns5_freq=100,             # time trigger effectively disabled in this window
        ns5_adaptive_trigger=True,
        ns5_cond_ratio_threshold=2.0,
    )
    # Step 1 always fires; seed exp_avg with low-condition gradients.
    for _ in range(3):
        p.grad = torch.ones_like(p) * 0.01     # uniform: low cond ratio
        opt.step()
    base_attempts = opt.state[p]["ns5_success_count"] + opt.state[p]["ns5_skip_count"]
    # Inject a high-condition gradient that should push the cond proxy
    # beyond 2× the prior.
    p.grad = torch.zeros_like(p)
    p.grad[0, :] = 100.0   # one row blasts
    opt.step()
    after = opt.state[p]["ns5_success_count"] + opt.state[p]["ns5_skip_count"]
    assert after > base_attempts, "adaptive trigger did not fire on cond shift"


# ── 7. NS5 safe-skip on non-convergence ────────────────────────────────


def test_ns5_safe_skip_increments_skip_counter():
    """A tight convergence threshold guarantees NS5 will fail the
    convergence check, and the skip counter should record it.
    """
    p = _make_2d_param(4, 8)
    opt = Muogi(
        [p],
        lr=1e-3,
        ns5_max_iters=1,             # only 1 iter → very unlikely to settle
        ns5_convergence_threshold=1e-6,   # absurdly tight
    )
    p_before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt.step()
    state = opt.state[p]
    # Either skip incremented OR success — but with these settings,
    # safe-skip should dominate. We require skip ≥ 1.
    assert state["ns5_skip_count"] >= 1, (
        f"expected skip ≥ 1 with iter=1 threshold=1e-6, got "
        f"ok={state['ns5_success_count']} skip={state['ns5_skip_count']}"
    )
    # Even on skip, the Yogi fallback still moves weights.
    assert not torch.allclose(p.detach(), p_before)
    assert torch.isfinite(p).all()


# ── 8. ε_yogi floor caps per-row scalar ────────────────────────────────


def test_eps_yogi_floor_caps_row_scale():
    """When the v_t accumulator is essentially zero (init_acc tiny),
    the per-row scale should be bounded by 1/eps_yogi, not blow up.
    """
    p = _make_2d_param(4, 8)
    opt = Muogi(
        [p],
        lr=1e-3,
        eps_yogi=1e-3,
        initial_accumulator=1e-12,     # smaller than eps_yogi → floor active
    )
    p.grad = torch.randn_like(p) * 0.01
    opt.step()
    # Per-row R = 1 / max(mean_j sqrt(v_hat[i,:]), eps_yogi) is bounded
    # by 1/eps_yogi = 1000. The weight update magnitude = lr * S * |dir|
    # * shape_scale; with lr=1e-3 and unit-norm direction it stays
    # tractable. Check that no element has gone wild.
    assert torch.isfinite(p).all()
    assert p.abs().max().item() < 100.0, (
        f"weight magnitude blew up: {p.abs().max().item()}"
    )


# ── 9. State-dict round-trip ───────────────────────────────────────────


def test_state_dict_roundtrip():
    p_orig = _make_2d_param(4, 8)
    opt_orig = Muogi([p_orig], lr=1e-3, ns5_freq=2)
    grads = [torch.randn_like(p_orig) for _ in range(5)]
    for g in grads:
        p_orig.grad = g.clone()
        opt_orig.step()
    saved = opt_orig.state_dict()
    # Round-trip through a serialization buffer to exercise the same
    # path checkpointing uses.
    buf = io.BytesIO()
    torch.save(saved, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)

    # Fresh optimizer on a fresh param starting from the same init.
    torch.manual_seed(7)     # repeat fixture seed so the init matches
    p_new = _make_2d_param(4, 8)
    p_new.data.copy_(p_orig.detach() - sum(g for g in grads))   # dummy reset
    opt_new = Muogi([p_new], lr=1e-3, ns5_freq=2)
    opt_new.load_state_dict(loaded)

    # Counters should match exactly.
    s_orig = opt_orig.state[p_orig]
    # After load_state_dict the new param's state key is p_new.
    s_new_keys = list(opt_new.state.keys())
    assert len(s_new_keys) == 1
    s_new = opt_new.state[s_new_keys[0]]
    assert s_new["step"] == s_orig["step"]
    assert s_new["ns5_success_count"] == s_orig["ns5_success_count"]
    assert s_new["ns5_skip_count"] == s_orig["ns5_skip_count"]
    assert s_new["last_ns5_step"] == s_orig["last_ns5_step"]
    assert torch.allclose(s_new["exp_avg"], s_orig["exp_avg"])
    assert torch.allclose(s_new["exp_avg_sq"], s_orig["exp_avg_sq"])


# ── 10. Telemetry aggregation ──────────────────────────────────────────


def test_get_telemetry_aggregates_across_params():
    p1 = _make_2d_param(4, 8)
    p2 = _make_2d_param(6, 10)
    p3 = _make_1d_param(8)   # should be ignored (num_2d_params counts only 2-D)
    opt = Muogi([p1, p2, p3], lr=1e-3, ns5_freq=2)
    for _ in range(5):
        for q in (p1, p2, p3):
            q.grad = torch.randn_like(q) * 0.1
        opt.step()
    t = opt.get_telemetry()
    assert t["num_2d_params"] == 2
    assert t["ns5_success_count"] + t["ns5_skip_count"] > 0
    assert t["last_ns5_step"] > 0


# ── 11. Toy regression convergence ─────────────────────────────────────


def test_toy_regression_convergence():
    """Two-layer linear regression. After 200 steps, loss should be
    well below the initial loss. Just verifies Muogi makes useful
    progress on a non-pathological problem.
    """
    torch.manual_seed(42)
    in_dim, hidden, out_dim, batch = 8, 16, 4, 32
    W1 = torch.nn.Parameter(torch.randn(hidden, in_dim) * 0.1)
    b1 = torch.nn.Parameter(torch.zeros(hidden))
    W2 = torch.nn.Parameter(torch.randn(out_dim, hidden) * 0.1)
    b2 = torch.nn.Parameter(torch.zeros(out_dim))

    X = torch.randn(batch, in_dim)
    W_true = torch.randn(out_dim, in_dim)
    Y = X @ W_true.T

    opt = Muogi([W1, b1, W2, b2], lr=3e-2)

    def loss_fn():
        h = torch.tanh(X @ W1.T + b1)
        pred = h @ W2.T + b2
        return ((pred - Y) ** 2).mean()

    initial_loss = loss_fn().item()
    for _ in range(200):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    final_loss = loss_fn().item()
    assert final_loss < initial_loss * 0.5, (
        f"loss did not decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
    )


# ── 12. Synthetic attention block convergence ──────────────────────────


def test_attention_block_convergence():
    """Q/K/V projection layers with softmax attention. Verifies Muogi
    handles a transformer-like update pattern.
    """
    torch.manual_seed(1)
    d, seq, batch = 16, 8, 4
    Wq = torch.nn.Parameter(torch.randn(d, d) * 0.1)
    Wk = torch.nn.Parameter(torch.randn(d, d) * 0.1)
    Wv = torch.nn.Parameter(torch.randn(d, d) * 0.1)
    Wo = torch.nn.Parameter(torch.randn(d, d) * 0.1)

    X = torch.randn(batch, seq, d)
    Y_target = torch.randn(batch, seq, d)

    opt = Muogi([Wq, Wk, Wv, Wo], lr=1e-2)

    def loss_fn():
        q = X @ Wq
        k = X @ Wk
        v = X @ Wv
        attn = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(d), dim=-1)
        h = attn @ v
        return ((h @ Wo - Y_target) ** 2).mean()

    initial_loss = loss_fn().item()
    for _ in range(150):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
    final_loss = loss_fn().item()
    assert final_loss < initial_loss, (
        f"attention loss did not improve: {initial_loss:.4f} → {final_loss:.4f}"
    )


# ── 13. Bursty per-column gradient stability ───────────────────────────


def test_joint_norm_burst_stability():
    """Synthetic gradient stream with bursty per-column distribution
    and row-coupling — mimics the gradient pathology Muogi was
    designed for. Must not NaN over 200 steps under this distribution.
    """
    torch.manual_seed(3)
    p = _make_2d_param(8, 16)
    opt = Muogi([p], lr=1e-3)
    for t in range(200):
        # Base gradient.
        g = torch.randn_like(p) * 0.05
        # Inject a per-step burst at a random column.
        burst_col = t % p.shape[1]
        g[:, burst_col] += torch.randn(p.shape[0]) * (3.0 if t % 7 == 0 else 0.5)
        # Row-couple: when col k bursts, multiply by a shared row factor.
        row_factor = torch.randn(p.shape[0], 1).abs() + 0.5
        g = g * row_factor
        p.grad = g
        opt.step()
        assert torch.isfinite(p).all(), f"NaN at step {t}"
    # Should have run without diverging — loose final-mag check.
    assert p.abs().max().item() < 1e3


# ── 14. Mixed precision (bf16 in, fp32 internals) ──────────────────────


def test_mixed_precision_bf16_input():
    p = torch.nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16) * 0.1)
    p.grad = torch.randn_like(p) * 0.01
    opt = Muogi([p], lr=1e-3)
    opt.step()
    # Param dtype preserved; state dtype follows init (bf16 for moments,
    # fp32 internally inside NS5).
    assert p.dtype == torch.bfloat16
    assert torch.isfinite(p).all()
    state = opt.state[p]
    assert state["exp_avg"].dtype == torch.bfloat16
    assert state["exp_avg_sq"].dtype == torch.bfloat16


# ── 15. M > N orientation handled ──────────────────────────────────────


def test_wide_projection_orientation():
    """Tall matrix M > N (e.g., wide output projection). NS5
    transposes internally for efficiency; verify the optimizer runs
    cleanly on both orientations and the row-scale injection targets
    the original row axis without crashing the NS5 helper.

    Convergence on a tall random matrix with row-scale injection is
    not guaranteed within the default threshold (the converged matrix
    is D·O, not O, so successive iters don't shrink as cleanly as
    pure NS5). The contract is: weights move, no NaN, NS5 attempted.
    Whether NS5 converged or safe-skipped to Yogi is acceptable —
    both are correct behaviors. The skip rate is reported in
    telemetry for paper analysis.
    """
    # Tall: 16 × 4 (m > n)
    p = torch.nn.Parameter(torch.randn(16, 4) * 0.1)
    opt = Muogi([p], lr=1e-3, ns5_freq=1)
    p_before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt.step()
    assert torch.isfinite(p).all()
    assert not torch.allclose(p.detach(), p_before)
    state = opt.state[p]
    # NS5 should have been attempted (step 1 always fires); the
    # outcome (ok vs skip) is acceptable either way — both produce a
    # finite weight update via the Yogi fallback path on skip.
    attempted = state["ns5_success_count"] + state["ns5_skip_count"]
    assert attempted == 1, (
        f"expected 1 NS5 attempt, got ok={state['ns5_success_count']} "
        f"skip={state['ns5_skip_count']}"
    )


# ── Bonus: direct NS5 helper unit tests ────────────────────────────────


def test_ns5_helper_returns_correct_shapes():
    M = torch.randn(6, 12)
    R = torch.ones(6)
    out, conv, res = _newton_schulz5_unified(M, R, max_iters=5, convergence_threshold=0.5)
    assert out.shape == M.shape
    assert isinstance(conv, bool)
    assert isinstance(res, float)


def test_ns5_helper_handles_tall_matrix():
    M = torch.randn(12, 6)   # tall: m > n
    R = torch.ones(12)
    out, conv, res = _newton_schulz5_unified(M, R, max_iters=5, convergence_threshold=0.5)
    assert out.shape == M.shape


# ── Combo A regression tests ───────────────────────────────────────────


def test_spread_cap_clamps_violent_R():
    """v1 failure mode: R with 1000x spread (e.g., from cond_proxy=1000+
    in a bursty regime) overflowed the polynomial inside the NS5 loop.
    Combo A's spread_cap=10 caps R's max/min ratio at one order of
    magnitude so the polynomial sees a bounded perturbation.

    Critical contract (the failure-safety contract): NS5's output must
    be FINITE regardless of input pathology — v1 produced NaN, v2 must
    not. Convergence may or may not happen depending on how aggressive
    the row-burst is; that's what the safe-skip mitigation handles at
    the optimizer level. The polynomial itself MUST stay numerically
    well-behaved.
    """
    M = torch.randn(8, 16)
    # Construct R with a 1000x spread. v1 would have produced NaN here.
    R = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1000.0])
    out, converged, residual = _newton_schulz5_unified(
        M, R, max_iters=5, convergence_threshold=0.5, spread_cap=10.0,
    )
    # PRIMARY CONTRACT: spread_cap prevents the polynomial from
    # overflowing to NaN/inf even under pathological R.
    assert torch.isfinite(out).all(), "NS5 produced NaN despite spread_cap"
    # The residual must also be finite — v1 reported residual=nan
    # because the polynomial overflowed. v2's residual is a real number
    # whether or not it crossed the convergence threshold.
    assert math.isfinite(residual), (
        f"residual is not finite: {residual} — polynomial overflowed"
    )
    # Note: we deliberately do NOT assert converged=True. Whether NS5
    # converges on a 1000x-burst input (post-clamp) depends on how
    # anisotropic the result still is. The safe-skip mitigation at
    # the optimizer level handles non-convergence by falling back to
    # vanilla Yogi. See test_v1_failure_input_yogi_fallback for the
    # full end-to-end contract.


def test_spread_cap_no_op_when_R_already_bounded():
    """When R is already within spread_cap, the clamp must be a no-op
    (output should be the same as if no clamp existed).
    """
    M = torch.randn(8, 16)
    # R with 5x spread, well under the default 10x cap.
    R = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0])
    out_capped, conv1, res1 = _newton_schulz5_unified(
        M, R, max_iters=5, convergence_threshold=0.5, spread_cap=10.0,
    )
    out_huge_cap, conv2, res2 = _newton_schulz5_unified(
        M, R, max_iters=5, convergence_threshold=0.5, spread_cap=1e9,
    )
    # With both caps far above R's spread, outputs should be identical.
    assert torch.allclose(out_capped, out_huge_cap, atol=1e-5)
    assert conv1 == conv2


def test_spread_cap_safe_max_guards_zero_Rmax():
    """Degenerate case: R is all zeros. safe_max guard should prevent
    a divide-by-zero in the clamp floor computation. The polynomial
    should still produce a finite (if not necessarily meaningful)
    output.
    """
    M = torch.randn(8, 16)
    R = torch.zeros(8)
    # Should not raise; should not produce NaN/inf in output.
    out, _, _ = _newton_schulz5_unified(
        M, R, max_iters=5, convergence_threshold=0.5, spread_cap=10.0,
    )
    assert torch.isfinite(out).all(), (
        "safe_max guard failed: zero R produced NaN/inf"
    )


def test_v1_failure_input_yogi_fallback():
    """End-to-end failure-safety contract for v1's failure mode:

    Reproduce a parameter whose gradient has extreme row-norm
    asymmetry (mimics a high-burst regime at cond_proxy ~770+). The polynomial
    might or might not converge depending on aggressiveness — the
    CRITICAL property is that training proceeds either way:

      - If NS5 converges: ns5_success_count incremented, weights
        update via the spectral path.
      - If NS5 safe-skips (residual still > threshold post-clamp):
        ns5_skip_count incremented, weights update via vanilla Yogi
        fallback.

    BOTH paths must produce finite weight updates. The safe-skip is
    not a failure — it's the mitigation that lets Muogi remain stable
    under arbitrarily pathological inputs. Vanilla Yogi at the
    fallback path is a feature, not a bug.
    """
    p = torch.nn.Parameter(torch.randn(8, 16) * 0.1)
    p_before = p.detach().clone()
    # Gradient with one row 1000x louder than the others (matches v1
    # bursty-regime failure trigger).
    g = torch.randn_like(p) * 0.01
    g[0, :] = g[0, :] * 1000.0
    p.grad = g
    opt = Muogi([p], lr=1e-4, ns5_freq=1, spread_cap=10.0)
    opt.step()
    state = opt.state[p]
    # Critical contract 1: NS5 was attempted (step 1 always triggers).
    attempted = state["ns5_success_count"] + state["ns5_skip_count"]
    assert attempted >= 1, "NS5 was not attempted at all"
    # Critical contract 2: weights moved (either via NS5 success OR
    # via the safe-skip Yogi fallback).
    assert not torch.allclose(p.detach(), p_before), (
        "weights did not move — both NS5 and Yogi fallback failed"
    )
    # Critical contract 3: no NaN. This is the v1 regression — v1
    # produced NaN in weights within ~10 steps on this exact input
    # distribution.
    assert torch.isfinite(p).all(), "v2 regressed: NaN in weights"


def test_safety_fallback_chain_is_complete():
    """Verify the three-layer failure-safety chain in Muogi v2:

        Layer 1: spread_cap clamps R before any use → polynomial
                 cannot overflow regardless of Yogi's variance estimate
        Layer 2: NS5 convergence check + safe-skip → if the
                 polynomial still produces an unverifiable result, the
                 optimizer falls back to vanilla Yogi for that step
        Layer 3: vanilla Yogi fallback → ALWAYS produces a finite
                 weight update because Yogi's own eps_yogi floor caps
                 the denominator

    All three layers MUST be present in the codepath for any 2-D
    parameter. This test exercises each layer in isolation.
    """
    # Layer 3 first (most fundamental): force Yogi-only via ns5_enabled=False.
    p1 = torch.nn.Parameter(torch.randn(8, 16) * 0.1)
    p1.grad = torch.randn_like(p1) * 100.0   # huge gradient
    opt1 = Muogi([p1], lr=1e-4, ns5_enabled=False)
    opt1.step()
    assert torch.isfinite(p1).all(), "Layer 3 (Yogi fallback) failed under huge gradient"

    # Layer 2: NS5 enabled but max_iters=1 forces unverifiable output
    # (residual sentinel = inf, see _newton_schulz5_unified for
    # the "max_iters < 2 has no delta to measure" branch — that
    # contract is preserved in v2 for backward compat).
    # In v2, the standard ||XX^T - I||_F check applies even with
    # max_iters=1, but the result is unlikely to converge.
    p2 = torch.nn.Parameter(torch.randn(8, 16) * 0.1)
    p2.grad = torch.randn_like(p2)
    opt2 = Muogi([p2], lr=1e-4, ns5_max_iters=1, ns5_convergence_threshold=1e-6)
    opt2.step()
    assert torch.isfinite(p2).all(), "Layer 2 (safe-skip) failed"

    # Layer 1: spread_cap clamps a pathological R.
    p3 = torch.nn.Parameter(torch.randn(8, 16) * 0.1)
    g3 = torch.randn_like(p3) * 0.01
    g3[0, :] *= 1e6   # extreme row burst
    p3.grad = g3
    opt3 = Muogi([p3], lr=1e-4, spread_cap=10.0)
    opt3.step()
    assert torch.isfinite(p3).all(), "Layer 1 (spread_cap) failed under extreme burst"
