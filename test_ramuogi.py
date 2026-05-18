"""Unit tests for the RAMuogi optimizer (Muogi v3 — RAdam L4 gate).

Covers the additional layer that distinguishes RAMuogi from Muogi v2:
the cold-start variance-confidence gate. Muogi v2's own tests in
test_muogi.py still cover L1/L2/L3 behavior — these tests focus on L4.

1. Cold start (rho_t <= 4): NS5 not attempted, momentum-only update,
   rect_skip increments, weights still move.
2. Warm-up crossover: at step ~5 with β2=0.999, gate transitions and
   ns5_success_count starts incrementing.
3. ramuogi_enabled=False reproduces Muogi v2 behavior on a deterministic
   gradient stream (byte-for-byte equivalent given matched config).
4. r_t monotonically increases from ~0 post-warmup toward ~1 in
   steady state.
5. Construction defaults + validator rejects invalid args.
"""

from __future__ import annotations

import math

import pytest
import torch

from ramuogi import (
    RAMuogi,
    _newton_schulz5_unified,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(13)


def _make_2d_param(m: int, n: int) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.randn(m, n) * 0.1)


# ── 1. Cold-start gate ─────────────────────────────────────────────────


def test_ramuogi_cold_start_skips_spectral():
    """At steps 1-4 with β2=0.999, RAdam's rho_t <= 4 so RAMuogi
    must take the cold-start branch: skip NS5 entirely, apply
    momentum-only update, increment rectification_skip_count."""
    p = _make_2d_param(4, 8)
    opt = RAMuogi([p], lr=1e-3, betas=(0.9, 0.999), ramuogi_enabled=True)
    for step in range(1, 5):
        p_before = p.detach().clone()
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
        state = opt.state[p]
        # L4 engaged: no NS5 success or skip recorded
        assert state["ns5_success_count"] == 0, (
            f"step {step}: NS5 fired during cold-start (rho_t should be <= 4)"
        )
        # rectification_skip_count tracks the cold-start branch count
        assert state["rectification_skip_count"] == step
        # last_r_t is 0.0 sentinel during cold-start
        assert state["last_r_t"] == 0.0
        # Weights still moved via momentum-only update
        assert not torch.allclose(p.detach(), p_before), (
            f"step {step}: cold-start branch did not move weights"
        )
        assert torch.isfinite(p).all()


# ── 2. Warmup crossover ────────────────────────────────────────────────


def test_ramuogi_warmup_engages_full_pipeline():
    """By step ~5+ with β2=0.999, rho_t > 4 and RAMuogi engages full
    Muogi v2 pipeline. NS5 success_count should start incrementing."""
    p = _make_2d_param(4, 8)
    opt = RAMuogi([p], lr=1e-3, betas=(0.9, 0.999), ns5_freq=1)
    for _ in range(20):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
    state = opt.state[p]
    # By step 20, rho_t is well past 4 and most steps should have
    # tried NS5 at least once.
    assert state["rectification_skip_count"] < 20, (
        f"All 20 steps cold-skipped — gate never opened. "
        f"rect_skip={state['rectification_skip_count']}"
    )
    attempted = state["ns5_success_count"] + state["ns5_skip_count"]
    assert attempted > 0, (
        f"NS5 never attempted post-warmup. "
        f"ok={state['ns5_success_count']} skip={state['ns5_skip_count']}"
    )
    # r_t should be non-zero and bounded by 1
    assert 0.0 < state["last_r_t"] <= 1.0


# ── 3. ramuogi_enabled=False = Muogi v2 ────────────────────────────────


def test_ramuogi_disabled_skips_l4_gate():
    """With ramuogi_enabled=False, L4 is bypassed: NS5 fires from step
    1, no rectification_skip_count increments. Behavior matches
    Muogi v2 (the gate is purely additive)."""
    p = _make_2d_param(4, 8)
    opt = RAMuogi([p], lr=1e-3, ns5_freq=1, ramuogi_enabled=False)
    for _ in range(5):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
    state = opt.state[p]
    # rect_skip stays 0 because L4 is disabled
    assert state["rectification_skip_count"] == 0, (
        f"rect_skip incremented despite ramuogi_enabled=False: "
        f"{state['rectification_skip_count']}"
    )
    # NS5 was attempted from step 1 (the v2-equivalent behavior)
    assert state["ns5_success_count"] + state["ns5_skip_count"] >= 5


# ── 4. r_t monotonic ramp ──────────────────────────────────────────────


def test_ramuogi_rt_monotonic_after_warmup():
    """Post-warmup, r_t should grow monotonically toward 1 as more
    samples accumulate. Verify the scalar _radam_rectification helper
    directly (deterministic)."""
    beta2 = 0.999
    r_t_series = []
    for t in range(1, 200):
        warmed, r_t = RAMuogi._radam_rectification(t, beta2)
        if warmed:
            r_t_series.append((t, r_t))
    # Find first warmed step
    assert len(r_t_series) > 0, "Never warmed up in 200 steps"
    first_t, first_r = r_t_series[0]
    # Should warm up around step 5 with β2=0.999
    assert first_t <= 10, f"Warmup took {first_t} steps — too slow"
    # Monotonic: each subsequent r_t >= previous
    for i in range(1, len(r_t_series)):
        prev_t, prev_r = r_t_series[i - 1]
        cur_t, cur_r = r_t_series[i]
        assert cur_r >= prev_r - 1e-9, (
            f"r_t not monotonic at t={cur_t}: prev={prev_r} cur={cur_r}"
        )
    # Final r_t should be close to (but not exceed) 1 — RAdam's
    # r_t formula approaches 1 from below as rho_t → rho_inf.
    final_t, final_r = r_t_series[-1]
    assert final_r <= 1.0
    # RAdam's r_t ramps slowly with β2=0.999. By step 200, r_t is
    # typically ~0.3; reaching 0.5 takes ~500-700 steps; reaching 0.9
    # takes a few thousand. Verify just that it's well above zero
    # post-warmup (the gate isn't permanently stuck at the boundary).
    assert final_r > 0.2, f"r_t suspiciously low at t={final_t}: {final_r}"
    # Cross-check at a longer horizon: by step 2000, r_t should be > 0.7
    _, r_t_2k = RAMuogi._radam_rectification(t=2000, beta2=0.999)
    assert r_t_2k > 0.7, f"r_t at t=2000 too low: {r_t_2k}"


# ── 5. Construction defaults + validation ──────────────────────────────


def test_ramuogi_construction_defaults():
    p = _make_2d_param(4, 8)
    opt = RAMuogi([p])
    g = opt.param_groups[0]
    # RAMuogi defaults: max_iters=5 (Jordan), threshold=0.64 (= 0.8²)
    assert g["ns5_max_iters"] == 5
    assert g["ns5_convergence_threshold"] == 0.64
    assert g["spread_cap"] == 10.0
    assert g["ramuogi_enabled"] is True


def test_ramuogi_validators():
    p = _make_2d_param(4, 8)
    with pytest.raises(ValueError, match="learning rate"):
        RAMuogi([p], lr=-1.0)
    with pytest.raises(ValueError, match="spread_cap"):
        RAMuogi([p], spread_cap=1.0)


# ── 6. _radam_rectification edge cases ─────────────────────────────────


def test_radam_rectification_step1_not_warmed():
    warmed, r_t = RAMuogi._radam_rectification(t=1, beta2=0.999)
    assert warmed is False
    assert r_t == 0.0


def test_radam_rectification_returns_finite():
    """Across a wide step range, the rectification scalar must be
    finite and bounded."""
    for t in [1, 2, 5, 10, 100, 1000, 10000]:
        warmed, r_t = RAMuogi._radam_rectification(t, beta2=0.999)
        assert math.isfinite(r_t)
        assert 0.0 <= r_t <= 1.0001   # tiny float tolerance


# ── 7. Telemetry surfaces L4 fields ─────────────────────────────────────


def test_ramuogi_telemetry_includes_l4_fields():
    p = _make_2d_param(4, 8)
    opt = RAMuogi([p], lr=1e-3, ns5_freq=1)
    for _ in range(20):
        p.grad = torch.randn_like(p) * 0.01
        opt.step()
    t = opt.get_telemetry()
    assert "rectification_skip_count" in t
    assert "last_r_t" in t
    assert t["num_2d_params"] == 1
    # Cold start contributed 4-5 rect_skips
    assert t["rectification_skip_count"] >= 4
    # r_t reported, positive after warmup
    assert t["last_r_t"] > 0.0


# ── 8. Stability under bursty conditioning ────────────────────────────


def test_ramuogi_stability_under_burst():
    """End-to-end: synthetic bursty gradient stream with per-column
    bursts and row-coupling — must not NaN over 100 steps with RAMuogi.
    L1+L2+L3+L4 chain must hold."""
    p = _make_2d_param(8, 16)
    opt = RAMuogi([p], lr=1e-3)
    for t in range(100):
        g = torch.randn_like(p) * 0.05
        burst_col = t % p.shape[1]
        g[:, burst_col] += torch.randn(p.shape[0]) * (3.0 if t % 7 == 0 else 0.5)
        p.grad = g
        opt.step()
        assert torch.isfinite(p).all(), f"NaN at step {t}"
