"""Phase 1 sanity tests for the Muogi benchmark infrastructure.

These tests do NOT depend on Phase 2 problem modules. They use small,
self-contained ``BenchProblem`` subclasses defined inline to verify the
contract, the wrapper dispatch (including the Naive Yogi-Muon
anti-baseline), the CSV schema, the plot-stub error messages, and
reproducibility under a fixed seed.

Run with::

    pytest Muogi/bench/tests/test_infrastructure.py
"""

from __future__ import annotations

import math
from typing import List

import pandas as pd
import pytest
import torch

from bench.optimizers.wrappers import KNOWN_OPTIMIZERS, build_optimizer
from bench.optimizers.naive_yogi_muon import NaiveYogiMuon, _naive_ns5
from bench.problems.base import BenchProblem
from bench.run_bench import CSV_COLUMNS, run_one
from bench import plot_bench


# ---------------------------------------------------------------------------
# Inline test problem.
# ---------------------------------------------------------------------------


class _TinyQuadratic(BenchProblem):
    """Minimal axis-aligned quadratic: f(w) = 0.5 * sum(w**2).

    Used only to exercise the contract in tests. Not a Phase 2 problem.
    """

    name = "test_tiny_quadratic"
    max_steps = 50
    converged_tol = 1e-6

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        w = torch.randn(4, generator=gen, dtype=torch.float64)
        w.requires_grad_(True)
        return [w]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (w,) = params
        return 0.5 * (w * w).sum()


class _TinyMatrix(BenchProblem):
    """2-D matrix problem to exercise NS5-eligible parameters.

    f(W) = 0.5 * ||W - target||_F^2 where target is a fixed seeded matrix.
    """

    name = "test_tiny_matrix"
    max_steps = 30
    converged_tol = 1e-6

    def init_params(self) -> List[torch.Tensor]:
        gen = self._generator
        W = torch.randn(4, 4, generator=gen, dtype=torch.float32)
        W.requires_grad_(True)
        self._target = torch.randn(4, 4, generator=gen, dtype=torch.float32)
        return [W]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (W,) = params
        return 0.5 * ((W - self._target) ** 2).sum()


# ---------------------------------------------------------------------------
# Contract — BenchProblem cannot be instantiated abstractly.
# ---------------------------------------------------------------------------


def test_benchproblem_is_abstract():
    with pytest.raises(TypeError):
        BenchProblem(seed=0)  # type: ignore[abstract]


def test_benchproblem_subclass_missing_init_params_fails():
    class _BadProblem(BenchProblem):
        name = "bad"
        max_steps = 1
        converged_tol = 0.0

    with pytest.raises(TypeError):
        _BadProblem(seed=0)  # type: ignore[abstract]


def test_benchproblem_default_loss_and_grad_uses_autograd():
    problem = _TinyQuadratic(seed=0)
    params = problem.init_params()
    loss, grads = problem.loss_and_grad(params)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert len(grads) == 1
    assert grads[0].shape == params[0].shape
    assert torch.allclose(grads[0], params[0].detach())


def test_benchproblem_converged_default():
    problem = _TinyQuadratic(seed=0)
    assert problem.converged(1e-9, step=10) is True
    assert problem.converged(1.0, step=10) is False


# ---------------------------------------------------------------------------
# build_optimizer — dispatch table
# ---------------------------------------------------------------------------


_IMPLEMENTED_BASELINES = ("adam", "adamw", "yogi", "naive_yogi_muon")


@pytest.mark.parametrize("name", _IMPLEMENTED_BASELINES)
def test_build_optimizer_returns_optimizer(name: str):
    params = [torch.randn(3, requires_grad=True)]
    opt = build_optimizer(name, params, lr=1e-3)
    assert isinstance(opt, torch.optim.Optimizer)


@pytest.mark.parametrize("name", ("muon", "lion", "sophia", "soap"))
def test_build_optimizer_not_vendored_raises(name: str):
    params = [torch.randn(3, requires_grad=True)]
    with pytest.raises(NotImplementedError) as excinfo:
        build_optimizer(name, params, lr=1e-3)
    assert "README" in str(excinfo.value) or "vendored" in str(excinfo.value)


@pytest.mark.parametrize("name", ("muogi", "ramuogi"))
def test_build_optimizer_muogi_lazy_imports(name: str):
    """Muogi/RAMuogi import lazily — in test isolation they may not be
    on sys.path. Either they import successfully (and we get an
    Optimizer) or they raise NotImplementedError with a clear message.
    """
    params = [torch.randn(4, 4, requires_grad=True)]
    try:
        opt = build_optimizer(name, params, lr=1e-3)
        assert isinstance(opt, torch.optim.Optimizer)
    except NotImplementedError as exc:
        assert "is not importable" in str(exc)


def test_build_optimizer_unknown_name_raises():
    params = [torch.randn(3, requires_grad=True)]
    with pytest.raises(ValueError):
        build_optimizer("nonexistent", params, lr=1e-3)


def test_build_optimizer_rejects_empty_params():
    with pytest.raises(ValueError):
        build_optimizer("adam", [], lr=1e-3)


def test_build_optimizer_rejects_nonpositive_lr():
    params = [torch.randn(3, requires_grad=True)]
    with pytest.raises(ValueError):
        build_optimizer("adam", params, lr=0.0)


def test_known_optimizers_includes_all_ten():
    expected = {
        "adam",
        "adamw",
        "yogi",
        "muon",
        "lion",
        "sophia",
        "soap",
        "naive_yogi_muon",
        "muogi",
        "ramuogi",
    }
    assert set(KNOWN_OPTIMIZERS) == expected


# ---------------------------------------------------------------------------
# Naive Yogi-Muon — construction + the cheater's-choice antibaseline math
# ---------------------------------------------------------------------------


def test_naive_yogi_muon_constructor_validates():
    params = [torch.randn(4, 4, requires_grad=True)]
    with pytest.raises(ValueError):
        NaiveYogiMuon(params, lr=-1.0)
    with pytest.raises(ValueError):
        NaiveYogiMuon(params, lr=1e-3, betas=(1.5, 0.999))
    with pytest.raises(ValueError):
        NaiveYogiMuon(params, lr=1e-3, eps_yogi=0.0)
    with pytest.raises(ValueError):
        NaiveYogiMuon(params, lr=1e-3, ns5_iters=0)


def test_naive_ns5_orthogonalizes():
    """NS5 of a random matrix produces an output whose Gram matrix
    approaches the identity (or projection) — the orthogonalization
    property that's the entire point of Muon-family methods.
    """
    torch.manual_seed(0)
    X = torch.randn(8, 8, dtype=torch.float32)
    Y = _naive_ns5(X, iters=5)
    # After 5 iterations on an 8x8 well-conditioned input, Y Y^T should
    # be close to identity.
    YYT = Y @ Y.T
    I = torch.eye(8, dtype=torch.float32)
    # 5 NS5 iterations bring well-conditioned inputs to within ~1e-2
    # of the orthogonal projection (the Schulz polynomial's natural rate).
    assert (YYT - I).abs().max().item() < 5e-1


def test_naive_yogi_muon_state_dict_round_trip():
    """The anti-baseline must expose a working state_dict so the harness
    treats it identically to other optimizers."""
    torch.manual_seed(42)
    params = [torch.randn(4, 4, requires_grad=True)]
    opt = NaiveYogiMuon(params, lr=1e-3)
    # Run a few steps to populate state.
    for _ in range(3):
        opt.zero_grad()
        loss = (params[0] ** 2).sum()
        loss.backward()
        opt.step()

    sd = opt.state_dict()

    # Build a fresh optimizer with same params and load.
    torch.manual_seed(42)
    params2 = [p.detach().clone().requires_grad_(True) for p in params]
    opt2 = NaiveYogiMuon(params2, lr=1e-3)
    opt2.load_state_dict(sd)

    # The two should have identical state.
    sd2 = opt2.state_dict()
    assert set(sd["state"].keys()) == set(sd2["state"].keys())


def test_naive_yogi_muon_runs_one_step_without_nan():
    """Smoke: a single step on a 4x4 problem produces finite updates."""
    torch.manual_seed(1)
    W = torch.randn(4, 4, requires_grad=True)
    opt = NaiveYogiMuon([W], lr=1e-3)
    target = torch.randn(4, 4)
    loss = ((W - target) ** 2).sum()
    loss.backward()
    opt.step()
    assert torch.isfinite(W).all().item()


def test_naive_yogi_muon_anti_baseline_applies_ns5_after_yogi():
    """**The defining test of the anti-baseline.**

    Verify that NaiveYogiMuon's single step:
      1. Forms D = m_hat / (sqrt(v_hat) + eps) — Yogi's preconditioned
         direction
      2. Feeds D into NS5 (rather than feeding row-scaled momentum into
         NS5 as Muogi's "cheater's choice" does)

    Compare a single step of NaiveYogiMuon against a manual reference
    that **inlines** the NS5 polynomial (does not call ``_naive_ns5``),
    so the test catches sign-flip / coefficient-typo bugs in the
    optimizer's NS5 helper. Plus an independent post-condition check
    that the NS5 output approximates an orthogonal projection — this
    fails loudly if the polynomial coefficients are wrong even if the
    optimizer-vs-reference equality somehow holds.
    """
    torch.manual_seed(7)
    W = torch.randn(4, 4, requires_grad=True)
    W_init = W.detach().clone()

    # Run NaiveYogiMuon for one step.
    opt = NaiveYogiMuon(
        [W], lr=1e-3, betas=(0.9, 0.999), eps_yogi=1e-3, ns5_iters=5,
    )
    # Use a fixed gradient to make the math reproducible.
    g = torch.randn(4, 4)
    W.grad = g.clone()
    opt.step()
    W_after = W.detach().clone()

    # Manual reference computation:
    init_acc = 1e-6
    beta1, beta2 = 0.9, 0.999
    eps_yogi = 1e-3

    m = torch.full_like(W_init, init_acc)
    v = torch.full_like(W_init, init_acc)
    # Yogi moment updates (one step, t=1).
    m_new = beta1 * m + (1.0 - beta1) * g
    g_sq = g * g
    v_new = v + torch.sign(v - g_sq) * g_sq * -(1.0 - beta2)
    # Bias correction at t=1.
    m_hat = m_new / (1.0 - beta1 ** 1)
    v_hat = v_new / (1.0 - beta2 ** 1)
    # Yogi's preconditioned direction — this is the "naive composition".
    D = m_hat / (v_hat.sqrt() + eps_yogi)

    # Inline 5-iteration NS5 — does NOT call _naive_ns5 so this is an
    # independent reference. Standard Keller Jordan coefficients:
    #   a = 3.4445, b = -4.7750, c = 2.0315
    # Per-iter update: X ← a·X + b·A·X + c·A²·X where A = X X^T
    a_ref, b_ref, c_ref = 3.4445, -4.7750, 2.0315
    X_ref = D / D.norm().clamp_min(1e-12)
    # 4x4 is square — no transpose path needed.
    for _ in range(5):
        A_ref = X_ref @ X_ref.T
        B_ref = b_ref * A_ref + c_ref * (A_ref @ A_ref)
        X_ref = a_ref * X_ref + B_ref @ X_ref
    update = X_ref
    W_manual = W_init - 1e-3 * update

    # Equality with the inlined reference: catches Yogi-stage typos
    # AND NS5-coefficient typos because the reference is independent.
    assert torch.allclose(W_after, W_manual, atol=1e-5), (
        f"NaiveYogiMuon step did not match inlined-reference computation; "
        f"max diff: {(W_after - W_manual).abs().max().item()}"
    )

    # Post-condition: NS5's output should be near-orthogonal — i.e. its
    # singular values cluster around 1. If the polynomial sign is wrong
    # the output blows up or shrinks to zero, both of which violate
    # this check independent of the equality test above.
    Y_post = update
    sv = torch.linalg.svdvals(Y_post)
    # Loose tolerance: 5 NS5 iters on a well-conditioned 4x4 puts SVs
    # within ~0.1 of 1.0. Anything beyond catches the sign bug.
    assert torch.isfinite(Y_post).all(), (
        "NS5 output is non-finite — coefficient sign bug suspected"
    )
    assert (sv > 0.5).all() and (sv < 2.0).all(), (
        f"NS5 output is not near-orthogonal; singular values: "
        f"{sv.tolist()} — should cluster near 1.0"
    )


def test_naive_yogi_muon_1d_skips_ns5():
    """1-D parameters skip NS5 (Muon's NS5 only applies to matrices)."""
    torch.manual_seed(2)
    w = torch.randn(4, requires_grad=True)
    w_init = w.detach().clone()
    opt = NaiveYogiMuon([w], lr=1e-2)
    g = torch.ones_like(w)
    w.grad = g.clone()
    opt.step()
    # 1-D path uses D directly. The update should be -lr * D ≈ -lr * sign(g)
    # via Yogi's normalization. Just verify it moved and is finite.
    assert torch.isfinite(w).all().item()
    assert not torch.allclose(w.detach(), w_init)


# ---------------------------------------------------------------------------
# CSV schema — run_one returns exactly the documented columns.
# ---------------------------------------------------------------------------


def test_run_one_csv_schema():
    problem = _TinyQuadratic(seed=0)
    row = run_one(problem, "adam", lr=1e-2, seed=0)
    assert set(row.keys()) == set(CSV_COLUMNS)
    assert isinstance(row["problem"], str)
    assert isinstance(row["optimizer"], str)
    assert isinstance(row["lr"], float)
    assert isinstance(row["seed"], int)
    assert isinstance(row["steps"], int)
    assert isinstance(row["loss_trajectory"], str)
    # Muogi-specific telemetry columns exist and are finite floats.
    assert isinstance(row["ns5_success_rate"], float)
    assert isinstance(row["r_t_value"], float)
    assert isinstance(row["variance_l2_norm"], float)
    # Trajectory string parses back.
    parsed = plot_bench._parse_trajectory(row["loss_trajectory"])
    assert len(parsed) == row["steps"]


def test_run_one_naive_yogi_muon_populates_variance_norm():
    """Naive Yogi-Muon uses ``exp_avg_sq`` (Yogi state) — variance_l2_norm
    should be > 0 after a few steps."""
    problem = _TinyMatrix(seed=0)
    row = run_one(problem, "naive_yogi_muon", lr=1e-3, seed=0)
    # After running for max_steps the optimizer has accumulated state.
    assert row["variance_l2_norm"] > 0.0


# ---------------------------------------------------------------------------
# Reproducibility — same (problem, optimizer, lr, seed) yields identical
# loss trajectories across two runs.
# ---------------------------------------------------------------------------


def test_run_one_reproducibility():
    """Same (problem, optimizer, lr, seed) → identical finite trajectory.

    Strict equality on finite floats. If a run produces NaN that's a
    separate optimizer bug to catch in its own test (e.g.
    ``test_naive_yogi_muon_no_nan_on_trivial_problem`` below), not
    something to paper over here.
    """
    p1 = _TinyQuadratic(seed=123)
    row1 = run_one(p1, "adam", lr=1e-2, seed=123)

    p2 = _TinyQuadratic(seed=123)
    row2 = run_one(p2, "adam", lr=1e-2, seed=123)

    traj1 = plot_bench._parse_trajectory(row1["loss_trajectory"])
    traj2 = plot_bench._parse_trajectory(row2["loss_trajectory"])
    # All values must be finite — Adam on a tiny quadratic does not
    # produce NaN. If this fails with NaN, fix the optimizer.
    assert all(math.isfinite(x) for x in traj1), (
        f"Adam on _TinyQuadratic produced non-finite trajectory: {traj1}"
    )
    assert traj1 == traj2
    assert row1["final_loss"] == row2["final_loss"]
    assert row1["steps"] == row2["steps"]


def test_run_one_naive_yogi_muon_reproducibility():
    """The anti-baseline must be reproducible across runs.

    Strict equality on finite trajectories. NaN production is caught
    by ``test_naive_yogi_muon_no_nan_on_trivial_problem`` as a separate
    correctness assertion.
    """
    p1 = _TinyMatrix(seed=99)
    row1 = run_one(p1, "naive_yogi_muon", lr=1e-3, seed=99)

    p2 = _TinyMatrix(seed=99)
    row2 = run_one(p2, "naive_yogi_muon", lr=1e-3, seed=99)

    traj1 = plot_bench._parse_trajectory(row1["loss_trajectory"])
    traj2 = plot_bench._parse_trajectory(row2["loss_trajectory"])
    assert all(math.isfinite(x) for x in traj1), (
        f"NaiveYogiMuon on _TinyMatrix produced non-finite trajectory: "
        f"{traj1[:10]}..."
    )
    assert traj1 == traj2


def test_naive_yogi_muon_no_nan_on_trivial_problem():
    """NaiveYogiMuon must produce finite outputs on a well-conditioned
    matrix-regression problem.

    Catches NS5 coefficient-sign regressions loudly. If the NS5
    polynomial is mis-signed, the optimizer's update explodes and the
    trajectory fills with NaN/inf. A correctness property of the
    implementation, not a behavioral claim about the naive composition.
    """
    p = _TinyMatrix(seed=42)
    row = run_one(p, "naive_yogi_muon", lr=1e-3, seed=42)
    traj = plot_bench._parse_trajectory(row["loss_trajectory"])
    assert row["nan_count"] == 0, (
        f"NaiveYogiMuon produced {row['nan_count']} NaN losses on a "
        "well-conditioned 4x4 problem — NS5 coefficient bug suspected."
    )
    assert all(math.isfinite(x) for x in traj), (
        f"NaiveYogiMuon trajectory has non-finite values: {traj[:10]}..."
    )


# ---------------------------------------------------------------------------
# Plot stubs — NotImplementedError with the right pointer.
# ---------------------------------------------------------------------------


def test_plot_stubs_raise_not_implemented():
    df = pd.DataFrame(columns=list(CSV_COLUMNS))
    stubs = [
        lambda: plot_bench.plot_loss_curves(df, "q1", "out.png"),
        lambda: plot_bench.plot_wall_clock_pareto(df, "q1", "out.png"),
        lambda: plot_bench.plot_lr_sensitivity(df, "q1", "out.png"),
        lambda: plot_bench.plot_optimizer_vs_problem_heatmap(df, "out.png"),
        lambda: plot_bench.plot_safety_chain_activations(df, "out.png"),
        lambda: plot_bench.plot_variance_fidelity(df, "q1", "out.png"),
        lambda: plot_bench.plot_direction_magnitude_separation(
            df, "q1", "out.png"
        ),
        lambda: plot_bench.plot_ns5_convergence_scatter(df, "q1", "out.png"),
        lambda: plot_bench.plot_rt_trajectory(df, "q5", "out.png"),
    ]
    for stub in stubs:
        with pytest.raises(NotImplementedError) as excinfo:
            stub()
        assert "Phase 4" in str(excinfo.value)


# ---------------------------------------------------------------------------
# load_results — parses trajectory column back.
# ---------------------------------------------------------------------------


def test_load_results_parses_trajectory(tmp_path):
    csv_path = tmp_path / "tiny.csv"
    header = ",".join(CSV_COLUMNS)
    traj = "1.0;0.5;0.25"
    row = ",".join(
        [
            "test_tiny_quadratic",  # problem
            "adam",                  # optimizer
            "0.01",                  # lr
            "0",                     # seed
            "3",                     # steps
            "-1",                    # convergence_step
            "0.25",                  # final_loss
            "1.0",                   # wall_clock_per_step_us
            "0",                     # nan_count
            "0", "0", "0", "0", "0",  # l1..l5
            "0.0",                   # ns5_success_rate
            "0.0",                   # r_t_value
            "0.0",                   # variance_l2_norm
            traj,
        ]
    )
    csv_path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    df = plot_bench.load_results(str(csv_path))
    assert len(df) == 1
    assert df.loc[0, "loss_trajectory"] == [1.0, 0.5, 0.25]
