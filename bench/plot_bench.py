"""Plotting harness — reads ``bench_results.csv`` and emits the figures
documented in ``plans/sharded-plotting-haven.md`` (Muogi benchmark
section).

Phase 1: ``load_results`` is fully implemented (parses the CSV, splits
the ``loss_trajectory`` column back into a list of floats). All plotting
functions are stubbed with ``NotImplementedError`` plus a docstring
describing what the plot should show; Phase 4 fills them in.

Plotting backend choice (matplotlib vs plotly) is deferred to Phase 4.
"""

from __future__ import annotations

import math
from typing import List

import pandas as pd


def _parse_trajectory(s: object) -> List[float]:
    """Parse the ``loss_trajectory`` column back to a Python list of floats.

    Empty / NaN cells return an empty list. Non-finite literals like
    ``nan`` and ``inf`` are preserved via ``float()``.
    """
    if s is None:
        return []
    if isinstance(s, float) and math.isnan(s):
        return []
    text = str(s).strip()
    if not text:
        return []
    out: List[float] = []
    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def load_results(csv_path: str) -> pd.DataFrame:
    """Load ``bench_results.csv`` into a DataFrame.

    Parses ``loss_trajectory`` from its semicolon-encoded string form
    back into a ``list[float]`` column. All other columns are loaded
    with their natural pandas dtypes.

    Args:
        csv_path: path to the CSV produced by ``run_bench.py``.

    Returns:
        A DataFrame with one row per (problem, optimizer, lr, seed) run.
    """
    df = pd.read_csv(csv_path)
    if "loss_trajectory" in df.columns:
        df["loss_trajectory"] = df["loss_trajectory"].map(_parse_trajectory)
    return df


def plot_loss_curves(df: pd.DataFrame, problem: str, out_path: str) -> None:
    """Loss-vs-step plot: 10 curves (one per optimizer), log-y, median +
    IQR shading across seeds, all on the same axes. One figure per
    problem; filtered by ``problem``.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_wall_clock_pareto(df: pd.DataFrame, problem: str, out_path: str) -> None:
    """Pareto curve: wall-clock-to-loss-X (x-axis) vs final-loss (y-axis)
    across optimizers. Highlights which optimizer is fastest to reach
    each target loss; one figure per problem.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_lr_sensitivity(df: pd.DataFrame, problem: str, out_path: str) -> None:
    """LR-sensitivity plot: small multiples (one panel per optimizer)
    showing best-final-loss as a function of LR. Sweeps across the
    documented LR set; one figure per problem.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_optimizer_vs_problem_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """Heatmap: optimizer (rows) × problem (cols) → relative final loss
    normalized so Adam=1.0 on each problem. Muogi should appear as the
    consistent leader on Q1/Q3/Q7; comparable to Muon on Q2 (polar
    decomposition, no row-burst pathology). Naive Yogi-Muon should be
    a visible loser on Q1 vs Yogi-alone and vs Muogi.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_safety_chain_activations(df: pd.DataFrame, out_path: str) -> None:
    """Stacked bar chart per problem showing what fraction of steps
    triggered each Muogi/RAMuogi safety-chain layer (L1 spread-cap clamp,
    L2 NS5 convergence safe-skip, L3 vanilla Yogi fallback, L4 RAdam
    variance gate). Validates the four-layer safety claim in the Muogi
    paper.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_variance_fidelity(df: pd.DataFrame, problem: str, out_path: str) -> None:
    """**Q1 HEADLINE PLOT — the cheater's-choice validation.**

    Renders Yogi-alone (gold standard) vs Muogi vs Naive_Yogi_Muon's
    ``||v_hat||`` trajectory across the Q1 bursty-gradient problem.

    The cheater's-choice claim is validated iff:
        - Muogi tracks Yogi within tolerance T (variance signal preserved
          when row-scale is injected at NS5 iter-0)
        - Naive Yogi-Muon diverges from Yogi by margin M (variance signal
          erased when NS5 sees the already-normalized direction and
          averages it to the spectral mean)

    This is the single most important figure in the Muogi paper's
    empirical section. M1 stands or falls on this plot.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_direction_magnitude_separation(
    df: pd.DataFrame, problem: str, out_path: str
) -> None:
    """For Q1's bursty-gradient regime: plot the NS5 output magnitude
    (should be ~1, orthogonalized) against the per-row scaling (should
    track Yogi variance for Muogi). Demonstrates Muogi separates
    direction and scale; naive composition collapses both onto the
    spectral mean.
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_ns5_convergence_scatter(
    df: pd.DataFrame, problem: str, out_path: str
) -> None:
    """Scatter: NS5 iteration count to convergence vs input condition
    number, across all Q1-Q5 runs that exercised NS5. Empirically maps
    the convergence radius and visualizes where the L2 safe-skip
    activates (the cluster of points above the convergence-radius
    threshold).
    """
    raise NotImplementedError("Phase 4 implements plotting")


def plot_rt_trajectory(df: pd.DataFrame, problem: str, out_path: str) -> None:
    """RAMuogi's RAdam variance-rectification gate trajectory.

    Plot ``r_t`` per step for RAMuogi runs on Q5 (cold-start regime).
    Should show the gate at 0 (NS5 disabled) for the first ~15 steps,
    then a smooth ramp to 1.0 (NS5 fully engaged) at t≈20. Visualizes
    the M5 claim that RAMuogi delays spectral orthogonalization until
    statistical variance tracking has stabilized.
    """
    raise NotImplementedError("Phase 4 implements plotting")
