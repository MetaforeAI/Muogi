"""Render the paper figures from a bench_results.csv produced by run_bench.

Synthetic problem figures:
  fig_q1_burst_variance.png         — Q1, loss-vs-step
  fig_q2_polar_decomposition.png    — Q2, loss-vs-step
  fig_q3_tiny_mlp_mixed.png         — Q3, loss-vs-step
  fig_q4_ns5_stress.png             — Q4, loss-vs-step
  fig_q5_radam_cold_start.png       — Q5, loss-vs-step

Real-task problem figures (if present):
  fig_r1_cifar10.png                — R1, loss-vs-step
  fig_r2_charlm.png                 — R2, loss-vs-step
  fig_r3_nanogpt.png                — R3, loss-vs-step

Safety-layer bar chart:
  fig_safety_counters.png           — l1..l5 totals per optimizer

Usage:
    python bench/plot_bench.py --input bench/results.csv --output bench/figs/
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_OPTIMIZER_COLORS = {
    "adam":              "#1f77b4",   # blue
    "adamw":             "#17becf",   # teal
    "yogi":              "#9467bd",   # purple
    "lion":              "#ff7f0e",   # orange
    "liger":             "#d62728",   # red
    "muogi":             "#2ca02c",   # green (this paper)
    "ramuogi":           "#8c564b",   # brown (this paper)
    "racaso":            "#e377c2",   # pink
    "naive_yogi_muon":   "#bcbd22",   # olive
}


def _read_rows(path: Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _parse_trajectory(s: str) -> List[float]:
    """Parse a semicolon-separated trajectory; NaN-safe."""
    if not s:
        return []
    out: List[float] = []
    for tok in s.split(";"):
        try:
            v = float(tok)
        except ValueError:
            v = float("nan")
        out.append(v)
    return out


def _filter(rows: List[dict], **kw) -> List[dict]:
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == str(v) or r.get(k) == v]
    return out


def _collect_best_per_opt(rows: List[dict], problem: str):
    """Return (avg_by_opt, final_by_opt, lr_by_opt) for the best LR per
    optimizer on a problem, averaging trajectories over seeds.

    Trajectories are NOT padded to max length — each line ends at the
    actual length its underlying runs reached. This matters because the
    harness stops a run at problem.converged_tol; padding short runs
    with their final value would falsely flat-line the curve.
    """
    sub = _filter(rows, problem=problem)
    if not sub:
        return {}, {}, {}
    avg_by_opt: Dict[str, List[float]] = {}
    final_by_opt: Dict[str, float] = {}
    lr_by_opt: Dict[str, str] = {}
    opts = sorted({r["optimizer"] for r in sub})
    for opt in opts:
        cand = [r for r in sub if r["optimizer"] == opt]
        by_lr: Dict[str, List[dict]] = defaultdict(list)
        for r in cand:
            by_lr[r["lr"]].append(r)
        def _score(lst: List[dict]) -> float:
            vals = []
            for r in lst:
                try:
                    v = float(r["final_loss"])
                    if math.isfinite(v):
                        vals.append(v)
                except (TypeError, ValueError):
                    continue
            return sum(vals) / len(vals) if vals else float("inf")
        if not by_lr:
            continue
        best_lr = min(by_lr, key=lambda k: _score(by_lr[k]))
        trajs = [_parse_trajectory(r["loss_trajectory"]) for r in by_lr[best_lr]]
        trajs = [t for t in trajs if t]
        if not trajs:
            continue
        # Average across seeds, truncating to the shortest seed's length —
        # this preserves the "ends where it converged" property.
        min_len = min(len(t) for t in trajs)
        if min_len == 0:
            continue
        truncated = [t[:min_len] for t in trajs]
        avg = [sum(c) / len(c) for c in zip(*truncated)]
        avg_by_opt[opt] = avg
        final_by_opt[opt] = avg[-1]
        lr_by_opt[opt] = best_lr
    return avg_by_opt, final_by_opt, lr_by_opt


def _filter_diverged(avg_by_opt, final_by_opt, lr_by_opt):
    """Exclude optimizers whose final loss > 3× median of converged.
    Returns (converged_avg, converged_final, lr_by_opt, diverged_dict)."""
    if not final_by_opt:
        return {}, {}, lr_by_opt, {}
    vals = sorted(final_by_opt.values())
    med = vals[len(vals) // 2]
    thresh = max(3.0 * med, med + 1.0)
    diverged = {o: v for o, v in final_by_opt.items() if v > thresh}
    converged_avg = {o: a for o, a in avg_by_opt.items() if o not in diverged}
    converged_final = {o: v for o, v in final_by_opt.items() if o not in diverged}
    return converged_avg, converged_final, lr_by_opt, diverged


def _loss_curves(rows: List[dict], problem: str, title: str, out: Path) -> None:
    """Single-panel loss-vs-step overlay for synthetic problems.

    Raw per-step traces (no smoothing) at thin alpha. Each line ends at
    its actual convergence step — line-end position is signal.
    """
    avg_by_opt, final_by_opt, lr_by_opt = _collect_best_per_opt(rows, problem)
    if not avg_by_opt:
        print(f"  no {problem} data; skipping {out}")
        return
    converged_avg, _, lr_by_opt, diverged = _filter_diverged(
        avg_by_opt, final_by_opt, lr_by_opt
    )
    if not converged_avg:
        print(f"  no converged runs for {problem}; skipping {out}")
        return
    diverged_note = ""
    if diverged:
        bits = [f"{o} ({v:.2g})" for o, v in sorted(diverged.items(), key=lambda kv: -kv[1])]
        diverged_note = f"  [diverged: {', '.join(bits)}]"

    fig, ax = plt.subplots(figsize=(10, 5))
    for opt, avg in converged_avg.items():
        color = _OPTIMIZER_COLORS.get(opt, "#000")
        ax.plot(
            range(1, len(avg) + 1), avg,
            color=color, linewidth=0.7, alpha=0.85,
            label=f"{opt} (lr={lr_by_opt[opt]})",
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.set_title(title + diverged_note)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _real_task_loss_curves(rows: List[dict], problem: str, title: str, out: Path) -> None:
    """Two-panel figure for a real-task problem (R1/R2/R3): raw curves
    on the left, final-loss bar chart (sorted) on the right."""
    avg_by_opt, final_by_opt, lr_by_opt = _collect_best_per_opt(rows, problem)
    if not avg_by_opt:
        print(f"  no {problem} data; skipping {out}")
        return
    converged_avg, converged_final, lr_by_opt, diverged = _filter_diverged(
        avg_by_opt, final_by_opt, lr_by_opt
    )
    if not converged_avg:
        print(f"  no converged runs for {problem}; skipping {out}")
        return
    diverged_note = ""
    if diverged:
        bits = [f"{o} ({v:.2g})" for o, v in sorted(diverged.items(), key=lambda kv: -kv[1])]
        diverged_note = f"  [diverged: {', '.join(bits)}]"

    fig, (ax_curve, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    for opt, avg in converged_avg.items():
        color = _OPTIMIZER_COLORS.get(opt, "#000")
        ax_curve.plot(
            range(1, len(avg) + 1), avg,
            color=color, linewidth=0.7, alpha=0.85,
            label=f"{opt} (lr={lr_by_opt[opt]})",
        )
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("loss (log)")
    ax_curve.set_title(title + diverged_note)
    ax_curve.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax_curve.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ordered = sorted(converged_final.items(), key=lambda kv: kv[1])
    opt_names = [o for o, _ in ordered]
    finals = [v for _, v in ordered]
    colors = [_OPTIMIZER_COLORS.get(o, "#000") for o in opt_names]
    ypos = list(range(len(opt_names)))
    ax_bar.barh(ypos, finals, color=colors, height=0.7)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(opt_names, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("final loss")
    ax_bar.set_title("final loss (lower = better)")
    ax_bar.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    for i, v in enumerate(finals):
        ax_bar.text(v, i, f" {v:.3g}",
                    va="center", ha="left", fontsize=8, color="#222")
    xmin = min(finals) * 0.9
    xmax = max(finals) * 1.15
    ax_bar.set_xlim(xmin, xmax)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _safety_counters(rows: List[dict], out: Path) -> None:
    """Bar chart of l1..l5 safety-counter totals per optimizer, summed
    across all problems and runs."""
    counters: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {f"l{i}": 0 for i in range(1, 6)}
    )
    for r in rows:
        opt = r["optimizer"]
        for i in range(1, 6):
            try:
                counters[opt][f"l{i}"] += int(r.get(f"l{i}_count", 0) or 0)
            except (TypeError, ValueError):
                continue
    opts = sorted(counters)
    if not opts:
        print(f"  no safety-counter data; skipping {out}")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    bottoms = [0] * len(opts)
    layer_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, layer in enumerate(("l1", "l2", "l3", "l4", "l5")):
        heights = [counters[o][layer] for o in opts]
        ax.bar(opts, heights, bottom=bottoms,
               color=layer_colors[i], label=layer.upper())
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_ylabel("total safety-counter firings")
    ax.set_title("Safety chain (L1-L5) firing counts per optimizer (all problems)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.legend(title="layer", loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="results.csv")
    ap.add_argument("--output", type=Path, default=Path("bench/figs"))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.input)
    print(f"loaded {len(rows)} rows from {args.input}")

    _loss_curves(rows, "q1_burst_variance",
                 "Q1 — Bursty variance preservation (M1/M2)",
                 args.output / "fig_q1_burst_variance.png")
    _loss_curves(rows, "q2_polar_decomposition",
                 "Q2 — Polar decomposition fidelity (NS5 core)",
                 args.output / "fig_q2_polar_decomposition.png")
    _loss_curves(rows, "q3_tiny_mlp_mixed",
                 "Q3 — Tiny MLP, mixed gradient distributions (M7)",
                 args.output / "fig_q3_tiny_mlp_mixed.png")
    _loss_curves(rows, "q4_ns5_stress",
                 "Q4 — NS5 convergence-failure stress (M3/M6)",
                 args.output / "fig_q4_ns5_stress.png")
    _loss_curves(rows, "q5_radam_cold_start",
                 "Q5 — RAdam r_t cold-start (M5)",
                 args.output / "fig_q5_radam_cold_start.png")

    _loss_curves(rows, "r1_cifar10_resnet18",
                 "R1 — CIFAR-10 ResNet-18: training loss",
                 args.output / "fig_r1_cifar10.png")
    _loss_curves(rows, "r2_charlm_shakespeare",
                 "R2 — Char-LM on tiny-shakespeare: training loss",
                 args.output / "fig_r2_charlm.png")
    _loss_curves(rows, "r3_nanogpt_wikitext2",
                 "R3 — NanoGPT on WikiText-2: training loss",
                 args.output / "fig_r3_nanogpt.png")

    _safety_counters(rows, args.output / "fig_safety_counters.png")


if __name__ == "__main__":
    main()
