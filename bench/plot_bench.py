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
    "adam":              "#888888",
    "adamw":             "#444444",
    "yogi":              "#1f77b4",
    "lion":              "#ff7f0e",
    "liger":             "#d62728",
    "muogi":             "#2ca02c",
    "ramuogi":           "#17becf",
    "racaso":            "#9467bd",
    "naive_yogi_muon":   "#bcbd22",
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


def _loss_curves(rows: List[dict], problem: str, title: str, out: Path) -> None:
    """Generic loss-vs-step overlay: one line per optimizer, best LR per
    optimizer, averaged over seeds."""
    sub = _filter(rows, problem=problem)
    if not sub:
        print(f"  no {problem} data; skipping {out}")
        return
    opts = sorted({r["optimizer"] for r in sub})
    fig, ax = plt.subplots(figsize=(10, 5))
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
        max_len = max(len(t) for t in trajs)
        padded = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        avg = [sum(c) / len(c) for c in zip(*padded)]
        ax.plot(
            range(1, len(avg) + 1), avg,
            color=_OPTIMIZER_COLORS.get(opt, "#000"),
            label=f"{opt} (lr={best_lr})",
            linewidth=1.5,
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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
