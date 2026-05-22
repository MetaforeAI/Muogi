"""Benchmark harness — run a single (problem, optimizer, lr, seed) config
or sweep all combinations and emit a CSV.

Phase 1 status:
    - ``run_one`` is implemented end-to-end against the ``BenchProblem``
      contract; once problems land in Phase 2, single-config runs will
      execute.
    - ``--sweep`` iterates over ``BenchProblem.__subclasses__()``. In
      Phase 1 this set is empty, so ``--sweep`` is a documented no-op
      that emits a header-only CSV and exits cleanly.

CSV schema (one row per (problem, optimizer, lr, seed) run):

    problem, optimizer, lr, seed, steps, convergence_step, final_loss,
    wall_clock_per_step_us, nan_count,
    l1_count, l2_count, l3_count, l4_count, l5_count,
    ns5_success_rate, r_t_value, variance_l2_norm,
    loss_trajectory

The three Muogi-specific telemetry columns:
    ns5_success_rate  — fraction of NS5 attempts that converged
    r_t_value         — final RAdam variance-rectification gate value
                        (RAMuogi only; 0 otherwise)
    variance_l2_norm  — end-of-run sum-of-norms of optimizer's
                        ``exp_avg_sq`` state (Yogi-family variance health)

``loss_trajectory`` is the full per-step loss history serialized as a
semicolon-separated list of floats.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from typing import Dict, List, Optional

import torch

from bench.optimizers.wrappers import KNOWN_OPTIMIZERS, build_optimizer
from bench.problems.base import BenchProblem


CSV_COLUMNS: tuple[str, ...] = (
    "problem",
    "optimizer",
    "lr",
    "seed",
    "steps",
    "convergence_step",
    "final_loss",
    "wall_clock_per_step_us",
    "nan_count",
    "l1_count",
    "l2_count",
    "l3_count",
    "l4_count",
    "l5_count",
    "ns5_success_rate",
    "r_t_value",
    "variance_l2_norm",
    "loss_trajectory",
)

LR_SWEEP: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
SEED_SWEEP: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)


def _registered_problems() -> Dict[str, type[BenchProblem]]:
    """Discover registered ``BenchProblem`` subclasses by ``.name``.

    Walks ``BenchProblem.__subclasses__()`` recursively so multi-level
    subclasses are visible too. Returns a name→class map. Subclasses
    with empty ``.name`` are skipped — they are treated as intermediate
    bases.
    """
    discovered: Dict[str, type[BenchProblem]] = {}
    stack: List[type[BenchProblem]] = list(BenchProblem.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls.name:
            discovered[cls.name] = cls
        stack.extend(cls.__subclasses__())
    return discovered


def _read_safety_counters(optimizer: torch.optim.Optimizer) -> Dict[str, int]:
    """Pull Muogi/RAMuogi safety-chain counters off the optimizer if exposed.

    Returns a dict with l1..l5 keys, defaulting to 0. Muogi/RAMuogi
    expose counters at ``optimizer.safety_counts`` (a dict keyed
    ``l1``..``l5``). Non-Muogi optimizers return zeros.
    """
    counters: Dict[str, int] = {f"l{i}_count": 0 for i in range(1, 6)}
    raw = getattr(optimizer, "safety_counts", None)
    if isinstance(raw, dict):
        for k in ("l1", "l2", "l3", "l4", "l5"):
            v = raw.get(k, 0)
            try:
                counters[f"{k}_count"] = int(v)
            except (TypeError, ValueError):
                counters[f"{k}_count"] = 0
    return counters


def _read_muogi_telemetry(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """Aggregate Muogi/RAMuogi/Naive-Yogi-Muon telemetry from optimizer state.

    Looks for per-param state entries:
        - ``ns5_success_count`` + ``ns5_skip_count`` → ns5_success_rate
        - ``r_t`` → median across params (RAMuogi only)
        - ``exp_avg_sq`` → sum of L2 norms (variance health)

    Optimizers that don't expose these keys get null/zero defaults.
    """
    ns5_success = 0
    ns5_attempts = 0
    r_t_vals: List[float] = []
    v_norm_sum = 0.0

    for p_state in optimizer.state.values():
        if not isinstance(p_state, dict):
            continue
        succ = p_state.get("ns5_success_count")
        skip = p_state.get("ns5_skip_count")
        if isinstance(succ, (int, float)) and isinstance(skip, (int, float)):
            ns5_success += int(succ)
            ns5_attempts += int(succ) + int(skip)
        r_t = p_state.get("r_t")
        if isinstance(r_t, (int, float)):
            r_t_vals.append(float(r_t))
        elif isinstance(r_t, torch.Tensor) and r_t.numel() == 1:
            r_t_vals.append(float(r_t.item()))
        v_hat = p_state.get("exp_avg_sq")
        if isinstance(v_hat, torch.Tensor):
            v_norm_sum += float(v_hat.detach().norm().item())

    ns5_success_rate = (
        ns5_success / ns5_attempts if ns5_attempts > 0 else 0.0
    )
    if r_t_vals:
        r_t_vals.sort()
        r_t_value = r_t_vals[len(r_t_vals) // 2]  # median
    else:
        r_t_value = 0.0

    return {
        "ns5_success_rate": float(ns5_success_rate),
        "r_t_value": float(r_t_value),
        "variance_l2_norm": float(v_norm_sum),
    }


def run_one(
    problem: BenchProblem,
    optimizer_name: str,
    lr: float,
    seed: int,
) -> Dict[str, object]:
    """Run one (problem, optimizer, lr, seed) configuration.

    Args:
        problem: an instantiated ``BenchProblem``.
        optimizer_name: one of ``bench.optimizers.wrappers.KNOWN_OPTIMIZERS``.
        lr: learning rate.
        seed: integer seed (also baked into ``problem`` at construction).

    Returns:
        A dict whose keys are exactly ``CSV_COLUMNS``.
    """
    if not isinstance(problem, BenchProblem):
        raise TypeError(
            f"problem must be a BenchProblem; got {type(problem).__name__}"
        )

    torch.manual_seed(seed)

    params = problem.init_params()
    if not isinstance(params, list) or not params:
        raise ValueError(
            f"{type(problem).__name__}.init_params() must return a "
            "non-empty list of tensors"
        )
    for i, p in enumerate(params):
        if not isinstance(p, torch.Tensor):
            raise TypeError(f"params[{i}] is not a Tensor")
        if not p.requires_grad:
            raise ValueError(f"params[{i}] must have requires_grad=True")
        if not p.is_leaf:
            raise ValueError(f"params[{i}] must be a leaf tensor")

    optimizer = build_optimizer(optimizer_name, params, lr=lr)

    trajectory: List[float] = []
    nan_count = 0
    convergence_step: int = -1
    final_loss: float = float("nan")
    total_wall_clock_s: float = 0.0
    measured_steps = 0

    for step in range(problem.max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss_val, grads = problem.loss_and_grad(params)
        for p, g in zip(params, grads):
            p.grad = g.detach() if isinstance(g, torch.Tensor) else None

        if not math.isfinite(loss_val):
            nan_count += 1
            trajectory.append(float("nan"))
            final_loss = float("nan")
            break

        trajectory.append(loss_val)
        final_loss = loss_val

        if convergence_step < 0 and problem.converged(loss_val, step):
            convergence_step = step

        t0 = time.perf_counter()
        optimizer.step()
        t1 = time.perf_counter()
        total_wall_clock_s += t1 - t0
        measured_steps += 1

    steps_completed = len(trajectory)
    if measured_steps > 0:
        wall_clock_per_step_us = (total_wall_clock_s / measured_steps) * 1e6
    else:
        wall_clock_per_step_us = float("nan")

    counters = _read_safety_counters(optimizer)
    muogi_tel = _read_muogi_telemetry(optimizer)

    return {
        "problem": problem.name,
        "optimizer": optimizer_name,
        "lr": lr,
        "seed": seed,
        "steps": steps_completed,
        "convergence_step": convergence_step,
        "final_loss": final_loss,
        "wall_clock_per_step_us": wall_clock_per_step_us,
        "nan_count": nan_count,
        "l1_count": counters["l1_count"],
        "l2_count": counters["l2_count"],
        "l3_count": counters["l3_count"],
        "l4_count": counters["l4_count"],
        "l5_count": counters["l5_count"],
        "ns5_success_rate": muogi_tel["ns5_success_rate"],
        "r_t_value": muogi_tel["r_t_value"],
        "variance_l2_norm": muogi_tel["variance_l2_norm"],
        "loss_trajectory": ";".join(repr(x) for x in trajectory),
    }


def _write_rows(rows: List[Dict[str, object]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_problem(problem_name: str, seed: int) -> BenchProblem:
    registry = _registered_problems()
    if not registry:
        raise RuntimeError(
            "no BenchProblem subclasses are registered; "
            "Phase 2 ships the problem modules — until then, single-"
            "config runs cannot proceed."
        )
    if problem_name not in registry:
        raise ValueError(
            f"unknown problem '{problem_name}'; "
            f"registered: {sorted(registry)}"
        )
    return registry[problem_name](seed=seed)


def _run_sweep(out_path: str) -> int:
    registry = _registered_problems()
    rows: List[Dict[str, object]] = []
    if not registry:
        _write_rows(rows, out_path)
        print(
            "[bench] no problems registered — Phase 1 sweep is a no-op; "
            f"wrote header-only CSV to {out_path}"
        )
        return 0

    for problem_name, cls in sorted(registry.items()):
        for opt_name in KNOWN_OPTIMIZERS:
            for lr in LR_SWEEP:
                for seed in SEED_SWEEP:
                    try:
                        problem = cls(seed=seed)
                        row = run_one(problem, opt_name, lr=lr, seed=seed)
                    except NotImplementedError as exc:
                        print(
                            f"[bench] skipping {problem_name} × "
                            f"{opt_name} × lr={lr} × seed={seed}: {exc}"
                        )
                        continue
                    rows.append(row)
    _write_rows(rows, out_path)
    print(f"[bench] wrote {len(rows)} rows to {out_path}")
    return 0


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench.run_bench",
        description="Run a single benchmark config or the full sweep.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="problem short name (e.g. q1_burst_variance)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        choices=sorted(KNOWN_OPTIMIZERS),
        help="optimizer short name",
    )
    parser.add_argument("--lr", type=float, default=None, help="learning rate")
    parser.add_argument("--seed", type=int, default=None, help="integer seed")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="run all (problem, optimizer, lr, seed) combinations",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="bench_results.csv",
        help="output CSV path (sweep mode); single-config mode prints to stdout",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.sweep:
        return _run_sweep(args.out)

    if args.problem is None or args.optimizer is None or args.lr is None \
            or args.seed is None:
        print(
            "error: single-config mode requires --problem, --optimizer, "
            "--lr, --seed (or use --sweep).",
            file=sys.stderr,
        )
        return 2

    problem = _build_problem(args.problem, args.seed)
    row = run_one(problem, args.optimizer, lr=args.lr, seed=args.seed)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(CSV_COLUMNS))
    writer.writeheader()
    writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
