"""Phase-2 smoke harness: confirm each problem runs 50 Adam steps cleanly.

Not part of the regular pytest suite (filename starts with underscore).
Invoked once during Phase-2 landing to verify the five new problem
modules instantiate, the first loss is finite, and ``run_one`` returns
a well-formed dict. Adam at lr=1e-3.
"""

from __future__ import annotations

import math
import os
import sys

# Make ``bench`` importable when this script is invoked directly.
_MUOGI_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _MUOGI_ROOT not in sys.path:
    sys.path.insert(0, _MUOGI_ROOT)

from bench.problems import (  # noqa: F401 — import triggers registry
    q1_burst_variance,
    q2_polar_decomposition,
    q3_tiny_mlp_mixed,
    q4_ns5_stress,
    q5_radam_cold_start,
)
from bench.run_bench import CSV_COLUMNS, _registered_problems, run_one


def main() -> None:
    registry = _registered_problems()
    print(f"registered: {sorted(registry)}")
    for name in sorted(registry):
        cls = registry[name]
        orig_max = cls.max_steps
        cls.max_steps = 50
        try:
            prob = cls(seed=0)
            row = run_one(prob, "adam", lr=1e-3, seed=0)
            traj = [float(x) for x in row["loss_trajectory"].split(";") if x]
            first = traj[0] if traj else float("nan")
            last = traj[-1] if traj else float("nan")
            schema_ok = set(row.keys()) == set(CSV_COLUMNS)
            print(
                f"{name}: steps={row['steps']:>2} first={first:.6f} "
                f"final={last:.6f} nan_count={row['nan_count']} "
                f"finite_first={math.isfinite(first)} "
                f"finite_last={math.isfinite(last)} "
                f"schema_ok={schema_ok}"
            )
        finally:
            cls.max_steps = orig_max


if __name__ == "__main__":
    main()
