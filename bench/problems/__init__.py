"""Benchmark problems live here.

Phase 2: the five problem modules Q1-Q5 land here. Importing this
package imports every problem module so that ``BenchProblem``'s
subclass registry is populated and ``run_bench.py`` can discover
them via ``_registered_problems()``.

Problem set (validates Muogi paper claims M1-M7):
    q1_burst_variance      — M1 + M2  (cheater's-choice variance preservation)
    q2_polar_decomposition — core NS5 property
    q3_tiny_mlp_mixed      — M7       (heterogeneous topology)
    q4_ns5_stress          — M3 + M6  (NS5 safe-skip + Yogi fallback)
    q5_radam_cold_start    — M5       (variance-rectification gate)
"""

from bench.problems import (
    q1_burst_variance,
    q2_polar_decomposition,
    q3_tiny_mlp_mixed,
    q4_ns5_stress,
    q5_radam_cold_start,
    r1_cifar10_resnet18,
    r2_charlm_shakespeare,
    r3_nanogpt_wikitext2,
)

__all__ = [
    "q1_burst_variance",
    "q2_polar_decomposition",
    "q3_tiny_mlp_mixed",
    "q4_ns5_stress",
    "q5_radam_cold_start",
    "r1_cifar10_resnet18",
    "r2_charlm_shakespeare",
    "r3_nanogpt_wikitext2",
]
