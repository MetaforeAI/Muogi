"""Optimizer wrappers for the Muogi benchmark suite.

A single canonical entry point — ``bench.optimizers.wrappers.build_optimizer``
— constructs every baseline. See ``README.md`` in this directory for the
canonical config table and the list of baselines that still need vendoring
before Phase 3.
"""

from bench.optimizers.wrappers import build_optimizer

__all__ = ["build_optimizer"]
