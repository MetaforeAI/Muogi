"""Q5 — RAdam ``r_t`` cold-start regime.

Validates Muogi paper claim **M5**: RAMuogi's variance-rectification
gate ``r_t`` correctly gates updates during the cold-start window
where the variance estimate ``v_t`` has not yet accumulated enough
mass to be trustworthy.

Problem
-------
Short-horizon training run: ``max_steps = 100`` deliberately, so the
optimizer never leaves the cold-start regime. Simple matrix
regression:

    f(W) = 1/2 * || W - M ||_F^2

with ``W`` of shape ``(4, 4)`` and ``M`` a fixed small Gaussian
matrix per seed.

The point isn't to converge. The point is to observe ``r_t`` behaviour
on RAMuogi during the first 100 steps — the harness captures
``r_t_value`` per run automatically via ``_read_muogi_telemetry`` in
``run_bench.py``. Phase 4's plotting code will visualize the gate
trajectory across these short runs.

Class attributes
----------------
- ``name = "q5_radam_cold_start"``
- ``max_steps = 100``
- ``converged_tol = 1e-2``
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


class Q5RAdamColdStart(BenchProblem):
    """RAdam r_t cold-start observation — validates M5."""

    name = "q5_radam_cold_start"
    max_steps = 100
    converged_tol = 1e-2

    _DIM = 4

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        # Small target so a 100-step horizon is not absurd, but also
        # not trivially solved by a single Adam step.
        self._target = (0.5 * torch.randn(
            (self._DIM, self._DIM), generator=self._generator
        )).to(self.device)

    def init_params(self) -> List[torch.Tensor]:
        w0 = torch.randn(
            (self._DIM, self._DIM), generator=self._generator
        ).to(self.device)
        w0.requires_grad_(True)
        return [w0]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (w,) = params
        diff = w - self._target
        return 0.5 * (diff * diff).sum()
