"""Q1 — Bursty variance preservation.

Validates Muogi paper claims **M1** (cheater's-choice preserves Yogi's
burst-aware variance signal through the NS5 orthogonal rotation) and
**M2** (the Naive Yogi-Muon composition destroys this signal). This is
the headline problem for the M1/M2 figure: it produces the gradient
distribution under which Yogi's per-element variance tracker is most
distinguishable from Muon's spectral averaging.

Problem
-------
Quadratic-distance loss against a fixed target matrix ``M``:

    f(W) = 1/2 * || W - M ||_F^2

with ``W`` of shape ``(8, 8)``. The natural gradient is ``W - M``,
which has a calm, near-Gaussian character. The interesting behaviour
emerges from the **bursty gradient injection**.

Bursty gradient pattern
-----------------------
``loss_and_grad`` is overridden to inject a burst pattern with period
11: 10 calm steps where the gradient is the natural ``W - M`` plus
unit-Gaussian noise, then 1 "burst" step where ~20% of elements
(chosen via a per-init seeded mask) are multiplied by 100x. The
mask is *element-wise*, not row-wise: the variance tracker should
see a small population of elements with explosive second moment
amongst a sea of calm ones. This is the regime where Yogi's
truncating sign trick is supposed to shine and where naive composition
with NS5 is supposed to wash the signal out.

The cycle position is determined by an instance counter ``self._step``
that increments on every call to ``loss_and_grad``. ``step`` isn't
passed explicitly because the base contract doesn't include it, so
the per-instance counter is the canonical mechanism.

Class attributes
----------------
- ``name = "q1_burst_variance"``
- ``max_steps = 2000``
- ``converged_tol = 1e-2`` — bursty pattern makes lower tolerance hard.
- ``burst_aware = True`` — Phase 4's variance-fidelity plot uses this
  attribute to identify the problem for the M1 overlay figure.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from bench.problems.base import BenchProblem


class Q1BurstVariance(BenchProblem):
    """Bursty quadratic problem — validates M1 + M2."""

    name = "q1_burst_variance"
    max_steps = 2000
    converged_tol = 1e-2
    burst_aware = True

    # Bursty pattern: 10 calm steps then 1 burst step (period 11).
    _CALM_STEPS = 10
    _BURST_PERIOD = 11
    _BURST_MULT = 100.0
    _BURST_FRACTION = 0.20

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        # Fixed target M and a fixed element-wise burst mask, both
        # baked in at construction time so they are stable across
        # the run. Element-wise mask: ~20% of elements get the 100x
        # boost on burst steps.
        self._target = torch.randn(
            (8, 8), generator=self._generator
        )
        mask_uniform = torch.rand((8, 8), generator=self._generator)
        self._burst_mask = (mask_uniform < self._BURST_FRACTION).to(
            torch.float32
        )
        # Instance step counter — increments on every loss_and_grad call.
        self._step = 0

    def init_params(self) -> List[torch.Tensor]:
        # Start away from M so there is real work to do.
        w0 = torch.randn((8, 8), generator=self._generator)
        w0.requires_grad_(True)
        return [w0]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        (w,) = params
        diff = w - self._target
        return 0.5 * (diff * diff).sum()

    def loss_and_grad(
        self, params: List[torch.Tensor]
    ) -> Tuple[float, List[torch.Tensor]]:
        (w,) = params
        with torch.no_grad():
            diff = w.detach() - self._target
            loss_val = 0.5 * float((diff * diff).sum().item())

            # Calm step: natural grad + unit-Gaussian noise.
            noise = torch.randn((8, 8), generator=self._generator)
            grad = diff + noise

            # Every BURST_PERIOD-th step is a burst: multiply masked
            # elements by BURST_MULT. The mask is fixed per instance
            # so the burst pattern is reproducible.
            cycle_pos = self._step % self._BURST_PERIOD
            if cycle_pos == self._CALM_STEPS:
                burst_factor = 1.0 + (self._BURST_MULT - 1.0) * self._burst_mask
                grad = grad * burst_factor

        self._step += 1
        return loss_val, [grad]
