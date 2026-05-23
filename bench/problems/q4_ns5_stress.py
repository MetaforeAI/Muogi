"""Q4 — NS5 convergence-failure stress.

Validates Muogi paper claims **M3** (NS5 safe-skip when the input
spectral norm exceeds the Schulz polynomial's convergence radius)
and **M6** (Yogi fallback when the NS5 safe-skip fires).

Background
----------
The 5-iteration Newton-Schulz polynomial used by Muon and the
Muogi family converges only when the input matrix's spectral norm
lies in ``[0, sqrt(3)]``. Outside that radius the polynomial
diverges. Muogi's L2 safety layer detects this and skips the NS5
step; Muogi's L3 layer falls back to vanilla Yogi on that param.
This problem deliberately injects gradients with known large
spectral norms to fire those safety layers.

Problem
-------
Quadratic loss against a fixed target M, shape ``(6, 6)``:

    f(W) = 1/2 * || W - M ||_F^2

The natural gradient is ``W - M``. ``loss_and_grad`` rescales this
gradient via SVD so that its spectral norm matches a target value
drawn from the cycle:

    [sqrt(3) + 0.1,  2*sqrt(3),  5*sqrt(3),  10*sqrt(3)]

cycling on every call. The rescaling preserves the singular vectors
and only replaces the singular values with a uniform vector of the
target spectral norm.

Reconstruction formula
----------------------
Given ``U, s, Vh = torch.linalg.svd(grad)``::

    g_rescaled = U @ torch.diag(s_target_vec) @ Vh

where ``s_target_vec = s_target * ones_like(s)``. All singular values
are mapped to ``s_target`` so the matrix has uniform spectrum at the
target norm. Recovering convergence under these conditions is hopeless
for a vanilla NS5 step — the point of the problem is the safety-layer
counts in the harness telemetry.

Class attributes
----------------
- ``name = "q4_ns5_stress"``
- ``max_steps = 1000``
- ``converged_tol = 1e-2``
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch

from bench.problems.base import BenchProblem


class Q4Ns5Stress(BenchProblem):
    """NS5 convergence-failure stress — validates M3 + M6."""

    name = "q4_ns5_stress"
    max_steps = 1000
    converged_tol = 1e-2

    _DIM = 6

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        self._target = torch.randn(
            (self._DIM, self._DIM), generator=self._generator
        ).to(self.device)
        sqrt3 = math.sqrt(3.0)
        # Spectral-norm cycle — first entry sits just outside the
        # convergence radius, the rest are deep outside.
        self._spectral_cycle: Tuple[float, ...] = (
            sqrt3 + 0.1,
            2.0 * sqrt3,
            5.0 * sqrt3,
            10.0 * sqrt3,
        )
        self._step = 0

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

    def loss_and_grad(
        self, params: List[torch.Tensor]
    ) -> Tuple[float, List[torch.Tensor]]:
        (w,) = params
        with torch.no_grad():
            diff = w.detach() - self._target
            loss_val = 0.5 * float((diff * diff).sum().item())

            grad_natural = diff
            # SVD of the natural gradient.
            u_mat, s_vec, vh_mat = torch.linalg.svd(
                grad_natural, full_matrices=False
            )

            s_target = self._spectral_cycle[
                self._step % len(self._spectral_cycle)
            ]
            s_target_vec = torch.full_like(s_vec, fill_value=s_target)

            grad_rescaled = u_mat @ torch.diag(s_target_vec) @ vh_mat

        self._step += 1
        return loss_val, [grad_rescaled]
