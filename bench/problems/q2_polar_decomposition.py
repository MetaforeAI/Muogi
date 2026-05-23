"""Q2 — Polar decomposition fidelity.

Validates the **core NS5 property**: Newton-Schulz iteration on a
gradient matrix recovers the orthogonal factor of its polar
decomposition. If an NS5-family optimizer is wired correctly, it
should solve this problem cleanly: the natural gradient direction
already points toward the target via the polar factor U, and
NS5 should preserve that direction while scaling moves toward the
PSD factor H.

Problem
-------
Minimize the Frobenius distance to a target matrix ``M`` of shape
``(6, 6)`` with a known polar decomposition ``M = U H`` where:

- ``U`` is orthogonal, constructed via QR of a Gaussian matrix.
- ``H`` is PSD, constructed via its known SVD ``H = U_h diag(lambda) U_h^T``
  with positive eigenvalues drawn uniformly from ``[0.5, 1.5]``.

Loss:

    f(W) = 1/2 * || W - M ||_F^2

The natural gradient is ``W - M``. The PSD factor H stretches the
gradient anisotropically in the directions of the eigenbasis of H,
and the orthogonal factor U rotates it. This is the cleanest
test of whether an NS5-family optimizer recovers the orthogonal
direction.

Class attributes
----------------
- ``name = "q2_polar_decomposition"``
- ``max_steps = 1000``
- ``converged_tol = 1e-4`` — clean problem; all NS5-family
  optimizers should hit this cleanly.
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


class Q2PolarDecomposition(BenchProblem):
    """Polar decomposition fidelity — validates the core NS5 property."""

    name = "q2_polar_decomposition"
    max_steps = 1000
    converged_tol = 1e-4

    _DIM = 6

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        # Construct U orthogonal via QR of a Gaussian.
        a = torch.randn(
            (self._DIM, self._DIM), generator=self._generator
        )
        q, r = torch.linalg.qr(a)
        # Stabilize sign so the QR factor is reproducibly orthogonal.
        sign = torch.sign(torch.diagonal(r))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        u_orth = q * sign.unsqueeze(0)

        # Construct H PSD via known SVD H = U_h diag(lambda) U_h^T.
        b = torch.randn(
            (self._DIM, self._DIM), generator=self._generator
        )
        q_h, r_h = torch.linalg.qr(b)
        sign_h = torch.sign(torch.diagonal(r_h))
        sign_h = torch.where(sign_h == 0, torch.ones_like(sign_h), sign_h)
        u_h = q_h * sign_h.unsqueeze(0)
        lam = 0.5 + torch.rand(
            (self._DIM,), generator=self._generator
        )  # in [0.5, 1.5]
        h_psd = u_h @ torch.diag(lam) @ u_h.T

        # Target M = U H.
        self._target = (u_orth @ h_psd).to(self.device)

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
