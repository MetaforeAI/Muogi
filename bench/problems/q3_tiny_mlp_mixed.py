"""Q3 — Tiny MLP with mixed gradient distributions.

Validates Muogi paper claim **M7**: Muogi beats Muon-alone and
Yogi-alone on heterogeneous-topology problems. The heterogeneity
matters because different parameter blocks see gradients with very
different magnitudes and distributions.

Problem
-------
2-layer MLP with structure ``input(10) -> hidden(32) -> output(4)``.
Activation: tanh on the hidden layer; linear output. Loss: MSE
against a small synthetic regression target.

The four parameter tensors:
    - W1 shape (32, 10), b1 shape (32,)
    - W2 shape (4, 32),  b2 shape (4,)

The narrow input embedding (W1) sees low-magnitude per-row
gradients; the wider hidden-to-output projection (W2) sees
medium-magnitude rows; the bias vectors b1 and b2 see 1-D gradients
that an NS5-family optimizer must safe-skip (no orthogonalization on
vectors). This is exactly the topology the M7 claim depends on.

Synthetic regression task
-------------------------
- Fixed input batch of size 32 generated at problem init via
  ``self._generator``.
- Target = ``A_true @ input + b_true + small_noise`` where
  ``A_true`` and ``b_true`` are fixed per seed.
- Loss = mean squared error over the batch.

Class attributes
----------------
- ``name = "q3_tiny_mlp_mixed"``
- ``max_steps = 5000``
- ``converged_tol = 0.01``
"""

from __future__ import annotations

from typing import List

import torch

from bench.problems.base import BenchProblem


class Q3TinyMlpMixed(BenchProblem):
    """Tiny MLP with mixed gradient distributions — validates M7."""

    name = "q3_tiny_mlp_mixed"
    max_steps = 5000
    converged_tol = 0.01

    _IN_DIM = 10
    _HIDDEN_DIM = 32
    _OUT_DIM = 4
    _BATCH_SIZE = 32

    def __init__(self, seed: int, device: str = "cpu") -> None:
        super().__init__(seed, device=device)
        # Fixed input batch (32, 10).
        self._x = torch.randn(
            (self._BATCH_SIZE, self._IN_DIM), generator=self._generator
        ).to(self.device)
        # Fixed teacher: small linear projection + bias + noise.
        a_true = 0.5 * torch.randn(
            (self._OUT_DIM, self._IN_DIM), generator=self._generator
        )
        b_true = 0.1 * torch.randn(
            (self._OUT_DIM,), generator=self._generator
        )
        noise = 0.05 * torch.randn(
            (self._BATCH_SIZE, self._OUT_DIM), generator=self._generator
        )
        self._y = (self._x @ a_true.to(self.device).T + b_true.to(self.device) + noise.to(self.device))

    def init_params(self) -> List[torch.Tensor]:
        # Xavier-ish small inits via the seeded generator.
        scale1 = (1.0 / self._IN_DIM) ** 0.5
        scale2 = (1.0 / self._HIDDEN_DIM) ** 0.5
        w1 = (scale1 * torch.randn(
            (self._HIDDEN_DIM, self._IN_DIM), generator=self._generator
        )).to(self.device)
        b1 = torch.zeros((self._HIDDEN_DIM,), device=self.device)
        w2 = (scale2 * torch.randn(
            (self._OUT_DIM, self._HIDDEN_DIM), generator=self._generator
        )).to(self.device)
        b2 = torch.zeros((self._OUT_DIM,), device=self.device)
        for t in (w1, b1, w2, b2):
            t.requires_grad_(True)
        return [w1, b1, w2, b2]

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        w1, b1, w2, b2 = params
        h = torch.tanh(self._x @ w1.T + b1)
        y_pred = h @ w2.T + b2
        diff = y_pred - self._y
        return (diff * diff).mean()
