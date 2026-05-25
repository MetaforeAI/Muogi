"""Muon optimizer — vendored reference implementation.

Newton-Schulz-orthogonalized SGD-momentum optimizer for 2-D parameter
matrices, originally by Keller Jordan.

Provenance:
    Author:     Keller Jordan
    Upstream:   https://github.com/KellerJordan/Muon/blob/master/muon.py
    License:    MIT
    Vendored:   2026-05 — single-file port for the Muogi bench harness.

Update rule per 2-D parameter ``p`` with gradient ``g`` at step ``t``:

    m_t      = β · m_{t-1} + g                          # SGD momentum
    g_eff    = β · m_t + g    (if nesterov)            # Nesterov lookahead
              | m_t           (otherwise)
    O_t      = NS5_orthogonalize(g_eff, ns_steps)       # polar factor
    p ← p · (1 - lr · weight_decay)                     # AdamW-style decay
    p ← p - lr · O_t · sqrt(max(1, m/n))                # Muon shape scale

For 1-D parameters (norms, biases, learned scalars) Muon's reference
implementation routes to AdamW — orthogonalization is undefined on
1-D tensors. The fallback here mirrors the same convention Muogi uses
(its own 1-D branch is vanilla Yogi; for the Muon baseline we use
AdamW so the comparison is to "Muon's intended deployment shape").

The Newton-Schulz polynomial uses the same Jordan coefficients
(3.4445, -4.7750, 2.0315) that Muogi and naive_yogi_muon use, so all
three optimizers' spectral paths compute on identical polynomials —
the differences are in how Muogi/RAMuogi inject Yogi variance and how
the safety chain catches non-convergent inputs.

This file is intentionally ~150 lines and free of imports from sibling
optimizers — no Muogi machinery leaks into the Muon baseline.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


# Newton-Schulz iteration coefficients from Keller Jordan's reference.
# Per-iteration update:   X ← a·X + (b·A + c·A²)·X    where A = X X^T
_NS5_A = 3.4445
_NS5_B = -4.7750
_NS5_C = 2.0315


@torch.no_grad()
def _zeropower_via_newtonschulz5(
    G: torch.Tensor, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Newton-Schulz orthogonalization (5-iteration polynomial).

    Returns an approximation to the orthogonal factor of ``G`` via the
    Jordan-coefficient polynomial. Operates in fp32 internally for
    numerical stability and returns in the input dtype.

    The matrix is transposed if ``M > N`` so the inner product ``XX^T``
    stays square in the smaller of the two dimensions (NS5 efficiency
    contract — polynomial cost is O(min(m,n)² · max(m,n))).
    """
    assert G.ndim == 2, f"NS5 expects 2-D, got {tuple(G.shape)}"
    orig_dtype = G.dtype
    X = G.to(torch.float32)
    # Frobenius normalize so spectral norm starts ≤ 1.
    X = X / (X.norm() + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = _NS5_B * A + _NS5_C * (A @ A)
        X = _NS5_A * X + B @ X
    if transposed:
        X = X.T
    return X.to(orig_dtype)


class Muon(Optimizer):
    """Muon optimizer — Newton-Schulz-orthogonalized SGD-momentum.

    Args:
        params: iterable of parameters.
        lr: learning rate. Muon typically wants Lion-family LRs
            (3-10x lower than Adam) because the orthogonalized update
            already has unit-magnitude singular values.
        momentum: SGD momentum coefficient. Default 0.95 (Jordan ref).
        nesterov: if True, use Nesterov lookahead. Default True.
        weight_decay: AdamW-style decoupled weight decay. Default 0.0.
        ns_steps: Newton-Schulz polynomial iterations. Default 5.
        adamw_betas: betas for the 1-D AdamW fallback path.
        adamw_eps: epsilon for the 1-D AdamW fallback path.
        adamw_wd: weight decay for the 1-D AdamW fallback path.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid Muon learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid Muon momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            adamw_b1, adamw_b2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]
            adamw_wd = group["adamw_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                # ── 1-D path: AdamW fallback ─────────────────────────
                if p.ndim != 2:
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(adamw_b1).add_(g, alpha=1.0 - adamw_b1)
                    exp_avg_sq.mul_(adamw_b2).addcmul_(g, g, value=1.0 - adamw_b2)
                    bc1 = 1.0 - adamw_b1 ** t
                    bc2 = 1.0 - adamw_b2 ** t
                    if adamw_wd != 0.0:
                        p.mul_(1.0 - lr * adamw_wd)
                    denom = (exp_avg_sq / bc2).sqrt().add_(adamw_eps)
                    p.addcdiv_(exp_avg / bc1, denom, value=-lr)
                    continue

                # ── 2-D path: NS5-orthogonalized SGD-momentum ────────
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g_eff = g.add(buf, alpha=momentum)
                else:
                    g_eff = buf

                O = _zeropower_via_newtonschulz5(g_eff, steps=ns_steps)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Muon shape scale: sqrt(max(1, m/n)).
                m_dim, n_dim = p.shape[-2], p.shape[-1]
                shape_scale = max(1.0, m_dim / n_dim) ** 0.5
                p.add_(O, alpha=-lr * shape_scale)

        return loss
