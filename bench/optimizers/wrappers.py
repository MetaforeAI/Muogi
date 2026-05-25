"""Canonical optimizer-construction entry point for the Muogi benchmark suite.

``build_optimizer(name, params, lr)`` returns a fully configured
``torch.optim.Optimizer``. All hyperparameters other than the learning
rate are pinned here.

Every optimizer is vendored as a standalone source file in this
``bench/optimizers/`` directory — no sibling-repo imports, no sys.path
gymnastics. This includes the optimizers from our own sibling research
projects (Liger, RACASO): we treat them exactly like we treat Lion and
Yogi — copy the source file, document the upstream commit at the top
of the copy, build via a normal Python import.

Canonical configs (single source of truth — see ``README.md``):

    adam              : torch.optim.Adam(lr, betas=(0.9, 0.999), eps=1e-8)
    adamw             : torch.optim.AdamW(lr, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=0.01)
    yogi              : Yogi(lr, betas=(0.9, 0.999), eps=1e-3,
                             initial_accumulator=1e-6, weight_decay=0.0)
                        — bench/optimizers/yogi.py (Zaheer et al. 2018)
    lion              : Lion(lr, betas=(0.9, 0.99), weight_decay=0.0)
                        — bench/optimizers/lion.py (Chen et al. 2023)
    naive_yogi_muon   : NaiveYogiMuon(lr, betas=(0.9, 0.999), eps_yogi=1e-3,
                                      ns5_iters=5)
                        — bench/optimizers/naive_yogi_muon.py
                        — the ANTI-BASELINE for Muogi paper claim M1
    muogi             : Muogi(lr, default Muogi config)
                        — bench/optimizers/muogi.py (this repo's optimizer)
    ramuogi           : RAMuogi(lr, default RAMuogi config)
                        — bench/optimizers/ramuogi.py (this repo's optimizer)
    liger             : Liger(lr, betas=(0.9, 0.99), eps_yogi=1e-3, wd=0.0)
                        — bench/optimizers/liger.py (sibling repo, vendored)
    racaso            : RACASO(lr, default RACASO config)
                        — bench/optimizers/racaso.py (sibling repo, vendored)
    muon              : Muon(lr, momentum=0.95, nesterov=True, ns_steps=5)
                        — bench/optimizers/muon.py (Keller Jordan, MIT)
    sophia            : NotImplementedError until vendored from the
                        official Sophia repo
    soap              : NotImplementedError until vendored from Vyas et al.
"""

from __future__ import annotations

from typing import List

import torch


KNOWN_OPTIMIZERS = (
    "adam",
    "adamw",
    "yogi",
    "muon",
    "lion",
    "sophia",
    "soap",
    "naive_yogi_muon",
    "muogi",
    "ramuogi",
    "liger",
    "racaso",
)


def _vendor_pointer(opt_name: str) -> str:
    return (
        f"{opt_name} is not vendored yet; "
        "see bench/optimizers/README.md for the canonical source and "
        "pinned version to drop into this wrapper before benchmarks run."
    )


# ── Constructors ─────────────────────────────────────────────────────────


def _build_adam(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


def _build_adamw(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )


def _build_yogi(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.yogi import Yogi

    return Yogi(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-3,
        initial_accumulator=1e-6,
        weight_decay=0.0,
    )


def _build_lion(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.lion import Lion

    return Lion(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0)


def _build_naive_yogi_muon(
    params: List[torch.Tensor], lr: float
) -> torch.optim.Optimizer:
    from bench.optimizers.naive_yogi_muon import NaiveYogiMuon

    return NaiveYogiMuon(
        params,
        lr=lr,
        betas=(0.9, 0.999),
        eps_yogi=1e-3,
        ns5_iters=5,
    )


def _build_muogi(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.muogi import Muogi

    return Muogi(params, lr=lr)


def _build_ramuogi(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.ramuogi import RAMuogi

    return RAMuogi(params, lr=lr)


def _build_liger(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.liger import Liger

    return Liger(params, lr=lr, betas=(0.9, 0.99), eps_yogi=1e-3, weight_decay=0.0)


def _build_racaso(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.racaso import RACASO

    return RACASO(params, lr=lr)


def _build_muon(params: List[torch.Tensor], lr: float) -> torch.optim.Optimizer:
    from bench.optimizers.muon import Muon

    return Muon(
        params,
        lr=lr,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    )


def build_optimizer(
    name: str, params: List[torch.Tensor], lr: float
) -> torch.optim.Optimizer:
    """Construct a baseline optimizer by canonical short name.

    Args:
        name: one of ``KNOWN_OPTIMIZERS``.
        params: list of parameter tensors with ``requires_grad=True``.
        lr: learning rate.

    Returns:
        A fully constructed ``torch.optim.Optimizer``.

    Raises:
        ValueError: if ``name`` is not in ``KNOWN_OPTIMIZERS``.
        NotImplementedError: for baselines whose implementation has not
            been vendored yet (muon / sophia / soap).
    """
    if name not in KNOWN_OPTIMIZERS:
        raise ValueError(
            f"unknown optimizer name '{name}'; "
            f"known: {sorted(KNOWN_OPTIMIZERS)}"
        )
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}")
    if not params:
        raise ValueError("params must be a non-empty list of tensors")

    builders = {
        "adam": _build_adam,
        "adamw": _build_adamw,
        "yogi": _build_yogi,
        "lion": _build_lion,
        "naive_yogi_muon": _build_naive_yogi_muon,
        "muogi": _build_muogi,
        "ramuogi": _build_ramuogi,
        "liger": _build_liger,
        "racaso": _build_racaso,
        "muon": _build_muon,
    }
    if name in builders:
        return builders[name](params, lr)
    if name in ("sophia", "soap"):
        raise NotImplementedError(_vendor_pointer(name))

    raise AssertionError(f"unreachable: optimizer {name} not dispatched")
