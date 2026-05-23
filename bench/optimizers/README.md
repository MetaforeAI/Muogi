# Benchmark Optimizer Wrappers — Canonical Configs (Muogi bench)

All optimizers are constructed via the single entry point
`bench.optimizers.wrappers.build_optimizer(name, params, lr)`. Everything
except `lr` is pinned here.

The Muogi bench's optimizer configs are **independent of the RACASO
bench's**. The two benches run different problem classes designed to
validate different claims; sharing configs across them is a hygiene
nicety but not a correctness requirement. Where the defaults happen to
match RACASO/bench's (Adam, AdamW, Yogi), they do; differences are
intentional and documented below.

## Canonical hyperparameter table

| name              | source                                                 | call                                                                                                | status        |
|-------------------|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------|---------------|
| adam              | `torch.optim`                                          | `Adam(lr=lr, betas=(0.9, 0.999), eps=1e-8)`                                                          | implemented   |
| adamw             | `torch.optim`                                          | `AdamW(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)`                                      | implemented   |
| yogi              | vendored at `bench/optimizers/yogi.py` (Zaheer et al. 2018) | `Yogi(lr=lr, betas=(0.9, 0.999), eps=1e-3, initial_accumulator=1e-6, weight_decay=0.0)`             | implemented   |
| muon              | Keller Jordan reference                                | `Muon(lr=lr, momentum=0.95, nesterov=True, ns_steps=5)`                                              | not vendored  |
| lion              | `lion-pytorch` or vendored                             | `Lion(lr=lr, betas=(0.9, 0.99), weight_decay=0.0)`                                                   | not vendored  |
| sophia            | official `Liuhong99/Sophia`                            | `SophiaG(lr=lr, betas=(0.965, 0.99), rho=0.04, weight_decay=0.0, eps=1e-15)`                          | not vendored  |
| soap              | official Vyas et al. reference                         | `SOAP(lr=lr, betas=(0.95, 0.95), shampoo_beta=0.95, eps=1e-8, weight_decay=0.0, precondition_frequency=10)` | not vendored  |
| naive_yogi_muon   | `bench/optimizers/naive_yogi_muon.py` (this repo)      | `NaiveYogiMuon(lr=lr, betas=(0.9, 0.999), eps_yogi=1e-3, ns5_iters=5)`                              | implemented   |
| muogi             | `Muogi/muogi.py`                                       | `Muogi(lr=lr)` — Muogi default config                                                                | implemented*  |
| ramuogi           | `Muogi/ramuogi.py`                                     | `RAMuogi(lr=lr)` — RAMuogi default config                                                            | implemented*  |

`*` Muogi / RAMuogi entries succeed when those modules are importable
from `sys.path`. The wrapper does not import them at module import time;
it imports lazily inside `build_optimizer` so bench infrastructure
tests can run without Muogi on the path.

## The Naive Yogi-Muon anti-baseline

This is the optimizer the Muogi paper says fails. It's implemented from
scratch in `naive_yogi_muon.py` (rather than calling Muogi's internal
helpers) so it's truly free of Muogi machinery — a clean reference
implementation of the composition the paper contradicts.

Design:
- Yogi moment updates (β₁ = 0.9, β₂ = 0.999, ε_yogi = 1e-3)
- Bias correction
- Form `D = m_hat / (sqrt(v_hat) + ε_yogi)` — Yogi's preconditioned direction
- For 2-D params: 5-iteration NS5 on `D`, no row-scale injection, no
  spread cap, no safety chain
- For 1-D params: use `D` directly
- Step: `W ← W - lr · output`

NS5 inline (`_naive_ns5`):
- Normalize spectral norm by dividing by Frobenius norm
- Transpose if M > N (so the smaller inner product is XX^T)
- 5 iterations of Jordan's polynomial: a=3.4445, b=-4.7750, c=2.0315

The contrast with Muogi:
- Muogi calls NS5 on `R · m_hat` — row-scaled momentum. NS5 sees an
  un-normalized input and produces `polar(R · m_hat)`. The variance
  signal R enters via row-scale and gets preserved in the orthogonal
  rotation.
- Naive Yogi-Muon calls NS5 on `m_hat / sqrt(v_hat)` — Yogi's full
  preconditioned direction. NS5 sees a near-orthogonal input and
  averages it to the spectral mean. Yogi's burst-aware variance signal
  is destroyed.

The Muogi paper's M1 claim is validated empirically iff Q1's variance
fidelity plot shows Muogi tracking Yogi-alone while Naive Yogi-Muon
diverges.

## Vendoring checklist for Phase 2

Before Phase 3 can sweep all 10 optimizers:

- [ ] **Muon** — copy the Keller Jordan reference implementation into
      `bench/optimizers/muon.py` (Apache-2.0 / public). Wire into
      `_build_muon` in `wrappers.py`.
- [ ] **Lion** — either `pip install lion-pytorch` or vendor
      `bench/optimizers/lion.py` from `lucidrains/lion-pytorch` (MIT).
- [ ] **Sophia** — vendor `bench/optimizers/sophia.py` from the official
      `Liuhong99/Sophia` repo. Use `SophiaG` (Gauss-Newton-Bartlett).
- [ ] **SOAP** — vendor `bench/optimizers/soap.py` from the Vyas et al.
      reference. Apache-2.0.

Each vendored file must include the upstream commit SHA and license
header. Hyperparameter defaults must match the table above exactly.

## Why a single wrapper module

Without this single source, drift creeps in (different default eps for
Adam between two problem classes → invisible bias in the comparison).
The wrapper pattern guarantees every benchmark in the repo calls
`build_optimizer("adam", ...)` and gets bit-identical optimizer state at
init.
