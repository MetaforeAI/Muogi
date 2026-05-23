# Muogi / RAMuogi Benchmark Suite

Controlled-conditions comparison of Muogi and RAMuogi against 7 baseline
optimizers + the Naive Yogi-Muon anti-baseline on 5 problems designed to
validate the Muogi paper's claims.

## Phase status

- **Phase 1 (this commit) — infrastructure only.** `BenchProblem` base,
  `build_optimizer` dispatch including the Naive Yogi-Muon anti-baseline,
  `run_bench.py` harness with Muogi-specific telemetry columns,
  `plot_bench.py` skeleton, `tests/` sanity suite. No problem modules.
  No runs. No figures.
- **Phase 2 — problem modules.** Five files land in `problems/`
  (`q1_burst_variance.py`, `q2_polar_decomposition.py`,
  `q3_tiny_mlp_mixed.py`, `q4_ns5_convergence_stress.py`,
  `q5_radam_cold_start.py`).
- **Phase 3 — execution.** The sweep runs on a clean host; results land
  in `bench_results.csv`.
- **Phase 4 — plotting.** `plot_bench.py` stubs become real figures.

## The Naive Yogi-Muon anti-baseline

`Muogi/bench/optimizers/naive_yogi_muon.py` implements **the optimizer
the Muogi paper says fails**. It's not a strawman — it's the specific
composition Muogi's "cheater's choice" design contradicts.

Naive Yogi-Muon:
1. Computes Yogi moments `m_t, v_t`
2. Bias-corrects to `m_hat, v_hat`
3. Forms `D = m_hat / (sqrt(v_hat) + ε)` — Yogi's full preconditioned direction
4. Feeds `D` into NS5 (5-iteration Newton-Schulz)
5. Steps with `W ← W - lr · NS5(D)`

The Muogi paper argues this loses Yogi's burst-aware variance signal
because NS5 sees a near-orthogonal input and averages it to the spectral
mean. Muogi avoids this by feeding `R · m_hat` (row-scaled momentum,
NOT pre-normalized direction) into NS5, so the variance signal enters
via row-scale and gets preserved in the orthogonal rotation.

**The headline plot** for the Muogi paper's M1 claim
(`plot_variance_fidelity` in `plot_bench.py`) overlays three
`||v_hat||` trajectories on the Q1 bursty-gradient problem:

- **Yogi-alone** (gold standard — what burst-aware variance looks like)
- **Muogi** (should track Yogi within tolerance)
- **Naive Yogi-Muon** (should diverge from Yogi by a clear margin)

If Muogi tracks Yogi and Naive doesn't, the cheater's-choice claim is
validated empirically. If they both track or both diverge, the claim
needs revising.

## How to run (when Phase 2 lands)

Single config:

```bash
python -m bench.run_bench --problem q1_burst_variance \
    --optimizer muogi --lr 1e-3 --seed 0
```

Full sweep (all problems × all optimizers × all LRs × all seeds):

```bash
python -m bench.run_bench --sweep --out bench_results.csv
```

In Phase 1 the sweep is a documented no-op (no problems registered) —
it writes a header-only CSV and exits cleanly. Single-config invocation
raises a clear error.

## CSV schema

Columns of `bench_results.csv`, in order:

| column                   | type       | meaning |
|--------------------------|------------|---------|
| `problem`                | str        | problem short name |
| `optimizer`              | str        | optimizer short name |
| `lr`                     | float      | learning rate |
| `seed`                   | int        | random seed |
| `steps`                  | int        | total steps recorded in the trajectory |
| `convergence_step`       | int        | first step where `converged()` returned True; `-1` if never |
| `final_loss`             | float      | last recorded loss (may be `nan`) |
| `wall_clock_per_step_us` | float      | mean `optimizer.step()` time in microseconds |
| `nan_count`              | int        | count of non-finite losses observed |
| `l1_count`               | int        | Muogi L1 spread-cap clamp activations |
| `l2_count`               | int        | Muogi L2 NS5 convergence safe-skip count |
| `l3_count`               | int        | Muogi L3 vanilla Yogi fallback count |
| `l4_count`               | int        | RAMuogi L4 RAdam variance gate count |
| `l5_count`               | int        | reserved (Muogi has 4 layers; column kept for schema parity) |
| `ns5_success_rate`       | float      | fraction of NS5 attempts that converged |
| `r_t_value`              | float      | final RAMuogi r_t gate value (0 for non-RAMuogi) |
| `variance_l2_norm`       | float      | end-of-run sum of `exp_avg_sq` norms |
| `loss_trajectory`        | str        | full per-step loss history, semicolon-separated floats |

## Baselines

| optimizer          | status (Phase 1)                | source |
|--------------------|---------------------------------|--------|
| adam               | implemented                     | `torch.optim.Adam` |
| adamw              | implemented                     | `torch.optim.AdamW` |
| yogi               | implemented (vendored)          | `bench/optimizers/yogi.py` (Zaheer et al. 2018) |
| muon               | NotImplementedError             | Keller Jordan reference — vendor in Phase 2 |
| lion               | NotImplementedError             | `lion-pytorch` or vendored |
| sophia             | NotImplementedError             | official `Liuhong99/Sophia` |
| soap               | NotImplementedError             | Vyas et al. reference |
| naive_yogi_muon    | implemented (THIS REPO)         | `bench/optimizers/naive_yogi_muon.py` |
| muogi              | implemented (lazy import)       | `Muogi/muogi.py` |
| ramuogi            | implemented (lazy import)       | `Muogi/ramuogi.py` |

See `bench/optimizers/README.md` for canonical hyperparameter configs
and the Phase 2 vendoring checklist.

## Files

```
bench/
├── README.md                       # this file
├── requirements_bench.txt          # pinned versions
├── __init__.py
├── problems/
│   ├── __init__.py                 # empty marker — Phase 2 fills it
│   └── base.py                     # BenchProblem ABC
├── optimizers/
│   ├── README.md                   # canonical configs + vendoring checklist
│   ├── __init__.py
│   ├── wrappers.py                 # build_optimizer dispatch
│   ├── yogi.py                     # vendored (Zaheer et al. 2018)
│   └── naive_yogi_muon.py          # THE ANTI-BASELINE (M1 contradiction)
├── run_bench.py                    # harness — run_one + --sweep
├── plot_bench.py                   # load_results + plot stubs
└── tests/
    ├── __init__.py
    └── test_infrastructure.py      # Phase 1 sanity tests
```

## Constraints (from `CLAUDE.md`)

- This bench imports only `torch` (plus `pandas` for plotting). It does
  not import any heavyweight upstream, Triton, or anything that fires CUDA/Triton
  autotune at import time.
- AST-parse every file before committing:
  `python -c "import ast; ast.parse(open(p).read())"`.
- Phase 3 runs on a clean host. Phase 1 ships infrastructure only.
