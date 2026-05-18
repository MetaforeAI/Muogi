# Muogi / RAMuogi

**Momentum-Yogi Orthogonalized by Newton-Schulz** (Muogi), and the RAdam-rectified extension (RAMuogi).

Muon-style spectral preconditioning but with Yogi's additive variance tracker injected into the NS5 loop, plus a relative-threshold spread clamp so heterogeneous architectures don't blow up the polynomial. RAMuogi adds a RAdam cold-start gate on top so the spectral pathway doesn't fire on uncalibrated variance estimates in the first few hundred steps.

Four-layer safety chain: L1 spread-cap clamp, L2 NS5 convergence safe-skip, L3 vanilla Yogi fallback, L4 RAdam variance gate (RAMuogi only). Every failure surfaces in `.get_telemetry()`.

## Install

Two files, no dependencies beyond PyTorch.

```bash
curl -O https://raw.githubusercontent.com/MetaforeAI/Muogi/main/muogi.py
curl -O https://raw.githubusercontent.com/MetaforeAI/Muogi/main/ramuogi.py
```

Or clone:

```bash
git clone https://github.com/MetaforeAI/Muogi.git
```

## Use

```python
from ramuogi import RAMuogi   # or: from muogi import Muogi

opt = RAMuogi(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    spread_cap=10.0,                  # L1 K-bound on row-scale spread
    ns5_max_iters=5,                  # Jordan canonical
    ns5_convergence_threshold=0.64,   # = 0.8² spectral bar
)

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
```

Use plain `Muogi` if your variance estimates are already warm (transfer learning from a checkpoint). Use `RAMuogi` otherwise.

## Telemetry

```python
t = opt.get_telemetry()
print(f"ns5: {t['ns5_success_count']} ok / {t['ns5_skip_count']} skip  "
      f"last_res={t['last_ns5_residual']:.3f}  "
      f"r_t={t['last_r_t']:.3f}  rect_skip={t['rectification_skip_count']}")
```

`ns5_skip_count` near zero post-warmup means the spectral side is doing real work. Skip rate near one means it's degenerating into vanilla Yogi with overhead — tune `spread_cap` lower, or accept that group isn't a fit for orthogonalization.

## Paper

[RAMuogi_Paper.pdf](RAMuogi_Paper.pdf) — Yogi+Muon composition, NS5 with the cheater's-choice injection, the four-layer safety chain, and step-1000 gradient-health traces across an architectural evolution.

## License

MIT.
