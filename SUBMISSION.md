# Faster Physics: our Differometor

This folder is a complete, runnable Differometor checkout. Only `differometor/` differs from the public
challenge repository; `benchmark.py`, `benchmark_aligo.py`, `start_here.py`, `examples/` and
`requirements.txt` are unchanged.

## Run it

Linux or WSL2, Python 3.12, from this folder:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python benchmark.py --cpus 1
.venv/bin/python benchmark_aligo.py --cpus 1
```

Both print `PASS`. The very first run after unpacking also compiles Python bytecode, so it is slower;
the second fresh process shows the steady first-run time.

## Results (complete first run, one logical CPU)

Measured on our team judge (Intel i7-13700K, WSL2, `--cpus 1`), original and ours in the same session:

| Benchmark | Original | Ours | Speedup |
|---|---:|---:|---:|
| `benchmark.py` (UIFO, 101 frequencies, 193 gradients) | 28.5 s | **0.33 s** | ~87× |
| `benchmark_aligo.py` (aLIGO, 101 frequencies, 29 gradients) | 15.4 s | **0.28 s** | ~55× |

Raw JSON reports, plots and PASS logs: [submission/reports/](submission/reports/).
About 0.18 s of our time is `import jax` itself.

## What we changed

The full explanation is in [submission/WRITEUP.md](submission/WRITEUP.md). In short:

- **Sparse solves on the host.** The optical systems are assembled sparse and solved with SciPy's SuperLU
  inside `jax.pure_callback`, with an exact hand-written reverse-mode rule (`jax.custom_vjp`) that reuses the
  forward LU factors for the adjoint solves.
- **Less linear algebra.** Simple optical blocks are eliminated before factoring (751 → 330 unknowns on the
  benchmark); quantum noise uses the identity Re(wᴴNw) with one adjoint solve; independent signal
  frequencies are factored in bounded block-diagonal batches.
- **Less compilation.** Detector readout runs as one small primitive with an analytical pullback, and
  `differometor/__init__.py` sets XLA CPU compile options that only change code generation (approved in
  #feyer, Sat 20:51).

Nothing is cached between runs, nothing depends on the benchmark's setup, sizes or counts, and everything
stays in float64/complex128.

## Correctness

- Both public benchmarks pass (reference sensitivity and finite-difference gradient check).
- Outputs **and** gradients were compared with the original Differometor on 11 setups (UIFO grids,
  aLIGO, Voyager with amplitude modulation, cavities, sweeps, lower-sideband and dark detectors, total
  internal reflection) and on several thousand randomly generated setups. Differences are at floating-point
  roundoff, except where the original itself is ill-conditioned.
- Results are deterministic across runs.
- Scope: reverse-mode gradients (`jax.grad`, `jax.vjp`, `jax.jacrev`) match the original. Forward-mode
  differentiation (`jax.jvp`, `jax.jacfwd`) through `simulate` is not supported and raises an error.
- Linked parameters are evaluated as real numbers: an unphysical negative input to a `sqrt` or `log` link
  (e.g. a negative laser power) gives NaN, where the original continues with complex values.
