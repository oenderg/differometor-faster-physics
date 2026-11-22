# Faster Physics

Differometor evaluates optical sensitivity and differentiates that objective so an optimizer can tune an interferometer. A full evaluation of the public UIFO runs 101 frequencies and returns gradients for 193 design variables. Our changes preserve the requested workload in float64 and complex128. All system sizes come from the optical setup.

## Development benchmark on the team judge

On the team judge, the paired draft took **27.913 s** for the dense original and **0.508 s** for the optimized code, about **55× faster**. Both passed on one logical CPU on the same i7-13700K desktop under WSL2. The tested commits were `8169dff` and `a8b6c67`. [Draft JSON, plots and validation logs](reports/)

**Second setup, not tuned for:** Feyer's public aLIGO benchmark (`benchmark_aligo.py`: mirror mechanics, squeezed light, 101 frequencies, 29 design variables) took **15.41 s** with the original and **0.28 s** with our code on the same judge (about **55× faster**), and passed. Our log sensitivity is within 3e-11 of Feyer's reference (original: 9e-11). We made no aLIGO-specific change. aLIGO run

The development history below uses separate judge sessions. The submission pair uses the dense baseline and frozen simulator in the same session.

| Stage | Complete first run | Evidence |
| --- | ---: | --- |
| Original dense simulator | 28.448 s | Dense control from the first sparse judge run |
| Sparse CSC and SuperLU | 0.832 s | Initial sparse result |
| Sparse correctness fixes and lower setup and callback overhead | 0.634 s | Revised sparse result |
| Optical block elimination | 0.593 s | Block solve result |
| Compiled detector readout | 0.502 s | Merged main after PR #9 |
| Faster XLA compilation | 0.404 s | Merged main after PR #13 |
| Later solver, setup and compile work (PRs #12, #14, #18, #21, #22, #28) | 0.344 s | Main after the overnight fixes |
| Batched signal-frequency SuperLU solves (PR #29) | 0.331 s | Merged main after PR #29 |

## What changed

- **Sparse optical systems.** Assemble the carrier, signal and noise terms in sparse form and solve with SciPy SuperLU. The sparse simulation path uses a JAX host callback and a custom reverse-mode rule. The backward pass solves the transposed systems and propagates cotangents to component parameters, including sweep values and tied design variables.
- **Quantum noise.** For the selection vector $c$, solve $A^H w = c$. Evaluating $\operatorname{Re}(w^H Nw)$ gives $\operatorname{Re}(c^H A^{-1} N A^{-H}c)$ and removes an additional solve. The expression follows directly from the same linear system.
- **Optical block elimination.** Eliminate simple blocks before factoring the reduced optical system. Reconstruct the fields and account for the eliminated variables in the backward pass.
- **Detector readout.** A NumPy host callback demodulates the complex128 field and returns analytical input cotangents, including the reductions for broadcast dimensions. A JAX tangent rule supports forward and higher derivatives of the detector itself. The original JAX expression handles other shapes and dtypes.
- **XLA settings.** The remaining JAX programs are small. On the pinned stack, we disable the CPU fusion emitters and most XLA optimizations for these programs, and compile each program's LLVM module as one part instead of splitting it for parallel code generation. The compiler change passed the eight-setup verification and Feyer approved portable compiler settings at 20:51. Compiler experiment · Feyer answer

## Correctness and measurement

The unchanged public benchmark checks positive, finite sensitivity, the reference log-sensitivity tolerance and a central-difference gradient spot check. The team’s eight-setup verifier also compares outputs, carrier powers, design and sweep gradients, and expected zero gradient entries against pre-merge main and the original dense simulator. It estimates the dense reference’s own floating-point variation by nudging sweep values by a relative $10^{-13}$. All eight setups passed for the merged readout candidate. This follows Feyer’s 17:15 clarification that the same inputs must yield the same outputs, including gradients. Verification summary · Feyer clarification

**Scope of differentiation:** reverse-mode gradients (`jax.grad`, `jax.vjp`, `jax.jacrev`, as used by the benchmark) match the original. Forward-mode differentiation (`jax.jvp`, `jax.jacfwd`) through `simulate` is not supported by the sparse solver yet and raises an error instead of returning a wrong value. Linked parameters are evaluated as real numbers: an unphysical negative input to a `sqrt` or `log` link (for example a negative laser power feeding an amplitude link) gives NaN, where the original continues with complex values.

The scored interval includes imports, initialization, input building, lowering and compilation, simulation, readout, the reverse pass and transfer to the host. Our judge protocol alternates the order over at least two paired rounds, runs validation each time and reports the minimum passing first-run time for each commit. The judge has shown about 3–4% variation between comparable single runs. Warm repeat timings are reported separately. [Measurement protocol](../SUBMISSION.md)

The sparse simulator supports reverse-mode differentiation through `simulate`. Forward-mode differentiation is available for the detector readout. Fresh-checkout imports have added roughly a quarter-second in measured judge runs. Voyager frequency modulation is numerically degenerate even for the dense reference, so the standard verifier uses other Voyager configurations. Feyer evaluates a separate optical setup.

The frozen before/after artifacts will be recorded in [final/](reports/) during the Sunday judge slot. Judge handoff and freeze
