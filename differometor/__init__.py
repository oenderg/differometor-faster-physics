import os as _os


def _configure_compiler():
    """Make XLA compile the small readout and gradient programs faster.

    Differometor's heavy numerical work runs in NumPy/SciPy host callbacks, so the XLA programs that
    remain are small and compile time dominates their cost. Both settings below only change how XLA
    generates code, not what it computes: on the public setups, results stay bit-identical (fusion
    emitters) or within ~1e-10 relative (reduced optimisation), far below the simulator's own
    floating-point noise. Allowed by the Feyer team (#feyer, Sat 21 Nov 20:51).

    `XLA_FLAGS` is read when the CPU backend starts, which happens after this import in normal use.
    Only add CPU flags for explicitly supported jaxlib versions. An unknown flag can abort XLA
    during backend initialization.
    """
    try:
        from jaxlib import version as _jaxlib_version

        jaxlib_version = _jaxlib_version.__version__
    except Exception:  # pragma: no cover - jaxlib layout differs
        jaxlib_version = ""
    flags = {
        "xla_cpu_use_fusion_emitters": "false",
        # With the fusion emitters off, all kernels share one LLVM module. By default XLA splits it into
        # up to 32 parts for parallel codegen; each part is printed, cloned via bitcode and compiled on
        # its own. Compiling it as one part does the same work with less overhead (bit-identical code).
        "xla_cpu_parallel_codegen_split_count": "1",
    }
    existing = _os.environ.get("XLA_FLAGS", "")
    if jaxlib_version in ("0.9.0", "0.9.0.1"):
        added = [f"--{name}={value}" for name, value in flags.items() if name not in existing]
        _os.environ["XLA_FLAGS"] = " ".join([existing, *added]).strip()

    try:
        import jax

        jax.config.update("jax_disable_most_optimizations", True)
    except Exception:  # pragma: no cover - option renamed or removed in another JAX release
        pass


_configure_compiler()

from differometor.setups import Setup
from differometor.simulate import run, run_build_step, run_simulation_step, simulate
from differometor.components import power_detector, signal_detector


def __getattr__(name: str):
    if name == "visualize_setup":
        from differometor.plot import visualize_setup
        return visualize_setup
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
