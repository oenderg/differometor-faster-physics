import os as _os


def _configure_compiler():
    """Make XLA compile the small readout and gradient programs faster.

    Differometor's heavy numerical work runs in NumPy/SciPy host callbacks, so the XLA programs that
    remain are small and compile time dominates their cost. Both settings below only change how XLA
    generates code, not what it computes: on the public setups, results stay bit-identical (fusion
    emitters) or within ~1e-10 relative (reduced optimisation), far below the simulator's own
    floating-point noise. Allowed by the Feyer team (#feyer, Sat 21 Nov 20:51).

    `XLA_FLAGS` is read when the CPU backend starts, which happens after this import in normal use.
    The flag is only added for the jaxlib release it was verified with, because XLA aborts the
    process on unknown flags.
    """
    try:
        from jaxlib import version as _jaxlib_version

        jaxlib_version = _jaxlib_version.__version__
    except Exception:  # pragma: no cover - jaxlib layout differs
        jaxlib_version = ""
    flag = "--xla_cpu_use_fusion_emitters=false"
    existing = _os.environ.get("XLA_FLAGS", "")
    if jaxlib_version.startswith("0.9.") and "xla_cpu_use_fusion_emitters" not in existing:
        _os.environ["XLA_FLAGS"] = f"{existing} {flag}".strip()

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
