"""
JAX-based optical simulation runner.

This module builds sparse linear systems from a `differometor` `Setup` and solves:
- a carrier (steady-state) system,
- a signal (sideband) system,
- and a quantum noise propagation system.

The main public entry points are:
- `run`: build once and simulate once
- `run_setups`: run multiple setups over a shared frequency grid
- `run_build_step` / `run_simulation_step` / `simulate`

Conventions used throughout:
- Double precision (`float64` / `complex128`) is preserved throughout.
- Forward evaluation and exact analytical reverse-mode AD run inside a single
  `jax.custom_vjp` + `jax.pure_callback` kernel using vectorized NumPy and
  SciPy SuperLU (`scipy.sparse.linalg.splu`), reusing each LU factorization
  across forward signal and adjoint quantum-noise solves while eliminating
  redundant XLA compilation and dense matrix allocations.
"""

from functools import partial
from time import time
import jax
import jax.numpy as jnp
import numpy as np
from threading import Lock
from jax import lax, vmap, jit

from differometor.setups import Setup
from differometor.build import (
    build,
    build_sparse_simulation_arrays,
    pairs_to_arrays,
    prepare_arrays,
    LINKING_FUNCTIONS,
)
from differometor.components import FUNCTIONS
from differometor._sparse_kernel import (
    SparseTopology,
    apply_host_links_fwd,
    apply_host_links_bwd,
    sparse_simulate_fwd_host,
    sparse_simulate_bwd_host,
)


jax.config.update("jax_enable_x64", True)


def solve(matrix: jnp.ndarray, right_hand_side: jnp.ndarray) -> jnp.ndarray:
    """Solve a dense linear system for each batch item (retained for API compatibility)."""
    right_hand_side = right_hand_side.astype(jnp.complex128)
    matrix = matrix.astype(jnp.complex128)
    return jnp.linalg.solve(matrix, right_hand_side)


def update(
    parameters: jnp.ndarray,
    matrix: jnp.ndarray,
    function_input_indices: jnp.ndarray,
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    function_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Legacy dense matrix update helper (retained for API compatibility)."""
    function_inputs = parameters[function_input_indices]
    output_matrix = vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))(function_indices, function_inputs)
    outputs = output_matrix[output_indices[0], output_indices[1]]
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)


def expand_parameters(array: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """Expand a single parameter row into multiple rows by replacing selected entries."""
    r = values.shape[1]
    expanded = jnp.broadcast_to(array, (r, array.shape[1]))
    return expanded.at[:, indices].set(values.T.astype(array.dtype))


# -----------------------------------------------------------------------------
# Single-callback Sparse Forward + Exact Analytical VJP
# -----------------------------------------------------------------------------

def _insert_optimized_parameters(parameters, optimized, topology, bind_parameters):
    if not bind_parameters:
        return parameters
    bound = np.array(parameters, copy=True)
    bound[0, topology.opt_param_idx] = np.asarray(optimized)[topology.opt_val_idx]
    return bound


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _sparse_simulate_core(
    optimized: jnp.ndarray,
    p_base: jnp.ndarray,
    car_ch_vals: jnp.ndarray,
    sig_ch_vals: jnp.ndarray,
    nsv: jnp.ndarray,
    topo: SparseTopology,
    bind_parameters: bool,
):
    return _sparse_forward(
        optimized, p_base, car_ch_vals, sig_ch_vals, nsv, topo, bind_parameters, save_residuals=False
    )


class _ForwardTape:
    """Keep the forward's host residuals for the matching pullback, outside JAX.

    The reverse pass needs large intermediate arrays (assembled matrices, adjoint fields). Returning
    them through `pure_callback` copies them into XLA buffers and back. Instead the forward keeps
    them here together with a copy of its (small) inputs. The pullback uses them only if its inputs
    are byte-identical and the topology is the same object; the forward is deterministic, so the
    residuals are then exactly the ones it would recompute. Otherwise it recomputes them.
    """

    def __init__(self):
        self._lock = Lock()
        self._entry = None

    @staticmethod
    def _key(arrays):
        return tuple((a.dtype.str, a.shape, a.tobytes()) for a in arrays)

    def save(self, topo, bind, inputs, residuals):
        entry = (topo, bind, self._key(inputs), tuple(residuals))
        with self._lock:
            self._entry = entry

    def get(self, topo, bind, inputs):
        with self._lock:
            entry = self._entry
        if entry is None or entry[0] is not topo or entry[1] != bind:
            return None
        if entry[2] != self._key(inputs):
            return None
        return entry[3]


_forward_tape = _ForwardTape()


def _sparse_forward(optimized, p_base, car_ch_vals, sig_ch_vals, nsv, topo, bind_parameters, *, save_residuals):
    b_car = car_ch_vals.shape[1] if topo.car_ch_idx.size > 0 else 1
    b_sig = sig_ch_vals.shape[1] if topo.sig_ch_idx.size > 0 else b_car
    D = topo.num_detectors

    specs = (
        jax.ShapeDtypeStruct((topo.cn, b_car), jnp.complex128),
        jax.ShapeDtypeStruct((topo.sn, b_sig), jnp.complex128),
        jax.ShapeDtypeStruct((D, b_sig), jnp.float64),
    )

    def _call(opt, base, car_sweep, sig_sweep, vectors):
        inputs = tuple(np.asarray(x) for x in (opt, base, car_sweep, sig_sweep, vectors))
        bound = _insert_optimized_parameters(inputs[1], inputs[0], topo, bind_parameters)
        if topo.host_linked_groups:
            bound = apply_host_links_fwd(bound, topo)
        carrier, signal, noise, *residuals = sparse_simulate_fwd_host(
            topo, bound, *inputs[2:], save_factors=save_residuals
        )
        if save_residuals:
            _forward_tape.save(topo, bind_parameters, inputs, residuals)
        return carrier.T, signal.T, noise

    return jax.pure_callback(
        _call, specs, optimized, p_base, car_ch_vals, sig_ch_vals, nsv, vmap_method="sequential"
    )


def _sparse_simulate_core_fwd(optimized, p_base, car_ch_vals, sig_ch_vals, nsv, topo, bind_parameters):
    inputs = (optimized, p_base, car_ch_vals, sig_ch_vals, nsv)
    results = _sparse_forward(*inputs, topo, bind_parameters, save_residuals=True)
    return results, (*inputs, *results)


def _sparse_simulate_core_bwd(topo, bind_parameters, res, g):
    specs = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in res[:5])

    def _call_bwd(opt, base, car_sweep, sig_sweep, vectors, carrier, signal, noise, g_carrier, g_signal, g_noise):
        inputs = tuple(np.asarray(x) for x in (opt, base, car_sweep, sig_sweep, vectors))
        bound_pre = _insert_optimized_parameters(inputs[1], inputs[0], topo, bind_parameters)
        bound = apply_host_links_fwd(bound_pre, topo) if topo.host_linked_groups else bound_pre
        residuals = _forward_tape.get(topo, bind_parameters, inputs)
        if residuals is None:
            # Another forward ran in between (or this pullback runs elsewhere): recompute exactly.
            _, _, _, *residuals = sparse_simulate_fwd_host(topo, bound, *inputs[2:], save_factors=True)
        g_base, *rest = sparse_simulate_bwd_host(
            topo, bound, *inputs[2:], np.asarray(carrier).T, np.asarray(signal).T, noise,
            *residuals, np.asarray(g_carrier).T, np.asarray(g_signal).T, g_noise,
        )
        if topo.host_linked_groups:
            apply_host_links_bwd(g_base, bound_pre, topo)
        g_opt = np.zeros(opt.shape, dtype=g_base.dtype)
        if bind_parameters:
            indices = topo.opt_param_idx
            np.add.at(g_opt, topo.opt_val_idx, g_base[0, indices])
            g_base[0, indices] = 0
        gradients = (g_opt, g_base, *rest)
        return tuple(gradient.astype(spec.dtype, copy=False) for gradient, spec in zip(gradients, specs))

    return jax.pure_callback(
        _call_bwd,
        specs,
        *res,
        *g,
        vmap_method="sequential",
    )


_sparse_simulate_core.defvjp(_sparse_simulate_core_fwd, _sparse_simulate_core_bwd)


def simulate(
    optimized_parameters: jnp.ndarray,
    parameters: jnp.ndarray,
    carrier_changing_values: jnp.ndarray,
    signal_changing_values: jnp.ndarray,
    noise_selection_vectors: jnp.ndarray,
    linked_groups: dict,
    topology: SparseTopology,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run one full sparse simulation pass: carrier, signal, and quantum noise.

    Returns
    -------
    carrier : jnp.ndarray
        Carrier solution, shape (CN, CV).
    signal : jnp.ndarray
        Signal solution, shape (SN, SV).
    noise : jnp.ndarray
        Quantum noise amplitude spectral density, shape (D, SV) after squeeze.
    """
    p_base = parameters
    bind_parameters = topology.opt_param_idx.size > 0
    if optimized_parameters is None and not bind_parameters:
        optimized_parameters = jnp.zeros((0,), dtype=p_base.dtype)
    if linked_groups:
        if bind_parameters:
            p_base = p_base.at[0, topology.opt_param_idx].set(
                optimized_parameters[topology.opt_val_idx].astype(p_base.dtype), unique_indices=True
            )
        bind_parameters = False
        link_input = p_base[0]
        for link_idx, (src_idx, dst_idx) in linked_groups.items():
            link_fn = LINKING_FUNCTIONS[link_idx]
            vals = vmap(link_fn)(link_input[src_idx])
            p_base = p_base.at[0, dst_idx].set(jnp.real(vals).astype(jnp.float64))

    carrier, signal, noise = _sparse_simulate_core(
        optimized_parameters,
        p_base,
        carrier_changing_values,
        signal_changing_values,
        noise_selection_vectors,
        topology,
        bind_parameters,
    )
    return carrier, signal, noise.squeeze()


def run_build_step(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
    timeit: bool = False,
) -> tuple:
    """Build a `Setup` into sparse simulation-ready arrays and metadata."""
    start = time()
    simulation_arrays, metadata = build_sparse_simulation_arrays(
        setup, changing_pairs, changing_values, optimization_pairs
    )

    if timeit:
        jax.block_until_ready(simulation_arrays)
        return time() - start, (simulation_arrays, metadata)

    return simulation_arrays, *metadata


def run_simulation_step(
    simulation_arrays: dict,
    jit_simulation: bool = False,
    timeit: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | tuple[float, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Run `simulate` using a pre-built set of arrays."""
    simulation_function = simulate

    if jit_simulation:
        simulation_function = jit(simulate)
        _ = simulation_function(**simulation_arrays)

    start = time()
    results = simulation_function(**simulation_arrays)

    if timeit:
        for result in results:
            jax.block_until_ready(result)
        return time() - start, results

    return results


def run(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convenience wrapper: build a setup and immediately run the simulation."""
    simulation_arrays, *metadata = run_build_step(setup, changing_pairs, changing_values, optimization_pairs)
    carrier, signal, noise = run_simulation_step(simulation_arrays)
    return carrier, signal, noise, *metadata


def run_setups(setups: list[Setup], frequencies: jnp.ndarray):
    """Run a list of setups over a shared frequency sweep."""
    results = []
    for setup in setups:
        results.append(run(setup, [("f", "frequency")], frequencies))
    return results
