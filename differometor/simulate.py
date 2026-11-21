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

@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _sparse_simulate_core(
    p_base: jnp.ndarray,
    car_ch_vals: jnp.ndarray,
    sig_ch_vals: jnp.ndarray,
    nsv: jnp.ndarray,
    topo: SparseTopology,
):
    b_car = car_ch_vals.shape[1] if topo.car_ch_idx.size > 0 else 1
    b_sig = sig_ch_vals.shape[1] if topo.sig_ch_idx.size > 0 else b_car
    b_n = b_sig if topo.noise_groups_sweep else b_car
    D = topo.num_detectors

    specs = (
        jax.ShapeDtypeStruct((b_car, topo.cn), jnp.complex128),
        jax.ShapeDtypeStruct((b_sig, topo.sn), jnp.complex128),
        jax.ShapeDtypeStruct((D, b_sig), jnp.float64),
        jax.ShapeDtypeStruct((b_car, topo.nnz_c), jnp.complex128),
        jax.ShapeDtypeStruct((b_sig, topo.nnz_s), jnp.complex128),
        jax.ShapeDtypeStruct((D, b_sig, topo.sn), jnp.complex128),
        jax.ShapeDtypeStruct((b_n, topo.nnz_n), jnp.complex128),
    )

    def _call(p_b, c_v, s_v, n_v):
        return sparse_simulate_fwd_host(topo, p_b, c_v, s_v, n_v)

    carrier, signal, noise, _, _, _, _ = jax.pure_callback(
        _call, specs, p_base, car_ch_vals, sig_ch_vals, nsv, vmap_method="sequential"
    )
    return carrier, signal, noise


def _sparse_simulate_core_fwd(p_base, car_ch_vals, sig_ch_vals, nsv, topo):
    b_car = car_ch_vals.shape[1] if topo.car_ch_idx.size > 0 else 1
    b_sig = sig_ch_vals.shape[1] if topo.sig_ch_idx.size > 0 else b_car
    b_n = b_sig if topo.noise_groups_sweep else b_car
    D = topo.num_detectors

    specs = (
        jax.ShapeDtypeStruct((b_car, topo.cn), jnp.complex128),
        jax.ShapeDtypeStruct((b_sig, topo.sn), jnp.complex128),
        jax.ShapeDtypeStruct((D, b_sig), jnp.float64),
        jax.ShapeDtypeStruct((b_car, topo.nnz_c), jnp.complex128),
        jax.ShapeDtypeStruct((b_sig, topo.nnz_s), jnp.complex128),
        jax.ShapeDtypeStruct((D, b_sig, topo.sn), jnp.complex128),
        jax.ShapeDtypeStruct((b_n, topo.nnz_n), jnp.complex128),
    )

    def _call(p_b, c_v, s_v, n_v):
        return sparse_simulate_fwd_host(topo, p_b, c_v, s_v, n_v)

    carrier, signal, noise, car_A, sig_A, w, n_vals = jax.pure_callback(
        _call, specs, p_base, car_ch_vals, sig_ch_vals, nsv, vmap_method="sequential"
    )
    res = (p_base, car_ch_vals, sig_ch_vals, nsv, carrier, signal, noise, car_A, sig_A, w, n_vals)
    return (carrier, signal, noise), res


def _sparse_simulate_core_bwd(topo, res, g):
    p_base, car_ch_vals, sig_ch_vals, nsv, carrier, signal, noise, car_A, sig_A, w, n_vals = res
    g_carrier, g_signal, g_noise = g
    gp_spec = jax.ShapeDtypeStruct(p_base.shape, jnp.float64)

    def _call_bwd(p_b, c_v, s_v, n_v, car, sig, noi, cA, sA, ww, nv, gc, gs, gn):
        return sparse_simulate_bwd_host(topo, p_b, c_v, s_v, n_v, car, sig, noi, cA, sA, ww, nv, gc, gs, gn)

    g_p_base = jax.pure_callback(
        _call_bwd,
        gp_spec,
        p_base,
        car_ch_vals,
        sig_ch_vals,
        nsv,
        carrier,
        signal,
        noise,
        car_A,
        sig_A,
        w,
        n_vals,
        g_carrier,
        g_signal,
        g_noise,
        vmap_method="sequential",
    )
    return g_p_base, None, None, None


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
    if topology.opt_param_idx.size > 0:
        p_base = p_base.at[0, topology.opt_param_idx].set(
            optimized_parameters[topology.opt_val_idx].astype(jnp.float64)
        )

    if linked_groups:
        for link_idx, (src_idx, dst_idx) in linked_groups.items():
            link_fn = LINKING_FUNCTIONS[link_idx]
            vals = vmap(link_fn)(p_base[0, src_idx])
            p_base = p_base.at[0, dst_idx].set(jnp.real(vals).astype(jnp.float64))

    carrier, signal, noise = _sparse_simulate_core(
        p_base,
        carrier_changing_values,
        signal_changing_values,
        noise_selection_vectors,
        topology,
    )
    return carrier.T, signal.T, noise.squeeze()


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
