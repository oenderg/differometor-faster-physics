"""
Pure-NumPy + SciPy SuperLU sparse optical simulation kernel and exact reverse-mode AD.

This module implements:
1. All 19 Differometor component functions in vectorized real-float64 -> complex128 NumPy
   along with their exact analytical reverse-mode VJPs (matching JAX's unconjugated
   complex cotangent convention: delta L = Re(sum(gy * delta y))).
2. A single-pass sparse forward + exact reverse-mode backward engine for carrier,
   signal, and quantum noise (using the exact identity Re(c^H A^-1 N A^-H c) = Re(w^H N w)
   and reusing one SuperLU factorization per frequency for both A x = b and A^H w = c).
"""

from __future__ import annotations

from importlib.machinery import EXTENSION_SUFFIXES
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path
from threading import Lock

import numpy as np
import jax

from differometor.components import (
    C_LIGHT,
    EPSILON_0C,
    F0,
    H_PLANCK,
    LAMBDA,
    UNIT_VACUUM,
    X_SCALE,
)

_DEG = np.pi / 180.0
_TWO_PI = 2.0 * np.pi
_INV_C = 1.0 / C_LIGHT
_INV_F0 = 1.0 / F0
_INV_CX = 1.0 / (C_LIGHT * X_SCALE)
_Z_SCALE = X_SCALE * _TWO_PI / LAMBDA
_SQ_K = 1.0 / (20.0 * np.log10(np.e))
_NOISE_SCALE = 2.0 * UNIT_VACUUM * H_PLANCK * F0 * 0.25


def _load_pinned_sparse_extension(name, *subdirectories):
    """Load the pinned wheel's native solver without eagerly importing SciPy sparse."""
    try:
        scipy = find_spec("scipy")
        if scipy is None or scipy.origin is None:
            return None
        package = Path(scipy.origin).parent
        # Same check as importlib.metadata.version("scipy") == "1.18.1" for a pip install,
        # without scanning every distribution on sys.path (~1.5 ms). Other layouts fall back
        # to the public SciPy import, which gives bit-identical results.
        if not (package.parent / "scipy-1.18.1.dist-info").is_dir():
            return None
        directory = package.joinpath("sparse", *subdirectories)
        for suffix in EXTENSION_SUFFIXES:
            binary = directory / f"{name}{suffix}"
            if binary.is_file():
                spec = spec_from_file_location(name, binary)
                if spec is None or spec.loader is None:
                    continue
                module = module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    except (ImportError, OSError):
        return None
    return None


_gstrf = getattr(_load_pinned_sparse_extension("_superlu", "linalg", "_dsolve"), "gstrf", None)
_sparsetools = _load_pinned_sparse_extension("_sparsetools")
_csr_matvec = getattr(_sparsetools, "csr_matvec", None)
_csr_matvecs = getattr(_sparsetools, "csr_matvecs", None)
_LU_PANEL_SIZE = 4
_LU_RELAX = 2


def _make_sparse_matrix(*args, **kwargs):
    from scipy.sparse import csc_matrix

    return csc_matrix(*args, **kwargs)


def _factorize(matrix, data, *, ordering="COLAMD", panel_size=_LU_PANEL_SIZE, relax=_LU_RELAX):
    """Return no factor for a singular optical system, matching the dense solve."""
    matrix.data = data
    try:
        if _gstrf is not None:
            return _gstrf(
                matrix.shape[0], data.size, data, matrix.indices, matrix.indptr,
                ilu=False,
                options={
                    "ColPerm": ordering,
                    "DiagPivotThresh": 1.0,
                    "PanelSize": panel_size,
                    "Relax": relax,
                },
                csc_construct_func=_make_sparse_matrix,
            )
        from scipy.sparse.linalg import splu

        csc = _make_sparse_matrix((data, matrix.indices, matrix.indptr), shape=matrix.shape)
        return splu(csc, permc_spec=ordering, diag_pivot_thresh=1.0, panel_size=panel_size, relax=relax)
    except RuntimeError as error:
        if "singular" not in str(error).lower():
            raise
        return None


class _CSCPattern:
    """Precomputed row/column reduction indices for a static CSC sparsity pattern."""

    __slots__ = (
        "shape", "indices", "indptr",
        "row_order", "cols_in_row_order", "rows_with_terms", "row_starts", "all_rows_filled",
        "cols_with_terms", "col_starts", "all_cols_filled",
    )

    def __init__(self, size, indices, indptr):
        self.shape = (size, size)
        self.indices = indices
        self.indptr = indptr
        self.row_order = None
        self.cols_in_row_order = None
        self.rows_with_terms = None
        self.row_starts = None
        self.all_rows_filled = False
        self.cols_with_terms = None
        self.col_starts = None
        self.all_cols_filled = False

    def ensure_ready(self):
        if self.row_order is not None:
            return
        # The pattern is shared through the topology, and `multiply` takes `row_order is not None` to
        # mean "ready": fill every other field first and publish `row_order` last, so a concurrent
        # first call (threads, or independent callbacks in one XLA program) never sees a half state.
        size = self.shape[0]
        row_order = np.argsort(self.indices, kind="stable")
        col_lens = np.diff(self.indptr)
        self.cols_in_row_order = np.repeat(np.arange(size, dtype=np.intp), col_lens)[row_order]
        counts = np.bincount(self.indices, minlength=size)
        rows_with_terms = np.flatnonzero(counts)
        self.row_starts = (np.cumsum(counts) - counts)[rows_with_terms]
        self.rows_with_terms = rows_with_terms
        self.all_rows_filled = int(rows_with_terms.size) == size
        cols_with_terms = np.flatnonzero(col_lens)
        self.col_starts = self.indptr[cols_with_terms]
        self.cols_with_terms = cols_with_terms
        self.all_cols_filled = int(cols_with_terms.size) == size
        self.row_order = row_order


class _CSCWorkspace:
    """Keep a sparsity pattern and apply the current matrix when refining a solve."""

    __slots__ = ("shape", "indices", "indptr", "pattern", "data")

    def __init__(self, size, indices, indptr, pattern=None):
        self.shape = (size, size)
        self.indices = indices
        self.indptr = indptr
        self.pattern = _CSCPattern(size, indices, indptr) if pattern is None else pattern
        self.data = None

    def multiply(self, values, trans):
        p = self.pattern
        if p.row_order is None:
            p.ensure_ready()

        if values.ndim == 1:
            if trans == "N":
                if p.all_rows_filled:
                    return np.add.reduceat(self.data[p.row_order] * values[p.cols_in_row_order], p.row_starts)
                product = np.zeros_like(values)
                if p.rows_with_terms.size:
                    product[p.rows_with_terms] = np.add.reduceat(
                        self.data[p.row_order] * values[p.cols_in_row_order], p.row_starts
                    )
                return product
            if trans == "H":
                if p.all_cols_filled:
                    out = np.add.reduceat(self.data * np.conj(values)[self.indices], p.col_starts)
                    np.conj(out, out=out)
                    return out
                product = np.zeros_like(values)
                if p.cols_with_terms.size:
                    res = np.add.reduceat(self.data * np.conj(values)[self.indices], p.col_starts)
                    np.conj(res, out=res)
                    product[p.cols_with_terms] = res
                return product
            if p.all_cols_filled:
                return np.add.reduceat(self.data * values[self.indices], p.col_starts)
            product = np.zeros_like(values)
            if p.cols_with_terms.size:
                product[p.cols_with_terms] = np.add.reduceat(self.data * values[self.indices], p.col_starts)
            return product

        if trans == "N":
            data = self.data[p.row_order]
            picked = values[p.cols_in_row_order]
            starts = p.row_starts
            filled = p.rows_with_terms
        else:
            data = np.conj(self.data) if trans == "H" else self.data
            picked = values[self.indices]
            starts = p.col_starts
            filled = p.cols_with_terms
        product = np.zeros_like(values)
        if filled.size:
            product[filled] = np.add.reduceat(data[:, None] * picked, starts, axis=0)
        return product


class _SweepFactorization:
    """Reuse the first successful column ordering within a single sweep."""

    def __init__(self, matrix):
        self.matrix = matrix
        self.order = None

    def factor(self, values, *, first_ordering="COLAMD"):
        self.matrix.data = values
        if self.order is None:
            factors = _factorize(self.matrix, values, ordering=first_ordering)
            if factors is not None:
                self.order = np.argsort(factors.perm_c)
                self.inverse_order = factors.perm_c
                lengths = np.diff(self.matrix.indptr)[self.order]
                pointers = np.empty_like(self.matrix.indptr)
                pointers[0] = 0
                np.cumsum(lengths, out=pointers[1:])
                offsets = self.matrix.indptr[self.order] - pointers[:-1]
                self.positions = np.repeat(offsets, lengths) + np.arange(values.size)
                rows = self.matrix.indices[self.positions]
                self.reordered = _CSCWorkspace(self.matrix.shape[0], rows, pointers)
            return factors

        factors = _factorize(self.reordered, values[self.positions], ordering="NATURAL")
        if factors is None:
            return _factorize(self.matrix, values)
        return _FrequencySolver(factors, self.order, self.inverse_order)


def _solve_refined(matrix, factors, rhs, trans="N"):
    """Correct the solve residual in float64 using the existing LU factors."""
    solution = factors.solve(rhs, trans=trans)
    residual = rhs - matrix.multiply(solution, trans)
    return solution + factors.solve(residual, trans=trans)


class _FrequencySolver:
    """One frequency's LU factors, kept in the ordering used to compute them."""

    __slots__ = ("lu", "order", "inverse_order")

    def __init__(self, lu, order, inverse_order):
        self.lu, self.order, self.inverse_order = lu, order, inverse_order

    def solve(self, rhs, trans="N"):
        if trans == "N":
            return self.lu.solve(rhs)[self.inverse_order]
        return self.lu.solve(rhs[self.order], trans=trans)


def _same_memory(a, b):
    """True if both arrays are the same object or identical views of one live buffer."""
    if a is b:
        return True
    root = a.base
    return (
        root is not None
        and root is b.base
        and a.dtype == b.dtype
        and a.shape == b.shape
        and a.strides == b.strides
        and a.ctypes.data == b.ctypes.data
    )


class _ForwardFactors:
    """Transfer one sweep's numerical factors from the forward callback to a pullback."""

    def __init__(self):
        self._lock = Lock()
        self._entry = None

    def save(self, topology, values, factors):
        # The forward never writes to `values` after this point, and neither does anyone else.
        entry = (topology, values, factors)
        with self._lock:
            self._entry = entry

    def take(self, topology, values):
        # Claim ownership before using SuperLU; simultaneous pullbacks can always refactor.
        with self._lock:
            entry = self._entry
            self._entry = None
        if entry is None or entry[0] is not topology:
            return None
        _, saved, factors = entry
        if _same_memory(saved, values):  # the host tape hands the pullback the forward's own array
            return factors
        if saved.dtype != values.dtype or saved.shape != values.shape:
            return None
        if saved.dtype == np.complex128 and values.strides[-1] == 16:
            if not np.array_equal(saved.view(np.uint64), values.view(np.uint64)):
                return None
        elif not np.array_equal(saved.view(np.void), values.view(np.void)):
            return None
        return factors


_signal_factors = _ForwardFactors()
_carrier_factors = _ForwardFactors()


_FREQUENCIES_AT_ONCE = 16


class _SweepBatches:
    """Factored block-diagonal groups of independent frequency systems."""

    def __init__(self, matrix, values):
        ordering = _SweepFactorization(matrix)
        # MMD_ATA can crash inside SuperLU on structurally singular matrices.
        # A present diagonal in every column is a sufficient structural guard.
        columns = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))
        diagonal = np.unique(matrix.indices[matrix.indices == columns])
        strategy = "MMD_ATA" if diagonal.size == matrix.shape[0] else "COLAMD"
        first = ordering.factor(values[0], first_ordering=strategy)
        if first is None:
            raise ValueError("first system is singular")
        width = matrix.shape[0]
        cols = ordering.reordered.indices
        ptr = ordering.reordered.indptr
        nnz = cols.size
        # Limit the native factor's working set as the optical core grows.
        size = min(_FREQUENCIES_AT_ONCE, 12288 // width, 65536 // nnz)
        if size < 2:
            raise ValueError("systems are too large to batch")
        self.batch_size = size
        block_cols = cols + np.arange(size, dtype=cols.dtype)[:, None] * width
        block_ptr = ptr[:-1] + np.arange(size, dtype=ptr.dtype)[:, None] * nnz
        permuted = np.ascontiguousarray(values[1:, ordering.positions])
        factor_patterns = {}
        self.groups = []
        self.first = first
        self.order = ordering.order
        self.inverse_order = ordering.inverse_order
        for start in range(1, len(values), size):
            stop = min(start + size, len(values))
            count = stop - start
            if count not in factor_patterns:
                pointers = np.empty(count * width + 1, dtype=ptr.dtype)
                pointers[:-1] = block_ptr[:count].ravel()
                pointers[-1] = count * nnz
                factor_patterns[count] = _CSCWorkspace(count * width, block_cols[:count].ravel(), pointers)
            workspace = factor_patterns[count]
            entries = permuted[start-1:stop-1].ravel()
            lu = _factorize(workspace, entries, ordering="NATURAL", panel_size=1, relax=1)
            if lu is None:
                raise ValueError("block contains a singular system")
            self.groups.append((slice(start,stop),lu))

    def refine_into(self, matrix, entries, rhs, trans, output):
        matrix.data = entries[0]
        output[0] = _solve_refined(matrix, self.first, rhs[0], trans=trans)
        pattern = matrix.pattern
        pattern.ensure_ready()
        n = matrix.shape[0]
        order = self.order
        inverse = self.inverse_order
        if trans == "N":
            coefficient_order = pattern.row_order
            pick = inverse[pattern.cols_in_row_order]
            starts = pattern.row_starts
            filled = pattern.rows_with_terms
        else:
            pick = matrix.indices
            starts = matrix.indptr[pattern.cols_with_terms]
            filled = pattern.cols_with_terms
        if _csr_matvec is not None and (_csr_matvecs is not None or rhs.ndim == 2):
            if trans == "N":
                counts = np.bincount(matrix.indices, minlength=n)
                indptr = np.empty(n + 1, dtype=matrix.indptr.dtype)
                indptr[0] = 0
                np.cumsum(counts, out=indptr[1:])
            else:
                indptr = matrix.indptr
            nnz = len(pick)
            block_cols = pick.astype(matrix.indices.dtype, copy=False) + np.arange(self.batch_size, dtype=matrix.indices.dtype)[:, None] * n
            block_starts = indptr[:-1] + np.arange(self.batch_size, dtype=indptr.dtype)[:, None] * nnz
            structures = {}
        else:
            structures = None
        for part, lu in self.groups:
            current = rhs[part]
            shape = current.shape
            flat_shape = (shape[0] * n,) + shape[2:]
            if trans != "N":
                current = current[:, order]
            solution = lu.solve(current.reshape(flat_shape), trans=trans).reshape(shape)
            coefs = entries[part]
            if trans == "N":
                coefs = coefs[:, coefficient_order]
            elif trans == "H":
                coefs = coefs.conj()
            if structures is None:
                selected = solution[:, pick]
                if len(shape) == 3:
                    coefs = coefs[..., None]
                selected *= coefs
                dot = np.add.reduceat(selected, starts, axis=1)
                if filled.size != n:
                    product = np.zeros_like(solution)
                    product[:, filled] = dot
                    dot = product
            else:
                count = shape[0]
                m = n * count
                if count not in structures:
                    ptr = np.empty(m + 1, dtype=indptr.dtype)
                    ptr[:-1] = block_starts[:count].reshape(-1)
                    ptr[-1] = nnz * count
                    structures[count] = ptr, block_cols[:count].reshape(-1)
                ptr, cols = structures[count]
                dot = np.zeros(shape, dtype=solution.dtype)
                x = np.ascontiguousarray(solution).reshape(-1)
                coef = np.ascontiguousarray(coefs).reshape(-1)
                if len(shape) == 2:
                    _csr_matvec(m, m, ptr, cols, coef, x, dot.reshape(-1))
                else:
                    _csr_matvecs(m, m, shape[2], ptr, cols, coef, x, dot.reshape(-1))
            if trans != "N":
                dot = dot[:, order]
            np.subtract(current, dot, out=dot)
            solution += lu.solve(dot.reshape(flat_shape), trans=trans).reshape(shape)
            output[part] = solution[:, inverse] if trans == "N" else solution


def _allows_frequency_batches(topo):
    # Laser frequency modulation subtracts sidebands that can cancel almost
    # completely at a detector. Keep the original evaluation order for these
    # optical sources; grouping changes floating-point roundoff near the null.
    frequency_modulation = (11, 18)
    return not any(
        group.f_idx in frequency_modulation
        for groups in (topo.signal_groups_static, topo.signal_groups_sweep)
        for group in groups
    )


def _solve_signal_systems(
    topo, values, first_rhs, first_trans, second_rhs, second_trans, *, cache="none", copy_second=True, allow_batches=True
):
    """Factor the coupled block once per frequency for both right-hand sides."""
    plan = topo.signal_block_plan
    finite = np.isfinite(values).all(axis=1)

    has_second = second_rhs.shape[-1] != 0
    if has_second and second_rhs.ndim == 3 and second_rhs.shape[-1] == 1:
        second_rhs = second_rhs[..., 0]
        squeeze_second = True
    else:
        squeeze_second = False

    if plan is None:
        matrix = _CSCWorkspace(topo.sn, topo.signal_csc_indices, topo.signal_csc_indptr, topo.signal_core_csc_pattern)
        first_out = np.empty_like(first_rhs)
        second_out = np.empty_like(second_rhs)
        first = first_rhs
        second = second_rhs
        block_values = values
    else:
        matrix = _CSCWorkspace(plan.core.size, plan.rows, plan.indptr, topo.signal_core_csc_pattern)
        first_out = plan.prepare(values, first_rhs, first_trans)
        second_out = plan.prepare(values, second_rhs, second_trans, copy=copy_second) if has_second else second_rhs
        first = np.take(first_out, plan.core, axis=1)
        second = np.take(second_out, plan.core, axis=1) if has_second else second_out
        block_values = values[:, :plan.rows.size]

    failed = ~finite
    if matrix.shape[0]:
        # The condensed right-hand sides are private buffers and can hold the solutions.
        first_solved = first_out if plan is None else first
        second_solved = second_out if plan is None else second
        solvers = _signal_factors.take(topo, block_values) if cache == "reuse" else None
        if not allow_batches and isinstance(solvers, _SweepBatches):
            # An interleaved call can leave batched factors for the same matrix.
            # A near-null readout must retain the independent solve's rounding,
            # including when factors would otherwise be reusable.
            solvers = None
        # A block diagonal factor is equivalent to independent systems.
        blocks = solvers if isinstance(solvers, _SweepBatches) else None
        if allow_batches and solvers is None and len(block_values) > 1 and finite.all() and matrix.indices.size and _allows_frequency_batches(topo):
            try:
                blocks = _SweepBatches(matrix, block_values)
            except ValueError:
                pass
        if blocks is not None:
            blocks.refine_into(matrix, block_values, first, first_trans, first_solved)
            if has_second:
                blocks.refine_into(matrix, block_values, second, second_trans, second_solved)
            if cache == "store":
                _signal_factors.save(topo, block_values, blocks)
        else:
            saved = [] if cache == "store" and finite.all() else None
            sweep = _SweepFactorization(matrix)
            for frequency, entries in enumerate(block_values):
                if not finite[frequency]:
                    factor = None
                elif solvers is not None:
                    factor = solvers[frequency]
                    matrix.data = entries
                else:
                    factor = sweep.factor(entries)
                if saved is not None:
                    saved.append(factor)
                if factor is None:
                    failed[frequency] = True
                    continue
                first_solved[frequency] = _solve_refined(matrix, factor, first[frequency], trans=first_trans)
                if has_second:
                    second_solved[frequency] = _solve_refined(matrix, factor, second[frequency], trans=second_trans)

            if saved is not None:
                _signal_factors.save(topo, block_values, saved)
        if plan is not None:
            # Ignore failed systems during substitutions, then mark the full row invalid.
            first_solved[failed] = 0
            first_out[:, plan.core] = first_solved
            if has_second:
                second_solved[failed] = 0
                second_out[:, plan.core] = second_solved

    if plan is not None:
        plan.restore(values, first_out, first_trans)
        if has_second:
            plan.restore(values, second_out, second_trans)
    first_out[failed] = np.nan
    if has_second:
        second_out[failed] = np.nan
    if squeeze_second:
        second_out = second_out[..., None]
    return first_out, second_out


# -----------------------------------------------------------------------------
# Area C: Scatter helpers, grouped component assembly, and analytical VJP
# -----------------------------------------------------------------------------

from differometor._sparse_blocks import (
    build_scatter_plan,
    scatter_add_1d,
    scatter_add_axis1,
)


def _cos_beta(alpha_deg, n_l, n_r):
    return np.cos(np.arcsin((n_l * (1.0 / n_r)) * np.sin(alpha_deg * _DEG)))


def _cos_beta_and_derivs(alpha_deg, n_l, n_r):
    """Return cos(beta) and partials wrt (alpha_deg, n_l, n_r) for Snell's law."""
    a_rad = alpha_deg * _DEG
    sin_a = np.sin(a_rad)
    cos_a = np.cos(a_rad)
    inv_nr = 1.0 / n_r
    u = (n_l * inv_nr) * sin_a
    beta = np.arcsin(u)
    cos_b = np.cos(beta)
    dcb_du = -u / cos_b
    d_alpha = dcb_du * (n_l * inv_nr * cos_a * _DEG)
    d_nl = dcb_du * (sin_a * inv_nr)
    d_nr = dcb_du * (-u * inv_nr)
    return cos_b, d_alpha, d_nl, d_nr


# -----------------------------------------------------------------------------
# Vectorized NumPy forward evaluators (indices 0..18 aligned with FUNCTIONS)
# Inputs: tuple of used parameter arrays, each shape (B, K) or (1, K) float64.
# Outputs: complex128 shape (B_eff, K) for C=1, or (B_eff, K*C) for C>1.
# -----------------------------------------------------------------------------

def _fwd_0(args):  # force_calculation_right
    alpha, n_l, n_r = args
    return (_cos_beta(alpha, n_l, n_r) * _INV_CX).astype(np.complex128)


def _fwd_1(args):  # laser
    power, phase = args
    amp = np.sqrt(2.0 * power / EPSILON_0C)
    return amp * np.exp(1j * (phase * _DEG))


def _fwd_2(args):  # surface
    loss, r_frac, tuning, freq, n_l, n_r, alpha = args
    abs_r = (1.0 - loss) * r_frac
    abs_t = (1.0 - loss) * (1.0 - r_frac)
    t_scale = 2.0 * (tuning * _DEG) * (1.0 + freq * _INV_F0)
    ps_l = t_scale * n_l
    ps_r = t_scale * n_r
    cl = np.cos(alpha * _DEG)
    cr_t = _cos_beta(alpha, n_l, n_r)
    cr_r = _cos_beta(alpha, n_r, n_l)
    sqrt_r = np.sqrt(abs_r)
    sqrt_t = np.sqrt(abs_t)
    ps_l_cl = ps_l * cl
    r_l = -sqrt_r * np.exp(1j * ps_l_cl)
    t_val = -sqrt_t * np.exp(1j * (0.5 * np.pi + 0.5 * (ps_l_cl - ps_r * cr_t)))
    r_r = -sqrt_r * np.exp(-1j * (ps_r * cr_r))
    b_eff = max(r_l.shape[0], t_val.shape[0], r_r.shape[0]); k = r_l.shape[1]
    out = np.empty((b_eff, k, 3), dtype=np.complex128)
    out[:, :, 0] = r_l
    out[:, :, 1] = t_val
    out[:, :, 2] = r_r
    return out.reshape(b_eff, k * 3)


def _fwd_3(args):  # space
    p0, p1, p2 = args
    theta = -_TWO_PI * p0 * p1 * p2 * _INV_C
    return -np.exp(1j * theta)


def _fwd_4(args):  # space_modulation
    p0, p1, p2 = args
    w_g = _TWO_PI * p0
    w_0 = _TWO_PI * F0
    arg = w_g * p1 * p2 * (0.5 * _INV_C)
    m_g = -0.5 * (w_0 / w_g) * np.sin(arg)
    return 1j * m_g * np.exp(-1j * arg)


def _fwd_5(args):  # signal_function
    p0, p1 = args
    return p0 * np.exp(1j * (p1 * _DEG))


def _fwd_8(args):  # loss_quantum_noise
    (p0,) = args
    return (p0 * 0.5).astype(np.complex128)


def _fwd_9(args):  # space_lower
    p0, p1, p2 = args
    theta = _TWO_PI * p0 * p1 * p2 * _INV_C
    return -np.exp(1j * theta)


def _fwd_11(args):  # laser_frequency_modulation
    (p0,) = args
    return (EPSILON_0C * 0.5 / p0).astype(np.complex128)


def _fwd_12(args):  # susceptibility
    p0, p1 = args
    w = _TWO_PI * p0
    return (1.0 / (p1 * w * w)).astype(np.complex128)


def _fwd_13(args):  # force_calculation_left
    (p0,) = args
    return (-np.cos(p0 * _DEG) * _INV_CX).astype(np.complex128)


def _fwd_14(args):  # corrected_optomechanical_phase_left
    freq, tuning, r_frac, loss, n_l, alpha = args
    abs_r = (1.0 - loss) * r_frac
    cl = np.cos(alpha * _DEG)
    t_rad = tuning * _DEG
    phase = t_rad * (freq * _INV_F0) + 2.0 * t_rad * n_l * (1.0 + freq * _INV_F0) * cl
    return 1j * _Z_SCALE * cl * np.sqrt(abs_r) * np.exp(1j * phase)


def _fwd_15(args):  # corrected_optomechanical_phase_right
    freq, tuning, r_frac, loss, n_l, n_r, alpha = args
    abs_r = (1.0 - loss) * r_frac
    cr_om = _cos_beta(alpha, n_l, n_r)
    cr_rf = _cos_beta(alpha, n_r, n_l)
    t_rad = tuning * _DEG
    phase = -t_rad * (freq * _INV_F0) - 2.0 * t_rad * n_r * (1.0 + freq * _INV_F0) * cr_rf
    return -1j * _Z_SCALE * cr_om * np.sqrt(abs_r) * np.exp(1j * phase)


def _fwd_16(args):  # space_modulation_lower
    p0, p1, p2 = args
    w_g = -_TWO_PI * p0
    w_0 = _TWO_PI * F0
    arg = w_g * p1 * p2 * (0.5 * _INV_C)
    m_g = -0.5 * (w_0 / w_g) * np.sin(arg)
    return 1j * m_g * np.exp(-1j * arg)


def _fwd_17(args):  # squeezer
    p0, p1 = args
    vu = UNIT_VACUUM * 0.5
    two_s = (2.0 * _SQ_K) * p0
    ph = np.exp(1j * (2.0 * _DEG) * p1)
    u_d = (vu * np.cosh(two_s)).astype(np.complex128)
    u_o = (vu * np.sinh(two_s)) * ph
    b_eff, k = u_o.shape
    out = np.empty((b_eff, k, 4), dtype=np.complex128)
    out[:, :, 0] = u_d
    out[:, :, 1] = u_o
    out[:, :, 2] = u_d
    out[:, :, 3] = np.conj(u_o)
    return out.reshape(b_eff, k * 4)


def _fwd_18(args):  # laser_frequency_modulation_lower
    (p0,) = args
    return (-EPSILON_0C * 0.5 / p0).astype(np.complex128)


NP_CONST_VALUES = {
    6: complex(1.0),
    7: complex(UNIT_VACUUM * 0.5),
    10: complex(EPSILON_0C * 0.5),
}

NP_FWD_FUNCTIONS = (
    _fwd_0, _fwd_1, _fwd_2, _fwd_3, _fwd_4, _fwd_5, None, None, _fwd_8,
    _fwd_9, None, _fwd_11, _fwd_12, _fwd_13, _fwd_14, _fwd_15, _fwd_16,
    _fwd_17, _fwd_18,
)


# -----------------------------------------------------------------------------
# Exact analytical reverse-mode VJPs (indices 0..18 aligned with FUNCTIONS)
# Inputs: args tuple of (B, K) or (1, K) float64; gy (B, K) for C=1 or (B, K, C).
# Output: tuple of `used` float64 arrays, each shape (B, K).
# -----------------------------------------------------------------------------

def _vjp_0(args, gy):  # force_calculation_right
    alpha, n_l, n_r = args
    _, d_a, d_nl, d_nr = _cos_beta_and_derivs(alpha, n_l, n_r)
    g = np.real(gy) * _INV_CX
    return (g * d_a, g * d_nl, g * d_nr)


def _vjp_1(args, gy):  # laser
    power, phase = args
    amp = np.sqrt(2.0 * power / EPSILON_0C)
    e = np.exp(1j * (phase * _DEG))
    y = amp * e
    inv_amp = np.where(amp > 0.0, 1.0 / np.where(amp > 0.0, amp, 1.0), 0.0)
    return (
        np.real(gy * e) * (inv_amp / EPSILON_0C),
        np.real(gy * (1j * _DEG * y)),
    )


def _vjp_2(args, gy):  # surface
    loss, r_frac, tuning, freq, n_l, n_r, alpha = args
    abs_r = (1.0 - loss) * r_frac
    abs_t = (1.0 - loss) * (1.0 - r_frac)
    f_fac = 1.0 + freq * _INV_F0
    t_rad = tuning * _DEG
    t_scale = 2.0 * t_rad * f_fac
    ps_l = t_scale * n_l
    ps_r = t_scale * n_r

    a_rad = alpha * _DEG
    cl = np.cos(a_rad)
    dcl_da = -np.sin(a_rad) * _DEG
    cr_t, dcr_t_da, dcr_t_dnl, dcr_t_dnr = _cos_beta_and_derivs(alpha, n_l, n_r)
    cr_r, dcr_r_da, dcr_r_dnr, dcr_r_dnl = _cos_beta_and_derivs(alpha, n_r, n_l)

    sqrt_r = np.sqrt(abs_r)
    sqrt_t = np.sqrt(abs_t)
    ps_l_cl = ps_l * cl
    e_rl = np.exp(1j * ps_l_cl)
    e_t = np.exp(1j * (0.5 * np.pi + 0.5 * (ps_l_cl - ps_r * cr_t)))
    e_rr = np.exp(-1j * (ps_r * cr_r))
    r_l = -sqrt_r * e_rl
    t_val = -sqrt_t * e_t
    r_r = -sqrt_r * e_rr

    g_rl, g_t, g_rr = gy[:, :, 0], gy[:, :, 1], gy[:, :, 2]

    inv_2sr = np.where(sqrt_r > 0.0, -0.5 / np.where(sqrt_r > 0.0, sqrt_r, 1.0), 0.0)
    inv_2st = np.where(sqrt_t > 0.0, -0.5 / np.where(sqrt_t > 0.0, sqrt_t, 1.0), 0.0)
    g_abs_r = np.real(g_rl * e_rl + g_rr * e_rr) * inv_2sr
    g_abs_t = np.real(g_t * e_t) * inv_2st

    gp0 = -g_abs_r * r_frac - g_abs_t * (1.0 - r_frac)
    gp1 = (g_abs_r - g_abs_t) * (1.0 - loss)

    g_phi_rl = np.real(g_rl * (1j * r_l))
    g_phi_t = np.real(g_t * (1j * t_val))
    g_phi_rr = np.real(g_rr * (1j * r_r))

    g_psl_cl = g_phi_rl + 0.5 * g_phi_t
    g_psr_crt = -0.5 * g_phi_t
    g_psr_crr = -g_phi_rr

    g_ps_l = g_psl_cl * cl
    g_ps_r = g_psr_crt * cr_t + g_psr_crr * cr_r
    g_cl = g_psl_cl * ps_l
    g_cr_t = g_psr_crt * ps_r
    g_cr_r = g_psr_crr * ps_r

    g_t_scale = g_ps_l * n_l + g_ps_r * n_r
    gp2 = g_t_scale * (2.0 * _DEG * f_fac)
    gp3 = g_t_scale * (2.0 * t_rad * _INV_F0)
    gp4 = g_ps_l * t_scale + g_cr_t * dcr_t_dnl + g_cr_r * dcr_r_dnl
    gp5 = g_ps_r * t_scale + g_cr_t * dcr_t_dnr + g_cr_r * dcr_r_dnr
    gp6 = g_cl * dcl_da + g_cr_t * dcr_t_da + g_cr_r * dcr_r_da
    return (gp0, gp1, gp2, gp3, gp4, gp5, gp6)


def _vjp_space_signed(args, gy, sign):
    p0, p1, p2 = args
    c = sign * _TWO_PI * _INV_C
    theta = c * p0 * p1 * p2
    y = -np.exp(1j * theta)
    g_theta = np.real(gy * (1j * y)) * c
    return (
        g_theta * (p1 * p2),
        g_theta * (p0 * p2),
        g_theta * (p0 * p1),
    )


def _vjp_3(args, gy):  # space
    return _vjp_space_signed(args, gy, -1.0)


def _vjp_9(args, gy):  # space_lower
    return _vjp_space_signed(args, gy, 1.0)


def _vjp_space_mod_signed(args, gy, sign):
    p0, p1, p2 = args
    w_g = (sign * _TWO_PI) * p0
    w_0 = _TWO_PI * F0
    half_inv_c = 0.5 * _INV_C
    ln_factor = p1 * p2 * half_inv_c
    arg = w_g * ln_factor
    sin_a = np.sin(arg)
    cos_a = np.cos(arg)
    inv_wg = 1.0 / w_g
    coef = -0.5 * w_0 * inv_wg
    m_g = coef * sin_a
    e = np.exp(-1j * arg)
    y = 1j * m_g * e

    g_mg = np.real(gy * (1j * e))
    g_arg = np.real(gy * (-1j * y)) + g_mg * (coef * cos_a)
    g_wg = g_mg * (-m_g * inv_wg) + g_arg * ln_factor

    return (
        g_wg * (sign * _TWO_PI),
        g_arg * (w_g * p2 * half_inv_c),
        g_arg * (w_g * p1 * half_inv_c),
    )


def _vjp_4(args, gy):  # space_modulation
    return _vjp_space_mod_signed(args, gy, 1.0)


def _vjp_16(args, gy):  # space_modulation_lower
    return _vjp_space_mod_signed(args, gy, -1.0)


def _vjp_5(args, gy):  # signal_function
    p0, p1 = args
    e = np.exp(1j * (p1 * _DEG))
    return (
        np.real(gy * e),
        np.real(gy * (1j * _DEG * p0 * e)),
    )


def _vjp_zero(args, gy):  # constants: dummy, vacuum_quantum_noise, laser_amplitude_modulation
    return ()


def _vjp_8(args, gy):  # loss_quantum_noise
    return (0.5 * np.real(gy),)


def _vjp_11(args, gy):  # laser_frequency_modulation
    (p0,) = args
    return (np.real(gy) * (-EPSILON_0C * 0.5 / (p0 * p0)),)


def _vjp_18(args, gy):  # laser_frequency_modulation_lower
    (p0,) = args
    return (np.real(gy) * (EPSILON_0C * 0.5 / (p0 * p0)),)


def _vjp_12(args, gy):  # susceptibility
    p0, p1 = args
    w = _TWO_PI * p0
    y = 1.0 / (p1 * w * w)
    g = np.real(gy) * y
    return (-2.0 * g / p0, -g / p1)


def _vjp_13(args, gy):  # force_calculation_left
    (p0,) = args
    return (np.real(gy) * (np.sin(p0 * _DEG) * _DEG * _INV_CX),)


def _vjp_14(args, gy):  # corrected_optomechanical_phase_left
    freq, tuning, r_frac, loss, n_l, alpha = args
    abs_r = (1.0 - loss) * r_frac
    sqrt_r = np.sqrt(abs_r)
    a_rad = alpha * _DEG
    cl = np.cos(a_rad)
    dcl_da = -np.sin(a_rad) * _DEG
    t_rad = tuning * _DEG
    f_fac = 1.0 + freq * _INV_F0
    phase = t_rad * (freq * _INV_F0) + 2.0 * t_rad * n_l * f_fac * cl
    e = np.exp(1j * phase)
    y = (1j * _Z_SCALE) * cl * sqrt_r * e

    g_cl = np.real(gy * ((1j * _Z_SCALE) * sqrt_r * e))
    inv_2sr = np.where(sqrt_r > 0.0, 0.5 / np.where(sqrt_r > 0.0, sqrt_r, 1.0), 0.0)
    g_abs_r = np.real(gy * ((1j * _Z_SCALE) * cl * inv_2sr * e))
    g_phase = np.real(gy * (1j * y))

    return (
        g_phase * (t_rad * _INV_F0 * (1.0 + 2.0 * n_l * cl)),
        g_phase * (_DEG * (freq * _INV_F0 + 2.0 * n_l * f_fac * cl)),
        g_abs_r * (1.0 - loss),
        -g_abs_r * r_frac,
        g_phase * (2.0 * t_rad * f_fac * cl),
        (g_cl + g_phase * (2.0 * t_rad * n_l * f_fac)) * dcl_da,
    )


def _vjp_15(args, gy):  # corrected_optomechanical_phase_right
    freq, tuning, r_frac, loss, n_l, n_r, alpha = args
    abs_r = (1.0 - loss) * r_frac
    sqrt_r = np.sqrt(abs_r)
    cr_om, dom_da, dom_dnl, dom_dnr = _cos_beta_and_derivs(alpha, n_l, n_r)
    cr_rf, drf_da, drf_dnr, drf_dnl = _cos_beta_and_derivs(alpha, n_r, n_l)
    t_rad = tuning * _DEG
    f_fac = 1.0 + freq * _INV_F0
    phase = -t_rad * (freq * _INV_F0) - 2.0 * t_rad * n_r * f_fac * cr_rf
    e = np.exp(1j * phase)
    y = (-1j * _Z_SCALE) * cr_om * sqrt_r * e

    g_cr_om = np.real(gy * ((-1j * _Z_SCALE) * sqrt_r * e))
    inv_2sr = np.where(sqrt_r > 0.0, 0.5 / np.where(sqrt_r > 0.0, sqrt_r, 1.0), 0.0)
    g_abs_r = np.real(gy * ((-1j * _Z_SCALE) * cr_om * inv_2sr * e))
    g_phase = np.real(gy * (1j * y))
    g_cr_rf = g_phase * (-2.0 * t_rad * n_r * f_fac)

    return (
        g_phase * (-t_rad * _INV_F0 * (1.0 + 2.0 * n_r * cr_rf)),
        g_phase * (-_DEG * (freq * _INV_F0 + 2.0 * n_r * f_fac * cr_rf)),
        g_abs_r * (1.0 - loss),
        -g_abs_r * r_frac,
        g_cr_om * dom_dnl + g_cr_rf * drf_dnl,
        g_cr_om * dom_dnr + g_cr_rf * drf_dnr + g_phase * (-2.0 * t_rad * f_fac * cr_rf),
        g_cr_om * dom_da + g_cr_rf * drf_da,
    )


def _vjp_17(args, gy):  # squeezer
    p0, p1 = args
    vu = UNIT_VACUUM * 0.5
    two_k = 2.0 * _SQ_K
    two_s = two_k * p0
    sh = np.sinh(two_s)
    ch = np.cosh(two_s)
    ph = np.exp(1j * (2.0 * _DEG) * p1)
    u_o = (vu * sh) * ph
    l_o = np.conj(u_o)

    g_ud, g_uo, g_ld, g_lo = gy[:, :, 0], gy[:, :, 1], gy[:, :, 2], gy[:, :, 3]
    gp0 = (
        np.real(g_ud + g_ld) * (two_k * vu * sh)
        + np.real(g_uo * ph + g_lo * np.conj(ph)) * (two_k * vu * ch)
    )
    gp1 = np.real(g_uo * (2j * _DEG * u_o) + g_lo * (-2j * _DEG * l_o))
    return (gp0, gp1)


NP_VJP_FUNCTIONS = (
    _vjp_0, _vjp_1, _vjp_2, _vjp_3, _vjp_4, _vjp_5, _vjp_zero, _vjp_zero, _vjp_8,
    _vjp_9, _vjp_zero, _vjp_11, _vjp_12, _vjp_13, _vjp_14, _vjp_15, _vjp_16,
    _vjp_17, _vjp_18,
)


# -----------------------------------------------------------------------------
# Complex fallback outside the real domain of sqrt / arcsin
# -----------------------------------------------------------------------------
# The original keeps `parameters` complex, so a negative reflectivity/transmissivity/laser power
# (e.g. R > 1) or total internal reflection (|n_l / n_r * sin(alpha)| > 1) gives finite complex
# couplings there, where the real-valued evaluators above give NaN. The functions below are only
# used when an input is outside that domain, so in-domain results stay bit-identical. They follow
# XLA's conventions: the branch cut ignores the sign of a zero imaginary part (arguments are cast
# with +0j), and d arcsin(u)/du = rsqrt(1 - u^2), which is what JAX differentiates.

def _snell_u(alpha_deg, n_l, n_r):
    return (n_l * (1.0 / n_r)) * np.sin(alpha_deg * _DEG)


def _beyond_unit(alpha_deg, n_a, n_b):
    return bool((np.abs(_snell_u(alpha_deg, n_a, n_b)) > 1.0).any() or (np.abs(_snell_u(alpha_deg, n_b, n_a)) > 1.0).any())


def _csqrt(x):
    return np.sqrt(np.asarray(x, dtype=np.float64).astype(np.complex128))


def _cos_beta_c(alpha_deg, n_l, n_r):
    return np.cos(np.arcsin(_snell_u(alpha_deg, n_l, n_r).astype(np.complex128)))


def _cos_beta_and_derivs_c(alpha_deg, n_l, n_r):
    a_rad = alpha_deg * _DEG
    sin_a = np.sin(a_rad)
    cos_a = np.cos(a_rad)
    inv_nr = 1.0 / n_r
    u = (n_l * inv_nr) * sin_a
    uc = u.astype(np.complex128)
    beta = np.arcsin(uc)
    cos_b = np.cos(beta)
    dcb_du = -np.sin(beta) / _csqrt(1.0 - u * u)
    d_alpha = dcb_du * (n_l * inv_nr * cos_a * _DEG)
    d_nl = dcb_du * (sin_a * inv_nr)
    d_nr = dcb_du * (-u * inv_nr)
    return cos_b, d_alpha, d_nl, d_nr


def _inv_2sqrt_c(x, sqrt_x, scale):
    nonzero = x != 0.0
    return np.where(nonzero, scale / np.where(nonzero, sqrt_x, 1.0), 0.0)


def _out_0(args):
    alpha, n_l, n_r = args
    return bool((np.abs(_snell_u(alpha, n_l, n_r)) > 1.0).any())


def _out_1(args):
    return bool((args[0] < 0.0).any())


def _out_2(args):
    loss, r_frac, _tuning, _freq, n_l, n_r, alpha = args
    if ((1.0 - loss) * r_frac < 0.0).any() or ((1.0 - loss) * (1.0 - r_frac) < 0.0).any():
        return True
    return _beyond_unit(alpha, n_l, n_r)


def _out_14(args):
    _freq, _tuning, r_frac, loss, _n_l, _alpha = args
    return bool(((1.0 - loss) * r_frac < 0.0).any())


def _out_15(args):
    _freq, _tuning, r_frac, loss, n_l, n_r, alpha = args
    if ((1.0 - loss) * r_frac < 0.0).any():
        return True
    return _beyond_unit(alpha, n_l, n_r)


def _fwd_0c(args):
    alpha, n_l, n_r = args
    return _cos_beta_c(alpha, n_l, n_r) * _INV_CX


def _fwd_1c(args):
    power, phase = args
    return _csqrt(2.0 * power / EPSILON_0C) * np.exp(1j * (phase * _DEG))


def _fwd_2c(args):
    loss, r_frac, tuning, freq, n_l, n_r, alpha = args
    abs_r = (1.0 - loss) * r_frac
    abs_t = (1.0 - loss) * (1.0 - r_frac)
    t_scale = 2.0 * (tuning * _DEG) * (1.0 + freq * _INV_F0)
    ps_l = t_scale * n_l
    ps_r = t_scale * n_r
    cl = np.cos(alpha * _DEG)
    cr_t = _cos_beta_c(alpha, n_l, n_r)
    cr_r = _cos_beta_c(alpha, n_r, n_l)
    sqrt_r = _csqrt(abs_r)
    sqrt_t = _csqrt(abs_t)
    ps_l_cl = ps_l * cl
    r_l = -sqrt_r * np.exp(1j * ps_l_cl)
    t_val = -sqrt_t * np.exp(1j * (0.5 * np.pi + 0.5 * (ps_l_cl - ps_r * cr_t)))
    r_r = -sqrt_r * np.exp(-1j * (ps_r * cr_r))
    r_l, t_val, r_r = np.broadcast_arrays(r_l, t_val, r_r)
    b_eff, k = r_l.shape
    out = np.empty((b_eff, k, 3), dtype=np.complex128)
    out[:, :, 0] = r_l
    out[:, :, 1] = t_val
    out[:, :, 2] = r_r
    return out.reshape(b_eff, k * 3)


def _fwd_14c(args):
    freq, tuning, r_frac, loss, n_l, alpha = args
    abs_r = (1.0 - loss) * r_frac
    cl = np.cos(alpha * _DEG)
    t_rad = tuning * _DEG
    phase = t_rad * (freq * _INV_F0) + 2.0 * t_rad * n_l * (1.0 + freq * _INV_F0) * cl
    return 1j * _Z_SCALE * cl * _csqrt(abs_r) * np.exp(1j * phase)


def _fwd_15c(args):
    freq, tuning, r_frac, loss, n_l, n_r, alpha = args
    abs_r = (1.0 - loss) * r_frac
    cr_om = _cos_beta_c(alpha, n_l, n_r)
    cr_rf = _cos_beta_c(alpha, n_r, n_l)
    t_rad = tuning * _DEG
    phase = -t_rad * (freq * _INV_F0) - 2.0 * t_rad * n_r * (1.0 + freq * _INV_F0) * cr_rf
    return -1j * _Z_SCALE * cr_om * _csqrt(abs_r) * np.exp(1j * phase)


# The complex VJPs are the real ones with np.real moved past every complex intermediate.

def _vjp_0c(args, gy):
    alpha, n_l, n_r = args
    _, d_a, d_nl, d_nr = _cos_beta_and_derivs_c(alpha, n_l, n_r)
    g = gy * _INV_CX
    return (np.real(g * d_a), np.real(g * d_nl), np.real(g * d_nr))


def _vjp_1c(args, gy):
    power, phase = args
    x = 2.0 * power / EPSILON_0C
    amp = _csqrt(x)
    e = np.exp(1j * (phase * _DEG))
    y = amp * e
    inv_amp = _inv_2sqrt_c(x, amp, 1.0)
    return (
        np.real(gy * e * inv_amp) / EPSILON_0C,
        np.real(gy * (1j * _DEG * y)),
    )


def _vjp_2c(args, gy):
    loss, r_frac, tuning, freq, n_l, n_r, alpha = args
    abs_r = (1.0 - loss) * r_frac
    abs_t = (1.0 - loss) * (1.0 - r_frac)
    f_fac = 1.0 + freq * _INV_F0
    t_rad = tuning * _DEG
    t_scale = 2.0 * t_rad * f_fac
    ps_l = t_scale * n_l
    ps_r = t_scale * n_r

    a_rad = alpha * _DEG
    cl = np.cos(a_rad)
    dcl_da = -np.sin(a_rad) * _DEG
    cr_t, dcr_t_da, dcr_t_dnl, dcr_t_dnr = _cos_beta_and_derivs_c(alpha, n_l, n_r)
    cr_r, dcr_r_da, dcr_r_dnr, dcr_r_dnl = _cos_beta_and_derivs_c(alpha, n_r, n_l)

    sqrt_r = _csqrt(abs_r)
    sqrt_t = _csqrt(abs_t)
    ps_l_cl = ps_l * cl
    e_rl = np.exp(1j * ps_l_cl)
    e_t = np.exp(1j * (0.5 * np.pi + 0.5 * (ps_l_cl - ps_r * cr_t)))
    e_rr = np.exp(-1j * (ps_r * cr_r))
    r_l = -sqrt_r * e_rl
    t_val = -sqrt_t * e_t
    r_r = -sqrt_r * e_rr

    g_rl, g_t, g_rr = gy[:, :, 0], gy[:, :, 1], gy[:, :, 2]

    g_abs_r = np.real((g_rl * e_rl + g_rr * e_rr) * _inv_2sqrt_c(abs_r, sqrt_r, -0.5))
    g_abs_t = np.real(g_t * e_t * _inv_2sqrt_c(abs_t, sqrt_t, -0.5))

    gp0 = -g_abs_r * r_frac - g_abs_t * (1.0 - r_frac)
    gp1 = (g_abs_r - g_abs_t) * (1.0 - loss)

    g_phi_t = g_t * (1j * t_val)  # w.r.t. the (complex) transmission phase
    g_psl_cl = np.real(g_rl * (1j * r_l)) + 0.5 * np.real(g_phi_t)
    g_psr_crt = -0.5 * g_phi_t
    g_psr_crr = -(g_rr * (1j * r_r))

    g_ps_l = g_psl_cl * cl
    g_ps_r = np.real(g_psr_crt * cr_t + g_psr_crr * cr_r)
    g_cl = g_psl_cl * ps_l
    g_cr_t = g_psr_crt * ps_r
    g_cr_r = g_psr_crr * ps_r

    g_t_scale = g_ps_l * n_l + g_ps_r * n_r
    gp2 = g_t_scale * (2.0 * _DEG * f_fac)
    gp3 = g_t_scale * (2.0 * t_rad * _INV_F0)
    gp4 = g_ps_l * t_scale + np.real(g_cr_t * dcr_t_dnl + g_cr_r * dcr_r_dnl)
    gp5 = g_ps_r * t_scale + np.real(g_cr_t * dcr_t_dnr + g_cr_r * dcr_r_dnr)
    gp6 = g_cl * dcl_da + np.real(g_cr_t * dcr_t_da + g_cr_r * dcr_r_da)
    return (gp0, gp1, gp2, gp3, gp4, gp5, gp6)


def _vjp_14c(args, gy):
    freq, tuning, r_frac, loss, n_l, alpha = args
    abs_r = (1.0 - loss) * r_frac
    sqrt_r = _csqrt(abs_r)
    a_rad = alpha * _DEG
    cl = np.cos(a_rad)
    dcl_da = -np.sin(a_rad) * _DEG
    t_rad = tuning * _DEG
    f_fac = 1.0 + freq * _INV_F0
    phase = t_rad * (freq * _INV_F0) + 2.0 * t_rad * n_l * f_fac * cl
    e = np.exp(1j * phase)
    y = (1j * _Z_SCALE) * cl * sqrt_r * e

    g_cl = np.real(gy * ((1j * _Z_SCALE) * sqrt_r * e))
    g_abs_r = np.real(gy * ((1j * _Z_SCALE) * cl * _inv_2sqrt_c(abs_r, sqrt_r, 0.5) * e))
    g_phase = np.real(gy * (1j * y))

    return (
        g_phase * (t_rad * _INV_F0 * (1.0 + 2.0 * n_l * cl)),
        g_phase * (_DEG * (freq * _INV_F0 + 2.0 * n_l * f_fac * cl)),
        g_abs_r * (1.0 - loss),
        -g_abs_r * r_frac,
        g_phase * (2.0 * t_rad * f_fac * cl),
        (g_cl + g_phase * (2.0 * t_rad * n_l * f_fac)) * dcl_da,
    )


def _vjp_15c(args, gy):
    freq, tuning, r_frac, loss, n_l, n_r, alpha = args
    abs_r = (1.0 - loss) * r_frac
    sqrt_r = _csqrt(abs_r)
    cr_om, dom_da, dom_dnl, dom_dnr = _cos_beta_and_derivs_c(alpha, n_l, n_r)
    cr_rf, drf_da, drf_dnr, drf_dnl = _cos_beta_and_derivs_c(alpha, n_r, n_l)
    t_rad = tuning * _DEG
    f_fac = 1.0 + freq * _INV_F0
    phase = -t_rad * (freq * _INV_F0) - 2.0 * t_rad * n_r * f_fac * cr_rf
    e = np.exp(1j * phase)
    y = (-1j * _Z_SCALE) * cr_om * sqrt_r * e

    g_cr_om = gy * ((-1j * _Z_SCALE) * sqrt_r * e)
    g_abs_r = np.real(gy * ((-1j * _Z_SCALE) * cr_om * _inv_2sqrt_c(abs_r, sqrt_r, 0.5) * e))
    g_phase = gy * (1j * y)  # the phase is complex once cr_rf is
    g_cr_rf = g_phase * (-2.0 * t_rad * n_r * f_fac)

    return (
        np.real(g_phase * (-t_rad * _INV_F0 * (1.0 + 2.0 * n_r * cr_rf))),
        np.real(g_phase * (-_DEG * (freq * _INV_F0 + 2.0 * n_r * f_fac * cr_rf))),
        g_abs_r * (1.0 - loss),
        -g_abs_r * r_frac,
        np.real(g_cr_om * dom_dnl + g_cr_rf * drf_dnl),
        np.real(g_cr_om * dom_dnr + g_cr_rf * drf_dnr + g_phase * (-2.0 * t_rad * f_fac * cr_rf)),
        np.real(g_cr_om * dom_da + g_cr_rf * drf_da),
    )


def _with_complex_fallback(real_fn, complex_fn, outside):
    def evaluate(args, *cotangent):
        if outside(args):
            return complex_fn(args, *cotangent)
        return real_fn(args, *cotangent)

    evaluate.__name__ = real_fn.__name__
    return evaluate


_COMPLEX_FALLBACKS = {
    0: (_out_0, _fwd_0c, _vjp_0c),
    1: (_out_1, _fwd_1c, _vjp_1c),
    2: (_out_2, _fwd_2c, _vjp_2c),
    14: (_out_14, _fwd_14c, _vjp_14c),
    15: (_out_15, _fwd_15c, _vjp_15c),
}
NP_FWD_FUNCTIONS = tuple(
    _with_complex_fallback(fn, _COMPLEX_FALLBACKS[i][1], _COMPLEX_FALLBACKS[i][0]) if i in _COMPLEX_FALLBACKS else fn
    for i, fn in enumerate(NP_FWD_FUNCTIONS)
)
NP_VJP_FUNCTIONS = tuple(
    _with_complex_fallback(fn, _COMPLEX_FALLBACKS[i][2], _COMPLEX_FALLBACKS[i][0]) if i in _COMPLEX_FALLBACKS else fn
    for i, fn in enumerate(NP_VJP_FUNCTIONS)
)


# -----------------------------------------------------------------------------
# Precomputed grouped operation descriptor & parameter-gradient scatter plans
# -----------------------------------------------------------------------------

def _build_col_reduce_plan(targets: np.ndarray):
    """Build an O(1)-dispatch accumulation plan for 1-D target indices (exact order).

    Modes:
      0: all entries map to a single target -> (0, int_target, None, None)
      1: all entries map to distinct targets -> (1, targets_intp, None, None)
      2: duplicate targets -> (2, uniq_targets, stable_order, reduceat_starts)
    """
    idx = np.asarray(targets, dtype=np.intp)
    n = idx.size
    if n == 0:
        return None
    if n == 1:
        return (0, int(idx[0]), None, None)
    if np.all(idx == idx[0]):
        return (0, int(idx[0]), None, None)
    if (idx[1:] > idx[:-1]).all():
        return (1, idx, None, None)
    order = np.argsort(idx, kind="stable")
    ordered = idx[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = ordered[1:] != ordered[:-1]
    if first.all():
        return (1, idx, None, None)
    starts = np.flatnonzero(first)
    uniq = ordered[starts]
    return (2, uniq, order, starts)


def build_op_group(
    f_idx: int,
    inp_sub: np.ndarray,
    out_row: np.ndarray,
    out_col: np.ndarray,
    flat_pos: np.ndarray,
    n_out_cols: int,
    *,
    bwd_flat_pos: np.ndarray | None = None,
    sweep_rows: np.ndarray | None = None,
    batch_sweep_set: set[int] | None = None,
):
    """Construct an `_OpGroup` with precomputed forward/backward gather & scatter plans."""
    k = inp_sub.shape[0]
    flat_pos_p = np.asarray(flat_pos, dtype=np.intp)
    const_val = NP_CONST_VALUES.get(f_idx)
    fwd_fn = NP_FWD_FUNCTIONS[f_idx]
    vjp_fn = NP_VJP_FUNCTIONS[f_idx]

    if const_val is not None:
        return _OpGroup(
            f_idx=f_idx,
            fwd_fn=None,
            vjp_fn=_vjp_zero,
            const_val=const_val,
            k=k,
            n_out_cols=n_out_cols,
            inp_cols=(),
            col_swept=(),
            flat_pos=flat_pos_p,
            fwd_out_idx=None,
            gy_plan=(),
            gy_full_first=False,
            col_scatter=(),
        )

    inp_cols = tuple(np.ascontiguousarray(inp_sub[:, j], dtype=np.intp) for j in range(inp_sub.shape[1]))
    if batch_sweep_set:
        if len(batch_sweep_set) == 1:
            sw0 = next(iter(batch_sweep_set))
            col_swept = tuple(bool((col == sw0).any()) for col in inp_cols)
        else:
            sw_arr = np.fromiter(batch_sweep_set, dtype=np.intp, count=len(batch_sweep_set))
            col_swept = tuple(bool(np.isin(col, sw_arr).any()) for col in inp_cols)
    else:
        col_swept = (False,) * len(inp_cols)

    flat_out = out_row.astype(np.intp) * n_out_cols + out_col.astype(np.intp)
    m_total = k * n_out_cols
    is_identity_out = (flat_out.size == m_total) and np.array_equal(flat_out, np.arange(m_total, dtype=np.intp))
    fwd_out_idx = None if is_identity_out else flat_out

    bwd_fp = flat_pos_p if bwd_flat_pos is None else np.asarray(bwd_flat_pos, dtype=np.intp)
    if is_identity_out:
        gy_plan = ((bwd_fp, None),) if m_total > 0 else ()
        gy_full_first = m_total > 0
    else:
        raw_gy = build_scatter_plan(flat_out)
        gy_batches = []
        arange_m = np.arange(m_total, dtype=np.intp)
        for pos, dst in raw_gy:
            fp_r = bwd_fp if pos is None else bwd_fp[pos]
            dst_r = None if (dst.size == m_total and np.array_equal(dst, arange_m)) else dst
            gy_batches.append((fp_r, dst_r))
        gy_plan = tuple(gy_batches)
        gy_full_first = bool(gy_plan) and (gy_plan[0][1] is None)

    col_scatter = []
    if sweep_rows is None:
        for col in inp_cols:
            col_scatter.append((0, _build_col_reduce_plan(col)))
    else:
        for col in inp_cols:
            slots = sweep_rows[col]
            is_sw = slots >= 0
            if not is_sw.any():
                col_scatter.append((1, _build_col_reduce_plan(col)))
            elif is_sw.all():
                col_scatter.append((2, _build_col_reduce_plan(slots)))
            else:
                sw_pos = np.flatnonzero(is_sw)
                ns_pos = np.flatnonzero(~is_sw)
                col_scatter.append((
                    3,
                    (sw_pos, _build_col_reduce_plan(slots[sw_pos])),
                    (ns_pos, _build_col_reduce_plan(col[ns_pos])),
                ))

    return _OpGroup(
        f_idx=f_idx,
        fwd_fn=fwd_fn,
        vjp_fn=vjp_fn,
        const_val=None,
        k=k,
        n_out_cols=n_out_cols,
        inp_cols=inp_cols,
        col_swept=col_swept,
        flat_pos=flat_pos_p,
        fwd_out_idx=fwd_out_idx,
        gy_plan=gy_plan,
        gy_full_first=gy_full_first,
        col_scatter=tuple(col_scatter),
    )


class _OpGroup:
    __slots__ = (
        "f_idx", "fwd_fn", "vjp_fn", "const_val", "k", "n_out_cols",
        "inp_cols", "col_swept", "flat_pos", "fwd_out_idx",
        "gy_plan", "gy_full_first", "col_scatter",
    )

    def __init__(
        self, *, f_idx, fwd_fn, vjp_fn, const_val, k, n_out_cols,
        inp_cols, col_swept, flat_pos, fwd_out_idx, gy_plan, gy_full_first, col_scatter,
    ):
        self.f_idx = f_idx
        self.fwd_fn = fwd_fn
        self.vjp_fn = vjp_fn
        self.const_val = const_val
        self.k = k
        self.n_out_cols = n_out_cols
        self.inp_cols = inp_cols
        self.col_swept = col_swept
        self.flat_pos = flat_pos
        self.fwd_out_idx = fwd_out_idx
        self.gy_plan = gy_plan
        self.gy_full_first = gy_full_first
        self.col_scatter = col_scatter


# -----------------------------------------------------------------------------
# Host-side parameter linking (eliminates XLA vmap/scatter compile cliff)
# -----------------------------------------------------------------------------

def _link_sqrt_fwd(x):
    return np.sqrt(x)


def _link_sqrt_deriv(x):
    return 0.5 / np.sqrt(x)


def _link_square_fwd(x):
    return x * x


def _link_square_deriv(x):
    return 2.0 * x


def _link_cos_deriv(x):
    return -np.sin(x)


def _link_log_deriv(x):
    return 1.0 / x


def _link_neg_deriv(x):
    return np.full_like(x, -1.0)


def _link_abs_deriv(x):
    return np.where(x >= 0.0, 1.0, -1.0)


_HOST_LINK_FUNCS = {
    jax.numpy.sqrt: (_link_sqrt_fwd, _link_sqrt_deriv),
    np.sqrt: (_link_sqrt_fwd, _link_sqrt_deriv),
    jax.numpy.square: (_link_square_fwd, _link_square_deriv),
    np.square: (_link_square_fwd, _link_square_deriv),
    jax.numpy.exp: (np.exp, np.exp),
    np.exp: (np.exp, np.exp),
    jax.numpy.sin: (np.sin, np.cos),
    np.sin: (np.sin, np.cos),
    jax.numpy.cos: (np.cos, _link_cos_deriv),
    np.cos: (np.cos, _link_cos_deriv),
    jax.numpy.log: (np.log, _link_log_deriv),
    np.log: (np.log, _link_log_deriv),
    jax.numpy.negative: (np.negative, _link_neg_deriv),
    np.negative: (np.negative, _link_neg_deriv),
    jax.numpy.abs: (np.abs, _link_abs_deriv),
    np.abs: (np.abs, _link_abs_deriv),
}


def resolve_host_link(fn):
    """Return ``(fwd_fn, deriv_fn)`` for known NumPy/JAX linking functions, else None."""
    try:
        return _HOST_LINK_FUNCS.get(fn)
    except TypeError:
        return None


def apply_host_links_fwd(bound: np.ndarray, topo: SparseTopology) -> np.ndarray:
    """Apply pre-resolved parameter links on the host, matching simulate.py semantics."""
    out = np.array(bound, copy=True)
    link_input = out[0].copy()
    for fwd_fn, _deriv_fn, src_idx, dst_idx in topo.host_linked_groups:
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = np.real(fwd_fn(link_input[src_idx])).astype(np.float64, copy=False)
        out[0, dst_idx] = vals
    return out


def apply_host_links_bwd(g_base: np.ndarray, bound_pre: np.ndarray, topo: SparseTopology) -> None:
    """In-place exact reverse-mode VJP of ``apply_host_links_fwd``."""
    link_input = np.asarray(bound_pre, dtype=np.float64)[0]
    g_out_row = g_base[0]
    g_link_in = np.zeros_like(g_out_row)
    for _fwd_fn, deriv_fn, src_idx, dst_idx in reversed(topo.host_linked_groups):
        g_dst = g_out_row[dst_idx].copy()
        g_out_row[dst_idx] = 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            g_src = g_dst * deriv_fn(link_input[src_idx])
        np.add.at(g_link_in, src_idx, g_src)
    g_out_row += g_link_in


# -----------------------------------------------------------------------------
# Sparse Topology Pytree Container (0 dynamic JAX leaves -> near-zero XLA graph)
# -----------------------------------------------------------------------------

class SparseTopology:
    """Immutable host-side sparse topology passed as static JAX pytree aux_data."""

    __slots__ = (
        "cn", "sn", "nnz_c", "nnz_s", "nnz_n", "num_detectors", "readout_ports",
        "opt_param_idx", "opt_val_idx",
        "car_ch_idx", "sig_ch_idx",
        "qhd_param_idx", "qhd_placing_idx", "qhd_scatter_plan",
        "host_linked_groups",
        "carrier_base_aug", "carrier_csc_rows", "carrier_csc_cols",
        "carrier_csc_indices", "carrier_csc_indptr", "carrier_csc_pattern", "carrier_groups",
        "signal_base_aug", "signal_csc_rows", "signal_csc_cols",
        "signal_csc_indices", "signal_csc_indptr", "signal_core_csc_pattern",
        "signal_block_plan",
        "signal_groups_static", "signal_groups_sweep",
        "signal_carrier_indices", "signal_carrier_flat_pos", "signal_carrier_scatter_plan",
        "signal_static_conj_idx", "signal_sweep_conj_idx",
        "bwd_sig_n_act", "bwd_sig_n_act_a",
        "bwd_sig_a_rows", "bwd_sig_a_cols", "bwd_sig_rhs_rows",
        "bwd_sig_conj_idx", "bwd_sig_car_pos", "bwd_sig_static_pos",
        "noise_base_vals", "noise_rows", "noise_cols",
        "noise_groups_static", "noise_groups_sweep",
        "noise_grad_order", "noise_grad_starts", "noise_grad_indices",
        "_uid",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._uid = id(self)

    def __hash__(self):
        return hash(self._uid)

    def __eq__(self, other):
        return self is other


jax.tree_util.register_pytree_node(
    SparseTopology,
    lambda x: ((), x),
    lambda aux, _: aux,
)


def _apply_np_groups(buf: np.ndarray, params: np.ndarray, groups: tuple) -> None:
    """In-place vectorized evaluation of grouped NumPy component functions."""
    b = params.shape[0]
    for g in groups:
        if g.const_val is not None:
            buf[:, g.flat_pos] = g.const_val
            continue
        if b > 1:
            args = tuple(
                params[:, col] if swept else params[:1, col]
                for col, swept in zip(g.inp_cols, g.col_swept)
            )
        else:
            args = tuple(params[:, col] for col in g.inp_cols)
        out = g.fwd_fn(args)
        if g.fwd_out_idx is None:
            buf[:, g.flat_pos] = out
        else:
            buf[:, g.flat_pos] = out[:, g.fwd_out_idx]


def _scatter_col_1d(dst_1d: np.ndarray, plan, vals_1d: np.ndarray) -> None:
    mode, target, order, starts = plan
    if mode == 0:
        dst_1d[target] += vals_1d.sum()
    elif mode == 1:
        dst_1d[target] += vals_1d
    else:
        dst_1d[target] += np.add.reduceat(vals_1d[order], starts)


def _scatter_col_2d(dst_2d: np.ndarray, plan, vals_2d: np.ndarray) -> None:
    mode, target, order, starts = plan
    if mode == 0:
        dst_2d[:, target] += vals_2d.sum(axis=1)
    elif mode == 1:
        dst_2d[:, target] += vals_2d
    else:
        dst_2d[:, target] += np.add.reduceat(vals_2d[:, order], starts, axis=1)


def _scatter_col_sweep(g_sweep: np.ndarray, plan, vals_2d: np.ndarray) -> None:
    mode, target, order, starts = plan
    if mode == 0:
        g_sweep[target] += vals_2d.sum(axis=1)
    elif mode == 1:
        g_sweep[target] += vals_2d.T
    else:
        g_sweep[target] += np.add.reduceat(vals_2d[:, order], starts, axis=1).T


def _vjp_np_groups(
    g_buf: np.ndarray, params: np.ndarray, groups: tuple, g_params: np.ndarray, g_sweep=None,
) -> None:
    """Accumulate parameter gradients from sparse buffer cotangents into g_params/g_sweep."""
    b = params.shape[0]
    for g in groups:
        vjp = g.vjp_fn
        if vjp is _vjp_zero:
            continue

        gy_plan = g.gy_plan
        if g.gy_full_first:
            fp0 = gy_plan[0][0]
            gy = g_buf[:, fp0]
            for fp_r, dst_r in gy_plan[1:]:
                if dst_r is None:
                    gy += g_buf[:, fp_r]
                else:
                    gy[:, dst_r] += g_buf[:, fp_r]
        else:
            gy = np.zeros((b, g.k * g.n_out_cols), dtype=np.complex128)
            for idx_r, (fp_r, dst_r) in enumerate(gy_plan):
                if idx_r == 0:
                    gy[:, dst_r] = g_buf[:, fp_r]
                elif dst_r is None:
                    gy += g_buf[:, fp_r]
                else:
                    gy[:, dst_r] += g_buf[:, fp_r]

        if g.n_out_cols > 1:
            gy = gy.reshape(b, g.k, g.n_out_cols)

        if b > 1:
            args = tuple(
                params[:, col] if swept else params[:1, col]
                for col, swept in zip(g.inp_cols, g.col_swept)
            )
        else:
            args = tuple(params[:, col] for col in g.inp_cols)

        gp_cols = vjp(args, gy)

        for gp_j, spec in zip(gp_cols, g.col_scatter):
            kind = spec[0]
            if kind == 0:
                if b == 1:
                    _scatter_col_1d(g_params[0], spec[1], gp_j[0])
                else:
                    _scatter_col_2d(g_params, spec[1], gp_j)
            elif kind == 1:
                _scatter_col_1d(g_params[0], spec[1], gp_j.sum(axis=0))
            elif kind == 2:
                _scatter_col_sweep(g_sweep, spec[1], gp_j)
            else:
                sw_pos, sw_plan = spec[1]
                ns_pos, ns_plan = spec[2]
                _scatter_col_sweep(g_sweep, sw_plan, gp_j[:, sw_pos])
                _scatter_col_1d(g_params[0], ns_plan, gp_j[:, ns_pos].sum(axis=0))


def _expand_np(p_row: np.ndarray, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    r = values.shape[1]
    out = np.repeat(p_row, r, axis=0)
    out[:, indices] = values.T
    return out


def _accumulate_sweep_vjp(g_parent, g_expanded, indices, sweep_values):
    """Route overwritten columns to the sweep and reduce the others into the parent."""
    if not indices.size:
        g_parent += g_expanded
        return np.zeros_like(sweep_values)

    g_sweep = np.zeros_like(sweep_values)
    # Assignment keeps the last value when a parameter occurs more than once.
    _, last_reversed = np.unique(indices[::-1], return_index=True)
    write_rows = indices.size - 1 - last_reversed
    g_sweep[write_rows] = g_expanded[:, indices[write_rows]].T
    g_expanded[:, indices] = 0
    g_parent += g_expanded.sum(axis=0, keepdims=True)
    return g_sweep


# -----------------------------------------------------------------------------
# Full Sparse Forward + Exact Reverse-Mode VJP Host Callbacks
# -----------------------------------------------------------------------------

def _has_near_null_readout(topo, carrier, signal):
    """Detect loss of significance in the optical demodulation of declared ports."""
    ports = topo.readout_ports
    if not ports.size:
        return False
    # A lower-sideband detector's index points into the signal vector (cn + port): check its port.
    ports = np.where(ports >= topo.cn, ports - topo.cn, ports)
    fields = carrier[:, ports]
    upper = fields.conj() * signal[:, ports]
    lower = fields * signal[:, topo.cn + ports]
    readout = upper + lower
    scale = np.abs(upper) + np.abs(lower)
    limit = 1e-7
    for i in range(ports.size):
        if np.any(np.abs(readout[:, i]) < limit * scale[:, i]):
            return True
        if i:
            difference = readout[:, :i] - readout[:, i:i+1]
            pair_scale = scale[:, :i] + scale[:, i:i+1]
            if np.any(np.abs(difference) < limit * pair_scale):
                return True
    return False


def sparse_simulate_fwd_host(
    topo: SparseTopology,
    p_base_in,
    car_ch_vals_in,
    sig_ch_vals_in,
    nsv_in,
    *,
    save_factors=False,
):
    """Run full carrier + signal + quantum-noise sparse forward pass in NumPy/SuperLU."""
    p_base = np.asarray(p_base_in, dtype=np.float64)  # (1, P)
    car_ch_vals = np.asarray(car_ch_vals_in, dtype=np.float64)
    sig_ch_vals = np.asarray(sig_ch_vals_in, dtype=np.float64)
    nsv_base = np.asarray(nsv_in, dtype=np.complex128)  # (D, 1, SN)

    cn, sn = topo.cn, topo.sn
    nnz_c, nnz_s = topo.nnz_c, topo.nnz_s
    D = topo.num_detectors

    # --- Carrier ---
    if topo.car_ch_idx.size > 0:
        p_car = _expand_np(p_base, topo.car_ch_idx, car_ch_vals)
    else:
        p_car = p_base
    b_car = p_car.shape[0]

    car_aug = np.broadcast_to(topo.carrier_base_aug[None, :], (b_car, nnz_c + cn)).copy()
    _apply_np_groups(car_aug, p_car, topo.carrier_groups)
    car_A = car_aug[:, :nnz_c]
    car_b = car_aug[:, nnz_c:]

    carrier = np.empty((b_car, cn), dtype=np.complex128)
    c_matrix = _CSCWorkspace(cn, topo.carrier_csc_indices, topo.carrier_csc_indptr, topo.carrier_csc_pattern)
    car_finite = np.isfinite(car_A).all(axis=1)
    saved_carrier = [] if save_factors else None
    car_sweep = _SweepFactorization(c_matrix) if b_car > 1 else None
    for bi in range(b_car):
        if not car_finite[bi]:
            lu_c = None
        elif car_sweep is not None:
            lu_c = car_sweep.factor(car_A[bi])
        else:
            lu_c = _factorize(c_matrix, car_A[bi])
        if saved_carrier is not None:
            saved_carrier.append(lu_c)
        carrier[bi] = _solve_refined(c_matrix, lu_c, car_b[bi]) if lu_c is not None else np.nan
    if saved_carrier is not None:
        _carrier_factors.save(topo, car_A, saved_carrier)

    # --- Signal assembly ---
    if topo.sig_ch_idx.size > 0:
        p_sig = _expand_np(p_car, topo.sig_ch_idx, sig_ch_vals)
    else:
        p_sig = p_car
    b_sig = p_sig.shape[0]

    sig_aug_static = np.broadcast_to(topo.signal_base_aug[None, :], (b_car, nnz_s + sn)).copy()
    _apply_np_groups(sig_aug_static, p_car, topo.signal_groups_static)
    if topo.signal_static_conj_idx.size > 0:
        sc_idx = topo.signal_static_conj_idx
        sig_aug_static[:, sc_idx] = np.conj(sig_aug_static[:, sc_idx])
    if b_sig != b_car:
        sig_aug = np.broadcast_to(sig_aug_static, (b_sig, nnz_s + sn)).copy()
    else:
        sig_aug = sig_aug_static
    _apply_np_groups(sig_aug, p_sig, topo.signal_groups_sweep)

    carrier_couplings = sig_aug[:, topo.signal_carrier_flat_pos]
    if topo.signal_carrier_flat_pos.size > 0:
        coupled = np.negative(carrier_couplings)
        coupled *= carrier[:, topo.signal_carrier_indices]
        sig_aug[:, topo.signal_carrier_flat_pos] = coupled

    if topo.signal_sweep_conj_idx.size > 0:
        c_idx = topo.signal_sweep_conj_idx
        sig_aug[:, c_idx] = np.conj(sig_aug[:, c_idx])

    sig_A = sig_aug[:, :nnz_s]
    if topo.signal_block_plan is not None:
        # The reverse pass reuses both this condensed matrix and its couplings.
        sig_A = topo.signal_block_plan.pack(sig_A)
    sig_b = sig_aug[:, nnz_s:]

    # --- Noise selection vectors ---
    nsv = nsv_base.copy()
    if topo.qhd_param_idx.size > 0:
        qhd_phases = np.exp(1j * (p_base[0, topo.qhd_param_idx] * _DEG))
        idx3 = topo.qhd_placing_idx.reshape(-1, 3)
        nsv[idx3[:, 0], idx3[:, 1], idx3[:, 2]] *= qhd_phases

    c_sel = np.broadcast_to(nsv, (D, b_car, sn)).copy()
    if D > 0:
        c_sel[:, :, :cn] *= carrier[None, :, :]
        c_sel[:, :, cn : 2 * cn] = np.conj(c_sel[:, :, cn : 2 * cn] * carrier[None, :, :])
    c_sel_full = np.broadcast_to(c_sel, (D, b_sig, sn))

    # --- Signal + Adjoint Noise SuperLU solve (1 factorization per freq) ---
    noise_rhs = c_sel_full[0] if D == 1 else np.moveaxis(c_sel_full, 0, -1)
    cache_mode = "store" if save_factors else "none"
    signal, w = _solve_signal_systems(topo, sig_A, sig_b, "N", noise_rhs, "H", cache=cache_mode)
    if b_sig > 1 and _allows_frequency_batches(topo) and _has_near_null_readout(topo, carrier, signal):
        signal, w = _solve_signal_systems(
            topo, sig_A, sig_b, "N", noise_rhs, "H", cache=cache_mode, allow_batches=False,
        )
    w = w[None, :, :] if D == 1 else np.moveaxis(w, -1, 0)

    # --- Quantum noise readout: Re(w^H N w) ---
    n_vals = np.broadcast_to(topo.noise_base_vals[None, :], (b_car, topo.nnz_n)).copy()
    _apply_np_groups(n_vals, p_car, topo.noise_groups_static)
    if topo.noise_groups_sweep:
        if b_sig != b_car:
            n_vals = np.broadcast_to(n_vals, (b_sig, topo.nnz_n)).copy()
        _apply_np_groups(n_vals, p_sig, topo.noise_groups_sweep)

    if D > 0 and topo.nnz_n > 0:
        w_term = np.conj(w[:, :, topo.noise_rows])
        w_term *= n_vals[None, :, :]
        w_term *= w[:, :, topo.noise_cols]
        temp_sum = np.sum(w_term.real, axis=2)
        noise = np.sqrt(_NOISE_SCALE * temp_sum)
    else:
        noise = np.zeros((D, b_sig), dtype=np.float64)

    return carrier, signal, noise, car_A, sig_A, w, n_vals, carrier_couplings


def sparse_simulate_bwd_host(
    topo: SparseTopology,
    p_base_in,
    car_ch_vals_in,
    sig_ch_vals_in,
    nsv_in,
    carrier_in,
    signal_in,
    noise_in,
    car_A_in,
    sig_A_in,
    w_in,
    n_vals_in,
    carrier_couplings_in,
    g_carrier_in,
    g_signal_in,
    g_noise_in,
):
    """Exact analytical reverse-mode VJP of the entire sparse simulation pass."""
    p_base = np.asarray(p_base_in, dtype=np.float64)
    car_ch_vals = np.asarray(car_ch_vals_in, dtype=np.float64)
    sig_ch_vals = np.asarray(sig_ch_vals_in, dtype=np.float64)
    nsv_base = np.asarray(nsv_in, dtype=np.complex128)
    carrier = np.asarray(carrier_in, dtype=np.complex128)
    signal = np.asarray(signal_in, dtype=np.complex128)
    noise = np.asarray(noise_in, dtype=np.float64)
    car_A = np.asarray(car_A_in, dtype=np.complex128)
    sig_A = np.asarray(sig_A_in, dtype=np.complex128)
    w = np.asarray(w_in, dtype=np.complex128)
    n_vals = np.asarray(n_vals_in, dtype=np.complex128)
    carrier_couplings = np.asarray(carrier_couplings_in, dtype=np.complex128)
    g_carrier = np.asarray(g_carrier_in, dtype=np.complex128).copy()
    g_signal = np.asarray(g_signal_in, dtype=np.complex128)
    g_noise = np.asarray(g_noise_in, dtype=np.float64)

    cn, sn = topo.cn, topo.sn
    nnz_c = topo.nnz_c
    D = topo.num_detectors

    if topo.car_ch_idx.size > 0:
        p_car = _expand_np(p_base, topo.car_ch_idx, car_ch_vals)
    else:
        p_car = p_base
    b_car = p_car.shape[0]

    if topo.sig_ch_idx.size > 0:
        p_sig = _expand_np(p_car, topo.sig_ch_idx, sig_ch_vals)
    else:
        p_sig = p_car
    b_sig = p_sig.shape[0]

    g_p_base = np.zeros_like(p_base)
    g_p_car = np.zeros_like(p_car)
    g_p_sig = np.zeros_like(p_car)
    g_sig_ch = np.zeros_like(sig_ch_vals)
    g_nsv = np.zeros_like(nsv_base)

    # JAX hands the original a symbolic zero for an output the loss does not use, and the original
    # then skips that whole branch. Here an unused output arrives as an all-zero cotangent: skip its
    # branch as well, so non-finite primal values on it (noise = 0 at a dark detector, a singular
    # signal system) do not turn 0 * inf into NaN gradients. A NaN cotangent counts as used.
    signal_used = bool(g_signal.any())
    noise_used = D > 0 and bool(g_noise.any())

    # 1. Backprop through noise = sqrt(_NOISE_SCALE * Re(w^H N w))
    g_w = np.zeros_like(w)  # (D, b_sig, sn) complex128
    g_n_vals = np.zeros((n_vals.shape[0], topo.nnz_n), dtype=np.complex128)

    if noise_used and topo.nnz_n > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            g_ts = g_noise * (0.5 * _NOISE_SCALE / noise)
        w_r_conj = np.conj(w[:, :, topo.noise_rows])
        w_c = w[:, :, topo.noise_cols]
        g_ts3 = g_ts[:, :, None]
        g_n_term = g_ts3 * w_r_conj
        g_n_term *= w_c
        if n_vals.shape[0] == 1:
            g_n_vals[0] = g_n_term.sum(axis=(0, 1))
        else:
            g_n_vals = g_n_term.sum(axis=0)

        g_wr = n_vals[None, :, :] * w_c
        np.conj(g_wr, out=g_wr)
        g_wr *= g_ts3
        g_wc = w_r_conj * n_vals[None, :, :]
        g_wc *= g_ts3
        terms = np.concatenate((g_wr, g_wc), axis=2)
        g_w[:, :, topo.noise_grad_indices] = np.add.reduceat(
            terms[:, :, topo.noise_grad_order], topo.noise_grad_starts, axis=2,
        )

        if topo.noise_groups_sweep:
            _vjp_np_groups(g_n_vals, p_sig, topo.noise_groups_sweep, g_p_sig, g_sig_ch)
            g_n_static = g_n_vals.sum(axis=0, keepdims=True) if b_car == 1 else g_n_vals
        else:
            g_n_static = g_n_vals
        _vjp_np_groups(g_n_static, p_car, topo.noise_groups_static, g_p_car)

    if signal_used or noise_used:
        # 2. Backprop through SuperLU signal + adjoint noise solve
        allow_batches = not _has_near_null_readout(topo, carrier, signal)
        if D == 1:
            bwd_second_rhs = np.conj(g_w[0])
            lam_x_all, adjoint_2d = _solve_signal_systems(
                topo, sig_A, g_signal, "T", bwd_second_rhs, "N", cache="reuse", copy_second=False, allow_batches=allow_batches
            )
            np.conj(adjoint_2d, out=adjoint_2d)
            g_c_sel_full = adjoint_2d[None, :, :]
        else:
            lam_x_all, adjoint = _solve_signal_systems(
                topo, sig_A, g_signal, "T", np.moveaxis(np.conj(g_w), 0, -1), "N", cache="reuse", copy_second=False, allow_batches=allow_batches
            )
            g_c_sel_full = np.moveaxis(np.conj(adjoint), -1, 0)

        # Compute gA and RHS cotangents ONLY for active signal slots in a compact buffer.
        n_act = topo.bwd_sig_n_act
        n_act_a = topo.bwd_sig_n_act_a
        g_sig_aug = np.empty((b_sig, n_act), dtype=np.complex128)
        if n_act_a > 0:
            a_rows = topo.bwd_sig_a_rows
            a_cols = topo.bwd_sig_a_cols
            if signal_used:
                gA = lam_x_all[:, a_rows]
                np.negative(gA, out=gA)
                gA *= signal[:, a_cols]
            else:
                gA = np.zeros((b_sig, n_act_a), dtype=np.complex128)
            if noise_used:
                for d in range(D):
                    t2 = w[d][:, a_rows]
                    t2 *= g_c_sel_full[d][:, a_cols]
                    np.conj(t2, out=t2)
                    gA -= t2
            g_sig_aug[:, :n_act_a] = gA
        if topo.bwd_sig_rhs_rows.size > 0:
            g_sig_aug[:, n_act_a:] = lam_x_all[:, topo.bwd_sig_rhs_rows] if signal_used else 0.0

        # 3. Backprop through carrier_selections (and optional QHD phases) -> g_carrier & g_p_base
        if noise_used:
            g_c_sel = g_c_sel_full.sum(axis=1, keepdims=True) if b_car == 1 else g_c_sel_full
            nsv = nsv_base.copy()
            if topo.qhd_param_idx.size > 0:
                qhd_phases = np.exp(1j * (p_base[0, topo.qhd_param_idx] * _DEG))
                idx3 = topo.qhd_placing_idx.reshape(-1, 3)
                nsv[idx3[:, 0], idx3[:, 1], idx3[:, 2]] *= qhd_phases
            nsv_b = np.broadcast_to(nsv, (D, b_car, sn))

            g_carrier += np.sum(g_c_sel[:, :, :cn] * nsv_b[:, :, :cn], axis=0)
            g_carrier += np.sum(np.conj(g_c_sel[:, :, cn : 2 * cn]) * nsv_b[:, :, cn : 2 * cn], axis=0)

            g_nsv[:, :, :cn] = np.sum(g_c_sel[:, :, :cn] * carrier[None, :, :], axis=1, keepdims=True)
            g_nsv[:, :, cn : 2 * cn] = np.sum(
                np.conj(g_c_sel[:, :, cn : 2 * cn]) * carrier[None, :, :], axis=1, keepdims=True
            )
            g_nsv[:, :, 2 * cn :] = np.sum(g_c_sel[:, :, 2 * cn :], axis=1, keepdims=True)
            if topo.qhd_param_idx.size > 0:
                phase_positions = tuple(idx3.T)
                g_qhd = np.real(g_nsv[phase_positions] * (1j * _DEG * nsv[phase_positions]))
                scatter_add_1d(g_p_base[0], topo.qhd_scatter_plan, g_qhd)
                g_nsv[phase_positions] *= qhd_phases

        # 4. Backprop through signal conjugation mask (active slots only)
        if topo.bwd_sig_conj_idx.size > 0:
            c_idx = topo.bwd_sig_conj_idx
            g_sig_aug[:, c_idx] = np.conj(g_sig_aug[:, c_idx])

        # 5. Backprop through carrier-signal coupling
        if topo.bwd_sig_car_pos.size > 0:
            pos = topo.bwd_sig_car_pos
            g_pos = g_sig_aug[:, pos]
            g_car_contrib = -g_pos * carrier_couplings
            if b_car == 1:
                scatter_add_1d(g_carrier[0], topo.signal_carrier_scatter_plan, g_car_contrib.sum(axis=0))
            else:
                scatter_add_axis1(g_carrier, topo.signal_carrier_scatter_plan, g_car_contrib)
            np.negative(g_pos, out=g_pos)
            g_pos *= carrier[:, topo.signal_carrier_indices]
            g_sig_aug[:, pos] = g_pos

        # 6. Backprop through signal component functions (sweep & static)
        _vjp_np_groups(g_sig_aug, p_sig, topo.signal_groups_sweep, g_p_sig, g_sig_ch)
        if topo.signal_groups_static:
            if b_car == 1:
                g_sig_static = np.empty((1, n_act), dtype=np.complex128)
                sp = topo.bwd_sig_static_pos
                g_sig_static[0, sp] = g_sig_aug[:, sp].sum(axis=0)
            else:
                g_sig_static = g_sig_aug
            _vjp_np_groups(g_sig_static, p_car, topo.signal_groups_static, g_p_car)

    # 7. Backprop through carrier SuperLU solve -> carrier component functions -> g_p_base
    g_car_aug = np.empty((b_car, nnz_c + cn), dtype=np.complex128)
    c_rows = topo.carrier_csc_rows
    c_cols = topo.carrier_csc_cols
    c_matrix = _CSCWorkspace(cn, topo.carrier_csc_indices, topo.carrier_csc_indptr, topo.carrier_csc_pattern)
    car_finite = np.isfinite(car_A).all(axis=1)
    saved_carrier = _carrier_factors.take(topo, car_A)
    car_sweep = _SweepFactorization(c_matrix) if (saved_carrier is None and b_car > 1) else None
    for bi in range(b_car):
        if saved_carrier is None:
            if not car_finite[bi]:
                lu_c = None
            elif car_sweep is not None:
                lu_c = car_sweep.factor(car_A[bi])
            else:
                lu_c = _factorize(c_matrix, car_A[bi])
        else:
            c_matrix.data = car_A[bi]
            lu_c = saved_carrier[bi]
        if lu_c is None:
            g_car_aug[bi] = np.nan
            continue
        lam_c = _solve_refined(c_matrix, lu_c, g_carrier[bi], trans="T")
        g_car_aug[bi, nnz_c:] = lam_c
        g_car_aug[bi, :nnz_c] = -lam_c[c_rows] * carrier[bi, c_cols]

    _vjp_np_groups(g_car_aug, p_car, topo.carrier_groups, g_p_car)

    g_p_car += g_p_sig
    g_car_ch = _accumulate_sweep_vjp(g_p_base, g_p_car, topo.car_ch_idx, car_ch_vals)
    return g_p_base, g_car_ch, g_sig_ch, g_nsv
