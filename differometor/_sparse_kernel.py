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
from importlib.metadata import version
from importlib.util import find_spec, module_from_spec, spec_from_file_location
from pathlib import Path

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


def _load_pinned_superlu():
    """Load the pinned wheel's native solver without eagerly importing SciPy sparse."""
    try:
        if version("scipy") != "1.18.1":
            return None
        scipy = find_spec("scipy")
        if scipy is None or scipy.origin is None:
            return None
        directory = Path(scipy.origin).parent / "sparse" / "linalg" / "_dsolve"
        for suffix in EXTENSION_SUFFIXES:
            binary = directory / f"_superlu{suffix}"
            if binary.is_file():
                spec = spec_from_file_location("_superlu", binary)
                if spec is None or spec.loader is None:
                    continue
                module = module_from_spec(spec)
                spec.loader.exec_module(module)
                return getattr(module, "gstrf", None)
    except (ImportError, OSError):
        return None
    return None


_gstrf = _load_pinned_superlu()
_LU_PANEL_SIZE = 4
_LU_RELAX = 2


def _make_sparse_matrix(*args, **kwargs):
    from scipy.sparse import csc_matrix

    return csc_matrix(*args, **kwargs)


def _factorize(matrix, data, *, ordering="COLAMD"):
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
                    "PanelSize": _LU_PANEL_SIZE,
                    "Relax": _LU_RELAX,
                },
                csc_construct_func=_make_sparse_matrix,
            )
        from scipy.sparse.linalg import splu

        csc = _make_sparse_matrix((data, matrix.indices, matrix.indptr), shape=matrix.shape)
        return splu(csc, permc_spec=ordering, diag_pivot_thresh=1.0, panel_size=_LU_PANEL_SIZE, relax=_LU_RELAX)
    except RuntimeError as error:
        if "singular" not in str(error).lower():
            raise
        return None


class _CSCWorkspace:
    """Keep a sparsity pattern and apply the current matrix when refining a solve."""

    def __init__(self, size, indices, indptr):
        self.shape = (size, size)
        self.indices = indices
        self.indptr = indptr
        self._row_order = None

    def multiply(self, values, trans):
        if self._row_order is None:
            size = self.shape[0]
            self._row_order = np.argsort(self.indices, kind="stable")
            self._cols_in_row_order = np.repeat(np.arange(size), np.diff(self.indptr))[self._row_order]
            counts = np.bincount(self.indices, minlength=size)
            self._rows_with_terms = np.flatnonzero(counts)
            self._row_starts = (np.cumsum(counts) - counts)[self._rows_with_terms]
            self._cols_with_terms = np.flatnonzero(np.diff(self.indptr))

        if trans == "N":
            data = self.data[self._row_order]
            picked = values[self._cols_in_row_order]
            starts = self._row_starts
            filled = self._rows_with_terms
        else:
            data = np.conj(self.data) if trans == "H" else self.data
            picked = values[self.indices]
            starts = self.indptr[self._cols_with_terms]
            filled = self._cols_with_terms
        product = np.zeros_like(values)
        if filled.size:
            terms = data * picked if values.ndim == 1 else data[:, None] * picked
            product[filled] = np.add.reduceat(terms, starts, axis=0)
        return product


class _SweepFactorization:
    """Reuse the first successful column ordering within a single sweep."""

    def __init__(self, matrix):
        self.matrix = matrix
        self.order = None

    def factor(self, values):
        self.matrix.data = values
        if self.order is None:
            factors = _factorize(self.matrix, values)
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

        self.factors = _factorize(self.reordered, values[self.positions], ordering="NATURAL")
        if self.factors is None:
            return _factorize(self.matrix, values)
        return self

    def solve(self, rhs, trans="N"):
        if trans == "N":
            return self.factors.solve(rhs)[self.inverse_order]
        return self.factors.solve(rhs[self.order], trans=trans)


def _solve_refined(matrix, factors, rhs, trans="N"):
    """Correct the solve residual in float64 using the existing LU factors."""
    solution = factors.solve(rhs, trans=trans)
    residual = rhs - matrix.multiply(solution, trans)
    return solution + factors.solve(residual, trans=trans)


def _solve_signal_systems(topo, values, first_rhs, first_trans, second_rhs, second_trans):
    """Factor the coupled block once per frequency for both right-hand sides."""
    plan = topo.signal_block_plan
    finite = np.isfinite(values).all(axis=1)

    has_second = second_rhs.shape[-1] != 0
    if has_second and second_rhs.shape[-1] == 1:
        second_rhs = second_rhs[..., 0]
        squeeze_second = True
    else:
        squeeze_second = False

    if plan is None:
        matrix = _CSCWorkspace(topo.sn, topo.signal_csc_indices, topo.signal_csc_indptr)
        first_out = np.empty_like(first_rhs)
        second_out = np.empty_like(second_rhs)
        first = first_rhs
        second = second_rhs
        block_values = values
    else:
        matrix = _CSCWorkspace(plan.core.size, plan.rows, plan.indptr)
        first_out = plan.prepare(values, first_rhs, first_trans)
        second_out = plan.prepare(values, second_rhs, second_trans) if has_second else second_rhs
        first = np.take(first_out, plan.core, axis=1)
        second = np.take(second_out, plan.core, axis=1) if has_second else second_out
        block_values = values[:, :plan.rows.size]

    failed = ~finite
    if matrix.shape[0]:
        # The condensed right-hand sides are private buffers and can hold the solutions.
        first_solved = first_out if plan is None else first
        second_solved = second_out if plan is None else second
        sweep = _SweepFactorization(matrix)
        for frequency, entries in enumerate(block_values):
            factor = sweep.factor(entries) if finite[frequency] else None
            if factor is None:
                failed[frequency] = True
                continue
            first_solved[frequency] = _solve_refined(matrix, factor, first[frequency], trans=first_trans)
            if has_second:
                second_solved[frequency] = _solve_refined(matrix, factor, second[frequency], trans=second_trans)

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
# Helper geometry / angle derivatives (real float64)
# -----------------------------------------------------------------------------

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
    # d(cos(arcsin(u)))/du = -u / cos(beta)
    dcb_du = -u / cos_b
    d_alpha = dcb_du * (n_l * inv_nr * cos_a * _DEG)
    d_nl = dcb_du * (sin_a * inv_nr)
    d_nr = dcb_du * (-u * inv_nr)
    return cos_b, d_alpha, d_nl, d_nr


# -----------------------------------------------------------------------------
# Vectorized NumPy forward evaluators (indices 0..18 aligned with FUNCTIONS)
# Input p: float64 shape (B, K, 7) -> Output y: complex128 shape (B, K, C)
# -----------------------------------------------------------------------------

def _fwd_0(p):  # force_calculation_right
    cos_b = _cos_beta(p[..., 0], p[..., 1], p[..., 2])
    return (cos_b * _INV_CX).astype(np.complex128)[..., None]


def _fwd_1(p):  # laser
    amp = np.sqrt(2.0 * p[..., 0] / EPSILON_0C)
    return (amp * np.exp(1j * (p[..., 1] * _DEG)))[..., None]


def _fwd_2(p):  # surface
    loss, r_frac, tuning, freq, n_l, n_r, alpha = (
        p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4], p[..., 5], p[..., 6]
    )
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
    r_l = -sqrt_r * np.exp(1j * (ps_l * cl))
    t_val = -sqrt_t * np.exp(1j * (0.5 * np.pi + 0.5 * (ps_l * cl - ps_r * cr_t)))
    r_r = -sqrt_r * np.exp(-1j * (ps_r * cr_r))
    return np.stack([r_l, t_val, r_r], axis=-1)


def _fwd_3(p):  # space
    theta = -_TWO_PI * p[..., 0] * p[..., 1] * p[..., 2] * _INV_C
    return (-np.exp(1j * theta))[..., None]


def _fwd_4(p):  # space_modulation
    w_g = _TWO_PI * p[..., 0]
    w_0 = _TWO_PI * F0
    arg = w_g * p[..., 1] * p[..., 2] * (0.5 * _INV_C)
    m_g = -0.5 * (w_0 / w_g) * np.sin(arg)
    return (1j * m_g * np.exp(-1j * arg))[..., None]


def _fwd_5(p):  # signal_function
    return (p[..., 0] * np.exp(1j * (p[..., 1] * _DEG)))[..., None]


def _fwd_6(p):  # dummy_function
    return np.ones(p.shape[:-1] + (1,), dtype=np.complex128)


def _fwd_7(p):  # vacuum_quantum_noise
    return np.full(p.shape[:-1] + (1,), UNIT_VACUUM * 0.5, dtype=np.complex128)


def _fwd_8(p):  # loss_quantum_noise
    return (p[..., 0] * 0.5).astype(np.complex128)[..., None]


def _fwd_9(p):  # space_lower
    theta = _TWO_PI * p[..., 0] * p[..., 1] * p[..., 2] * _INV_C
    return (-np.exp(1j * theta))[..., None]


def _fwd_10(p):  # laser_amplitude_modulation
    return np.full(p.shape[:-1] + (1,), EPSILON_0C * 0.5, dtype=np.complex128)


def _fwd_11(p):  # laser_frequency_modulation
    return (EPSILON_0C * 0.5 / p[..., 0]).astype(np.complex128)[..., None]


def _fwd_12(p):  # susceptibility
    w = _TWO_PI * p[..., 0]
    return (1.0 / (p[..., 1] * w * w)).astype(np.complex128)[..., None]


def _fwd_13(p):  # force_calculation_left
    return (-np.cos(p[..., 0] * _DEG) * _INV_CX).astype(np.complex128)[..., None]


def _fwd_14(p):  # corrected_optomechanical_phase_left
    freq, tuning, r_frac, loss, n_l, alpha = (
        p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4], p[..., 5]
    )
    abs_r = (1.0 - loss) * r_frac
    cl = np.cos(alpha * _DEG)
    t_rad = tuning * _DEG
    phase = t_rad * (freq * _INV_F0) + 2.0 * t_rad * n_l * (1.0 + freq * _INV_F0) * cl
    return (1j * _Z_SCALE * cl * np.sqrt(abs_r) * np.exp(1j * phase))[..., None]


def _fwd_15(p):  # corrected_optomechanical_phase_right
    freq, tuning, r_frac, loss, n_l, n_r, alpha = (
        p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4], p[..., 5], p[..., 6]
    )
    abs_r = (1.0 - loss) * r_frac
    cr_om = _cos_beta(alpha, n_l, n_r)
    cr_rf = _cos_beta(alpha, n_r, n_l)
    t_rad = tuning * _DEG
    phase = -t_rad * (freq * _INV_F0) - 2.0 * t_rad * n_r * (1.0 + freq * _INV_F0) * cr_rf
    return (-1j * _Z_SCALE * cr_om * np.sqrt(abs_r) * np.exp(1j * phase))[..., None]


def _fwd_16(p):  # space_modulation_lower
    w_g = -_TWO_PI * p[..., 0]
    w_0 = _TWO_PI * F0
    arg = w_g * p[..., 1] * p[..., 2] * (0.5 * _INV_C)
    m_g = -0.5 * (w_0 / w_g) * np.sin(arg)
    return (1j * m_g * np.exp(-1j * arg))[..., None]


def _fwd_17(p):  # squeezer
    vu = UNIT_VACUUM * 0.5
    two_s = (2.0 * _SQ_K) * p[..., 0]
    ph = np.exp(1j * (2.0 * _DEG) * p[..., 1])
    u_d = (vu * np.cosh(two_s)).astype(np.complex128)
    u_o = (vu * np.sinh(two_s)) * ph
    return np.stack([u_d, u_o, u_d, np.conj(u_o)], axis=-1)


def _fwd_18(p):  # laser_frequency_modulation_lower
    return (-EPSILON_0C * 0.5 / p[..., 0]).astype(np.complex128)[..., None]


NP_FWD_FUNCTIONS = (
    _fwd_0, _fwd_1, _fwd_2, _fwd_3, _fwd_4, _fwd_5, _fwd_6, _fwd_7, _fwd_8,
    _fwd_9, _fwd_10, _fwd_11, _fwd_12, _fwd_13, _fwd_14, _fwd_15, _fwd_16,
    _fwd_17, _fwd_18,
)


# -----------------------------------------------------------------------------
# Exact analytical reverse-mode VJPs (indices 0..18 aligned with FUNCTIONS)
# Inputs: p (B, K, 7) float64, gy (B, K, C) complex128
# Output: gp (B, K, 7) float64  (only used columns populated; unused stay 0)
# -----------------------------------------------------------------------------

def _vjp_0(p, gy):  # force_calculation_right
    gp = np.zeros_like(p)
    _, d_a, d_nl, d_nr = _cos_beta_and_derivs(p[..., 0], p[..., 1], p[..., 2])
    g = np.real(gy[..., 0]) * _INV_CX
    gp[..., 0] = g * d_a
    gp[..., 1] = g * d_nl
    gp[..., 2] = g * d_nr
    return gp


def _vjp_1(p, gy):  # laser
    gp = np.zeros_like(p)
    gy0 = gy[..., 0]
    amp = np.sqrt(2.0 * p[..., 0] / EPSILON_0C)
    e = np.exp(1j * (p[..., 1] * _DEG))
    y = amp * e
    inv_amp = np.where(amp > 0.0, 1.0 / np.where(amp > 0.0, amp, 1.0), 0.0)
    gp[..., 0] = np.real(gy0 * e) * (inv_amp / EPSILON_0C)
    gp[..., 1] = np.real(gy0 * (1j * _DEG * y))
    return gp


def _vjp_2(p, gy):  # surface
    gp = np.zeros_like(p)
    loss, r_frac, tuning, freq, n_l, n_r, alpha = (
        p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4], p[..., 5], p[..., 6]
    )
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
    e_rl = np.exp(1j * (ps_l * cl))
    e_t = np.exp(1j * (0.5 * np.pi + 0.5 * (ps_l * cl - ps_r * cr_t)))
    e_rr = np.exp(-1j * (ps_r * cr_r))
    r_l = -sqrt_r * e_rl
    t_val = -sqrt_t * e_t
    r_r = -sqrt_r * e_rr

    g_rl, g_t, g_rr = gy[..., 0], gy[..., 1], gy[..., 2]

    # Derivatives wrt abs_r and abs_t
    inv_2sr = np.where(sqrt_r > 0.0, -0.5 / np.where(sqrt_r > 0.0, sqrt_r, 1.0), 0.0)
    inv_2st = np.where(sqrt_t > 0.0, -0.5 / np.where(sqrt_t > 0.0, sqrt_t, 1.0), 0.0)
    g_abs_r = np.real(g_rl * e_rl + g_rr * e_rr) * inv_2sr
    g_abs_t = np.real(g_t * e_t) * inv_2st

    gp[..., 0] = -g_abs_r * r_frac - g_abs_t * (1.0 - r_frac)
    gp[..., 1] = (g_abs_r - g_abs_t) * (1.0 - loss)

    # Derivatives wrt phases: phi_rl = ps_l*cl, phi_t = pi/2 + 0.5*(ps_l*cl - ps_r*cr_t), phi_rr = -ps_r*cr_r
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
    gp[..., 2] = g_t_scale * (2.0 * _DEG * f_fac)
    gp[..., 3] = g_t_scale * (2.0 * t_rad * _INV_F0)
    gp[..., 4] = g_ps_l * t_scale + g_cr_t * dcr_t_dnl + g_cr_r * dcr_r_dnl
    gp[..., 5] = g_ps_r * t_scale + g_cr_t * dcr_t_dnr + g_cr_r * dcr_r_dnr
    gp[..., 6] = g_cl * dcl_da + g_cr_t * dcr_t_da + g_cr_r * dcr_r_da
    return gp


def _vjp_space_signed(p, gy, sign):
    gp = np.zeros_like(p)
    c = sign * _TWO_PI * _INV_C
    theta = c * p[..., 0] * p[..., 1] * p[..., 2]
    y = -np.exp(1j * theta)
    g_theta = np.real(gy[..., 0] * (1j * y)) * c
    gp[..., 0] = g_theta * (p[..., 1] * p[..., 2])
    gp[..., 1] = g_theta * (p[..., 0] * p[..., 2])
    gp[..., 2] = g_theta * (p[..., 0] * p[..., 1])
    return gp


def _vjp_3(p, gy):  # space
    return _vjp_space_signed(p, gy, -1.0)


def _vjp_9(p, gy):  # space_lower
    return _vjp_space_signed(p, gy, 1.0)


def _vjp_space_mod_signed(p, gy, sign):
    gp = np.zeros_like(p)
    w_g = (sign * _TWO_PI) * p[..., 0]
    w_0 = _TWO_PI * F0
    half_inv_c = 0.5 * _INV_C
    ln_factor = p[..., 1] * p[..., 2] * half_inv_c
    arg = w_g * ln_factor
    sin_a = np.sin(arg)
    cos_a = np.cos(arg)
    inv_wg = 1.0 / w_g
    coef = -0.5 * w_0 * inv_wg
    m_g = coef * sin_a
    e = np.exp(-1j * arg)
    y = 1j * m_g * e
    gy0 = gy[..., 0]

    g_mg = np.real(gy0 * (1j * e))
    g_arg = np.real(gy0 * (-1j * y)) + g_mg * (coef * cos_a)
    g_wg = g_mg * (-m_g * inv_wg) + g_arg * ln_factor

    gp[..., 0] = g_wg * (sign * _TWO_PI)
    gp[..., 1] = g_arg * (w_g * p[..., 2] * half_inv_c)
    gp[..., 2] = g_arg * (w_g * p[..., 1] * half_inv_c)
    return gp


def _vjp_4(p, gy):  # space_modulation
    return _vjp_space_mod_signed(p, gy, 1.0)


def _vjp_16(p, gy):  # space_modulation_lower
    return _vjp_space_mod_signed(p, gy, -1.0)


def _vjp_5(p, gy):  # signal_function
    gp = np.zeros_like(p)
    e = np.exp(1j * (p[..., 1] * _DEG))
    gy0 = gy[..., 0]
    gp[..., 0] = np.real(gy0 * e)
    gp[..., 1] = np.real(gy0 * (1j * _DEG * p[..., 0] * e))
    return gp


def _vjp_zero(p, gy):  # constants: dummy, vacuum_quantum_noise, laser_amplitude_modulation
    return np.zeros_like(p)


def _vjp_8(p, gy):  # loss_quantum_noise
    gp = np.zeros_like(p)
    gp[..., 0] = 0.5 * np.real(gy[..., 0])
    return gp


def _vjp_11(p, gy):  # laser_frequency_modulation
    gp = np.zeros_like(p)
    gp[..., 0] = np.real(gy[..., 0]) * (-EPSILON_0C * 0.5 / (p[..., 0] ** 2))
    return gp


def _vjp_18(p, gy):  # laser_frequency_modulation_lower
    gp = np.zeros_like(p)
    gp[..., 0] = np.real(gy[..., 0]) * (EPSILON_0C * 0.5 / (p[..., 0] ** 2))
    return gp


def _vjp_12(p, gy):  # susceptibility
    gp = np.zeros_like(p)
    w = _TWO_PI * p[..., 0]
    y = 1.0 / (p[..., 1] * w * w)
    g = np.real(gy[..., 0]) * y
    gp[..., 0] = -2.0 * g / p[..., 0]
    gp[..., 1] = -g / p[..., 1]
    return gp


def _vjp_13(p, gy):  # force_calculation_left
    gp = np.zeros_like(p)
    gp[..., 0] = np.real(gy[..., 0]) * (np.sin(p[..., 0] * _DEG) * _DEG * _INV_CX)
    return gp


def _vjp_14(p, gy):  # corrected_optomechanical_phase_left
    gp = np.zeros_like(p)
    freq, tuning, r_frac, loss, n_l, alpha = (
        p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4], p[..., 5]
    )
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
    gy0 = gy[..., 0]

    g_cl = np.real(gy0 * ((1j * _Z_SCALE) * sqrt_r * e))
    inv_2sr = np.where(sqrt_r > 0.0, 0.5 / np.where(sqrt_r > 0.0, sqrt_r, 1.0), 0.0)
    g_abs_r = np.real(gy0 * ((1j * _Z_SCALE) * cl * inv_2sr * e))
    g_phase = np.real(gy0 * (1j * y))

    gp[..., 0] = g_phase * (t_rad * _INV_F0 * (1.0 + 2.0 * n_l * cl))
    gp[..., 1] = g_phase * (_DEG * (freq * _INV_F0 + 2.0 * n_l * f_fac * cl))
    gp[..., 2] = g_abs_r * (1.0 - loss)
    gp[..., 3] = -g_abs_r * r_frac
    gp[..., 4] = g_phase * (2.0 * t_rad * f_fac * cl)
    gp[..., 5] = (g_cl + g_phase * (2.0 * t_rad * n_l * f_fac)) * dcl_da
    return gp


def _vjp_15(p, gy):  # corrected_optomechanical_phase_right
    gp = np.zeros_like(p)
    freq, tuning, r_frac, loss, n_l, n_r, alpha = (
        p[..., 0], p[..., 1], p[..., 2], p[..., 3], p[..., 4], p[..., 5], p[..., 6]
    )
    abs_r = (1.0 - loss) * r_frac
    sqrt_r = np.sqrt(abs_r)
    cr_om, dom_da, dom_dnl, dom_dnr = _cos_beta_and_derivs(alpha, n_l, n_r)
    cr_rf, drf_da, drf_dnr, drf_dnl = _cos_beta_and_derivs(alpha, n_r, n_l)
    t_rad = tuning * _DEG
    f_fac = 1.0 + freq * _INV_F0
    phase = -t_rad * (freq * _INV_F0) - 2.0 * t_rad * n_r * f_fac * cr_rf
    e = np.exp(1j * phase)
    y = (-1j * _Z_SCALE) * cr_om * sqrt_r * e
    gy0 = gy[..., 0]

    g_cr_om = np.real(gy0 * ((-1j * _Z_SCALE) * sqrt_r * e))
    inv_2sr = np.where(sqrt_r > 0.0, 0.5 / np.where(sqrt_r > 0.0, sqrt_r, 1.0), 0.0)
    g_abs_r = np.real(gy0 * ((-1j * _Z_SCALE) * cr_om * inv_2sr * e))
    g_phase = np.real(gy0 * (1j * y))
    g_cr_rf = g_phase * (-2.0 * t_rad * n_r * f_fac)

    gp[..., 0] = g_phase * (-t_rad * _INV_F0 * (1.0 + 2.0 * n_r * cr_rf))
    gp[..., 1] = g_phase * (-_DEG * (freq * _INV_F0 + 2.0 * n_r * f_fac * cr_rf))
    gp[..., 2] = g_abs_r * (1.0 - loss)
    gp[..., 3] = -g_abs_r * r_frac
    gp[..., 4] = g_cr_om * dom_dnl + g_cr_rf * drf_dnl
    gp[..., 5] = g_cr_om * dom_dnr + g_cr_rf * drf_dnr + g_phase * (-2.0 * t_rad * f_fac * cr_rf)
    gp[..., 6] = g_cr_om * dom_da + g_cr_rf * drf_da
    return gp


def _vjp_17(p, gy):  # squeezer
    gp = np.zeros_like(p)
    vu = UNIT_VACUUM * 0.5
    two_k = 2.0 * _SQ_K
    two_s = two_k * p[..., 0]
    sh = np.sinh(two_s)
    ch = np.cosh(two_s)
    ph = np.exp(1j * (2.0 * _DEG) * p[..., 1])
    u_o = (vu * sh) * ph
    l_o = np.conj(u_o)

    g_ud, g_uo, g_ld, g_lo = gy[..., 0], gy[..., 1], gy[..., 2], gy[..., 3]
    gp[..., 0] = (
        np.real(g_ud + g_ld) * (two_k * vu * sh)
        + np.real(g_uo * ph + g_lo * np.conj(ph)) * (two_k * vu * ch)
    )
    gp[..., 1] = np.real(g_uo * (2j * _DEG * u_o) + g_lo * (-2j * _DEG * l_o))
    return gp


NP_VJP_FUNCTIONS = (
    _vjp_0, _vjp_1, _vjp_2, _vjp_3, _vjp_4, _vjp_5, _vjp_zero, _vjp_zero, _vjp_8,
    _vjp_9, _vjp_zero, _vjp_11, _vjp_12, _vjp_13, _vjp_14, _vjp_15, _vjp_16,
    _vjp_17, _vjp_18,
)


# -----------------------------------------------------------------------------
# Sparse Topology Pytree Container (0 dynamic JAX leaves -> near-zero XLA graph)
# -----------------------------------------------------------------------------

class SparseTopology:
    """Immutable host-side sparse topology passed as static JAX pytree aux_data."""

    __slots__ = (
        "cn", "sn", "nnz_c", "nnz_s", "nnz_n", "num_detectors",
        "opt_param_idx", "opt_val_idx",
        "car_ch_idx", "sig_ch_idx",
        "qhd_param_idx", "qhd_placing_idx",
        "carrier_base_aug", "carrier_csc_rows", "carrier_csc_cols",
        "carrier_csc_indices", "carrier_csc_indptr", "carrier_groups",
        "signal_base_aug", "signal_csc_rows", "signal_csc_cols",
        "signal_csc_indices", "signal_csc_indptr",
        "signal_block_plan",
        "signal_groups_static", "signal_groups_sweep",
        "signal_carrier_indices", "signal_carrier_flat_pos", "signal_conj_mask",
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
    for f_idx, inp_idx, out_row, out_col, flat_pos, _ in groups:
        out = NP_FWD_FUNCTIONS[f_idx](params[:, inp_idx])  # (B, K, C)
        buf[:, flat_pos] = out[:, out_row, out_col]


def _vjp_np_groups(
    g_buf: np.ndarray, params: np.ndarray, groups: tuple, g_params: np.ndarray, sweep=None,
) -> None:
    """Accumulate parameter gradients from sparse buffer cotangents into g_params."""
    B = params.shape[0]
    for f_idx, inp_idx, out_row, out_col, flat_pos, n_out_cols in groups:
        vjp = NP_VJP_FUNCTIONS[f_idx]
        if vjp is _vjp_zero:
            continue
        K = inp_idx.shape[0]
        # Gather upstream cotangent for each (b, k, c)
        gy = np.zeros((B, K, n_out_cols), dtype=np.complex128)
        np.add.at(gy, (slice(None), out_row, out_col), g_buf[:, flat_pos])
        gp_local = vjp(params[:, inp_idx], gy)
        if sweep is not None:
            sweep_rows, g_sweep = sweep
            flat_indices = inp_idx.ravel()
            flat_grad = gp_local.reshape(B, -1)
            slots = sweep_rows[flat_indices]
            is_swept = slots >= 0
            np.add.at(g_sweep, (slots[is_swept], slice(None)), flat_grad[:, is_swept].T)
            np.add.at(g_params[0], flat_indices[~is_swept], flat_grad[:, ~is_swept].sum(axis=0))
        elif B == 1:
            np.add.at(g_params[0], inp_idx.ravel(), gp_local[0].ravel())
        else:
            np.add.at(g_params, (slice(None), inp_idx.ravel()), gp_local.reshape(B, -1))


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

def sparse_simulate_fwd_host(
    topo: SparseTopology,
    p_base_in,
    car_ch_vals_in,
    sig_ch_vals_in,
    nsv_in,
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
    c_matrix = _CSCWorkspace(cn, topo.carrier_csc_indices, topo.carrier_csc_indptr)
    car_finite = np.isfinite(car_A).all(axis=1)
    for bi in range(b_car):
        lu_c = _factorize(c_matrix, car_A[bi]) if car_finite[bi] else None
        carrier[bi] = _solve_refined(c_matrix, lu_c, car_b[bi]) if lu_c is not None else np.nan

    # --- Signal assembly ---
    if topo.sig_ch_idx.size > 0:
        p_sig = _expand_np(p_car, topo.sig_ch_idx, sig_ch_vals)
    else:
        p_sig = p_car
    b_sig = p_sig.shape[0]

    sig_aug_static = np.broadcast_to(topo.signal_base_aug[None, :], (b_car, nnz_s + sn)).copy()
    _apply_np_groups(sig_aug_static, p_car, topo.signal_groups_static)
    if b_sig != b_car:
        sig_aug = np.broadcast_to(sig_aug_static, (b_sig, nnz_s + sn)).copy()
    else:
        sig_aug = sig_aug_static
    _apply_np_groups(sig_aug, p_sig, topo.signal_groups_sweep)

    carrier_couplings = sig_aug[:, topo.signal_carrier_flat_pos]
    if topo.signal_carrier_flat_pos.size > 0:
        sig_aug[:, topo.signal_carrier_flat_pos] = -carrier_couplings * carrier[:, topo.signal_carrier_indices]

    if topo.signal_conj_mask.any():
        sig_aug[:, topo.signal_conj_mask] = np.conj(sig_aug[:, topo.signal_conj_mask])

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
    signal, w = _solve_signal_systems(topo, sig_A, sig_b, "N", np.moveaxis(c_sel_full, 0, -1), "H")
    w = np.moveaxis(w, -1, 0)

    # --- Quantum noise readout: Re(w^H N w) ---
    n_vals = np.broadcast_to(topo.noise_base_vals[None, :], (b_car, topo.nnz_n)).copy()
    _apply_np_groups(n_vals, p_car, topo.noise_groups_static)
    if topo.noise_groups_sweep:
        if b_sig != b_car:
            n_vals = np.broadcast_to(n_vals, (b_sig, topo.nnz_n)).copy()
        _apply_np_groups(n_vals, p_sig, topo.noise_groups_sweep)

    if D > 0 and topo.nnz_n > 0:
        temp_sum = np.sum(
            np.real(np.conj(w[:, :, topo.noise_rows]) * n_vals[None, :, :] * w[:, :, topo.noise_cols]),
            axis=2,
        )
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
    nnz_c, nnz_s = topo.nnz_c, topo.nnz_s
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
    signal_sweep = None
    if topo.sig_ch_idx.size:
        sweep_rows = np.full(p_car.shape[1], -1, dtype=np.int32)
        sweep_rows[topo.sig_ch_idx] = np.arange(topo.sig_ch_idx.size)
        signal_sweep = (sweep_rows, g_sig_ch)
    g_nsv = np.zeros_like(nsv_base)

    # 1. Backprop through noise = sqrt(_NOISE_SCALE * Re(w^H N w))
    g_w = np.zeros_like(w)  # (D, b_sig, sn) complex128
    g_n_vals = np.zeros((n_vals.shape[0], topo.nnz_n), dtype=np.complex128)

    if D > 0 and topo.nnz_n > 0:
        # g_temp_sum = g_noise * (_NOISE_SCALE / (2 * noise))
        with np.errstate(divide="ignore", invalid="ignore"):
            g_ts = g_noise * (0.5 * _NOISE_SCALE / noise)
        w_r = w[:, :, topo.noise_rows]  # (D, b_sig, nnz_n)
        w_c = w[:, :, topo.noise_cols]  # (D, b_sig, nnz_n)
        # Re(conj(w_r) * n_k * w_c):
        # wrt n_k: delta Re(conj(w_r)*w_c * dn_k) -> g_n_k = g_ts * conj(w_r) * w_c
        g_n_term = g_ts[:, :, None] * np.conj(w_r) * w_c  # (D, b_sig, nnz_n)
        if n_vals.shape[0] == 1:
            g_n_vals[0] = g_n_term.sum(axis=(0, 1))
        else:
            g_n_vals = g_n_term.sum(axis=0)

        # wrt w_r: Re(conj(dw_r) * (n_k * w_c)) = Re(dw_r * conj(n_k * w_c)) -> g_wr = g_ts * conj(n_k * w_c)
        # wrt w_c: Re(dw_c * (conj(w_r) * n_k)) -> g_wc = g_ts * conj(w_r) * n_k
        g_wr = g_ts[:, :, None] * np.conj(n_vals[None, :, :] * w_c)
        g_wc = g_ts[:, :, None] * (np.conj(w_r) * n_vals[None, :, :])
        terms = np.concatenate((g_wr, g_wc), axis=2)
        g_w[:, :, topo.noise_grad_indices] = np.add.reduceat(
            terms[:, :, topo.noise_grad_order], topo.noise_grad_starts, axis=2,
        )

        if topo.noise_groups_sweep:
            _vjp_np_groups(g_n_vals, p_sig, topo.noise_groups_sweep, g_p_sig, signal_sweep)
            g_n_static = g_n_vals.sum(axis=0, keepdims=True) if b_car == 1 else g_n_vals
        else:
            g_n_static = g_n_vals
        _vjp_np_groups(g_n_static, p_car, topo.noise_groups_static, g_p_car)

    # 2. Backprop through SuperLU signal + adjoint noise solve
    g_sig_aug = np.empty((b_sig, nnz_s + sn), dtype=np.complex128)
    s_rows = topo.signal_csc_rows
    s_cols = topo.signal_csc_cols

    lam_x_all, adjoint = _solve_signal_systems(
        topo, sig_A, g_signal, "T", np.moveaxis(np.conj(g_w), 0, -1), "N"
    )
    g_c_sel_full = np.moveaxis(np.conj(adjoint), -1, 0)

    g_sig_aug[:, nnz_s:] = lam_x_all
    gA_all = -lam_x_all[:, s_rows] * signal[:, s_cols]
    if D == 1:
        gA_all -= np.conj(w[0][:, s_rows] * g_c_sel_full[0][:, s_cols])
    elif D > 1:
        gA_all -= np.sum(np.conj(w[:, :, s_rows] * g_c_sel_full[:, :, s_cols]), axis=0)
    g_sig_aug[:, :nnz_s] = gA_all

    # 3. Backprop through carrier_selections (and optional QHD phases) -> g_carrier & g_p_base
    if D > 0:
        g_c_sel = g_c_sel_full.sum(axis=1, keepdims=True) if b_car == 1 else g_c_sel_full  # (D, b_car, sn)
        nsv = nsv_base.copy()
        if topo.qhd_param_idx.size > 0:
            qhd_phases = np.exp(1j * (p_base[0, topo.qhd_param_idx] * _DEG))
            idx3 = topo.qhd_placing_idx.reshape(-1, 3)
            nsv[idx3[:, 0], idx3[:, 1], idx3[:, 2]] *= qhd_phases
        nsv_b = np.broadcast_to(nsv, (D, b_car, sn))

        # upper: c_sel[:cn] = nsv[:cn] * carrier -> g_carrier += sum_d(g_c_sel[:cn] * nsv[:cn])
        g_carrier += np.sum(g_c_sel[:, :, :cn] * nsv_b[:, :, :cn], axis=0)
        # lower: c_sel[cn:2cn] = conj(nsv[cn:2cn] * carrier)
        # Re(g * conj(nsv * dcar)) = Re(conj(g) * nsv * dcar) -> g_carrier += sum_d(conj(g) * nsv)
        g_carrier += np.sum(np.conj(g_c_sel[:, :, cn : 2 * cn]) * nsv_b[:, :, cn : 2 * cn], axis=0)

        g_nsv[:, :, :cn] = np.sum(g_c_sel[:, :, :cn] * carrier[None, :, :], axis=1, keepdims=True)
        g_nsv[:, :, cn : 2 * cn] = np.sum(
            np.conj(g_c_sel[:, :, cn : 2 * cn]) * carrier[None, :, :], axis=1, keepdims=True
        )
        g_nsv[:, :, 2 * cn :] = np.sum(g_c_sel[:, :, 2 * cn :], axis=1, keepdims=True)
        if topo.qhd_param_idx.size > 0:
            phase_positions = tuple(idx3.T)
            g_qhd = np.real(g_nsv[phase_positions] * (1j * _DEG * nsv[phase_positions]))
            np.add.at(g_p_base[0], topo.qhd_param_idx, g_qhd)
            g_nsv[phase_positions] *= qhd_phases

    # 4. Backprop through signal conjugation mask
    if topo.signal_conj_mask.any():
        g_sig_aug[:, topo.signal_conj_mask] = np.conj(g_sig_aug[:, topo.signal_conj_mask])

    # 5. Backprop through carrier-signal coupling:
    #    sig_after[pos] = -sig_before[pos] * carrier[c_idx]
    if topo.signal_carrier_flat_pos.size > 0:
        pos = topo.signal_carrier_flat_pos
        c_idx = topo.signal_carrier_indices
        g_pos = g_sig_aug[:, pos]  # (b_sig, C_len)
        g_car_contrib = -g_pos * carrier_couplings
        if b_car == 1:
            np.add.at(g_carrier[0], c_idx, g_car_contrib.sum(axis=0))
        else:
            for bi in range(b_car):
                np.add.at(g_carrier[bi], c_idx, g_car_contrib[bi])
        g_sig_aug[:, pos] = -g_pos * carrier[:, c_idx]

    # 6. Backprop through signal component functions (sweep & static)
    _vjp_np_groups(g_sig_aug, p_sig, topo.signal_groups_sweep, g_p_sig, signal_sweep)
    if topo.signal_groups_static:
        g_sig_static = g_sig_aug.sum(axis=0, keepdims=True) if b_car == 1 else g_sig_aug
        _vjp_np_groups(g_sig_static, p_car, topo.signal_groups_static, g_p_car)

    # 7. Backprop through carrier SuperLU solve -> carrier component functions -> g_p_base
    g_car_aug = np.empty((b_car, nnz_c + cn), dtype=np.complex128)
    c_rows = topo.carrier_csc_rows
    c_cols = topo.carrier_csc_cols
    c_matrix = _CSCWorkspace(cn, topo.carrier_csc_indices, topo.carrier_csc_indptr)
    car_finite = np.isfinite(car_A).all(axis=1)
    for bi in range(b_car):
        lu_c = _factorize(c_matrix, car_A[bi]) if car_finite[bi] else None
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
