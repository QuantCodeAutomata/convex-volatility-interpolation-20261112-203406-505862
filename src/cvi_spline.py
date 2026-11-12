"""CVI cubic-spline / cubic-B-spline dual parameterization of variance.

Implements the parameterization of Section "The CVI Parameterization" /
"CVI Cubic Spline" / "CVI B-Spline" (paper pp. 1-2). The variance
v(z) = sigma^2(z, T) is parameterized in normalized log-moneyness
z := k / (sigma_star * sqrt(T)), k = log(K/F), by either:

  * n + 2 cubic-spline parameters p = (v(0), dv/dz(0), c_0, ..., c_{n-1}),
    where c_i := d2v/dz2(z_i) are the "anchored convexities" at the n knots
    z_0 < ... < z_{n-1} (one of which, z_l, equals 0); or
  * n + 2 cubic B-spline weights w = (w_0, ..., w_{n+1}).

Both representations span the same (n + 2)-dimensional space of C^2
piecewise-cubic functions over the knots, so they are linked by a square
linear transformation w = M @ p. The paper describes this transformation
as "known" without writing it out explicitly; M is derived here from
standard spline theory via collocation at the B-spline Greville abscissae
(scipy.interpolate.BSpline).

Outside [z_0, z_{n-1}] both representations are linearly extrapolated,
consistent with setting d2v/dz2 = 0 at the edge knots (Section "CVI Cubic
Spline", p.2).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

DEGREE = 3  # cubic


def _check_knots(z_knots: np.ndarray) -> np.ndarray:
    z_knots = np.asarray(z_knots, dtype=float)
    if z_knots.ndim != 1 or z_knots.size < 3:
        raise ValueError("z_knots must be a 1-D array with at least 3 knots")
    if np.any(np.diff(z_knots) <= 0):
        raise ValueError("z_knots must be strictly increasing")
    return z_knots


def cubic_spline_knot_values(
    z_knots: np.ndarray, v0: float, d0: float, c: np.ndarray, l: int
) -> tuple[np.ndarray, np.ndarray]:
    """Value and first derivative at every knot, by marching from z_l = 0.

    Within segment [z_i, z_{i+1}] the second derivative is, by
    construction, the linear interpolation of the anchored convexities
    c_i, c_{i+1} (paper, relation (a): "the second derivative of the cubic
    spline should match the convexities ... at each knot"). Integrating
    this linear function twice from the known value/derivative at one
    endpoint gives the value/derivative at the other endpoint; marching
    outward from z_l = 0 (where v=v0, v'=d0 are given directly) produces
    v, v' at every knot without needing to solve the full 4(n+1)x4(n+1)
    system explicitly.
    """
    z_knots = _check_knots(z_knots)
    n = z_knots.size
    c = np.asarray(c, dtype=float)
    if c.shape != (n,):
        raise ValueError("c must have one convexity per knot")
    if not np.isclose(z_knots[l], 0.0):
        raise ValueError("z_knots[l] must equal 0 (the ATM knot)")

    v_knots = np.empty(n)
    d_knots = np.empty(n)
    v_knots[l] = v0
    d_knots[l] = d0

    # March to the right of the ATM knot: z_{i+1} from z_i.
    for i in range(l, n - 1):
        h = z_knots[i + 1] - z_knots[i]
        d_knots[i + 1] = d_knots[i] + h * (c[i] + c[i + 1]) / 2.0
        v_knots[i + 1] = (
            v_knots[i] + d_knots[i] * h + h * h * (2.0 * c[i] + c[i + 1]) / 6.0
        )

    # March to the left of the ATM knot: z_i from z_{i+1}.
    for i in range(l - 1, -1, -1):
        h = z_knots[i + 1] - z_knots[i]
        d_knots[i] = d_knots[i + 1] - h * (c[i] + c[i + 1]) / 2.0
        v_knots[i] = (
            v_knots[i + 1] - d_knots[i] * h - h * h * (2.0 * c[i] + c[i + 1]) / 6.0
        )
    return v_knots, d_knots


def evaluate_cubic_spline(
    z_eval: np.ndarray,
    z_knots: np.ndarray,
    v0: float,
    d0: float,
    c: np.ndarray,
    l: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate v(z), dv/dz(z), d2v/dz2(z) for the CVI cubic spline.

    For z outside [z_knots[0], z_knots[-1]] the function is linearly
    extrapolated using the value and slope at the nearest edge knot
    (paper: "the variance is linearly extrapolated in log-strike in the
    tails").
    """
    z_knots = _check_knots(z_knots)
    n = z_knots.size
    if l is None:
        l = int(np.argmin(np.abs(z_knots)))
    v_knots, d_knots = cubic_spline_knot_values(z_knots, v0, d0, c, l)
    c = np.asarray(c, dtype=float)

    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))
    v = np.empty_like(z_eval)
    dv = np.empty_like(z_eval)
    d2v = np.empty_like(z_eval)

    # Segment index for each point: i such that z_knots[i] <= z < z_knots[i+1].
    seg = np.clip(np.searchsorted(z_knots, z_eval, side="right") - 1, 0, n - 2)

    left_tail = z_eval < z_knots[0]
    right_tail = z_eval > z_knots[-1]
    interior = ~(left_tail | right_tail)

    for i in np.unique(seg[interior]):
        mask = interior & (seg == i)
        u = z_eval[mask] - z_knots[i]
        h = z_knots[i + 1] - z_knots[i]
        v[mask] = (
            v_knots[i]
            + d_knots[i] * u
            + 0.5 * c[i] * u**2
            + (c[i + 1] - c[i]) / (6.0 * h) * u**3
        )
        dv[mask] = d_knots[i] + c[i] * u + (c[i + 1] - c[i]) / (2.0 * h) * u**2
        d2v[mask] = c[i] + (c[i + 1] - c[i]) * u / h

    if np.any(left_tail):
        u = z_eval[left_tail] - z_knots[0]
        v[left_tail] = v_knots[0] + d_knots[0] * u
        dv[left_tail] = d_knots[0]
        d2v[left_tail] = 0.0
    if np.any(right_tail):
        u = z_eval[right_tail] - z_knots[-1]
        v[right_tail] = v_knots[-1] + d_knots[-1] * u
        dv[right_tail] = d_knots[-1]
        d2v[right_tail] = 0.0
    return v, dv, d2v


def clamped_knot_vector(z_knots: np.ndarray) -> np.ndarray:
    """Clamped cubic knot vector: boundary knots repeated degree+1 times.

    For n simple knots this yields n + 2*DEGREE + 2 - ... giving exactly
    n + 2 cubic B-spline basis functions over the domain [z_0, z_{n-1}]
    (paper: "the n + 2 B-splines associated with the n knots").
    """
    z_knots = _check_knots(z_knots)
    pad_left = np.full(DEGREE + 1, z_knots[0])
    pad_right = np.full(DEGREE + 1, z_knots[-1])
    return np.concatenate([pad_left, z_knots[1:-1], pad_right])


def n_bspline_basis(z_knots: np.ndarray) -> int:
    t = clamped_knot_vector(z_knots)
    return len(t) - DEGREE - 1


def greville_abscissae(t: np.ndarray, degree: int = DEGREE) -> np.ndarray:
    """Greville abscissae: standard, well-conditioned B-spline collocation
    points (average of degree consecutive interior knots per basis
    function)."""
    n_basis = len(t) - degree - 1
    return np.array([t[i + 1 : i + degree + 1].mean() for i in range(n_basis)])


def bspline_basis_matrix(z_eval: np.ndarray, z_knots: np.ndarray) -> np.ndarray:
    """Design matrix B[a, i] = B_i(z_eval[a]) of the clamped cubic
    B-splines, evaluated strictly inside [z_0, z_{n-1}] (scipy.interpolate
    .BSpline, as recommended in the paper's implementation notes)."""
    t = clamped_knot_vector(z_knots)
    n_basis = n_bspline_basis(z_knots)
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))
    basis = np.zeros((z_eval.size, n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spline = BSpline(t, coeffs, DEGREE, extrapolate=False)
        basis[:, i] = np.nan_to_num(spline(z_eval), nan=0.0)
    return basis


def build_transformation_matrix(z_knots: np.ndarray) -> np.ndarray:
    """Linear map M (shape (n+2, n+2)) such that w = M @ p, with
    p = (v(0), dv/dz(0), c_0, ..., c_{n-1}) the cubic-spline parameters and
    w the cubic B-spline weights representing the same function.

    Derivation (the paper leaves the explicit transform undescribed,
    calling it only "known"): both parameterizations span the same (n+2)
    dimensional space of C^2 piecewise-cubic functions over the knots.
    M is found by evaluating the cubic-spline representation of each
    parameter basis vector e_i at the B-spline's Greville collocation
    points and solving the resulting square linear system for the
    corresponding B-spline weights, i.e. by evaluating both
    representations at sufficient collocation points and solving a
    linear system.
    """
    z_knots = _check_knots(z_knots)
    n = z_knots.size
    n_basis = n_bspline_basis(z_knots)
    assert n_basis == n + 2

    l = int(np.argmin(np.abs(z_knots)))
    t = clamped_knot_vector(z_knots)
    greville = greville_abscissae(t)
    # Clip strictly inside the domain to avoid extrapolate=False edge NaNs.
    eps = 1e-9 * (z_knots[-1] - z_knots[0])
    greville_clipped = np.clip(greville, z_knots[0] + eps, z_knots[-1] - eps)

    collocation = bspline_basis_matrix(greville_clipped, z_knots)

    n_params = n + 2
    g_matrix = np.zeros((n_basis, n_params))
    for col in range(n_params):
        p = np.zeros(n_params)
        p[col] = 1.0
        v0, d0, c = p[0], p[1], p[2:]
        v_vals, _, _ = evaluate_cubic_spline(greville_clipped, z_knots, v0, d0, c, l)
        g_matrix[:, col] = v_vals

    m_matrix = np.linalg.solve(collocation, g_matrix)
    return m_matrix


def evaluate_bspline(
    z_eval: np.ndarray, z_knots: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the cubic B-spline representation v(z) = sum_i w_i B_i(z),
    linearly extrapolated outside [z_0, z_{n-1}]."""
    z_knots = _check_knots(z_knots)
    t = clamped_knot_vector(z_knots)
    w = np.asarray(w, dtype=float)
    spline = BSpline(t, w, DEGREE, extrapolate=False)
    d1 = spline.derivative(1)
    d2 = spline.derivative(2)

    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))
    v = np.empty_like(z_eval)
    dv = np.empty_like(z_eval)
    d2v = np.empty_like(z_eval)

    left_tail = z_eval < z_knots[0]
    right_tail = z_eval > z_knots[-1]
    interior = ~(left_tail | right_tail)

    v[interior] = spline(z_eval[interior])
    dv[interior] = d1(z_eval[interior])
    d2v[interior] = d2(z_eval[interior])

    if np.any(left_tail):
        v0, dv0 = spline(z_knots[0]), d1(z_knots[0])
        v[left_tail] = v0 + dv0 * (z_eval[left_tail] - z_knots[0])
        dv[left_tail] = dv0
        d2v[left_tail] = 0.0
    if np.any(right_tail):
        v1, dv1 = spline(z_knots[-1]), d1(z_knots[-1])
        v[right_tail] = v1 + dv1 * (z_eval[right_tail] - z_knots[-1])
        dv[right_tail] = dv1
        d2v[right_tail] = 0.0
    return v, dv, d2v
