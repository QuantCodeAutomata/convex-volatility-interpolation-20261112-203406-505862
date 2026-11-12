"""CVI no-arbitrage and structural linear constraints (paper, Sections
"Linear Extrapolation" through "PDF >= 0 at the edge knots and beyond",
pp. 3-6).

All eight constraint types are assembled as rows of l <= A x <= u on the
stacked free-parameter vector x = (x_1, ..., x_m), x_j = (v(0), dv/dz(0),
c_1, ..., c_{n-2}) per expiry (same layout as src.cvi_objective). The two
linearized no-butterfly-arbitrage constraints (items 7 and 8) require a
reference solution (v_ref, s_ref, d1_ref, d2_ref); they are rebuilt at
each calibration iteration from the previous solution, per the paper's
two-step process (Section "Putting It All Together").
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.cvi_objective import ExpiryQuotes, VariableLayout, free_param_design_matrices
from src.qp_types import ConstraintSet


def _stack(blocks: list[ConstraintSet], n_var: int) -> ConstraintSet:
    blocks = [b for b in blocks if b.A.shape[0] > 0]
    if not blocks:
        return ConstraintSet(sp.csc_matrix((0, n_var)), np.array([]), np.array([]), [])
    A = sp.vstack([b.A for b in blocks], format="csc")
    l = np.concatenate([b.l for b in blocks])
    u = np.concatenate([b.u for b in blocks])
    names = sum([b.names for b in blocks], [])
    return ConstraintSet(A, l, u, names)


def _row(idx: np.ndarray, coeffs: np.ndarray, n_var: int) -> sp.csc_matrix:
    return sp.csc_matrix((coeffs, (np.zeros_like(idx), idx)), shape=(1, n_var))


def _design_at(z_knots: np.ndarray, z_points: np.ndarray):
    """V, D, C design matrices (n_points x n_free) at z_points."""
    return free_param_design_matrices(z_points, z_knots)


def positivity_of_variance(
    layout: VariableLayout, z_knots: np.ndarray, n_strikes: int, first_expiry_only: bool = True
) -> ConstraintSet:
    """v(z) >= 0 on n_strikes points in [z_0, z_{n-1}], first expiry only
    (paper: "subsequent expiries are bounded by calendar spread
    constraints")."""
    z_grid = np.linspace(z_knots[0], z_knots[-1], n_strikes)
    V, _, _ = _design_at(z_knots, z_grid)
    n_var = layout.total_size
    rows, names = [], []
    expiries_idx = [0] if first_expiry_only else range(layout.n_expiries)
    for j in expiries_idx:
        idx = np.arange(layout.spline_slices[j].start, layout.spline_slices[j].stop)
        for r in range(n_strikes):
            rows.append(_row(idx, V[r], n_var))
            names.append(f"variance_positivity_T{j}_k{r}")
    A = sp.vstack(rows, format="csc") if rows else sp.csc_matrix((0, n_var))
    l = np.zeros(A.shape[0])
    u = np.full(A.shape[0], np.inf)
    return ConstraintSet(A, l, u, names)


def positivity_in_tails(layout: VariableLayout, z_knots: np.ndarray) -> ConstraintSet:
    """dv/dz(z_0) <= 0 and dv/dz(z_{n-1}) >= 0, for every expiry (paper,
    Section "Positivity of the Variance in the Tails")."""
    edges = np.array([z_knots[0], z_knots[-1]])
    _, D, _ = _design_at(z_knots, edges)
    n_var = layout.total_size
    rows, l, u, names = [], [], [], []
    for j in range(layout.n_expiries):
        idx = np.arange(layout.spline_slices[j].start, layout.spline_slices[j].stop)
        rows.append(_row(idx, D[0], n_var)); l.append(-np.inf); u.append(0.0)
        names.append(f"tail_positivity_left_T{j}")
        rows.append(_row(idx, D[1], n_var)); l.append(0.0); u.append(np.inf)
        names.append(f"tail_positivity_right_T{j}")
    A = sp.vstack(rows, format="csc")
    return ConstraintSet(A, np.array(l), np.array(u), names)


def lee_tail_slope_bounds(
    layout: VariableLayout, z_knots: np.ndarray, v_star: np.ndarray, T: np.ndarray, margin: float
) -> ConstraintSet:
    """Lee's moment formula tail slope bounds (paper, Section "Lee's Tail
    Slope Bounds"): s(z_{n-1}) < 2/sqrt(v_star T) and s(z_0) > -2/sqrt(v_star T),
    enforced with a small numerical margin for strictness."""
    edges = np.array([z_knots[0], z_knots[-1]])
    _, D, _ = _design_at(z_knots, edges)
    n_var = layout.total_size
    rows, l, u, names = [], [], [], []
    for j in range(layout.n_expiries):
        idx = np.arange(layout.spline_slices[j].start, layout.spline_slices[j].stop)
        bound = 2.0 / np.sqrt(v_star[j] * T[j])
        # s = (dv/dz)/v_star => dv/dz <= v_star * bound - margin
        rows.append(_row(idx, D[1], n_var)); l.append(-np.inf); u.append(v_star[j] * bound - margin)
        names.append(f"lee_upper_T{j}")
        rows.append(_row(idx, D[0], n_var)); l.append(-v_star[j] * bound + margin); u.append(np.inf)
        names.append(f"lee_lower_T{j}")
    A = sp.vstack(rows, format="csc")
    return ConstraintSet(A, np.array(l), np.array(u), names)


def generic_no_calendar_spread(
    layout: VariableLayout,
    z_knots: np.ndarray,
    v_star: np.ndarray,
    T: np.ndarray,
    n_strikes: int,
) -> ConstraintSet:
    """v(K_i, T_j) T_j <= v(K_i F_{j+1}/F_j, T_{j+1}) T_{j+1} on a common
    absolute log-moneyness grid k_i (paper, Section "Generic
    No-Calendar-Spread-Arbitrage Constraints"). The grid is built from the
    first expiry's anchor volatility, matching the knot range in z-space;
    each expiry evaluates it at its own normalized coordinate
    z_{j,i} = k_i / (sigma_star_j sqrt(T_j))."""
    m = layout.n_expiries
    n_var = layout.total_size
    k_grid = np.linspace(z_knots[0], z_knots[-1], n_strikes) * v_star[0] ** 0.5 * T[0] ** 0.5
    rows, l, u, names = [], [], [], []
    for j in range(m - 1):
        z_j = k_grid / (np.sqrt(v_star[j]) * np.sqrt(T[j]))
        z_j1 = k_grid / (np.sqrt(v_star[j + 1]) * np.sqrt(T[j + 1]))
        V_j, _, _ = _design_at(z_knots, z_j)
        V_j1, _, _ = _design_at(z_knots, z_j1)
        idx_j = np.arange(layout.spline_slices[j].start, layout.spline_slices[j].stop)
        idx_j1 = np.arange(layout.spline_slices[j + 1].start, layout.spline_slices[j + 1].stop)
        for r in range(n_strikes):
            row = sp.csc_matrix(
                (
                    np.concatenate([T[j] * V_j[r], -T[j + 1] * V_j1[r]]),
                    (np.zeros(2 * layout.n_free, dtype=int), np.concatenate([idx_j, idx_j1])),
                ),
                shape=(1, n_var),
            )
            rows.append(row)
            l.append(-np.inf); u.append(0.0)
            names.append(f"calendar_spread_T{j}_T{j+1}_k{r}")
    A = sp.vstack(rows, format="csc") if rows else sp.csc_matrix((0, n_var))
    return ConstraintSet(A, np.array(l), np.array(u), names)


def no_calendar_spread_tails(
    layout: VariableLayout, z_knots: np.ndarray, v_star: np.ndarray, T: np.ndarray
) -> ConstraintSet:
    """No-calendar-spread in the tails (paper, Section
    "No-Calendar-Spread-Arbitrage Constraints in the Tails"):
    s(z_0,T_j) sqrt(v_star_j T_j) >= s(z_0,T_{j+1}) sqrt(v_star_{j+1} T_{j+1})
    s(z_{n-1},T_j) sqrt(v_star_j T_j) <= s(z_{n-1},T_{j+1}) sqrt(v_star_{j+1} T_{j+1})."""
    m = layout.n_expiries
    n_var = layout.total_size
    edges = np.array([z_knots[0], z_knots[-1]])
    _, D, _ = _design_at(z_knots, edges)
    rows, l, u, names = [], [], [], []
    for j in range(m - 1):
        idx_j = np.arange(layout.spline_slices[j].start, layout.spline_slices[j].stop)
        idx_j1 = np.arange(layout.spline_slices[j + 1].start, layout.spline_slices[j + 1].stop)
        # s(z,T_j) sqrt(v_star_j T_j) = (D@x_j / v_star_j) * sqrt(v_star_j T_j) = D@x_j * sqrt(T_j/v_star_j)
        scale_j = np.sqrt(T[j] / v_star[j])
        scale_j1 = np.sqrt(T[j + 1] / v_star[j + 1])
        row_left = sp.csc_matrix(
            (
                np.concatenate([scale_j * D[0], -scale_j1 * D[0]]),
                (np.zeros(2 * layout.n_free, dtype=int), np.concatenate([idx_j, idx_j1])),
            ),
            shape=(1, n_var),
        )
        rows.append(row_left); l.append(0.0); u.append(np.inf)
        names.append(f"calendar_spread_tail_left_T{j}_T{j+1}")

        row_right = sp.csc_matrix(
            (
                np.concatenate([scale_j * D[1], -scale_j1 * D[1]]),
                (np.zeros(2 * layout.n_free, dtype=int), np.concatenate([idx_j, idx_j1])),
            ),
            shape=(1, n_var),
        )
        rows.append(row_right); l.append(-np.inf); u.append(0.0)
        names.append(f"calendar_spread_tail_right_T{j}_T{j+1}")
    A = sp.vstack(rows, format="csc") if rows else sp.csc_matrix((0, n_var))
    return ConstraintSet(A, np.array(l), np.array(u), names)


def _d1_d2(k: np.ndarray, vT: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sqrt_vT = np.sqrt(vT)
    d1 = (-k + vT / 2.0) / sqrt_vT
    d2 = (-k - vT / 2.0) / sqrt_vT
    return d1, d2


def pdf_nonnegative_interior(
    layout: VariableLayout,
    z_knots: np.ndarray,
    v_star: np.ndarray,
    T: np.ndarray,
    n_strikes: int,
    reference: list[dict],
) -> ConstraintSet:
    """Linearized PDF >= 0 between z_0 and z_{n-1} (paper, Section "PDF >= 0
    between z_0 and z_{n-1}"): starting from the Martini-Mingone condition
    expressed in z-space,

        c >= (v_star/(2v)) s^2 + sqrt(v_star T) s
             - 2(1 + d1 sqrt(v_star/v) s + d1 d2 (v_star/(4v)) s^2),

    linearized around a reference solution (v_ref, s_ref) with d1, d2
    evaluated at the reference and held fixed:

        c >= beta_0 + beta_1 (s - s_ref) + beta_2 (v - v_ref).

    ``reference`` holds, per expiry, {"v": array, "s": array, "d1": array,
    "d2": array} evaluated at the same z_grid used here (produced by
    ``src.cvi_calibration`` from the previous iteration's solution).
    """
    m = layout.n_expiries
    n_var = layout.total_size
    z_grid = np.linspace(z_knots[0], z_knots[-1], n_strikes)
    V, D, C = _design_at(z_knots, z_grid)
    rows, l, u, names = [], [], [], []
    for j in range(m):
        idx = np.arange(layout.spline_slices[j].start, layout.spline_slices[j].stop)
        ref = reference[j]
        v_ref, s_ref, d1_ref, d2_ref = ref["v"], ref["s"], ref["d1"], ref["d2"]
        vstar = v_star[j]
        A_coef = vstar / 2.0 * (1.0 - d1_ref * d2_ref)
        # f(s, v) = A_coef * s^2 / v + sqrt(vstar*T) s - 2*d1_ref*sqrt(vstar)*s/sqrt(v) - 2
        beta1 = 2 * A_coef * s_ref / v_ref + np.sqrt(vstar * T[j]) - 2 * d1_ref * np.sqrt(vstar) / np.sqrt(v_ref)
        beta2 = -A_coef * s_ref**2 / v_ref**2 + d1_ref * np.sqrt(vstar) * s_ref * v_ref ** (-1.5)
        f_ref = A_coef * s_ref**2 / v_ref + np.sqrt(vstar * T[j]) * s_ref - 2 * d1_ref * np.sqrt(vstar) * s_ref / np.sqrt(v_ref) - 2
        beta0 = f_ref - beta1 * s_ref - beta2 * v_ref  # so that beta0 + beta1*s + beta2*v == f_ref at reference
        for r in range(n_strikes):
            # c/vstar - beta1*(D_row@x)/vstar - beta2*(V_row@x) >= beta0
            coeff = C[r] / vstar - beta1[r] / vstar * D[r] - beta2[r] * V[r]
            rows.append(_row(idx, coeff, n_var))
            l.append(beta0[r]); u.append(np.inf)
            names.append(f"pdf_interior_T{j}_k{r}")
    A = sp.vstack(rows, format="csc") if rows else sp.csc_matrix((0, n_var))
    return ConstraintSet(A, np.array(l), np.array(u), names)


def pdf_nonnegative_edges(
    layout: VariableLayout,
    z_knots: np.ndarray,
    v_star: np.ndarray,
    T: np.ndarray,
    reference: list[dict],
) -> ConstraintSet:
    """Linearized PDF >= 0 at and beyond the edge knots (paper, Section
    "PDF >= 0 at the edge knots and beyond"): where c = 0, the condition is
    quadratic in s with discriminant Delta = v_star (T + 4/v); evaluating
    its two roots at the reference solution's v gives a numeric bound on
    s that is then imposed directly as a linear constraint (the
    "linearization" the paper refers to: "Like (2), the two equations
    above must be linearized", i.e. v is frozen at its reference value)."""
    m = layout.n_expiries
    n_var = layout.total_size
    edges = np.array([z_knots[0], z_knots[-1]])
    _, D, _ = _design_at(z_knots, edges)
    rows, l, u, names = [], [], [], []
    for j in range(m):
        idx = np.arange(layout.spline_slices[j].start, layout.spline_slices[j].stop)
        ref = reference[j]
        vstar, Tj = v_star[j], T[j]
        for edge_name, edge_i, D_row in (("left", 0, D[0]), ("right", 1, D[1])):
            v_ref = ref["v_edge"][edge_i]
            k_edge = edges[edge_i] * np.sqrt(vstar) * np.sqrt(Tj)
            vT_ref = v_ref * Tj
            sqrt_term = np.sqrt(max(vT_ref * (1.0 + vT_ref / 4.0), 0.0))
            denom = k_edge**2 - vT_ref**2 / 4.0 - vT_ref
            if abs(denom) < 1e-12:
                continue  # degenerate: skip (reference too close to singularity)
            root_plus = (2 * v_ref * np.sqrt(Tj) * (k_edge + sqrt_term)) / (np.sqrt(vstar) * denom)
            root_minus = (2 * v_ref * np.sqrt(Tj) * (k_edge - sqrt_term)) / (np.sqrt(vstar) * denom)
            lower_root, upper_root = sorted([root_plus, root_minus])
            if edge_name == "right":
                # s(z_{n-1}) <= lower (positive) root
                bound = lower_root
                rows.append(_row(idx, D_row, n_var)); l.append(-np.inf); u.append(vstar * bound)
                names.append(f"pdf_edge_right_T{j}")
            else:
                # s(z_0) >= upper (negative) root
                bound = upper_root
                rows.append(_row(idx, D_row, n_var)); l.append(vstar * bound); u.append(np.inf)
                names.append(f"pdf_edge_left_T{j}")
    A = sp.vstack(rows, format="csc") if rows else sp.csc_matrix((0, n_var))
    return ConstraintSet(A, np.array(l), np.array(u), names)


def assemble_all_constraints(
    expiries: list[ExpiryQuotes],
    layout: VariableLayout,
    z_knots: np.ndarray,
    n_strikes: int,
    lee_margin: float,
    include_linearized_butterfly: bool,
    reference: list[dict] | None = None,
) -> ConstraintSet:
    """Assemble all applicable constraint blocks (paper, Section "Putting
    It All Together"). The two linearized butterfly blocks (items 7, 8)
    are included only from the second calibration iteration onward, once
    a reference solution is available."""
    v_star = np.array([e.v_star for e in expiries])
    T = np.array([e.T for e in expiries])
    blocks = [
        positivity_of_variance(layout, z_knots, n_strikes),
        positivity_in_tails(layout, z_knots),
        generic_no_calendar_spread(layout, z_knots, v_star, T, n_strikes),
        no_calendar_spread_tails(layout, z_knots, v_star, T),
        lee_tail_slope_bounds(layout, z_knots, v_star, T, lee_margin),
    ]
    if include_linearized_butterfly:
        assert reference is not None, "linearized butterfly constraints require a reference solution"
        blocks.append(pdf_nonnegative_interior(layout, z_knots, v_star, T, n_strikes, reference))
        blocks.append(pdf_nonnegative_edges(layout, z_knots, v_star, T, reference))
    return _stack(blocks, layout.total_size)
