"""Full CVI calibration: QP assembly and the two-step solve with Clarabel
(paper, Sections "Putting It All Together" and "Canonical QP Problem",
pp. 6-7).

The optimization problem is

    minimize    1/2 x^T P x + q^T x
    subject to  l <= A x <= u

where x stacks, for each expiry, the free cubic-spline parameters
(v(0), dv/dz(0), c_1, ..., c_{n-2}) plus the Above Ask / Below Bid /
Strike Regularization epigraph slacks (src.cvi_objective). The solve is
run twice per the paper's calibration procedure: first without the
linearized no-butterfly-arbitrage constraints, then with them, using the
first solve as the reference solution for the linearization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import clarabel
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix

from src.cvi_constraints import assemble_all_constraints
from src.cvi_objective import ExpiryQuotes, VariableLayout, assemble_objective, epigraph_constraints, \
    free_param_design_matrices
from src.qp_types import ConstraintSet


@dataclass
class CalibrationResult:
    x: np.ndarray
    layout: VariableLayout
    iterations_time: list[float]
    total_time: float
    n_nonzero_P: int
    n_nonzero_A: int
    n_constraints: int
    chi_squared_per_expiry: list[float]
    chi_squared_mean: float


def _solve_clarabel(
    P: csc_matrix, q: np.ndarray, constraints: ConstraintSet
) -> np.ndarray:
    """Solve the QP with the Clarabel interior-point solver (paper, Section
    "Canonical QP Problem": "we use the Clarabel interior-point solver").

    Clarabel expects one-sided cones; a two-sided row l <= a^T x <= u is
    split into a <= u (NonnegativeConeT on u - a^T x >= 0) and a >= l
    (on a^T x - l >= 0), following Clarabel's native ``solve`` API.
    """
    A = constraints.A.tocsc()
    l, u = constraints.l, constraints.u
    has_upper = np.isfinite(u)
    has_lower = np.isfinite(l)

    rows_A = []
    rows_b = []
    cones = []
    if np.any(has_upper):
        rows_A.append(A[has_upper])
        rows_b.append(u[has_upper])
        cones.append(clarabel.NonnegativeConeT(int(has_upper.sum())))
    if np.any(has_lower):
        rows_A.append(-A[has_lower])
        rows_b.append(-l[has_lower])
        cones.append(clarabel.NonnegativeConeT(int(has_lower.sum())))

    A_stacked = sp.vstack(rows_A, format="csc") if rows_A else sp.csc_matrix((0, P.shape[0]))
    b_stacked = np.concatenate(rows_b) if rows_b else np.array([])

    P_upper = sp.triu(P, format="csc")
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(P_upper, q, A_stacked, b_stacked, cones, settings)
    solution = solver.solve()
    if str(solution.status) not in ("Solved", "AlmostSolved", "SolverStatus.Solved", "SolverStatus.AlmostSolved"):
        raise RuntimeError(f"Clarabel did not converge: status={solution.status}")
    return np.array(solution.x)


def reference_from_solution(
    x: np.ndarray, expiries: list[ExpiryQuotes], layout: VariableLayout, z_knots: np.ndarray, n_strikes: int
) -> list[dict]:
    """Diagnostics (v, s, d1, d2 at the discretized interior grid, and v at
    the edge knots) needed to linearize the no-butterfly-arbitrage
    constraints for the next calibration iteration (paper: "the linearized
    constraints ... depend on a reference solution")."""
    z_grid = np.linspace(z_knots[0], z_knots[-1], n_strikes)
    V, D, _ = free_param_design_matrices(z_grid, z_knots)
    V_edge, D_edge, _ = free_param_design_matrices(np.array([z_knots[0], z_knots[-1]]), z_knots)
    reference = []
    for j, exp in enumerate(expiries):
        xj = x[layout.spline_slices[j]]
        v = V @ xj
        dv = D @ xj
        s = dv / exp.v_star
        k = z_grid * np.sqrt(exp.v_star) * np.sqrt(exp.T)
        vT = v * exp.T
        sqrt_vT = np.sqrt(np.maximum(vT, 1e-12))
        d1 = (-k + vT / 2.0) / sqrt_vT
        d2 = (-k - vT / 2.0) / sqrt_vT
        v_edge = V_edge @ xj
        reference.append({"v": v, "s": s, "d1": d1, "d2": d2, "v_edge": v_edge})
    return reference


def calibrate(
    expiries: list[ExpiryQuotes],
    z_knots: np.ndarray,
    lam: float,
    n_strikes: int,
    lee_margin: float,
    n_iterations: int = 2,
) -> CalibrationResult:
    """Two-step CVI calibration (paper, Section "Putting It All Together"):
    first solve without the linearized butterfly constraints, then
    rebuild them from that reference solution and solve again."""
    P, q, layout = assemble_objective(expiries, z_knots, lam)
    epi = epigraph_constraints(expiries, layout, z_knots)

    iter_times = []
    reference = None
    x = None
    last_constraints = None
    for it in range(n_iterations):
        t0 = time.perf_counter()
        base = assemble_all_constraints(
            expiries, layout, z_knots, n_strikes, lee_margin,
            include_linearized_butterfly=(it > 0), reference=reference,
        )
        A = sp.vstack([epi.A, base.A], format="csc")
        l = np.concatenate([epi.l, base.l])
        u = np.concatenate([epi.u, base.u])
        constraints = ConstraintSet(A, l, u, epi.names + base.names)
        x = _solve_clarabel(P, q, constraints)
        iter_times.append(time.perf_counter() - t0)
        last_constraints = constraints
        if it < n_iterations - 1:
            reference = reference_from_solution(x, expiries, layout, z_knots, n_strikes)

    chi2 = []
    for j, exp in enumerate(expiries):
        xj = x[layout.spline_slices[j]]
        z_mid = exp.z_mid
        mask = (z_mid >= z_knots[0]) & (z_mid <= z_knots[-1])
        if mask.sum() == 0:
            chi2.append(np.nan)
            continue
        V, _, _ = free_param_design_matrices(z_mid[mask], z_knots)
        v_fit = V @ xj
        v_mid = 0.5 * (exp.v_bid_mid[mask] + exp.v_ask_mid[mask])
        w = 1.0 / np.maximum(exp.v_ask_mid[mask] - exp.v_bid_mid[mask], 1e-12) ** 2
        chi2.append(float(np.sum(w * (v_fit - v_mid) ** 2) / mask.sum()))

    return CalibrationResult(
        x=x,
        layout=layout,
        iterations_time=iter_times,
        total_time=sum(iter_times),
        n_nonzero_P=P.nnz,
        n_nonzero_A=last_constraints.A.nnz,
        n_constraints=last_constraints.A.shape[0],
        chi_squared_per_expiry=chi2,
        chi_squared_mean=float(np.nanmean(chi2)),
    )
