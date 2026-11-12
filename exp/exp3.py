"""exp_3: No-arbitrage constraints end-to-end exercise and validation.

Exercises the already-implemented CVI no-arbitrage constraints (paper,
Sections "Linear Extrapolation", "Positivity of the Variance", "Positivity
of the Variance in the Tails", "No-Calendar-Spread-Arbitrage Constraints",
"Lee's Tail Slope Bounds", "No-Butterfly-Arbitrage Constraints", "Putting
It All Together", pp. 3-6), as built by ``src.cvi_constraints`` and used by
``src.cvi_calibration``'s two-step ``calibrate()``, on a synthetic
multi-expiry surface. This module does not reimplement any constraint; it
calibrates, then post-hoc verifies the fitted solution against the paper's
no-arbitrage conditions.

Run with::

    python -m exp.exp3
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.generate_synthetic import make_synthetic_surface
from src import config
from src.cvi_calibration import CalibrationResult, calibrate
from src.cvi_objective import ExpiryQuotes, VariableLayout, free_param_design_matrices

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Number of fine interior points used for the unlinearized PDF >= 0 check
# (step 4e). Not specified in the paper for this diagnostic; chosen dense
# enough to detect localized violations between the discretized
# constraint points.
PDF_CHECK_N_POINTS = 101
# Small inward margin (in z) kept away from the exact edge knots when
# evaluating the strictly-interior PDF condition (paper, Section "PDF >= 0
# between z_0 and z_{n-1}").
PDF_CHECK_EDGE_MARGIN = 1e-3


def build_symmetric_knots(n: int, edge: float) -> np.ndarray:
    """Symmetric knot vector of exactly ``n`` knots (odd) spanning
    [-edge, edge] with zero as one of the knots (the ATM knot)."""
    assert n % 2 == 1, "use an odd knot count so 0 is exactly a knot"
    half = np.linspace(0.0, edge, n // 2 + 1)
    return np.concatenate([-half[-1:0:-1], half])


def _compute_d1_d2(z: np.ndarray, v_star: float, T: float, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Black-Scholes d1, d2 in normalized log-moneyness z-space (paper's
    notation, matching ``src.cvi_calibration.reference_from_solution`` and
    ``src.cvi_constraints._d1_d2``): k = z * sigma_star * sqrt(T),
    vT = v * T, d1 = (-k + vT/2)/sqrt(vT), d2 = (-k - vT/2)/sqrt(vT).
    ``v`` may be a scalar (anchor variance v_star) or an array (fitted
    variance curve)."""
    z = np.asarray(z, dtype=float)
    v = np.asarray(v, dtype=float)
    k = z * np.sqrt(v_star) * np.sqrt(T)
    vT = v * T
    sqrt_vT = np.sqrt(np.maximum(vT, 1e-300))
    d1 = (-k + vT / 2.0) / sqrt_vT
    d2 = (-k - vT / 2.0) / sqrt_vT
    return d1, d2


def check_edge_d1_d2(expiries: list[ExpiryQuotes], z_knots: np.ndarray) -> list[dict[str, Any]]:
    """Paper, Section "PDF >= 0 at the edge knots": d1*d2 > 1 at both edge
    knots is required for the edge butterfly condition (discriminant of
    the quadratic-in-s condition) to apply. Evaluated using the anchor
    variance v_star (no fitted solution exists yet at this stage)."""
    edge_z = np.array([z_knots[0], z_knots[-1]])
    checks = []
    for exp in expiries:
        d1, d2 = _compute_d1_d2(edge_z, exp.v_star, exp.T, exp.v_star)
        prod = d1 * d2
        passed = bool(prod[0] > 1.0 and prod[1] > 1.0)
        checks.append(
            {
                "T": exp.T,
                "v_star": exp.v_star,
                "d1_d2_left": float(prod[0]),
                "d1_d2_right": float(prod[1]),
                "pass": passed,
            }
        )
    return checks


def check_linear_extrapolation(
    x: np.ndarray, layout: VariableLayout, z_knots: np.ndarray
) -> dict[str, Any]:
    """Paper, Section "Linear Extrapolation": d2v/dz2 is structurally fixed
    to zero at both edge knots (the free-parameter mapping fixes c_0 =
    c_{n-1} = 0). Verified here by evaluating C @ x_j at the two edge
    knots for every expiry."""
    edges = np.array([z_knots[0], z_knots[-1]])
    _, _, C = free_param_design_matrices(edges, z_knots)
    max_abs = 0.0
    per_expiry = []
    for j in range(layout.n_expiries):
        xj = x[layout.spline_slices[j]]
        c_edges = C @ xj
        max_abs = max(max_abs, float(np.max(np.abs(c_edges))))
        per_expiry.append({"expiry": j, "c_left": float(c_edges[0]), "c_right": float(c_edges[1])})
    passed = max_abs < 1e-10
    assert passed, f"d2v/dz2 not structurally zero at edge knots: max|c| = {max_abs:.3e}"
    return {"max_abs_c_at_edges": max_abs, "pass": passed, "per_expiry": per_expiry}


def check_positivity_of_variance(
    x: np.ndarray, expiries: list[ExpiryQuotes], layout: VariableLayout, z_knots: np.ndarray, n_strikes: int
) -> dict[str, Any]:
    """Paper, Sections "Positivity of the Variance": the linear constraint
    only directly binds the first expiry; here we check v(z) >= 0 on the
    same discretization grid for EVERY expiry, to see whether positivity
    still holds through the calendar-spread chain."""
    z_grid = np.linspace(z_knots[0], z_knots[-1], n_strikes)
    V, _, _ = free_param_design_matrices(z_grid, z_knots)
    per_expiry = []
    total_violations = 0
    max_violation = 0.0
    for j, exp in enumerate(expiries):
        xj = x[layout.spline_slices[j]]
        v = V @ xj
        bad = v < -1e-6
        n_bad = int(bad.sum())
        total_violations += n_bad
        worst = float((-v[bad]).max()) if n_bad > 0 else 0.0
        max_violation = max(max_violation, worst)
        per_expiry.append({"expiry": j, "T": exp.T, "n_violations": n_bad, "max_violation": worst})
    return {
        "total_violations": total_violations,
        "max_violation": max_violation,
        "per_expiry": per_expiry,
    }


def check_calendar_spread(
    x: np.ndarray, expiries: list[ExpiryQuotes], layout: VariableLayout, z_knots: np.ndarray, n_strikes: int
) -> dict[str, Any]:
    """Paper, Section "No-Calendar-Spread-Arbitrage Constraints": reuses
    the exact absolute log-moneyness grid construction of
    ``src.cvi_constraints.generic_no_calendar_spread`` (same formula, read
    from that function) to check v(K_i,T_j) T_j <= v(K_i,T_{j+1}) T_{j+1}
    on the fitted solution."""
    v_star = np.array([e.v_star for e in expiries])
    T = np.array([e.T for e in expiries])
    k_grid = np.linspace(z_knots[0], z_knots[-1], n_strikes) * v_star[0] ** 0.5 * T[0] ** 0.5
    pairs = []
    total_violations = 0
    max_violation = 0.0
    for j in range(len(expiries) - 1):
        z_j = k_grid / (np.sqrt(v_star[j]) * np.sqrt(T[j]))
        z_j1 = k_grid / (np.sqrt(v_star[j + 1]) * np.sqrt(T[j + 1]))
        V_j, _, _ = free_param_design_matrices(z_j, z_knots)
        V_j1, _, _ = free_param_design_matrices(z_j1, z_knots)
        xj = x[layout.spline_slices[j]]
        xj1 = x[layout.spline_slices[j + 1]]
        lhs = (V_j @ xj) * T[j]
        rhs = (V_j1 @ xj1) * T[j + 1]
        viol = lhs - rhs
        bad = viol > 1e-6
        n_bad = int(bad.sum())
        total_violations += n_bad
        worst = float(viol[bad].max()) if n_bad > 0 else 0.0
        max_violation = max(max_violation, worst)
        pairs.append({"pair": [j, j + 1], "n_violations": n_bad, "max_violation": worst})
    return {"total_violations": total_violations, "max_violation": max_violation, "per_pair": pairs}


def check_lee_bounds(
    x: np.ndarray, expiries: list[ExpiryQuotes], layout: VariableLayout, z_knots: np.ndarray, margin: float
) -> dict[str, Any]:
    """Paper, Section "Lee's Tail Slope Bounds": |s| < 2/sqrt(v_star T) at
    both edge knots, enforced with ``margin``."""
    edges = np.array([z_knots[0], z_knots[-1]])
    _, D, _ = free_param_design_matrices(edges, z_knots)
    per_expiry = []
    total_violations = 0
    max_violation = 0.0
    for j, exp in enumerate(expiries):
        xj = x[layout.spline_slices[j]]
        dv = D @ xj
        s = dv / exp.v_star
        bound = 2.0 / np.sqrt(exp.v_star * exp.T) - margin
        excess = np.abs(s) - bound
        bad = excess > 1e-9
        n_bad = int(bad.sum())
        total_violations += n_bad
        worst = float(excess[bad].max()) if n_bad > 0 else 0.0
        max_violation = max(max_violation, worst)
        per_expiry.append(
            {
                "expiry": j,
                "T": exp.T,
                "s_left": float(s[0]),
                "s_right": float(s[1]),
                "bound": float(bound),
                "n_violations": n_bad,
                "max_violation": worst,
            }
        )
    return {"total_violations": total_violations, "max_violation": max_violation, "per_expiry": per_expiry}


def check_pdf_condition(
    x: np.ndarray, expiries: list[ExpiryQuotes], layout: VariableLayout, z_knots: np.ndarray
) -> dict[str, Any]:
    """Paper, Section "PDF >= 0 between z_0 and z_{n-1}" (Martini-Mingone
    condition, unlinearized): c >= (v_star/(2v)) s^2 + sqrt(v_star T) s
    - 2(1 + d1 sqrt(v_star/v) s + d1 d2 (v_star/(4v)) s^2), evaluated on a
    fine interior grid of z points, using the fitted v, s = (dv/dz)/v_star,
    c = (d2v/dz2)/v_star, and d1, d2 computed from the fitted v."""
    z_interior = np.linspace(
        z_knots[0] + PDF_CHECK_EDGE_MARGIN, z_knots[-1] - PDF_CHECK_EDGE_MARGIN, PDF_CHECK_N_POINTS
    )
    V, D, C = free_param_design_matrices(z_interior, z_knots)
    per_expiry = []
    total_violations = 0
    max_violation = 0.0
    for j, exp in enumerate(expiries):
        xj = x[layout.spline_slices[j]]
        v = V @ xj
        dv = D @ xj
        d2v = C @ xj
        vstar = exp.v_star
        T = exp.T
        s = dv / vstar
        c = d2v / vstar

        v_safe = np.maximum(v, 1e-10)
        d1, d2 = _compute_d1_d2(z_interior, vstar, T, v_safe)
        rhs = (
            (vstar / (2.0 * v_safe)) * s**2
            + np.sqrt(vstar * T) * s
            - 2.0
            * (
                1.0
                + d1 * np.sqrt(vstar / v_safe) * s
                + d1 * d2 * (vstar / (4.0 * v_safe)) * s**2
            )
        )
        lhs = c
        bad = (lhs < rhs - 1e-9) | (v <= 1e-10)
        n_bad = int(bad.sum())
        total_violations += n_bad
        diff = rhs - lhs
        worst = float(diff[bad].max()) if n_bad > 0 else 0.0
        max_violation = max(max_violation, worst)
        per_expiry.append({"expiry": j, "T": exp.T, "n_violations": n_bad, "max_violation": worst})
    return {"total_violations": total_violations, "max_violation": max_violation, "per_expiry": per_expiry}


def plot_surface(
    result: CalibrationResult, expiries: list[ExpiryQuotes], z_knots: np.ndarray, path: Path
) -> None:
    """Plot v(z) for three representative expiries (shortest, middle,
    longest T) over the knot range extended ~2 units beyond each edge
    knot, to visually confirm linear extrapolation in the tails (paper,
    Section "Linear Extrapolation")."""
    n = len(expiries)
    rep_idx = sorted({0, n // 2, n - 1})
    z_plot = np.linspace(z_knots[0] - 2.0, z_knots[-1] + 2.0, 400)
    V, _, _ = free_param_design_matrices(z_plot, z_knots)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for j in rep_idx:
        xj = result.x[result.layout.spline_slices[j]]
        v = V @ xj
        ax.plot(z_plot, v, label=f"T={expiries[j].T:.3f}")
    ax.axvline(z_knots[0], color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(z_knots[-1], color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Normalized log-moneyness z")
    ax.set_ylabel("Fitted annualized variance v(z)")
    ax.set_title("CVI fitted smile, incl. linear-extrapolated tails (exp3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    try:
        # Step 1: knots.
        z_knots = build_symmetric_knots(config.EXP3_N_KNOTS, config.EXP3_EDGE_Z)

        # Step 2: synthetic multi-expiry surface + edge d1*d2 check.
        expiries = make_synthetic_surface(
            n_expiries=config.EXP3_N_EXPIRIES,
            z_range=(-config.EXP3_EDGE_Z + 0.2, config.EXP3_EDGE_Z - 0.2),
            seed=config.SYNTHETIC_SEED,
        )
        edge_checks = check_edge_d1_d2(expiries, z_knots)
        summary["expiries"] = [{"T": e.T, "v_star": e.v_star} for e in expiries]
        summary["edge_d1_d2"] = edge_checks
        all_edges_pass = all(c["pass"] for c in edge_checks)
        summary["edge_d1_d2_all_pass"] = all_edges_pass
        if not all_edges_pass:
            print("WARNING: d1*d2 > 1 fails at an edge knot for at least one expiry:")
            for c in edge_checks:
                if not c["pass"]:
                    print(f"  T={c['T']:.4f}: d1_d2_left={c['d1_d2_left']:.4f}, d1_d2_right={c['d1_d2_right']:.4f}")

        # Step 3: full two-step calibration.
        result = calibrate(
            expiries,
            z_knots,
            config.STRIKE_REG_LAMBDA,
            config.N_DISCRETIZATION_STRIKES,
            config.LEE_BOUND_MARGIN,
            config.N_CALIBRATION_ITERATIONS,
        )
        summary["calibration"] = {
            "n_constraints": result.n_constraints,
            "n_nonzero_P": result.n_nonzero_P,
            "n_nonzero_A": result.n_nonzero_A,
            "iterations_time": result.iterations_time,
            "total_time": result.total_time,
            "chi_squared_per_expiry": result.chi_squared_per_expiry,
            "chi_squared_mean": result.chi_squared_mean,
        }

        # Step 4a: linear extrapolation (structural c=0 at edge knots).
        summary["linear_extrapolation"] = check_linear_extrapolation(result.x, result.layout, z_knots)

        # Step 4b: positivity of variance, every expiry.
        summary["positivity_of_variance"] = check_positivity_of_variance(
            result.x, expiries, result.layout, z_knots, config.N_DISCRETIZATION_STRIKES
        )

        # Step 4c: calendar-spread check.
        summary["calendar_spread"] = check_calendar_spread(
            result.x, expiries, result.layout, z_knots, config.N_DISCRETIZATION_STRIKES
        )

        # Step 4d: Lee bound check.
        summary["lee_bounds"] = check_lee_bounds(result.x, expiries, result.layout, z_knots, config.LEE_BOUND_MARGIN)

        # Step 4e: butterfly/PDF check, before vs. after linearization.
        result_iter1 = calibrate(
            expiries,
            z_knots,
            config.STRIKE_REG_LAMBDA,
            config.N_DISCRETIZATION_STRIKES,
            config.LEE_BOUND_MARGIN,
            n_iterations=1,
        )
        pdf_before = check_pdf_condition(result_iter1.x, expiries, result_iter1.layout, z_knots)
        pdf_after = check_pdf_condition(result.x, expiries, result.layout, z_knots)
        summary["pdf_condition"] = {"before_linearization": pdf_before, "after_linearization": pdf_after}

        # Step 5: write JSON summary.
        summary_path = RESULTS_DIR / "exp3_arbitrage_checks.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Step 6: surface plot.
        plot_path = RESULTS_DIR / "exp3_surface_plot.png"
        plot_surface(result, expiries, z_knots, plot_path)

        # Step 7: human-readable report.
        print("=== exp3: No-arbitrage constraints ===")
        print(f"Expiries (T, v_star): {[(round(e.T, 4), round(e.v_star, 5)) for e in expiries]}")
        print(f"Edge d1*d2 > 1 for all expiries: {all_edges_pass}")
        print(
            f"Calibration: n_constraints={result.n_constraints}, "
            f"n_nonzero_P={result.n_nonzero_P}, n_nonzero_A={result.n_nonzero_A}, "
            f"total_time={result.total_time:.3f}s"
        )
        print(
            f"chi_squared_per_expiry={['%.4f' % c for c in result.chi_squared_per_expiry]}, "
            f"chi_squared_mean={result.chi_squared_mean:.4f}"
        )
        le = summary["linear_extrapolation"]
        print(f"Linear extrapolation: max|c| at edge knots = {le['max_abs_c_at_edges']:.3e} (pass={le['pass']})")
        pv = summary["positivity_of_variance"]
        print(f"Positivity of variance (all expiries): {pv['total_violations']} violations, max={pv['max_violation']:.3e}")
        cs = summary["calendar_spread"]
        print(f"Calendar-spread: {cs['total_violations']} violations, max={cs['max_violation']:.3e}")
        lb = summary["lee_bounds"]
        print(f"Lee tail-slope bound: {lb['total_violations']} violations, max={lb['max_violation']:.3e}")
        print(
            f"PDF/butterfly condition: before linearization = "
            f"{pdf_before['total_violations']} violations (max={pdf_before['max_violation']:.3e}); "
            f"after linearization = {pdf_after['total_violations']} violations (max={pdf_after['max_violation']:.3e})"
        )
        print(f"Summary written to {summary_path}")
        print(f"Plot written to {plot_path}")
    except Exception:
        print("exp3 FAILED with an exception:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
