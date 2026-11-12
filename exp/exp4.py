"""Experiment 4: full CVI calibration on live option chains.

Paper, Section "Numerical Results" (p.6) and "Scaling" (p.7): the complete
QP (objective, src.cvi_objective, + all constraints, src.cvi_constraints)
is assembled and solved with the Clarabel interior-point solver
(src.cvi_calibration.calibrate) for SPX and SX5E surfaces, benchmarking
calibration time, chi-squared, and the number of non-zero coefficients in
P and A as the number of knots per expiry grows from 10 to 40 (Table 3).

SX5E is not served by the Massive REST API and SPX contracts carry only
zero-zero NBBO quotes on this entitlement (both confirmed empirically,
src.market_surface, results/RESULTS.md), so this replication fits live
NDX and NVDA option chains instead -- underlyings the paper itself
benchmarks in its knot-convergence figure (Section "Scaling", Figure 6).

The surfaces available on this entitlement are small enough (<= 20
expiries each) that the paper's full knot grid, 10 to 40 per expiry,
completes in about a minute; ``--full`` is therefore a no-op kept only so
the command line matches the other experiments.

Run with::

    python -m exp.exp4
    python -m exp.exp4 --full
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.cvi_calibration import CalibrationResult, calibrate
from src.cvi_objective import ExpiryQuotes, free_param_design_matrices
from src.market_surface import load_market_surface

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Edge-knot placement in normalized log-moneyness, as in exp_2/exp_3: far
# enough in the tails that d1*d2 > 1 at the edge knots for every expiry
# tested here (required by "PDF >= 0 at the edge knots and beyond").
EDGE_Z = 6.0


def build_nm_dense_knots(n: int, edge: float, power: float = 2.2) -> np.ndarray:
    """Knots concentrated near-the-money, symmetric, 0 included (paper,
    "Numerical Results": knot placement with "increasing density near the
    money"; the exact density law is not specified -- a power-law warp of
    an evenly spaced grid in [0, 1], as in exp_1/exp_1's
    ``build_nm_dense_knots``)."""
    assert n % 2 == 1, "use an odd knot count so 0 is exactly a knot"
    t = np.linspace(0.0, 1.0, n // 2 + 1)
    half = edge * t**power
    return np.concatenate([-half[-1:0:-1], half])


def _check_edge_d1_d2(expiries: list[ExpiryQuotes], z_knots: np.ndarray) -> bool:
    """Paper, "PDF >= 0 at the edge knots": d1*d2 > 1 at both edge knots,
    evaluated at the anchor variance v_star (no fit exists yet)."""
    edge_z = np.array([z_knots[0], z_knots[-1]])
    for exp in expiries:
        k = edge_z * np.sqrt(exp.v_star) * np.sqrt(exp.T)
        vT = exp.v_star * exp.T
        d1 = (-k + vT / 2.0) / np.sqrt(vT)
        d2 = (-k - vT / 2.0) / np.sqrt(vT)
        if not np.all(d1 * d2 > 1.0):
            return False
    return True


def _post_hoc_arbitrage_checks(
    result: CalibrationResult, expiries: list[ExpiryQuotes], z_knots: np.ndarray, n_strikes: int
) -> dict[str, Any]:
    """Positivity of variance (every expiry) and generic no-calendar-spread
    (adjacent expiry pairs) on the fitted solution (paper, Sections
    "Positivity of the Variance" and "No-Calendar-Spread-Arbitrage
    Constraints"), as a compact post-hoc arbitrage-free check."""
    z_grid = np.linspace(z_knots[0], z_knots[-1], n_strikes)
    V, _, _ = free_param_design_matrices(z_grid, z_knots)
    neg_violations = 0
    for j in range(len(expiries)):
        xj = result.x[result.layout.spline_slices[j]]
        neg_violations += int(np.sum(V @ xj < -1e-6))

    v_star = np.array([e.v_star for e in expiries])
    T = np.array([e.T for e in expiries])
    k_grid = z_grid * np.sqrt(v_star[0]) * np.sqrt(T[0])
    cal_violations = 0
    for j in range(len(expiries) - 1):
        z_j = k_grid / (np.sqrt(v_star[j]) * np.sqrt(T[j]))
        z_j1 = k_grid / (np.sqrt(v_star[j + 1]) * np.sqrt(T[j + 1]))
        V_j, _, _ = free_param_design_matrices(z_j, z_knots)
        V_j1, _, _ = free_param_design_matrices(z_j1, z_knots)
        lhs = (V_j @ result.x[result.layout.spline_slices[j]]) * T[j]
        rhs = (V_j1 @ result.x[result.layout.spline_slices[j + 1]]) * T[j + 1]
        cal_violations += int(np.sum(lhs - rhs > 1e-6))
    return {"negative_variance_violations": neg_violations, "calendar_spread_violations": cal_violations}


def run_underlying(underlying: str, knot_counts: tuple[int, ...]) -> dict[str, Any]:
    expiries, data_diag = load_market_surface(
        underlying, config.EXP4_N_EXPIRIES_TARGET, config.EXP4_N_STRIKES_TARGET
    )
    print(
        f"{underlying}: loaded {len(expiries)} expiries, "
        f"{data_diag['n_quotes_total']} quotes, spot={data_diag['spot']}"
    )

    rows = []
    for n_knots in knot_counts:
        n = n_knots if n_knots % 2 == 1 else n_knots + 1  # ATM knot needs an odd count
        z_knots = build_nm_dense_knots(n, EDGE_Z)
        if not _check_edge_d1_d2(expiries, z_knots):
            print(f"  WARNING: d1*d2 <= 1 at an edge knot for n_knots={n}; skipping")
            continue
        t0 = time.perf_counter()
        result = calibrate(
            expiries, z_knots, config.STRIKE_REG_LAMBDA, config.N_DISCRETIZATION_STRIKES,
            config.LEE_BOUND_MARGIN, config.N_CALIBRATION_ITERATIONS,
        )
        wall_time = time.perf_counter() - t0
        arb = _post_hoc_arbitrage_checks(result, expiries, z_knots, config.N_DISCRETIZATION_STRIKES)
        rows.append({
            "underlying": underlying,
            "n_knots": n,
            "n_expiries": len(expiries),
            "n_quotes": data_diag["n_quotes_total"],
            "n_variables": result.layout.total_size,
            "n_constraints": result.n_constraints,
            "n_nonzero_P": result.n_nonzero_P,
            "n_nonzero_A": result.n_nonzero_A,
            "iterations_time": result.iterations_time,
            "total_time": result.total_time,
            "wall_time": wall_time,
            "chi_squared_mean": result.chi_squared_mean,
            **arb,
        })
        print(
            f"  n_knots={n}: total_time={result.total_time:.4f}s, "
            f"chi2_mean={result.chi_squared_mean:.4f}, "
            f"n_constraints={result.n_constraints}, nnz(P)={result.n_nonzero_P}, nnz(A)={result.n_nonzero_A}, "
            f"neg_var_viol={arb['negative_variance_violations']}, cal_spread_viol={arb['calendar_spread_violations']}"
        )
    return {"underlying": underlying, "data_diagnostics": data_diag, "rows": rows}


def plot_benchmarks(all_rows: list[dict[str, Any]], path_chi2: Path, path_time: Path) -> None:
    underlyings = sorted({r["underlying"] for r in all_rows})

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for u in underlyings:
        rows = sorted([r for r in all_rows if r["underlying"] == u], key=lambda r: r["n_knots"])
        ax.plot([r["n_knots"] for r in rows], [r["chi_squared_mean"] for r in rows], marker="o", label=u)
    ax.set_xlabel("Number of knots per expiry")
    ax.set_ylabel("Chi-squared of the fit (average across expiries)")
    ax.set_title("CVI knot-count convergence (exp4, Figure 6 reproduction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_chi2, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for u in underlyings:
        rows = sorted([r for r in all_rows if r["underlying"] == u], key=lambda r: r["n_knots"])
        ax.plot([r["n_knots"] for r in rows], [r["total_time"] for r in rows], marker="o", label=u)
    ax.set_xlabel("Number of knots per expiry")
    ax.set_ylabel("Two-step calibration time (s)")
    ax.set_title("CVI calibration time vs. knot count (exp4, Table 3 reproduction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_time, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="use the paper's knot grid (10..40) instead of the reduced grid")
    args = parser.parse_args()
    knot_counts = config.KNOT_COUNTS_PAPER if args.full else config.KNOT_COUNTS_RUN

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"knot_counts_requested": list(knot_counts), "underlyings": {}}
    all_rows: list[dict[str, Any]] = []
    try:
        for underlying in config.UNDERLYINGS_RUN:
            out = run_underlying(underlying, knot_counts)
            summary["underlyings"][underlying] = out
            all_rows.extend(out["rows"])

        summary_path = RESULTS_DIR / "exp4_benchmark.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        csv_path = RESULTS_DIR / "exp4_benchmark.csv"
        if all_rows:
            cols = [c for c in all_rows[0] if c != "iterations_time"]
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(cols) + "\n")
                for row in all_rows:
                    f.write(",".join(str(row[c]) for c in cols) + "\n")

        if all_rows:
            plot_benchmarks(all_rows, RESULTS_DIR / "exp4_chi2_vs_knots.png", RESULTS_DIR / "exp4_time_vs_knots.png")

        print(f"Summary written to {summary_path}")
        print(f"Table written to {csv_path}")
    except Exception:
        print("exp4 FAILED with an exception:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
