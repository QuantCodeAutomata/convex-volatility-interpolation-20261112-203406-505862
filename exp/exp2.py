"""exp_2: CVI Objective Function (penalties) demonstration.

Exercises the already-implemented CVI objective (paper, Sections
"Objective Function", "Least Squares Penalty", "Above Ask Penalty",
"Below Bid Penalty", "Strike Regularization Penalty", pp. 2-3), as built
by ``src.cvi_objective`` and used end-to-end by ``src.cvi_calibration``.

Run with::

    python -m exp.exp2
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

from data.generate_synthetic import make_synthetic_surface, reference_smile
from src import config
from src.cvi_calibration import calibrate
from src.cvi_objective import assemble_objective, free_param_design_matrices

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_symmetric_knots(n: int, edge: float) -> np.ndarray:
    """Symmetric knot vector of exactly ``n`` knots (odd) spanning
    [-edge, edge] with zero as one of the knots (the ATM knot)."""
    assert n % 2 == 1, "use an odd knot count so 0 is exactly a knot"
    half = np.linspace(0.0, edge, n // 2 + 1)
    return np.concatenate([-half[-1:0:-1], half])


def _check_psd(P: Any) -> tuple[bool, float]:
    """Symmetry + positive semi-definiteness check on the dense objective
    Hessian P (paper: the objective is a sum of squared/epigraph penalties,
    hence necessarily convex, Section "Objective Function")."""
    P_dense = P.toarray()
    symmetric = bool(np.allclose(P_dense, P_dense.T, atol=1e-8, rtol=1e-6))
    eigvals = np.linalg.eigvalsh(0.5 * (P_dense + P_dense.T))
    min_eig = float(eigvals.min())
    return symmetric, min_eig


def demo_wide_spread_negligible() -> dict[str, Any]:
    """Paper, Section "Least Squares Penalty": "options with very wide
    bid-ask quotes have a negligible impact on the fit" because the
    least-squares weight w_i = 1/(ask-bid)^2 decays quadratically with the
    spread. Uses ``reference_smile`` (data/generate_synthetic.py) to build
    a representative ATM mid quote and evaluates the weight formula
    directly for a range of half-spreads expressed as a fraction of
    v_mid (matching ``make_synthetic_expiry``'s convention: half_spread =
    bid_ask_half_spread * v_mid, so ask - bid = 2 * bid_ask_half_spread *
    v_mid)."""
    half_spread_fractions = np.array([0.005, 0.02, 0.05, 0.1, 0.3, 0.6])
    v_star = 0.04  # representative anchor variance level (matches make_synthetic_surface's default)
    z_fixed = 0.0  # ATM, representative fixed strike position
    v_mid = float(reference_smile(np.array([z_fixed]), v_star)[0])

    ask_minus_bid = 2.0 * half_spread_fractions * v_mid
    weights = 1.0 / ask_minus_bid**2
    weight_ratio_to_tightest = weights / weights[0]

    rows = list(
        zip(half_spread_fractions.tolist(), weights.tolist(), weight_ratio_to_tightest.tolist())
    )
    csv_path = RESULTS_DIR / "exp2_weight_vs_spread.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("half_spread_fraction,weight,weight_ratio_to_tightest\n")
        for hs, w, r in rows:
            f.write(f"{hs},{w},{r}\n")

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.loglog(half_spread_fractions, weights, marker="o", label="w_i = 1/(ask-bid)^2")
    ax.set_xlabel("Bid-ask half-spread (fraction of v_mid)")
    ax.set_ylabel("Least-squares weight w_i")
    ax.set_title("Least-squares weight decays with bid-ask spread width")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_weight_vs_spread.png", dpi=150)
    plt.close(fig)

    return {
        "half_spread_fractions": half_spread_fractions.tolist(),
        "weights": weights.tolist(),
        "weight_ratio_to_tightest": weight_ratio_to_tightest.tolist(),
        "widest_to_tightest_ratio": float(weight_ratio_to_tightest[-1]),
    }


def demo_chi_squared_vs_n() -> dict[str, Any]:
    """Paper, Section "Objective Function": the penalties "produce
    dimensionless numbers that scale consistently ... regardless of the
    number of ... options". Builds a one-expiry surface with a varying
    number of mid quotes and checks that ``chi_squared_mean`` from a full
    calibration stays roughly constant."""
    z_knots = build_symmetric_knots(config.EXP2_N_KNOTS, 6.0)
    n_mid_values = [10, 30, 60]
    chi2_means = []
    for n_mid in n_mid_values:
        expiries_n = make_synthetic_surface(
            n_expiries=1,
            z_range=(-5.5, 5.5),
            seed=config.SYNTHETIC_SEED,
            n_mid=n_mid,
        )
        result = calibrate(
            expiries_n,
            z_knots,
            config.STRIKE_REG_LAMBDA,
            config.N_DISCRETIZATION_STRIKES,
            config.LEE_BOUND_MARGIN,
            config.N_CALIBRATION_ITERATIONS,
        )
        chi2_means.append(result.chi_squared_mean)

    csv_path = RESULTS_DIR / "exp2_chi_squared_vs_n.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("n_mid,chi_squared_mean\n")
        for n_mid, chi2 in zip(n_mid_values, chi2_means):
            f.write(f"{n_mid},{chi2}\n")

    chi2_array = np.array(chi2_means)
    ratio = float(chi2_array.max() / chi2_array.min())
    return {
        "n_mid_values": n_mid_values,
        "chi_squared_mean_values": chi2_means,
        "max_over_min_ratio": ratio,
        "within_factor_of_3": bool(ratio <= 3.0),
    }


def demo_fit_quality(expiries: list, z_knots: np.ndarray, result: Any) -> dict[str, Any]:
    """Fit-quality checks (paper's stated goals of the four penalties):
    (a) fraction of fitted mid variances within the bid-ask band;
    (b) maximum above-ask / below-bid violation for ask-only/bid-only
    quotes (the epigraph slacks should keep these small)."""
    x = result.x
    layout = result.layout

    n_within = 0
    n_total = 0
    max_above_ask_violation = 0.0
    max_below_bid_violation = 0.0

    for j, exp in enumerate(expiries):
        xj = x[layout.spline_slices[j]]

        if exp.z_mid.size > 0:
            V_mid, _, _ = free_param_design_matrices(exp.z_mid, z_knots)
            v_fit_mid = V_mid @ xj
            within = (v_fit_mid >= exp.v_bid_mid) & (v_fit_mid <= exp.v_ask_mid)
            n_within += int(within.sum())
            n_total += within.size

        if exp.z_ask_only.size > 0:
            V_ask, _, _ = free_param_design_matrices(exp.z_ask_only, z_knots)
            v_fit_ask = V_ask @ xj
            violation = np.maximum(v_fit_ask - exp.v_ask_only, 0.0)
            max_above_ask_violation = max(max_above_ask_violation, float(violation.max()))

        if exp.z_bid_only.size > 0:
            V_bid, _, _ = free_param_design_matrices(exp.z_bid_only, z_knots)
            v_fit_bid = V_bid @ xj
            violation = np.maximum(exp.v_bid_only - v_fit_bid, 0.0)
            max_below_bid_violation = max(max_below_bid_violation, float(violation.max()))

    frac_within = float(n_within / n_total) if n_total > 0 else float("nan")
    return {
        "n_mid_within_bid_ask": n_within,
        "n_mid_total": n_total,
        "fraction_mid_within_bid_ask": frac_within,
        "max_above_ask_violation": max_above_ask_violation,
        "max_below_bid_violation": max_below_bid_violation,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    try:
        # Step 1-2: knots and synthetic surface.
        z_knots = build_symmetric_knots(config.EXP2_N_KNOTS, 6.0)
        expiries = make_synthetic_surface(
            n_expiries=config.EXP2_N_EXPIRIES,
            z_range=(-5.5, 5.5),
            seed=config.SYNTHETIC_SEED,
        )

        # Step 3: assemble objective, check symmetry/PSD.
        P, q, layout = assemble_objective(expiries, z_knots, config.STRIKE_REG_LAMBDA)
        symmetric, min_eig = _check_psd(P)
        assert min_eig >= -1e-8, f"P is not PSD: min eigenvalue = {min_eig}"
        summary["psd_check"] = {
            "symmetric": symmetric,
            "min_eigenvalue": min_eig,
            "total_size": layout.total_size,
        }

        # Step 4: wide bid-ask quotes have negligible impact.
        summary["weight_vs_spread"] = demo_wide_spread_negligible()

        # Step 5: chi-squared normalization across n_mid.
        summary["chi_squared_vs_n"] = demo_chi_squared_vs_n()

        # Step 6: full calibration on the step-2 surface + fit quality.
        result = calibrate(
            expiries,
            z_knots,
            config.STRIKE_REG_LAMBDA,
            config.N_DISCRETIZATION_STRIKES,
            config.LEE_BOUND_MARGIN,
            config.N_CALIBRATION_ITERATIONS,
        )
        summary["fit_quality"] = demo_fit_quality(expiries, z_knots, result)
        summary["calibration"] = {
            "chi_squared_per_expiry": result.chi_squared_per_expiry,
            "chi_squared_mean": result.chi_squared_mean,
            "total_time": result.total_time,
            "n_nonzero_P": result.n_nonzero_P,
            "n_nonzero_A": result.n_nonzero_A,
            "n_constraints": result.n_constraints,
        }

        # Step 7: write JSON summary.
        summary_path = RESULTS_DIR / "exp2_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Step 8: human-readable report.
        print("=== exp2: CVI Objective Function ===")
        print(f"P symmetric: {symmetric}, min eigenvalue: {min_eig:.3e} (PSD check passed)")
        print(
            "Weight ratio (widest/tightest spread): "
            f"{summary['weight_vs_spread']['widest_to_tightest_ratio']:.3e}"
        )
        print(
            "chi_squared_mean vs n_mid: "
            f"{summary['chi_squared_vs_n']['chi_squared_mean_values']}, "
            f"max/min ratio = {summary['chi_squared_vs_n']['max_over_min_ratio']:.3f} "
            f"(within factor of 3: {summary['chi_squared_vs_n']['within_factor_of_3']})"
        )
        fq = summary["fit_quality"]
        print(
            f"Fraction of mid quotes with fit within [bid, ask]: "
            f"{fq['fraction_mid_within_bid_ask']:.3f} "
            f"({fq['n_mid_within_bid_ask']}/{fq['n_mid_total']})"
        )
        print(
            f"Max above-ask violation: {fq['max_above_ask_violation']:.3e}, "
            f"max below-bid violation: {fq['max_below_bid_violation']:.3e}"
        )
        print(
            f"Full calibration: chi_squared_mean={result.chi_squared_mean:.4f}, "
            f"total_time={result.total_time:.3f}s, n_constraints={result.n_constraints}"
        )
        print(f"Summary written to {summary_path}")
    except Exception:
        print("exp2 FAILED with an exception:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
