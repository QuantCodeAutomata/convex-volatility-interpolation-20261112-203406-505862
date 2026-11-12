"""Experiment 1: CVI cubic-spline / B-spline dual parameterization.

Paper, Section "The CVI Parameterization" (p.1) and its subsections "CVI
Cubic Spline" and "CVI B-Spline" (p.2): variance v(z) in normalized
log-moneyness z is represented either by n+2 cubic-spline parameters
p = (v(0), dv/dz(0), c_0, ..., c_{n-1}) or by n+2 cubic B-spline weights w,
linked by a linear map w = M @ p. This script:

1. verifies, for several knot configurations (evenly spaced and
   near-the-money dense, several knot counts), that both representations
   produce identical v(z), dv/dz(z), d2v/dz2(z), including in the linearly
   extrapolated tails beyond z_0 and z_{n-1} (research objectives 1-4);
2. confirms that the physical CVI construction (edge convexities
   c_0 = c_{n-1} = 0) gives zero second derivative at the edge knots and a
   purely linear (zero quadratic/cubic coefficient) extrapolating segment
   (research objective 5);
3. reproduces the shape of Figure 1 (p.2) from its (z, v, dv/dz, d2v/dz2)
   table, by least-squares fitting the CVI cubic spline to the table.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.cvi_spline import (
    build_transformation_matrix,
    evaluate_bspline,
    evaluate_cubic_spline,
    n_bspline_basis,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_symmetric_knots(n: int, edge: float) -> np.ndarray:
    """Evenly spaced knots in [-edge, edge], with 0 exactly one of them."""
    assert n % 2 == 1, "use an odd knot count so 0 is exactly a knot"
    half = np.linspace(0.0, edge, n // 2 + 1)
    return np.concatenate([-half[-1:0:-1], half])


def build_nm_dense_knots(n: int, edge: float, power: float = 2.2) -> np.ndarray:
    """Knots concentrated near-the-money (|z| small), symmetric, 0 included.

    Paper's data requirement for exp_1: "synthetic knot configurations with
    varying knot densities (evenly spaced and increasing density near-the-
    money)". The exact density law is not specified; a power-law warp of an
    evenly spaced grid in [0, 1] is used here.
    """
    assert n % 2 == 1
    t = np.linspace(0.0, 1.0, n // 2 + 1)
    half = edge * t**power
    return np.concatenate([-half[-1:0:-1], half])


def _random_physical_params(n: int, rng: np.random.Generator) -> tuple[float, float, np.ndarray]:
    """Random (v0, d0, c) with the physical CVI convention c_0 = c_{n-1} = 0
    (edge convexities fixed for linear extrapolation, Section "CVI Cubic
    Spline")."""
    v0 = float(rng.uniform(0.02, 0.1))
    d0 = float(rng.uniform(-0.05, 0.05))
    c = rng.uniform(-0.2, 0.2, size=n)
    c[0] = 0.0
    c[-1] = 0.0
    return v0, d0, c


def verify_dual_parameterization(z_knots: np.ndarray, rng: np.random.Generator) -> dict:
    """Check that the cubic-spline and B-spline representations of the same
    physical CVI parameter vector agree everywhere, including the
    extrapolated tails (research objectives 3-4)."""
    n = z_knots.size
    l = int(np.argmin(np.abs(z_knots)))
    v0, d0, c = _random_physical_params(n, rng)

    tail = config.EXP1_TAIL_EXTENSION
    n_pts = config.EXP1_EVAL_POINTS_PER_SEGMENT * (n - 1)
    z_eval = np.linspace(z_knots[0] - tail, z_knots[-1] + tail, n_pts)

    v_cs, dv_cs, d2v_cs = evaluate_cubic_spline(z_eval, z_knots, v0, d0, c, l)

    M = build_transformation_matrix(z_knots)
    p = np.concatenate([[v0, d0], c])
    w = M @ p
    assert w.shape == (n_bspline_basis(z_knots),)
    v_bs, dv_bs, d2v_bs = evaluate_bspline(z_eval, z_knots, w)

    interior = (z_eval >= z_knots[0]) & (z_eval <= z_knots[-1])
    tails = ~interior

    out = {
        "n_knots": n,
        "max_abs_diff_v_interior": float(np.max(np.abs(v_cs[interior] - v_bs[interior]))),
        "max_abs_diff_dv_interior": float(np.max(np.abs(dv_cs[interior] - dv_bs[interior]))),
        "max_abs_diff_d2v_interior": float(np.max(np.abs(d2v_cs[interior] - d2v_bs[interior]))),
        "max_abs_diff_v_tails": float(np.max(np.abs(v_cs[tails] - v_bs[tails]))) if np.any(tails) else 0.0,
        "max_abs_diff_dv_tails": float(np.max(np.abs(dv_cs[tails] - dv_bs[tails]))) if np.any(tails) else 0.0,
        "cond_M": float(np.linalg.cond(M)),
        # Research objective 5: second derivative zero at edge knots, and
        # the tail extrapolation has no quadratic/cubic term (d2v == 0
        # identically beyond the edge knots, by construction).
        "d2v_at_z0": float(evaluate_cubic_spline(np.array([z_knots[0]]), z_knots, v0, d0, c, l)[2][0]),
        "d2v_at_zn": float(evaluate_cubic_spline(np.array([z_knots[-1]]), z_knots, v0, d0, c, l)[2][0]),
        "max_abs_d2v_in_tails": float(np.max(np.abs(d2v_cs[tails]))) if np.any(tails) else 0.0,
    }
    return out


def reproduce_figure1() -> dict:
    """Approximate reproduction of Figure 1 (p.2) from its (z, v, dv/dz,
    d2v/dz2) table.

    The paper gives the table only at nine z values and states (per this
    replication's reading of "CVI Cubic Spline") that these are knots of a
    cubic spline, but does not disclose the full knot grid used to produce
    the figure (see exp_1's recorded uncertainty: "the exact knot positions
    used for the example in Figure 1 are not fully specified"). Treating the
    nine table z-values themselves as the only knots over- or under-
    determines v, dv/dz and d2v/dz2 simultaneously (verified numerically:
    matching v exactly leaves dv/dz, d2v/dz2 far from the table, and
    vice-versa), which confirms the table was generated on a denser,
    undisclosed knot grid. This function instead fits the CVI cubic spline,
    on a denser symmetric knot grid obtained by inserting the midpoint
    between each pair of consecutive table z-values, by weighted least
    squares against the table's (v, dv/dz, d2v/dz2) jointly. The result
    reproduces the table's qualitative shape (smile with negative skew and
    positive convexity) but not its exact digits; both are reported.
    """
    z_table = np.array(config.EXP1_FIGURE1_Z)
    v_table = np.array(config.EXP1_FIGURE1_V)
    dv_table = np.array(config.EXP1_FIGURE1_DV)
    d2v_table = np.array(config.EXP1_FIGURE1_D2V)

    mids = (z_table[:-1] + z_table[1:]) / 2.0
    z_knots = np.sort(np.concatenate([z_table, mids]))
    n = z_knots.size
    l = int(np.argmin(np.abs(z_knots)))

    # Free-parameter design matrices (v0, d0, c_1..c_{n-2}; c_0=c_{n-1}=0
    # fixed) evaluated at the nine table points.
    V = np.zeros((9, n))
    D = np.zeros((9, n))
    C = np.zeros((9, n))
    for col in range(n):
        v0 = 1.0 if col == 0 else 0.0
        d0 = 1.0 if col == 1 else 0.0
        c = np.zeros(n)
        if col >= 2:
            c[col - 1] = 1.0
        v, dv, d2v = evaluate_cubic_spline(z_table, z_knots, v0, d0, c, l)
        V[:, col], D[:, col], C[:, col] = v, dv, d2v

    # Weights compensate for the very different scales of v, dv/dz,
    # d2v/dz2 in the table (not specified in the paper; chosen so the fit
    # balances all three quantities).
    w = np.concatenate([np.full(9, 40.0), np.full(9, 60.0), np.full(9, 40.0)])
    A = np.vstack([V, D, C])
    b = np.concatenate([v_table, dv_table, d2v_table])
    x, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)

    z_fine = np.linspace(z_knots[0] - 4.0, z_knots[-1] + 4.0, 400)
    v0, d0, c = x[0], x[1], np.concatenate([[0.0], x[2:], [0.0]])
    v_fit, dv_fit, d2v_fit = evaluate_cubic_spline(z_fine, z_knots, v0, d0, c, l)
    v_at_table, dv_at_table, d2v_at_table = evaluate_cubic_spline(z_table, z_knots, v0, d0, c, l)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, fit, table, name in zip(
        axes, (v_fit, dv_fit, d2v_fit), (v_table, dv_table, d2v_table), ("v(z)", "dv/dz(z)", "d2v/dz2(z)")
    ):
        ax.plot(z_fine, fit, label="fitted CVI cubic spline")
        ax.plot(z_table, table, "o", label="Figure 1 table")
        ax.set_xlabel("z (normalized log-moneyness)")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
    fig.suptitle("CVI Figure 1: fitted cubic spline vs. the paper's table")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_figure1_reproduction.png", dpi=140)
    plt.close(fig)

    return {
        "z_table": z_table.tolist(),
        "v_table": v_table.tolist(),
        "dv_table": dv_table.tolist(),
        "d2v_table": d2v_table.tolist(),
        "v_fit_at_table": v_at_table.tolist(),
        "dv_fit_at_table": dv_at_table.tolist(),
        "d2v_fit_at_table": d2v_at_table.tolist(),
        "max_abs_err_v": float(np.max(np.abs(v_at_table - v_table))),
        "max_abs_err_dv": float(np.max(np.abs(dv_at_table - dv_table))),
        "max_abs_err_d2v": float(np.max(np.abs(d2v_at_table - d2v_table))),
        "note": (
            "Approximate reproduction: the paper does not disclose the knot "
            "grid used for Figure 1, so this is a weighted least-squares fit "
            "of the CVI cubic spline to the published table, not an exact "
            "digit-level match."
        ),
    }


def main() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SYNTHETIC_SEED)

    dual_param_results = []
    for n in config.EXP1_KNOT_COUNTS:
        for style, builder in (("even", build_symmetric_knots), ("nm_dense", build_nm_dense_knots)):
            z_knots = builder(n, 8.0)
            res = verify_dual_parameterization(z_knots, rng)
            res["knot_style"] = style
            dual_param_results.append(res)
            print(
                f"n={n:2d} {style:8s}: max|dv diff|={res['max_abs_diff_v_interior']:.2e} "
                f"(interior), {res['max_abs_diff_v_tails']:.2e} (tails); cond(M)={res['cond_M']:.2f}"
            )

    fig1 = reproduce_figure1()
    print(
        f"Figure 1 fit: max abs error v={fig1['max_abs_err_v']:.3f}, "
        f"dv/dz={fig1['max_abs_err_dv']:.3f}, d2v/dz2={fig1['max_abs_err_d2v']:.3f}"
    )

    summary = {"dual_parameterization": dual_param_results, "figure1_reproduction": fig1}
    with open(RESULTS_DIR / "exp1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot of one representative dual-parameterization check (evenly spaced,
    # largest knot count) for visual confirmation of extrapolation.
    n = config.EXP1_KNOT_COUNTS[-1]
    z_knots = build_symmetric_knots(n, 8.0)
    v0, d0, c = _random_physical_params(n, np.random.default_rng(config.SYNTHETIC_SEED))
    z_eval = np.linspace(z_knots[0] - 4.0, z_knots[-1] + 4.0, 400)
    v_cs, _, _ = evaluate_cubic_spline(z_eval, z_knots, v0, d0, c, int(np.argmin(np.abs(z_knots))))
    M = build_transformation_matrix(z_knots)
    w = M @ np.concatenate([[v0, d0], c])
    v_bs, _, _ = evaluate_bspline(z_eval, z_knots, w)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(z_eval, v_cs, label="cubic-spline parameterization", lw=2)
    ax.plot(z_eval, v_bs, "--", label="B-spline parameterization", lw=2)
    ax.axvline(z_knots[0], color="gray", ls=":", label="edge knots")
    ax.axvline(z_knots[-1], color="gray", ls=":")
    ax.set_xlabel("z (normalized log-moneyness)")
    ax.set_ylabel("v(z)")
    ax.set_title("CVI dual parameterization: cubic spline vs. B-spline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_dual_parameterization.png", dpi=140)
    plt.close(fig)

    return summary


if __name__ == "__main__":
    main()
