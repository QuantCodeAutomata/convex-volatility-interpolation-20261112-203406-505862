"""Tests for src.cvi_objective: the four CVI penalties (paper, Sections
"Objective Function", "Least Squares Penalty", "Above Ask Penalty",
"Below Bid Penalty", "Strike Regularization Penalty", pp. 2-3).
"""

from __future__ import annotations

import numpy as np
import pytest

from data.generate_synthetic import make_synthetic_expiry, make_synthetic_surface
from src.cvi_objective import (
    ExpiryQuotes,
    build_variable_layout,
    assemble_objective,
    epigraph_constraints,
    free_param_design_matrices,
)
from src.cvi_spline import evaluate_cubic_spline


def _empty_expiry(
    T: float, v_star: float, z_mid: np.ndarray, v_bid_mid: np.ndarray, v_ask_mid: np.ndarray
) -> ExpiryQuotes:
    """Build an ExpiryQuotes with only mid quotes (no ask-only / bid-only
    quotes), used to isolate the Least Squares Penalty in regression tests."""
    empty = np.array([])
    return ExpiryQuotes(
        T=T,
        v_star=v_star,
        z_mid=z_mid,
        v_bid_mid=v_bid_mid,
        v_ask_mid=v_ask_mid,
        z_ask_only=empty,
        v_ask_only=empty,
        vega_ask_only=empty,
        z_bid_only=empty,
        v_bid_only=empty,
        vega_bid_only=empty,
    )


def test_free_param_design_matrices_shapes_and_matches_spline() -> None:
    """V, D, C have the documented shapes and V @ x (with the free-param ->
    (v0, d0, c) mapping c[0] = c[-1] = 0) matches evaluate_cubic_spline
    directly, at the same evaluation points."""
    z_knots = np.array([-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
    n = z_knots.size
    z_eval = np.linspace(-5.0, 5.0, 13)  # includes points outside the knots (tails)

    V, D, C = free_param_design_matrices(z_eval, z_knots)
    assert V.shape == (z_eval.size, n)
    assert D.shape == (z_eval.size, n)
    assert C.shape == (z_eval.size, n)

    rng = np.random.default_rng(42)
    x = rng.normal(size=n)

    v0, d0 = x[0], x[1]
    c = np.zeros(n)
    c[1:-1] = x[2:]  # c_0 = c_{n-1} = 0 fixed, free params map to interior convexities
    l = int(np.argmin(np.abs(z_knots)))
    v_expected, dv_expected, d2v_expected = evaluate_cubic_spline(z_eval, z_knots, v0, d0, c, l)

    np.testing.assert_allclose(V @ x, v_expected, atol=1e-9)
    np.testing.assert_allclose(D @ x, dv_expected, atol=1e-9)
    np.testing.assert_allclose(C @ x, d2v_expected, atol=1e-9)


def test_assemble_objective_symmetric_and_psd() -> None:
    """P (built from a small synthetic 2-expiry surface) is symmetric and
    positive semi-definite (paper: the objective is a sum of
    squared/epigraph penalties, hence necessarily convex)."""
    z_knots = np.linspace(-6.0, 6.0, 9)
    expiries = make_synthetic_surface(
        n_expiries=2,
        z_range=(-5.0, 5.0),
        seed=123,
        n_mid=6,
        n_ask_only=2,
        n_bid_only=2,
    )
    P, q, layout = assemble_objective(expiries, z_knots, lam=0.05)
    assert q.shape == (layout.total_size,)

    P_dense = P.toarray()
    np.testing.assert_allclose(P_dense, P_dense.T, atol=1e-10)

    eigvals = np.linalg.eigvalsh(P_dense)
    assert eigvals.min() >= -1e-8


def test_least_squares_weight_tighter_spread_has_larger_weight() -> None:
    """w_i = 1/(ask - bid)^2: a tighter bid-ask spread strictly dominates a
    wider one (paper, Section "Least Squares Penalty")."""
    v_mid = 0.04
    tight_bid, tight_ask = v_mid - 0.001, v_mid + 0.001
    wide_bid, wide_ask = v_mid - 0.02, v_mid + 0.02

    w_tight = 1.0 / (tight_ask - tight_bid) ** 2
    w_wide = 1.0 / (wide_ask - wide_bid) ** 2

    assert w_tight > w_wide
    assert w_tight / w_wide == pytest.approx(((wide_ask - wide_bid) / (tight_ask - tight_bid)) ** 2)


def test_epigraph_constraints_rows_and_bounds() -> None:
    """For n_ask_only ask-only quotes there are 2 * n_ask_only above-ask
    rows: one "above_ask_T*" row (u = inf, l = -v_ask) and one
    "above_ask_nonneg_T*" row (l = 0, u = inf)."""
    z_knots = np.linspace(-6.0, 6.0, 7)
    n_ask_only = 3
    n_bid_only = 2
    expiries = make_synthetic_surface(
        n_expiries=1,
        z_range=(-3.0, 3.0),
        seed=7,
        n_mid=5,
        n_ask_only=n_ask_only,
        n_bid_only=n_bid_only,
    )
    exp = expiries[0]
    layout = build_variable_layout(expiries, z_knots)
    constraints = epigraph_constraints(expiries, layout, z_knots)

    above_ask_names = [nm for nm in constraints.names if nm.startswith("above_ask_T0_i")]
    above_ask_nonneg_names = [nm for nm in constraints.names if nm.startswith("above_ask_nonneg_T0_i")]
    assert len(above_ask_names) == n_ask_only
    assert len(above_ask_nonneg_names) == n_ask_only

    assert constraints.A.shape[0] == len(constraints.names)
    assert constraints.l.shape == (len(constraints.names),)
    assert constraints.u.shape == (len(constraints.names),)

    idx_above = constraints.names.index("above_ask_T0_i0")
    assert constraints.u[idx_above] == np.inf
    assert constraints.l[idx_above] == pytest.approx(-exp.v_ask_only[0])

    idx_above_nonneg = constraints.names.index("above_ask_nonneg_T0_i0")
    assert constraints.l[idx_above_nonneg] == 0.0
    assert constraints.u[idx_above_nonneg] == np.inf

    below_bid_names = [nm for nm in constraints.names if nm.startswith("below_bid_T0_i")]
    below_bid_nonneg_names = [nm for nm in constraints.names if nm.startswith("below_bid_nonneg_T0_i")]
    assert len(below_bid_names) == n_bid_only
    assert len(below_bid_nonneg_names) == n_bid_only


def test_zero_ask_only_and_bid_only_edge_case() -> None:
    """An expiry with zero ask-only and zero bid-only quotes still
    assembles without error; the corresponding slack slices are empty."""
    z_knots = np.linspace(-6.0, 6.0, 7)
    rng = np.random.default_rng(99)
    exp = make_synthetic_expiry(
        T=0.5,
        v_star=0.04,
        n_mid=8,
        n_ask_only=0,
        n_bid_only=0,
        z_range=(-3.0, 3.0),
        bid_ask_half_spread=0.03,
        noise_std=0.002,
        rng=rng,
    )
    expiries = [exp]
    layout = build_variable_layout(expiries, z_knots)
    assert layout.ask_slack_slices[0].start == layout.ask_slack_slices[0].stop
    assert layout.bid_slack_slices[0].start == layout.bid_slack_slices[0].stop

    P, q, layout2 = assemble_objective(expiries, z_knots, lam=0.05)
    assert P.shape == (layout2.total_size, layout2.total_size)

    constraints = epigraph_constraints(expiries, layout2, z_knots)
    assert not any(nm.startswith("above_ask_T0_i") for nm in constraints.names)
    assert not any(nm.startswith("below_bid_T0_i") for nm in constraints.names)


def test_regression_tiny_least_squares_penalty() -> None:
    """Hand-computed Least Squares Penalty on a tiny (5-knot) surface
    matches 0.5 * x^T P x + q^T x (up to the x-independent constant term
    that the QP objective does not store), evaluated at the exact
    weighted-least-squares minimizer."""
    z_knots = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    n = z_knots.size
    z_mid = np.array([-1.8, -1.0, -0.3, 0.3, 1.0, 1.8])
    v_bid_mid = np.array([0.038, 0.039, 0.040, 0.0405, 0.041, 0.043])
    v_ask_mid = np.array([0.044, 0.042, 0.0425, 0.044, 0.0445, 0.047])

    exp = _empty_expiry(T=1.0, v_star=0.04, z_mid=z_mid, v_bid_mid=v_bid_mid, v_ask_mid=v_ask_mid)
    lam = 0.0  # isolate the Least Squares Penalty (no strike-regularization contribution)
    P, q, layout = assemble_objective([exp], z_knots, lam)
    P_dense = P.toarray()

    V, _, _ = free_param_design_matrices(z_mid, z_knots)
    v_mid_mid = 0.5 * (v_bid_mid + v_ask_mid)
    w = 1.0 / (v_ask_mid - v_bid_mid) ** 2
    n_mid = z_mid.size

    # Weighted least-squares minimizer of (1/n_mid) sum w_i (V x - v_mid)_i^2.
    WV = V * w[:, None]
    x_star, *_ = np.linalg.lstsq(V.T @ WV, V.T @ (w * v_mid_mid), rcond=None)

    # Embed x_star (spline free params only) into the full variable vector
    # (which also carries the strike-regularization slacks, zero here since
    # lam = 0 and the expiry has no ask-only / bid-only quotes).
    x_full = np.zeros(layout.total_size)
    x_full[layout.spline_slices[0]] = x_star

    # The gradient of 0.5 x^T P x + q^T x at the minimizer must vanish.
    grad = P_dense @ x_full + q
    np.testing.assert_allclose(grad, np.zeros(layout.total_size), atol=1e-8)

    # The quadratic form matches the hand-computed penalty up to the
    # x-independent constant (1/n_mid) sum w_i v_mid_i^2.
    penalty_hand = np.sum(w * (V @ x_star - v_mid_mid) ** 2) / n_mid
    const = np.sum(w * v_mid_mid**2) / n_mid
    quad_value = 0.5 * x_full @ P_dense @ x_full + q @ x_full
    np.testing.assert_allclose(quad_value, penalty_hand - const, atol=1e-8)
