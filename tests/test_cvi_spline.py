"""Unit tests for the CVI dual parameterization (``src/cvi_spline.py``),
paper Sections "CVI Cubic Spline" and "CVI B-Spline" (Experiment 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cvi_spline import (
    build_transformation_matrix,
    evaluate_bspline,
    evaluate_cubic_spline,
    n_bspline_basis,
)


def _evenly_spaced_knots(n: int, half_width: float = 4.0) -> np.ndarray:
    return np.linspace(-half_width, half_width, n)


def _dense_near_money_knots(n: int, half_width: float = 4.0) -> np.ndarray:
    """Knots denser near z = 0, still containing z = 0 exactly."""
    u = np.linspace(-1.0, 1.0, n)
    z = np.sign(u) * half_width * np.abs(u) ** 1.7
    z[np.argmin(np.abs(z))] = 0.0
    return np.unique(np.round(z, 10))


@pytest.mark.parametrize("n", [7, 9, 15])
@pytest.mark.parametrize("knot_fn", [_evenly_spaced_knots, _dense_near_money_knots])
def test_cubic_spline_and_bspline_agree_including_tails(knot_fn, n):
    """The cubic-spline parameterization p and the B-spline weights
    w = M @ p must give identical v, dv/dz, d2v/dz2 everywhere, including
    in the linearly extrapolated tails (paper, "CVI B-Spline" section)."""
    z_knots = knot_fn(n)
    rng = np.random.default_rng(0)
    v0, d0 = 0.04, 0.01 * rng.standard_normal()
    c = rng.standard_normal(z_knots.size) * 0.01
    l = int(np.argmin(np.abs(z_knots)))
    c[0] = 0.0
    c[-1] = 0.0  # linear extrapolation requires zero edge convexity

    m_matrix = build_transformation_matrix(z_knots)
    p = np.concatenate([[v0, d0], c])
    w = m_matrix @ p
    assert w.size == n_bspline_basis(z_knots)

    z_eval = np.concatenate(
        [
            np.linspace(z_knots[0] - 3.0, z_knots[0], 10),  # left tail
            np.linspace(z_knots[0], z_knots[-1], 25 * (n - 1)),  # interior
            np.linspace(z_knots[-1], z_knots[-1] + 3.0, 10),  # right tail
        ]
    )
    v_cs, dv_cs, d2v_cs = evaluate_cubic_spline(z_eval, z_knots, v0, d0, c, l)
    v_bs, dv_bs, d2v_bs = evaluate_bspline(z_eval, z_knots, w)

    assert np.max(np.abs(v_cs - v_bs)) < 1e-8
    assert np.max(np.abs(dv_cs - dv_bs)) < 1e-6
    assert np.max(np.abs(d2v_cs - d2v_bs)) < 1e-4


def test_edge_second_derivative_and_cubic_coefficient_are_zero():
    """Setting c_0 = c_{n-1} = 0 must reproduce linear extrapolation with
    zero second derivative at and beyond the edge knots (paper, "CVI Cubic
    Spline": linear extrapolation requires d2v/dz2(z_0) = d2v/dz2(z_{n-1}) = 0)."""
    z_knots = _evenly_spaced_knots(9)
    c = np.array([0.0, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2, -0.3, 0.0])
    v, dv, d2v = evaluate_cubic_spline(
        np.array([z_knots[0], z_knots[-1], z_knots[0] - 1.0, z_knots[-1] + 1.0]),
        z_knots,
        v0=0.05,
        d0=0.01,
        c=c,
        l=4,
    )
    assert d2v[0] == pytest.approx(0.0, abs=1e-12)
    assert d2v[1] == pytest.approx(0.0, abs=1e-12)
    assert d2v[2] == pytest.approx(0.0, abs=1e-12)  # left tail
    assert d2v[3] == pytest.approx(0.0, abs=1e-12)  # right tail
    # In the tails the slope must equal the edge-knot slope (linear extrapolation).
    assert dv[2] == pytest.approx(dv[0], abs=1e-12)
    assert dv[3] == pytest.approx(dv[1], abs=1e-12)


def test_transformation_matrix_is_square_and_well_conditioned():
    z_knots = _evenly_spaced_knots(11)
    m_matrix = build_transformation_matrix(z_knots)
    n_params = z_knots.size + 2
    assert m_matrix.shape == (n_params, n_params)
    cond = np.linalg.cond(m_matrix)
    assert np.isfinite(cond) and cond < 1e6


def test_atm_knot_matches_v0_and_slope():
    """v(z=0) and dv/dz(z=0) must equal the input parameters exactly
    (paper: "the cubic spline parameterized by its value and first
    derivative at z = 0")."""
    z_knots = _evenly_spaced_knots(9)
    l = int(np.argmin(np.abs(z_knots)))
    v0, d0 = 0.0625, -0.004
    c = np.zeros(z_knots.size)
    v, dv, _ = evaluate_cubic_spline(np.array([0.0]), z_knots, v0, d0, c, l)
    assert v[0] == pytest.approx(v0, abs=1e-12)
    assert dv[0] == pytest.approx(d0, abs=1e-12)


def test_rejects_non_increasing_or_too_few_knots():
    with pytest.raises(ValueError):
        evaluate_cubic_spline(np.array([0.0]), np.array([1.0, 0.0, 2.0]), 0.04, 0.0, np.zeros(3), 0)
    with pytest.raises(ValueError):
        evaluate_cubic_spline(np.array([0.0]), np.array([-1.0, 1.0]), 0.04, 0.0, np.zeros(2), 0)
