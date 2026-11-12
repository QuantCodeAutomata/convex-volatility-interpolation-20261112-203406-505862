"""Tests for src.cvi_constraints: the CVI no-arbitrage linear constraints
(paper, Sections "Linear Extrapolation" through "Putting It All Together",
pp. 3-6).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cvi_constraints import (
    assemble_all_constraints,
    generic_no_calendar_spread,
    lee_tail_slope_bounds,
    no_calendar_spread_tails,
    positivity_in_tails,
    positivity_of_variance,
)
from src.cvi_objective import ExpiryQuotes, build_variable_layout
from src.qp_types import ConstraintSet


def _empty_expiry(T: float, v_star: float) -> ExpiryQuotes:
    """A minimal ExpiryQuotes with no quotes at all, sufficient to build a
    VariableLayout for constraint-assembly tests (the constraints under
    test depend only on the layout/z_knots geometry, not on the quotes)."""
    empty = np.array([])
    return ExpiryQuotes(
        T=T,
        v_star=v_star,
        z_mid=empty,
        v_bid_mid=empty,
        v_ask_mid=empty,
        z_ask_only=empty,
        v_ask_only=empty,
        vega_ask_only=empty,
        z_bid_only=empty,
        v_bid_only=empty,
        vega_bid_only=empty,
    )


def _make_layout(n_expiries: int, z_knots: np.ndarray, v_star: float = 0.04, t_start: float = 0.5):
    """Build a VariableLayout for ``n_expiries`` synthetic empty expiries
    with distinct, increasing T values (needed by the calendar-spread and
    Lee-bound constraints, which index v_star/T per expiry)."""
    expiries = [_empty_expiry(T=t_start * (j + 1), v_star=v_star) for j in range(n_expiries)]
    layout = build_variable_layout(expiries, z_knots)
    return expiries, layout


# ---------------------------------------------------------------------------
# positivity_of_variance
# ---------------------------------------------------------------------------


def test_positivity_of_variance_row_count_and_bounds_first_expiry_only() -> None:
    """Paper, Section "Positivity of the Variance": v(z) >= 0 discretized
    over n_strikes points, first expiry only -> exactly n_strikes rows,
    all referencing only expiry-0 spline indices, with l=0, u=+inf."""
    z_knots = np.linspace(-4.0, 4.0, 7)
    n_strikes = 9
    expiries, layout = _make_layout(3, z_knots)

    cs = positivity_of_variance(layout, z_knots, n_strikes, first_expiry_only=True)

    assert cs.A.shape[0] == n_strikes
    assert np.all(cs.l == 0.0)
    assert np.all(np.isinf(cs.u)) and np.all(cs.u > 0)

    # Every row must only touch expiry-0's spline indices.
    expiry0_idx = set(range(layout.spline_slices[0].start, layout.spline_slices[0].stop))
    A_coo = cs.A.tocoo()
    touched_cols = set(A_coo.col.tolist())
    assert touched_cols <= expiry0_idx


def test_positivity_of_variance_all_expiries_scales_rows() -> None:
    """With first_expiry_only=False, row count scales with n_expiries."""
    z_knots = np.linspace(-4.0, 4.0, 7)
    n_strikes = 5
    expiries, layout = _make_layout(4, z_knots)

    cs = positivity_of_variance(layout, z_knots, n_strikes, first_expiry_only=False)
    assert cs.A.shape[0] == n_strikes * layout.n_expiries


# ---------------------------------------------------------------------------
# positivity_in_tails (sign convention)
# ---------------------------------------------------------------------------


def test_positivity_in_tails_sign_convention() -> None:
    """Paper, Section "Positivity of the Variance in the Tails": slope
    must be <= 0 in the left tail (dv/dz(z_0) <= 0) and >= 0 in the right
    tail (dv/dz(z_{n-1}) >= 0). This is the critical sign/direction test:
    the row touching dv/dz(z_0) must have u=0 (upper-bounded at 0) and the
    row touching dv/dz(z_{n-1}) must have l=0 (lower-bounded at 0)."""
    z_knots = np.linspace(-3.0, 3.0, 5)
    expiries, layout = _make_layout(1, z_knots)

    cs = positivity_in_tails(layout, z_knots)

    assert cs.A.shape[0] == 2
    assert cs.names[0] == "tail_positivity_left_T0"
    assert cs.names[1] == "tail_positivity_right_T0"

    # Left-tail row: dv/dz(z_0) <= 0 -> l=-inf, u=0.
    assert np.isneginf(cs.l[0])
    assert cs.u[0] == 0.0

    # Right-tail row: dv/dz(z_{n-1}) >= 0 -> l=0, u=+inf.
    assert cs.l[1] == 0.0
    assert np.isinf(cs.u[1]) and cs.u[1] > 0


def test_positivity_in_tails_row_count_scales_with_expiries() -> None:
    z_knots = np.linspace(-3.0, 3.0, 5)
    expiries, layout = _make_layout(3, z_knots)
    cs = positivity_in_tails(layout, z_knots)
    assert cs.A.shape[0] == 2 * layout.n_expiries


# ---------------------------------------------------------------------------
# lee_tail_slope_bounds
# ---------------------------------------------------------------------------


def test_lee_tail_slope_bounds_exact_value() -> None:
    """Paper, Section "Lee's Tail Slope Bounds": the bound is exactly
    2/sqrt(v_star T). Regression-tests the literal formula (not some other
    expression) for v_star=0.04, T=0.5, with margin subtracted/added."""
    z_knots = np.linspace(-3.0, 3.0, 5)
    v_star_val = 0.04
    T_val = 0.5
    margin = 1e-4
    expiries, layout = _make_layout(1, z_knots, v_star=v_star_val, t_start=T_val)

    v_star_arr = np.array([v_star_val])
    T_arr = np.array([T_val])
    cs = lee_tail_slope_bounds(layout, z_knots, v_star_arr, T_arr, margin)

    expected_bound = 2.0 / np.sqrt(v_star_val * T_val)

    # lee_upper_T0: u = v_star * bound - margin
    upper_name_idx = cs.names.index("lee_upper_T0")
    assert cs.u[upper_name_idx] == pytest.approx(v_star_val * expected_bound - margin)
    assert np.isneginf(cs.l[upper_name_idx])

    # lee_lower_T0: l = -v_star * bound + margin
    lower_name_idx = cs.names.index("lee_lower_T0")
    assert cs.l[lower_name_idx] == pytest.approx(-v_star_val * expected_bound + margin)
    assert np.isinf(cs.u[lower_name_idx]) and cs.u[lower_name_idx] > 0

    # Sanity: the bound is not, e.g., 1/sqrt(v_star*T) or 2/sqrt(v_star) or
    # 2*sqrt(v_star*T) -- confirm numerically distinct from common mistakes.
    assert expected_bound != pytest.approx(1.0 / np.sqrt(v_star_val * T_val))
    assert expected_bound != pytest.approx(2.0 / np.sqrt(v_star_val))
    assert expected_bound != pytest.approx(2.0 * np.sqrt(v_star_val * T_val))


# ---------------------------------------------------------------------------
# generic_no_calendar_spread
# ---------------------------------------------------------------------------


def test_generic_no_calendar_spread_row_count() -> None:
    """Paper, Section "No-Calendar-Spread-Arbitrage Constraints": one row
    per strike per adjacent expiry pair -> (n_expiries-1) * n_strikes rows."""
    z_knots = np.linspace(-3.0, 3.0, 5)
    n_expiries = 3
    n_strikes = 5
    expiries, layout = _make_layout(n_expiries, z_knots)
    v_star = np.array([e.v_star for e in expiries])
    T = np.array([e.T for e in expiries])

    cs = generic_no_calendar_spread(layout, z_knots, v_star, T, n_strikes)

    assert cs.A.shape[0] == (n_expiries - 1) * n_strikes
    assert np.all(np.isneginf(cs.l))
    assert np.all(cs.u == 0.0)


def test_generic_no_calendar_spread_n_expiries_1_produces_no_rows() -> None:
    """Edge case: with a single expiry there is no adjacent pair, so no
    calendar-spread rows should be generated."""
    z_knots = np.linspace(-3.0, 3.0, 5)
    expiries, layout = _make_layout(1, z_knots)
    v_star = np.array([e.v_star for e in expiries])
    T = np.array([e.T for e in expiries])

    cs = generic_no_calendar_spread(layout, z_knots, v_star, T, n_strikes=5)
    assert cs.A.shape[0] == 0
    assert cs.names == []

    cs_tails = no_calendar_spread_tails(layout, z_knots, v_star, T)
    assert cs_tails.A.shape[0] == 0
    assert cs_tails.names == []


def test_generic_no_calendar_spread_n_strikes_2() -> None:
    """Edge case: smallest meaningful discretization, n_strikes=2."""
    z_knots = np.linspace(-3.0, 3.0, 5)
    n_expiries = 3
    expiries, layout = _make_layout(n_expiries, z_knots)
    v_star = np.array([e.v_star for e in expiries])
    T = np.array([e.T for e in expiries])

    cs = generic_no_calendar_spread(layout, z_knots, v_star, T, n_strikes=2)
    assert cs.A.shape[0] == (n_expiries - 1) * 2


# ---------------------------------------------------------------------------
# assemble_all_constraints
# ---------------------------------------------------------------------------


def test_assemble_all_constraints_butterfly_rows_are_additive() -> None:
    """Paper, Section "Putting It All Together": the linearized butterfly
    rows (PDF >= 0 interior + edges) are additive on top of the other six
    constraint types. include_linearized_butterfly=True must therefore
    produce strictly more rows than False, given a valid reference."""
    z_knots = np.linspace(-3.0, 3.0, 7)
    n_strikes = 4
    lee_margin = 1e-4
    expiries, layout = _make_layout(2, z_knots)

    cs_without = assemble_all_constraints(
        expiries, layout, z_knots, n_strikes, lee_margin, include_linearized_butterfly=False
    )

    z_grid = np.linspace(z_knots[0], z_knots[-1], n_strikes)
    reference = []
    for exp in expiries:
        reference.append(
            {
                "v": np.full(n_strikes, exp.v_star),
                "s": np.zeros(n_strikes),
                "d1": np.full(n_strikes, 2.0),
                "d2": np.full(n_strikes, 2.0),
                "v_edge": np.full(2, exp.v_star),
            }
        )

    cs_with = assemble_all_constraints(
        expiries, layout, z_knots, n_strikes, lee_margin,
        include_linearized_butterfly=True, reference=reference,
    )

    assert cs_with.A.shape[0] > cs_without.A.shape[0]
    assert cs_with.A.shape[1] == cs_without.A.shape[1] == layout.total_size


def test_assemble_all_constraints_butterfly_without_reference_is_handled() -> None:
    """Inspecting the actual implementation: assemble_all_constraints has
    an explicit ``assert reference is not None`` guard when
    include_linearized_butterfly=True. Calling it with reference=None must
    therefore raise an AssertionError with a clear message, not silently
    proceed or crash with an unrelated error."""
    z_knots = np.linspace(-3.0, 3.0, 7)
    expiries, layout = _make_layout(2, z_knots)

    with pytest.raises(AssertionError, match="reference"):
        assemble_all_constraints(
            expiries, layout, z_knots, n_strikes=4, lee_margin=1e-4,
            include_linearized_butterfly=True, reference=None,
        )


def test_assemble_all_constraints_n_expiries_1_has_no_calendar_rows() -> None:
    """Edge case: n_expiries=1 means the full assembled constraint set must
    contain no calendar-spread rows (generic or tails), identifiable by
    name prefix."""
    z_knots = np.linspace(-3.0, 3.0, 7)
    expiries, layout = _make_layout(1, z_knots)

    cs = assemble_all_constraints(
        expiries, layout, z_knots, n_strikes=4, lee_margin=1e-4, include_linearized_butterfly=False
    )
    assert not any("calendar_spread" in name for name in cs.names)


def test_assemble_all_constraints_n_strikes_2() -> None:
    """Edge case: smallest meaningful discretization n_strikes=2 does not
    break constraint assembly and still produces a well-formed ConstraintSet."""
    z_knots = np.linspace(-3.0, 3.0, 7)
    expiries, layout = _make_layout(2, z_knots)

    cs = assemble_all_constraints(
        expiries, layout, z_knots, n_strikes=2, lee_margin=1e-4, include_linearized_butterfly=False
    )
    assert isinstance(cs, ConstraintSet)
    assert cs.A.shape[0] == len(cs.l) == len(cs.u) == len(cs.names)
    assert cs.A.shape[0] > 0
    assert cs.A.shape[1] == layout.total_size
