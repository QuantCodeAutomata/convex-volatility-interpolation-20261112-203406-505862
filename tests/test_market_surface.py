"""Market-data tests for ``src/market_surface.py`` (paper, Experiment 4
data pipeline). Uses the session-scoped ``market_surface_slice`` fixture
from ``tests/conftest.py``, which loads one small live NVDA option surface
through ``data/data_loader.py`` / the Massive REST API and is skipped when
``MASSIVE_API_KEY`` is not set.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.market_data
def test_market_surface_slice_has_usable_expiries(market_surface_slice):
    expiries, diagnostics = market_surface_slice
    assert diagnostics["underlying"] == "NVDA"
    assert diagnostics["n_expiries_used"] >= 1
    assert len(expiries) == diagnostics["n_expiries_used"]


@pytest.mark.market_data
def test_market_surface_expiry_quotes_are_well_formed(market_surface_slice):
    """Each fitted expiry must have a positive maturity and ATM variance,
    mid-quote strikes sorted in increasing normalized log-moneyness ``z``
    (paper, Section "Notation": z := k / (sigma_star sqrt(T))), and
    ask >= bid total variance on every two-sided quote.
    """
    expiries, _ = market_surface_slice
    for eq in expiries:
        assert eq.T > 0.0
        assert eq.v_star > 0.0
        assert eq.z_mid.size >= 3
        assert np.all(np.diff(eq.z_mid) >= 0.0)
        assert np.all(eq.v_ask_mid >= eq.v_bid_mid)
        assert np.all(eq.v_ask_mid > 0.0)
        assert np.all(eq.v_bid_mid >= 0.0)


@pytest.mark.market_data
def test_market_surface_ask_and_bid_only_quotes_have_vega(market_surface_slice):
    """Ask-only / bid-only quotes used by the Above-Ask / Below-Bid
    penalties carry a strictly positive Black-Scholes vega weight (paper,
    Section "Above Ask Penalty"); an option with zero vega cannot drive a
    weighted penalty."""
    expiries, _ = market_surface_slice
    for eq in expiries:
        if eq.z_ask_only.size:
            assert np.all(eq.vega_ask_only > 0.0)
        if eq.z_bid_only.size:
            assert np.all(eq.vega_bid_only > 0.0)
