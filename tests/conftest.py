"""Shared pytest configuration: registers the ``market_data`` marker used by
tests that call the live Massive REST API (``data/data_loader.py``,
``src/market_surface.py``), and a session-scoped fixture that loads one
small live expiry once per test session.
"""

from __future__ import annotations

import os

import pytest

from dotenv import load_dotenv


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: test calls the live Massive REST API (needs MASSIVE_API_KEY)"
    )


@pytest.fixture(scope="session")
def market_surface_slice():
    """One small live NVDA option surface: two near-dated expiries, 15
    strikes each. Session-scoped so the network call happens at most once
    per test run; skipped entirely when ``MASSIVE_API_KEY`` is absent.
    """
    load_dotenv()
    if not os.environ.get("MASSIVE_API_KEY"):
        pytest.skip("MASSIVE_API_KEY not set: skipping market-data test")
    from src.market_surface import load_market_surface

    expiries, diagnostics = load_market_surface("NVDA", n_expiries_target=2, n_strikes_target=15)
    return expiries, diagnostics
