"""Synthetic option-quote surfaces used by the CVI experiments that do not
require market data (paper, exp_1 "No market data required"; exp_2/exp_3
"synthetic or real SPX/SX5E option chains can be used").

A smooth reference total-variance smile (a simple quadratic-in-moneyness
model, flattened in the wings) is sampled at a set of normalized
log-moneyness points per expiry, perturbed with a bid-ask spread and
Gaussian pricing noise, to build ``ExpiryQuotes`` objects usable directly
by ``src.cvi_objective`` / ``src.cvi_constraints`` / ``src.cvi_calibration``.
"""

from __future__ import annotations

import numpy as np

from src.black_scholes import black_vega
from src.cvi_objective import ExpiryQuotes


def reference_smile(z: np.ndarray, v_star: float, skew: float = -0.02, curv: float = 0.01) -> np.ndarray:
    """A simple, strictly positive reference annualized-variance smile in
    normalized log-moneyness (quadratic near the money, flattened in the
    wings), used only to generate synthetic option quotes."""
    z = np.asarray(z, dtype=float)
    z_clipped = np.clip(z, -6.0, 6.0)
    return v_star * (1.0 + skew * z_clipped + curv * z_clipped**2)


def make_synthetic_expiry(
    T: float,
    v_star: float,
    n_mid: int,
    n_ask_only: int,
    n_bid_only: int,
    z_range: tuple[float, float],
    bid_ask_half_spread: float,
    noise_std: float,
    rng: np.random.Generator,
) -> ExpiryQuotes:
    """Build one synthetic ``ExpiryQuotes`` with mid, ask-only and bid-only
    quotes drawn from ``reference_smile`` plus a bid-ask spread and
    Gaussian noise (bounded so that bid < ask always holds)."""
    forward = 100.0  # arbitrary synthetic forward; only z enters the CVI fit

    def sample(n: int) -> np.ndarray:
        return rng.uniform(z_range[0], z_range[1], size=n)

    def quotes(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v_mid = reference_smile(z, v_star) + rng.normal(0.0, noise_std, size=z.size)
        v_mid = np.maximum(v_mid, 1e-6)
        half_spread = bid_ask_half_spread * v_mid
        return np.maximum(v_mid - half_spread, 1e-7), v_mid + half_spread

    z_mid = np.sort(sample(n_mid))
    v_bid_mid, v_ask_mid = quotes(z_mid)

    z_ask_only = np.sort(sample(n_ask_only))
    _, v_ask_only = quotes(z_ask_only)
    strikes_ask = forward * np.exp(z_ask_only * np.sqrt(v_star) * np.sqrt(T))
    vega_ask_only = black_vega(np.full_like(strikes_ask, forward), strikes_ask, v_ask_only * T)

    z_bid_only = np.sort(sample(n_bid_only))
    v_bid_only, _ = quotes(z_bid_only)
    strikes_bid = forward * np.exp(z_bid_only * np.sqrt(v_star) * np.sqrt(T))
    vega_bid_only = black_vega(np.full_like(strikes_bid, forward), strikes_bid, v_bid_only * T)

    return ExpiryQuotes(
        T=T,
        v_star=v_star,
        z_mid=z_mid,
        v_bid_mid=v_bid_mid,
        v_ask_mid=v_ask_mid,
        z_ask_only=z_ask_only,
        v_ask_only=v_ask_only,
        vega_ask_only=vega_ask_only,
        z_bid_only=z_bid_only,
        v_bid_only=v_bid_only,
        vega_bid_only=vega_bid_only,
    )


def make_synthetic_surface(
    n_expiries: int,
    z_range: tuple[float, float],
    seed: int,
    n_mid: int = 15,
    n_ask_only: int = 3,
    n_bid_only: int = 3,
    t_min: float = 1.0 / 12.0,
    t_max: float = 2.0,
    v_star_level: float = 0.04,
    bid_ask_half_spread: float = 0.03,
    noise_std: float = 0.002,
) -> list[ExpiryQuotes]:
    """A term structure of ``n_expiries`` synthetic expiries, term
    structure of anchor variance decaying towards ``v_star_level`` as T
    grows (a simple, strictly decreasing-term-structure proxy)."""
    rng = np.random.default_rng(seed)
    T_grid = np.linspace(t_min, t_max, n_expiries)
    expiries = []
    for T in T_grid:
        v_star = v_star_level * (1.0 + 0.3 / (1.0 + T))
        expiries.append(
            make_synthetic_expiry(
                T, v_star, n_mid, n_ask_only, n_bid_only, z_range,
                bid_ask_half_spread, noise_std, rng,
            )
        )
    return expiries
