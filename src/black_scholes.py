"""Black-Scholes pricing and vega in forward/variance terms.

Used to weight the Above Ask / Below Bid penalties (paper, Section
"Above Ask Penalty": "the weights are proportional to the vega derived
from the ask ... price"). Options are European on the forward F_T with
total variance v*T (vT), as is standard for a total-variance volatility
surface.

Custom implementation: the paper does not specify a vega library; the
closed-form Black (1976) vega is the elementary, well-known formula and is
implemented directly rather than pulled from a general options-pricing
package, matching standard total-variance notation v*T = sigma^2*T.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def d1_d2_from_total_variance(k: np.ndarray, vT: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """d1, d2 of Black's formula expressed with log-forward-moneyness k and
    total variance vT = sigma^2 * T (paper's notation, Section "No-Butterfly
    -Arbitrage Constraints": d_{1,2} = (-k +/- vT/2) / sqrt(vT))."""
    k = np.asarray(k, dtype=float)
    vT = np.asarray(vT, dtype=float)
    sqrt_vT = np.sqrt(vT)
    d1 = (-k + vT / 2.0) / sqrt_vT
    d2 = (-k - vT / 2.0) / sqrt_vT
    return d1, d2


def black_vega(forward: np.ndarray, strike: np.ndarray, total_variance: np.ndarray) -> np.ndarray:
    """Black (1976) vega d(Price)/d(sigma), undiscounted, per unit forward.

    vega = F * phi(d1) * sqrt(T), expressed directly from total variance
    vT = sigma^2 * T via d1 = (-k + vT/2)/sqrt(vT), k = log(K/F):
    vega = F * phi(d1) * sqrt(vT) / sigma = F * phi(d1) * sqrt(T).
    Since sigma = sqrt(vT / T) and we only need vega up to a positive
    common scale (it enters the Above Ask / Below Bid weights only through
    a ratio that normalizes it away), we return F * phi(d1) * sqrt(vT),
    which is proportional to the standard Black vega for a fixed T.
    """
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    total_variance = np.asarray(total_variance, dtype=float)
    if np.any(forward <= 0) or np.any(strike <= 0):
        raise ValueError("forward and strike must be strictly positive")
    if np.any(total_variance <= 0):
        raise ValueError("total_variance must be strictly positive")
    k = np.log(strike / forward)
    d1, _ = d1_d2_from_total_variance(k, total_variance)
    return forward * norm.pdf(d1) * np.sqrt(total_variance)


def black_undiscounted_price(
    forward: float, strike: float, total_variance: float, is_call: bool
) -> float:
    """Undiscounted Black (1976) price of a European option on the forward,
    F * [N(d1) - K/F * N(d2)] for a call (put via parity), with total
    variance vT = sigma^2*T. Used only to invert market bid/ask quotes to
    total variance (Section "Notation": options are priced on the forward,
    so no discount factor is required to recover v(K,T) itself)."""
    if total_variance <= 0.0:
        return max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    k = np.log(strike / forward)
    d1, d2 = d1_d2_from_total_variance(np.array(k), np.array(total_variance))
    d1, d2 = float(d1), float(d2)
    if is_call:
        return forward * norm.cdf(d1) - strike * norm.cdf(d2)
    return strike * norm.cdf(-d2) - forward * norm.cdf(-d1)


def implied_total_variance(
    price: float, forward: float, strike: float, is_call: bool, t_max: float = 100.0
) -> float | None:
    """Invert the undiscounted Black (1976) price for the total variance
    vT = sigma^2*T via bisection (``scipy.optimize.brentq``). Returns
    ``None`` when the price violates no-arbitrage bounds (outside the
    intrinsic/forward bracket) so the caller can drop the quote rather than
    fabricate a variance.

    Custom implementation: the paper does not specify an inversion routine;
    Black's formula is monotonically increasing in total variance, so a
    scalar root find is the standard, library-free approach.
    """
    intrinsic = max(forward - strike, 0.0) if is_call else max(strike - forward, 0.0)
    upper_bound = forward if is_call else strike
    if price <= intrinsic + 1e-12 or price >= upper_bound - 1e-12:
        return None

    def objective(vT: float) -> float:
        return black_undiscounted_price(forward, strike, vT, is_call) - price

    lo, hi = 1e-8, t_max
    if objective(lo) > 0.0 or objective(hi) < 0.0:
        return None
    return float(brentq(objective, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200))
