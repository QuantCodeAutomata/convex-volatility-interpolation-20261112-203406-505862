"""Build CVI ``ExpiryQuotes`` surfaces from live NDX / NVDA option chains.

The paper's exp_4 benchmark (Section "Numerical Results") fits SPX and SX5E
option chains. Confirmed empirically against the Massive REST API (see
``results/RESULTS.md``): SX5E (Eurex-listed) is not served at all
(``GET /v3/snapshot/options/SX5E`` returns an empty result set), and SPX
contracts are listed but every quoted NBBO is zero on both sides
(``GET /v3/quotes/{ticker}`` returns ``bid_price=ask_price=0`` for every
SPX and SPXW contract tested, across near- and far-dated expiries), so no
live SPX smile can be fit either. This replication therefore substitutes
NDX and NVDA, two of the single-name/index underlyings the paper itself
benchmarks in its knot-convergence figure (Section "Scaling"), both of
which return genuine two-sided NBBO quotes.

For every selected expiry this module: locates the forward price via
put-call parity on the quoted strikes nearest the spot, inverts each
quoted bid/ask option price to total variance (``src.black_scholes``),
estimates the ATM anchor variance v_star(T) from the quote nearest the
forward, and partitions strikes into the mid / ask-only / bid-only sets
used by ``src.cvi_objective.ExpiryQuotes`` (paper, Section "Objective
Function": "bids and asks refer to the tightest quotes across puts and
calls" -- implemented here by keeping, for every strike, whichever side
(call or put) gives the narrower total-variance bid-ask spread).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np

from data.data_loader import ContractRef, fetch_option_contracts, fetch_quotes_parallel
from src.black_scholes import black_vega, implied_total_variance
from src.cvi_objective import ExpiryQuotes

# Band around the spot price, in absolute log-moneyness, within which
# strikes are considered candidates for quoting. Not specified in the
# paper; wide enough to cover the smile wings of a 1-day-to-2-year option
# chain while keeping the number of NBBO requests bounded.
LOG_MONEYNESS_BAND = 0.9


@dataclass
class _Quote:
    bid: float
    ask: float

    @property
    def two_sided(self) -> bool:
        return self.bid > 0.0 and self.ask > self.bid

    @property
    def ask_only(self) -> bool:
        return self.bid <= 0.0 and self.ask > 0.0


def _year_fraction(expiration: str, asof: dt.date) -> float:
    """ACT/365 year fraction from ``asof`` to the option's expiration date.
    Not specified in the paper (a live snapshot has no paper-given as-of
    date); ACT/365 is the standard day-count convention absent a paper
    specification."""
    exp_date = dt.date.fromisoformat(expiration)
    return max((exp_date - asof).days, 1) / 365.0


def _approx_atm_strike(calls: dict[float, ContractRef], puts: dict[float, ContractRef]) -> float:
    """A rough at-the-money strike located without a spot price, from the
    previous-session reference prices alone: at the forward, put-call parity
    gives call price ~= put price, so the strike minimizing the absolute
    reference-price gap between the call and the put is close to the
    forward. Used only to center the strike window before quoting; the
    forward itself is re-estimated precisely from live NBBO parity
    (``build_expiry_quotes``)."""
    common = sorted(set(calls) & set(puts))
    if not common:
        raise RuntimeError("no strike has both a call and a put contract")
    gaps = {
        k: abs((calls[k].reference_price or np.nan) - (puts[k].reference_price or np.nan))
        for k in common
    }
    gaps = {k: v for k, v in gaps.items() if np.isfinite(v)}
    if not gaps:
        return common[len(common) // 2]
    return min(gaps, key=gaps.get)


def _select_strikes(
    calls: dict[float, ContractRef], puts: dict[float, ContractRef], spot: float | None, n_target: int
) -> list[float]:
    strikes = sorted(set(calls) | set(puts))
    if spot is not None:
        log_money = {k: abs(np.log(k / spot)) for k in strikes}
        strikes = [k for k in strikes if log_money[k] <= LOG_MONEYNESS_BAND]
        strikes.sort(key=lambda k: log_money[k])
        return sorted(strikes[:n_target])
    # No spot available (index underlyings on this entitlement): center the
    # window on the strike nearest the forward implied by reference prices.
    atm = _approx_atm_strike(calls, puts)
    strikes.sort(key=lambda k: abs(k - atm))
    return sorted(strikes[:n_target])


def _invert(quote: _Quote, forward: float, strike: float, is_call: bool) -> tuple[float | None, float | None]:
    """Total variance implied by the bid and ask legs of ``quote`` (``None``
    when a leg violates the no-arbitrage price bounds)."""
    vT_bid = implied_total_variance(quote.bid, forward, strike, is_call) if quote.bid > 0.0 else None
    vT_ask = implied_total_variance(quote.ask, forward, strike, is_call) if quote.ask > 0.0 else None
    return vT_bid, vT_ask


def build_expiry_quotes(
    underlying: str,
    expiration: str,
    contracts: list[ContractRef],
    spot: float,
    asof: dt.date,
    n_strikes_target: int,
) -> tuple[ExpiryQuotes | None, dict[str, Any]]:
    """One ``ExpiryQuotes`` for a single expiration date, built from live
    NBBO quotes. Returns ``(None, diagnostics)`` if too few usable quotes
    remain (fewer than 3 strikes with any usable side)."""
    calls = {c.strike: c for c in contracts if c.contract_type == "call"}
    puts = {c.strike: c for c in contracts if c.contract_type == "put"}
    strikes = _select_strikes(calls, puts, spot, n_strikes_target)
    tickers = [calls[k].ticker for k in strikes if k in calls] + [puts[k].ticker for k in strikes if k in puts]
    raw_quotes = fetch_quotes_parallel(tickers)

    T = _year_fraction(expiration, asof)

    def quote_of(ticker: str | None) -> _Quote | None:
        if ticker is None or ticker not in raw_quotes:
            return None
        bid, ask = raw_quotes[ticker]
        return _Quote(bid=bid, ask=ask)

    # Step 1: forward price via put-call parity (undiscounted: C - P = F - K,
    # paper, Section "Notation": options are priced on the forward), using
    # the strikes nearest the spot (or, absent a spot, the approximate
    # at-the-money strike) where both legs are two-sided.
    center = spot if spot is not None else _approx_atm_strike(calls, puts)
    parity_strikes = sorted(strikes, key=lambda k: abs(k - center))
    parity_estimates = []
    for k in parity_strikes:
        c, p = quote_of(calls.get(k) and calls[k].ticker), quote_of(puts.get(k) and puts[k].ticker)
        if c is not None and c.two_sided and p is not None and p.two_sided:
            c_mid, p_mid = (c.bid + c.ask) / 2.0, (p.bid + p.ask) / 2.0
            parity_estimates.append(k + c_mid - p_mid)
        if len(parity_estimates) >= 5:
            break
    forward = float(np.median(parity_estimates)) if parity_estimates else float(center)

    # Step 2: classify every strike's tightest leg into mid / ask-only / bid-only.
    mid_k, mid_vb, mid_va = [], [], []
    ask_only_k, ask_only_v = [], []
    bid_only_k, bid_only_v = [], []
    for k in strikes:
        legs = []
        for is_call, ticker in ((True, calls.get(k) and calls[k].ticker), (False, puts.get(k) and puts[k].ticker)):
            q = quote_of(ticker)
            if q is None:
                continue
            vb, va = _invert(q, forward, k, is_call)
            legs.append((is_call, q, vb, va))

        two_sided_legs = [(vb, va) for _, q, vb, va in legs if q.two_sided and vb is not None and va is not None]
        if two_sided_legs:
            vb, va = min(two_sided_legs, key=lambda pair: pair[1] - pair[0])
            mid_k.append(k)
            mid_vb.append(vb)
            mid_va.append(va)
            continue

        ask_candidates = [va for _, q, _, va in legs if q.ask_only and va is not None]
        if ask_candidates:
            ask_only_k.append(k)
            ask_only_v.append(min(ask_candidates))
            continue

        bid_candidates = [vb for _, q, vb, _ in legs if (not q.two_sided and not q.ask_only) and vb is not None]
        if bid_candidates:
            bid_only_k.append(k)
            bid_only_v.append(max(bid_candidates))

    diagnostics = {
        "expiration": expiration,
        "T": T,
        "forward": forward,
        "n_mid": len(mid_k),
        "n_ask_only": len(ask_only_k),
        "n_bid_only": len(bid_only_k),
        "n_strikes_requested": len(strikes),
    }
    if len(mid_k) < 3:
        return None, diagnostics

    # Step 3: anchor ATM variance v_star(T) from the mid quote nearest the forward.
    mid_k_arr = np.array(mid_k)
    atm_idx = int(np.argmin(np.abs(mid_k_arr - forward)))
    v_star = float((mid_vb[atm_idx] + mid_va[atm_idx]) / 2.0 / T)
    diagnostics["v_star"] = v_star

    def to_z(k_arr: np.ndarray) -> np.ndarray:
        return np.log(k_arr / forward) / (np.sqrt(v_star) * np.sqrt(T))

    z_mid = to_z(mid_k_arr)
    order = np.argsort(z_mid)
    z_mid, v_bid_mid, v_ask_mid = z_mid[order], np.array(mid_vb)[order], np.array(mid_va)[order]

    ask_k_arr = np.array(ask_only_k)
    z_ask_only = to_z(ask_k_arr) if ask_k_arr.size else np.zeros(0)
    v_ask_only = np.array(ask_only_v)
    vega_ask_only = black_vega(np.full(ask_k_arr.size, forward), ask_k_arr, v_ask_only * T) if ask_k_arr.size else np.zeros(0)

    bid_k_arr = np.array(bid_only_k)
    z_bid_only = to_z(bid_k_arr) if bid_k_arr.size else np.zeros(0)
    v_bid_only = np.array(bid_only_v)
    vega_bid_only = black_vega(np.full(bid_k_arr.size, forward), bid_k_arr, v_bid_only * T) if bid_k_arr.size else np.zeros(0)

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
    ), diagnostics


def load_market_surface(
    underlying: str,
    n_expiries_target: int,
    n_strikes_target: int,
    min_days: int = 2,
    max_days: int = 760,
) -> tuple[list[ExpiryQuotes], dict[str, Any]]:
    """Live option surface for ``underlying``: the ``n_expiries_target``
    nearest-term listed expiries within ``[min_days, max_days]`` (the
    paper gives no explicit expiry-selection rule for the live chain; the
    nearest-dated, most liquid expiries are used, consistent with the
    download-time liquidity-ranking rule applied elsewhere in this
    replication), each built by ``build_expiry_quotes``.
    """
    asof = dt.datetime.utcnow().date()
    exp_gte = (asof + dt.timedelta(days=min_days)).isoformat()
    exp_lte = (asof + dt.timedelta(days=max_days)).isoformat()
    contracts, spot = fetch_option_contracts(underlying, exp_gte, exp_lte)

    by_expiration: dict[str, list[ContractRef]] = {}
    for c in contracts:
        by_expiration.setdefault(c.expiration, []).append(c)
    expirations = sorted(by_expiration)[:n_expiries_target]

    expiries: list[ExpiryQuotes] = []
    per_expiry_diagnostics = []
    dropped = []
    for expiration in expirations:
        eq, diag = build_expiry_quotes(
            underlying, expiration, by_expiration[expiration], spot, asof, n_strikes_target
        )
        per_expiry_diagnostics.append(diag)
        if eq is None:
            dropped.append(expiration)
        else:
            expiries.append(eq)

    n_quotes = sum(e.z_mid.size + e.z_ask_only.size + e.z_bid_only.size for e in expiries)
    diagnostics = {
        "underlying": underlying,
        "spot": spot,
        "asof": asof.isoformat(),
        "n_contracts_listed": len(contracts),
        "n_expiries_requested": len(expirations),
        "n_expiries_used": len(expiries),
        "dropped_expiries": dropped,
        "n_quotes_total": n_quotes,
        "per_expiry": per_expiry_diagnostics,
    }
    if not expiries:
        raise RuntimeError(f"{underlying}: no expiry produced a usable surface ({dropped})")
    return expiries, diagnostics
