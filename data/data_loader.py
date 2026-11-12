"""Live option-chain data for the CVI calibration experiment (exp_4), loaded
directly from the Massive REST API.

The CVI paper fits a total-variance surface to bid/ask option quotes
(Section "Objective Function"); it does not use OHLCV bars, so the QuantX
Dataloader (which only serves OHLCV bars) is not applicable here and the
option chain, quotes and contract reference data are loaded directly from
Massive, as the ``massive-rest-market-data`` skill prescribes for data the
Dataloader route does not serve.

Two Massive endpoints are used:

* ``GET /v3/snapshot/options/{underlying}`` (paginated) lists contracts
  (strike, expiration, type) together with a previous-session reference
  price (``day.close``), used only to locate the at-the-money region of
  each expiry before selecting which contracts to quote.
* ``GET /v3/quotes/{option_ticker}`` returns the most recent National Best
  Bid and Offer for one contract; one request per selected contract is
  needed because the snapshot endpoint does not carry live bid/ask fields
  for index-option underlyings on this entitlement.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

API_ROOT = "https://api.massive.com"
RETRY_STATUSES = {429, 503, 504}
MAX_PARALLEL = 8  # concurrent NBBO quote requests


class MassiveDataError(RuntimeError):
    """A failed Massive REST request, or a response missing an expected field."""


def _headers() -> dict[str, str]:
    load_dotenv()
    key = os.environ.get("MASSIVE_API_KEY")
    if not key:
        raise MassiveDataError("MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {key}"}


def _get(path: str, params: dict | None = None) -> dict:
    headers = _headers()
    response = None
    for attempt in range(4):
        try:
            response = requests.get(API_ROOT + path, headers=headers, params=params, timeout=30)
        except requests.exceptions.RequestException as exc:
            if attempt == 3:
                raise MassiveDataError(f"{path}: connection error after retries: {exc}") from exc
            time.sleep(2 * 2**attempt)
            continue
        if response.status_code not in RETRY_STATUSES or attempt == 3:
            break
        time.sleep(float(response.headers.get("Retry-After") or 2 * 2**attempt))
    if response.status_code != 200:
        raise MassiveDataError(f"{path}: HTTP {response.status_code}: {response.text[:300]}")
    return response.json()


@dataclass
class ContractRef:
    ticker: str
    strike: float
    expiration: str
    contract_type: str  # "call" or "put"
    reference_price: float | None  # previous-session close of the option


def fetch_option_contracts(
    underlying: str, expiration_gte: str, expiration_lte: str, max_pages: int = 50
) -> tuple[list[ContractRef], float | None]:
    """All call/put contracts for ``underlying`` expiring in
    ``[expiration_gte, expiration_lte]`` (``GET /v3/snapshot/options/{underlying}``,
    paginated via ``next_url``), their previous-session reference price, and the
    underlying's current price when the snapshot reports one
    (``underlying_asset.price``; absent for index underlyings such as SPX and
    NDX on this entitlement -- the caller then locates the at-the-money
    region from the options' own reference prices via put-call parity)."""
    out: list[ContractRef] = []
    spot: float | None = None
    params = {
        "expiration_date.gte": expiration_gte,
        "expiration_date.lte": expiration_lte,
        "limit": 250,
    }
    path, next_params = f"/v3/snapshot/options/{underlying}", params
    seen_urls: set[str] = set()
    for _ in range(max_pages):
        answer = _get(path, next_params)
        results = answer.get("results")
        if results is None:
            raise MassiveDataError(f"/v3/snapshot/options/{underlying}: no 'results' field")
        for row in results:
            details = row.get("details", {})
            day = row.get("day", {})
            ticker = details.get("ticker")
            if not ticker:
                continue
            if spot is None:
                price = row.get("underlying_asset", {}).get("price")
                if price:
                    spot = float(price)
            out.append(ContractRef(
                ticker=ticker,
                strike=float(details["strike_price"]),
                expiration=details["expiration_date"],
                contract_type=details["contract_type"],
                reference_price=day.get("close"),
            ))
        next_url = answer.get("next_url")
        if not next_url or next_url in seen_urls:
            break
        seen_urls.add(next_url)
        path, next_params = next_url[len(API_ROOT):], None
    if not out:
        raise MassiveDataError(f"{underlying}: no option contracts in [{expiration_gte}, {expiration_lte}]")
    return out, spot


def fetch_latest_quote(contract_ticker: str) -> tuple[float, float] | None:
    """Most recent National Best Bid and Offer for one option contract
    (``GET /v3/quotes/{contract_ticker}``, ``limit=1``). Returns ``(bid, ask)``
    as reported, including a zero bid (offer-only quote) or a zero ask
    (bid-only quote); returns ``None`` only when there is no usable quote at
    all (ask <= 0, i.e. no market on either side) or ``ask <= bid``."""
    answer = _get(f"/v3/quotes/{contract_ticker}", {"limit": 1})
    results = answer.get("results")
    if not results:
        return None
    row = results[0]
    bid, ask = float(row.get("bid_price", 0.0)), float(row.get("ask_price", 0.0))
    if ask <= 0.0 or ask <= bid:
        return None
    return bid, ask


def fetch_quotes_parallel(tickers: list[str]) -> dict[str, tuple[float, float]]:
    """NBBO quotes for many contracts, ``MAX_PARALLEL`` requests at a time."""
    quotes: dict[str, tuple[float, float]] = {}
    with ThreadPoolExecutor(MAX_PARALLEL) as pool:
        for ticker, quote in zip(tickers, pool.map(fetch_latest_quote, tickers)):
            if quote is not None:
                quotes[ticker] = quote
    return quotes
