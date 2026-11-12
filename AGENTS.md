# Repository guide

Reproduction of *Convex Volatility Interpolation* (Cutting Edge, Risk,
February 2026): a convex quadratic-program fit of an arbitrage-free,
bid-ask aware implied total-variance surface (the CVI parameterization).

## Layout

- `src/` — library modules: `cvi_spline.py` (dual cubic-spline / B-spline
  parameterization), `cvi_objective.py` (bid-ask aware penalties),
  `cvi_constraints.py` (no-arbitrage constraints), `cvi_calibration.py`
  (QP assembly and the two-step Clarabel solve), `black_scholes.py`
  (vega weighting), `qp_types.py` (shared QP data types), `config.py`
  (all experiment parameters, as module-level variables with a source
  comment each).
- `exp/` — one script per experiment (`exp1`.. `exp4`), run as
  `python -m exp.exp1`, etc. Only call the `src/` functions with the
  signatures they expose.
- `data/` — `generate_synthetic.py` builds the synthetic option-quote
  surfaces used by exp_1-3; `data_loader.py` fetches live option-chain
  contracts and NBBO quotes from the Massive REST API for exp_4 (used by
  `src/market_surface.py`, which turns raw quotes into `ExpiryQuotes`
  surfaces). No market data is ever committed.
- `tests/` — pytest suite for `src/` (synthetic fixtures) and for the
  market-data path of exp_4 (marked `market_data`, skipped without
  `MASSIVE_API_KEY`).
- `results/` — tables, plots and JSON/CSV logs written by the experiment
  scripts, plus `results/RESULTS.md` (paper comparison, deviations, not
  reproduced).

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"     # unit tests, no network
python -m pytest                          # full suite (needs API keys)
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4                        # reduced knot grid / universe
python -m exp.exp4 --full                 # paper's knot grid and universe
```

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` variables; no classes or dicts there.
- Market data is loaded only through `data/data_loader.py`; synthetic
  data only through `data/generate_synthetic.py`. Neither `src/` nor
  `exp/` fetches or generates data directly.
- `MASSIVE_API_KEY` is read from an uncommitted `.env` (see
  `.env.example`); it never appears in code, logs or results. The CVI
  surface is fit directly from option bid/ask quotes, so the QuantX
  Dataloader (OHLCV bars only) is not used and `DATALOADER_API_KEY` is
  not required by this repository.
- After any rerun, the experiment commands overwrite the files in
  `results/`; the numbers and verdicts in `results/RESULTS.md` must then
  be checked against the new files and updated by hand.
