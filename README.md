# Convex Volatility Interpolation

Reproduction of *Convex Volatility Interpolation* (author version of the
article published in *Cutting Edge, Risk*, February 2026).

This repository implements the CVI (Convex Volatility Interpolation)
framework for fitting an arbitrage-free, bid-ask aware implied total
variance surface with a convex quadratic program, and reproduces the
paper's four worked examples:

1. **exp_1** — the dual cubic-spline / cubic B-spline parameterization of
   variance in normalized log-moneyness, and the linear map between them
   ("The CVI Parameterization", "CVI Cubic Spline", "CVI B-Spline").
2. **exp_2** — the four-term bid-ask aware objective function (least
   squares, above-ask, below-bid, strike regularization penalties,
   "Objective Function" through "Strike Regularization Penalty").
3. **exp_3** — the eight no-arbitrage / structural linear constraints and
   the two-step linearized-butterfly calibration ("Linear Extrapolation"
   through "Putting It All Together").
4. **exp_4** — full QP assembly, solving with the Clarabel interior-point
   solver, and the knot-count convergence/benchmarking study on market
   option data ("Numerical Results", Tables 2-3, Figure 6).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY / MASSIVE_API_KEY for exp_4
```

## Tests

```bash
python -m pytest -m "not market_data"   # fixtures / synthetic inputs only
python -m pytest                        # includes exp_4's market-data tests
```

## Experiments

```bash
python -m exp.exp1   # dual parameterization + Figure 1 reproduction
python -m exp.exp2   # objective-function penalties
python -m exp.exp3   # no-arbitrage constraints, two-step calibration
python -m exp.exp4   # full calibration, Clarabel benchmarking, knot study
python -m exp.exp4 --full   # paper's knot grid (10..40) and universe size
```

Every run overwrites the tables, plots and logs in `results/` and the
run's JSON/CSV summaries; `results/RESULTS.md` compares them against the
paper and must be checked and updated by hand after a rerun.

See `AGENTS.md` for the repository layout and conventions, and
`results/RESULTS.md` for the comparison against the paper, deviations and
non-reproduced items.
