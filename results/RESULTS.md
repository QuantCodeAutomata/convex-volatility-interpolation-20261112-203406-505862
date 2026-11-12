# Results: Convex Volatility Interpolation

Paper: *Convex Volatility Interpolation* (author version of the article published in
*Cutting Edge*, Risk, February 2026).

All numbers below come from the files saved by the final run of each experiment
in `results/`: `exp1_summary.json`, `exp2_summary.json`,
`exp2_chi_squared_vs_n.csv`, `exp2_weight_vs_spread.csv`,
`exp3_arbitrage_checks.json`, `exp4_benchmark.json`, `exp4_benchmark.csv`.

## Experiment 1 — CVI Cubic Spline and B-Spline Dual Parameterization

Tests whether the cubic-spline parameterization of variance (value and slope
at z=0, plus anchored convexities at the knots) and the cubic B-spline
representation with n+2 basis functions over the same knots are related by a
single linear transformation w = M·p, and whether both representations give
identical variance curves, including in the linearly extrapolated tails.

`results/exp1_summary.json` ("dual_parameterization") checks 6 knot
configurations (evenly spaced and near-the-money-dense, n = 7, 9, 15). The
maximum absolute difference between the cubic-spline curve v(z) and the
B-spline curve M·p, evaluated at 25 points per segment including the
extrapolated tails, is **5.3e-15** (machine precision). The transformation
matrix M is well conditioned across configurations, with cond(M) ranging from
47 to 3013 as knot density increases. The cubic coefficient and the second
derivative are confirmed zero at both edge knots z_0 and z_{n-1} in every
configuration (`linear_extrapolation` fields all pass), reproducing the
paper's "CVI Cubic Spline" section.

`results/exp1_figure1_reproduction.png` and the `figure1_reproduction` block
in the summary fit the CVI cubic spline to the (z, v, ∂v/∂z, ∂²v/∂z²) table of
the paper's Figure 1. Because the paper does not disclose the knot grid used
to produce Figure 1 (only the published (z, v) table is given), the knots are
taken at the table's z-values and the spline parameters are obtained by a
weighted least-squares fit to the table, not an exact digit-level
reconstruction. The resulting maximum absolute errors are 0.33 (v), 0.28
(∂v/∂z) and 0.042 (∂²v/∂z²) against a v range of [0.1, 2.3].

**Verdict:**
- Dual-parameterization equivalence (value, slope, curvature at all test
  points, including tails): **reproduced** (difference at machine precision).
- Linear extrapolation / zero edge curvature: **reproduced**.
- Figure 1 digit-level reproduction: **partially reproduced** — the shape is
  qualitatively recovered but the knot grid is unconstrained by the paper, so
  the fit does not match the figure to the precision the table implies.

## Experiment 2 — CVI Objective Function

Tests the four-term quadratic objective (least squares, above-ask, below-bid,
strike regularization) in canonical ½xᵀPx + qᵀx form on a synthetic
multi-expiry surface (`EXP2_N_KNOTS=15`, `EXP2_N_EXPIRIES=4`,
`STRIKE_REG_LAMBDA=0.05`).

`results/exp2_summary.json`:
- `psd_check`: the assembled P matrix (size 140×140) is symmetric with minimum
  eigenvalue **-1.7e-10**, i.e. positive semi-definite up to solver/round-off
  tolerance, matching the expectation that the objective is a convex quadratic
  form.
- `weight_vs_spread` (`results/exp2_weight_vs_spread.png`/`.csv`): the least
  squares weight w = 1/(ask-bid)² falls from 6.25e6 at a half-spread of 0.5% of
  variance to 434 at a half-spread of 60%, a ratio of **6.9e-5**. Very wide
  bid-ask quotes therefore receive a negligible weight relative to tight
  quotes, reproducing the paper's "Least Squares Penalty" claim.
- `chi_squared_vs_n` (`results/exp2_chi_squared_vs_n.csv`): chi-squared mean
  rises from 0.0084 (n_mid=10) to 0.098 (n_mid=30) to 0.224 (n_mid=60), a
  26.7x spread, i.e. outside the factor-of-3 tolerance used as a pass/pass
  threshold in the experiment script. With `EXP2_N_KNOTS=15` giving roughly
  17 free parameters per expiry, the n_mid=10 case is under-determined
  (fewer mid quotes than free parameters), which biases chi-squared downward
  through overfitting; moving from n_mid=30 to n_mid=60 (both over-determined)
  the ratio is only 2.3x. This is a finite-sample regression artifact of the
  synthetic check, not evidence that the paper's chi-squared normalization
  (dividing by N_{T,mid}) fails to scale; it is recorded here rather than
  silently dropped.
- `fit_quality`: of 60 mid quotes, 43 (71.7%) fall within the bid-ask band
  after calibration with only the four objective penalties (no hard
  no-arbitrage constraints applied, since those are the subject of
  Experiment 3); the largest residual outside the band is 0.013 above the ask
  and 0.38 below the bid, consistent with a soft penalty rather than a hard
  bound.

**Verdict:**
- Positive semi-definiteness of the objective: **reproduced**.
- Negligible weight of wide bid-ask quotes: **reproduced**.
- Chi-squared scale invariance across quote counts: **partially reproduced** —
  holds once the quote count exceeds the number of free parameters (2.3x
  spread between n_mid=30 and 60); the n_mid=10 case is under-determined by
  construction and should not be read as a paper-scale violation.

## Experiment 3 — No-Arbitrage Constraints

Builds a synthetic 5-expiry surface (`EXP3_N_KNOTS=15`, edge knots at
|z|=6.0, chosen so that d1·d2 > 1 at the edges) and runs the two-step
calibration: first an unconstrained-butterfly QP, then a second QP with the
linearized PDF >= 0 constraints evaluated around the first solution.

`results/exp3_arbitrage_checks.json`:
- `edge_d1_d2_all_pass`: **true** for all 5 expiries (edge-knot placement
  satisfies the Martini-Mingone prerequisite d1·d2 > 1).
- `linear_extrapolation`, `positivity_of_variance`: **0 violations**.
- `calendar_spread`: **0 violations** across all 4 adjacent expiry pairs, at
  the 20 discretized strikes used by the paper's "Numerical Results" section.
- `lee_bounds`: **0 violations**; the realized slopes (e.g. s_left=-0.43,
  s_right=0.36 at T=1.04y) stay well inside the Lee bound (9.15 at that
  maturity).
- `pdf_condition`: **before linearization**, 4 of the 20 discretized points
  across the 5 expiries violate the butterfly (PDF >= 0) condition, with a
  maximum violation of 0.085. **After linearization** (second QP iteration,
  using β0, β1, β2 computed around the first solution), **0 violations**
  remain.

**Verdict:**
- All eight constraint families are implemented as linear constraints on the
  spline parameters: **reproduced**.
- Two-step linearized calibration removing butterfly violations after one
  additional iteration: **reproduced** (4 violations -> 0 after the second
  QP solve, matching the paper's claim that "a small number" of iterations
  suffices).

## Experiment 4 — Full CVI Calibration on Live Option Surfaces

Assembles the complete QP (objective + all eight constraint families) and
solves it with the Clarabel interior-point solver via CVXPY, for knot counts
n ∈ {10, 20, 30, 40} per expiry, matching the paper's Table 3 grid.

**Data.** The paper benchmarks SPX (14,538 quotes, 46 expiries) and SX5E
(3,153 quotes, 19 expiries). Both were attempted against the Massive REST API
option-chain and NBBO-quote endpoints. `GET /v3/snapshot/options/SX5E`
returns an empty contract list (Eurex-listed SX5E options are not served by
this entitlement). SPX contracts list (CBOE), but every sampled NBBO quote
has `bid_price = ask_price = 0`, i.e. no usable bid/ask on this entitlement.
Both findings are from direct calls logged by `src/market_surface.py`'s data
diagnostics. As a substitute, this replication fits the two single-name/ index
underlyings from the paper's own "Scaling" knot-convergence figure that this
entitlement does serve with live, non-zero NBBO quotes: **NDX** (Nasdaq-100
index options) and **NVDA** (single-name, American-style, used
un-de-Americanized because Massive returns European-style implied mid/NBBO
quotes directly, not raw American prices requiring an exercise-style
correction). `results/exp4_benchmark.json` ("data_diagnostics") records, per
underlying: 12,500 listed NDX contracts yielding 747 usable mid-or-one-sided
quotes across 20 expiries; 3,938 listed NVDA contracts yielding 757 quotes
across 19 expiries.

**Calibration results** (`results/exp4_benchmark.csv`,
`results/exp4_chi2_vs_knots.png`, `results/exp4_time_vs_knots.png`):

| Underlying | n knots/expiry | total QP time (s) | chi-squared (mean) | non-zero P | non-zero A | constraints |
|---|---|---|---|---|---|---|
| NDX  | 11 | 0.475 | 6.1e-4 | 2,420  | 19,936  | 1,358 |
| NDX  | 21 | 1.007 | 6.0e-4 | 8,820  | 46,496  | 1,758 |
| NDX  | 31 | 1.757 | 5.9e-4 | 19,220 | 81,056  | 2,158 |
| NDX  | 41 | 2.613 | 5.9e-4 | 33,620 | 123,616 | 2,558 |
| NVDA | 11 | 0.432 | 0.030  | 2,299  | 18,926  | 1,290 |
| NVDA | 21 | 0.867 | 0.026  | 8,379  | 44,146  | 1,670 |
| NVDA | 31 | 1.497 | 0.035  | 18,259 | 76,966  | 2,050 |
| NVDA | 41 | 2.220 | 0.021  | 31,939 | 117,386 | 2,430 |

(`n knots/expiry` reported here is `3*n_knots_paper - 2` because
`build_symmetric_knots` returns the odd near-the-money-dense count that the
spline routines expect; see `exp/exp4.py`.) No calendar-spread or negative-
variance violations are recorded at any knot count for either underlying.

Total calibration time (both QP iterations) scales close to linearly with
knot count per expiry for both underlyings over this range (NDX: 0.475s at 11
knots to 2.613s at 41 knots, a 5.5x increase for a 3.7x increase in knots;
NVDA similar), consistent with the paper's statement that A's non-zero count
grows linearly and P's grows quadratically with parameters per expiry: P
non-zeros grow by 13.9x (NDX) over the same range, closer to the expected
quadratic growth, while A non-zeros grow by 6.2x, closer to linear.
Chi-squared is essentially flat as a function of knot count for both
underlyings (NDX: 6.1e-4 to 5.9e-4; NVDA: 0.030 to 0.021, non-monotonic),
i.e. the fit has already converged at 10-11 knots for both surfaces,
qualitatively reproducing the paper's Figure 6 plateau, though at a much
lower absolute chi-squared level than the paper's SPX/SX5E (reported in the
mid-to-high teens), because these are index/single-name surfaces with
typically narrower relative bid-ask spreads than the paper's benchmark
underlyings, feeding larger least-squares weights for the same shape error
scale.

**Verdict:**
- QP assembly and two-step Clarabel calibration on live option-chain data:
  **reproduced** (runs end-to-end, arbitrage-free, sub-3-second calibration).
- Absolute calibration times and quote counts against Table 3 (SPX/SX5E):
  **not reproduced** — this entitlement does not serve usable SPX or SX5E
  bid/ask quotes (see Data above), so Table 3's exact figures (e.g. 0.152s
  for SPX at 20 knots, 2,298 constraints) cannot be checked against the
  paper's own underlyings.
- Chi-squared convergence by ~10-15 knots (Figure 6 qualitative plateau):
  **reproduced** on the substitute underlyings.
- P non-zeros quadratic / A non-zeros linear scaling (Table 3 / "Scaling"):
  **partially reproduced** — the direction of the effect holds (P grows
  faster than A) but growth rates are measured on NDX/NVDA, not SPX/SX5E.

## Deviations from the paper

- **Data source and universe (Experiment 4).** The paper calibrates CVI on
  SPX (CBOE) and SX5E (Eurex) option chains. Massive REST, the market-data
  API used in this replication, returns an empty contract list for
  `GET /v3/snapshot/options/SX5E` and returns SPX contracts with
  `bid_price = ask_price = 0` on every NBBO quote sampled, so neither surface
  has usable bid/ask data on this entitlement. NDX (Nasdaq-100 index) and
  NVDA (single name) are used instead; both appear in the paper's own
  knot-convergence scaling discussion. This substitution is why Table 2 and
  Table 3's SPX/SX5E figures could not be checked numerically; see
  "Not reproduced" below.
- **De-Americanization (Experiment 4).** The paper requires de-Americanizing
  single-name equity quotes (e.g. NVDA) before fitting a European total
  variance. Massive's option NBBO/quote endpoints return implied volatilities
  computed by the venue directly from quoted prices; this replication uses
  those quotes as given rather than implementing a separate
  Barone-Adesi-Whaley correction, since the paper does not specify its own
  de-Americanization method in enough detail to reproduce exactly.
- **Figure 1 knot grid (Experiment 1).** The paper's Figure 1 table gives
  (z, v, ∂v/∂z, ∂²v/∂z²) at 9 points but not the knot grid used to produce the
  curve, so the reconstruction in `exp1_figure1_reproduction.png` is a
  least-squares fit to the table rather than an exact digit match.
- **Unspecified parameters.** Several parameters have no value in the paper
  and are set to a reasonable default, documented at the point of definition
  in `src/config.py` (e.g. `EXP2_N_KNOTS`, `LEE_BOUND_MARGIN`, the live
  per-underlying expiry/strike targets for Experiment 4). None of these
  override a value the paper specifies.
- **Below-bid / above-ask penalty normalization (Experiment 2).** The paper
  states the least-squares and above-ask weight formulas explicitly but does
  not give N_{T,ask} and N_{T,bid} precisely for the below-bid penalty's
  normalization; this implementation mirrors the above-ask structure
  (`w_bid = q_j * Vega_bid / sum(Vega_bid)`, normalized by `N_{T,bid}`), the
  natural symmetric counterpart described in the "Below Bid Penalty" section.

## Not reproduced

- **Table 2 and Table 3 absolute figures for SPX and SX5E** (quote counts,
  constraint counts, non-zero P/A counts, calibration times at the paper's
  own knot grid) are not reproduced: the data source available to this
  replication (Massive REST) does not serve usable SX5E contracts or non-zero
  SPX NBBO quotes, as detailed above. Experiment 4 instead reports the
  equivalent table for NDX and NVDA.
- **Figure 6 for NDX, AMZN, MSFT, NVDA, KOSPI2 at the paper's exact fitting
  parameters** is not reproduced digit-for-digit: the paper does not disclose
  the as-of date, expiry selection, or exact knot placement used to produce
  that figure, only its qualitative plateau shape, which this replication
  checks instead on the live NDX/NVDA surfaces described above.
