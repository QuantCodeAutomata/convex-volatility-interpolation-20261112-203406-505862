"""Parameters for the Convex Volatility Interpolation (CVI) replication.

All experiment parameters live here as module-level variables (no classes, no
dicts). Each variable carries a comment stating its meaning, unit and source
(a paper section, or "not specified in the paper").

Paper: Convex Volatility Interpolation (author version of the article
published in Cutting Edge, Risk, February 2026).
"""

# ---------------------------------------------------------------------------
# exp_1: CVI Cubic Spline and B-Spline Dual Parameterization
# ---------------------------------------------------------------------------

# Knot counts tested for the dual-parameterization check (n knots per
# expiry). Paper, "The CVI Parameterization": knot count n is a modeling
# choice, not fixed by the paper; we test several densities.
EXP1_KNOT_COUNTS = (7, 9, 15)

# Figure 1 reproduction table (z, v, dv/dz, d2v/dz2), paper p.2, "CVI Cubic
# Spline", Figure 1 caption table.
EXP1_FIGURE1_Z = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
EXP1_FIGURE1_V = (2.3, 1.5, 0.9, 0.4, 0.1, 0.2, 0.4, 0.7, 1.0)
EXP1_FIGURE1_DV = (-0.00, -0.00, -0.02, -0.08, -0.02, 0.00, 0.00, 0.00, 0.00)
EXP1_FIGURE1_D2V = (0.00, 0.00, 0.03, 0.06, 0.12, 0.03, 0.02, 0.00, 0.00)

# Number of evaluation points used to compare the cubic-spline and B-spline
# representations over each interval, including the extrapolated tails.
# Not specified in the paper; chosen for numerical verification.
EXP1_EVAL_POINTS_PER_SEGMENT = 25
EXP1_TAIL_EXTENSION = 4.0  # extra |z| range tested in the extrapolated tails

# ---------------------------------------------------------------------------
# exp_2: CVI Objective Function (penalties)
# ---------------------------------------------------------------------------

# Strike regularization factor lambda. Paper, "Strike Regularization
# Penalty": "lambda = 0.05 is a sensible choice across underlyings."
STRIKE_REG_LAMBDA = 0.05

# Number of knots per expiry used for the synthetic objective-function
# demonstration. Not specified in the paper for this illustrative check;
# chosen in the "mid-to-high teens" range the paper reports for convergence.
EXP2_N_KNOTS = 15

# Number of expiries in the synthetic surface used to exercise the
# multi-expiry objective assembly. Not specified in the paper for this
# experiment; a small value keeps the demonstration fast.
EXP2_N_EXPIRIES = 4

# ---------------------------------------------------------------------------
# exp_3: No-arbitrage constraints
# ---------------------------------------------------------------------------

# Number of discretized strikes used for the discretized constraints
# (positivity of variance, generic no-calendar-spread, PDF >= 0 between
# edge knots). Paper, "Numerical Results": "Discretized constraints are
# applied across 20 strikes."
N_DISCRETIZATION_STRIKES = 20

# Small margin used to enforce Lee's strict tail-slope bounds numerically.
# Not specified in the paper ("we enforce them numerically with a small
# margin"); a conservative small value is used.
LEE_BOUND_MARGIN = 1e-4

# Number of linearization (two-step calibration) iterations. Paper,
# "Putting It All Together": "One or a few iterations ... are sufficient in
# practice." We use two iterations (first without, second with the
# linearized butterfly constraints), matching the paper's reported timings.
N_CALIBRATION_ITERATIONS = 2

# Knot counts and expiries for the synthetic no-arbitrage demonstration.
# Not specified in the paper for this experiment; chosen to be large enough
# that d1*d2 > 1 holds at the edge knots (see "PDF >= 0 at the edge knots").
EXP3_N_KNOTS = 15
EXP3_N_EXPIRIES = 5
EXP3_EDGE_Z = 6.0  # |z_0| = |z_{n-1}|, in normalized log-moneyness

# ---------------------------------------------------------------------------
# exp_4: Full CVI calibration (QP assembly, benchmarking, knot convergence)
# ---------------------------------------------------------------------------

# Knot counts per expiry at which the QP is solved and benchmarked. Paper,
# Table 3: "ranging from 10 to 40." The live surfaces fit in this
# replication are small enough (<= 20 expiries) that the full paper grid
# runs in well under a minute, so no reduction is needed; KNOT_COUNTS_RUN
# is identical to KNOT_COUNTS_PAPER and ``--full`` is a no-op kept only for
# interface compatibility with the other experiments.
KNOT_COUNTS_PAPER = (10, 20, 30, 40)
KNOT_COUNTS_RUN = KNOT_COUNTS_PAPER

# Underlyings targeted by the paper's benchmark (Table 2): SPX and SX5E.
UNDERLYINGS_PAPER = ("SPX", "SX5E")
# Underlyings actually fit in this replication: SX5E (Eurex) is not served
# by the Massive REST API, and SPX contracts carry only zero-zero NBBO
# quotes on this entitlement (both confirmed empirically, see
# results/RESULTS.md), so NDX and NVDA are used instead -- underlyings the
# paper itself fits in its knot-convergence figure (Section "Scaling").
UNDERLYINGS_RUN = ("NDX", "NVDA")

# Paper-reported market-data sizes (Table 2), used only to compare against
# the data actually obtained in this replication.
SPX_N_QUOTES_PAPER = 14538
SPX_N_EXPIRIES_PAPER = 46
SX5E_N_QUOTES_PAPER = 3153
SX5E_N_EXPIRIES_PAPER = 19

# Number of nearest-dated listed expiries fetched per underlying for the
# live surface (paper: 46 for SPX, 19 for SX5E). Not specified in the
# paper for a live snapshot; chosen close to the SX5E count while keeping
# the NBBO download time bounded.
EXP4_N_EXPIRIES_TARGET = 20

# Number of strikes requested per expiry (bounds the number of NBBO
# fetches: 2 requests per strike). Not specified in the paper for a live
# snapshot; chosen to keep each expiry's mid-quote count in the same
# order of magnitude as the paper's average per-expiry quote count
# (14538/46 ~ 316 for SPX, 3153/19 ~ 166 for SX5E) while keeping the
# total download time bounded.
EXP4_N_STRIKES_TARGET = 40

# QP solver tolerance (Clarabel defaults). Paper, "Numerical Results":
# "the reported timings assume Clarabel's default tolerance."
CLARABEL_DEFAULT_TOLERANCE = True
