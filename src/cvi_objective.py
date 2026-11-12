"""CVI objective function: the four bid-ask aware penalties (paper,
Section "Objective Function", pp. 2-3).

Builds the quadratic objective 1/2 x^T P x + q^T x (scipy.sparse matrices)
for the free per-expiry parameters x_j = (v(0), dv/dz(0), c_1, ..., c_{n-2})
(the n-2 interior anchored convexities; the edge convexities c_0 = c_{n-1}
= 0 are fixed, as required for linear extrapolation, Section "CVI Cubic
Spline"), plus the auxiliary epigraph variables needed to express the
Above Ask, Below Bid and Strike Regularization penalties as a quadratic
program (paper's implementation note: "The max() terms ... require
auxiliary variables in canonical QP form").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from src.cvi_spline import evaluate_cubic_spline
from src.qp_types import ConstraintSet


def free_param_design_matrices(
    z_eval: np.ndarray, z_knots: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design matrices V, D, C (shape (len(z_eval), n)) such that, for the
    free parameter vector x = (v(0), dv/dz(0), c_1, ..., c_{n-2}) (edge
    convexities c_0 = c_{n-1} = 0 fixed):

        v(z_eval)    = V @ x
        dv/dz(z_eval)  = D @ x
        d2v/dz2(z_eval) = C @ x

    Each column is obtained by evaluating the (linear) cubic-spline
    parameterization at a unit basis vector of x (src.cvi_spline).
    """
    z_knots = np.asarray(z_knots, dtype=float)
    n = z_knots.size
    l = int(np.argmin(np.abs(z_knots)))
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))

    V = np.zeros((z_eval.size, n))
    D = np.zeros((z_eval.size, n))
    C = np.zeros((z_eval.size, n))
    for col in range(n):
        v0 = 1.0 if col == 0 else 0.0
        d0 = 1.0 if col == 1 else 0.0
        c = np.zeros(n)
        if col >= 2:
            c[col - 1] = 1.0  # col-1 in 1..n-2 -> interior knot index
        v, dv, d2v = evaluate_cubic_spline(z_eval, z_knots, v0, d0, c, l)
        V[:, col], D[:, col], C[:, col] = v, dv, d2v
    return V, D, C


@dataclass
class ExpiryQuotes:
    """Per-expiry option data in normalized log-moneyness z-space.

    All variances are *total* variances sigma^2 * T (paper's notation,
    Section "Notation"). ``z_mid``/``v_bid_mid``/``v_ask_mid`` index the
    mid set S_{T,mid} = S_{T,ask} ∩ S_{T,bid}; ``z_ask_only``/``z_bid_only``
    index options quoted only on the ask/bid side (paper's reported
    calibration restricts the Above Ask / Below Bid penalties to options
    with a bid or an ask but not both, Section "Above Ask Penalty": "a
    good compromise ... the calibration times reported in this paper are
    based on this choice").
    """

    T: float
    v_star: float
    z_mid: np.ndarray
    v_bid_mid: np.ndarray
    v_ask_mid: np.ndarray
    z_ask_only: np.ndarray
    v_ask_only: np.ndarray
    vega_ask_only: np.ndarray
    z_bid_only: np.ndarray
    v_bid_only: np.ndarray
    vega_bid_only: np.ndarray


@dataclass
class VariableLayout:
    """Index ranges of the stacked optimization vector x."""

    n_expiries: int
    n_free: int  # free spline parameters per expiry
    spline_slices: list[slice]
    ask_slack_slices: list[slice]
    bid_slack_slices: list[slice]
    tv_slack_slices: list[slice]
    total_size: int


def _exclude_outside_knots(z: np.ndarray, z_knots: np.ndarray, *others: np.ndarray):
    """Paper, Section "Above Ask Penalty": options whose normalized
    log-moneyness falls outside [z_0, z_{n-1}] are excluded from the Least
    Squares and Below Bid penalties."""
    mask = (z >= z_knots[0]) & (z <= z_knots[-1])
    return (z[mask],) + tuple(o[mask] for o in others)


def build_variable_layout(expiries: list[ExpiryQuotes], z_knots: np.ndarray) -> VariableLayout:
    n = z_knots.size
    m = len(expiries)
    offset = m * n
    spline_slices = [slice(j * n, (j + 1) * n) for j in range(m)]

    ask_slack_slices, bid_slack_slices, tv_slack_slices = [], [], []
    for exp in expiries:
        n_ask = exp.z_ask_only.size
        ask_slack_slices.append(slice(offset, offset + n_ask))
        offset += n_ask
    for exp in expiries:
        n_bid = exp.z_bid_only.size
        bid_slack_slices.append(slice(offset, offset + n_bid))
        offset += n_bid
    for _ in expiries:
        n_tv = n - 1
        tv_slack_slices.append(slice(offset, offset + n_tv))
        offset += n_tv
    return VariableLayout(m, n, spline_slices, ask_slack_slices, bid_slack_slices, tv_slack_slices, offset)


def assemble_objective(
    expiries: list[ExpiryQuotes], z_knots: np.ndarray, lam: float
) -> tuple[sp.csc_matrix, np.ndarray, VariableLayout]:
    """Assemble P (sparse, PSD), q and the variable layout for the four
    CVI penalties, summed over all expiries (paper, Section "Objective
    Function": "the objective function is the sum of the four penalties
    ... Each penalty is the sum of dimensionless numbers across expiries").
    """
    z_knots = np.asarray(z_knots, dtype=float)
    n = z_knots.size
    layout = build_variable_layout(expiries, z_knots)
    n_var = layout.total_size

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    q = np.zeros(n_var)

    def add_block(block: np.ndarray, idx: np.ndarray) -> None:
        """Add a dense symmetric block (len(idx) x len(idx)) into P at the
        global indices idx (scipy.sparse COO accumulation)."""
        r, c = np.meshgrid(idx, idx, indexing="ij")
        rows.append(r.ravel())
        cols.append(c.ravel())
        data.append(block.ravel())

    for j, exp in enumerate(expiries):
        idx_spline = np.arange(layout.spline_slices[j].start, layout.spline_slices[j].stop)

        # --- Least Squares Penalty ---
        z_mid, v_bid_mid, v_ask_mid = _exclude_outside_knots(
            exp.z_mid, z_knots, exp.v_bid_mid, exp.v_ask_mid
        )
        v_mid_mid = 0.5 * (v_bid_mid + v_ask_mid)
        n_mid = z_mid.size
        if n_mid > 0:
            w = 1.0 / np.maximum(v_ask_mid - v_bid_mid, 1e-12) ** 2
            V, _, _ = free_param_design_matrices(z_mid, z_knots)
            WV = V * w[:, None]
            block = 2.0 / n_mid * (V.T @ WV)
            add_block(block, idx_spline)
            q[idx_spline] += -2.0 / n_mid * (V.T @ (w * v_mid_mid))

        # --- Above Ask Penalty (epigraph slack s_ask >= v - v_ask, s_ask >= 0) ---
        n_ask_set = z_mid.size  # S_{T,ask} includes mid set in the general definition
        q_j = np.sum(1.0 / np.maximum(v_ask_mid - v_bid_mid, 1e-12) ** 2) if n_mid > 0 else 0.0
        idx_ask = np.arange(layout.ask_slack_slices[j].start, layout.ask_slack_slices[j].stop)
        n_ask_only = exp.z_ask_only.size
        if n_ask_only > 0 and q_j > 0:
            total_vega = np.sum(exp.vega_ask_only)
            w_ask = q_j * exp.vega_ask_only / max(total_vega, 1e-12)
            n_ask_denom = n_ask_only  # N_{T,ask}: options entering this penalty (ask-only compromise)
            diag = 2.0 * w_ask / n_ask_denom
            add_block(np.diag(diag), idx_ask)

        # --- Below Bid Penalty (epigraph slack s_bid >= v_bid - v, s_bid >= 0) ---
        idx_bid = np.arange(layout.bid_slack_slices[j].start, layout.bid_slack_slices[j].stop)
        n_bid_only = exp.z_bid_only.size
        if n_bid_only > 0 and q_j > 0:
            total_vega_bid = np.sum(exp.vega_bid_only)
            w_bid = q_j * exp.vega_bid_only / max(total_vega_bid, 1e-12)
            n_bid_denom = n_bid_only
            diag = 2.0 * w_bid / n_bid_denom
            add_block(np.diag(diag), idx_bid)

        # --- Strike Regularization Penalty: lambda * sum |c_{i+1}-c_i|, linear in slack t ---
        idx_tv = np.arange(layout.tv_slack_slices[j].start, layout.tv_slack_slices[j].stop)
        q[idx_tv] += lam

    P = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_var, n_var),
    ).tocsc()
    return P, q, layout


def epigraph_constraints(
    expiries: list[ExpiryQuotes], layout: VariableLayout, z_knots: np.ndarray
) -> ConstraintSet:
    """Linear constraints coupling the Above Ask / Below Bid epigraph
    slacks to the spline parameters (paper's implementation note: "The
    max() terms ... require auxiliary variables in canonical QP form"):

        s_ask >= v(K, T) - v_ask(K, T),   s_ask >= 0
        s_bid >= v_bid(K, T) - v(K, T),   s_bid >= 0

    plus the strike-regularization epigraph |c_{i+1} - c_i| <= t
    (Section "Strike Regularization Penalty").
    """
    n_var = layout.total_size
    rows, l, u, names = [], [], [], []

    def add(idx: np.ndarray, coeffs: np.ndarray, lo: float, hi: float, name: str) -> None:
        rows.append(sp.csc_matrix((coeffs, (np.zeros_like(idx), idx)), shape=(1, n_var)))
        l.append(lo); u.append(hi); names.append(name)

    for j, exp in enumerate(expiries):
        idx_spline = np.arange(layout.spline_slices[j].start, layout.spline_slices[j].stop)
        idx_ask = np.arange(layout.ask_slack_slices[j].start, layout.ask_slack_slices[j].stop)
        idx_bid = np.arange(layout.bid_slack_slices[j].start, layout.bid_slack_slices[j].stop)
        idx_tv = np.arange(layout.tv_slack_slices[j].start, layout.tv_slack_slices[j].stop)

        if exp.z_ask_only.size > 0:
            V_ask, _, _ = free_param_design_matrices(exp.z_ask_only, z_knots)
            for i in range(exp.z_ask_only.size):
                add(
                    np.concatenate([idx_spline, idx_ask[i : i + 1]]),
                    np.concatenate([-V_ask[i], [1.0]]),
                    -exp.v_ask_only[i], np.inf,
                    f"above_ask_T{j}_i{i}",
                )
                add(idx_ask[i : i + 1], np.array([1.0]), 0.0, np.inf, f"above_ask_nonneg_T{j}_i{i}")

        if exp.z_bid_only.size > 0:
            V_bid, _, _ = free_param_design_matrices(exp.z_bid_only, z_knots)
            for i in range(exp.z_bid_only.size):
                add(
                    np.concatenate([idx_spline, idx_bid[i : i + 1]]),
                    np.concatenate([V_bid[i], [1.0]]),
                    exp.v_bid_only[i], np.inf,
                    f"below_bid_T{j}_i{i}",
                )
                add(idx_bid[i : i + 1], np.array([1.0]), 0.0, np.inf, f"below_bid_nonneg_T{j}_i{i}")

        n = layout.n_free
        for k in range(n - 1):
            # c_k with c_0 = c_{n-1} = 0 fixed: free-parameter index of c_k is
            # 2 + (k - 1) for interior knots 1..n-2; edges contribute 0.
            coef = np.zeros(n)
            if 1 <= k <= n - 2:
                coef[2 + (k - 1)] = -1.0
            if 1 <= k + 1 <= n - 2:
                coef[2 + (k + 1 - 1)] = 1.0
            full_idx = np.concatenate([idx_spline, idx_tv[k : k + 1]])
            full_coef_hi = np.concatenate([-coef, [1.0]])  # t - (c_{k+1}-c_k) >= 0
            full_coef_lo = np.concatenate([coef, [1.0]])  # t + (c_{k+1}-c_k) >= 0
            add(full_idx, full_coef_hi, 0.0, np.inf, f"strike_reg_hi_T{j}_k{k}")
            add(full_idx, full_coef_lo, 0.0, np.inf, f"strike_reg_lo_T{j}_k{k}")

    A = sp.vstack(rows, format="csc") if rows else sp.csc_matrix((0, n_var))
    return ConstraintSet(A, np.array(l), np.array(u), names)
