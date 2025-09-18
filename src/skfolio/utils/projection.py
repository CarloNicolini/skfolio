"""Projection utilities for portfolio weights."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from numpy.typing import ArrayLike
import cvxpy as cp
import numpy as np

from skfolio.utils.equations import equations_to_matrix


def _as_array_like(x, n: int, default: float) -> np.ndarray:
    if x is None:
        return np.full(n, default, dtype=float)
    if np.isscalar(x):
        return np.full(n, float(x), dtype=float)  # type: ignore[arg-type]
    arr = np.asarray(x, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"Expected shape ({n},) got {arr.shape}.")
    return arr


def project_box_and_sum(
    y: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    budget: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Project y onto {w | sum(w)=budget, <=w<=upper} via bisection.

    This solves for theta such that w = clip(y - theta, lower, upper) sums to budget.
    """
    low_sum = float(np.sum(lower))
    up_sum = float(np.sum(upper))
    b = float(budget)
    if b < low_sum - 1e-15 or b > up_sum + 1e-15:
        # infeasible request; fall back to nearest feasible budget
        b = min(max(b, low_sum), up_sum)

    # Brackets for theta: when theta -> -inf, w -> upper; when +inf, w -> lower
    # Use values based on y- upper/lower bounds
    lo = float(np.min(y - upper)) - 1.0
    hi = float(np.max(y - lower)) + 1.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        w = np.clip(y - mid, lower, upper)
        s = float(np.sum(w))
        if abs(s - b) <= tol:
            return w
        if s > b:
            # Need larger theta to reduce sum
            lo = mid
        else:
            hi = mid
    # Final projection
    return np.clip(y - 0.5 * (lo + hi), lower, upper)


def project_with_turnover(
    w_raw: ArrayLike,
    previous_weights: ArrayLike | None,
    max_turnover: float | None,
    lower: ArrayLike | float | None = None,
    upper: ArrayLike | float | None = None,
    budget: float = 1.0,
) -> np.ndarray:
    """Project weights with box-and-sum, then apply optional L1 turnover cap.

    Steps:
      1) Box-and-sum projection to enforce bounds and budget
      2) If turnover cap provided and previous weights available, shrink the
         change toward previous weights to satisfy L1 norm, then re-project.
    """
    n = int(np.asarray(w_raw).shape[0])  # type: ignore
    lower_bounds = _as_array_like(lower, n, 0.0)
    u = _as_array_like(upper, n, 1.0)
    y = np.asarray(w_raw, dtype=float)

    # First project to box-and-sum
    w = project_box_and_sum(y=y, lower=lower_bounds, upper=u, budget=budget)

    if previous_weights is None or max_turnover is None:
        return w

    tau = float(max_turnover)
    if tau <= 0:
        # No trade allowed; stick to previous feasible weights projected to box/sum
        w_prev = project_box_and_sum(
            y=np.asarray(previous_weights, dtype=float),
            lower=lower_bounds,
            upper=u,
            budget=budget,
        )
        return w_prev

    prev = np.asarray(previous_weights, dtype=float)
    delta = w - prev
    l1 = float(np.sum(np.abs(delta)))
    if l1 <= tau + 1e-15:
        return w
    # Shrink toward previous weights uniformly
    factor = tau / l1
    w_shrunk = prev + factor * delta
    # Re-enforce constraints post-shrink
    return project_box_and_sum(y=w_shrunk, lower=lower_bounds, upper=u, budget=budget)


def project_convex(
    w_raw: ArrayLike,
    budget: float | None = 1.0,
    lower: ArrayLike | float | None = 0.0,
    upper: ArrayLike | float | None = 1.0,
    min_budget: float | None = None,
    max_budget: float | None = None,
    max_short: float | None = None,
    max_long: float | None = None,
    # group/linear constraints
    groups: ArrayLike | None = None,
    linear_constraints: list[str] | None = None,
    left_inequality: ArrayLike | None = None,
    right_inequality: ArrayLike | None = None,
    # tracking error (benchmark time-series)
    X: ArrayLike | None = None,
    tracking_error_benchmark: ArrayLike | None = None,
    max_tracking_error: float | None = None,
    # simple variance-style bound
    covariance: ArrayLike | None = None,
    variance_bound: float | None = None,
    # solver
    solver: str | None = None,
) -> np.ndarray:
    """Convex projection with advanced constraints using cvxpy.

    Solves: minimize 0.5 * ||w - w_raw||_2^2 subject to
        sum(w) = budget
        lower <= w <= upper
        group_min <= G @ w <= group_max  (if provided)
        w' Sigma w <= max_tracking_error^2 (if provided)
    """

    n = w_raw.shape[0]
    lower_bounds = _as_array_like(lower, n, 0.0)
    u = _as_array_like(upper, n, 1.0)
    w0 = np.asarray(w_raw, dtype=float)

    w = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.sum_squares(w - w0))
    constraints: list[Any] = [w >= lower_bounds, w <= u]

    if min_budget is not None:
        constraints.append(cp.sum(w) >= float(min_budget))
    if max_budget is not None:
        constraints.append(cp.sum(w) <= float(max_budget))
    if budget is not None:
        if min_budget is not None or max_budget is not None:
            raise ValueError("Cannot use budget with min_budget/max_budget.")
        constraints.append(cp.sum(w) == float(budget))

    if max_long is not None:
        if max_long <= 0:
            raise ValueError("max_long must be > 0")
        constraints.append(cp.sum(cp.pos(w)) <= float(max_long))
    if max_short is not None:
        if max_short <= 0:
            raise ValueError("max_short must be > 0")
        constraints.append(cp.sum(cp.neg(w)) >= -float(max_short))

    if linear_constraints is not None and len(linear_constraints) > 0:
        if groups is None:
            raise ValueError(
                "groups must be provided (as matrix) to parse linear_constraints."
            )
        a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
            groups=groups,
            equations=linear_constraints,
            raise_if_group_missing=False,
        )
        if len(a_eq) != 0:
            constraints.append(a_eq @ w - b_eq == 0)
        if len(a_ineq) != 0:
            constraints.append(a_ineq @ w - b_ineq <= 0)

    if left_inequality is not None and right_inequality is not None:
        A = np.asarray(left_inequality)
        b = np.asarray(right_inequality)
        if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0] or A.shape[1] != n:
            raise ValueError("Invalid left/right inequality shapes.")
        constraints.append(A @ w - b <= 0)

    if (
        X is not None
        and tracking_error_benchmark is not None
        and max_tracking_error is not None
    ):
        # TE constraint using time-series definition
        Rt = np.asarray(X, dtype=float)
        y = np.asarray(tracking_error_benchmark, dtype=float)
        if Rt.ndim != 2 or y.ndim != 1 or Rt.shape[0] != y.shape[0] or Rt.shape[1] != n:
            raise ValueError("Invalid shapes for TE: X (T,n), benchmark (T,)")
        T = Rt.shape[0]
        constraints.append(
            cp.norm(Rt @ w - y, 2) / np.sqrt(max(T - 1, 1)) <= float(max_tracking_error)
        )
    elif covariance is not None and variance_bound is not None:
        # Variance-style bound alternative
        Sigma = np.asarray(covariance, dtype=float)
        constraints.append(cp.quad_form(w, Sigma) <= float(variance_bound))

    prob = cp.Problem(objective, constraints)
    if solver is None:
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(verbose=False)
    else:
        prob.solve(solver=solver, verbose=False)

    if w.value is None:
        raise RuntimeError("Convex projection failed to converge.")
    return np.asarray(w.value, dtype=float)
