"""Projection utilities and projector classes for portfolio weights."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import softmax

import skfolio.typing as skt
from skfolio.utils.equations import equations_to_matrix
from skfolio.utils.tools import input_to_array


@dataclass
class ProjectionConfig:
    """Container for projection-related constraints.

    Parameters
    ----------
    lower, upper : array-like or float or None
        Box bounds. Defaults to [0, 1] if None.
    budget : float or None
        Equality budget. If None, no equality budget is enforced.
    min_budget, max_budget : float or None
        Budget bounds (mutually exclusive with equality `budget`).
    max_short, max_long : float or None
        Aggregate short/long caps.
    groups, linear_constraints, left_inequality, right_inequality : array-like or None
        Rich linear/group constraints.
    X_tracking, tracking_error_benchmark, max_tracking_error : array-like or None
        Time-series tracking error constraint components.
    covariance, variance_bound : array-like or None
        Variance-style quadratic bound.
    previous_weights, max_turnover : array-like or float or None
        Turnover information for L1 cap.
    solver : str or None
        Optional cvxpy solver name.
    """

    lower: Any | None = 0.0
    upper: Any | None = 1.0
    budget: float | None = 1.0
    min_budget: float | None = None
    max_budget: float | None = None
    max_short: float | None = None
    max_long: float | None = None
    groups: Any | None = None
    linear_constraints: skt.LinearConstraints | None = None
    left_inequality: Any | None = None
    right_inequality: Any | None = None
    X_tracking: Any | None = None
    tracking_error_benchmark: Any | None = None
    max_tracking_error: float | None = None
    covariance: Any | None = None
    variance_bound: float | None = None
    previous_weights: Any | None = None
    max_turnover: float | None = None
    solver: str | None = None


class BaseProjector:
    def project(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FastProjector(BaseProjector):
    """Fast projector using box+sum and optional L1 turnover cap."""

    def __init__(self, config: ProjectionConfig):
        self.config = config

    def project(self, y: np.ndarray) -> np.ndarray:
        if self.config.previous_weights is None or self.config.max_turnover is None:
            return project_box_and_sum(
                y=y,
                lower=0.0 if self.config.lower is None else self.config.lower,
                upper=1.0 if self.config.upper is None else self.config.upper,
                budget=1.0 if self.config.budget is None else float(self.config.budget),
            )
        return project_with_turnover(
            w_raw=y,
            previous_weights=self.config.previous_weights,
            max_turnover=self.config.max_turnover,
            lower=0.0 if self.config.lower is None else self.config.lower,
            upper=1.0 if self.config.upper is None else self.config.upper,
            budget=1.0 if self.config.budget is None else float(self.config.budget),
        )


class ConvexProjector(BaseProjector):
    """Convex projector using cvxpy with rich constraints (see ProjectionConfig)."""

    def __init__(self, config: ProjectionConfig):
        self.config = config

    def project(self, y: np.ndarray) -> np.ndarray:
        return project_convex(
            w_raw=y,
            budget=self.config.budget,
            lower=self.config.lower,
            upper=self.config.upper,
            min_budget=self.config.min_budget,
            max_budget=self.config.max_budget,
            max_short=self.config.max_short,
            max_long=self.config.max_long,
            groups=self.config.groups,
            linear_constraints=self.config.linear_constraints,
            left_inequality=self.config.left_inequality,
            right_inequality=self.config.right_inequality,
            X=self.config.X_tracking,
            tracking_error_benchmark=self.config.tracking_error_benchmark,
            max_tracking_error=self.config.max_tracking_error,
            covariance=self.config.covariance,
            variance_bound=self.config.variance_bound,
            previous_weights=self.config.previous_weights,
            max_turnover=self.config.max_turnover,
            solver=self.config.solver,
        )


class AutoProjector(BaseProjector):
    """
    Automatically choose fast or convex projector based on provided constraints.
    Fast projector is used when no rich constraints are provided.
    Convex projector is used when rich constraints are provided.
    """

    def __init__(self, config: ProjectionConfig):
        self.config = config
        self._fast = FastProjector(config)
        self._cvx = ConvexProjector(config)

    def _needs_cvx(self) -> bool:
        return any(
            v is not None
            for v in [
                self.config.groups,
                self.config.linear_constraints,
                self.config.left_inequality,
                self.config.right_inequality,
                self.config.X_tracking,
                self.config.tracking_error_benchmark,
                self.config.max_tracking_error,
                self.config.covariance,
                self.config.variance_bound,
                self.config.min_budget,
                self.config.max_budget,
                self.config.max_short,
                self.config.max_long,
            ]
        )

    def project(self, y: np.ndarray) -> np.ndarray:
        if self._needs_cvx():
            return self._cvx.project(y)
        return self._fast.project(y)


def project_box_and_sum(
    y: ArrayLike,
    lower: ArrayLike | float,
    upper: ArrayLike | float,
    budget: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Project onto box with budget equality using bisection.

    Projects a vector ``y`` onto the set
    ``{ w ∈ R^n | sum(w) = budget, lower ≤ w ≤ upper }``
    via a 1D bisection on a dual parameter ``theta`` such that
    ``w = clip(y - theta, lower, upper)`` satisfies the budget.

    Parameters
    ----------
    y : ArrayLike of shape (n,)
        Raw weights to be projected.
    lower : ArrayLike or float
        Lower bounds per coordinate (broadcasted if float).
    upper : ArrayLike or float
        Upper bounds per coordinate (broadcasted if float).
    budget : float, default=1.0
        Target sum of weights.
    tol : float, default=1e-12
        Tolerance on the budget equality.
    max_iter : int, default=100
        Maximum bisection iterations.

    Returns
    -------
    w : ndarray of shape (n,)
        Projected weights.
    """
    y_arr = np.asarray(y, dtype=float)
    n = int(y_arr.shape[0])
    if lower is None:
        lo_arr = np.full(n, 0.0, dtype=float)
    elif np.isscalar(lower):
        lo_arr = np.full(n, float(lower), dtype=float)
    else:
        lo_arr = input_to_array(lower, n, 0.0, 1, None, "lower_bounds")

    if upper is None:
        up_arr = np.full(n, 1.0, dtype=float)
    elif np.isscalar(upper):
        up_arr = np.full(n, float(upper), dtype=float)
    else:
        up_arr = input_to_array(upper, n, 1.0, 1, None, "upper_bounds")

    low_sum = float(np.sum(lo_arr))
    up_sum = float(np.sum(up_arr))
    b = float(budget)
    if b < low_sum - 1e-15 or b > up_sum + 1e-15:
        b = min(max(b, low_sum), up_sum)

    lo = float(np.min(y_arr - up_arr)) - 1.0
    hi = float(np.max(y_arr - lo_arr)) + 1.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        w = np.clip(y_arr - mid, lo_arr, up_arr)
        s = float(np.sum(w))
        if abs(s - b) <= tol:
            return w
        if s > b:
            lo = mid
        else:
            hi = mid
    return np.clip(y_arr - 0.5 * (lo + hi), lo_arr, up_arr)


def project_with_turnover(
    w_raw: ArrayLike,
    previous_weights: ArrayLike | None,
    max_turnover: float | None,
    lower: ArrayLike | float | None = None,
    upper: ArrayLike | float | None = None,
    budget: float = 1.0,
) -> np.ndarray:
    """
    Project with box/budget and optional L1 turnover cap.

    Parameters
    ----------
    w_raw : ArrayLike of shape (n,)
        Proposed weights before projection.
    previous_weights : ArrayLike of shape (n,), optional
        Previous portfolio weights used to enforce turnover cap.
    max_turnover : float, optional
        L1 distance cap between projected weights and previous weights.
    lower, upper : ArrayLike or float, optional
        Box bounds; broadcasted if float. Defaults to [0, 1].
    budget : float, default=1.0
        Target sum of weights.

    Returns
    -------
    w : ndarray of shape (n,)
        Projected weights.
    """
    y = np.asarray(w_raw, dtype=float)
    n = int(y.shape[0])
    if lower is None:
        lower_bounds = np.full(n, 0.0, dtype=float)
    elif np.isscalar(lower):
        lower_bounds = np.full(n, float(lower), dtype=float)
    else:
        lower_bounds = input_to_array(lower, n, 0.0, 1, None, "lower_bounds")

    if upper is None:
        u = np.full(n, 1.0, dtype=float)
    elif np.isscalar(upper):
        u = np.full(n, float(upper), dtype=float)
    else:
        u = input_to_array(upper, n, 1.0, 1, None, "upper_bounds")

    # Iteratively enforce both box/budget and L1-turnover constraints
    w = project_box_and_sum(y=y, lower=lower_bounds, upper=u, budget=budget)

    if previous_weights is None or max_turnover is None:
        return w

    if max_turnover <= 0:
        return project_box_and_sum(
            y=np.asarray(previous_weights, dtype=float),
            lower=lower_bounds,
            upper=u,
            budget=budget,
        )

    prev = np.asarray(previous_weights, dtype=float)
    for _ in range(10):  # small fixed-point iterations
        delta = w - prev
        l1 = float(np.sum(np.abs(delta)))
        if l1 <= float(max_turnover) + 1e-12:
            break
        factor = float(max_turnover) / l1
        w = prev + factor * delta
        w = project_box_and_sum(y=w, lower=lower_bounds, upper=u, budget=budget)
    return w


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
    groups: skt.Groups | None = None,
    linear_constraints: skt.LinearConstraints | None = None,
    left_inequality: skt.Inequality | None = None,
    right_inequality: skt.Inequality | None = None,
    # tracking error (benchmark time-series)
    X: ArrayLike | None = None,
    tracking_error_benchmark: ArrayLike | None = None,
    max_tracking_error: float | None = None,
    # simple variance-style bound
    covariance: ArrayLike | None = None,
    variance_bound: float | None = None,
    # turnover
    previous_weights: ArrayLike | None = None,
    max_turnover: float | None = None,
    # solver
    solver: str | None = None,
) -> np.ndarray:
    r"""
    Convex projection with advanced constraints using cvxpy.

    Solves: minimize 0.5 * ||w - w_raw||_2^2 subject to
    sum(w) = budget
    lower <= w <= upper
    group_min <= G @ w <= group_max  (if provided)
    w' Sigma w <= max_tracking_error^2 (if provided)
    """
    w0 = np.asarray(w_raw, dtype=float)
    n = int(w0.shape[0])
    if lower is None:
        lower_bounds = np.full(n, 0.0, dtype=float)
    elif np.isscalar(lower):
        lower_bounds = np.full(n, float(lower), dtype=float)
    else:
        lower_bounds = input_to_array(lower, n, 0.0, 1, None, "lower_bounds")

    if upper is None:
        u = np.full(n, 1.0, dtype=float)
    elif np.isscalar(upper):
        u = np.full(n, float(upper), dtype=float)
    else:
        u = input_to_array(upper, n, 1.0, 1, None, "upper_bounds")

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
        # Tracking error as sample standard deviation of active returns:
        # std(R w - y) = ||(R w - y) - mean(R w - y)||_2 / sqrt(T-1)
        Rt = np.asarray(X, dtype=float)
        y = np.asarray(tracking_error_benchmark, dtype=float)
        if Rt.ndim != 2 or y.ndim != 1 or Rt.shape[0] != y.shape[0] or Rt.shape[1] != n:
            raise ValueError("Invalid shapes for TE: X (T,n), benchmark (T,)")
        T = Rt.shape[0]
        ones = np.ones((T, 1))
        resid = Rt @ w - y
        mean_resid = (1.0 / T) * (ones.T @ resid)
        centered = resid - ones @ mean_resid
        constraints.append(
            cp.norm(centered, 2) / np.sqrt(max(T - 1, 1)) <= float(max_tracking_error)
        )
    elif covariance is not None and variance_bound is not None:
        # Variance-style bound alternative
        Sigma = np.asarray(covariance, dtype=float)
        constraints.append(cp.quad_form(w, Sigma) <= float(variance_bound))

    # Turnover L1 cap if requested
    if previous_weights is not None and max_turnover is not None:
        w_prev = np.asarray(previous_weights, dtype=float)
        if w_prev.shape != (n,):
            raise ValueError("previous_weights must have shape (n,)")
        if max_turnover < 0:
            raise ValueError("max_turnover must be nonnegative")
        constraints.append(cp.norm1(w - w_prev) <= float(max_turnover))

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
