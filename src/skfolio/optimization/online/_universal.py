"""Universal Portfolio estimator.

Thomas Cover's Universal Portfolio algorithm estimates the next-period
portfolio as the expected portfolio over all constant-rebalanced portfolios,
weighted by their past wealth. We approximate the integral over the simplex
either via a uniform grid (when ``grid_step`` is provided) or via Monte Carlo
sampling from a Dirichlet distribution (uniform over the simplex).

References
----------
- Cover, T. M. (1991, 1996). Universal Portfolios.
- Blog overview: Andrew C. Jones, "Universal Portfolios".
"""

# Author: skfolio contributors
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv  # type: ignore

from skfolio.optimization._base import BaseOptimization
from skfolio.utils.projection import project_with_turnover, project_convex


def _integer_simplex_grid(n_assets: int, m: int) -> np.ndarray:
    """
    Generate all compositions of integer m into n_assets parts.

    Returns an array of shape (n_points, n_assets) with nonnegative integers
    summing to m. Uses stars-and-bars with combinations.
    """
    # Number of points = C(m + n - 1, n - 1)
    # We construct by placing n-1 bars among m + n - 1 positions.
    idx_iter = combinations(range(m + n_assets - 1), n_assets - 1)
    points = []
    for bars in idx_iter:
        # prepend -1 and append last index to compute gaps (stars per bin)
        last = -1
        counts = []
        for b in bars:
            counts.append(b - last - 1)
            last = b
        counts.append(m + n_assets - 1 - last - 1)
        points.append(counts)
    return np.asarray(points, dtype=float)


class Universal(BaseOptimization):
    """
    Universal Portfolio estimator (Cover, 1991/1996).

    On input returns ``X`` with shape (n_observations, n_assets), computes the next-period portfolio weights ``p_{T+1}`` as the expected value of constant rebalanced portfolios under a wealth-weighted measure derived from the historical returns.

    Parameters
    ----------
    grid_step : float, optional
        If provided, discretize the simplex on a uniform grid with step size
        ``grid_step`` (e.g., 0.1). The number of grid points grows combinatorially as ``comb(1/grid_step + n_assets - 1, n_assets - 1)``.
        For larger universes prefer the Monte Carlo approximation via ``n_samples``.

    n_samples : int, default=10000
        Number of Dirichlet samples to approximate the universal integral when ``grid_step`` is not provided.

    random_state : int | np.random.Generator, optional
        Random state for reproducibility of the Dirichlet sampling.

    weight_scheme : {"wealth", "uniform", "top_k"}, default="wealth"
        How to weight experts (candidate portfolios):
        - "wealth": proportional to cumulative wealth (Cover's UP).
        - "uniform": equal weight across all experts.
        - "top_k": equal weight across the top-k experts by cumulative wealth.

    top_k : int, default=1
        Number of experts when ``weight_scheme="top_k"``. If larger than the
        number of candidates, it is clipped.

    save_experts : bool, default=False
        If True, save diagnostic information from the fit:
        - ``candidates_``: candidate constant-rebalanced portfolios (k, n_assets)
        - ``log_wealth_``: log cumulative wealth per candidate (k,)
        - ``wealth_softmax_``: softmax-normalized wealth weights (k,)
        - ``expert_weights_``: weights used over candidates per ``weight_scheme`` (k,)

    portfolio_params : dict, optional
        Portfolio parameters passed to the :class:`~skfolio.portfolio.Portfolio`
        produced by ``predict`` and ``score``.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Estimated next-period portfolio weights.
    """

    def __init__(
        self,
        grid_step: float | None = None,
        n_samples: int = 10_000,
        random_state: int | np.random.Generator | None = None,
        weight_scheme: str = "wealth",
        top_k: int = 1,
        save_experts: bool = False,
        # projection & online options
        min_weights: float | np.ndarray | None = 0.0,
        max_weights: float | np.ndarray | None = 1.0,
        budget: float | None = 1.0,
        min_budget: float | None = None,
        max_budget: float | None = None,
        max_short: float | None = None,
        max_long: float | None = None,
        max_turnover: float | None = None,
        previous_weights: np.ndarray | None = None,
        use_convex_projection: bool = False,
        groups: np.ndarray | None = None,
        linear_constraints: list[str] | None = None,
        left_inequality: np.ndarray | None = None,
        right_inequality: np.ndarray | None = None,
        X: np.ndarray | None = None,
        tracking_error_benchmark: np.ndarray | None = None,
        max_tracking_error: float | None = None,
        covariance: np.ndarray | None = None,
        variance_bound: float | None = None,
        solver: str | None = None,
        portfolio_params: dict | None = None,
    ) -> None:
        super().__init__(portfolio_params=portfolio_params)
        self.grid_step = grid_step
        self.n_samples = n_samples
        self.random_state = random_state
        self.weight_scheme = weight_scheme
        self.top_k = top_k
        self.save_experts = save_experts
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.budget = budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.max_short = max_short
        self.max_long = max_long
        self.max_turnover = max_turnover
        # Expose previous_weights so BaseOptimization.predict can pass it to Portfolio
        self.previous_weights = previous_weights
        self.use_convex_projection = use_convex_projection
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.left_inequality = left_inequality
        self.right_inequality = right_inequality
        self.tracking_error_benchmark = tracking_error_benchmark
        self.max_tracking_error = max_tracking_error
        self.covariance = covariance
        self.variance_bound = variance_bound
        self.solver = solver

    def _rng(self) -> np.random.Generator:
        rs = self.random_state
        if isinstance(rs, np.random.Generator):
            return rs
        return np.random.default_rng(rs)

    def _generate_portfolios(self, n_assets: int) -> np.ndarray:
        if self.grid_step is not None:
            if not (0 < self.grid_step <= 1):
                raise ValueError("grid_step must be in (0, 1].")
            m = int(round(1.0 / self.grid_step))
            # sanity check for combinatorial explosion
            n_points = math.comb(m + n_assets - 1, n_assets - 1)
            if n_points > 1_000_000:
                raise ValueError(
                    "Grid size too large. Reduce grid_step or use sampling via n_samples."
                )
            grid = _integer_simplex_grid(n_assets, m)
            return grid / grid.sum(axis=1, keepdims=True)
        # Dirichlet sampling (uniform over the simplex)
        rng = self._rng()
        return rng.dirichlet(alpha=np.ones(n_assets), size=self.n_samples)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "Universal":
        """Fit the Universal Portfolio estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets (arithmetic). We internally use gross
            returns ``1 + r_t`` in the wealth computation.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : UniversalPortfolio
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        n_assets = int(np.asarray(X).shape[1])  # type: ignore

        # Convert to gross returns and guard against non-positive values
        gross = 1.0 + np.asarray(X, dtype=float)
        if np.any(gross <= 0):
            raise ValueError(
                "Gross returns (1 + r) must be strictly positive for all observations."
            )

        # Candidate constant-rebalanced portfolios
        P = self._generate_portfolios(n_assets)  # (k, n)

        # Wealth of each candidate: W_k = prod_t (x_t^T p_k)
        # Compute daily portfolio gross returns for all candidates at once
        # (T, n) @ (n, k) -> (T, k)
        daily = gross @ P.T
        # Numerical stability: work in log-domain
        daily = np.clip(daily, 1e-12, None)
        log_wealth = np.log(daily).sum(axis=0)  # (k,)

        # Choose expert weighting scheme
        scheme = self.weight_scheme.lower()
        if scheme == "wealth":
            log_wealth -= log_wealth.max()
            candidate_probs = np.exp(log_wealth)
            candidate_probs /= candidate_probs.sum()
        elif scheme == "uniform":
            candidate_probs = np.ones(P.shape[0], dtype=float)
            candidate_probs /= candidate_probs.size
        elif scheme == "top_k":
            k = int(self.top_k) if isinstance(self.top_k, (int, np.integer)) else 1
            k = max(1, min(k, P.shape[0]))
            # pick top-k by wealth (equivalently log_wealth)
            idx = np.argpartition(log_wealth, -k)[-k:]
            candidate_probs = np.zeros(P.shape[0], dtype=float)
            candidate_probs[idx] = 1.0 / k
        else:
            raise ValueError(
                "weight_scheme must be one of {'wealth','uniform','top_k'}"
            )

        # Expected portfolio under chosen expert weights
        w_raw = candidate_probs @ P  # (n,)
        # Project with constraints
        if self.use_convex_projection:
            self.weights_ = project_convex(
                w_raw=w_raw,
                budget=self.budget,
                lower=self.min_weights,
                upper=self.max_weights,
                min_budget=self.min_budget,
                max_budget=self.max_budget,
                max_short=self.max_short,
                max_long=self.max_long,
                groups=self.groups,
                linear_constraints=self.linear_constraints,
                left_inequality=self.left_inequality,
                right_inequality=self.right_inequality,
                tracking_error_benchmark=self.tracking_error_benchmark,
                max_tracking_error=self.max_tracking_error,
                covariance=self.covariance,
                variance_bound=self.variance_bound,
                solver=self.solver,
            )
        else:
            self.weights_ = project_with_turnover(
                w_raw=w_raw,
                previous_weights=self.previous_weights,
                max_turnover=self.max_turnover,
                lower=self.min_weights,
                upper=self.max_weights,
                budget=self.budget if self.budget is not None else 1.0,
            )

        # Optionally save diagnostics
        if self.save_experts:
            self.candidates_ = P
            self.log_wealth_ = log_wealth
            # Wealth softmax independent of chosen scheme for diagnostics
            lw = log_wealth - log_wealth.max()
            ws = np.exp(lw)
            self.wealth_softmax_ = ws / ws.sum()
            self.expert_weights_ = candidate_probs
        return self
