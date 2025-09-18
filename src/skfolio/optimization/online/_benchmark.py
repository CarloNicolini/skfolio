"""Benchmarks for online portfolio selection.

Implements:
- CRP: Constant Rebalanced Portfolio with fixed weights (UCRP when uniform).
- BCRP: Best Constant Rebalanced Portfolio in hindsight, maximizing
  sum_t log(b^T x_t) over the simplex.

References
----------
- Li, B., & Hoi, S. C. H. (2013). Online Portfolio Selection: A Survey.
  arXiv:1212.2129 (`https://arxiv.org/abs/1212.2129`).
- Notes in `OCO/benchmarks.md`.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import numpy.typing as npt

import skfolio.typing as skt
from skfolio.measures import RiskMeasure
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.optimization.convex._mean_risk import MeanRisk
from skfolio.optimization.online._mixins import OnlineMixin
from skfolio.optimization.online._utils import net_to_relatives

# Keep a direct reference to cp to satisfy static linters
assert cp is not None


class _OnceFittable:
    """Mixin ensuring one-shot fitting semantics.

    Subsequent calls to fit/partial_fit return immediately without modifying state.
    """

    _finalized: bool

    def _mark_finalized(self) -> None:
        self._finalized = True

    def _is_finalized(self) -> bool:
        return getattr(self, "_finalized", False)


class CRP(BaseOptimization, _OnceFittable):
    """Constant Rebalanced Portfolio (CRP).

    Keeps a fixed portfolio w and rebalances to it every period.

    Parameters
    ----------
    weights : array-like of shape (n_assets,), optional
        Fixed weights to rebalance to. If None, uses uniform weights leading to the
        Uniform Constant Rebalanced Portfolio (UCRP).
    portfolio_params : dict, optional
        Portfolio parameters propagated to the produced portfolio.
    """

    weights_: np.ndarray
    n_features_in_: int

    def __init__(
        self,
        weights: npt.ArrayLike | None = None,
        portfolio_params: dict | None = None,
    ) -> None:
        super().__init__(portfolio_params=portfolio_params)
        self.weights = weights
        self.portfolio_params = portfolio_params

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        if self.weights is None:
            n = X.shape[1]
            self.weights_ = np.ones(n) / n
            self.portfolio_params["name"] = "UCRP"
        else:
            self.weights_ = self.weights
            self.portfolio_params["name"] = "CRP"
        return self

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):  # type: ignore[override]
        return self.partial_fit(X, y)


class BestStock(BaseOptimization, _OnceFittable):
    def __init__(self, portfolio_params: dict | None = None):
        super().__init__(portfolio_params=portfolio_params)

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        relatives = net_to_relatives(X)
        best_stock = np.argmax(np.sum(np.log(relatives), axis=0, keepdims=True), axis=1)
        self.weights_ = np.zeros(relatives.shape[1])
        self.weights_[best_stock] = 1.0
        return self

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        return self.partial_fit(X, y)


class BCRP(MeanRisk, OnlineMixin):
    """Best Constant Rebalanced Portfolio (BCRP) in hindsight.

    Maximizes cumulative log-wealth on the sample:

        maximize    sum_t log((1 + r_t)^T w)
        subject to  w in feasible set (simplex/bounds/budget/linear/etc.)
    """

    name: str = "BCRP"
    n_features_in_: int

    @staticmethod
    def _log_wealth_expr(w: cp.Variable, estimator: "BCRP") -> cp.Expression:
        # Use returns estimated by the prior to build price relatives
        rd = estimator.prior_estimator_.return_distribution_
        relatives = 1.0 + rd.returns  # TODO use the _to_relatives method here
        # Portfolio gross relatives per period, then sum of logs
        return cp.sum(cp.log(relatives @ w))

    def __init__(
        self,
        l2_coef: float = 0.0,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        previous_weights: skt.MultiInput | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1.0,
        solver: str = "CLARABEL",
        solver_params: dict | None = None,
        scale_objective: float | None = None,
        scale_constraints: float | None = None,
        save_problem: bool = False,
        raise_on_failure: bool = True,
        add_objective: skt.ExpressionFunction | None = None,
        add_constraints: skt.ExpressionFunction | None = None,
        portfolio_params: dict | None = None,
    ):
        super().__init__(
            objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
            risk_measure=RiskMeasure.VARIANCE,  # placeholder; not used in objective
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            l2_coef=l2_coef,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            previous_weights=previous_weights,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            solver=solver,
            solver_params=solver_params,
            scale_objective=scale_objective,
            scale_constraints=scale_constraints,
            save_problem=True,
            raise_on_failure=raise_on_failure,
            add_objective=add_objective,
            add_constraints=add_constraints,
            overwrite_expected_return=BCRP._log_wealth_expr,
            portfolio_params=portfolio_params,
        )
        self.max_turnover = max_turnover

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        return self.fit(X, y)

    def fit_dynamic(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params
    ) -> MeanRisk:
        """
        Helper method to fit over increasingly large folds of data from t=1 to T.
        Useful for use with regret calculation in dynamic regret settings.

        Parameters
        ----------
        X : npt.ArrayLike
            _description_
        y : npt.ArrayLike | None, optional
            _description_, by default None

        Returns
        -------
        MeanRisk
            _description_
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D array of shape (T, n_assets)")
        T, n = X_arr.shape
        weights_list: list[np.ndarray] = []
        for t in range(1, T + 1):
            if t == 1:
                weights_list.append(BestStock().fit(X_arr[:t, :]).weights_.copy())
            else:
                # Fit on prefix up to time t (inclusive)
                self.fit(X_arr[:t, :], y)
                weights_list.append(self.weights_.copy())
        self.all_weights_ = np.vstack(weights_list)
        return self


if __name__ == "__main__":
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.optimization.online._regret import compute_regret_curve

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = BCRP()
    model.fit_dynamic(X)
