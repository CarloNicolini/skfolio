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
from skfolio.measures import PerfMeasure, RiskMeasure
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
            if self.portfolio_params is not None:
                self.portfolio_params["name"] = "UCRP"
        else:
            self.weights_ = self.weights
            if self.portfolio_params is not None:
                self.portfolio_params["name"] = "CRP"
        return self

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):  # type: ignore[override]
        return self.partial_fit(X, y)


# utility class for UCRP
class UCRP(CRP):
    pass


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

    Optimizes portfolio weights on historical data using any convex objective:
    - Risk minimization (variance, CVaR, CDaR, etc.)
    - Log-wealth maximization (Kelly criterion)
    - Mean-risk utility maximization

    Parameters
    ----------
    objective_measure : RiskMeasure | PerfMeasure, default=PerfMeasure.LOG_WEALTH
        The measure to optimize:

        - If ``PerfMeasure.LOG_WEALTH``: maximizes cumulative log-wealth
          ``sum_t log((1 + r_t)^T w)``
        - If ``RiskMeasure``: uses specified ``objective_function``
          (minimize risk, maximize utility, etc.)

    objective_function : ObjectiveFunction, default=ObjectiveFunction.MAXIMIZE_RETURN
        Objective function when using a RiskMeasure:

        - ``MINIMIZE_RISK``: minimize the risk measure
        - ``MAXIMIZE_UTILITY``: maximize mean - risk_aversion * risk
        - ``MAXIMIZE_RATIO``: maximize Sharpe-like ratio

        Ignored when ``objective_measure`` is ``LOG_WEALTH``.

    risk_aversion : float, default=1.0
        Risk aversion parameter for ``MAXIMIZE_UTILITY`` objective.
        Higher values lead to more conservative portfolios.

    l2_coef : float, default=0.0
        L2 regularization coefficient for weight smoothness.

    transaction_costs : float | dict | array-like, default=0.0
        Transaction costs per asset (proportional).

    management_fees : float | dict | array-like, default=0.0
        Management fees per asset (per-period).

    previous_weights : float | dict | array-like | None, default=None
        Previous weights for computing transaction costs and turnover.

    groups : dict | array-like | None, default=None
        Asset groups for linear constraints.

    linear_constraints : list[str] | None, default=None
        Linear constraints on weights (e.g., ``"Equity >= 0.5"``).

    left_inequality : array-like | None, default=None
        Left-hand side matrix A for inequality Aw <= b.

    right_inequality : array-like | None, default=None
        Right-hand side vector b for inequality Aw <= b.

    max_turnover : float | None, default=None
        Maximum allowed turnover per period (L1 norm of weight changes).

    min_weights : float | dict | array-like | None, default=0.0
        Minimum weight per asset (lower bound).

    max_weights : float | dict | array-like | None, default=1.0
        Maximum weight per asset (upper bound).

    budget : float | None, default=1.0
        Investment budget (sum of weights). If None, no budget constraint.

    solver : str, default="CLARABEL"
        CVXPY solver to use.

    solver_params : dict | None, default=None
        Additional parameters passed to the solver.

    scale_objective : float | None, default=None
        Scale factor for the objective function.

    scale_constraints : float | None, default=None
        Scale factor for constraints.

    save_problem : bool, default=False
        If True, save the CVXPY problem in ``problem_`` attribute.

    raise_on_failure : bool, default=True
        If True, raise an error when optimization fails.

    add_objective : callable | None, default=None
        Custom objective term to add to the optimization.

    add_constraints : callable | None, default=None
        Custom constraints to add to the optimization.

    portfolio_params : dict | None, default=None
        Additional portfolio parameters.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Optimized portfolio weights.

    all_weights_ : ndarray of shape (T, n_assets)
        Weights from ``fit_dynamic`` method (dynamic regret).

    Examples
    --------
    >>> from skfolio.optimization.online import BCRP
    >>> from skfolio.measures import RiskMeasure, PerfMeasure
    >>> from skfolio.optimization.convex._base import ObjectiveFunction
    >>> import numpy as np
    >>>
    >>> # Log-wealth maximization (default)
    >>> bcrp_log = BCRP()
    >>>
    >>> # Variance minimization
    >>> bcrp_var = BCRP(
    ...     objective_measure=RiskMeasure.VARIANCE,
    ...     objective_function=ObjectiveFunction.MINIMIZE_RISK
    ... )
    >>>
    >>> # CVaR minimization
    >>> bcrp_cvar = BCRP(
    ...     objective_measure=RiskMeasure.CVAR,
    ...     objective_function=ObjectiveFunction.MINIMIZE_RISK,
    ...     cvar_beta=0.95
    ... )
    >>>
    >>> # Mean-variance utility
    >>> bcrp_utility = BCRP(
    ...     objective_measure=RiskMeasure.VARIANCE,
    ...     objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
    ...     risk_aversion=2.0
    ... )

    References
    ----------
    .. [1] Li, B., & Hoi, S. C. H. (2013). Online Portfolio Selection: A Survey.
           arXiv:1212.2129.
    """

    name: str = "BCRP"
    n_features_in_: int

    @staticmethod
    def _log_wealth_expr(w: cp.Variable, estimator) -> cp.Expression:
        """Log-wealth objective: sum_t log(r_t^T w)."""
        rd = estimator.prior_estimator_.return_distribution_
        relatives = 1.0 + rd.returns
        return cp.sum(cp.log(relatives @ w))

    def __init__(
        self,
        objective_measure: RiskMeasure | PerfMeasure = PerfMeasure.LOG_WEALTH,
        objective_function: ObjectiveFunction = ObjectiveFunction.MAXIMIZE_RETURN,
        risk_aversion: float = 1.0,
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
        # Additional MeanRisk parameters for risk measures
        cvar_beta: float = 0.95,
        evar_beta: float = 0.95,
        cdar_beta: float = 0.95,
        edar_beta: float = 0.95,
        min_acceptable_return: skt.Target | None = None,
        risk_free_rate: float = 0.0,
    ):
        # Determine configuration based on objective_measure
        if objective_measure == PerfMeasure.LOG_WEALTH:
            # Log-wealth maximization: override expected return with log-sum
            risk_measure = RiskMeasure.VARIANCE  # Placeholder, not used
            obj_func = ObjectiveFunction.MAXIMIZE_RETURN
            overwrite_expected_return = BCRP._log_wealth_expr
        else:
            # Standard risk measure optimization
            if not isinstance(objective_measure, RiskMeasure):
                raise ValueError(
                    f"objective_measure must be RiskMeasure or PerfMeasure.LOG_WEALTH, "
                    f"got {type(objective_measure).__name__}"
                )
            risk_measure = objective_measure
            obj_func = objective_function
            overwrite_expected_return = None

        super().__init__(
            objective_function=obj_func,
            risk_measure=risk_measure,
            risk_aversion=risk_aversion,
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
            save_problem=save_problem,
            raise_on_failure=raise_on_failure,
            add_objective=add_objective,
            add_constraints=add_constraints,
            overwrite_expected_return=overwrite_expected_return,
            portfolio_params=portfolio_params,
            cvar_beta=cvar_beta,
            evar_beta=evar_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
            min_acceptable_return=min_acceptable_return,
            risk_free_rate=risk_free_rate,
        )
        self.max_turnover = max_turnover
        self.objective_measure = objective_measure

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

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = BCRP()
    model.fit_dynamic(X)
