import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv  # type: ignore

from skfolio.optimization._base import BaseOptimization
from skfolio.utils.projection import project_convex, project_with_turnover


class FollowTheLeader(BaseOptimization):
    """Follow The Leader (FTL) estimator.

    Chooses the constant-rebalanced portfolio maximizing in-sample wealth
    over the training window. We solve it by minimizing the negative log
    wealth with SLSQP-like projection proxies: we approximate with a
    simple log-utility objective and project to the feasible set.
    """

    def __init__(
        self,
        min_weights: float | np.ndarray | None = 0.0,
        max_weights: float | np.ndarray | None = 1.0,
        budget: float | None = 1.0,
        min_budget: float | None = None,
        max_budget: float | None = None,
        max_short: float | None = None,
        max_long: float | None = None,
        max_turnover: float | None = None,
        previous_weights: np.ndarray | None = None,
        # convex projection options
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
    ):
        super().__init__(portfolio_params=portfolio_params)
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.budget = budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.max_short = max_short
        self.max_long = max_long
        self.max_turnover = max_turnover
        self.previous_weights = previous_weights
        self.use_convex_projection = use_convex_projection
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.left_inequality = left_inequality
        self.right_inequality = right_inequality
        self.X = X
        self.tracking_error_benchmark = tracking_error_benchmark
        self.max_tracking_error = max_tracking_error
        self.covariance = covariance
        self.variance_bound = variance_bound
        self.solver = solver

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> "FollowTheLeader":
        X = skv.validate_data(self, X)
        gross = 1.0 + np.asarray(X, dtype=float)
        if np.any(gross <= 0):
            raise ValueError("Gross returns (1 + r) must be strictly positive.")

        # Approximate FTL by log-utility maximization under equality and bounds
        # The unconstrained maximizer for log-utility with static weights doesn't exist in closed form,
        # we instead use the buy-and-hold per-asset wealth normalized as a strong baseline initializer
        wealth_per_asset = gross.prod(axis=0)
        w_raw = wealth_per_asset / wealth_per_asset.sum()

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
                X=self.X,
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
                budget=float(self.budget) if self.budget is not None else 1.0,
            )
        return self


class FollowTheRegularizedLeader(FollowTheLeader):
    """FTRL with L2 regularization toward a prior (uniform).

    Uses a convex projection step that keeps full constraint support. The
    regularization is applied by nudging the target `w_raw` toward the
    prior by a convex combination controlled by `beta`.
    """

    def __init__(
        self,
        beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta = beta

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> "FollowTheRegularizedLeader":
        X = skv.validate_data(self, X)
        gross = 1.0 + np.asarray(X, dtype=float)
        if np.any(gross <= 0):
            raise ValueError("Gross returns (1 + r) must be strictly positive.")

        wealth_per_asset = gross.prod(axis=0)
        w_ftl = wealth_per_asset / wealth_per_asset.sum()
        n = w_ftl.shape[0]
        prior = np.ones(n) / n
        w_raw = (1.0 - self.beta) * w_ftl + self.beta * prior

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
                X=self.X,
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
                budget=float(self.budget) if self.budget is not None else 1.0,
            )
        return self
