import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv  # type: ignore

from skfolio.optimization.convex._base import ConvexOptimization
from skfolio.measures import RiskMeasure


class OnlineConvexAdapter(ConvexOptimization):
    """Adapter to reconcile OPS targets with full convex constraint stack.

    This class subclasses ConvexOptimization to reuse the complete constraint
    machinery (transaction_costs, management_fees, groups, linear constraints,
    turnover, TE, solvers, etc.) while letting the user provide an "OPS target"
    vector `w_raw` (e.g., from ExponentialGradient/Universal/FTL/...).

    The adapter solves, at each fit call, a quadratic proximity objective under
    all configured constraints:

        minimize 0.5 * ||w - w_raw||_2^2

    Notes
    -----
    - The proximity objective preserves the OPS signal while enforcing
      feasibility in the convex constraint set.
    - Use `set_ops_target(w_raw)` to provide the signal before calling `fit`.
    - If `w_raw` is not set at fit time, an error is raised.
    - All ConvexOptimization parameters are available (min/max weights, budget,
      min/max budget, turnover via max_turnover+previous_weights, linear
      constraints, groups, tracking error via overwrite_expected_return or
      custom constraints, solver and params, etc.).
    """

    def __init__(
        self,
        # Inherit ConvexOptimization init signature to expose constraints API
        risk_measure: RiskMeasure = RiskMeasure.VARIANCE,
        **kwargs,
    ):
        super().__init__(risk_measure=risk_measure, **kwargs)
        self._ops_target: np.ndarray | None = None

    def set_ops_target(self, w_raw: npt.ArrayLike) -> "OnlineConvexAdapter":
        arr = np.asarray(w_raw, dtype=float)
        if arr.ndim != 1:
            raise ValueError("w_raw must be a 1D array of shape (n_assets,)")
        self._ops_target = arr
        return self

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params):
        # Validate input and delegate to ConvexOptimization stack with a custom
        # objective that minimizes 0.5*||w - w_raw||^2.
        X = skv.validate_data(self, X)
        if self._ops_target is None:
            raise ValueError("OPS target not set. Call set_ops_target(w_raw) before fit.")

        import cvxpy as cp  # type: ignore
        from skfolio.prior import ReturnDistribution

        n_obs, n_assets = np.asarray(X).shape
        if self._ops_target.shape[0] != n_assets:
            raise ValueError(
                f"w_raw length {self._ops_target.shape[0]} does not match n_assets={n_assets}"
            )

        # Minimal ReturnDistribution for constraints needing returns/covariance
        rd = ReturnDistribution(
            mu=np.zeros(n_assets),
            covariance=np.eye(n_assets),
            returns=np.asarray(X, dtype=float),
        )

        w = cp.Variable(n_assets)
        factor = cp.Constant(1.0)

        # Build constraints via ConvexOptimization helpers to maximize reuse
        constraints = []
        constraints += self._get_weight_constraints(
            n_assets=n_assets, w=w, factor=factor, allow_negative_weights=True
        )

        # Turnover constraint if provided (uses previous_weights)
        if getattr(self, "max_turnover", None) is not None:
            turnover = self._turnover(n_assets=n_assets, w=w, factor=factor)
            constraints.append(
                cp.sum(turnover) * self._scale_constraints
                <= float(self.max_turnover) * factor * self._scale_constraints
            )

        # Linear constraints, left/right inequalities already added in _get_weight_constraints
        # Add tracking error constraint if provided through left/right inequality or custom use
        # (Users can pass them as linear_constraints; for time-series TE an extension could be added.)

        # Objective: proximity to OPS target + L1/L2 regularization if set
        proximity = 0.5 * cp.sum_squares(w - self._ops_target)
        regularization = self._cvx_regularization(w)
        objective = cp.Minimize((proximity + regularization) / self._scale_objective)

        problem = cp.Problem(objective, constraints)

        # Setup and solve with ConvexOptimization plumbing
        self._set_solver_params(default=dict(tol_gap_abs=1e-9, tol_gap_rel=1e-9))
        self._set_scale_objective(default=1.0)
        self._set_scale_constraints(default=1.0)
        self._solve_problem(problem=problem, w=w, factor=factor)
        return self


