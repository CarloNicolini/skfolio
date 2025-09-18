import warnings

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv  # type: ignore

from skfolio.optimization._base import BaseOptimization
from skfolio.utils.projection import project_convex, project_with_turnover


class AntiCor(BaseOptimization):
    """AntiCor optimizer adapted to skfolio (Borodin et al., 2003).

    Transfers weight from outperformers to anti-correlated underperformers
    using two consecutive windows.
    """

    def __init__(
        self,
        window_size: int = 10,
        max_window: int | None = None,
        use_multiple_windows: bool = False,
        min_history: int | None = None,
        # projection options
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
    ):
        super().__init__(portfolio_params=portfolio_params)
        self.window_size = window_size
        self.max_window = max_window or window_size
        self.use_multiple_windows = use_multiple_windows
        self.min_history = min_history
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

    def _compute_log_returns(
        self, returns: np.ndarray, start: int, end: int
    ) -> np.ndarray:
        rel = 1.0 + returns[start:end]
        rel = np.clip(rel, 1e-8, None)
        return np.log(rel)

    def _correlation_claims(self, LX1: np.ndarray, LX2: np.ndarray) -> np.ndarray:
        n = LX1.shape[1]
        mu1 = LX1.mean(axis=0)
        mu2 = LX2.mean(axis=0)
        std1 = LX1.std(axis=0, ddof=1)
        std2 = LX2.std(axis=0, ddof=1)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if std1[i] > 1e-8 and std2[j] > 1e-8:
                    cov = np.mean((LX1[:, i] - mu1[i]) * (LX2[:, j] - mu2[j]))
                    M[i, j] = cov / (std1[i] * std2[j])
        claims = np.zeros_like(M)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (mu2[i] > mu2[j]) and (M[i, j] > 0):
                    A_i = abs(M[i, i]) if M[i, i] < 0 else 0
                    A_j = abs(M[j, j]) if M[j, j] < 0 else 0
                    claims[i, j] = M[i, j] + A_i + A_j
        return claims

    def _apply_claims(self, w: np.ndarray, claims: np.ndarray) -> np.ndarray:
        n = len(w)
        new_w = w.copy()
        for i in range(n):
            total_out = claims[i, :].sum()
            if total_out > 0:
                for j in range(n):
                    if claims[i, j] > 0:
                        transfer = w[i] * claims[i, j] / total_out
                        new_w[i] -= transfer
                        new_w[j] += transfer
        new_w = np.maximum(new_w, 0)
        s = new_w.sum()
        return new_w / s if s > 0 else np.ones(n) / n

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "AntiCor":
        X = skv.validate_data(self, X)
        R = np.asarray(X, dtype=float)
        T, n = R.shape
        min_t = self.min_history or (4 * self.window_size)
        if T < min_t:
            self.weights_ = np.ones(n) / n
            return self

        if self.use_multiple_windows:
            all_w = []
            for w in range(2, self.max_window + 1):
                try:
                    all_w.append(self._fit_single(R, w))
                except Exception as e:  # pragma: no cover - defensive
                    warnings.warn(f"AntiCor failed for window={w}: {e}")
            w_raw = np.mean(all_w, axis=0) if all_w else np.ones(n) / n
        else:
            w_raw = self._fit_single(R, self.window_size)

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

    def _fit_single(self, R: np.ndarray, window_size: int) -> np.ndarray:
        T, n = R.shape
        w = np.ones(n) / n
        for t in range(2 * window_size, T):
            LX1 = self._compute_log_returns(R, t - 2 * window_size, t - window_size)
            LX2 = self._compute_log_returns(R, t - window_size, t)
            claims = self._correlation_claims(LX1, LX2)
            w = self._apply_claims(w, claims)
        return w
