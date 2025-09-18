import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from skfolio.optimization.online._base import OPS

from skfolio.optimization.online._utils import integer_simplex_grid

from ._mixins import (
    CORNMixin,
    FollowTheLoserMixin,
    OnlineMethod,
    OnlineMixin,
)

CLIP_EPSILON = 1e-12


def gradient_log_wealth(
    w: np.ndarray, x_gross: np.ndarray, l2_coef: float = 0.0
) -> np.ndarray:
    """Kelly-style gradient of log-wealth with optional L2 regularization.

    Uses CLIP_EPSILON for numerical safety and estimator.l2_coef.
    """
    denom = float(np.dot(w, x_gross))
    denom = max(denom, CLIP_EPSILON)
    return -x_gross / denom + 2.0 * l2_coef * w


def hinge_gradient(
    w: np.ndarray, x_tilde: np.ndarray, C: float, tau: float
) -> np.ndarray:
    """Passive-Aggressive style hinge subgradient toward feature x_tilde."""
    val = float(tau) - float(np.dot(w, x_tilde))
    if val > 0:
        return -float(C) * x_tilde
    return np.zeros_like(w)


class Loss(ABC):
    def __init__(self, estimator: "OPS") -> None:
        self.estimator = estimator

    @abstractmethod
    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError("Must implement gradient computation")

    @abstractmethod
    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Must implement loss computation")

    def log_wealth(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        return np.log(np.dot(w, x_gross)).sum()


class BuyAndHoldLoss(Loss, OnlineMixin):
    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        return np.zeros_like(w)

    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        return 0.0


class KellyLoss(Loss, OnlineMixin):
    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        return gradient_log_wealth(w, x_gross)

    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        return 0.0


class FollowTheLeaderLoss(Loss, OnlineMixin):
    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        # Important: return only the current gradient.
        # Cumulative gradients are handled implicitly by the EMD update,
        # and passing cumulative gradients here would double-count.
        return gradient_log_wealth(w, x_gross)

    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        return 0.0


class FollowTheLoserLoss(Loss, OnlineMixin, FollowTheLoserMixin):
    def _default_reversion_feature(self) -> np.ndarray:
        if len(self.estimator._history_gross_relatives) == 0:
            return np.ones(int(self.estimator.n_features_in_), dtype=float)
        window = min(
            self.estimator.reversion_window,
            len(self.estimator._history_gross_relatives),
        )
        windowed = np.vstack(self.estimator._history_gross_relatives[-window:])
        moving_average = np.maximum(windowed.mean(axis=0), CLIP_EPSILON)
        return 1.0 / moving_average

    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        g = gradient_log_wealth(w, x_gross)
        x_tilde = self._default_reversion_feature()
        g += hinge_gradient(w, x_tilde, C=1.0, tau=1.0)
        return g

    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        return 0.0


class OLMARLoss(Loss, OnlineMixin):
    """Online Moving Average Reversion (OLMAR) loss.

    Uses Kelly-style gradient on a moving-average prediction feature.
    """

    def _moving_average_prediction(self, estimator) -> np.ndarray:
        """Compute moving average prediction for OLMAR."""
        if len(estimator._history_gross_relatives) == 0:
            return np.ones(int(estimator.n_features_in_), dtype=float)
        # Use a simple moving average window (could be parameterized)
        window = min(5, len(estimator._history_gross_relatives))
        windowed = np.vstack(estimator._history_gross_relatives[-window:])
        return windowed.mean(axis=0)

    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        x_pred = self._moving_average_prediction(self.estimator)
        return gradient_log_wealth(w, x_pred)

    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        return 0.0


class CORNLoss(Loss, OnlineMixin, CORNMixin):
    def _corn_scenarios(self) -> list[np.ndarray]:
        if len(self.estimator._history_gross_relatives) <= self.estimator.corn_window:
            return []
        X_all = np.vstack(self.estimator._history_gross_relatives)
        num_rows = X_all.shape[0]
        last_block = X_all[num_rows - self.estimator.corn_window : num_rows].reshape(
            self.estimator.corn_window, -1
        )
        correlations: list[tuple[float, int]] = []
        for t_index in range(
            self.estimator.corn_window, num_rows - self.estimator.corn_window + 1
        ):
            segment = X_all[t_index - self.estimator.corn_window : t_index]
            corr = float(np.corrcoef(last_block.ravel(), segment.ravel())[0, 1])
            correlations.append((corr, t_index))
        correlations.sort(reverse=True, key=lambda z: (z[0], -z[1]))
        selected = [t for _, t in correlations[: self.estimator.corn_k]]
        return [X_all[t] for t in selected]

    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        scenarios = self._corn_scenarios()
        if len(scenarios) == 0:
            return gradient_log_wealth(w, x_gross)
        grad = np.zeros_like(w)
        for xs in scenarios:
            denom = float(np.dot(w, xs))
            denom = max(denom, CLIP_EPSILON)
            grad += -xs / denom
        grad /= float(len(scenarios))
        l2 = float(kwargs.get("l2_coef", 0.0))
        return grad + 2.0 * l2 * w

    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        return 0.0


class SmoothPredictionLoss(Loss, OnlineMixin):
    """Smooth Prediction loss with log-barrier regularization.

    Implements the Follow-the-Regularized-Leader (FTRL) algorithm with log-barrier
    regularization from Rakhlin & Tewari (2008) and others.

    The algorithm solves:
        x_t = arg max_{x ∈ Δ_n} ∑_{i=1}^{t-1} log(r_i^T x) + ε ∑_{j=1}^n log(x_j)

    where ε is the regularization parameter (smooth_epsilon).

    Key properties:
    - Regret bound: O(n/(min r_t) * log(T)) for known bounds
    - Runtime: O(n^2.5 T) per iteration (due to projection complexity)
    - Can be universalized for O(n√(T log T)) regret

    References
    ----------
    - Rakhlin, A., & Tewari, A. (2008). Online learning via sequential complexities.
    - Blog post: https://sudeepraja.github.io/OPS4/
    """

    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        """Compute gradient combining Kelly objective and log-barrier regularization.

        Returns the negative gradient for minimization of:
        -log(w^T x) - ε * ∑ log(w_i)

        which is equivalent to maximizing:
        log(w^T x) + ε * ∑ log(w_i)
        """
        # Standard Kelly gradient: -x_gross / (w^T x_gross)
        kelly_grad = gradient_log_wealth(w, x_gross, l2_coef=self.estimator.l2_coef)

        # Log-barrier regularization gradient: -ε / w_i for each component
        # Use numerical clipping for stability near zero
        w_clipped = np.maximum(w, CLIP_EPSILON)
        smooth_epsilon = getattr(self.estimator, "smooth_epsilon", 1.0)
        regularization_grad = -float(smooth_epsilon) / w_clipped

        return kelly_grad + regularization_grad

    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        """Compute the loss (currently returns 0 as loss tracking is optional)."""
        return 0.0


class EGTildeLoss(Loss, OnlineMixin):
    """EG-Tilde loss with time-varying regularization and modified returns.

    Implements the Exponentiated Gradient algorithm with tilting (EG-Tilde) from
    the blog post https://sudeepraja.github.io/OPS3/. This is a variant of the
    Hedge algorithm that uses:

    1. Time-varying regularization parameters:
       - alpha_t = (n^2 * log(n) / (t+1))^0.25
       - eta_t = (alpha_t / n) * sqrt(log(n) / (t+1))

    2. Modified returns for gradient computation:
       - tilde_r_t = (1 - alpha_t/n) * r_t + alpha_t/n

    3. Portfolio mixing after update:
       - tilde_x_{t+1} = (1 - alpha_t) * x_{t+1} + alpha_t/n

    Key properties:
    - Improved regret bounds compared to standard Hedge
    - Adaptive regularization based on problem dimension and time
    - Automatic balancing between exploration and exploitation

    References
    ----------
    - Blog post: https://sudeepraja.github.io/OPS3/
    """

    def __init__(self, estimator: "OPS") -> None:
        super().__init__(estimator)
        self._t = 0  # Time counter for parameters

    def _compute_time_varying_params(self, n: int, t: int) -> tuple[float, float]:
        """Compute alpha_t and eta_t for time step t."""
        if t == 0:
            return 0.0, 0.0

        log_n = np.log(n)
        alpha_t = (n * n * log_n / (t + 1)) ** 0.25
        eta_t = (alpha_t / n) * np.sqrt(log_n / (t + 1))
        return alpha_t, eta_t

    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        """Compute EG-Tilde gradient with modified returns and time-varying parameters."""
        n = w.shape[0]
        self._t += 1

        # Get time-varying parameters
        alpha_t, eta_t = self._compute_time_varying_params(n, self._t)

        # Store parameters for descent algorithm access
        self._current_alpha_t = alpha_t
        self._current_eta_t = eta_t

        if alpha_t == 0.0:  # First iteration
            return gradient_log_wealth(w, x_gross, l2_coef=self.estimator.l2_coef)

        # Modified returns: tilde_r = (1 - alpha_t/n) * r + alpha_t/n
        tilde_r = (1.0 - alpha_t / n) * x_gross + alpha_t / n

        # Kelly gradient on modified returns: -tilde_r / (w^T tilde_r)
        denom = float(np.dot(w, tilde_r))
        denom = max(denom, CLIP_EPSILON)
        gradient = -tilde_r / denom + 2.0 * self.estimator.l2_coef * w

        return gradient

    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        """Compute the loss (currently returns 0 as loss tracking is optional)."""
        return 0.0


class UniversalLoss(Loss):
    def __init__(self, estimator: "OPS") -> None:
        self.estimator = estimator
        self._alpha: np.ndarray | None = None
        self._experts: np.ndarray | None = None

    def _build_experts(self, n: int) -> np.ndarray:
        """Build the experts matrix. Experts are portfolios in the simplex.
        They can either be generated by a simplex grid or by sampling from a Dirichlet prior.
        """
        # 1) User-supplied experts matrix has priority
        if getattr(self.estimator, "experts", None) is not None:
            M_user = np.asarray(self.estimator.experts, dtype=float)
            if M_user.ndim != 2 or M_user.shape[0] != n:
                raise ValueError("experts matrix must have shape (n_assets, n_experts)")
            return M_user

        # 2) Grid discretization of the simplex if requested
        grid_step = self.estimator.universal_grid_step
        max_pts = self.estimator.universal_max_grid_points
        if grid_step is not None:
            if not (0.0 < float(grid_step) <= 1.0):
                raise ValueError("universal_grid_step must be in (0, 1].")
            m = round(1.0 / float(grid_step))
            n_points = math.comb(m + n - 1, n - 1)
            if n_points <= max_pts:
                grid = integer_simplex_grid(n_assets=n, m=m)
                # normalize to sum to 1 (points sum to m)
                P = grid / grid.sum(axis=1, keepdims=True)
                return P.T  # shape (n, k)
            # fall-through to sampling if grid is too large

        # 3) Dirichlet sampling as Monte Carlo / discretization
        rng = np.random.default_rng()
        n_samples = self.estimator.universal_n_samples
        alpha = self.estimator.universal_dirichlet_alpha
        if np.isscalar(alpha):
            # robust scalar conversion without relying on float() typing
            alpha_scalar = np.asarray(alpha, dtype=np.float64).item()
            alpha_vec = np.full((n,), alpha_scalar, dtype=np.float64)
        else:
            alpha_vec = np.asarray(alpha, dtype=np.float64)
            if alpha_vec.shape != (n,):
                raise ValueError(
                    "universal_dirichlet_alpha must be scalar or shape (n_assets,)"
                )
        P = rng.dirichlet(
            alpha=alpha_vec.astype(np.float64, copy=False), size=n_samples
        )
        return P.T  # shape (n, k)

    def update_weights(self, w: np.ndarray, x_gross: np.ndarray) -> np.ndarray | None:
        if self._experts is None:
            n = x_gross.shape[0]
            self._experts = self._build_experts(n)
        M = self._experts  # shape (n, k)
        # Use gross relatives provided by caller (fees modeled upstream)
        z = np.maximum(M.T @ x_gross, CLIP_EPSILON)
        losses = -np.log(z)
        # Add transaction-cost penalty per expert relative to previous weights
        tx = self.estimator._transaction_costs_arr
        prev = self.estimator.previous_weights
        if tx is not None and prev is not None:
            prev_arr = np.asarray(prev, dtype=float)
            if prev_arr.shape == (M.shape[0],):
                # cost_i = sum_j c_j * |M_{j,i} - prev_j|
                tx_arr = (
                    np.asarray(tx, dtype=float)
                    if not np.isscalar(tx)
                    else np.full(M.shape[0], tx, dtype=float)
                )
                cost_per_expert = (tx_arr[:, None] * np.abs(M - prev_arr[:, None])).sum(
                    axis=0
                )
                losses = losses + cost_per_expert
        if self._alpha is None:
            self._alpha = np.ones(M.shape[1], dtype=float) / float(M.shape[1])
        log_alpha = (
            np.log(np.maximum(self._alpha, CLIP_EPSILON)) - self.estimator.eta0 * losses
        )
        log_alpha -= np.max(log_alpha)
        alpha_raw = np.exp(log_alpha)
        s = float(alpha_raw.sum())
        self._alpha = (
            (alpha_raw / s) if s > 0 else (np.ones_like(alpha_raw) / alpha_raw.size)
        )
        w_proposed = M @ self._alpha  # convex combination of expert portfolios

        projector = self.estimator._projector
        if projector is None:
            return w_proposed
        return projector.project(w_proposed)

    def gradient(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> np.ndarray:
        return np.zeros_like(w)

    def loss(self, w: np.ndarray, x_gross: np.ndarray, **kwargs) -> float:
        return 0.0


losses_map = {
    OnlineMethod.BUY_AND_HOLD: BuyAndHoldLoss,
    OnlineMethod.HEDGE: KellyLoss,
    OnlineMethod.EG_TILDE: EGTildeLoss,
    OnlineMethod.FOLLOW_THE_LEADER: FollowTheLeaderLoss,
    OnlineMethod.FOLLOW_THE_LOSER: FollowTheLoserLoss,
    OnlineMethod.OLMAR: OLMARLoss,
    OnlineMethod.CORN: CORNLoss,
    OnlineMethod.SMOOTH_PRED: SmoothPredictionLoss,
    OnlineMethod.UNIVERSAL: UniversalLoss,
}
