from __future__ import annotations

from dataclasses import dataclass
from enum import auto
from numbers import Integral, Real
from typing import Any, Optional, Protocol

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from sklearn.utils._param_validation import (  # type: ignore
    Interval,
    StrOptions,
)
from sklearn.utils.validation import _check_sample_weight, validate_data

from skfolio.optimization.online._base import OnlinePortfolioSelection
from skfolio.optimization.online._ftrl import FTRL, LastGradPredictor
from skfolio.optimization.online._mirror_maps import (
    BaseMirrorMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
)
from skfolio.optimization.online._mixins import LoserStrategy, OLMARVariant
from skfolio.optimization.online._projection import AutoProjector
from skfolio.optimization.online._utils import CLIP_EPSILON, net_to_relatives
from skfolio.utils.tools import AutoEnum


class Surrogate(Protocol):
    def value(self, w: np.ndarray, phi: np.ndarray) -> float: ...
    def grad(self, w: np.ndarray, phi: np.ndarray) -> np.ndarray: ...


@dataclass
class HingeSurrogate:
    epsilon: float

    def value(self, w, phi):
        m = float(phi @ w)
        return max(0.0, self.epsilon - m)

    def grad(self, w, phi):
        return -phi if (self.epsilon - float(phi @ w) > 0.0) else np.zeros_like(phi)


@dataclass
class SquaredHingeSurrogate:
    epsilon: float

    def value(self, w, phi):
        m = self.epsilon - float(phi @ w)
        return (m * m) if m > 0.0 else 0.0

    def grad(self, w, phi):
        m = self.epsilon - float(phi @ w)
        return (-2.0 * m) * phi if m > 0.0 else np.zeros_like(phi)


@dataclass
class SoftplusSurrogate:
    epsilon: float
    beta: float = 5.0

    def value(self, w, phi):
        z = self.beta * (self.epsilon - float(phi @ w))
        if z > 50:
            return z / self.beta
        if z < -50:
            return np.exp(z) / self.beta
        return np.log1p(np.exp(z)) / self.beta

    def grad(self, w, phi):
        z = self.beta * (self.epsilon - float(phi @ w))
        if z >= 0:
            s = 1.0 / (1.0 + np.exp(-z))
        else:
            ez = np.exp(z)
            s = ez / (1.0 + ez)
        return -s * phi


class BaseReversionPredictor:
    def reset(self, d: int) -> None:
        self._d = d

    def update_and_predict(self, x_t: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OLMAR1Predictor(BaseReversionPredictor):
    """
    OLMAR-1 reversion predictor φ_{t+1} with explicit variant.

    Parameters
    ----------
    window : int, default=5
        Window size for the predictor.
    variant : OLMARVariant, default=OLMARVariant.OLPS
        Variant of the predictor.
      - variant="olps":     average of [1, 1/x_T, 1/(x_T x_{T-1}), ..., W terms]
      - variant="cumprod":  average of [1/x_T, 1/(x_T x_{T-1}), ..., W terms] (no leading 1)

    Cold-start rule: if T < W+1, return last observed relative x_T (OLPS schedule).
    """

    def __init__(self, window: int = 5, variant: OLMARVariant = OLMARVariant.OLPS):
        if window < 1:
            raise ValueError("window must be >= 1.")
        if variant not in (OLMARVariant.OLPS, OLMARVariant.CUMPROD):
            raise ValueError(
                "variant must be one of {OLMARVariant.OLPS, OLMARVariant.CUMPROD}"
            )
        self.window = int(window)
        self.variant = variant
        self._history: list[np.ndarray] = []

    def reset(self, d: int) -> None:
        super().reset(d)
        self._history = []

    def update_and_predict(self, x_t: np.ndarray) -> np.ndarray:
        x_t = np.asarray(x_t, dtype=float)
        self._history.append(x_t)
        T = len(self._history)
        W = self.window

        if T < W + 1:
            return self._history[-1].copy()

        if self.variant == OLMARVariant.OLPS:
            d = x_t.shape[0]
            tmp = np.ones(d, dtype=float)
            phi = np.zeros(d, dtype=float)
            for i in range(W):
                phi += 1.0 / np.maximum(tmp, CLIP_EPSILON)
                x_idx = T - i - 1  # x_{T - i}
                tmp = tmp * np.maximum(self._history[x_idx], CLIP_EPSILON)
            return phi * (1.0 / float(W))

        # variant == OLMARVariant.CUMPROD
        recent = np.stack(self._history[-W:], axis=0)[::-1, :]  # x_T, x_{T-1}, ...
        cumprods = np.cumprod(np.maximum(recent, CLIP_EPSILON), axis=0)
        inv = 1.0 / cumprods
        return inv.mean(axis=0)


class OLMAR2Predictor(BaseReversionPredictor):
    """
    OLMAR-2 reversion predictor φ_{t+1} = α·1 + (1−α) (φ_t ./ x_t),  φ_1 = 1.

    Parameters
    ----------
    alpha : float, default=0.5
        Alpha parameter for the predictor.
    """

    def __init__(self, alpha: float = 0.5):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = float(alpha)
        self._phi: Optional[np.ndarray] = None

    def reset(self, d: int) -> None:
        super().reset(d)
        self._phi = np.ones(d, dtype=float)

    def update_and_predict(self, x_t: np.ndarray) -> np.ndarray:
        x_t = np.asarray(x_t, dtype=float)
        if self._phi is None:
            self._phi = np.ones_like(x_t)
        self._phi = self.alpha * np.ones_like(x_t) + (1.0 - self.alpha) * (
            self._phi / np.maximum(x_t, CLIP_EPSILON)
        )
        return self._phi.copy()


class FTLoser(OnlinePortfolioSelection):
    r"""
    Follow-the-Loser (FTL) estimator that unifies mean-reversion strategies.

    Parameters
    ----------
    strategy : {'olmar', 'pamr', 'cwmr'} or LoserStrategy, default='olmar'
        Mean-reversion family to employ. Enum members from
        :class:`~skfolio.optimization.online._mixins.LoserStrategy` are accepted.
    olmar_order : {1, 2}, default=1
        Selects OLMAR-1 (moving-average predictor) or OLMAR-2 (recursive
        predictor) when ``strategy='olmar'``.
    olmar_window : int, default=5
        Window length for the OLMAR-1 predictor (used when ``olmar_order=1``).
    olmar_variant : {'olps', 'cumprod'}, default='olps'
        OLMAR-1 variant. ``'olps'`` reproduces the original OLPS Matlab
        implementation; ``'cumprod'`` removes the leading unit term.
    olmar_alpha : float, default=0.5
        Exponential smoothing parameter of OLMAR-2 (used when ``olmar_order=2``).
    cwmr_eta : float, default=0.95
        Confidence level of CWMR. Must lie in the open interval (0.5, 1).
    cwmr_sigma0 : float, default=1.0
        Initial diagonal variance for CWMR.
    cwmr_min_var : float or None, default=1e-12
        Optional lower bound applied to the CWMR diagonal covariance.
    cwmr_max_var : float or None, default=None
        Optional upper bound applied to the CWMR diagonal covariance.
    cwmr_mean_lr : float, default=1.0
        Learning rate for the CWMR mean update when ``update_mode='md'``.
    cwmr_var_lr : float, default=1.0
        Learning rate for the CWMR variance update when ``update_mode='md'``.
    epsilon : float, default=2.0
        Margin parameter used by all strategies. It is the target prediction
        level for OLMAR, the tolerance for passive-aggressive PAMR, and the
        confidence threshold in CWMR.
    loss : {'hinge', 'squared_hinge', 'softplus'}, default='hinge'
        Surrogate loss used by mirror-descent updates (``update_mode='md'``) for
        OLMAR/PAMR strategies.
    beta : float, default=5.0
        Sharpness parameter of the ``'softplus'`` surrogate loss.
    update_mode : {'pa', 'md'}, default='pa'
        Update regime. ``'pa'`` reproduces the classical passive-aggressive
        closed forms; ``'md'`` activates mirror-descent/FTRL style updates. For
        CWMR, ``'md'`` performs the OCO formulation introduced in this module.
    learning_rate : float or callable, default=1.0
        Learning-rate schedule for mirror-descent updates of OLMAR/PAMR. The
        callable signature must be ``learning_rate(t: int) -> float``.
    apply_fees_to_phi : bool, default=True
        Whether to apply management fees to the OLMAR prediction vector before
        executing the passive-aggressive or mirror-descent step.
    mirror : {'euclidean', 'entropy'}, default='euclidean'
        Mirror map used when ``update_mode='md'`` and the strategy is OLMAR or
        PAMR.
    **kwargs : dict
        Additional convex constraints forwarded to
        :class:`~skfolio.optimization.online._base.OnlinePortfolioSelection`.

    References
    ----------
    .. [1] Li, B., & Hoi, S. C. H. (2013). *Online Portfolio Selection: A Survey*.
           ACM Computing Surveys.
    .. [2] Li, B., Zhao, P., Hoi, S. C. H., & Gopalkrishnan, V. (2012).
           Confidence Weighted Mean Reversion Strategy for Online Portfolio Selection.
           ACM Transactions on Intelligent Systems and Technology.
    """

    _parameter_constraints: dict = {
        "strategy": [StrOptions({m.value.lower() for m in LoserStrategy}), None],
        "olmar_order": [Interval(Integral, 1, 2, closed="both")],
        "olmar_window": [Interval(Integral, 1, None, closed="left")],
        "olmar_variant": [StrOptions({m.value.lower() for m in OLMARVariant})],
        "olmar_alpha": [Interval(Real, 0, 1, closed="both")],
        "cwmr_eta": [Interval(Real, 0.5000001, 1.0, closed="neither")],
        "cwmr_sigma0": [Interval(Real, 1e-14, None, closed="left")],
        "cwmr_min_var": [Interval(Real, 0.0, None, closed="left"), None],
        "cwmr_max_var": [Interval(Real, 0.0, None, closed="left"), None],
        "cwmr_mean_lr": [Interval(Real, 0.0, None, closed="neither")],
        "cwmr_var_lr": [Interval(Real, 0.0, None, closed="left")],
        "epsilon": [Interval(Real, 0, None, closed="left")],
        "loss": [StrOptions({"hinge", "squared_hinge", "softplus"})],
        "beta": [Interval(Real, 0, None, closed="neither")],
        "update_mode": [StrOptions({"pa", "md"})],
        "learning_rate": [Interval(Real, 0, None, closed="neither"), callable],
        "apply_fees_to_phi": ["boolean"],
        "mirror": [StrOptions({"euclidean", "entropy"})],
    }

    def __init__(
        self,
        *,
        strategy: str | LoserStrategy | None = "olmar",
        olmar_order: int = 1,
        olmar_window: int = 5,
        olmar_alpha: float = 0.5,
        olmar_variant: str = "olps",
        cwmr_eta: float = 0.95,
        cwmr_sigma0: float = 1.0,
        cwmr_min_var: float | None = 1e-12,
        cwmr_max_var: float | None = None,
        cwmr_mean_lr: float = 1.0,
        cwmr_var_lr: float = 1.0,
        epsilon: float = 2.0,
        loss: str = "hinge",
        beta: float = 5.0,
        update_mode: str = "pa",
        learning_rate: float | int | callable = 1.0,
        apply_fees_to_phi: bool = True,
        mirror: str = "euclidean",
        warm_start: bool = True,
        initial_weights: npt.ArrayLike | None = None,
        previous_weights: npt.ArrayLike | None = None,
        transaction_costs: Any = 0.0,
        management_fees: Any = 0.0,
        groups: Any | None = None,
        linear_constraints: Any | None = None,
        left_inequality: Any | None = None,
        right_inequality: Any | None = None,
        max_turnover: float | None = None,
        min_weights: Any | None = 0.0,
        max_weights: Any | None = 1.0,
        budget: float | None = 1.0,
        X_tracking: Any | None = None,
        tracking_error_benchmark: Any | None = None,
        max_tracking_error: float | None = None,
        covariance: Any | None = None,
        variance_bound: float | None = None,
        portfolio_params: dict | None = None,
    ) -> None:
        super().__init__(
            warm_start=warm_start,
            initial_weights=initial_weights,
            previous_weights=previous_weights,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            max_turnover=max_turnover,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            X_tracking=X_tracking,
            tracking_error_benchmark=tracking_error_benchmark,
            max_tracking_error=max_tracking_error,
            covariance=covariance,
            variance_bound=variance_bound,
            portfolio_params=portfolio_params,
        )

        self.strategy = self._normalise_strategy(strategy)
        self.olmar_order = int(olmar_order)
        self.olmar_window = int(olmar_window)
        self.olmar_variant = str(olmar_variant).lower()
        self.olmar_alpha = float(olmar_alpha)
        self.cwmr_eta = float(cwmr_eta)
        self.cwmr_sigma0 = float(cwmr_sigma0)
        self.cwmr_min_var = cwmr_min_var
        self.cwmr_max_var = cwmr_max_var
        self.cwmr_mean_lr = float(cwmr_mean_lr)
        self.cwmr_var_lr = float(cwmr_var_lr)
        self.epsilon = float(epsilon)
        self.loss = loss
        self.beta = float(beta)
        self.update_mode = update_mode
        self.learning_rate = learning_rate
        self.apply_fees_to_phi = bool(apply_fees_to_phi)
        self.mirror = mirror

        self._predictor: BaseReversionPredictor | None = None
        self._surrogate: Surrogate | None = None
        self._engine: FTRL | None = None
        # status tracking variables
        self._t: int = 0
        self._last_trade_weights_: np.ndarray | None = None

        self._cwmr_mu: np.ndarray | None = None
        self._cwmr_Sdiag: np.ndarray | None = None
        self._cwmr_quantile: float | None = None

    def _normalise_strategy(
        self, strategy: str | LoserStrategy | None
    ) -> str:
        if isinstance(strategy, LoserStrategy):
            return strategy.value.lower()
        if strategy is None:
            return "olmar"
        return str(strategy).lower()

    def _validate_strategy_combinations(self) -> None:
        if self.strategy not in {"olmar", "pamr", "cwmr"}:
            raise ValueError("strategy must be one of {'olmar', 'pamr', 'cwmr'}.")

        if self.strategy == "olmar":
            if self.olmar_order not in (1, 2):
                raise ValueError("olmar_order must be 1 or 2 when strategy='olmar'.")
            if self.olmar_order == 1 and self.olmar_window < 1:
                raise ValueError("olmar_window must be >= 1 when olmar_order=1.")
            valid_variants = {m.value.lower() for m in OLMARVariant}
            if self.olmar_variant not in valid_variants:
                raise ValueError("olmar_variant must be 'olps' or 'cumprod'.")
            if not 0.0 <= self.olmar_alpha <= 1.0:
                raise ValueError("olmar_alpha must lie in [0, 1].")

        if self.strategy == "cwmr":
            if not 0.5 < self.cwmr_eta < 1.0:
                raise ValueError("cwmr_eta must belong to the open interval (0.5, 1).")
            if self.cwmr_sigma0 <= 0.0:
                raise ValueError("cwmr_sigma0 must be strictly positive.")
            if self.cwmr_min_var is not None and self.cwmr_min_var < 0.0:
                raise ValueError("cwmr_min_var cannot be negative.")
            if self.cwmr_max_var is not None and self.cwmr_max_var < 0.0:
                raise ValueError("cwmr_max_var cannot be negative.")
            if (
                self.cwmr_min_var is not None
                and self.cwmr_max_var is not None
                and self.cwmr_max_var < self.cwmr_min_var
            ):
                raise ValueError("cwmr_max_var cannot be smaller than cwmr_min_var.")
            if self.cwmr_mean_lr <= 0.0:
                raise ValueError("cwmr_mean_lr must be strictly positive.")
            if self.cwmr_var_lr < 0.0:
                raise ValueError("cwmr_var_lr cannot be negative.")

    def _clip_cwmr_variances(self, diag: np.ndarray) -> np.ndarray:
        """Clip CWMR diagonal variances to the configured bounds."""

        out = np.array(diag, dtype=float, copy=True)
        if self.cwmr_min_var is not None:
            out = np.maximum(out, float(self.cwmr_min_var))
        if self.cwmr_max_var is not None:
            out = np.minimum(out, float(self.cwmr_max_var))
        return np.maximum(out, 1e-18)

    def _ensure_components(self, d: int) -> None:
        if self._projector is None:
            self._initialize_projector()

        if self.strategy == "olmar" and self.olmar_order == 1:
            if self._predictor is None:
                self._predictor = OLMAR1Predictor(
                    window=self.olmar_window, variant=self.olmar_variant
                )
                self._predictor.reset(d)
            elif getattr(self._predictor, "_d", None) != d:
                self._predictor.reset(d)
        elif self.strategy == "olmar" and self.olmar_order == 2:
            if self._predictor is None:
                self._predictor = OLMAR2Predictor(alpha=self.olmar_alpha)
                self._predictor.reset(d)
            elif getattr(self._predictor, "_d", None) != d:
                self._predictor.reset(d)

        if self.strategy == "olmar" and self._surrogate is None:
            if self.loss == "hinge":
                self._surrogate = HingeSurrogate(self.epsilon)
            elif self.loss == "squared_hinge":
                self._surrogate = SquaredHingeSurrogate(self.epsilon)
            elif self.loss == "softplus":
                self._surrogate = SoftplusSurrogate(self.epsilon, self.beta)
            else:
                raise ValueError("Unknown surrogate loss.")

        if self.update_mode == "md" and (
            self.strategy == "pamr"
            or (self.strategy == "olmar" and self.olmar_order in {1, 2})
        ):
            if self._engine is None:
                if self.mirror == "euclidean":
                    mm: BaseMirrorMap = EuclideanMirrorMap()
                elif self.mirror == "entropy":
                    mm = EntropyMirrorMap()
                else:
                    raise ValueError("Unknown mirror map.")
                self._engine = FTRL(
                    mirror_map=mm,
                    projector=self._projector,
                    eta=self.learning_rate,
                    predictor=LastGradPredictor(),
                    mode="omd",
                )

        if self.strategy == "cwmr":
            if self._cwmr_quantile is None:
                self._cwmr_quantile = float(norm.ppf(self.cwmr_eta))
            if not self._weights_initialized:
                self._initialize_weights(d)
            if self._cwmr_mu is None:
                self._cwmr_mu = self.weights_.copy()
            if self._cwmr_Sdiag is None:
                self._cwmr_Sdiag = np.full(d, self.cwmr_sigma0, dtype=float)
            self._cwmr_Sdiag = self._clip_cwmr_variances(self._cwmr_Sdiag)

    def _should_update_today(self) -> bool:
        if self.strategy == "olmar" and self.olmar_order == 1:
            return self._t >= 1
        return True

    @staticmethod
    def _stable_y(a: float, s: float) -> float:
        # y = 0.5(-a + sqrt(a^2 + 4s)) = 2s / (sqrt(a^2 + 4s) + a)
        return (2.0 * s) / (np.sqrt(a * a + 4.0 * s) + a)

    def _cwmr_solve_tau(self, m0: float, s: float, eps: float, phi: float) -> float:
        if s <= 1e-18:
            return 0.0

        def f(tau: float) -> float:
            a = tau * phi * s
            y = self._stable_y(a, s)
            return (m0 - tau * s + phi * y) - eps

        f0 = f(0.0)
        if f0 <= 0.0:
            return 0.0

        tau_hi = max((m0 - eps) / max(s, 1e-18), 1.0)
        val = f(tau_hi)
        it = 0
        while val > 0.0 and it < 60:
            tau_hi *= 2.0
            val = f(tau_hi)
            it += 1
            if tau_hi > 1e12:
                break
        tau_lo = 0.0
        for _ in range(60):
            mid = 0.5 * (tau_lo + tau_hi)
            v = f(mid)
            if v <= 0.0:
                tau_hi = mid
            else:
                tau_lo = mid
        return tau_hi

    def _cwmr_pa_distribution_update(
        self, mu: np.ndarray, diag: np.ndarray, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi = float(self._cwmr_quantile)
        s = float(np.dot(diag * x, x))
        m0 = float(np.dot(mu, x))
        if m0 + phi * np.sqrt(max(s, 0.0)) <= self.epsilon + 1e-18:
            return mu.copy(), diag.copy()

        tau = self._cwmr_solve_tau(m0, s, self.epsilon, phi)
        sigma_x = diag * x
        mu_new = mu - tau * sigma_x

        a = tau * phi * s
        y = self._stable_y(a, s)
        denom = s * (y + a)
        if denom <= 0.0:
            coeff = 0.0
        else:
            coeff = a / denom
        diag_new = diag - coeff * (sigma_x * sigma_x)
        return mu_new, self._clip_cwmr_variances(diag_new)

    def _cwmr_pa_step(self, trade_w: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        mu = self._cwmr_mu
        diag = self._cwmr_Sdiag
        assert mu is not None and diag is not None
        mu_new, diag_new = self._cwmr_pa_distribution_update(mu, diag, x_t)
        if isinstance(self._projector, AutoProjector):
            self._projector.config.previous_weights = trade_w
        w_next = self._projector.project(mu_new)
        self._cwmr_mu = w_next.copy()
        self._cwmr_Sdiag = diag_new.copy()
        return w_next

    def _cwmr_oco_step(self, trade_w: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        mu = self._cwmr_mu
        diag = self._cwmr_Sdiag
        assert mu is not None and diag is not None

        phi = float(self._cwmr_quantile)
        s = float(np.dot(diag * x_t, x_t))
        sqrt_s = np.sqrt(max(s, 0.0))
        margin = float(np.dot(mu, x_t))
        violation = margin + phi * sqrt_s - self.epsilon
        if violation <= 0.0:
            self._cwmr_mu = trade_w.copy()
            self._cwmr_Sdiag = self._clip_cwmr_variances(diag)
            return trade_w.copy()

        mean_lr = self.cwmr_mean_lr
        var_lr = self.cwmr_var_lr
        sigma_x = diag * x_t
        mu_candidate = mu - mean_lr * sigma_x
        if isinstance(self._projector, AutoProjector):
            self._projector.config.previous_weights = trade_w
        w_next = self._projector.project(mu_candidate)
        self._cwmr_mu = w_next.copy()

        if var_lr > 0.0 and sqrt_s > 0.0:
            grad_sigma = (phi / (2.0 * max(sqrt_s, 1e-18))) * (x_t**2)
            log_diag = np.log(np.maximum(diag, 1e-18))
            log_diag_new = log_diag - var_lr * grad_sigma
            diag_new = np.exp(log_diag_new)
        else:
            diag_new = diag
        self._cwmr_Sdiag = self._clip_cwmr_variances(diag_new)
        return w_next

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> FTLoser:
        self._validate_params()
        self._validate_strategy_combinations()

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        first_call = not hasattr(self, "n_features_in_")
        X = validate_data(
            self, X=X_arr, y=None, reset=first_call, dtype=float, ensure_2d=True
        )
        if sample_weight is not None:
            _ = _check_sample_weight(sample_weight, X)

        x_t = np.asarray(net_to_relatives(X).squeeze(), dtype=float)
        d = int(x_t.shape[0])

        if not self._is_initialized:
            self.n_features_in_ = d
        if not self._weights_initialized or not self.warm_start:
            self._initialize_weights(d)
        self._ensure_components(d)

        trade_w = self.weights_.copy()
        self._last_trade_weights_ = trade_w

        is_olmar = self.strategy == "olmar"

        if is_olmar:
            assert self._predictor is not None
            phi_raw = self._predictor.update_and_predict(x_t)
            if self.apply_fees_to_phi:
                fees = self._clean_input(
                    self.management_fees, d, 0.0, "management_fees"
                )
                if np.isscalar(fees):
                    phi_eff = np.maximum(phi_raw * (1.0 - float(fees)), CLIP_EPSILON)
                else:
                    phi_eff = np.maximum(
                        phi_raw * (1.0 - np.asarray(fees, dtype=float)), CLIP_EPSILON
                    )
            else:
                phi_eff = np.maximum(phi_raw, CLIP_EPSILON)
        else:
            phi_eff = np.maximum(x_t, CLIP_EPSILON)

        next_w = trade_w.copy()
        if self._should_update_today():
            if self.update_mode == "pa":
                if is_olmar:
                    margin = float(phi_eff @ trade_w)
                    ell = max(0.0, self.epsilon - margin)
                    if ell > 0.0:
                        c = phi_eff - np.mean(phi_eff)
                        denom = float(np.dot(c, c))
                        if denom > 0.0:
                            lam = ell / denom
                            if isinstance(self._projector, AutoProjector):
                                self._projector.config.previous_weights = trade_w
                            next_w = self._projector.project(trade_w + lam * c)
                elif self.strategy == "pamr":
                    margin = float(phi_eff @ trade_w)
                    ell = max(0.0, margin - self.epsilon)
                    if ell > 0.0:
                        c = phi_eff - np.mean(phi_eff)
                        denom = float(np.dot(c, c))
                        if denom > 0.0:
                            tau = ell / denom
                            if isinstance(self._projector, AutoProjector):
                                self._projector.config.previous_weights = trade_w
                            next_w = self._projector.project(trade_w - tau * c)
                else:  # CWMR
                    next_w = self._cwmr_pa_step(trade_w, x_t)
            elif self.update_mode == "md":
                if self.strategy == "cwmr":
                    next_w = self._cwmr_oco_step(trade_w, x_t)
                else:
                    assert self._engine is not None
                    if is_olmar:
                        assert self._surrogate is not None
                        g = self._surrogate.grad(trade_w, phi_eff)
                    else:  # PAMR gradient
                        margin = float(phi_eff @ trade_w)
                        g = (
                            phi_eff
                            if (margin - self.epsilon) > 0.0
                            else np.zeros_like(phi_eff)
                        )
                    if self.transaction_costs and self.previous_weights is not None:
                        prev = np.asarray(self.previous_weights, dtype=float)
                        if prev.shape == trade_w.shape:
                            tc = self._clean_input(
                                self.transaction_costs, d, 0.0, "transaction_costs"
                            )
                            tc_arr = np.asarray(tc, dtype=float)
                            g = g + tc_arr * np.sign(trade_w - prev)
                    if isinstance(self._engine.map, EuclideanMirrorMap):
                        g = g - np.mean(g)
                    if isinstance(self._projector, AutoProjector):
                        self._projector.config.previous_weights = trade_w
                    next_w = self._engine.step(g)
            else:
                raise ValueError("update_mode must be 'pa' or 'md'.")

        self.weights_ = next_w
        self.previous_weights = trade_w.copy()

        final_return = float(np.dot(trade_w, np.maximum(x_t, CLIP_EPSILON)))
        self.loss_ = -np.log(max(final_return, CLIP_EPSILON))

        self._t += 1
        return self

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> "FTLoser":
        if not self.warm_start:
            self._weights_initialized = False
            self._is_initialized = False
            self._projector = None
            self._predictor = None
            self._surrogate = None
            self._engine = None
            self._cwmr_mu = None
            self._cwmr_Sdiag = None
            self._cwmr_quantile = None
            self._t = 0

        trade_list: list[np.ndarray] = []
        X_arr = np.asarray(X, dtype=float)
        for t in range(X_arr.shape[0]):
            self.partial_fit(X_arr[t][None, :], y, sample_weight=None, **fit_params)
            assert self._last_trade_weights_ is not None
            trade_list.append(self._last_trade_weights_.copy())

        self.all_weights_ = np.vstack(trade_list)
        return self
