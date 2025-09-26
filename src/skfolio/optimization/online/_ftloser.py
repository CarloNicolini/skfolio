from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Optional, Protocol

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from sklearn.utils._param_validation import Interval, StrOptions  # type: ignore
from sklearn.utils.validation import _check_sample_weight, validate_data

from skfolio.optimization.online._base import OnlinePortfolioSelection
from skfolio.optimization.online._ftrl import FTRL, LastGradPredictor
from skfolio.optimization.online._mirror_maps import (
    BaseMirrorMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
)
from skfolio.optimization.online._projection import AutoProjector
from skfolio.optimization.online._utils import CLIP_EPSILON, net_to_relatives
from skfolio.optimization.online._mixins import LoserFamily


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
    OLMAR-1 reversion predictor φ_{t+1} with explicit variant:
      - variant="olps":     average of [1, 1/x_T, 1/(x_T x_{T-1}), ..., W terms]
      - variant="cumprod":  average of [1/x_T, 1/(x_T x_{T-1}), ..., W terms] (no leading 1)

    Cold-start rule: if T < W+1, return last observed relative x_T (OLPS schedule).
    """

    def __init__(self, window: int = 5, variant: str = "olps"):
        if window < 1:
            raise ValueError("window must be >= 1.")
        if variant not in ("olps", "cumprod"):
            raise ValueError('variant must be one of {"olps","cumprod"}')
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

        if self.variant == "olps":
            d = x_t.shape[0]
            tmp = np.ones(d, dtype=float)
            phi = np.zeros(d, dtype=float)
            for i in range(W):
                phi += 1.0 / np.maximum(tmp, CLIP_EPSILON)
                x_idx = T - i - 1  # x_{T - i}
                tmp = tmp * np.maximum(self._history[x_idx], CLIP_EPSILON)
            return phi * (1.0 / float(W))

        # variant == "cumprod"
        recent = np.stack(self._history[-W:], axis=0)[::-1, :]  # x_T, x_{T-1}, ...
        cumprods = np.cumprod(np.maximum(recent, CLIP_EPSILON), axis=0)
        inv = 1.0 / cumprods
        return inv.mean(axis=0)


class OLMAR2Predictor(BaseReversionPredictor):
    """
    φ_{t+1} = α·1 + (1−α) (φ_t ./ x_t),  φ_1 = 1.
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
    strategy : {'olmar', 'olmar1', 'olmar2', 'pamr', 'cwmr'} or LoserFamily, default='olmar1'
        Strategy identifier. Passing ``'olmar'`` allows ``strategy_params['order']`` to
        choose between OLMAR-1 and OLMAR-2. Enum members from
        :class:`~skfolio.optimization.online._mixins.LoserFamily` are also accepted.
    strategy_params : dict, optional
        Strategy-specific hyperparameters. Recognised keys are:

        * ``olmar1`` – ``window`` (int ≥ 1) and ``variant`` in {``'olps'``, ``'cumprod'``}.
        * ``olmar2`` – ``alpha`` in [0, 1].
        * ``olmar`` – optional ``order`` in {1, 2} delegating to OLMAR-1/2.
        * ``cwmr`` – ``eta`` in (0.5, 1), ``sigma0`` > 0, ``min_var`` ≥ 0 or ``None``,
          ``max_var`` ≥ ``min_var`` or ``None``, ``mean_lr`` > 0, and ``var_lr`` ≥ 0.
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
        "strategy": [StrOptions({"olmar", "olmar1", "olmar2", "pamr", "cwmr"}), None],
        "strategy_params": [dict, None],
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
        strategy: str | LoserFamily | None = "olmar1",
        strategy_params: dict | None = None,
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

        self.strategy, self._strategy_params = self._canonicalize_strategy(
            strategy, strategy_params
        )
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

        self._t: int = 0
        self._last_trade_weights_: np.ndarray | None = None

        self._cwmr_mu: np.ndarray | None = None
        self._cwmr_Sdiag: np.ndarray | None = None
        self._cwmr_quantile: float | None = None

    def _canonicalize_strategy(
        self, strategy: str | LoserFamily | None, overrides: dict | None
    ) -> tuple[str, dict]:
        if isinstance(strategy, LoserFamily):
            strategy_str = strategy.value.lower()
        elif strategy is None:
            strategy_str = "olmar1"
        else:
            strategy_str = str(strategy).lower()

        params = dict(overrides or {})

        if strategy_str == "olmar":
            order = params.pop("order", 1)
            if not isinstance(order, Integral):
                raise ValueError(
                    "`strategy_params['order']` must be an integer when strategy='olmar'."
                )
            order = int(order)
            if order not in (1, 2):
                raise ValueError(
                    "`strategy_params['order']` must be 1 or 2 when strategy='olmar'."
                )
            strategy_str = f"olmar{order}"

        if strategy_str not in {"olmar1", "olmar2", "pamr", "cwmr"}:
            raise ValueError(
                "strategy must be one of {'olmar1', 'olmar2', 'pamr', 'cwmr', 'olmar'}."
            )

        defaults = self._default_strategy_params(strategy_str)
        defaults.update(params)
        self._validate_strategy_params(strategy_str, defaults)
        return strategy_str, defaults

    def _default_strategy_params(self, strategy: str) -> dict:
        if strategy == "olmar1":
            return {"window": 5, "variant": "olps"}
        if strategy == "olmar2":
            return {"alpha": 0.5}
        if strategy == "pamr":
            return {}
        if strategy == "cwmr":
            return {
                "eta": 0.95,
                "sigma0": 1.0,
                "min_var": 1e-12,
                "max_var": None,
                "mean_lr": 1.0,
                "var_lr": 1.0,
            }
        raise ValueError(f"Unsupported strategy '{strategy}'.")

    def _validate_strategy_params(self, strategy: str, params: dict) -> None:
        if strategy == "olmar1":
            window = params.get("window", 5)
            if not isinstance(window, Integral) or window < 1:
                raise ValueError("OLMAR-1 requires `window` to be an integer ≥ 1.")
            params["window"] = int(window)
            variant = str(params.get("variant", "olps")).lower()
            if variant not in {"olps", "cumprod"}:
                raise ValueError("OLMAR-1 variant must be 'olps' or 'cumprod'.")
            params["variant"] = variant
        elif strategy == "olmar2":
            alpha = float(params.get("alpha", 0.5))
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("OLMAR-2 `alpha` must lie in [0, 1].")
            params["alpha"] = alpha
        elif strategy == "cwmr":
            eta = float(params.get("eta", 0.95))
            if not 0.5 < eta < 1.0:
                raise ValueError("CWMR `eta` must belong to the open interval (0.5, 1).")
            sigma0 = float(params.get("sigma0", 1.0))
            if sigma0 <= 0.0:
                raise ValueError("CWMR `sigma0` must be strictly positive.")
            min_var = params.get("min_var", 1e-12)
            if min_var is not None:
                min_var = float(min_var)
                if min_var < 0.0:
                    raise ValueError("CWMR `min_var` cannot be negative.")
            max_var = params.get("max_var", None)
            if max_var is not None:
                max_var = float(max_var)
                if max_var < 0.0:
                    raise ValueError("CWMR `max_var` cannot be negative.")
                if min_var is not None and max_var < min_var:
                    raise ValueError("CWMR `max_var` cannot be smaller than `min_var`.")
            mean_lr = float(params.get("mean_lr", 1.0))
            if mean_lr <= 0.0:
                raise ValueError("CWMR `mean_lr` must be strictly positive.")
            var_lr = float(params.get("var_lr", 1.0))
            if var_lr < 0.0:
                raise ValueError("CWMR `var_lr` cannot be negative.")
            params["eta"] = eta
            params["sigma0"] = sigma0
            params["min_var"] = min_var
            params["max_var"] = max_var
            params["mean_lr"] = mean_lr
            params["var_lr"] = var_lr
            params["quantile"] = float(norm.ppf(eta))

    def _clip_cwmr_variances(self, diag: np.ndarray) -> np.ndarray:
        """Clip CWMR diagonal variances to the configured bounds."""

        out = np.array(diag, dtype=float, copy=True)
        min_var = self._strategy_params.get("min_var")
        max_var = self._strategy_params.get("max_var")
        if min_var is not None:
            out = np.maximum(out, float(min_var))
        if max_var is not None:
            out = np.minimum(out, float(max_var))
        return np.maximum(out, 1e-18)

    def _ensure_components(self, d: int) -> None:
        if self._projector is None:
            self._initialize_projector()

        if self.strategy == "olmar1":
            if self._predictor is None:
                params = self._strategy_params
                self._predictor = OLMAR1Predictor(
                    window=params["window"], variant=params["variant"]
                )
                self._predictor.reset(d)
            elif getattr(self._predictor, "_d", None) != d:
                self._predictor.reset(d)
        elif self.strategy == "olmar2":
            if self._predictor is None:
                params = self._strategy_params
                self._predictor = OLMAR2Predictor(alpha=params["alpha"])
                self._predictor.reset(d)
            elif getattr(self._predictor, "_d", None) != d:
                self._predictor.reset(d)

        if self.strategy in {"olmar1", "olmar2"} and self._surrogate is None:
            if self.loss == "hinge":
                self._surrogate = HingeSurrogate(self.epsilon)
            elif self.loss == "squared_hinge":
                self._surrogate = SquaredHingeSurrogate(self.epsilon)
            elif self.loss == "softplus":
                self._surrogate = SoftplusSurrogate(self.epsilon, self.beta)
            else:
                raise ValueError("Unknown surrogate loss.")

        if self.update_mode == "md" and self.strategy in {"olmar1", "olmar2", "pamr"}:
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
                self._cwmr_quantile = float(self._strategy_params["quantile"])
            if not self._weights_initialized:
                self._initialize_weights(d)
            if self._cwmr_mu is None:
                self._cwmr_mu = self.weights_.copy()
            if self._cwmr_Sdiag is None:
                sigma0 = float(self._strategy_params["sigma0"])
                self._cwmr_Sdiag = np.full(d, sigma0, dtype=float)
            self._cwmr_Sdiag = self._clip_cwmr_variances(self._cwmr_Sdiag)

    def _should_update_today(self) -> bool:
        return self.strategy != "olmar1" or self._t >= 1

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

        mean_lr = float(self._strategy_params["mean_lr"])
        var_lr = float(self._strategy_params["var_lr"])
        sigma_x = diag * x_t
        mu_candidate = mu - mean_lr * sigma_x
        if isinstance(self._projector, AutoProjector):
            self._projector.config.previous_weights = trade_w
        w_next = self._projector.project(mu_candidate)
        self._cwmr_mu = w_next.copy()

        if var_lr > 0.0 and sqrt_s > 0.0:
            grad_sigma = (phi / (2.0 * max(sqrt_s, 1e-18))) * (x_t ** 2)
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
    ) -> "FTLoser":
        first_call = not hasattr(self, "n_features_in_")
        X = validate_data(
            self, X=X, y=None, reset=first_call, dtype=float, ensure_2d=True
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

        if self.strategy in {"olmar1", "olmar2"}:
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
                if self.strategy in {"olmar1", "olmar2"}:
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
                    if self.strategy in {"olmar1", "olmar2"}:
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
