"""
Mean-reversion portfolio selection strategies (refactored version).

Unified interface for OLMAR, PAMR, and CWMR strategies with both
passive-aggressive and mirror-descent update modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar, Protocol

import numpy as np
import numpy.typing as npt
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import _check_sample_weight, validate_data

from skfolio.optimization.online._base import OnlinePortfolioSelection
from skfolio.optimization.online._ftrl import LastGradPredictor, _FTRLEngine
from skfolio.optimization.online._mirror_maps import (
    BaseMirrorMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
)
from skfolio.optimization.online._mixins import (
    MeanReversionStrategy,
    OLMARVariant,
    UpdateMode,
)
from skfolio.optimization.online._strategies import (
    BaseStrategy,
    CWMRStrategy,
    OLMARStrategy,
    PAMRStrategy,
)
from skfolio.optimization.online._utils import CLIP_EPSILON, net_to_relatives

# ============================================================================
# Surrogate Loss Functions
# ============================================================================


class SurrogateLoss(Protocol):
    def value(self, w: np.ndarray, phi: np.ndarray) -> float: ...
    def grad(self, w: np.ndarray, phi: np.ndarray) -> np.ndarray: ...


@dataclass
class HingeSurrogateLoss:
    r"""Hinge surrogate loss: :math:`L(w) = \max(0, \epsilon - \phi^T w)`."""

    epsilon: float

    def value(self, w, phi):
        return max(0.0, self.epsilon - float(phi @ w))

    def grad(self, w, phi):
        return -phi if (self.epsilon - float(phi @ w) > 0.0) else np.zeros_like(phi)


@dataclass
class SquaredHingeSurrogateLoss:
    r"""Squared hinge loss: :math:`L(w) = (\max(0, \epsilon - \phi^T w))^2`."""

    epsilon: float

    def value(self, w, phi):
        m = self.epsilon - float(phi @ w)
        return (m * m) if m > 0.0 else 0.0

    def grad(self, w, phi):
        m = self.epsilon - float(phi @ w)
        return (-2.0 * m) * phi if m > 0.0 else np.zeros_like(phi)


@dataclass
class SoftplusSurrogateLoss:
    r"""Softplus loss: :math:`L(w) = (1/\beta) \log(1 + \exp(\beta (\epsilon - \phi^T w)))`."""

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


# ============================================================================
# Reversion Predictors
# ============================================================================


class BaseReversionPredictor:
    def reset(self, d: int) -> None:
        self._d = d

    def update_and_predict(self, x_t: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OLMAR1Predictor(BaseReversionPredictor):
    """OLMAR-1 moving-average reversion predictor."""

    def __init__(self, window: int = 5, variant: OLMARVariant = OLMARVariant.OLPS):
        if window < 1:
            raise ValueError("window must be >= 1.")
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

        match self.variant:
            case OLMARVariant.OLPS:
                d = x_t.shape[0]
                tmp = np.ones(d, dtype=float)
                phi = np.zeros(d, dtype=float)
                for i in range(W):
                    phi += 1.0 / np.maximum(tmp, CLIP_EPSILON)
                    x_idx = T - i - 1
                    tmp = tmp * np.maximum(self._history[x_idx], CLIP_EPSILON)
                return phi * (1.0 / float(W))
            case OLMARVariant.CUMPROD:
                recent = np.stack(self._history[-W:], axis=0)[::-1, :]
                cumprods = np.cumprod(np.maximum(recent, CLIP_EPSILON), axis=0)
                inv = 1.0 / cumprods
                return inv.mean(axis=0)


class OLMAR2Predictor(BaseReversionPredictor):
    """OLMAR-2 exponential-smoothing reversion predictor."""

    def __init__(self, alpha: float = 0.5):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = float(alpha)
        self._phi: np.ndarray | None = None

    def reset(self, d: int) -> None:
        super().reset(d)
        self._phi = np.ones(d, dtype=float)

    def update_and_predict(self, x_t: np.ndarray) -> np.ndarray:
        if self._phi is None:
            self._phi = np.ones_like(x_t)
        self._phi = self.alpha * np.ones_like(x_t) + (1.0 - self.alpha) * (
            self._phi / np.maximum(x_t, CLIP_EPSILON)
        )
        return self._phi.copy()


# ============================================================================
# Main MeanReversion Estimator
# ============================================================================


class MeanReversion(OnlinePortfolioSelection):
    r"""
    Mean-reversion estimator (OLMAR, PAMR, CWMR).

    Parameters
    ----------
    strategy : {'olmar', 'pamr', 'cwmr'}, default='olmar'
        Mean-reversion family.
    olmar_order : {1, 2}, default=1
        OLMAR version (1=moving-average, 2=recursive).
    olmar_window : int, default=5
        OLMAR-1 window length.
    olmar_variant : {'olps', 'cumprod'}, default='olps'
        OLMAR-1 variant.
    olmar_alpha : float, default=0.5
        OLMAR-2 smoothing parameter.
    pamr_variant : {0, 1, 2}, default=0
        PAMR variant (0=basic, 1=capped, 2=soft-reg).
    pamr_C : float, default=500.0
        PAMR aggressiveness parameter.
    cwmr_eta : float, default=0.95
        CWMR confidence level (0.5, 1).
    cwmr_sigma0 : float, default=1.0
        CWMR initial variance.
    cwmr_min_var, cwmr_max_var : float or None
        CWMR variance bounds.
    cwmr_mean_lr, cwmr_var_lr : float, default=1.0
        CWMR learning rates (MD mode).
    epsilon : float, default=2.0
        Margin threshold.
    loss : {'hinge', 'squared_hinge', 'softplus'}, default='hinge'
        Surrogate loss (MD mode).
    beta : float, default=5.0
        Softplus temperature.
    update_mode : {'pa', 'md'}, default='pa'
        Update mode (PA=passive-aggressive, MD=mirror-descent).
    learning_rate : float or callable, default=1.0
        Learning rate (MD mode).
    apply_fees_to_phi : bool, default=True
        Apply management fees to predictor.
    mirror : {'euclidean', 'entropy'}, default='euclidean'
        Mirror map (MD mode).
    **kwargs
        Additional constraints.

    References
    ----------
    .. [1] Li, B., & Hoi, S. C. H. (2013). Online Portfolio Selection: A Survey.
    """

    _parameter_constraints: ClassVar[dict] = {
        "strategy": [
            StrOptions({m.value.lower() for m in MeanReversionStrategy}),
            None,
        ],
        "olmar_order": [Interval(Integral, 1, 2, closed="both")],
        "olmar_window": [Interval(Integral, 1, None, closed="left")],
        "olmar_variant": [StrOptions({m.value.lower() for m in OLMARVariant})],
        "olmar_alpha": [Interval(Real, 0, 1, closed="both")],
        "pamr_variant": [Interval(Integral, 0, 2, closed="both")],
        "pamr_C": [Interval(Real, 0, None, closed="neither")],
        "cwmr_eta": [Interval(Real, 0.5000001, 1.0, closed="neither")],
        "cwmr_sigma0": [Interval(Real, 1e-14, None, closed="left")],
        "cwmr_min_var": [Interval(Real, 0.0, None, closed="left"), None],
        "cwmr_max_var": [Interval(Real, 0.0, None, closed="left"), None],
        "cwmr_mean_lr": [Interval(Real, 0.0, None, closed="neither")],
        "cwmr_var_lr": [Interval(Real, 0.0, None, closed="left")],
        "epsilon": [Interval(Real, 0, None, closed="left")],
        "loss": [StrOptions({"hinge", "squared_hinge", "softplus"})],
        "beta": [Interval(Real, 0, None, closed="neither")],
        "update_mode": [StrOptions({m.value.lower() for m in UpdateMode})],
        "learning_rate": [Interval(Real, 0, None, closed="neither"), callable],
        "apply_fees_to_phi": ["boolean"],
        "mirror": [StrOptions({"euclidean", "entropy"})],
    }

    def __init__(
        self,
        *,
        strategy: str | MeanReversionStrategy | None = MeanReversionStrategy.OLMAR,
        olmar_order: int = 1,
        olmar_window: int = 5,
        olmar_alpha: float = 0.5,
        olmar_variant: str = "olps",
        pamr_variant: int = 0,
        pamr_C: float = 500.0,
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

        self.strategy = strategy
        self.olmar_order = olmar_order
        self.olmar_window = olmar_window
        self.olmar_variant = olmar_variant
        self.olmar_alpha = olmar_alpha
        self.pamr_variant = pamr_variant
        self.pamr_C = pamr_C
        self.cwmr_eta = cwmr_eta
        self.cwmr_sigma0 = cwmr_sigma0
        self.cwmr_min_var = cwmr_min_var
        self.cwmr_max_var = cwmr_max_var
        self.cwmr_mean_lr = cwmr_mean_lr
        self.cwmr_var_lr = cwmr_var_lr
        self.epsilon = epsilon
        self.loss = loss
        self.beta = beta
        self.update_mode = update_mode
        self.learning_rate = learning_rate
        self.apply_fees_to_phi = apply_fees_to_phi
        self.mirror = mirror

        # Internal state
        self._strategy_impl: BaseStrategy | None = None
        self._predictor: BaseReversionPredictor | None = None
        self._surrogate: SurrogateLoss | None = None
        self._engine: _FTRLEngine | None = None

    def _normalize_enums(self) -> None:
        """Normalize string parameters to enums."""
        if isinstance(self.strategy, str):
            try:
                self.strategy = MeanReversionStrategy(self.strategy.lower())
            except ValueError as e:
                raise ValueError(
                    f"strategy must be one of {{{', '.join([s.value for s in MeanReversionStrategy])}}}, "
                    f"got '{self.strategy}'"
                ) from e

        if isinstance(self.update_mode, str):
            try:
                self.update_mode = UpdateMode(self.update_mode.lower())
            except ValueError as e:
                raise ValueError(
                    f"update_mode must be 'pa' or 'md', got '{self.update_mode}'"
                ) from e

        if isinstance(self.olmar_variant, str):
            try:
                self.olmar_variant = OLMARVariant(self.olmar_variant.lower())
            except ValueError as e:
                raise ValueError(
                    f"olmar_variant must be 'olps' or 'cumprod', got '{self.olmar_variant}'"
                ) from e

    def _validate_strategy_params(self) -> None:
        """Validate strategy-specific parameters."""
        match self.strategy:
            case MeanReversionStrategy.OLMAR:
                if self.olmar_order == 1 and self.olmar_window < 1:
                    raise ValueError("olmar_window must be >= 1 when olmar_order=1.")
            case MeanReversionStrategy.CWMR:
                if not 0.5 < self.cwmr_eta < 1.0:
                    raise ValueError("cwmr_eta must be in (0.5, 1).")
                if (
                    self.cwmr_min_var is not None
                    and self.cwmr_max_var is not None
                    and self.cwmr_max_var < self.cwmr_min_var
                ):
                    raise ValueError(
                        "cwmr_max_var cannot be smaller than cwmr_min_var."
                    )

    def _initialize_components(self, d: int) -> None:
        """Initialize all strategy components."""
        if self._projector is None:
            self._initialize_projector()

        if not self._weights_initialized:
            self._initialize_weights(d)

        # Initialize predictor and surrogate
        match self.strategy:
            case MeanReversionStrategy.OLMAR:
                self._init_olmar_components(d)
            case MeanReversionStrategy.PAMR:
                self._init_pamr_components(d)
            case MeanReversionStrategy.CWMR:
                self._init_cwmr_components(d)

        # Initialize strategy implementation
        if self._strategy_impl is None:
            self._strategy_impl = self._create_strategy(d)

    def _init_olmar_components(self, d: int) -> None:
        """Initialize OLMAR-specific components."""
        if self._predictor is None:
            if self.olmar_order == 1:
                self._predictor = OLMAR1Predictor(
                    window=self.olmar_window, variant=self.olmar_variant
                )
            else:
                self._predictor = OLMAR2Predictor(alpha=self.olmar_alpha)
            self._predictor.reset(d)

        if self._surrogate is None:
            match self.loss:
                case "hinge":
                    self._surrogate = HingeSurrogateLoss(self.epsilon)
                case "squared_hinge":
                    self._surrogate = SquaredHingeSurrogateLoss(self.epsilon)
                case "softplus":
                    self._surrogate = SoftplusSurrogateLoss(self.epsilon, self.beta)

        if self.update_mode == UpdateMode.MD and self._engine is None:
            self._engine = self._create_engine()

    def _init_pamr_components(self, d: int) -> None:
        """Initialize PAMR-specific components."""
        if self.update_mode == UpdateMode.MD and self._engine is None:
            self._engine = self._create_engine()

    def _init_cwmr_components(self, d: int) -> None:
        """Initialize CWMR-specific components."""
        # CWMR doesn't need separate initialization
        pass

    def _create_engine(self) -> _FTRLEngine:
        """Create FTRL engine for MD mode."""
        match self.mirror:
            case "euclidean":
                mm: BaseMirrorMap = EuclideanMirrorMap()
            case "entropy":
                mm = EntropyMirrorMap()
            case _:
                raise ValueError(f"Unknown mirror: {self.mirror}")

        return _FTRLEngine(
            mirror_map=mm,
            projector=self._projector,
            eta=self.learning_rate,
            predictor=LastGradPredictor(),
            mode="omd",
        )

    def _create_strategy(self, d: int) -> BaseStrategy:
        """Factory method to create strategy implementation."""
        match self.strategy:
            case MeanReversionStrategy.OLMAR:
                strategy = OLMARStrategy(
                    predictor=self._predictor,
                    surrogate=self._surrogate,
                    engine=self._engine,
                    epsilon=self.epsilon,
                    olmar_order=self.olmar_order,
                )
            case MeanReversionStrategy.PAMR:
                strategy = PAMRStrategy(
                    engine=self._engine,
                    epsilon=self.epsilon,
                    pamr_variant=self.pamr_variant,
                    pamr_C=self.pamr_C,
                )
            case MeanReversionStrategy.CWMR:
                strategy = CWMRStrategy(
                    cwmr_eta=self.cwmr_eta,
                    cwmr_sigma0=self.cwmr_sigma0,
                    cwmr_min_var=self.cwmr_min_var,
                    cwmr_max_var=self.cwmr_max_var,
                    cwmr_mean_lr=self.cwmr_mean_lr,
                    cwmr_var_lr=self.cwmr_var_lr,
                    epsilon=self.epsilon,
                    initial_weights=self.weights_,
                )

        # Initialize the strategy
        strategy.reset(d)
        return strategy

    def _compute_phi_effective(self, x_t: np.ndarray, d: int) -> np.ndarray:
        """Compute effective reversion predictor (with optional fees)."""
        if self.strategy == MeanReversionStrategy.OLMAR:
            phi_raw = self._predictor.update_and_predict(x_t)
            if self.apply_fees_to_phi:
                fees = self._clean_input(
                    self.management_fees, d, 0.0, "management_fees"
                )
                if np.isscalar(fees):
                    return np.maximum(phi_raw * (1.0 - float(fees)), CLIP_EPSILON)
                return np.maximum(
                    phi_raw * (1.0 - np.asarray(fees, dtype=float)), CLIP_EPSILON
                )
            return np.maximum(phi_raw, CLIP_EPSILON)
        # PAMR and CWMR use x_t directly
        return np.maximum(x_t, CLIP_EPSILON)

    def _execute_strategy_update(
        self, trade_w: np.ndarray, x_t: np.ndarray, phi_eff: np.ndarray
    ) -> np.ndarray:
        """Execute the strategy update step."""
        if not self._strategy_impl.should_update(self._t):
            return trade_w.copy()

        return self._strategy_impl.execute_step(
            trade_w=trade_w,
            x_t=x_t,
            phi_eff=phi_eff,
            update_mode=self.update_mode,
            projector=self._projector,
        )

    def _compute_loss(self, trade_w: np.ndarray, x_t: np.ndarray) -> float:
        """Compute realized loss for the period."""
        final_return = float(np.dot(trade_w, np.maximum(x_t, CLIP_EPSILON)))
        return -np.log(max(final_return, CLIP_EPSILON))

    @property
    def _cwmr_Sdiag(self) -> np.ndarray | None:
        """Expose CWMR diagonal covariance for backward compatibility with tests."""
        if self.strategy == MeanReversionStrategy.CWMR and self._strategy_impl:
            return getattr(self._strategy_impl, "_Sdiag", None)
        return None

    @property
    def _cwmr_mu(self) -> np.ndarray | None:
        """Expose CWMR mean for backward compatibility with tests."""
        if self.strategy == MeanReversionStrategy.CWMR and self._strategy_impl:
            return getattr(self._strategy_impl, "_mu", None)
        return None

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> MeanReversion:
        # Validate parameters
        self._validate_params()
        self._normalize_enums()
        self._validate_strategy_params()

        # Validate and preprocess input
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        first_call = not hasattr(self, "n_features_in_")
        X = validate_data(
            self, X=X_arr, y=None, reset=first_call, dtype=float, ensure_2d=True
        )
        if sample_weight is not None:
            _ = _check_sample_weight(sample_weight, X)

        # Convert to relatives
        x_t = np.asarray(net_to_relatives(X).squeeze(), dtype=float)
        d = int(x_t.shape[0])

        # Initialize components
        if not self._is_initialized:
            self.n_features_in_ = d
        if not self._weights_initialized or not self.warm_start:
            self._initialize_weights(d)
        self._initialize_components(d)

        # Store current weights for trading
        trade_w = self.weights_.copy()
        self._last_trade_weights_ = trade_w

        # Compute effective predictor
        phi_eff = self._compute_phi_effective(x_t, d)

        # Execute strategy update
        next_w = self._execute_strategy_update(trade_w, x_t, phi_eff)

        # Update state
        self.weights_ = next_w
        self.previous_weights = trade_w.copy()
        self.loss_ = self._compute_loss(trade_w, x_t)
        self._t += 1

        return self

    def _reset_state_for_fit(self) -> None:
        """Reset internal state when warm_start=False."""
        super()._reset_state_for_fit()
        self._strategy_impl = None
        self._predictor = None
        self._surrogate = None
        self._engine = None
