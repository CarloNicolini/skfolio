"""
Mean-reversion portfolio selection strategies.
Unified interface for OLMAR, PAMR, and CWMR strategies with both passive-aggressive and mirror-descent update modes.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import validate_data

from skfolio.optimization.online._base import OnlinePortfolioSelection
from skfolio.optimization.online._ftrl import _FTRLEngine
from skfolio.optimization.online._mirror_maps import (
    BaseMirrorMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
)
from skfolio.optimization.online._mixins import (
    MeanReversionStrategy,
    OLMARPredictor,
    PAMRVariant,
    UpdateMode,
)
from skfolio.optimization.online._prediction import (
    BaseReversionPredictor,
    LastGradPredictor,
    OLMAR1Predictor,
    OLMAR2Predictor,
)
from skfolio.optimization.online._strategies import (
    BaseStrategy,
    CWMRStrategy,
    OLMARStrategy,
    PAMRStrategy,
)
from skfolio.optimization.online._surrogates import (
    HingeLoss,
    SoftplusLoss,
    SquaredHingeLoss,
    SurrogateLoss,
    SurrogateLossType,
)
from skfolio.optimization.online._utils import CLIP_EPSILON, net_to_relatives


class MeanReversion(OnlinePortfolioSelection):
    r"""
    Mean-reversion estimator (OLMAR, PAMR, CWMR).

    Parameters
    ----------
    strategy : {'olmar', 'pamr', 'cwmr'}, default='olmar'
        Mean-reversion family.
    # OLMAR parameters
    olmar_predictor : {"sma", "ewma"}
        OLMAR reversion predictor (sma=for simple moving-average, ewma for exponentially weighted).
    olmar_window : int, default=5
        OLMAR window size (applies only for SMA version).
    olmar_alpha : float, default=0.5
        OLMAR smoothing parameter (applies only for EWMA).
    # PAMR parameters
    pamr_variant : {"simple", "slack_linear", "slack_quadratic"}, default="simple"
        PAMR variant "simple", "slack_linear", or "slack_quadratic".
        These variants are the 0, 1 and 2 versions of PAMR in Li & Hoi book (2012).
    pamr_C : float, default=500.0
        PAMR aggressiveness parameter.
    # CWMR parameters
    cwmr_eta : float, default=0.95
        CWMR confidence level (0.5, 1).
    cwmr_sigma0 : float, default=1.0
        CWMR initial variance.
    cwmr_min_var, cwmr_max_var : float or None
        CWMR variance bounds.
    cwmr_mean_lr, cwmr_var_lr : float, default=1.0
        CWMR learning rates (MD mode).
    # Mirror Descent (MD) and Passive-Aggressive (PA) parameters
    epsilon : float, default=2.0
        Margin threshold.
    loss : {'hinge', 'squared_hinge', 'softplus'}, default='hinge'
        Surrogate loss (MD mode).
    beta : float, default=5.0
        Softplus temperature.
    update_mode : {'pa', 'md'}, default='pa'
        Update mode (PA=passive-aggressive, MD=mirror-descent).
        The passive aggressive update mode reflects the original algorithms.
        The MD uses Online Mirror Descent and supports more flexibility (e.g., learning rate, mirror map).
    learning_rate : float or callable, default=1.0
        Learning rate (MD mode).
    mirror : {'euclidean', 'entropy'}, default='euclidean'
        Mirror map (MD mode).
    **kwargs
        Additional constraints.

    References
    ----------
    .. [1] Li, B., & Hoi, S. C. H. (2013). Online Portfolio Selection: A Survey.
    """

    _parameter_constraints: ClassVar[dict] = {
        # Strategy to replicate mean reversion algorithm
        "strategy": [
            StrOptions({m.value.lower() for m in MeanReversionStrategy}),
            None,
        ],
        # OLMAR parameters
        "olmar_predictor": [StrOptions({m.value.lower() for m in OLMARPredictor})],
        "olmar_window": [Interval(Integral, 1, None, closed="left")],
        "olmar_alpha": [Interval(Real, 0, 1, closed="both")],
        # PAMR parameters
        "pamr_variant": [StrOptions({m.value.lower() for m in PAMRVariant})],
        "pamr_C": [Interval(Real, 0, None, closed="neither")],
        # CWMR parameters
        "cwmr_eta": [Interval(Real, 0.5000001, 1.0, closed="neither")],
        "cwmr_sigma0": [Interval(Real, 1e-14, None, closed="left")],
        "cwmr_min_var": [Interval(Real, 0.0, None, closed="left"), None],
        "cwmr_max_var": [Interval(Real, 0.0, None, closed="left"), None],
        "cwmr_mean_lr": [Interval(Real, 0.0, None, closed="neither")],
        "cwmr_var_lr": [Interval(Real, 0.0, None, closed="left")],
        # PA/MD parameters
        "epsilon": [Interval(Real, 0, None, closed="left")],
        "loss": [StrOptions({"hinge", "squared_hinge", "softplus"})],
        "beta": [Interval(Real, 0, None, closed="neither")],
        "update_mode": [StrOptions({m.value.lower() for m in UpdateMode})],
        "learning_rate": [Interval(Real, 0, None, closed="neither"), callable],
        "mirror": [StrOptions({"euclidean", "entropy"})],
    }

    def __init__(
        self,
        *,
        strategy: str | MeanReversionStrategy,
        olmar_predictor: OLMARPredictor | str = OLMARPredictor.SMA,
        olmar_window: int = 5,
        olmar_alpha: float = 0.5,
        pamr_variant: PAMRVariant = "simple",
        pamr_C: float = 500.0,
        cwmr_eta: float = 0.95,
        cwmr_sigma0: float = 1.0,
        cwmr_min_var: float | None = 1e-12,
        cwmr_max_var: float | None = None,
        cwmr_mean_lr: float = 1.0,
        cwmr_var_lr: float = 1.0,
        epsilon: float = 2.0,
        loss: str | SurrogateLoss = SurrogateLossType.HINGE,
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
        self.olmar_predictor = olmar_predictor
        self.olmar_window = olmar_window
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
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        # Initialize strategy implementation
        if self._strategy_impl is None:
            self._strategy_impl = self._create_strategy(d)

    def _init_olmar_components(self, d: int) -> None:
        """Initialize OLMAR-specific components."""
        if self._predictor is None:
            if self.olmar_predictor == 1:
                self._predictor = OLMAR1Predictor(window=self.olmar_window)
            else:
                self._predictor = OLMAR2Predictor(alpha=self.olmar_alpha)
            self._predictor.reset(d)

        if self._surrogate is None:
            match self.loss:
                case SurrogateLossType.HINGE:
                    self._surrogate = HingeLoss(self.epsilon)
                case SurrogateLossType.SQUARED_HINGE:
                    self._surrogate = SquaredHingeLoss(self.epsilon)
                case SurrogateLossType.SOFTPLUS:
                    self._surrogate = SoftplusLoss(self.epsilon, self.beta)

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
                    olmar_order=self.olmar_predictor,
                )
            case MeanReversionStrategy.PAMR:
                strategy = PAMRStrategy(
                    surrogate=self._surrogate,
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

    def _compute_predictor(self, x_t: np.ndarray, d: int) -> np.ndarray:
        """Compute effective reversion predictor (only for OLMAR)."""
        if self.strategy == MeanReversionStrategy.OLMAR:
            return self._predictor.update_and_predict(x_t)
        # PAMR and CWMR use x_t directly
        return x_t

    def _execute_strategy_update(
        self, trade_w: np.ndarray, x_t: np.ndarray, phi_eff: np.ndarray
    ) -> np.ndarray:
        """Execute the strategy update step."""
        if not self._strategy_impl.should_update(self._t):
            return trade_w.copy()

        return self._strategy_impl.step(
            trade_w=trade_w,
            arr=x_t,
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
        # Initial checking and parameters validation
        self._validate_params()
        first_call = not hasattr(self, "n_features_in_")
        X = validate_data(self, X=X, y=y, reset=first_call, dtype=float, ensure_2d=True)

        # Convert to relatives
        x_t = np.asarray(net_to_relatives(X).squeeze(), dtype=float)
        d = int(x_t.shape[0])

        # Initialize components
        if not self._is_initialized:
            self.n_features_in_ = d
            self._is_initialized = True
        if not self._weights_initialized:
            self._initialize_weights(d)
        # initialize projector if needed and engine if needed
        self._initialize_components(d)

        # Store current weights for trading
        trade_w = self.weights_.copy()
        self._last_trade_weights_ = trade_w

        # Compute effective predictor (OLMAR only)
        phi = self._compute_predictor(x_t, d)

        # Execute strategy update
        next_w = self._execute_strategy_update(trade_w, x_t, phi)

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
