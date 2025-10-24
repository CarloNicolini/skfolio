"""
Mean-reversion portfolio selection strategies.
Unified interface for OLMAR, PAMR, and CWMR strategies with both passive-aggressive and mirror-descent update modes.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import validate_data

import skfolio.typing as skt
from skfolio.optimization.online._autograd_objectives import (
    BaseObjective,
    create_mean_reversion_objective,
)
from skfolio.optimization.online._base import OnlinePortfolioSelection
from skfolio.optimization.online._ftrl import FirstOrderOCO
from skfolio.optimization.online._mirror_maps import (
    BaseMirrorMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
)
from skfolio.optimization.online._mixins import (
    FTLStrategy,
    OLMARPredictor,
    OnlineParameterConstraintsMixin,
    PAMRVariant,
    UpdateMode,
)
from skfolio.optimization.online._prediction import (
    BaseReversionPredictor,
    LastGradPredictor,
    OLMAR1Predictor,
    OLMAR2Predictor,
    RMRPredictor,
)
from skfolio.optimization.online._strategies import (
    BaseStrategy,
    CWMRStrategy,
    OLMARStrategy,
    PAMRStrategy,
)
from skfolio.optimization.online._utils import CLIP_EPSILON, net_to_relatives


class FollowTheLoser(OnlinePortfolioSelection):
    r"""Follow-the-Loser: Online Portfolio Selection via mean-reversion strategies.

    This family of algorithms transfers wealth from recent winners to recent losers,
    exploiting mean reversion in asset prices:

    - OLMAR: Online Moving Average Reversion (multi-period)
    - PAMR: Passive-Aggressive Mean Reversion (single-period)
    - CWMR: Confidence-Weighted Mean Reversion (second-order distributional)
    - RMR: Robust Median Reversion (L1-median, outlier-robust)

    Projection:

    - Fast path: box + budget (+ turnover) via project_box_and_sum/project_with_turnover
    - Rich constraints (groups/linear/tracking error/variance): fallback to project_convex

    Data conventions
    ----------------
    Inputs ``X`` to :meth:`fit` and :meth:`partial_fit` must be NET returns
    (i.e., arithmetic returns r_t in [-1, +inf)). Internally, the estimator
    converts each row to gross returns via ``1.0 + r_t`` before computing
    losses/gradients.

    Parameters
    ----------
    strategy : str | FTLStrategy
        Mean-reversion strategy: 'olmar', 'pamr', or 'cwmr'.

        - OLMAR: Online Moving Average Reversion
        - PAMR: Passive-Aggressive Mean Reversion
        - CWMR: Confidence-Weighted Mean Reversion

    olmar_predictor : OLMARPredictor | str, default=OLMARPredictor.SMA
        OLMAR reversion predictor type. Only used with ``strategy="olmar"``.

        - "sma": Simple Moving Average (OLMAR-1 from Li & Hoi 2012)
        - "ewma": Exponentially Weighted Moving Average (OLMAR-2 from Li & Hoi 2012)

    olmar_window : int, default=5
        OLMAR window size for SMA predictor. Only used with ``strategy="olmar"``
        and ``olmar_predictor="sma"``.

    olmar_alpha : float, default=0.5
        OLMAR smoothing parameter for EWMA predictor (in [0, 1]).
        Only used with ``strategy="olmar"`` and ``olmar_predictor="ewma"``.

    rmr_window : int, default=5
        RMR window size for L1-median computation. Only used with ``strategy="rmr"``.

    rmr_max_iter : int, default=200
        RMR maximum iterations for L1-median algorithm (Weiszfeld).
        Only used with ``strategy="rmr"``. This parameter follows sklearn
        convention for iterative algorithms (similar to other estimators' max_iter).

    rmr_tolerance : float, default=1e-9
        RMR convergence tolerance for L1-median algorithm.
        Only used with ``strategy="rmr"``. This parameter follows sklearn
        convention for iterative convergence (similar to other estimators' tol).

    pamr_variant : PAMRVariant | str, default="simple"
        PAMR variant. Only used with ``strategy="pamr"``.

        - "simple": Original PAMR (PAMR-0)
        - "slack_linear": Linear slack variable (PAMR-1)
        - "slack_quadratic": Quadratic slack regularization (PAMR-2)

    pamr_C : float, default=500.0
        PAMR aggressiveness parameter. Denotes the aggressiveness of reverting to a particular strategy. Only used with ``strategy="pamr"``.
        Typically, 100 has the highest returns for 'slack-linear' and 10000 has the highest returns for 'slack-quadratic'.

    cwmr_eta : float, default=0.95
        CWMR confidence level in (0.5, 1.0). Higher values enforce tighter
        probabilistic margin constraints. Only used with ``strategy="cwmr"``.

    cwmr_sigma0 : float, default=1.0
        CWMR initial diagonal variance. Only used with ``strategy="cwmr"``.

    cwmr_min_var : float | None, default=1e-12
        CWMR minimum variance bound (prevents numerical underflow).
        Only used with ``strategy="cwmr"``.

    cwmr_max_var : float | None, default=None
        CWMR maximum variance bound. If None, no upper clipping.
        Only used with ``strategy="cwmr"``.

    cwmr_mean_lr : float, default=1.0
        CWMR mean learning rate (MD mode). Only used with ``strategy="cwmr"``
        and ``update_mode="md"``.

    cwmr_var_lr : float, default=1.0
        CWMR variance learning rate (MD mode). Only used with ``strategy="cwmr"``
        and ``update_mode="md"``.

    epsilon : float, default=2.0
        Margin threshold for detecting mean reversion opportunities.
        Larger values → more tolerant (fewer false positives, less turnover).
        Smaller values → more sensitive (higher turnover).

    objective : str, default="hinge"
        Objective function for gradient computation (both PA and MD modes):

        - "hinge": max(0, ε - margin) (standard, non-smooth)
        - "squared_hinge": [max(0, ε - margin)]² (smooth variant)
        - "softplus": (1/β) log(1 + exp(β(ε - margin))) (smooth, differentiable)

    beta : float, default=5.0
        Softplus temperature parameter. Larger β → sharper hinge approximation.
        Only used with ``objective="softplus"``.

    update_mode : str, default="pa"
        Update mode:

        - "pa": Passive-Aggressive (closed-form, original algorithms)
        - "md": Mirror Descent (OCO-style, supports flexible geometries and learning rates)

    learning_rate : float | int | callable, default=1.0
        Learning rate for MD mode. Can be constant or callable ``f(t)``.
        Only used with ``update_mode="md"``.

    apply_fees_to_phi : bool, default=True
        Whether to apply management fees to reversion predictor phi.
        If True, phi is scaled by (1 - fees) to account for fee drag.

    mirror : str, default="euclidean"
        Mirror map for MD mode:

        - "euclidean": Euclidean geometry (OGD-style)
        - "entropy": Entropy geometry (EG-style, multiplicative weights)

        Only used with ``update_mode="md"``.

    warm_start : bool, default=True
        Whether to warm start the estimator. If False, resets state before fitting.

    initial_weights : array-like of shape (n_assets,) | None, default=None
        Initial portfolio weights. If None, uses uniform weights (1/n_assets).

    initial_wealth : float | array-like of shape (n_assets,) | None, default=None
        Initial portfolio wealth for tracking purposes only (does not affect optimization).

        - If float: Total initial wealth (e.g., 100000.0 for $100k)
        - If array: Per-asset initial dollar amounts (e.g., [10000, 20000, 30000]).
          If `initial_weights` not provided, weights are computed proportionally.
        - If None: Defaults to 1.0 (unit wealth)

        Note: If both `initial_wealth` and `initial_weights` are arrays, raises ValueError.

        The estimator tracks:

        - `wealth_` : Current portfolio wealth (updated each period)
        - `all_wealth_` : Historical wealth after each period (shape (T+1,))

    previous_weights : array-like of shape (n_assets,) | None, default=None
        Previous weights for computing transaction costs and turnover.

    transaction_costs : float | dict[str, float] | array-like of shape (n_assets,), default=0.0
        Proportional transaction costs per asset.

    management_fees : float | dict[str, float] | array-like of shape (n_assets,), default=0.0
        Management fees per asset (per-period).

    groups : dict[str, list[str]] | array-like of shape (n_groups, n_assets) | None, default=None
        Asset groups for linear constraints.

    linear_constraints : list[str] | None, default=None
        Linear constraints on weights (e.g., ``"Equity >= 0.5"``).

    left_inequality : array-like of shape (n_constraints, n_assets) | None, default=None
        Left-hand side matrix :math:`A` for inequality :math:`A w \leq b`.

    right_inequality : array-like of shape (n_constraints,) | None, default=None
        Right-hand side vector :math:`b` for inequality :math:`A w \leq b`.

    max_turnover : float | None, default=None
        Maximum allowed turnover per period (L1 norm of weight changes).

    min_weights : float | dict[str, float] | array-like of shape (n_assets,) | None, default=0.0
        Minimum weight per asset (lower bound).

    max_weights : float | dict[str, float] | array-like of shape (n_assets,) | None, default=1.0
        Maximum weight per asset (upper bound).

    budget : float | None, default=1.0
        Investment budget (sum of weights). If None, no budget constraint.

    X_tracking : array-like of shape (n_observations, n_assets) | None, default=None
        Tracking portfolio returns for tracking error constraint.

    tracking_error_benchmark : array-like of shape (n_observations,) | None, default=None
        Benchmark returns for tracking error calculation.

    max_tracking_error : float | None, default=None
        Maximum allowed tracking error (standard deviation of tracking difference).

    covariance : array-like of shape (n_assets, n_assets) | None, default=None
        Covariance matrix for variance constraint.

    variance_bound : float | None, default=None
        Maximum allowed portfolio variance.

    portfolio_params : dict | None, default=None
        Additional portfolio parameters passed to base class.

    References
    ----------
    .. [1] Li, B., & Hoi, S. C. H. (2013). Online Portfolio Selection: A Survey.
           arXiv:1212.2129.
    .. [2] Li, B., Zhao, P., Hoi, S. C. H., & Gopalkrishnan, V. (2012). PAMR:
           Passive aggressive mean reversion strategy for portfolio selection.
           Machine Learning, 87(2), 221-258.
    .. [3] Li, B., Hoi, S. C. H., Zhao, P., & Gopalkrishnan, V. (2011). Confidence
           weighted mean reversion strategy for on-line portfolio selection. In
           AISTATS (pp. 434-442).
    .. [4] Huang, D., Zhou, J., Li, B., Hoi, S. C. H., & Zhou, S. (2013).
           Robust Median Reversion Strategy for On-Line Portfolio Selection.
           IJCAI 2013.
    """

    _parameter_constraints: ClassVar[dict] = {
        **OnlineParameterConstraintsMixin._parameter_constraints,
        # Strategy to replicate mean reversion algorithm
        "strategy": [
            StrOptions({m.value.lower() for m in FTLStrategy}),
            None,
        ],
        # OLMAR parameters
        "olmar_predictor": [StrOptions({m.value.lower() for m in OLMARPredictor})],
        "olmar_window": [Interval(Integral, 1, None, closed="left")],
        "olmar_alpha": [Interval(Real, 0, 1, closed="both")],
        # RMR parameters
        "rmr_window": [Interval(Integral, 1, None, closed="left")],
        "rmr_max_iter": [Interval(Integral, 1, 10000, closed="both")],
        "rmr_tolerance": [Interval(Real, 1e-12, 1e-3, closed="both")],
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
        "epsilon": [Interval(Real, 0.0, None, closed="left")],
        "objective": [StrOptions({"hinge", "squared_hinge", "softplus"})],
        "beta": [Interval(Real, 0, None, closed="neither")],
        "learning_rate": [
            Interval(Real, 0, None, closed="neither"),
            callable,
            StrOptions({"auto"}),
        ],
        "update_mode": [StrOptions({m.value.lower() for m in UpdateMode})],
        "apply_fees_to_phi": ["boolean"],
        "mirror": [StrOptions({"euclidean", "entropy"})],
    }

    def __init__(
        self,
        *,
        strategy: str | FTLStrategy,
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
        rmr_window: int = 5,
        rmr_max_iter: int = 200,
        rmr_tolerance: float = 1e-9,
        epsilon: float = 1.0,
        objective: str = "hinge",
        beta: float = 5.0,
        update_mode: str = "pa",
        learning_rate: float | int | callable = 1.0,
        apply_fees_to_phi: bool = True,
        mirror: Literal["euclidean", "entropy"] = "euclidean",
        warm_start: bool = True,
        initial_weights: npt.ArrayLike | None = None,
        initial_wealth: float | npt.ArrayLike | None = None,
        previous_weights: npt.ArrayLike | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
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
            initial_wealth=initial_wealth,
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
        self.rmr_window = rmr_window
        self.rmr_max_iter = rmr_max_iter
        self.rmr_tolerance = rmr_tolerance
        self.pamr_variant = pamr_variant
        self.pamr_C = pamr_C
        self.cwmr_eta = cwmr_eta
        self.cwmr_sigma0 = cwmr_sigma0
        self.cwmr_min_var = cwmr_min_var
        self.cwmr_max_var = cwmr_max_var
        self.cwmr_mean_lr = cwmr_mean_lr
        self.cwmr_var_lr = cwmr_var_lr
        self.epsilon = epsilon
        self.objective = objective
        self.beta = beta
        self.update_mode = update_mode
        self.learning_rate = learning_rate
        self.apply_fees_to_phi = apply_fees_to_phi
        self.mirror = mirror

        # Internal state
        self._strategy_impl: BaseStrategy | None = None
        self._predictor: BaseReversionPredictor | None = None
        self._objective_fn: BaseObjective | None = None
        self._engine: FirstOrderOCO | None = None

    def _initialize_components(self, d: int) -> None:
        """Initialize all strategy components."""
        self._projector = self._initialize_projector()

        # Initialize fee arrays
        self._transaction_costs_arr = self._clean_input(
            self.transaction_costs,
            n_assets=d,
            fill_value=0.0,
            name="transaction_costs",
        )
        self._management_fees_arr = self._clean_input(
            self.management_fees,
            n_assets=d,
            fill_value=0.0,
            name="management_fees",
        )

        if not self._weights_initialized:
            self._initialize_weights(d)

        # Initialize wealth tracking
        if not self._wealth_initialized:
            self._initialize_wealth(d)

        # Initialize predictor and surrogate
        match self.strategy:
            case FTLStrategy.OLMAR:
                self._init_olmar_components(d)
            case FTLStrategy.PAMR:
                self._init_pamr_components(d)
            case FTLStrategy.CWMR:
                self._init_cwmr_components(d)
            case FTLStrategy.RMR:
                self._init_rmr_components(d)
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy}")

        # Initialize strategy implementation
        if self._strategy_impl is None:
            self._strategy_impl = self._create_strategy(d)

    def _init_olmar_components(self, d: int) -> None:
        """Initialize OLMAR-specific components: predictor, objective, and engine.

        We also allow for alternative kinds of objective, thus extending the family of
        mean-reversion strategies in OLMAR that was initially thought to only support
        Hinge loss (max(0, epsilon - phi^T weights)).
        """
        if self._predictor is None:
            if self.olmar_predictor in (OLMARPredictor.SMA, 1, "sma"):
                self._predictor = OLMAR1Predictor(window=self.olmar_window)
            elif self.olmar_predictor in (OLMARPredictor.EWMA, 2, "ewma"):
                self._predictor = OLMAR2Predictor(alpha=self.olmar_alpha)
            else:
                raise ValueError(f"Unknown OLMAR predictor: {self.olmar_predictor}")
            self._predictor.reset(d)

        if self._objective_fn is None:
            self._objective_fn = create_mean_reversion_objective(
                objective_type=self.objective,
                epsilon=self.epsilon,
                beta=self.beta,
            )

        if self.update_mode == UpdateMode.MD and self._engine is None:
            self._engine = self._create_engine()

    def _init_pamr_components(self, d: int) -> None:
        """Initialize PAMR-specific components."""
        if self._objective_fn is None:
            self._objective_fn = create_mean_reversion_objective(
                objective_type=self.objective,
                epsilon=self.epsilon,
                beta=self.beta,
            )

        if self.update_mode == UpdateMode.MD and self._engine is None:
            self._engine = self._create_engine()

    def _init_cwmr_components(self, d: int) -> None:
        """Initialize CWMR-specific components."""
        # CWMR doesn't need separate initialization
        pass

    def _init_rmr_components(self, d: int) -> None:
        """Initialize RMR-specific components: predictor, objective, and engine.

        RMR uses L1-median predictor but same update logic as OLMAR.
        """
        if self._predictor is None:
            self._predictor = RMRPredictor(
                window=self.rmr_window,
                max_iter=self.rmr_max_iter,
                tolerance=self.rmr_tolerance,
            )
            self._predictor.reset(d)

        if self._objective_fn is None:
            self._objective_fn = create_mean_reversion_objective(
                objective_type=self.objective,
                epsilon=self.epsilon,
                beta=self.beta,
            )

        if self.update_mode == UpdateMode.MD and self._engine is None:
            self._engine = self._create_engine()

    def _create_engine(self) -> FirstOrderOCO:
        """Create FTRL engine for MD mode."""
        match self.mirror:
            case "euclidean":
                mm: BaseMirrorMap = EuclideanMirrorMap()
            case "entropy":
                mm = EntropyMirrorMap()
            case _:
                raise ValueError(f"Unknown mirror: {self.mirror}")

        return FirstOrderOCO(
            mirror_map=mm,
            projector=self._projector,
            eta=self.learning_rate,
            predictor=LastGradPredictor(),
            mode="omd",
        )

    def _create_strategy(self, d: int) -> BaseStrategy:
        """Factory method to create strategy implementation."""
        match self.strategy:
            case FTLStrategy.OLMAR:
                strategy = OLMARStrategy(
                    predictor=self._predictor,
                    objective=self._objective_fn,
                    engine=self._engine,
                    epsilon=self.epsilon,
                    olmar_order=self.olmar_predictor,
                    transaction_costs_arr=self._transaction_costs_arr,
                )
            case FTLStrategy.PAMR:
                strategy = PAMRStrategy(
                    objective=self._objective_fn,
                    engine=self._engine,
                    epsilon=self.epsilon,
                    pamr_variant=self.pamr_variant,
                    pamr_C=self.pamr_C,
                    transaction_costs_arr=self._transaction_costs_arr,
                )
            case FTLStrategy.CWMR:
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
            case FTLStrategy.RMR:
                strategy = OLMARStrategy(
                    predictor=self._predictor,
                    objective=self._objective_fn,
                    engine=self._engine,
                    epsilon=self.epsilon,
                    olmar_order="rmr",
                    transaction_costs_arr=self._transaction_costs_arr,
                )

        # Initialize the strategy
        strategy.reset(d)
        return strategy

    def _compute_predictor(self, x_t: np.ndarray, d: int) -> np.ndarray:
        """Compute effective reversion predictor (only for OLMAR/RMR)."""
        if self.strategy in (FTLStrategy.OLMAR, FTLStrategy.RMR):
            phi = self._predictor.update_and_predict(x_t)
            # Apply fees to phi if requested
            if self.apply_fees_to_phi:
                phi = phi * (1.0 - self._management_fees_arr)
            return phi
        # PAMR and CWMR use x_t directly
        return x_t

    def _execute_strategy_update(
        self, trade_w: np.ndarray, x_t: np.ndarray, phi_eff: np.ndarray
    ) -> np.ndarray:
        """Execute the strategy update step."""
        if not self._strategy_impl.should_update(self._t):
            return trade_w.copy()

        # For OLMAR/RMR, pass phi_eff; for PAMR/CWMR, pass x_t
        arr = phi_eff if self.strategy in (FTLStrategy.OLMAR, FTLStrategy.RMR) else x_t

        return self._strategy_impl.step(
            trade_w=trade_w,
            arr=arr,
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
        if self.strategy == FTLStrategy.CWMR and self._strategy_impl:
            return getattr(self._strategy_impl, "_Sdiag", None)
        return None

    @property
    def _cwmr_mu(self) -> np.ndarray | None:
        """Expose CWMR mean for backward compatibility with tests."""
        if self.strategy == FTLStrategy.CWMR and self._strategy_impl:
            return getattr(self._strategy_impl, "_mu", None)
        return None

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> FollowTheLoser:
        # Initial checking and parameters validation
        self._validate_params()
        first_call = not hasattr(self, "n_features_in_")
        X = validate_data(self, X=X, y=y, reset=first_call, dtype=float, ensure_2d=True)

        # Convert to relatives
        x_t_gross = np.asarray(net_to_relatives(X).squeeze(), dtype=float)
        d = int(x_t_gross.shape[0])

        # Initialize components
        if not self._is_initialized:
            self.n_features_in_ = d
            self._is_initialized = True
        if not self._weights_initialized:
            self._initialize_weights(d)
        # initialize projector if needed and engine if needed
        self._initialize_components(d)

        # Apply management fees to gross relatives
        x_t = self._compute_effective_relatives(x_t_gross)

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

        # Update wealth tracking
        if hasattr(self, "_wealth_initialized") and self._wealth_initialized:
            prev_weights = self.previous_weights if self._t > 0 else None
            self._update_wealth(
                trade_weights=trade_w,
                effective_relatives=x_t,
                previous_weights=prev_weights,
            )

        self._t += 1
        return self

    def _reset_state_for_fit(self) -> None:
        """Reset internal state when warm_start=False."""
        super()._reset_state_for_fit()
        self._strategy_impl = None
        self._predictor = None
        self._objective_fn = None
        self._engine = None
