from collections.abc import Callable
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt
from sklearn.utils._param_validation import StrOptions

import skfolio.typing as skt
from skfolio.measures._enums import ExtraRiskMeasure, PerfMeasure, RiskMeasure
from skfolio.optimization.online._autograd_objectives import create_objective
from skfolio.optimization.online._base import OnlinePortfolioSelection
from skfolio.optimization.online._ftrl import (
    FirstOrderOCO,
    Predictor,
    SwordMeta,
)
from skfolio.optimization.online._learning_rate import get_auto_learning_rate
from skfolio.optimization.online._mirror_maps import (
    AdaptiveMahalanobisMap,
    AdaptiveVariationMap,
    BaseMirrorMap,
    BurgMirrorMap,
    CompositeMirrorMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
    make_ada_barrons_mirror_map,
)
from skfolio.optimization.online._mixins import FTWStrategy
from skfolio.optimization.online._prediction import LastGradPredictor, SmoothPredictor
from skfolio.optimization.online._projection import IdentityProjector
from skfolio.optimization.online._utils import CLIP_EPSILON


class FollowTheWinner(OnlinePortfolioSelection):
    """FollowTheWinner: Online Portfolio Selection via Online Convex Optimization and momentum-based strategies.

    It unifies FTRL and OMD through the lens of FTRL-Proximal, which is a unified framework for FTRL and OMD.

    This family of algorithms increases weights on recently successful assets,
    implementing first-order OCO methods including:
    - Mirror Descent family: OGD, EG, PROD (Soft-Bayes)
    - Adaptive methods: AdaGrad, AdaBARRONS
    - Optimistic methods: Smooth Prediction, SWORD family

    Projection:
    - Fast path: box + budget (+ turnover) via project_box_and_sum/project_with_turnover
    - Rich constraints (groups/linear/tracking error/variance): fallback to project_convex

    Data conventions
    ----------------
    - Inputs ``X`` to :meth:`fit` and :meth:`partial_fit` must be NET returns
        (i.e., arithmetic returns r_t in [-1, +inf)). Internally, the estimator
        converts each row to gross returns via ``1.0 + r_t`` before computing
        losses/gradients. This keeps interfaces consistent with most skfolio
        preprocessing pipelines that output net returns.

    Fitting
    -------
    - :meth:`fit` iterates over ``X`` row by row, calling :meth:`partial_fit` on each row to preserve online/sequential updates.

    References
    ----------
    .. [1] Orseau, L., Lattimore, T., & Legg, S. (2017). Soft-Bayes: Prod for
           Mixtures of Experts with Log-Loss. In Algorithmic Learning Theory,
           PMLR 76:73-90.
    """

    _ftrl_engine: FirstOrderOCO
    cumulative_loss_: float

    _parameter_constraints: ClassVar[dict] = {
        "strategy": [StrOptions({m.value.lower() for m in FTWStrategy})],
        "learning_rate_scale": [StrOptions({"theory", "moderate", "empirical"})],
        "objective": [
            StrOptions({m.value.lower() for m in RiskMeasure}),
            StrOptions({m.value.lower() for m in PerfMeasure}),
            None,
        ],
        "update_mode": [StrOptions({"omd", "ftrl"})],
        "grad_predictor": [StrOptions({"last", "smooth"}), None],
    }

    def __init__(
        self,
        strategy: FTWStrategy | str = FTWStrategy.EG,
        *,
        objective: RiskMeasure | ExtraRiskMeasure | PerfMeasure | None = None,
        learning_rate: float | Callable[[int], float] = "auto",
        learning_rate_scale: Literal["theory", "moderate", "empirical"] = "empirical",
        update_mode: Literal["ftrl", "omd"] = "ftrl",
        warm_start: bool = True,
        initial_weights: npt.ArrayLike | None = None,
        initial_wealth: float | npt.ArrayLike | None = None,
        previous_weights: skt.MultiInput | None = None,
        grad_predictor: Literal["last", "smooth"] | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        ## Smooth Prediction
        smooth_epsilon: float = 1.0,
        ## AdaGrad
        adagrad_D: float | npt.ArrayLike | None = None,
        adagrad_eps: float = 1e-8,
        ## Ada-BARRONS
        adabarrons_barrier_coef: float = 1.0,
        adabarrons_alpha: float = 1.0,
        adabarrons_euclidean_coef: float = 1.0,
        adabarrons_beta: float = 0.1,
        eg_tilde: bool = False,
        eg_tilde_alpha: float | Callable[[int], float] = 0.1,
        # Projection constraints (fast path)
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1.0,
        # Rich constraints (fallback)
        X_tracking: npt.ArrayLike | None = None,
        tracking_error_benchmark: npt.ArrayLike | None = None,
        max_tracking_error: float | None = None,
        covariance: npt.ArrayLike | None = None,
        variance_bound: float | None = None,
        portfolio_params: dict | None = None,
    ):
        r"""Follow-The-Winner online portfolio selection estimator.

        Parameters
        ----------
        strategy : FTWStrategy | str, default=FTWStrategy.EG
            Follow-the-Winner strategy to use. Available strategies:

            - 'ogd': Online Gradient Descent (Euclidean geometry)
            - 'eg': Exponential Gradient (entropy geometry)
            - 'prod': Soft-Bayes Product algorithm (Burg entropy)
            - 'adagrad': Adaptive Gradient with diagonal preconditioning
            - 'adabarrons': Ada-BARRONS with adaptive barrier and second-order updates
            - 'sword', 'sword_var': SWORD-Var (variation-adaptive)
            - 'sword_small': SWORD-Small (AdaGrad geometry with optimistic gradients)
            - 'sword_best': SWORD-Best (meta-aggregation of Var + Small)
            - 'sword_pp': SWORD++ (meta-aggregation of Var + Small + EG)

        objective : RiskMeasure | ExtraRiskMeasure | PerfMeasure | None, default=None
            Objective function to minimize. If None (default), uses the standard
            log-wealth objective (maximizing cumulative logarithmic returns).

            - For RiskMeasure or ExtraRiskMeasure: Minimizes the risk measure on portfolio returns.
            - For PerfMeasure: Maximizes the performance measure (minimizes its negative).

            The gradient is computed using automatic differentiation (autograd).
            Log-wealth (default) uses analytical gradients for better performance.

            .. note::
                Only convex objectives guarantee convergence to global optimum.
                Non-convex measures may converge to local optima.

        learning_rate : float | Callable[[int], float] | "auto", default=0.05
            Learning rate :math:`\eta_t` for the optimization step.

            - float: Constant :math:`\eta`
            - Callable: Time-varying :math:`\eta(t)`
            - **"auto"**: Empirically-validated rate per strategy (default: "empirical" scale)

            For EG, the learning rate acts as an inverse temperature parameter.
            As :math:`\eta \to 0`, converges to uniform weights; as :math:`\eta \to \infty`,
            converges to best-performing asset (one-hot).

        learning_rate_scale : {"theory", "moderate", "empirical"}, default="empirical"
            Scaling mode for learning rates:

            - **"empirical"**: n/√(t+1) - Validated on real data (90-95% of optimal log-wealth)
            - **"moderate"**: √(8·log(n)/(t+1)) - Hazan's book constant (75-85% of optimal log-wealth)
            - **"theory"**: √(log(n)/(t+1)) - Hazan's worst-case OCO bound (70-80% of optimal log-wealth)

            **Note:** The 'empirical' scaling is **recommended for financial portfolio selection** where trading does not change the returns too much (oblivious environemnt).
            For cases where slippage is significant (adversarial/worst-case scenarios or safety-critical applications), use explicit learning_rate with conservative values.


        update_mode : {"ftrl", "omd"}, default="ftrl"
            Update rule to use:

            - "omd": Online Mirror Descent (local Bregman/proximal step)
            - "ftrl": Follow-the-Regularized-Leader (global cumulative minimization)

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

        previous_weights : float | dict[str, float] | array-like of shape (n_assets,) | None, default=None
            Previous weights for computing transaction costs and turnover.
            If float, applied to all assets. If dict, maps asset names to weights.

        grad_predictor : {"last", "smooth"} | None, default=None
            Gradient predictor for optimistic updates:

            - "last": Use last gradient as predictor (optimistic OMD)
            - "smooth": Smooth prediction with log-barrier regularization
            - None: No prediction (standard update)

        transaction_costs : float | dict[str, float] | array-like of shape (n_assets,), default=0.0
            Proportional transaction costs per asset. Added as L1 penalty on turnover.
            If float, applied to all assets. If dict, maps asset names to costs.

        management_fees : float | dict[str, float] | array-like of shape (n_assets,), default=0.0
            Management fees per asset (per-period). Modeled as multiplicative drag
            on gross relatives: gradients use net-of-fee relatives.
            If float, applied to all assets. If dict, maps asset names to fees.

        groups : dict[str, list[str]] | array-like of shape (n_groups, n_assets) | None, default=None
            Asset groups for linear constraints. If dict, maps asset names to group labels.

        linear_constraints : list[str] | None, default=None
            Linear constraints on weights (e.g., ``"Equity >= 0.5"``).
            References asset names or group names from `groups`.

        left_inequality : array-like of shape (n_constraints, n_assets) | None, default=None
            Left-hand side matrix :math:`A` for inequality :math:`A w \leq b`.

        right_inequality : array-like of shape (n_constraints,) | None, default=None
            Right-hand side vector :math:`b` for inequality :math:`A w \leq b`.

        max_turnover : float | None, default=None
            Maximum allowed turnover per period (L1 norm of weight changes).

        smooth_epsilon : float, default=1.0
            Log-barrier regularization for Smooth Prediction method.
            Only used with ``grad_predictor="smooth"``.

        adagrad_D : float | array-like of shape (n_assets,) | None, default=None
            Diameter bound(s) for AdaGrad. If float, same for all assets.
            If None, defaults to :math:`\sqrt{2}` (simplex diameter).
            Only used with ``strategy="adagrad"``.

        adagrad_eps : float, default=1e-8
            Numerical stability constant for AdaGrad denominator.
            Only used with ``strategy="adagrad"``.

        adabarrons_barrier_coef : float, default=1.0
            Barrier coefficient for Ada-BARRONS weight-proximity adaptation.
            Only used with ``strategy="adabarrons"``.

        adabarrons_alpha : float, default=1.0
            Barrier scaling parameter for Ada-BARRONS.
            Only used with ``strategy="adabarrons"``.

        adabarrons_euclidean_coef : float, default=1.0
            Euclidean regularization coefficient for Ada-BARRONS.
            Only used with ``strategy="adabarrons"``.

        adabarrons_beta : float, default=0.1
            Second-order curvature coefficient for Ada-BARRONS full quadratic component.
            Only used with ``strategy="adabarrons"``.

        eg_tilde : bool, default=False
            Whether to apply EG-tilde mixing (uniform weight mixing) for EG strategy.
            Only used with ``strategy="eg"``.

        eg_tilde_alpha : float | Callable[[int], float], default=0.1
            Mixing coefficient for EG-tilde. Final weights are
            :math:`(1 - \\alpha) w_{EG} + \\alpha / n`.
            Can be constant or callable. Only used if ``eg_tilde=True``.

        min_weights : float | dict[str, float] | array-like of shape (n_assets,) | None, default=0.0
            Minimum weight per asset (lower bound). If float, applied to all assets.

        max_weights : float | dict[str, float] | array-like of shape (n_assets,) | None, default=1.0
            Maximum weight per asset (upper bound). If float, applied to all assets.

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

        """
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

        # Public configuration
        self.strategy = strategy
        self.objective = objective
        self.learning_rate = learning_rate
        self.learning_rate_scale = learning_rate_scale
        self.update_mode = update_mode
        self.smooth_epsilon = smooth_epsilon
        self.adagrad_D = adagrad_D
        self.adagrad_eps = adagrad_eps
        self.adabarrons_barrier_coef = adabarrons_barrier_coef
        self.adabarrons_alpha = adabarrons_alpha
        self.adabarrons_euclidean_coef = adabarrons_euclidean_coef
        self.adabarrons_beta = adabarrons_beta
        self.eg_tilde = eg_tilde
        self.eg_tilde_alpha = eg_tilde_alpha
        self.warm_start = warm_start
        self.initial_weights = initial_weights
        self.grad_predictor = grad_predictor

        # Costs and fees (public attributes preserved for predict())
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees

        # Projection parameters
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.budget = budget
        self.previous_weights = previous_weights
        self.max_turnover = max_turnover

        # Rich constraints
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.left_inequality = left_inequality
        self.right_inequality = right_inequality
        self.X_tracking = X_tracking
        self.tracking_error_benchmark = tracking_error_benchmark
        self.max_tracking_error = max_tracking_error
        self.covariance = covariance
        self.variance_bound = variance_bound

        # Internal state (initialized deterministically)
        self._ftrl_engine: FirstOrderOCO | None = None
        self.cumulative_loss_: float = 0.0

    def _ensure_initialized(self, gross_relatives: np.ndarray) -> None:
        num_assets = int(gross_relatives.shape[0])

        if not self._is_initialized:
            self.n_features_in_ = num_assets

        self._transaction_costs_arr = self._clean_input(
            self.transaction_costs,
            n_assets=num_assets,
            fill_value=0,
            name="transaction_costs",
        )

        self._management_fees_arr = self._clean_input(
            self.management_fees,
            n_assets=num_assets,
            fill_value=0,
            name="management_fees",
        )

        # Resolve transaction costs and management fees once number of assets is known

        self._projector = self._initialize_projector()

        # Initialize objective function
        if not hasattr(self, "_objective_fn"):
            self._objective_fn = create_objective(self.objective, use_autograd=False)

        if self._ftrl_engine is None:
            mirror_map: BaseMirrorMap | None = None
            predictor: Predictor | None = None

            # Resolve learning rate if "auto"
            effective_learning_rate = self.learning_rate
            if self.learning_rate == "auto":
                effective_learning_rate = get_auto_learning_rate(
                    strategy=self.strategy,
                    n_assets=num_assets,
                    min_weights=self.min_weights,
                    max_weights=self.max_weights,
                    budget=self.budget,
                    objective=self.objective,  # pass enum directly
                    gradient_bound=None,  # use automatic estimation
                    historical_returns=None,  # no historical data at init
                    scale=self.learning_rate_scale,
                )

            match self.grad_predictor:
                case "last":
                    predictor = LastGradPredictor()
                case "smooth":
                    predictor = SmoothPredictor()
                case None:
                    predictor = None

            match self.strategy:
                case FTWStrategy.EG | "eg":
                    mirror_map = EntropyMirrorMap()
                case FTWStrategy.OGD | "ogd":
                    mirror_map = EuclideanMirrorMap()
                case FTWStrategy.PROD | "prod":
                    mirror_map = BurgMirrorMap()
                case FTWStrategy.ADAGRAD | "adagrad":
                    mirror_map = AdaptiveMahalanobisMap(eps=self.adagrad_eps)
                case FTWStrategy.ADABARRONS | "adabarrons":
                    # Ada-BARRONS uses a compositional mirror map:
                    # - AdaBarronsBarrierMap: weight-proximity adaptive barrier
                    # - EuclideanMap: Euclidean regularization
                    # - FullQuadraticMap: second-order curvature
                    mirror_map = make_ada_barrons_mirror_map(
                        d=num_assets,
                        barrier_coef=self.adabarrons_barrier_coef,
                        alpha=self.adabarrons_alpha,
                        euclidean_coef=self.adabarrons_euclidean_coef,
                        beta=self.adabarrons_beta,
                    )
                case FTWStrategy.SWORD_VAR | "sword_var":
                    # SWORD-Var: variation-adaptive OMD with optimistic gradients
                    mirror_map = AdaptiveVariationMap(eps=self.adagrad_eps)
                case FTWStrategy.SWORD_SMALL | "sword_small":
                    # SWORD-Small: AdaGrad geometry with optimistic gradients
                    mirror_map = AdaptiveMahalanobisMap(eps=self.adagrad_eps)
                case (
                    FTWStrategy.SWORD_BEST
                    | FTWStrategy.SWORD_PP
                    | "sword_best"
                    | "sword_pp"
                ):
                    # Meta-aggregator combining SWORD-Var and SWORD-Small, plus EG for Sword++
                    var_engine = FirstOrderOCO(
                        mirror_map=AdaptiveVariationMap(eps=self.adagrad_eps),
                        projector=IdentityProjector(),  # inner unconstrained
                        eta=effective_learning_rate,
                        predictor=LastGradPredictor(),
                        mode="omd",
                    )
                    small_engine = FirstOrderOCO(
                        mirror_map=AdaptiveMahalanobisMap(eps=self.adagrad_eps),
                        projector=IdentityProjector(),
                        eta=effective_learning_rate,
                        predictor=LastGradPredictor(),
                        mode="omd",
                    )
                    experts: list[FirstOrderOCO] = [var_engine, small_engine]
                    if self.strategy in (FTWStrategy.SWORD_PP, "sword_pp"):
                        # add an entropy-geometry expert to stabilize/explore (only for Sword++)
                        eg_engine = FirstOrderOCO(
                            mirror_map=EntropyMirrorMap(),
                            projector=IdentityProjector(),
                            eta=effective_learning_rate,
                            predictor=LastGradPredictor(),
                            mode="omd",
                        )
                        experts.append(eg_engine)
                    # specific case treating SwordMeta differently
                    # we initialize it here because it needs to be initialized with the experts (only for SwordMeta)
                    self._ftrl_engine = SwordMeta(
                        experts=experts,
                        projector=self._projector,
                        eta_meta=effective_learning_rate,  # tie meta-eta to learning_rate
                    )
                case _:
                    raise ValueError(f"Unknown strategy provided {self.strategy}")

            # Finally initialize the FTRL engine to be used in the fit method
            if self._ftrl_engine is None:
                skip_auto_update = self.strategy in (
                    FTWStrategy.ADABARRONS,
                    "adabarrons",
                )
                self._ftrl_engine = FirstOrderOCO(
                    mirror_map=mirror_map,
                    projector=self._projector,
                    eta=effective_learning_rate,
                    predictor=predictor,
                    mode=self.update_mode,
                    skip_auto_update=skip_auto_update,
                )

        if not self._weights_initialized or not self.warm_start:
            self._initialize_weights(num_assets)
            # Initialize wealth tracking
            if not self._wealth_initialized:
                self._initialize_wealth(num_assets)
            if self._ftrl_engine is not None and hasattr(self._ftrl_engine, "_x_t"):
                if self._ftrl_engine._x_t is None:
                    self._ftrl_engine._x_t = self.weights_.copy()

        # Mark overall init complete
        self._is_initialized = True

    def _compute_effective_relatives(self, gross_relatives: np.ndarray) -> np.ndarray:
        """Apply management fees to gross relatives.

        Parameters
        ----------
        gross_relatives : np.ndarray
            Gross price relatives for one period.

        Returns
        -------
        np.ndarray
            Effective relatives after applying management fees.
        """
        # Apply management fees multiplicatively to gross relatives for Kelly-like gradients
        effective_relatives = np.maximum(
            gross_relatives * (1 - self._management_fees_arr), CLIP_EPSILON
        )
        return effective_relatives

    def _compute_portfolio_gradient(
        self, effective_relatives: np.ndarray
    ) -> np.ndarray:
        """Compute the portfolio gradient including transaction costs.

        Parameters
        ----------
        effective_relatives : np.ndarray
            Effective price relatives after fees.

        Returns
        -------
        np.ndarray
            Gradient vector for the portfolio optimization.
        """
        # Convert effective relatives to net returns for objective function
        # Objective functions now work with net returns (r = x - 1)
        effective_net_returns = effective_relatives - 1.0

        # Compute gradient using objective function (log-wealth by default, or custom risk/perf measure)
        gradient = self._objective_fn.grad(self.weights_, effective_net_returns)

        # Add L1 turnover subgradient for transaction costs: grad C_t(b) = c * sign(b - b_prev)
        if self.transaction_costs and self.previous_weights is not None:
            prev = np.asarray(self.previous_weights, dtype=float)
            if prev.shape == self.weights_.shape:
                delta = self.weights_ - prev
                gradient += self._transaction_costs_arr * np.sign(delta)

        return gradient

    def _execute_ftrl_step(self, gradient: np.ndarray) -> np.ndarray:
        """Execute the FTRL optimization step.

        Parameters
        ----------
        gradient : np.ndarray
            Portfolio gradient vector.

        Returns
        -------
        np.ndarray
            New portfolio weights from FTRL step.
        """
        if self._ftrl_engine:
            w_ftrl = self._ftrl_engine.step(gradient)
        else:
            raise RuntimeError("FTRL Engine not initialized")
        return w_ftrl

    def _apply_weights_mixing(self, w_ftrl: np.ndarray) -> np.ndarray:
        """Apply weights mixing (only for EG-tilde mixing).

        Parameters
        ----------
        w_ftrl : np.ndarray
            Weights from FTRL step.

        Returns
        -------
        np.ndarray
            Final weights.
        """
        if not (self.eg_tilde and self.strategy == FTWStrategy.EG):
            return w_ftrl

        alpha_t = (
            self.eg_tilde_alpha(self._ftrl_engine._t)
            if callable(self.eg_tilde_alpha)
            else self.eg_tilde_alpha
        )

        if alpha_t <= 0:
            return w_ftrl

        n = w_ftrl.shape[0]
        mixed = (1.0 - alpha_t) * w_ftrl + alpha_t / n
        return self._projector.project(mixed)

    def _update_adabarrons_components(
        self, weights: np.ndarray, gradient: np.ndarray
    ) -> None:
        """Special update logic for Ada-BARRONS compositional mirror map.

        Ada-BARRONS requires:
        - Barrier component updated with weights (weight-proximity adaptation)
        - Full quadratic component updated with gradients (second-order curvature)

        Parameters
        ----------
        weights : np.ndarray
            Current portfolio weights.
        gradient : np.ndarray
            Portfolio gradient vector.
        """
        if self.strategy == FTWStrategy.ADABARRONS and isinstance(
            self._ftrl_engine.map, CompositeMirrorMap
        ):
            # Access the components of the Ada-BARRONS mirror map:
            # [0] = AdaBarronsBarrierMap (weight-proximity adaptive)
            # [1] = EuclideanMap (static, no update needed)
            # [2] = FullQuadraticMap (gradient-based second-order)
            components = self._ftrl_engine.map.components_

            # Update barrier with weights (NOT gradients)
            components[0].update_state(weights)

            # Update full quadratic with gradients
            components[2].update_state(gradient)

    def _finalize_partial_fit_state(self, effective_relatives: np.ndarray) -> None:
        """Update internal state and compute loss after weight update.

        Parameters
        ----------
        effective_relatives : np.ndarray
            Effective price relatives used for loss computation.
        """
        # Compute loss for inspection (optional, can be simplified)
        final_return = np.dot(self.weights_, effective_relatives)
        self.loss_ = -np.log(np.maximum(final_return, CLIP_EPSILON))
        self.cumulative_loss_ += self.loss_
        self.previous_weights = self._last_trade_weights_.copy()
        self._t += 1

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> "FollowTheWinner":
        """Perform one online update with a single period of net returns.

        In OCO, ``partial_fit`` is the core update and must receive exactly one
        row of data (one period). Use :meth:`fit` to iterate over multiple rows
        in sequential order.

        Parameters
        ----------
        X : array-like of shape (n_assets,) or (1, n_assets)
            Net returns for a single period. Internally converted to price
            relatives via ``1.0 + r_t``.
        y : Ignored
            Present for API consistency.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently ignored but present for API consistency.
        **fit_params : Any
            Additional parameters (unused).

        Returns
        -------
        OPS
            The estimator instance.
        """
        # Step 1: Validate input and preprocess to gross relatives
        gross_relatives = self._validate_and_preprocess_partial_fit_input(
            X, y, sample_weight
        )

        # Step 2: Initialize engine and weights if needed
        self._ensure_initialized(gross_relatives)

        # Step 3: Apply management fees to get effective relatives
        effective_relatives = self._compute_effective_relatives(gross_relatives)

        # Step 4: Compute portfolio gradient (including transaction costs)
        gradient = self._compute_portfolio_gradient(effective_relatives)

        # Step 5: Execute FTRL optimization step
        w_ftrl = self._execute_ftrl_step(gradient)

        # Step 5.5: Store trading weights before update
        self._last_trade_weights_ = self.weights_.copy()

        # Step 6: Apply post-processing (e.g., EG-tilde mixing)
        self.weights_ = self._apply_weights_mixing(w_ftrl)

        # Step 6.5: Manual component updates for Ada-BARRONS
        # (barrier with weights, full quadratic with gradients)
        self._update_adabarrons_components(self.weights_, gradient)

        # Step 7: Update internal state and compute loss
        self._finalize_partial_fit_state(effective_relatives)

        # Step 8: Update wealth tracking
        if hasattr(self, "_wealth_initialized") and self._wealth_initialized:
            prev_weights = self.previous_weights if self._t > 1 else None
            self._update_wealth(
                trade_weights=self._last_trade_weights_,
                effective_relatives=effective_relatives,
                previous_weights=prev_weights,
            )

        return self

    def _reset_state_for_fit(self) -> None:
        """Reset internal state when warm_start=False."""
        super()._reset_state_for_fit()
        self._ftrl_engine = None
        self.cumulative_loss_ = 0.0
