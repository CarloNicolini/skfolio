import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import _check_sample_weight, validate_data

import skfolio.typing as skt
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.online._ftrl import (
    LastGradPredictor,
    Predictor,
    SwordMeta,
    _FTRLEngine,
)
from skfolio.optimization.online._mirror_maps import (
    AdaptiveMahalanobisMap,
    AdaptiveVariationMap,
    BaseMirrorMap,
    CompositeMirrorMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
    make_ada_barrons_mirror_map,
)
from skfolio.optimization.online._mixins import (
    FTRLStrategy,
    OnlineMixin,
    OnlineParameterConstraintsMixin,
)
from skfolio.optimization.online._projection import (
    AutoProjector,
    IdentityProjector,
    ProjectionConfig,
)
from skfolio.optimization.online._utils import CLIP_EPSILON, net_to_relatives
from skfolio.utils.tools import input_to_array


class OnlinePortfolioSelection(
    BaseOptimization, OnlineMixin, OnlineParameterConstraintsMixin
):
    """Online Portfolio Selection (OPS) base class.

    This class serves as a foundation for implementing various online portfolio
    selection algorithms. It handles the common logic for fitting data sequentially,
    managing weights, and applying projections. Subclasses should implement the
    core update logic.
    """

    def __init__(
        self,
        *,
        warm_start: bool = True,
        initial_weights: npt.ArrayLike | None = None,
        previous_weights: skt.MultiInput | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1.0,
        X_tracking: npt.ArrayLike | None = None,
        tracking_error_benchmark: npt.ArrayLike | None = None,
        max_tracking_error: float | None = None,
        covariance: npt.ArrayLike | None = None,
        variance_bound: float | None = None,
        portfolio_params: dict | None = None,
    ):
        super().__init__(portfolio_params=portfolio_params)
        self.warm_start = warm_start
        self.initial_weights = initial_weights
        self.previous_weights = previous_weights
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.left_inequality = left_inequality
        self.right_inequality = right_inequality
        self.max_turnover = max_turnover
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.budget = budget
        self.X_tracking = X_tracking
        self.tracking_error_benchmark = tracking_error_benchmark
        self.max_tracking_error = max_tracking_error
        self.covariance = covariance
        self.variance_bound = variance_bound

        self._is_initialized: bool = False
        self._weights_initialized: bool = False
        self._projector: AutoProjector | None = None
        self._t: int = 0
        self._last_trade_weights_: np.ndarray | None = None

    def _initialize_projector(self):
        projection_config = ProjectionConfig(
            lower=self.min_weights,
            upper=self.max_weights,
            budget=self.budget,
            groups=self.groups,
            linear_constraints=self.linear_constraints,
            left_inequality=self.left_inequality,
            right_inequality=self.right_inequality,
            X_tracking=self.X_tracking,
            tracking_error_benchmark=self.tracking_error_benchmark,
            max_tracking_error=self.max_tracking_error,
            covariance=self.covariance,
            variance_bound=self.variance_bound,
            previous_weights=self.previous_weights,
            max_turnover=self.max_turnover,
        )
        self._projector = AutoProjector(projection_config)

    def _initialize_weights(self, num_assets: int):
        if self.initial_weights is not None:
            initial = np.asarray(self.initial_weights, dtype=float)
            if initial.shape != (num_assets,):
                raise ValueError("initial_weights has incompatible shape")
            self.weights_ = initial
        else:
            self.weights_ = np.ones(num_assets, dtype=float) / float(num_assets)
        self._weights_initialized = True

    def _validate_and_preprocess_partial_fit_input(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """Validate and preprocess input for partial_fit.

        Returns
        -------
        np.ndarray
            Gross relatives for a single period as 1D array.
        """
        # Validate parameters
        self._validate_params()

        # Check if this is the first call to partial_fit
        first_call = not hasattr(self, "n_features_in_")

        # Validate input data - reset=True only on first call
        X = validate_data(
            self,
            X=X,
            y=None,  # y is always ignored in OPS
            reset=first_call,
            dtype=float,
            ensure_2d=False,
            allow_nd=False,
            accept_sparse=False,
        )

        # Handle sample_weight if provided (though it's ignored)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            warnings.warn(
                "sample_weight is ignored in OPS.partial_fit (online convex optimization).",
                UserWarning,
                stacklevel=3,
            )

        # Convert to proper shape for partial_fit (single row)
        net_returns = np.asarray(X, dtype=float)

        # Ensure we have the right shape for partial_fit
        if net_returns.ndim == 1:
            # Single sample as 1D array - this is expected for partial_fit
            pass
        elif net_returns.ndim == 2:
            if net_returns.shape[0] > 1:
                raise ValueError(
                    "partial_fit expects a single row (one period). Use fit for multiple rows."
                )
            # Single sample as 2D array (1, n_features) - squeeze to 1D
            net_returns = net_returns.squeeze(0)
        else:
            raise ValueError("Input must be 1D or 2D array")

        gross_relatives = net_to_relatives(net_returns).squeeze()
        return gross_relatives

    def _reset_state_for_fit(self) -> None:
        """Reset internal state when warm_start=False. Subclasses override to add specific state."""
        self._weights_initialized = False
        self._is_initialized = False
        self._projector = None
        self._t = 0
        self._last_trade_weights_ = None

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> "OnlinePortfolioSelection":
        """Iterate over rows and call partial_fit for each period.

        In OCO, ``fit`` is a convenience wrapper. It does not aggregate gradients or
        perform multi-epoch training. Each row of ``X`` is processed once in order.

        Parameters
        ----------
        X : array-like of shape (T, n_assets) or (n_assets,)
            Net returns per period.
        y : Ignored
            Present for API consistency.
        **fit_params : Any
            Additional parameters (unused).

        Returns
        -------
        self
            The estimator instance.
        """
        if not self.warm_start:
            self._reset_state_for_fit()

        trade_list: list[np.ndarray] = []
        X_arr = np.asarray(X, dtype=float)
        for t in range(X_arr.shape[0]):
            self.partial_fit(X_arr[t][None, :], y, sample_weight=None, **fit_params)
            if self._last_trade_weights_ is not None:
                trade_list.append(self._last_trade_weights_.copy())

        if trade_list:
            self.all_weights_ = np.vstack(trade_list)
        return self

    def _clean_input(
        self,
        value: float | dict | npt.ArrayLike | None,
        n_assets: int,
        fill_value: Any,
        name: str,
    ) -> float | np.ndarray:
        """Convert input to cleaned float or ndarray."""
        if value is None:
            return fill_value
        if np.isscalar(value):
            return float(value)
        return input_to_array(
            items=value,
            n_assets=n_assets,
            fill_value=fill_value,
            dim=1,
            assets_names=(
                self.feature_names_in_ if hasattr(self, "feature_names_in_") else None
            ),
            name=name,
        )


class FTRLProximal(OnlinePortfolioSelection):
    """FTRLProximal: Online Portfolio Selection via Online Convex Optimization.

    It unifies FTRL and OMD through the lens of FTRL-Proximal, which is a unified framework for FTRL and OMD.

    Implements standard first-order OCO methods:
    - Mirror Descent family: OGD, EG
    - Adaptive methods: AdaGrad, AdaBARRONS
    - Optimistic methods: Smooth Prediction

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
    """

    def __init__(
        self,
        strategy: FTRLStrategy = FTRLStrategy.EG,
        *,
        ftrl: bool = False,
        warm_start: bool = True,
        initial_weights: npt.ArrayLike | None = None,
        previous_weights: skt.MultiInput | None = None,
        learning_rate: float | Callable[[int], float] = 0.05,
        smooth_prediction: bool = False,
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
        r"""
        The Online Portfolio Selection estimator.

        Parameters
        ----------
            strategy: FTRLStrategy, default=FTRLStrategy.EG
                The FTRL strategy to use.
            ftrl: bool, default=False
                If `True`, use the Follow-the-regularized-leader (FTRL) update.
                If `False` (default), use the Online Mirror Descent (OMD) update.
            learning_rate: float, default=0.05
                Step size :math:`\eta_t` scaling factor for the learning rate schedule.
                Can be a float or a callable `lambda t: f(t)`.
            transaction_costs : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
                Transaction costs of the assets. It is used to add linear transaction costs to
                the optimization problem:

                .. math:: total\_cost = \sum_{i=1}^{N} c_{i} \times |w_{i} - w\_prev_{i}|

                with :math:`c_{i}` the transaction cost of asset i, :math:`w_{i}` its weight
                and :math:`w\_prev_{i}` its previous weight (defined in `previous_weights`).
                The float :math:`total\_cost` is impacting the portfolio expected return in the optimization:

                .. math:: expected\_return = \mu^{T} \cdot w - total\_cost

                with :math:`\mu` the vector af assets' expected returns and :math:`w` the
                vector of assets weights.

                If a float is provided, it is applied to each asset.
                If a dictionary is provided, its (key/value) pair must be the
                (asset name/asset cost) and the input `X` of the `fit` method must be a
                DataFrame with the assets names in columns.
                The default value is `0.0`.

                .. warning::

                    Based on the above formula, the periodicity of the transaction costs
                    needs to be homogenous to the periodicity of :math:`\mu`. For example, if
                    the input `X` is composed of **daily** returns, the `transaction_costs` need
                    to be expressed as **daily** costs.
                    (See :ref:`sphx_glr_auto_examples_mean_risk_plot_6_transaction_costs.py`)
            management_fees : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
                Management fees of the assets. Fees are modeled as multiplicative
                drags on gross relatives: internally, gradients are computed using
                net-of-fee relatives ``x_eff = x ⊙ (1 - fee)`` (for small fees). This
                aligns with log-wealth objectives and avoids mixing linear and
                multiplicative fee models.

                If a float is provided, it is applied to each asset.
                If a dictionary is provided, its (key/value) pair must be the
                (asset name/asset fee) and the input `X` of the `fit` method must be a
                DataFrame with the assets names in columns.
                The default value is `0.0`.

                .. warning::

                    Based on the above formula, the periodicity of the management fees needs to
                    be homogenous to the periodicity of :math:`\mu`. For example, if the input
                    `X` is composed of **daily** returns, the `management_fees` need to be
                    expressed in **daily** fees.
            warm_start: bool
                Whether to warm start the estimator.
            initial_weights: np.ndarray
                The initial weights to use.
            min_weights : float | dict[str, float] | array-like of shape (n_assets, ) | None, default=0.0
                Minimum assets weights (weights lower bounds).
                Differently from MeanRisk in online setting, set weights must be between 0 and 1.
                If a float is provided, it is applied to each asset.
                `None` is equivalent to `0`.
                If a dictionary is provided, its (key/value) pair must be the
                (asset name/asset minium weight) and the input `X` of the `fit` method must
                be a DataFrame with the assets names in columns.
                When using a dictionary, assets values that are not provided are assigned
                a minimum weight of `0.0`.
                The default value is `0.0` (no short selling).

                Example:

                * `min_weights = 0` --> long only portfolio (no short selling).
                * `min_weights = None` --> 0.
                * `min_weights = {"SX5E": 0.2, "SPX":0.1}`
                * `min_weights = [0.2, 0.3]`

            max_weights : float | dict[str, float] | array-like of shape (n_assets, ) | None, default=1.0
                Maximum assets weights (weights upper bounds).
                Differently from MeanRisk in online setting, set weights must be between 0 and 1.
                If a float is provided, it is applied to each asset.
                `None` is equivalent to 1.
                If a dictionary is provided, its (key/value) pair must be the
                (asset name/asset maximum weight) and the input `X` of the `fit` method must
                be a DataFrame with the assets names in columns.
                When using a dictionary, assets values that are not provided are assigned
                a minimum weight of `1.0`.
                The default value is `1.0` (each asset is below 100%).

                Example:

                * `max_weights = None` --> default to 1
                * `max_weights = {"SX5E": 0.8, "SPX": 0.9}`
                * `max_weights = [0.8, 0.9]`

            budget : float | None, default=1.0
                Investment budget. It is the sum of long positions and short positions (sum of
                all weights). `None` means no budget constraints.
                Budget must be between 0 and 1.
                The default value is `1.0` (fully invested portfolio).

            previous_weights : float | dict[str, float] | array-like of shape (n_assets, ), optional
                Previous weights of the assets. Previous weights are used to compute the
                portfolio cost and the portfolio turnover.
                If a float is provided, it is applied to each asset.
                If a dictionary is provided, its (key/value) pair must be the
                (asset name/asset previous weight) and the input `X` of the `fit` method must
                be a DataFrame with the assets names in columns.
                The default (`None`) means no previous weights.

            max_turnover : float, optional
                Upper bound constraint of the turnover.
                The turnover is defined as the absolute difference between the portfolio weights
                and the `previous_weights`. Note that another way to control for turnover is by
                using the `transaction_costs` parameter.
            groups : dict[str, list[str]] or array-like of shape (n_groups, n_assets), optional
                The assets groups referenced in `linear_constraints`.
                If a dictionary is provided, its (key/value) pair must be the
                (asset name/asset groups) and the input `X` of the `fit` method must be a
                DataFrame with the assets names in columns.

                For example:

                    * `groups = {"SX5E": ["Equity", "Europe"], "SPX": ["Equity", "US"], "TLT": ["Bond", "US"]}`
                    * `groups = [["Equity", "Equity", "Bond"], ["Europe", "US", "US"]]`

            left_inequality : array-like of shape (n_constraints, n_assets), optional
                Left inequality matrix :math:`A` of the linear
                constraint :math:`A \cdot w \leq b`.

            right_inequality : array-like of shape (n_constraints, ), optional
                Right inequality vector :math:`b` of the linear
                constraint :math:`A \cdot w \leq b`.
            linear_constraints : array-like of shape (n_constraints,), optional
                Linear constraints.
                The linear constraints must match any of following patterns:

                * `"2.5 * ref1 + 0.10 * ref2 + 0.0013 <= 2.5 * ref3"`
                * `"ref1 >= 2.9 * ref2"`
                * `"ref1 == ref2"`
                * `"ref1 >= ref1"`

                With `"ref1"`, `"ref2"` ... the assets names or the groups names provided
                in the parameter `groups`. Assets names can be referenced without the need of
                `groups` if the input `X` of the `fit` method is a DataFrame with these
                assets names in columns.

                For example:

                    * `"SPX >= 0.10"` --> SPX weight must be greater than 10% (note that you can also use `min_weights`)
                    * `"SX5E + TLT >= 0.2"` --> the sum of SX5E and TLT weights must be greater than 20%
                    * `"US == 0.7"` --> the sum of all US weights must be equal to 70%
                    * `"Equity == 3 * Bond"` --> the sum of all Equity weights must be equal to 3 times the sum of all Bond weights.
                    * `"2*SPX + 3*Europe <= Bond + 0.05"` --> mixing assets and group constraints
            l2_coef: float
                The L2 regularization coefficient to use.
            smooth_epsilon: float
                Log-barrier regularization parameter for Smooth Prediction method.
                Controls the strength of the regularization term ε * ∑ log(w_i) in the
                FTRL objective. Larger values provide more regularization (smoother weights).
                Typical values: 1.0 (standard), 1/r for known return lower bound r.
                Only used when method=OnlineMethod.SMOOTH_PRED.
            adagrad_D: float or array-like, optional
                Diameter bound(s) for AdaGrad algorithm. Controls the scaling of coordinate-wise adaptive learning rates: η_{t,i} = D_i / (√(∑ g_{j,i}²) + ε). If float, uses same bound for all coordinates. If array-like, should have length equal to problem dimension. If None, defaults to √2 (simplex diameter).
                Only used when objective=OnlineObjective.ADAGRAD.
            adagrad_eps: float
                Small constant added to denominator for numerical stability.
                Set to 0 for truly scale-free updates.
                Only used when objective=OnlineObjective.ADAGRAD.
            covariance: np.ndarray
                The covariance to use.
            variance_bound: float
                The variance bound to use.
            portfolio_params: dict
                The portfolio parameters to use.

        """
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

        # Public configuration
        self.strategy = strategy
        self.learning_rate = learning_rate
        self.ftrl = ftrl
        self.smooth_epsilon = float(smooth_epsilon)
        self.adagrad_D = adagrad_D
        self.adagrad_eps = float(adagrad_eps)
        self.adabarrons_barrier_coef = float(adabarrons_barrier_coef)
        self.adabarrons_alpha = float(adabarrons_alpha)
        self.adabarrons_euclidean_coef = float(adabarrons_euclidean_coef)
        self.adabarrons_beta = float(adabarrons_beta)
        self.warm_start = bool(warm_start)
        self.initial_weights = initial_weights
        self.smooth_prediction = smooth_prediction
        self.eg_tilde = eg_tilde
        self.eg_tilde_alpha = eg_tilde_alpha

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
        self._ftrl_engine: _FTRLEngine | None = None
        self._cumulative_loss: float = 0.0

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
        if self._projector is None:
            self._initialize_projector()

        if self._ftrl_engine is None:
            mirror_map: BaseMirrorMap | None = None
            predictor: Predictor | None = None

            if self.smooth_prediction:
                predictor = LastGradPredictor()

            match self.strategy:
                case FTRLStrategy.EG:
                    mirror_map = EntropyMirrorMap()
                case FTRLStrategy.OGD:
                    mirror_map = EuclideanMirrorMap()
                case FTRLStrategy.ADAGRAD:
                    mirror_map = AdaptiveMahalanobisMap(eps=self.adagrad_eps)
                case FTRLStrategy.ADABARRONS:
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
                case (FTRLStrategy.SWORD_VAR, FTRLStrategy.SWORD):
                    # SWORD-Var: variation-adaptive OMD with optimistic gradients
                    predictor = LastGradPredictor()
                    mirror_map = AdaptiveVariationMap(eps=self.adagrad_eps)
                case FTRLStrategy.SWORD_SMALL:
                    # SWORD-Small: AdaGrad geometry with optimistic gradients
                    predictor = LastGradPredictor()
                    mirror_map = AdaptiveMahalanobisMap(eps=self.adagrad_eps)
                case (FTRLStrategy.SWORD_BEST, FTRLStrategy.SWORD_PP):
                    # Meta-aggregator combining SWORD-Var and SWORD-Small, plus EG for ++
                    predictor_var = LastGradPredictor()
                    predictor_small = LastGradPredictor()
                    var_engine = _FTRLEngine(
                        mirror_map=AdaptiveVariationMap(eps=self.adagrad_eps),
                        projector=IdentityProjector(),  # inner unconstrained
                        eta=self.learning_rate,
                        predictor=predictor_var,
                        mode="omd",
                    )
                    small_engine = _FTRLEngine(
                        mirror_map=AdaptiveMahalanobisMap(eps=self.adagrad_eps),
                        projector=IdentityProjector(),
                        eta=self.learning_rate,
                        predictor=predictor_small,
                        mode="omd",
                    )
                    experts: list[_FTRLEngine] = [var_engine, small_engine]
                case FTRLStrategy.SWORD_PP:
                    # add an entropy-geometry expert to stabilize/explore
                    eg_engine = _FTRLEngine(
                        mirror_map=EntropyMirrorMap(),
                        projector=IdentityProjector(),
                        eta=self.learning_rate,
                        predictor=LastGradPredictor(),
                        mode="omd",
                    )
                    experts.append(eg_engine)
                    self._ftrl_engine = SwordMeta(
                        experts=experts,
                        projector=self._projector,
                        eta_meta=self.learning_rate,  # tie meta-eta to learning_rate
                    )
                case _:
                    raise ValueError("Unknown objective provided.")

            skip_auto_update = self.strategy == FTRLStrategy.ADABARRONS
            self._ftrl_engine = _FTRLEngine(
                mirror_map=mirror_map,
                projector=self._projector,
                eta=self.learning_rate,
                predictor=predictor,
                mode="ftrl" if self.ftrl else "omd",
                skip_auto_update=skip_auto_update,
            )

        if not self._weights_initialized or not self.warm_start:
            self._initialize_weights(num_assets)

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
        # Gradient for log-wealth objective
        denominator = np.dot(self.weights_, effective_relatives)
        if float(denominator) <= 0:
            # Handle non-positive return, e.g. by using a small positive constant
            # or by returning without updating, depending on desired behavior.
            # For now, we skip the update to avoid division by zero.
            warnings.warn(
                "Non-positive portfolio return, skipping update.",
                UserWarning,
                stacklevel=3,
            )
            gradient = np.zeros_like(self.weights_)
        else:
            gradient = -effective_relatives / denominator

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
        if not (self.eg_tilde and self.strategy == FTRLStrategy.EG):
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
        if self.strategy == FTRLStrategy.ADABARRONS and isinstance(
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
        self._cumulative_loss += self.loss_
        self.previous_weights = self.weights_.copy()
        self._t += 1

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> "FTRLProximal":
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

        return self

    def _reset_state_for_fit(self) -> None:
        """Reset internal state when warm_start=False."""
        super()._reset_state_for_fit()
        self._ftrl_engine = None
        self._cumulative_loss = 0.0
