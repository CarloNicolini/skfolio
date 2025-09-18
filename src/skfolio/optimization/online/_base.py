import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import _check_sample_weight, validate_data

import skfolio.typing as skt
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.online._descent import (
    AdaGrad,
    AdaBARRONS,
    BaseDescent,
    EGTildeEntropicMirrorDescent,
    EntropicMirrorDescent,
    OnlineFrankWolfe,
    OnlineGradientDescent,
    OnlineNewtonStep,
    SwordEntropicMirrorDescent,
)
from skfolio.optimization.online._loss import CLIP_EPSILON, Loss, losses_map
from skfolio.optimization.online._mixins import (
    ALGORITHM_SPECIFIC_METHODS,
    OnlineMethod,
    OnlineMixin,
    OnlineParameterConstraintsMixin,
    UpdateRule,
)
from skfolio.optimization.online._projection import AutoProjector, ProjectionConfig
from skfolio.optimization.online._utils import net_to_relatives
from skfolio.utils.tools import input_to_array


class OPS(BaseOptimization, OnlineMixin, OnlineParameterConstraintsMixin):
    """First-order Online Portfolio Selection with projection.

    This estimator implements standard online convex optimization updates:
    - EMD: Entropic Mirror Descent (multiplicative weights) - default
    - OGD: Projected (Euclidean) Gradient Descent
    - OFW: Online Frank-Wolfe (projection-free on the simplex; projected if needed)
    - ONS: Online Newton Step (Sherman-Morrison inverse update).

    Methods (loss/gradient providers):

    **Generic methods** (work with any update_rule):
    - BUY_AND_HOLD: no update
    - EG: Kelly gradient on current gross returns (standard Hedge/EG)
    - FOLLOW_THE_LEADER: cumulative Kelly gradients (entropy geometry)
    - FOLLOW_THE_LOSER: Kelly + PA hinge toward a reversion feature
    - OLMAR: Kelly gradient on a moving-average prediction
    - CORN: scenario-based Kelly gradient over correlation-selected neighbors
    - SMOOTH_PRED: Kelly gradient + log-barrier regularization (FTRL variant)

    **Algorithm-specific methods** (use integrated descent, ignore update_rule):
    - EG_TILDE: Exponentiated Gradient with tilting - uses specialized EMD with
      time-varying parameters and portfolio mixing
    - UNIVERSAL: Expert aggregation - bypasses gradient-based updates entirely
    - SWORD_VAR: Sword with gradient variation tracking - adaptive EMD using
      accumulated gradient variation V_T for O(√((1+P_T+V_T)(1+P_T))) dynamic regret
    - SWORD_SMALL: Sword with small-loss tracking - adaptive EMD using cumulative
      comparator loss F_T for O(√((1+P_T+F_T)(1+P_T))) dynamic regret
    - SWORD_BEST: Sword best-of-both-worlds - adaptive EMD using min{V_T,F_T}
      for O(√((1+P_T+min{V_T,F_T})(1+P_T))) dynamic regret


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
        method: OnlineMethod = OnlineMethod.EG,
        update_rule: UpdateRule = UpdateRule.EMD,
        *,
        initial_weights: npt.ArrayLike | None = None,
        eta0: float = 0.05,
        warm_start: bool = True,
        gamma: float | None = None,
        l2_coef: float = 0.0,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        previous_weights: skt.MultiInput | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        # Algorihm specific parameters
        ## Follow the loser
        reversion_window: int = 5,
        ## CORN
        corn_window: int = 5,
        corn_k: int = 5,
        ## Smooth Prediction
        smooth_epsilon: float = 1.0,
        ## AdaGrad
        adagrad_D: float | npt.ArrayLike | None = None,
        adagrad_eps: float = 1e-8,
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
        # Universal provider options
        experts: np.ndarray | None = None,
        universal_grid_step: float | None = None,
        universal_n_samples: int = 10_000,
        universal_dirichlet_alpha: float | npt.ArrayLike = 1.0,
        universal_max_grid_points: int = 1_000_000,
    ):
        r"""
        The Online Portfolio Selection estimator.

        Parameters
        ----------
            method: OnlineMethod
                The online method to use.
            update_rule: UpdateRule
                The update rule to use.
            eta0: float
                Initial learning-rate.

                - EMD/OGD: the actual step-size defaults to ``eta_t = eta_t / sqrt(t)``
                  following Hazan's OGD/MD schedule (ensures ``O(\sqrt{T})`` regret for
                  convex losses). For exp-concave losses you may keep a constant small
                  ``eta``.
                - ONS: the effective step-size is chosen automatically as
                  ``eta_eff = 1 / gamma`` with ``gamma = 0.5 / (G D)`` where ``G`` is an
                  online estimate of the gradient norm upper bound and ``D`` is the
                  Euclidean diameter (default ``sqrt(2)`` for the simplex). This follows
                  Hazan's guidance for logarithmic regret up to the unknown
                  exp-concavity.
                - UNIVERSAL: for expert mixtures (Hedge/EWOO), ``eta`` controls how fast
                  expert weights adapt; ``eta=1`` approximates wealth weighting.
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

            gamma: float
                The gamma parameter to use for the Online Newton Step
            l2_coef: float
                The L2 regularization coefficient to use.
            reversion_window: int
                The reversion window to use. Applies only to FOLLOW_THE_LOSER method.
            corn_window: int
                The corn window to use. Applies only to CORN method.
            corn_k: int
                The corn k parameter to use. Applies only to CORN method.
            smooth_epsilon: float
                Log-barrier regularization parameter for Smooth Prediction method.
                Controls the strength of the regularization term ε * ∑ log(w_i) in the
                FTRL objective. Larger values provide more regularization (smoother weights).
                Typical values: 1.0 (standard), 1/r for known return lower bound r.
                Only used when method=OnlineMethod.SMOOTH_PRED.
            adagrad_D: float or array-like, optional
                Diameter bound(s) for AdaGrad algorithm. Controls the scaling of coordinate-wise
                adaptive learning rates: η_{t,i} = D_i / (√(∑ g_{j,i}²) + ε).
                If float, uses same bound for all coordinates. If array-like, should have
                length equal to problem dimension. If None, defaults to √2 (simplex diameter).
                Only used when update_rule=UpdateRule.ADAGRAD.
            adagrad_eps: float
                Small constant added to denominator for numerical stability.
                Set to 0 for truly scale-free updates.
                Only used when update_rule=UpdateRule.ADAGRAD.
            experts: np.ndarray
                Optional expert matrix of shape ``(n_assets, n_experts)``. Each
                column is a valid portfolio on the simplex (nonnegative, sums to 1).
                If ``None``, experts are generated automatically using either a
                simplex grid (when small enough) or Dirichlet sampling (see
                ``universal_grid_step``, ``universal_n_samples``,
                ``universal_dirichlet_alpha``).
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

            X_tracking: np.ndarray
                The tracking error benchmark to use.
            tracking_error_benchmark: np.ndarray
                The tracking error benchmark to use.
            max_tracking_error : float, optional
                Upper bound constraint on the tracking error.
                The tracking error is defined as the RMSE (root-mean-square error) of the
                portfolio returns compared to a target returns. If `max_tracking_error` is
                provided, the target returns `y` must be provided in the `fit` method.
            covariance: np.ndarray
                The covariance to use.
            variance_bound: float
                The variance bound to use.
            portfolio_params: dict
                The portfolio parameters to use.
            universal_grid_step: float
                If provided, attempts to discretize the simplex with grid step
                ``h`` such that the number of grid points is
                ``comb(round(1/h)+n_assets-1, n_assets-1)``. If this count exceeds
                ``universal_max_grid_points``, the implementation falls back to
                Dirichlet sampling (see below). Use small ``h`` only for small
                universes (few assets).
            universal_n_samples: int
                Number of Dirichlet samples used to generate experts when a grid is
                not used. Increase to improve coverage of the simplex for larger
                universes; trades off memory/compute.
            universal_dirichlet_alpha: float
                Dirichlet concentration. ``1.0`` draws are uniform over the simplex;
                values < 1 concentrate mass near the corners (extreme allocations); values > 1
                concentrate near the center (balanced allocations).
            universal_max_grid_points: int
                Maximum number of grid points allowed before falling back to
                sampling when ``universal_grid_step`` is specified.

        Notes
        -----
        - FTL vs MD: the provider for ``FTL`` returns the instantaneous gradient;
            cumulative effects are produced by the mirror descent dynamics. This avoids
            double-counting that would occur if cumulative gradients were fed to MD.
        - OLMAR: this implementation uses a Kelly-style convex surrogate at the
            forecast vector rather than the original PA constraint. Behavior is
            similar in spirit but not identical to Li & Hoi (2012).
        - Tracking error: the TE constraint in projection uses sample standard
            deviation of active returns (centered residuals) divided by sqrt(T-1).
        """
        super().__init__(portfolio_params=portfolio_params)

        # Validate method-descent compatibility
        if method in ALGORITHM_SPECIFIC_METHODS and update_rule != UpdateRule.EMD:
            import warnings

            warnings.warn(
                f"Method {method.value} uses its own integrated descent algorithm "
                f"and ignores update_rule parameter (specified: {update_rule.value}). "
                f"This is correct behavior - {method.value} requires specific update rules.",
                UserWarning,
                stacklevel=2,
            )

        # Public configuration
        self.method = method
        self.update_rule = update_rule
        self.eta0 = float(eta0)
        self.gamma = None if gamma is None else float(gamma)
        self.l2_coef = float(l2_coef)
        self.reversion_window = int(reversion_window)
        self.corn_window = int(corn_window)
        self.corn_k = int(corn_k)
        self.smooth_epsilon = float(smooth_epsilon)
        self.adagrad_D = adagrad_D
        self.adagrad_eps = float(adagrad_eps)
        self.experts = experts
        self.warm_start = bool(warm_start)
        self.initial_weights = (
            None
            if initial_weights is None
            else np.asarray(initial_weights, dtype=float)
        )

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

        # Universal options
        self.universal_grid_step = universal_grid_step
        self.universal_n_samples = int(universal_n_samples)
        self.universal_dirichlet_alpha = universal_dirichlet_alpha
        self.universal_max_grid_points = int(universal_max_grid_points)

        # Internal state (initialized deterministically)
        self.t_: int = 0
        self._history_gross_relatives: list[np.ndarray] = []
        self._descent: BaseDescent | None = None
        self._loss_method: Loss | None = None
        self._projector: Any | None = None
        # Initialization flags to avoid hasattr/getattr checks
        self._is_initialized: bool = False
        self._weights_initialized: bool = False
        # ONS numerical parameters
        self.eps = 1e-3
        self.jitter = 1e-9
        self.recompute_every = 1000

    def _clean_input(
        self,
        value: float | dict | npt.ArrayLike | None,
        n_assets: int,
        fill_value: Any,
        name: str,
    ) -> float | np.ndarray:
        """Convert input to cleaned float or ndarray.

        Parameters
        ----------
        value : float, dict, array-like or None.
            Input value to clean.

        n_assets : int
            Number of assets. Used to verify the shape of the converted array.

        fill_value : Any
            When `items` is a dictionary, elements that are not in `asset_names` are
            filled with `fill_value` in the converted array.

        name : str
            Name used for error messages.

        Returns
        -------
        value :  float or ndarray of shape (n_assets,)
            The cleaned float or 1D array.
        """
        if value is None:
            return fill_value
        if np.isscalar(value):
            return float(value)  # type: ignore[arg-type]
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
        if (self._projector is None) or (self._descent is None):
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

            # Special case for EG_TILDE: always use specialized EMD regardless of update_rule
            if self.method == OnlineMethod.EG_TILDE:
                self._descent = EGTildeEntropicMirrorDescent(
                    self._projector, self, eta=self.eta0
                )
            # Special cases for Sword algorithms: always use specialized EMD with adaptive step sizes
            elif self.method == OnlineMethod.SWORD_VAR:
                self._descent = SwordEntropicMirrorDescent(
                    self._projector, self, eta=self.eta0, sword_variant="var"
                )
            elif self.method == OnlineMethod.SWORD_SMALL:
                self._descent = SwordEntropicMirrorDescent(
                    self._projector, self, eta=self.eta0, sword_variant="small"
                )
            elif self.method == OnlineMethod.SWORD_BEST:
                self._descent = SwordEntropicMirrorDescent(
                    self._projector, self, eta=self.eta0, sword_variant="best"
                )
            elif self.update_rule == UpdateRule.EMD:
                self._descent = EntropicMirrorDescent(
                    self._projector, eta=self.eta0, use_schedule=True
                )
            elif self.update_rule == UpdateRule.OGD:
                self._descent = OnlineGradientDescent(
                    self._projector, eta=self.eta0, use_schedule=True
                )
            elif self.update_rule == UpdateRule.OFW:
                self._descent = OnlineFrankWolfe(
                    self._projector, gamma=self.gamma if self.gamma is not None else 0.0
                )
            elif self.update_rule == UpdateRule.ONS:
                self._descent = OnlineNewtonStep(
                    self._projector,
                    eta=self.eta0,
                    eps=self.eps,
                    jitter=self.jitter,
                    recompute_every=self.recompute_every,
                    auto_eta=True,
                )
            elif self.update_rule == UpdateRule.ADAGRAD:
                self._descent = AdaGrad(
                    self._projector,
                    D=self.adagrad_D,
                    eps=self.adagrad_eps,
                )
            elif self.update_rule == UpdateRule.ADABARRONS:
                self._descent = AdaBARRONS(
                    self._projector,
                    eta=self.eta0,
                    eps=self.adagrad_eps,
                    jitter=self.jitter,
                    # lambda_init=self.lambda_init,
                    # backtracking=self.backtracking,
                    # max_backtrack=self.max_backtrack,
                )
            else:
                raise ValueError("Unknown update_rule provided.")

        if self._loss_method is None:
            if self.method not in losses_map:
                raise ValueError("Unknown method provided.")

            self._loss_method = losses_map[self.method](self)  # type: ignore[abstract]

        if (not self._weights_initialized) or (not self.warm_start):
            if self.initial_weights is not None:
                initial = np.asarray(self.initial_weights, dtype=float)
                if initial.shape != (num_assets,):
                    raise ValueError("initial_weights has incompatible shape")
                self.weights_ = initial
            else:
                self.weights_ = np.ones(num_assets, dtype=float) / float(num_assets)
            self._weights_initialized = True

        # Mark overall init complete
        self._is_initialized = True

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> "OPS":
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
            ensure_2d=True,
            allow_nd=False,
            accept_sparse=False,
        )

        # Handle sample_weight if provided (though it's ignored)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            warnings.warn(
                "sample_weight is ignored in OPS.partial_fit (online convex optimization).",
                UserWarning,
                stacklevel=2,
            )

        # Convert to proper shape for partial_fit (single row)
        net_returns = np.asarray(X, dtype=float)
        if net_returns.ndim == 2 and net_returns.shape[0] > 1:
            raise ValueError(
                "partial_fit expects a single row (one period). Use fit for multiple rows."
            )

        gross_relatives = net_to_relatives(net_returns).squeeze()
        self._ensure_initialized(gross_relatives)
        self.t_ += 1

        # Apply management fees multiplicatively to gross relatives for Kelly-like gradients
        effective_relatives = np.maximum(
            gross_relatives * (1 - self._management_fees_arr), CLIP_EPSILON
        )

        # Method-specific provider may directly update weights (e.g. UNIVERSAL)
        if self._loss_method is None:
            raise RuntimeError("Provider not initialized")

        # UNIVERSAL method bypasses gradient-based updates
        if self.method == OnlineMethod.UNIVERSAL:
            if hasattr(self._loss_method, "update_weights"):
                new_weights = self._loss_method.update_weights(
                    self.weights_, effective_relatives
                )
                if new_weights is not None:
                    self.weights_ = new_weights
        else:
            gradient = self._loss_method.gradient(self.weights_, effective_relatives)
            # Add L1 turnover subgradient for transaction costs: grad C_t(b) = c * sign(b - b_prev)
            if not self.transaction_costs and self.previous_weights is not None:
                prev = np.asarray(self.previous_weights, dtype=float)
                if prev.shape == self.weights_.shape:
                    delta = self.weights_ - prev
                    gradient += self._transaction_costs_arr * np.sign(delta)

            if self._descent:
                y = self._descent.step(self.weights_, gradient)
                self.weights_ = self._descent.project(y)
            else:
                raise RuntimeError("Descent not initialized")

        self._history_gross_relatives.append(gross_relatives)
        self.previous_weights = self.weights_.copy()
        self.loss_ = self._loss_method.loss(self.weights_, effective_relatives)
        return self

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> "OPS":
        """Iterate over rows and call :meth:`partial_fit` for each period.

        In OCO, ``fit`` is a convenience wrapper. It does not aggregate gradients or
        perform multi-epoch training. Each row of ``X`` is processed once in order.

        Parameters
        ----------
        X : array-like of shape (T, n_assets) or (n_assets,)
            Net returns per period.
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
        # Validate parameters
        self._validate_params()

        if not self.warm_start:
            # Reset online state when warm_start=False
            self.t_ = 0
            self._history_gross_relatives = []
            # Do not assign None to typed attributes; just mark flags and reset internals
            self._weights_initialized = False
            self._is_initialized = False
            self._descent = None
            self._projector = None
            self._loss_method = None

        # Handle sample_weight if provided (though it's ignored)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            warnings.warn(
                "sample_weight is ignored in OPS.fit (online convex optimization).",
                UserWarning,
                stacklevel=2,
            )
        self.all_weights_ = np.vstack(
            [
                self.partial_fit(
                    row[None, :], y, sample_weight=None, **fit_params
                ).weights_
                for row in np.asarray(X)
            ]
        )
        return self
