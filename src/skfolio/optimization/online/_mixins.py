from enum import auto
from numbers import Integral, Real
from typing import ClassVar

import numpy as np
from sklearn.utils._param_validation import Interval, StrOptions  # mypy: ignore

from skfolio.utils.tools import AutoEnum


class RegretType(AutoEnum):
    DYNAMIC = auto()
    STATIC = auto()


class UpdateRule(AutoEnum):
    """Update rule for first-order online optimization.

    - EMD: Entropic Mirror Descent (multiplicative weights) - default.
    - OGD: Projected (Euclidean) Gradient Descent.
    - OFW: Online Frank-Wolfe (projection-free on the simplex; projected if needed).
    - ONS: Online Newton Step (Sherman-Morrison inverse update).
    - ADAGRAD: Adaptive Gradient with coordinate-wise adaptive learning rates.
    - ADABARRONS: Adaptive Barrier-Regularized Online Newton Step.
    """

    EMD = auto()
    OGD = auto()
    OFW = auto()
    ONS = auto()
    ADAGRAD = auto()
    ADABARRONS = auto()


class OnlineMethod(AutoEnum):
    # Generic methods (work with any descent algorithm)
    BUY_AND_HOLD = auto()
    EG = auto()  # Standard Hedge/EG - Kelly gradient
    FOLLOW_THE_LEADER = auto()
    FOLLOW_THE_LOSER = auto()
    OLMAR = auto()
    CORN = auto()
    SMOOTH_PRED = auto()

    # Specific algorithms (use integrated descent methods)
    EG_TILDE = auto()  # EG with tilting - requires specialized EMD
    UNIVERSAL = auto()  # Expert mixing - bypasses gradient descent

    # Sword algorithms - smoothness-aware dynamic regret optimization
    SWORD_VAR = auto()  # Sword with gradient variation tracking
    SWORD_SMALL = auto()  # Sword with small-loss tracking
    SWORD_BEST = auto()  # Sword best-of-both-worlds


# Methods that ignore update_rule parameter (algorithm-specific)
ALGORITHM_SPECIFIC_METHODS = {
    OnlineMethod.EG_TILDE,
    OnlineMethod.UNIVERSAL,
    OnlineMethod.SWORD_VAR,
    OnlineMethod.SWORD_SMALL,
    OnlineMethod.SWORD_BEST,
}


class OnlineMixin:
    _history_gross_relatives: list[np.ndarray]  # mypy: ignore
    n_features_in_: int


class OnlineParameterConstraintsMixin:
    _parameter_constraints: ClassVar[dict] = {
        "method": [StrOptions({m.value for m in OnlineMethod})],
        "update_rule": [StrOptions(set(UpdateRule))],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "eta0": [Interval(Real, 0, None, closed="neither")],
        "warm_start": ["boolean"],
        "gamma": [Interval(Real, 0, None, closed="left"), None],
        "l2_coef": [Interval(Real, 0, None, closed="left")],
        "transaction_costs": [
            Interval(Real, 0, None, closed="left"),
            "array-like",
            dict,
            None,
        ],
        "management_fees": [
            Interval(Real, 0, None, closed="left"),
            "array-like",
            dict,
            None,
        ],
        "reversion_window": [Interval(Integral, 1, None, closed="left")],
        "corn_window": [Interval(Integral, 1, None, closed="left")],
        "corn_k": [Interval(Integral, 1, None, closed="left")],
        "min_weights": [Interval(Real, 0, 1, closed="both"), "array-like", None],
        "max_weights": [Interval(Real, 0, 1, closed="both"), "array-like", None],
        "budget": [Interval(Real, 0, 1.0, closed="right"), None],
        "previous_weights": ["array-like", None],
        "max_turnover": [Interval(Real, 0, None, closed="left"), None],
        "groups": ["array-like", dict, None],
        "linear_constraints": [list, None],
        "left_inequality": ["array-like", None],
        "right_inequality": ["array-like", None],
        "X_tracking": ["array-like", None],
        "tracking_error_benchmark": ["array-like", None],
        "max_tracking_error": [Interval(Real, 0, None, closed="neither"), None],
        "covariance": ["array-like", None],
        "variance_bound": [Interval(Real, 0, None, closed="neither"), None],
        "portfolio_params": [dict, None],
        "experts": ["array-like", None],
        "initial_weights": ["array-like", None],
        "universal_grid_step": [Interval(Real, 0, None, closed="neither"), None],
        "universal_n_samples": [Interval(Integral, 1, None, closed="left")],
        "universal_dirichlet_alpha": [
            Interval(Real, 0, None, closed="neither"),
            "array-like",
        ],
        "universal_max_grid_points": [Interval(Integral, 1, None, closed="left")],
        "smooth_epsilon": [Interval(Real, 0, None, closed="neither")],
        "adagrad_D": [Interval(Real, 0, None, closed="neither"), "array-like", None],
        "adagrad_eps": [Interval(Real, 0, 1e-3, closed="both"), None],
    }


class FollowTheLoserMixin:
    reversion_window: int


class CORNMixin:
    corn_window: int
    corn_k: int
