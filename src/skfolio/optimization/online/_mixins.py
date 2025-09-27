from enum import auto
from numbers import Integral, Real
from typing import ClassVar

from sklearn.utils._param_validation import Interval, StrOptions  # mypy: ignore

from skfolio.utils.tools import AutoEnum


class RegretType(AutoEnum):
    DYNAMIC = auto()
    STATIC = auto()


class OnlineFamily(AutoEnum):
    OGD = auto()
    EG = auto()
    FOLLOW_THE_LEADER = auto()
    FOLLOW_THE_LOSER = auto()
    SMOOTH_PRED = auto()
    ADAGRAD = auto()
    ADABARRONS = auto()  # https://arxiv.org/pdf/2202.07574
    # Sword algorithms - smoothness-aware dynamic regret optimization
    SWORD_SMALL = auto()
    SWORD_VAR = auto()
    SWORD_BEST = auto()
    SWORD_PP = auto()
    SWORD = SWORD_VAR


class LoserStrategy(AutoEnum):
    """
    Follow-the-Loser families (mean-reversion):
    - OLMAR: Online Moving Average Reversion (1/2)
    - PAMR: Passive-Aggressive Mean Reversion
    - CWMR: Confidence-Weighted Mean Reversion (distributional, second-order)
    """

    OLMAR = auto()
    PAMR = auto()
    CWMR = auto()


class OLMARVariant(AutoEnum):
    OLPS = auto()
    CUMPROD = auto()


class OnlineMixin:
    n_features_in_: int


class OnlineParameterConstraintsMixin:
    _parameter_constraints: ClassVar[dict] = {
        "objective": [StrOptions({m.value for m in OnlineFamily})],
        "ftrl": ["boolean"],
        "batch_size": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither"), callable],
        "warm_start": ["boolean"],
        "initial_weights": ["array-like", None],
        "previous_weights": ["array-like", None],
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
        "min_weights": [Interval(Real, 0, 1, closed="both"), "array-like", None],
        "max_weights": [Interval(Real, 0, 1, closed="both"), "array-like", None],
        "budget": [Interval(Real, 0, 1.0, closed="right"), None],
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
        "smooth_epsilon": [Interval(Real, 0, None, closed="neither")],
        "adagrad_D": [Interval(Real, 0, None, closed="neither"), "array-like", None],
        "adagrad_eps": [Interval(Real, 0, 1e-3, closed="both"), None],
        "eg_tilde": ["boolean"],
        "eg_tilde_alpha": [Interval(Real, 0, 1, closed="both"), callable, None],
    }
