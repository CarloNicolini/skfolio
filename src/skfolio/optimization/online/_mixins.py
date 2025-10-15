from enum import auto
from numbers import Real
from typing import ClassVar

from sklearn.utils._param_validation import Interval, StrOptions

from skfolio.utils.tools import AutoEnum


class RegretType(AutoEnum):
    STATIC = auto()
    DYNAMIC = auto()
    DYNAMIC_UNIVERSAL = auto()
    DYNAMIC_WORST_CASE = auto()
    DYNAMIC_LEGACY = auto()


class FTWStrategy(AutoEnum):
    OGD = auto()
    EG = auto()
    ADAGRAD = auto()
    ADABARRONS = auto()
    SWORD_SMALL = auto()
    SWORD_VAR = auto()
    SWORD_BEST = auto()
    SWORD_PP = auto()
    PROD = auto()  # Soft-Bayes Prod algorithm


class FTLStrategy(AutoEnum):
    """
    Follow-The-Loser families (mean-reversion).

    - OLMAR: Online Moving Average Reversion
    - PAMR: Passive-Aggressive Mean Reversion
    - CWMR: Confidence-Weighted Mean Reversion (distributional, second-order)
    - RMR: Robust Median Reversion (L1-median, outlier-robust)
    """

    OLMAR = auto()
    PAMR = auto()
    CWMR = auto()
    RMR = auto()


class PAMRVariant(AutoEnum):
    """Variants for Passive-Aggressive Mean Reversion (PAMR).

    - SIMPLE: the original PAMR formula with τ = loss / ||c||²
    - SLACK_LINEAR: introduces a linear slack variable ξ with weight C (PAMR-1)
                    τ = min(C, loss / ||c||²)
    - SLACK_QUADRATIC: quadratic slack regularisation ξ² with weight C (PAMR-2)
                       τ = loss / (||c||² + 1/(2C))
    See Li & Hoi book (2012), Equations 9.2, 9.3, 9.4.
    """

    SIMPLE = auto()  # PAMR (original)
    SLACK_LINEAR = auto()  # PAMR-1
    SLACK_QUADRATIC = auto()  # PAMR-2


class UpdateMode(AutoEnum):
    """Update mode for mean-reversion strategies."""

    PA = auto()  # Passive-Aggressive (closed-form)
    MD = auto()  # Mirror Descent (OCO-style)


class OLMARPredictor(AutoEnum):
    """Predictor for OLMAR strategy.

    Two variants are implemented from the Li and Hoi "Online Portfolio Selection" book:
    - MAR-1 (SMA) Simple Moving Average Eq. 11.1 ("1" version in OLMAR)
    - MAR-2 (EWMA) Exponentially Weighted Moving Average Eq. 11.2 ("2" version in OLMAR)
    """

    SMA = "sma"
    EWMA = "ewma"  # Exponentially Weighted Moving Average


class OnlineMixin:
    n_features_in_: int


class OnlineParameterConstraintsMixin:
    _parameter_constraints: ClassVar[dict] = {
        "learning_rate": [
            Interval(Real, 0, None, closed="neither"),
            callable,
            StrOptions({"auto"}),
        ],
        "warm_start": ["boolean"],
        "initial_weights": ["array-like", None],
        "initial_wealth": [Interval(Real, 0, None, closed="neither"), None],
        "previous_weights": ["array-like", None],
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
        "adagrad_eps": [Interval(Real, 1e-12, 1e-3, closed="both"), None],
        "eg_tilde": ["boolean"],
        "eg_tilde_alpha": [Interval(Real, 0, 1, closed="both"), callable, None],
    }
