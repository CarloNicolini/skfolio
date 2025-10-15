"""Autograd-compatible objective functions for online portfolio optimization.

This module provides automatic and analytical differentiation for risk and
performance measures in the Online Convex Optimization (OCO) framework.

OCO-Compatible Measures
------------------------
The following measures are convex (or exp-concave) in portfolio weights and
suitable for online optimization with regret guarantees:

**Strongly Convex (O(log T) regret with appropriate step sizes)**:
- Variance (quadratic)
- Log-Wealth (exp-concave)

**Convex (O(√T) regret)**:
- Standard Deviation
- Semi-Variance, Semi-Deviation
- Mean Absolute Deviation
- First Lower Partial Moment
- CVaR (Conditional Value at Risk)
- EVaR (Entropic Value at Risk, also exp-concave)
- Worst Realization
- Gini Mean Difference
- CDaR (Conditional Drawdown at Risk)
- EDaR (Entropic Drawdown at Risk, also exp-concave)

**Linear (O(√T) regret)**:
- Mean (expected return)

Non-Convex Measures (Not Included)
-----------------------------------
The following measures are NOT included because they are non-convex and
violate OCO assumptions:
- Value at Risk (VaR): quantile function, non-convex
- Max Drawdown: max operator over path-dependent values, non-convex
- Average Drawdown: path-dependent, non-convex
- Ulcer Index: involves squared drawdown, non-convex

For these measures, standard OCO regret bounds do not hold.

Analytical vs Autograd Gradients
---------------------------------
Where available, analytical gradients are significantly faster than autograd.
The following measures have analytical gradient implementations:
- Log-Wealth: -r / (1 + w^T r)
- Mean: mean(X)
- Variance: 2Σw / T (requires sample covariance)
- Semi-Variance: closed-form with indicators
- Mean Absolute Deviation: closed-form with signs
- First Lower Partial Moment: closed-form with indicators
- Worst Realization: one-hot gradient at worst asset
- Gini Mean Difference: OWA weights

References
----------
See OCO/gradients_risk_metrics.md for detailed mathematical formulations.
"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np
from autograd import grad

from skfolio.measures import (
    cdar,
    cvar,
    edar,
    evar,
    first_lower_partial_moment,
    gini_mean_difference,
    mean,
    mean_absolute_deviation,
    semi_deviation,
    semi_variance,
    standard_deviation,
    variance,
    worst_realization,
)
from skfolio.measures._enums import BaseMeasure, PerfMeasure, RiskMeasure


class BaseObjective(ABC):
    """Base class for all objective functions in online portfolio optimization.

    Parameters
    ----------
    use_autograd : bool, default=True
        If True, use automatic differentiation to compute gradients.
        If False, use analytical gradients (when available).
    """

    def __init__(self, use_autograd: bool = True):
        self.use_autograd = use_autograd

    @abstractmethod
    def loss(self, weights: np.ndarray, net_returns: np.ndarray, **kwargs) -> float:
        """Compute the loss value.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        net_returns : ndarray of shape (n_assets,) or (n_observations, n_assets)
            Net returns of assets.
        **kwargs
            Additional measure-specific parameters (e.g., beta for CVaR).

        Returns
        -------
        float
            Loss value.
        """
        pass

    @abstractmethod
    def grad(
        self, weights: np.ndarray, net_returns: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Compute the gradient of the loss.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        net_returns : ndarray of shape (n_assets,) or (n_observations, n_assets)
            Net returns of assets.
        **kwargs
            Additional measure-specific parameters (e.g., beta for CVaR).

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        pass


# =============================================================================
# Helper functions for analytical gradients
# =============================================================================


def _compute_sample_covariance(net_returns: np.ndarray) -> np.ndarray:
    """Compute sample covariance matrix from net returns.

    Parameters
    ----------
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.

    Returns
    -------
    ndarray of shape (n_assets, n_assets)
        Sample covariance matrix.
    """
    T = net_returns.shape[0]
    mean_returns = np.mean(net_returns, axis=0, keepdims=True)
    centered = net_returns - mean_returns
    return (centered.T @ centered) / T


def _analytical_log_wealth_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs
) -> np.ndarray:
    """Analytical gradient of log-wealth: ∇_w (-log(1 + w^T r)) = -r / (1 + w^T r).

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_assets,)
        Net returns of assets (single observation).
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    portfolio_net_return = np.dot(weights, net_returns)
    gross_return = 1.0 + portfolio_net_return
    if gross_return <= 0:
        warnings.warn(
            "Non-positive portfolio return, returning zero gradient.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros_like(weights)
    return -net_returns / gross_return


def _analytical_mean_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs
) -> np.ndarray:
    """Analytical gradient of mean (expected return).

    Mean gradient: ∇_w mean = mean(X)
    where X is the returns matrix.

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights (unused, included for signature consistency).
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    # Gradient of mean(w^T r) w.r.t. w is mean(r)
    return np.mean(net_returns, axis=0)


def _analytical_variance_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs
) -> np.ndarray:
    """Analytical gradient of variance: ∇_w var = 2Σw.

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    Sigma = _compute_sample_covariance(net_returns)
    return 2.0 * (Sigma @ weights)


def _analytical_semi_variance_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs
) -> np.ndarray:
    """Analytical gradient of semi-variance.

    Semi-variance gradient: ∇_w SV = (2/T) Σ_{t: r_t^T w < τ} (r_t^T w - τ) r_t

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Additional parameters (min_acceptable_return).

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    min_acceptable_return = kwargs.get("min_acceptable_return", 0.0)
    T = net_returns.shape[0]
    portfolio_returns = net_returns @ weights
    # Indicator for returns below threshold
    below_threshold = portfolio_returns < min_acceptable_return
    # Gradient: (2/T) sum over bad outcomes
    grad = np.zeros_like(weights)
    if np.any(below_threshold):
        bad_returns = net_returns[below_threshold]
        bad_portfolio_returns = portfolio_returns[below_threshold]
        # Vectorized computation
        deviations = bad_portfolio_returns - min_acceptable_return
        grad = 2.0 * np.sum(deviations[:, np.newaxis] * bad_returns, axis=0) / T
    return grad


def _analytical_mad_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs
) -> np.ndarray:
    """Analytical gradient of Mean Absolute Deviation.

    MAD gradient: ∇_w MAD = (1/T) Σ_t sign(r_t^T w - μ) r_t
    where μ = mean(r^T w)

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    T = net_returns.shape[0]
    portfolio_returns = net_returns @ weights
    mean_return = np.mean(portfolio_returns)
    deviations = portfolio_returns - mean_return
    signs = np.sign(deviations)
    # Gradient: (1/T) Σ sign(deviation) * r_t
    grad = (net_returns.T @ signs) / T
    return grad


def _analytical_flpm_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs
) -> np.ndarray:
    """Analytical gradient of First Lower Partial Moment.

    FLPM gradient: ∇_w FLPM = -(1/T) Σ_{t: r_t^T w < τ} r_t

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Additional parameters (min_acceptable_return).

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    min_acceptable_return = kwargs.get("min_acceptable_return", 0.0)
    T = net_returns.shape[0]
    portfolio_returns = net_returns @ weights
    below_threshold = portfolio_returns < min_acceptable_return
    grad = np.zeros_like(weights)
    if np.any(below_threshold):
        # Sum returns where portfolio return is below threshold
        grad = -np.sum(net_returns[below_threshold], axis=0) / T
    return grad


def _analytical_worst_realization_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs
) -> np.ndarray:
    """Analytical gradient of Worst Realization (negative minimum return).

    Worst gradient is one-hot at the index of the worst portfolio return.
    ∇_w (-min r^T w) = -r_{worst}

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector (one-hot at worst observation).
    """
    portfolio_returns = net_returns @ weights
    worst_idx = np.argmin(portfolio_returns)
    # Gradient of -min is negative of the return vector at worst time
    return -net_returns[worst_idx]


def _analytical_gini_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs
) -> np.ndarray:
    """Analytical gradient of Gini Mean Difference.

    GMD gradient uses OWA (Ordered Weighted Averaging) weights.
    ∇_w GMD = Σ_t w_π(t) r_t
    where π is the sorting permutation and w_k = (2k - T - 1)/T

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    T = net_returns.shape[0]
    portfolio_returns = net_returns @ weights
    # Sort returns and get permutation
    sort_idx = np.argsort(portfolio_returns)
    # OWA weights: w_k = (2k - T - 1) / T for k = 1, ..., T
    owa_weights = (2.0 * np.arange(1, T + 1) - T - 1) / T
    # Apply OWA weights to sorted returns
    sorted_returns = net_returns[sort_idx]
    grad = sorted_returns.T @ owa_weights
    return grad


# Measure properties including convexity info and measure functions from skfolio.measures
MEASURE_PROPERTIES = {
    # Special measure for log-wealth (default objective)
    PerfMeasure.LOG_WEALTH: {
        "convexity": "exp-concave",
        "sign": -1,  # maximize wealth = minimize negative log
        "notes": "1-exp-concave, O(log T) regret. Analytical gradient available.",
    },
    # Performance measures (maximize = minimize negative)
    PerfMeasure.MEAN: {
        "measure_func": mean,
        "convexity": "linear",
        "sign": -1,  # maximize
        "notes": "Linear, smooth. Analytical gradient available.",
    },
    # Risk measures (minimize)
    RiskMeasure.VARIANCE: {
        "measure_func": variance,
        "convexity": "strongly_convex",
        "sign": 1,  # minimize
        "notes": "Strongly convex, quadratic. Analytical gradient available (requires Σ).",
    },
    RiskMeasure.STANDARD_DEVIATION: {
        "measure_func": standard_deviation,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, smooth. Uses autograd (square root complicates analytical form).",
    },
    RiskMeasure.SEMI_VARIANCE: {
        "measure_func": semi_variance,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Analytical gradient available.",
    },
    RiskMeasure.SEMI_DEVIATION: {
        "measure_func": semi_deviation,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Uses autograd.",
    },
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION: {
        "measure_func": mean_absolute_deviation,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, L1 norm, non-smooth. Analytical gradient available.",
    },
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT: {
        "measure_func": first_lower_partial_moment,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Analytical gradient available.",
    },
    RiskMeasure.CVAR: {
        "measure_func": cvar,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Uses autograd.",
    },
    RiskMeasure.EVAR: {
        "measure_func": evar,
        "convexity": "exp-concave",
        "sign": 1,
        "notes": "Convex, exp-concave, smooth. Uses autograd.",
    },
    RiskMeasure.WORST_REALIZATION: {
        "measure_func": worst_realization,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth (max operator). Analytical gradient available (one-hot).",
    },
    RiskMeasure.GINI_MEAN_DIFFERENCE: {
        "measure_func": gini_mean_difference,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Analytical gradient available (OWA).",
    },
    RiskMeasure.CDAR: {
        "measure_func": cdar,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Uses autograd.",
    },
    RiskMeasure.EDAR: {
        "measure_func": edar,
        "convexity": "exp-concave",
        "sign": 1,
        "notes": "Convex, exp-concave, smooth. Uses autograd.",
    },
}


@dataclass
class MeasureObjective(BaseObjective):
    """Unified objective for all measures (risk, performance, and log-wealth).

    Parameters
    ----------
    measure : RiskMeasure | PerfMeasure | str | None
        The measure to optimize. If None, uses log-wealth objective.
    use_autograd : bool, default=True
        Whether to use autograd for gradient computation. If False, uses analytical
        gradients when available (faster but limited to certain measures).
    risk_free_rate : float, default=0.0
        Risk-free rate (used for some measures).
    min_acceptable_return : float, default=0.0
        Minimum acceptable return threshold (used for semi-variance, FLPM, etc.).
    cvar_beta : float, default=0.95
        Confidence level for CVaR.
    evar_beta : float, default=0.95
        Confidence level for EVaR.
    cdar_beta : float, default=0.95
        Confidence level for CDaR.
    edar_beta : float, default=0.95
        Confidence level for EDaR.
    """

    measure: RiskMeasure | PerfMeasure | str | None = None
    use_autograd: bool = True
    risk_free_rate: float = 0.0
    min_acceptable_return: float = 0.0
    cvar_beta: float = 0.95
    evar_beta: float = 0.95
    cdar_beta: float = 0.95
    edar_beta: float = 0.95

    def __post_init__(self):
        # Default to log-wealth if measure is None
        if self.measure is None:
            self.measure = PerfMeasure.LOG_WEALTH

        # Get measure properties
        if self.measure not in MEASURE_PROPERTIES:
            raise ValueError(f"Unknown measure: {self.measure}")

        self.properties = MEASURE_PROPERTIES[self.measure]
        self._sign = self.properties["sign"]

        # Create gradient function
        if self.use_autograd:
            self._grad_fn = grad(self._loss_fn, argnum=0)
        else:
            # Use analytical gradient
            self._grad_fn = self._get_analytical_grad_fn()
            if self._grad_fn is None:
                raise NotImplementedError(
                    f"Analytical gradient not implemented for {self.measure}. "
                    "Set `use_autograd=True` or choose a different measure."
                )

    def _loss_fn(self, weights, net_returns, **kwargs):
        """Autograd-compatible loss computation.

        Parameters
        ----------
        weights : array-like
            Portfolio weights.
        net_returns : array-like
            Net returns of assets (not price relatives). Can be:
            - 1D array of shape (n_assets,) for single observation
            - 2D array of shape (n_observations, n_assets) for time series
        **kwargs
            Additional parameters (e.g., beta for CVaR, min_acceptable_return, etc.).
        """
        if self.measure == PerfMeasure.LOG_WEALTH:
            # Log-wealth: -log(1 + w^T r_t) for single period
            # where r_t are net returns
            portfolio_net_return = anp.dot(weights, net_returns)
            return -anp.log(1.0 + portfolio_net_return)
        else:
            # Compute portfolio returns using autograd-compatible operations
            # If net_returns is 1D (single observation), portfolio_returns is scalar
            # If net_returns is 2D (time series), portfolio_returns is 1D
            if net_returns.ndim == 1:
                # Single observation: net_returns shape (n_assets,)
                portfolio_returns = anp.dot(weights, net_returns)
                # For single observations, use simple proxies
                return self._compute_measure_single(portfolio_returns, **kwargs)
            else:
                # Time series: net_returns shape (n_observations, n_assets)
                portfolio_returns = anp.dot(net_returns, weights)
                return self._compute_measure_timeseries(portfolio_returns, **kwargs)

    def _compute_measure_single(self, portfolio_return, **kwargs):
        """Compute measure for single observation using autograd-compatible operations.

        For single observations, we use simple approximations since measures
        like variance are undefined for a single sample.
        """
        min_acceptable_return = kwargs.get(
            "min_acceptable_return", self.min_acceptable_return
        )

        match self.measure:
            case PerfMeasure.MEAN:
                return self._sign * portfolio_return
            case RiskMeasure.VARIANCE:
                # For single observation, use squared return as proxy
                return self._sign * portfolio_return**2
            case RiskMeasure.STANDARD_DEVIATION:
                return self._sign * anp.abs(portfolio_return)
            case RiskMeasure.SEMI_VARIANCE:
                # Downside deviation squared
                downside = anp.maximum(0, min_acceptable_return - portfolio_return)
                return self._sign * downside**2
            case RiskMeasure.SEMI_DEVIATION:
                downside = anp.maximum(0, min_acceptable_return - portfolio_return)
                return self._sign * downside
            case RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
                return self._sign * anp.abs(portfolio_return - min_acceptable_return)
            case RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT:
                downside = anp.maximum(0, min_acceptable_return - portfolio_return)
                return self._sign * downside
            case RiskMeasure.WORST_REALIZATION:
                return self._sign * (-portfolio_return)
            case _:
                # For other measures that need time series, use absolute value as fallback
                return self._sign * anp.abs(portfolio_return)

    def _compute_measure_timeseries(self, portfolio_returns, **kwargs):
        """Compute measure for time series using autograd-compatible operations."""
        min_acceptable_return = kwargs.get(
            "min_acceptable_return", self.min_acceptable_return
        )
        beta = kwargs.get("beta", self.cvar_beta)

        match self.measure:
            case PerfMeasure.MEAN:
                return self._sign * anp.mean(portfolio_returns)
            case RiskMeasure.VARIANCE:
                # Use unbiased variance (ddof=1 equivalent)
                mean_ret = anp.mean(portfolio_returns)
                return self._sign * anp.mean((portfolio_returns - mean_ret) ** 2)
            case RiskMeasure.STANDARD_DEVIATION:
                mean_ret = anp.mean(portfolio_returns)
                var = anp.mean((portfolio_returns - mean_ret) ** 2)
                return self._sign * anp.sqrt(var + 1e-10)
            case RiskMeasure.SEMI_VARIANCE:
                downside = anp.maximum(0, min_acceptable_return - portfolio_returns)
                return self._sign * anp.mean(downside**2)
            case RiskMeasure.SEMI_DEVIATION:
                downside = anp.maximum(0, min_acceptable_return - portfolio_returns)
                return self._sign * anp.sqrt(anp.mean(downside**2) + 1e-10)
            case RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
                mean_ret = anp.mean(portfolio_returns)
                return self._sign * anp.mean(anp.abs(portfolio_returns - mean_ret))
            case RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT:
                downside = anp.maximum(0, min_acceptable_return - portfolio_returns)
                return self._sign * anp.mean(downside)
            case RiskMeasure.CVAR:
                # CVaR: mean of worst (1-beta) losses
                k = int((1 - beta) * len(portfolio_returns))
                if k == 0:
                    k = 1
                # Sort in descending order of losses (ascending returns)
                sorted_returns = anp.sort(portfolio_returns)
                return self._sign * (-anp.mean(sorted_returns[:k]))
            case RiskMeasure.EVAR:
                # EVAR: use the measure function directly (complex formula)
                # For time series, approximate via CVaR-like computation
                k = int((1 - beta) * len(portfolio_returns))
                if k == 0:
                    k = 1
                sorted_returns = anp.sort(portfolio_returns)
                return self._sign * (-anp.mean(sorted_returns[:k]))
            case RiskMeasure.WORST_REALIZATION:
                return self._sign * (-anp.min(portfolio_returns))
            case RiskMeasure.GINI_MEAN_DIFFERENCE:
                n = len(portfolio_returns)
                sorted_rets = anp.sort(portfolio_returns)
                # OWA weights for Gini
                weights_owa = (2.0 * anp.arange(1, n + 1) - n - 1) / n
                return self._sign * anp.dot(weights_owa, sorted_rets)
            case RiskMeasure.CDAR | RiskMeasure.EDAR:
                # Compute drawdowns
                cumulative_returns = anp.cumsum(portfolio_returns)
                running_max = anp.maximum.accumulate(cumulative_returns)
                drawdowns = running_max - cumulative_returns

                if self.measure == RiskMeasure.CDAR:
                    # CDaR: mean of worst (1-beta) drawdowns
                    k = int((1 - beta) * len(drawdowns))
                    if k == 0:
                        k = 1
                    sorted_dd = anp.sort(drawdowns)[::-1]  # descending
                    return self._sign * anp.mean(sorted_dd[:k])
                elif self.measure == RiskMeasure.EDAR:
                    # EDAR: exponentially weighted worst drawdown (simplified)
                    return self._sign * anp.max(drawdowns)
            case _:
                raise ValueError(f"Unsupported measure: {self.measure}")

    def _get_analytical_grad_fn(self):
        """Get analytical gradient function for the measure.

        Returns
        -------
        callable or None
            Analytical gradient function if available, None otherwise.
        """
        # Get the base gradient function and wrap it with sign multiplication
        match self.measure:
            case PerfMeasure.LOG_WEALTH:
                # Log-wealth gradient already has the sign baked in (negative)
                return _analytical_log_wealth_gradient

            case PerfMeasure.MEAN:
                base_grad_fn = _analytical_mean_gradient

            case RiskMeasure.VARIANCE:
                base_grad_fn = _analytical_variance_gradient

            case RiskMeasure.SEMI_VARIANCE:

                def semi_var_grad(weights, net_returns, **kwargs):
                    mar = kwargs.get(
                        "min_acceptable_return", self.min_acceptable_return
                    )
                    return _analytical_semi_variance_gradient(
                        weights, net_returns, min_acceptable_return=mar
                    )

                base_grad_fn = semi_var_grad

            case RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
                base_grad_fn = _analytical_mad_gradient

            case RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT:

                def flpm_grad(weights, net_returns, **kwargs):
                    mar = kwargs.get(
                        "min_acceptable_return", self.min_acceptable_return
                    )
                    return _analytical_flpm_gradient(
                        weights, net_returns, min_acceptable_return=mar
                    )

                base_grad_fn = flpm_grad

            case RiskMeasure.WORST_REALIZATION:
                base_grad_fn = _analytical_worst_realization_gradient

            case RiskMeasure.GINI_MEAN_DIFFERENCE:
                base_grad_fn = _analytical_gini_gradient

            case _:
                # No analytical gradient available for this measure
                return None

        # Wrap with sign multiplication to match loss function
        def signed_grad_fn(weights, net_returns, **kwargs):
            return self._sign * base_grad_fn(weights, net_returns, **kwargs)

        return signed_grad_fn

    def loss(self, weights: np.ndarray, net_returns: np.ndarray, **kwargs) -> float:
        """Compute loss value.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        net_returns : ndarray of shape (n_assets,) or (n_observations, n_assets)
            Net returns of assets.
        **kwargs
            Additional measure-specific parameters (e.g., beta=0.95 for CVaR).

        Returns
        -------
        float
            Loss value.
        """
        return float(self._loss_fn(weights, net_returns, **kwargs))

    def grad(
        self, weights: np.ndarray, net_returns: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Compute gradient of the loss.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        net_returns : ndarray of shape (n_assets,) or (n_observations, n_assets)
            Net returns of assets.
        **kwargs
            Additional measure-specific parameters (e.g., beta=0.95 for CVaR).

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        return self._grad_fn(weights, net_returns, **kwargs)


@dataclass
class MeanReversionHinge(BaseObjective):
    """Hinge loss for mean reversion: L(w) = max(0, ε - φ^T w).

    This loss encourages the portfolio to satisfy φ^T w ≥ ε, where φ is a
    reversion predictor (e.g., inverse moving average for OLMAR or current
    price relatives for PAMR).

    Parameters
    ----------
    epsilon : float
        Margin threshold. The loss is zero when φ^T w ≥ ε.
    use_autograd : bool, default=False
        If True, verify analytical gradient against autograd.

    Notes
    -----
    Gradient: ∇_w L(w) = -φ if φ^T w < ε, else 0 (subgradient at equality).
    The loss is convex and piecewise linear.
    """

    epsilon: float
    use_autograd: bool = False

    def loss(self, weights: np.ndarray, phi: np.ndarray, **kwargs) -> float:
        """Compute hinge loss value.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor (fixed during gradient computation).
        **kwargs
            Unused, included for signature consistency.

        Returns
        -------
        float
            Loss value.
        """
        margin = self.epsilon - np.dot(weights, phi)
        return float(max(0.0, margin))

    def grad(self, weights: np.ndarray, phi: np.ndarray, **kwargs) -> np.ndarray:
        """Compute hinge loss gradient w.r.t. weights.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor (fixed).
        **kwargs
            Unused, included for signature consistency.

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        margin = self.epsilon - np.dot(weights, phi)
        if margin > 0.0:
            return -phi
        else:
            return np.zeros_like(phi)


@dataclass
class MeanReversionSquaredHinge(BaseObjective):
    """Squared hinge loss: L(w) = [max(0, ε - φ^T w)]².

    Smooth variant of hinge loss with Lipschitz continuous gradient.

    Parameters
    ----------
    epsilon : float
        Margin threshold.
    use_autograd : bool, default=False
        If True, verify analytical gradient against autograd.

    Notes
    -----
    Gradient: ∇_w L(w) = -2(ε - φ^T w)φ if φ^T w < ε, else 0.
    The loss is convex and differentiable everywhere (smooth).
    """

    epsilon: float
    use_autograd: bool = False

    def loss(self, weights: np.ndarray, phi: np.ndarray, **kwargs) -> float:
        """Compute squared hinge loss value.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor.
        **kwargs
            Unused.

        Returns
        -------
        float
            Loss value.
        """
        margin = self.epsilon - np.dot(weights, phi)
        return float(max(0.0, margin) ** 2)

    def grad(self, weights: np.ndarray, phi: np.ndarray, **kwargs) -> np.ndarray:
        """Compute squared hinge gradient w.r.t. weights.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor.
        **kwargs
            Unused.

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        margin = self.epsilon - np.dot(weights, phi)
        if margin > 0.0:
            return -2.0 * margin * phi
        else:
            return np.zeros_like(phi)


@dataclass
class MeanReversionSoftplus(BaseObjective):
    """Softplus loss: L(w) = (1/β) log(1 + exp(β(ε - φ^T w))).

    Smooth approximation to hinge loss. As β → ∞, approaches hinge loss.

    Parameters
    ----------
    epsilon : float
        Margin threshold.
    beta : float, default=5.0
        Temperature parameter. Larger values → sharper hinge approximation.
    use_autograd : bool, default=False
        If True, verify analytical gradient against autograd.

    Notes
    -----
    Gradient: ∇_w L(w) = -σ(β(ε - φ^T w))φ where σ is the sigmoid function.
    The loss is convex, infinitely differentiable, and exp-concave for bounded β.
    Admits O(log T) regret in OCO with appropriate algorithms.
    """

    epsilon: float
    beta: float = 5.0
    use_autograd: bool = False

    def loss(self, weights: np.ndarray, phi: np.ndarray, **kwargs) -> float:
        """Compute softplus loss value.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor.
        **kwargs
            Unused.

        Returns
        -------
        float
            Loss value.
        """
        margin = self.epsilon - np.dot(weights, phi)
        # Use log1p for numerical stability when beta*margin is small
        return float(np.log(1.0 + np.exp(self.beta * margin)) / self.beta)

    def grad(self, weights: np.ndarray, phi: np.ndarray, **kwargs) -> np.ndarray:
        """Compute softplus gradient w.r.t. weights.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor.
        **kwargs
            Unused.

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        margin = self.epsilon - np.dot(weights, phi)
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        sigmoid = 1.0 / (1.0 + np.exp(-self.beta * margin))
        return -sigmoid * phi


def create_mean_reversion_objective(
    objective_type: str,
    epsilon: float,
    beta: float = 5.0,
    use_autograd: bool = False,
) -> BaseObjective:
    """Factory for mean-reversion objectives.

    Creates objective functions for Follow-the-Loser (mean-reversion) strategies
    like OLMAR, PAMR, and CWMR. These objectives measure how well portfolio weights
    satisfy a mean-reversion constraint.

    Parameters
    ----------
    objective_type : str
        One of "hinge", "squared_hinge", "softplus".

        - "hinge": Piecewise linear, non-smooth at kink
        - "squared_hinge": Smooth, Lipschitz gradient
        - "softplus": Infinitely smooth, exp-concave

    epsilon : float
        Margin threshold. Loss is zero when φ^T w ≥ ε.

    beta : float, default=5.0
        Temperature parameter for softplus. Only used when objective_type="softplus".
        Larger values approximate hinge more closely.

    use_autograd : bool, default=False
        If True, verify analytical gradient against autograd (for testing).

    Returns
    -------
    BaseObjective
        Configured mean-reversion objective with analytical gradients.

    Examples
    --------
    >>> # Hinge loss for PAMR
    >>> obj = create_mean_reversion_objective("hinge", epsilon=1.0)
    >>> weights = np.array([0.5, 0.5])
    >>> phi = np.array([1.1, 0.9])  # Price relatives
    >>> loss_val = obj.loss(weights, phi)
    >>> gradient = obj.grad(weights, phi)

    >>> # Smooth softplus for OLMAR
    >>> obj = create_mean_reversion_objective("softplus", epsilon=2.0, beta=10.0)
    """
    match objective_type.lower():
        case "hinge":
            return MeanReversionHinge(epsilon=epsilon, use_autograd=use_autograd)
        case "squared_hinge":
            return MeanReversionSquaredHinge(epsilon=epsilon, use_autograd=use_autograd)
        case "softplus":
            return MeanReversionSoftplus(
                epsilon=epsilon, beta=beta, use_autograd=use_autograd
            )
        case _:
            raise ValueError(
                f"Unknown mean-reversion objective: {objective_type}. "
                f"Must be one of: 'hinge', 'squared_hinge', 'softplus'"
            )


def create_objective(
    measure: BaseMeasure | None,
    use_autograd: bool = True,
    **kwargs,
) -> BaseObjective:
    """Factory function to create objective from measure enum.

    Parameters
    ----------
    measure : BaseMeasure or None
        Measure enum (RiskMeasure, ExtraRiskMeasure, or PerfMeasure).
        If None, uses log-wealth (default).
    use_autograd : bool, default=True
        Whether to use autograd for gradients.
    **kwargs
        Additional parameters passed to MeasureObjective.

    Returns
    -------
    BaseObjective
        Configured objective function.

    Examples
    --------
    >>> # Default log-wealth
    >>> obj = create_objective(None)

    >>> # CVaR with analytical gradient (not available, will fall back to autograd)
    >>> obj = create_objective(RiskMeasure.CVAR, use_autograd=False, cvar_beta=0.95)

    >>> # Variance with autograd
    >>> obj = create_objective(RiskMeasure.VARIANCE)
    """
    return MeasureObjective(measure=measure, use_autograd=use_autograd, **kwargs)
