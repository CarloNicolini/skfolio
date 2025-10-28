import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import _check_sample_weight, validate_data

import skfolio.typing as skt
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.online._mixins import (
    OnlineMixin,
    OnlineParameterConstraintsMixin,
)
from skfolio.optimization.online._projection import (
    AutoProjector,
    ProjectionConfig,
)
from skfolio.optimization.online._utils import net_to_relatives
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
        initial_wealth: float | npt.ArrayLike | None = None,
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
        self.initial_wealth = initial_wealth
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
        self._wealth_initialized: bool = False
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
        return AutoProjector(projection_config)

    def _initialize_weights(self, num_assets: int):
        if self.initial_weights is not None:
            initial = np.asarray(self.initial_weights, dtype=float)
            if initial.shape != (num_assets,):
                raise ValueError("initial_weights has incompatible shape")
            self.weights_ = initial
        else:
            self.weights_ = np.ones(num_assets, dtype=float) / float(num_assets)
        self._weights_initialized = True

    def _initialize_wealth(self, num_assets: int) -> None:
        """Initialize wealth tracking based on initial_wealth parameter.

        Handles three cases:
        1. Array initial_wealth → compute initial_weights if not provided
        2. Scalar initial_wealth → use as starting wealth
        3. No initial_wealth → default to 1.0

        Raises
        ------
        ValueError
            If both initial_weights and initial_wealth are arrays (incompatible).
        """
        # Case: Both arrays → error
        if self.initial_weights is not None and self.initial_wealth is not None:
            iw_arr = np.asarray(self.initial_weights, dtype=float)
            ih_arr = np.asarray(self.initial_wealth, dtype=float)
            if iw_arr.ndim > 0 and ih_arr.ndim > 0:
                raise ValueError(
                    "initial_weights and initial_wealth cannot both be arrays. "
                    "Provide either initial_weights (for allocation) or "
                    "initial_wealth (for dollar amounts), not both."
                )

        # Case: Array initial_wealth
        if self.initial_wealth is not None:
            ih_arr = np.asarray(self.initial_wealth, dtype=float)
            if ih_arr.ndim > 0:  # Array
                if ih_arr.shape != (num_assets,):
                    raise ValueError(
                        f"initial_wealth array must have shape ({num_assets},), "
                        f"got {ih_arr.shape}"
                    )
                total = float(np.sum(ih_arr))
                if total <= 0:
                    raise ValueError("Sum of initial_wealth must be positive")

                # Set wealth
                self.wealth_ = total

                # Compute weights if not provided
                if not self._weights_initialized:
                    budget = self.budget if self.budget is not None else 1.0
                    self.weights_ = (ih_arr / total) * budget
                    self._weights_initialized = True
            else:  # Scalar
                self.wealth_ = float(self.initial_wealth)
                if self.wealth_ <= 0:
                    raise ValueError("initial_wealth must be positive")
        else:
            # Default: unit wealth
            self.wealth_ = 1.0

        self._wealth_initialized = True

    def _update_wealth(
        self,
        trade_weights: np.ndarray,
        effective_relatives: np.ndarray,
        previous_weights: np.ndarray | None,
        *,
        drift_aware: bool = True,
    ) -> None:
        """Update wealth after observing period returns.

        Computes net portfolio return accounting for:
        - Gross return: trade_weights^T * effective_relatives
        - Transaction costs: sum(c_i * |trade_weights_i - previous_weights_i|)

        Parameters
        ----------
        trade_weights : np.ndarray
            Weights used for trading (before update).
        effective_relatives : np.ndarray
            Price relatives after management fees.
        previous_weights : np.ndarray | None
            Weights from previous period (for turnover).
        """
        # Gross portfolio return (after management fees)
        gross_return = float(np.dot(trade_weights, effective_relatives))

        # Transaction costs (proportional to turnover)
        txn_cost = 0.0
        if previous_weights is not None and hasattr(self, "_transaction_costs_arr"):
            prev_arr = np.asarray(previous_weights, dtype=float)
            if prev_arr.shape == trade_weights.shape:
                # Drift-aware previous holdings: \tilde w_{t-1} = (w_{t-1} ⊙ x_t) / (w_{t-1}^T x_t)
                if drift_aware:
                    denom = float(np.dot(prev_arr, effective_relatives))
                    if denom <= 0:
                        denom = 1e-16
                    prev_drifted = (prev_arr * effective_relatives) / denom
                    turnover = np.abs(trade_weights - prev_drifted)
                else:
                    turnover = np.abs(trade_weights - prev_arr)
                # Total cost as fraction of portfolio
                if np.isscalar(self._transaction_costs_arr):
                    txn_cost = float(self._transaction_costs_arr * np.sum(turnover))
                else:
                    txn_cost = float(np.dot(self._transaction_costs_arr, turnover))

        # Net return after costs
        net_return = gross_return - txn_cost

        # Update wealth: W_{t+1} = W_t * net_return
        self.wealth_ *= max(net_return, 1e-16)  # Prevent negative/zero wealth

    def _compute_effective_relatives(self, gross_relatives: np.ndarray) -> np.ndarray:
        """Apply management fees to gross relatives.

        Parameters
        ----------
        gross_relatives : np.ndarray
            Gross price relatives for one period (shape: (n_assets,)).

        Returns
        -------
        np.ndarray
            Effective relatives after applying management fees multiplicatively.
            If no fees configured, returns gross_relatives unchanged.

        Notes
        -----
        Management fees are applied as multiplicative discount:
        effective_relative_i = gross_relative_i * (1 - fee_i)
        """
        if not hasattr(self, "_management_fees_arr"):
            return gross_relatives

        # If fees are scalar zero, no adjustment needed
        if np.isscalar(self._management_fees_arr):
            if self._management_fees_arr <= 0:
                return gross_relatives
            return gross_relatives * (1.0 - self._management_fees_arr)

        # Array of fees: element-wise multiplication
        return gross_relatives * (
            1.0 - np.asarray(self._management_fees_arr, dtype=float)
        )

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
        self._wealth_initialized = False
        self._is_initialized = False
        self._projector = None
        self._t = 0
        self._last_trade_weights_ = None
        # Note: wealth_ is a learned attribute, only created after fit/partial_fit

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

        # Initialize wealth before loop if needed (creates wealth_ attribute)
        if X_arr.shape[0] > 0 and not self._wealth_initialized:
            num_assets = X_arr.shape[1]
            if not self._weights_initialized:
                self._initialize_weights(num_assets)
            self._initialize_wealth(num_assets)

        # Start wealth tracking list (wealth_ now exists as a learned attribute)
        wealth_list: list[float] = [self.wealth_] if hasattr(self, "wealth_") else []

        for t in range(X_arr.shape[0]):
            self.partial_fit(X_arr[t][None, :], y, sample_weight=None, **fit_params)
            if self._last_trade_weights_ is not None:
                trade_list.append(self._last_trade_weights_.copy())
            # Append wealth after each period (wealth_ created in first partial_fit if not already)
            if hasattr(self, "wealth_"):
                wealth_list.append(self.wealth_)

        if trade_list:
            self.all_weights_ = np.vstack(trade_list)

        # Store wealth history (only if wealth tracking was enabled)
        if wealth_list:
            self.all_wealth_ = np.array(wealth_list, dtype=float)

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
