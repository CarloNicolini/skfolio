"""
AntiCor Algorithm Implementation for skfolio

This module implements the AntiCor algorithm from Borodin et al. (2003) 
"Can We Learn to Beat the Best Stock" as a portfolio optimization strategy
compatible with the skfolio library.

The AntiCor algorithm exploits predictable statistical relationships between 
pairs of stocks rather than trying to predict individual stock winners.
It uses correlation analysis over sliding windows to transfer wealth between
anti-correlated assets.

Author: Assistant (based on Borodin, El-Yaniv, Gogan 2003)
"""

import warnings
from typing import Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import check_random_state
import sklearn.utils.validation as check
from numbers import Real
from abc import ABC


class BaseOptimizer(BaseEstimator, ABC):
    """Base class for portfolio optimizers in skfolio-compatible format."""
    
    def __init__(self):
        pass
    
    def _validate_data(self, X, reset=True):
        """Validate input data following sklearn patterns."""
        X = check_array(X, ensure_2d=True, allow_nd=False, dtype='numeric')
        if reset:
            self.n_features_in_ = X.shape[1]
            if hasattr(X, 'columns'):
                self.feature_names_in_ = np.array(X.columns)
        return X
    
    def fit(self, X, y=None):
        """Fit the optimizer. Must be implemented by subclasses."""
        raise NotImplementedError
        
    def predict(self, X):
        """Predict portfolio returns. Basic implementation."""
        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != len(self.weights_):
            raise ValueError(f"X has {X.shape[1]} features, but optimizer was fitted with {len(self.weights_)} features")
        
        # Return portfolio returns: dot product of weights and returns
        return np.dot(X, self.weights_)


class AntiCorOptimizer(BaseOptimizer):
    """
    AntiCor Portfolio Optimization Algorithm.
    
    The AntiCor algorithm exploits statistical relationships between pairs of assets
    by analyzing correlations over sliding time windows. It transfers wealth from
    outperforming assets to anti-correlated underperforming assets.
    
    The algorithm uses two consecutive windows to model cross-correlations between
    assets and makes portfolio adjustments based on predictable relationships.
    
    Parameters
    ----------
    window_size : int, default=10
        Size of each time window for correlation analysis. The algorithm uses
        two consecutive windows of this size.
        
    max_window : int, default=30
        Maximum window size to consider when using multiple windows.
        
    use_multiple_windows : bool, default=True
        If True, use multiple window sizes from 2 to max_window and combine them.
        This provides robustness against parameter selection.
        
    use_nested_anticor : bool, default=False
        If True, apply AntiCor to the outputs of different window-size AntiCor
        algorithms. This is the ANTICOR(ANTICOR) approach from the paper.
        
    min_history : int, default=None
        Minimum history required before making portfolio adjustments. If None,
        set to 4 * window_size.
        
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.
        
    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        The portfolio weights computed during fit.
        
    n_features_in_ : int
        Number of assets seen during fit.
        
    feature_names_in_ : ndarray of shape (n_features_in_,), optional
        Names of assets seen during fit. Only available if input X has feature names.
        
    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> # Load data and convert to returns
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)
    >>> 
    >>> # Fit AntiCor optimizer
    >>> model = AntiCorOptimizer(window_size=15, use_multiple_windows=True)
    >>> model.fit(X_train)
    >>> 
    >>> # Get portfolio weights
    >>> print(model.weights_)
    >>> 
    >>> # Predict returns (if using skfolio's Portfolio evaluation)
    >>> portfolio_returns = model.predict(X_test)
    
    References
    ----------
    Borodin, A., El-Yaniv, R., & Gogan, V. (2003). Can We Learn to Beat the Best Stock. 
    In Advances in Neural Information Processing Systems (pp. 345-352).
    """
    
    def __init__(
        self,
        window_size: int = 10,
        max_window: int = 30, 
        use_multiple_windows: bool = True,
        use_nested_anticor: bool = False,
        min_history: Optional[int] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        super().__init__()
        self.window_size = window_size
        self.max_window = max_window
        self.use_multiple_windows = use_multiple_windows
        self.use_nested_anticor = use_nested_anticor
        self.min_history = min_history
        self.random_state = random_state
        
    def _validate_parameters(self):
        """Validate the algorithm parameters."""
        if self.window_size < 2:
            raise ValueError("window_size must be >= 2")
        if self.max_window < self.window_size:
            raise ValueError("max_window must be >= window_size")
        if self.min_history is not None and self.min_history < 2 * self.window_size:
            raise ValueError("min_history must be >= 2 * window_size")
            
    def _compute_log_returns_matrix(self, returns: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Compute log returns matrix for a given time window.
        
        Parameters
        ----------
        returns : ndarray of shape (n_periods, n_assets)
            Returns data
        start_idx : int
            Start index of the window
        end_idx : int
            End index of the window
            
        Returns
        -------
        log_returns : ndarray of shape (window_size, n_assets)
            Log returns for the specified window
        """
        # Convert returns to price relatives (1 + return) and take log
        price_relatives = 1 + returns[start_idx:end_idx]
        # Clip to avoid log(0) issues
        price_relatives = np.clip(price_relatives, 1e-8, None)
        return np.log(price_relatives)
    
    def _compute_correlation_matrix(self, LX1: np.ndarray, LX2: np.ndarray) -> tuple:
        """
        Compute cross-correlation matrix between two windows.
        
        Parameters
        ----------
        LX1 : ndarray of shape (window_size, n_assets)
            Log returns for first window
        LX2 : ndarray of shape (window_size, n_assets) 
            Log returns for second window
            
        Returns
        -------
        M_cor : ndarray of shape (n_assets, n_assets)
            Cross-correlation matrix
        mu1, mu2 : ndarray
            Mean vectors for each window
        """
        n_assets = LX1.shape[1]
        
        # Compute means and standard deviations
        mu1 = np.mean(LX1, axis=0)
        mu2 = np.mean(LX2, axis=0)
        std1 = np.std(LX1, axis=0, ddof=1)
        std2 = np.std(LX2, axis=0, ddof=1)
        
        # Initialize correlation matrix
        M_cor = np.zeros((n_assets, n_assets))
        
        # Compute cross-correlations
        for i in range(n_assets):
            for j in range(n_assets):
                if std1[i] > 1e-8 and std2[j] > 1e-8:
                    # Cross-covariance
                    cov_ij = np.mean((LX1[:, i] - mu1[i]) * (LX2[:, j] - mu2[j]))
                    # Cross-correlation  
                    M_cor[i, j] = cov_ij / (std1[i] * std2[j])
                else:
                    M_cor[i, j] = 0.0
                    
        return M_cor, mu1, mu2
    
    def _compute_claims(self, M_cor: np.ndarray, mu2: np.ndarray) -> np.ndarray:
        """
        Compute transfer claims between assets.
        
        Parameters
        ----------
        M_cor : ndarray of shape (n_assets, n_assets)
            Cross-correlation matrix
        mu2 : ndarray
            Mean returns for second window
            
        Returns
        -------
        claims : ndarray of shape (n_assets, n_assets)
            Claims matrix where claims[i,j] is the claim from asset i to asset j
        """
        n_assets = M_cor.shape[0]
        claims = np.zeros((n_assets, n_assets))
        
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j and mu2[i] > mu2[j] and M_cor[i, j] > 0:
                    # A(h) term: add absolute autocorrelation if negative
                    A_i = abs(M_cor[i, i]) if M_cor[i, i] < 0 else 0
                    A_j = abs(M_cor[j, j]) if M_cor[j, j] < 0 else 0
                    
                    claims[i, j] = M_cor[i, j] + A_i + A_j
                    
        return claims
        
    def _update_portfolio(self, current_weights: np.ndarray, claims: np.ndarray) -> np.ndarray:
        """
        Update portfolio weights based on transfer claims.
        
        Parameters
        ----------
        current_weights : ndarray of shape (n_assets,)
            Current portfolio weights
        claims : ndarray of shape (n_assets, n_assets)
            Claims matrix
            
        Returns
        -------
        new_weights : ndarray of shape (n_assets,)
            Updated portfolio weights
        """
        n_assets = len(current_weights)
        new_weights = current_weights.copy()
        
        for i in range(n_assets):
            # Outgoing transfers from asset i
            total_claim_from_i = np.sum(claims[i, :])
            
            # Incoming transfers to asset i  
            total_claim_to_i = np.sum(claims[:, i])
            
            if total_claim_from_i > 0:
                for j in range(n_assets):
                    if claims[i, j] > 0:
                        # Transfer from i to j
                        transfer_amount = current_weights[i] * claims[i, j] / total_claim_from_i
                        new_weights[i] -= transfer_amount
                        new_weights[j] += transfer_amount
                        
        # Ensure weights sum to 1 and are non-negative
        new_weights = np.maximum(new_weights, 0)
        weight_sum = np.sum(new_weights)
        if weight_sum > 0:
            new_weights /= weight_sum
        else:
            # Fallback to uniform weights
            new_weights = np.ones(n_assets) / n_assets
            
        return new_weights
    
    def _anticor_single_window(self, returns: np.ndarray, window_size: int) -> np.ndarray:
        """
        Run AntiCor algorithm with a single window size.
        
        Parameters
        ----------
        returns : ndarray of shape (n_periods, n_assets)
            Returns data
        window_size : int
            Size of correlation window
            
        Returns
        -------
        weights : ndarray of shape (n_assets,)
            Final portfolio weights
        """
        n_periods, n_assets = returns.shape
        min_periods = self.min_history or (4 * window_size)
        
        if n_periods < min_periods:
            # Not enough data - return uniform weights
            return np.ones(n_assets) / n_assets
        
        # Initialize with uniform portfolio
        weights = np.ones(n_assets) / n_assets
        
        # Run AntiCor algorithm day by day
        for t in range(2 * window_size, n_periods):
            # Define two windows
            window1_start = t - 2 * window_size
            window1_end = t - window_size
            window2_start = t - window_size  
            window2_end = t
            
            # Compute log returns for both windows
            LX1 = self._compute_log_returns_matrix(returns, window1_start, window1_end)
            LX2 = self._compute_log_returns_matrix(returns, window2_start, window2_end)
            
            # Compute cross-correlation matrix
            M_cor, mu1, mu2 = self._compute_correlation_matrix(LX1, LX2)
            
            # Compute transfer claims
            claims = self._compute_claims(M_cor, mu2)
            
            # Update portfolio weights
            weights = self._update_portfolio(weights, claims)
            
        return weights
    
    def _anticor_multiple_windows(self, returns: np.ndarray) -> np.ndarray:
        """
        Run AntiCor with multiple window sizes and combine results.
        
        This implements the BAHW(ANTICOR) approach from the paper.
        
        Parameters
        ----------
        returns : ndarray of shape (n_periods, n_assets)
            Returns data
            
        Returns
        -------
        weights : ndarray of shape (n_assets,)
            Combined portfolio weights
        """
        window_sizes = range(2, self.max_window + 1)
        all_weights = []
        
        # Compute weights for each window size
        for w in window_sizes:
            try:
                weights_w = self._anticor_single_window(returns, w)
                all_weights.append(weights_w)
            except Exception as e:
                warnings.warn(f"Failed to compute AntiCor for window size {w}: {e}")
                continue
                
        if not all_weights:
            # Fallback to uniform weights
            return np.ones(returns.shape[1]) / returns.shape[1]
        
        # Uniform combination of all window-size results
        combined_weights = np.mean(all_weights, axis=0)
        
        # Normalize to sum to 1
        combined_weights = combined_weights / np.sum(combined_weights)
        
        return combined_weights
    
    def _anticor_nested(self, returns: np.ndarray) -> np.ndarray:
        """
        Run nested AntiCor (AntiCor of AntiCor results).
        
        This implements the BAH30(ANTICOR(ANTICOR)) approach from the paper.
        
        Parameters
        ----------
        returns : ndarray of shape (n_periods, n_assets)
            Returns data
            
        Returns  
        -------
        weights : ndarray of shape (n_assets,)
            Final portfolio weights from nested approach
        """
        window_sizes = range(2, self.max_window + 1)
        
        # First level: compute returns for each window-size AntiCor
        anticor_returns = []
        anticor_weights = []
        
        n_periods = len(returns)
        min_periods = self.min_history or (4 * self.max_window)
        
        if n_periods < min_periods:
            return np.ones(returns.shape[1]) / returns.shape[1]
        
        for w in window_sizes:
            try:
                # Simulate online AntiCor for this window size
                weights_series = []
                weights = np.ones(returns.shape[1]) / returns.shape[1]
                
                for t in range(2 * w, n_periods):
                    # Update weights at time t
                    window1_start = t - 2 * w
                    window1_end = t - w
                    window2_start = t - w
                    window2_end = t
                    
                    if window1_start >= 0:
                        LX1 = self._compute_log_returns_matrix(returns, window1_start, window1_end)
                        LX2 = self._compute_log_returns_matrix(returns, window2_start, window2_end)
                        M_cor, mu1, mu2 = self._compute_correlation_matrix(LX1, LX2)
                        claims = self._compute_claims(M_cor, mu2)
                        weights = self._update_portfolio(weights, claims)
                    
                    weights_series.append(weights.copy())
                
                # Compute cumulative returns for this AntiCor variant
                if len(weights_series) > 0:
                    weights_array = np.array(weights_series)
                    # Use weights from t-1 to compute returns at t
                    portfolio_returns = []
                    for i, w_t in enumerate(weights_array):
                        t_actual = 2 * w + i
                        if t_actual < n_periods:
                            ret = np.dot(w_t, returns[t_actual])
                            portfolio_returns.append(ret)
                    
                    if portfolio_returns:
                        anticor_returns.append(portfolio_returns)
                        anticor_weights.append(weights_series[-1])  # Final weights
                        
            except Exception as e:
                warnings.warn(f"Failed nested AntiCor for window {w}: {e}")
                continue
        
        if not anticor_returns:
            return np.ones(returns.shape[1]) / returns.shape[1]
        
        # Second level: Apply AntiCor to the AntiCor returns
        if len(anticor_returns) < 2:
            # Not enough algorithms for nesting, return combined weights
            return np.mean(anticor_weights, axis=0)
        
        # Create matrix of AntiCor algorithm returns
        min_length = min(len(ret_series) for ret_series in anticor_returns)
        anticor_returns_matrix = np.array([ret_series[:min_length] for ret_series in anticor_returns]).T
        
        # Apply AntiCor to these algorithm returns
        nested_weights = self._anticor_single_window(anticor_returns_matrix, self.window_size)
        
        # Final portfolio weights are weighted combination of individual AntiCor weights
        final_weights = np.zeros(returns.shape[1])
        for i, alg_weight in enumerate(nested_weights):
            if i < len(anticor_weights):
                final_weights += alg_weight * anticor_weights[i]
                
        # Normalize
        final_weights = final_weights / np.sum(final_weights)
        
        return final_weights
    
    def fit(self, X, y=None):
        """
        Fit the AntiCor optimizer.
        
        Parameters
        ----------
        X : array-like of shape (n_periods, n_assets)
            Asset returns data. Each row represents a time period and each
            column represents an asset.
        y : None
            Ignored. Present for API consistency.
            
        Returns
        -------
        self : AntiCorOptimizer
            Fitted estimator instance.
        """
        # Validate parameters
        self._validate_parameters()
        
        # Validate and store input data
        X = self._validate_data(X, reset=True)
        
        # Ensure we have enough data
        min_periods_required = 2 * self.max_window if self.use_multiple_windows else 2 * self.window_size
        if X.shape[0] < min_periods_required:
            raise ValueError(f"Need at least {min_periods_required} periods of data, got {X.shape[0]}")
        
        # Run appropriate AntiCor variant
        if self.use_nested_anticor:
            self.weights_ = self._anticor_nested(X)
        elif self.use_multiple_windows:
            self.weights_ = self._anticor_multiple_windows(X)  
        else:
            self.weights_ = self._anticor_single_window(X, self.window_size)
        
        # Ensure weights are properly normalized
        if np.sum(self.weights_) <= 0:
            warnings.warn("All weights are zero or negative. Using uniform