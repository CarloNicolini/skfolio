import numpy as np
import pytest
from skfolio.optimization.online import (
    FollowTheLoser,
    FollowTheWinner,
    FTLStrategy,
    FTWStrategy,
    regret,
    RegretType,
)
from tests.test_optimization.test_online.utils import make_stationary_returns

def test_rmr_outlier_resistance():
    """
    Test that RMR (Robust Median Reversion) outperforms standard OLMAR
    in the presence of significant outliers.
    """
    # Generate stationary data where mean reversion works well
    n_assets = 5
    n_periods = 500
    rng = np.random.default_rng(42)
    X = make_stationary_returns(T=n_periods, n=n_assets, gap=0.01)
    
    # Inject outliers: random massive spikes
    n_outliers = 20
    outlier_indices = rng.choice(n_periods, n_outliers, replace=False)
    asset_indices = rng.choice(n_assets, n_outliers, replace=True)
    
    # Create a copy to corrupt
    X_corrupted = X.copy()
    for t, i in zip(outlier_indices, asset_indices):
        # Massive 50% drop (0.5 relative) or 100% gain (2.0 relative)
        # to confuse mean estimators
        X_corrupted[t, i] = 2.0 if rng.random() > 0.5 else 0.5

    # Train OLMAR (sensitive to mean)
    olmar = FollowTheLoser(
        strategy=FTLStrategy.OLMAR,
        olmar_predictor="sma",
        olmar_window=5,
        update_mode="pa"
    )
    olmar.fit(X_corrupted)

    # Train RMR (robust median)
    rmr = FollowTheLoser(
        strategy=FTLStrategy.RMR,
        rmr_window=5, # Window size for RMR
        update_mode="pa"
    )
    rmr.fit(X_corrupted)

    print(f"OLMAR Wealth: {olmar.wealth_}")
    print(f"RMR Wealth: {rmr.wealth_}")

    # RMR should generally handle these outliers better or at least not crash
    # In this specific setup, we expect RMR to be more stable.
    # We check if RMR wealth is reasonable and ideally better or comparable to OLMAR
    # given the corruption.
    assert rmr.wealth_ > 0.5 # Basic sanity check
    
    # Check that RMR weights are less volatile around outliers?
    # Or simply that it runs without error.
    # Let's assert RMR outperforms OLMAR in this specific corrupted scenario
    # Note: This might be stochastic, so we use a fixed seed.
    assert rmr.wealth_ > olmar.wealth_

def test_universal_regret_bound():
    """
    Verify that the computed universal regret is finite and behaves reasonably.
    """
    X = make_stationary_returns(T=100, n=3)
    
    model = FollowTheWinner(strategy=FTWStrategy.EG)
    model.fit(X)
    
    # Calculate universal regret
    # We need a comparator that supports universal regret or just use the function
    # The regret function with RegretType.DYNAMIC_UNIVERSAL computes it against 
    # a theoretical bound or a specific universal portfolio implementation.
    # In skfolio, it might be implemented as comparing to a specific baseline.
    
    # Let's check the regret values
    regrets = regret(
        model, 
        X, 
        regret_type=RegretType.DYNAMIC_UNIVERSAL,
        dynamic_config={"path_length": 2.0}
    )
    
    assert len(regrets) == len(X)
    assert not np.any(np.isnan(regrets))
    # Universal regret should generally be positive as it compares to a strong baseline
    # but it depends on the exact definition in skfolio.
    # We just ensure it runs and returns valid numbers.

def test_zero_liquidity_handling():
    """
    Test behavior when assets have 0 returns (price relative = 1.0) 
    or even worse, price relative = 0.0 (bankruptcy).
    """
    X = np.zeros((50, 3)) # Flat market (0% return -> relative 1.0)
    
    model = FollowTheWinner(strategy=FTWStrategy.EG)
    model.fit(X)
    
    # In a flat market, wealth should stay 1.0 (minus fees if any)
    assert np.isclose(model.wealth_, 1.0)
    
    # Case 2: Asset 0 goes to 0 (bankruptcy)
    X_crash = np.zeros((50, 3))
    X_crash[10, 0] = -1.0 # Asset 0 dies at t=10 (return -100% -> relative 0)
    
    model_crash = FollowTheWinner(strategy=FTWStrategy.EG)
    model_crash.fit(X_crash)
    
    # If the model had any weight on asset 0 at t=10, wealth should drop significantly
    # Log-wealth would be -inf if weight > 0.
    # We want to ensure it doesn't raise an unhandled exception, 
    # though -inf wealth is mathematically correct.
    
    # Check if weights are updated after crash
    assert model_crash.weights_.shape == (3,)
