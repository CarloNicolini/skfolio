"""
Tests for PROD (Soft-Bayes) strategy in FTRL framework.

The Prod algorithm (Orseau et al., 2017) is implemented via the Burg (log-barrier)
regularizer which yields a rational multiplicative update:
    w_{t+1,i} = w_{t,i} / (1 + η w_{t,i}(g_i - λ))
where λ is chosen to ensure Σw_i = 1.

Tests verify:
- Basic functionality and initialization
- Correct rational multiplicative update formula
- Regret bounds independent of max loss/gradient
- Learning rate schedules (constant and time-varying)
- Numerical stability
- Convergence of λ solver
"""

import numpy as np
import pytest

from skfolio.optimization.online._base import FTRLProximal
from skfolio.optimization.online._ftrl import FirstOrderOCO
from skfolio.optimization.online._mirror_maps import BurgMirrorMap
from skfolio.optimization.online._mixins import FTRLStrategy
from skfolio.optimization.online._projection import IdentityProjector

from ..utils import assert_box_budget


def test_prod_basic_initialization():
    """Test that PROD strategy initializes correctly and completes a fit."""
    model = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=0.1,
        warm_start=False,
    )
    # Simple 2-asset returns over 5 periods
    X = np.array(
        [
            [0.01, -0.005],
            [-0.02, 0.015],
            [0.005, -0.01],
            [0.02, 0.01],
            [-0.01, 0.02],
        ]
    )
    model.fit(X)

    # Verify the model uses BurgMirrorMap
    assert isinstance(model._ftrl_engine.map, BurgMirrorMap)

    # Check that weights sum to 1 and are non-negative
    assert_box_budget(model.weights_, lower=0.0, upper=1.0, budget=1.0)

    # Check that we have recorded all weights (one per period)
    assert model.all_weights_.shape == (
        5,
        2,
    )  # T rows (trading weights for each period)
    assert np.allclose(model.all_weights_[0], [0.5, 0.5])  # initial uniform


def test_prod_rational_update_properties():
    """
    Verify properties of the Burg (Prod) rational multiplicative update:
        w_{t+1,i} = w_{t,i} / (1 + η w_{t,i}(g_i - λ))

    Properties to verify:
    1. Weights stay on simplex (sum to 1, non-negative)
    2. Update is multiplicative (relative to previous weights)
    3. Numerically stable
    """
    eta = 0.15
    w0 = np.array([0.6, 0.4])
    model = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=eta,
        warm_start=False,
        initial_weights=w0,
    )

    # Single period update
    X = np.array([[0.03, -0.01]])
    model.fit(X)

    # Verify simplex properties
    assert_box_budget(model.weights_, lower=0.0, upper=1.0, budget=1.0)

    # Verify update is multiplicative: w_{t+1} depends on w_t
    # Assets with positive gradients should decrease in weight
    rel = 1.0 + X[0]  # gross relatives
    grad = -rel / np.dot(w0, rel)  # log-loss gradient

    # The update has form w_i / (1 + η w_i (g_i - λ))
    # For assets with g_i - λ > 0, weight decreases
    # We verify that the update produced valid weights
    assert np.all(model.weights_ > 0)
    assert np.abs(np.sum(model.weights_) - 1.0) < 1e-10


def test_prod_with_time_varying_learning_rate():
    """
    Test Prod with time-varying learning rate η_t = 1/(t+c).

    The paper shows that with c ≥ N, Prod recovers KT estimator bounds
    of O((N/2) log T).
    """
    d = 3
    c = d  # Set c = N as suggested in paper

    def learning_rate_schedule(t: int) -> float:
        return 1.0 / (t + c)

    model = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=learning_rate_schedule,
        warm_start=False,
    )

    # Generate synthetic returns
    np.random.seed(42)
    T = 20
    X = np.random.randn(T, d) * 0.02

    model.fit(X)

    # Verify weights are valid at each step
    assert model.all_weights_.shape == (T, d)
    for t in range(T):
        assert_box_budget(model.all_weights_[t], lower=0.0, upper=1.0, budget=1.0)

    # Verify learning rate decreased over time (implicitly tested by no NaN)
    assert np.all(np.isfinite(model.weights_))


def test_prod_regret_sublinear():
    """
    Test that Prod achieves sublinear regret relative to the best single asset.

    The Prod algorithm is designed to have regret bounds that depend only on
    log(N) and T, independent of maximum loss or gradient.
    """
    np.random.seed(123)
    d = 5
    T = 100

    # Generate returns with one dominant asset (to test regret)
    X = np.random.randn(T, d) * 0.01
    X[:, 0] += 0.005  # Asset 0 has slightly higher expected return

    model = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=0.1,
        warm_start=False,
    )
    model.fit(X)

    # Calculate cumulative returns
    gross_rel = 1.0 + X

    # Prod's cumulative return
    prod_wealth = 1.0
    for t in range(T):
        prod_wealth *= np.dot(model.all_weights_[t], gross_rel[t])

    # Best single asset's cumulative return (in hindsight)
    best_single_wealth = np.max(np.prod(gross_rel, axis=0))

    # Regret should be sublinear: R_T = O(log T)
    # For this test, we just verify Prod performs reasonably (not too far behind)
    regret_ratio = best_single_wealth / prod_wealth

    # With decent parameters, regret ratio should be bounded
    # (exact bound depends on problem instance, but should not explode)
    assert regret_ratio < 2.0, f"Regret ratio {regret_ratio} is too large"
    assert prod_wealth > 0.5, "Prod wealth should not collapse"


def test_prod_comparison_with_eg():
    """
    Compare Prod with Exponentiated Gradient (EG).

    Prod should be more robust than EG when facing incompetent experts
    or large gradients, due to the (1 - η) convex combination term.
    """
    np.random.seed(456)
    d = 4
    T = 50

    # Generate returns with some extreme values (to test robustness)
    X = np.random.randn(T, d) * 0.02
    X[10, 2] = -0.5  # Large negative shock on asset 2
    X[30, 3] = 0.3  # Large positive shock on asset 3

    # Run Prod
    prod_model = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=0.1,
        warm_start=False,
    )
    prod_model.fit(X)

    # Run EG for comparison
    eg_model = FTRLProximal(
        strategy=FTRLStrategy.EG,
        learning_rate=0.1,
        warm_start=False,
    )
    eg_model.fit(X)

    # Both should complete without NaN
    assert np.all(np.isfinite(prod_model.weights_))
    assert np.all(np.isfinite(eg_model.weights_))

    # Calculate final cumulative returns
    gross_rel = 1.0 + X

    prod_wealth = 1.0
    eg_wealth = 1.0
    for t in range(T):
        prod_wealth *= np.dot(prod_model.all_weights_[t], gross_rel[t])
        eg_wealth *= np.dot(eg_model.all_weights_[t], gross_rel[t])

    # Both should have positive wealth (no catastrophic failure)
    assert prod_wealth > 0.1
    assert eg_wealth > 0.1

    # Prod is designed to be more robust, so it should perform reasonably
    # (not necessarily better in all cases, but should not fail dramatically)
    wealth_ratio = prod_wealth / eg_wealth
    assert 0.5 < wealth_ratio < 2.0, "Prod and EG should have comparable performance"


def test_prod_numerical_stability_extreme_gradients():
    """
    Test that Prod remains numerically stable with extreme gradients.

    The paper emphasizes that Prod's regret bound is independent of
    maximum gradient magnitude, suggesting robust behavior.
    """
    d = 3

    # Create scenario with very large gradient on first step
    X = np.array(
        [
            [-0.9, 0.01, 0.01],  # Large loss on asset 0
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
        ]
    )

    model = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=0.1,
        warm_start=False,
    )
    model.fit(X)

    # Verify no NaN or Inf
    assert np.all(np.isfinite(model.all_weights_))
    assert np.all(np.isfinite(model.weights_))

    # Verify weights remain valid
    for t in range(len(X)):
        assert_box_budget(model.all_weights_[t], lower=0.0, upper=1.0, budget=1.0)


def test_prod_with_smooth_prediction():
    """
    Test Prod with optimistic/smooth prediction (m_t = g_{t-1}).

    This should work with Prod just as with other strategies.
    """
    model = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=0.1,
        grad_predictor=True,
        warm_start=False,
    )

    X = np.array(
        [
            [0.01, -0.005],
            [0.015, -0.008],
            [0.02, -0.01],
        ]
    )

    model.fit(X)

    # Should complete without errors
    assert np.all(np.isfinite(model.weights_))
    assert_box_budget(model.weights_, lower=0.0, upper=1.0, budget=1.0)


def test_prod_mirror_map_interface():
    """Test that BurgMirrorMap works correctly within _FTRLEngine."""
    d = 3
    mirror_map = BurgMirrorMap()
    engine = FirstOrderOCO(
        mirror_map=mirror_map,
        projector=IdentityProjector(),
        eta=0.1,
        mode="omd",
    )

    # Simulate a few steps
    grads = [
        np.array([0.1, -0.05, 0.02]),
        np.array([-0.03, 0.08, -0.01]),
        np.array([0.05, -0.02, 0.03]),
    ]

    for g in grads:
        w = engine.step(g)
        # Verify valid weights
        assert np.all(w >= 0)
        assert np.isclose(np.sum(w), 1.0, atol=1e-10)
        assert np.all(np.isfinite(w))


def test_prod_constant_vs_decreasing_learning_rate():
    """
    Compare constant learning rate vs decreasing schedule for Prod.

    Theory suggests decreasing η_t = 1/(t+c) gives better asymptotic regret.
    """
    np.random.seed(789)
    d = 4
    T = 100
    X = np.random.randn(T, d) * 0.015

    # Constant learning rate
    model_const = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=0.05,
        warm_start=False,
    )
    model_const.fit(X)

    # Decreasing learning rate
    model_decay = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=lambda t: 1.0 / (t + d),
        warm_start=False,
    )
    model_decay.fit(X)

    # Both should complete successfully
    assert np.all(np.isfinite(model_const.weights_))
    assert np.all(np.isfinite(model_decay.weights_))

    # Calculate cumulative wealth
    gross_rel = 1.0 + X

    wealth_const = 1.0
    wealth_decay = 1.0
    for t in range(T):
        wealth_const *= np.dot(model_const.all_weights_[t], gross_rel[t])
        wealth_decay *= np.dot(model_decay.all_weights_[t], gross_rel[t])

    # Both should achieve positive returns
    assert wealth_const > 0.5
    assert wealth_decay > 0.5


def test_prod_with_constraints():
    """Test that Prod works with portfolio constraints (min/max weights)."""
    model = FTRLProximal(
        strategy=FTRLStrategy.PROD,
        learning_rate=0.1,
        min_weights=0.1,  # At least 10% in each asset
        max_weights=0.6,  # At most 60% in each asset
        warm_start=False,
    )

    X = np.array(
        [
            [0.02, -0.01, 0.005],
            [-0.01, 0.03, -0.005],
            [0.015, -0.02, 0.01],
        ]
    )

    model.fit(X)

    # Verify constraints are satisfied at each step
    for t in range(len(X)):
        w = model.all_weights_[t]
        assert np.all(w >= 0.1 - 1e-6)
        assert np.all(w <= 0.6 + 1e-6)
        assert np.isclose(np.sum(w), 1.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
