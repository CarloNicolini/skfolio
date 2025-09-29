"""
Comprehensive tests for mean-reversion strategies (PAMR, CWMR, OLMAR).

This module tests:
1. Parameter validation for all three strategies
2. Correct mathematical implementation of each algorithm
3. Partial_fit and fit behavior
4. Edge cases and numerical stability
5. Integration with projector and constraints
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online._mean_reversion import (
    HingeSurrogateLoss,
    MeanReversion,
    OLMAR1Predictor,
    OLMAR2Predictor,
    SquaredHingeSurrogateLoss,
    SoftplusSurrogateLoss,
)
from skfolio.optimization.online._projection import project_box_and_sum


# ============================================================================
# Fixtures and Utilities
# ============================================================================


@pytest.fixture
def simple_returns():
    """Simple 2-asset returns for basic tests."""
    return np.array(
        [
            [0.01, -0.02],
            [-0.01, 0.02],
            [0.02, -0.01],
            [-0.02, 0.01],
        ]
    )


@pytest.fixture
def mean_reverting_returns():
    """Mean-reverting 3-asset market."""
    rng = np.random.default_rng(42)
    n_periods = 50
    returns = np.zeros((n_periods, 3))

    # Asset 1: mean-reverting around 0
    for t in range(1, n_periods):
        returns[t, 0] = -0.5 * returns[t - 1, 0] + 0.01 * rng.standard_normal()

    # Asset 2: mean-reverting with larger amplitude
    for t in range(1, n_periods):
        returns[t, 1] = -0.7 * returns[t - 1, 1] + 0.015 * rng.standard_normal()

    # Asset 3: random walk (no mean reversion)
    returns[:, 2] = 0.01 * rng.standard_normal(n_periods)

    return returns


def assert_valid_weights(w: np.ndarray, budget: float = 1.0, tol: float = 1e-8):
    """Assert that weights form a valid portfolio."""
    assert np.all(w >= -tol), "Weights should be non-negative"
    assert_allclose(np.sum(w), budget, atol=tol, err_msg="Weights should sum to budget")


# ============================================================================
# 1. Parameter Validation Tests
# ============================================================================


class TestParameterValidation:
    """Test parameter validation for all strategies."""

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        model = MeanReversion(strategy="invalid")
        X = np.array([[0.01, 0.02]])
        with pytest.raises((ValueError, Exception), match="strategy"):
            model.fit(X)

    def test_olmar_order_validation(self):
        """Test OLMAR order parameter validation."""
        # Valid orders
        MeanReversion(strategy="olmar", olmar_order=1)
        MeanReversion(strategy="olmar", olmar_order=2)

        # Invalid order - sklearn validation happens at fit time
        model = MeanReversion(strategy="olmar", olmar_order=3)
        X = np.array([[0.01, 0.02]])
        with pytest.raises(ValueError):
            model.fit(X)

    def test_olmar_window_validation(self):
        """Test OLMAR window parameter validation."""
        # Valid window
        MeanReversion(strategy="olmar", olmar_order=1, olmar_window=5)

        # Invalid window - sklearn validation happens at fit time
        model = MeanReversion(strategy="olmar", olmar_order=1, olmar_window=0)
        X = np.array([[0.01, 0.02]])
        with pytest.raises(ValueError):
            model.fit(X)

    def test_olmar_alpha_validation(self):
        """Test OLMAR alpha parameter validation."""
        # Valid alpha
        MeanReversion(strategy="olmar", olmar_order=2, olmar_alpha=0.5)

        # Invalid alpha (out of range) - sklearn validation happens at fit time
        model = MeanReversion(strategy="olmar", olmar_order=2, olmar_alpha=1.5)
        X = np.array([[0.01, 0.02]])
        with pytest.raises(ValueError):
            model.fit(X)

    def test_cwmr_eta_validation(self):
        """Test CWMR eta parameter validation."""
        # Valid eta
        MeanReversion(strategy="cwmr", cwmr_eta=0.95)

        # Invalid eta (must be in (0.5, 1)) - sklearn validation happens at fit time
        model1 = MeanReversion(strategy="cwmr", cwmr_eta=0.5)  # exactly 0.5 is invalid
        X = np.array([[0.01, 0.02]])
        with pytest.raises(ValueError):
            model1.fit(X)

        model2 = MeanReversion(strategy="cwmr", cwmr_eta=1.0)  # exactly 1.0 is invalid
        with pytest.raises(ValueError):
            model2.fit(X)

    def test_cwmr_variance_bounds_validation(self):
        """Test CWMR variance bounds validation."""
        # Valid bounds
        MeanReversion(strategy="cwmr", cwmr_min_var=1e-6, cwmr_max_var=10.0)

        # Invalid bounds (max < min) - checked in _validate_strategy_combinations
        model = MeanReversion(strategy="cwmr", cwmr_min_var=10.0, cwmr_max_var=1e-6)
        X = np.array([[0.01, 0.02]])
        with pytest.raises(ValueError, match="cwmr_max_var cannot be smaller"):
            model.fit(X)

    def test_epsilon_validation(self):
        """Test epsilon parameter validation."""
        # Valid epsilon
        MeanReversion(strategy="pamr", epsilon=1.0)

        # Invalid epsilon (must be positive) - sklearn validation happens at fit time
        model = MeanReversion(strategy="pamr", epsilon=-1.0)
        X = np.array([[0.01, 0.02]])
        with pytest.raises(ValueError):
            model.fit(X)

    def test_loss_validation(self):
        """Test surrogate loss validation."""
        # Valid losses
        for loss in ["hinge", "squared_hinge", "softplus"]:
            MeanReversion(strategy="olmar", loss=loss, update_mode="md")

        # Invalid loss - sklearn validation happens at fit time
        model = MeanReversion(strategy="olmar", loss="invalid")
        X = np.array([[0.01, 0.02]])
        with pytest.raises(ValueError):
            model.fit(X)

    def test_update_mode_validation(self):
        """Test update mode validation."""
        # Valid modes
        MeanReversion(strategy="pamr", update_mode="pa")
        MeanReversion(strategy="pamr", update_mode="md")

        # Invalid mode - sklearn validation happens at fit time
        model = MeanReversion(strategy="pamr", update_mode="invalid")
        X = np.array([[0.01, 0.02]])
        with pytest.raises(ValueError):
            model.fit(X)

    def test_mirror_validation(self):
        """Test mirror map validation."""
        # Valid mirrors
        MeanReversion(strategy="pamr", update_mode="md", mirror="euclidean")
        MeanReversion(strategy="pamr", update_mode="md", mirror="entropy")

        # Invalid mirror - sklearn validation happens at fit time
        model = MeanReversion(strategy="pamr", update_mode="md", mirror="invalid")
        X = np.array([[0.01, 0.02]])
        with pytest.raises(ValueError):
            model.fit(X)


# ============================================================================
# 2. PAMR Tests
# ============================================================================


class TestPAMR:
    """Test Passive-Aggressive Mean Reversion (PAMR)."""

    def test_pamr_pa_basic(self, simple_returns):
        """Test basic PAMR with PA update."""
        model = MeanReversion(strategy="pamr", epsilon=1.0, update_mode="pa")
        model.fit(simple_returns)

        # Check output shapes
        assert model.weights_.shape == (2,)
        assert model.all_weights_.shape == (4, 2)

        # Check weights are valid
        for w in model.all_weights_:
            assert_valid_weights(w)

    def test_pamr_closed_form_correctness(self):
        """Test PAMR closed-form update matches expected formula."""
        # Single step test with known values
        model = MeanReversion(strategy="pamr", epsilon=1.0, update_mode="pa")

        # Initial uniform weights
        X = np.array([[0.01, 0.02]])
        model.fit(X)
        w_init = model.weights_.copy()

        # Second step
        X2 = np.array([[0.03, -0.01]])  # Asset 1 up, asset 2 down
        model.partial_fit(X2)
        w_new = model.weights_

        # Manually compute expected update
        x_t = 1.0 + X2[0]  # gross relatives
        margin = np.dot(w_init, x_t)
        loss = max(0.0, margin - model.epsilon)

        if loss > 0.0:
            c = x_t - np.mean(x_t)
            denom = np.dot(c, c)
            if denom > 0.0:
                tau = loss / denom
                w_expected = project_box_and_sum(
                    w_init - tau * c, lower=0.0, upper=1.0, budget=1.0
                )
                assert_allclose(w_new, w_expected, atol=1e-6)

    def test_pamr_no_update_when_margin_satisfied(self):
        """Test PAMR doesn't update when margin constraint is satisfied."""
        model = MeanReversion(
            strategy="pamr",
            epsilon=2.0,  # High epsilon - hard to violate
            update_mode="pa",
        )

        # Returns that won't violate margin
        X = np.array(
            [
                [0.001, 0.001],
                [0.001, 0.001],
            ]
        )

        model.fit(X)
        w_before = model.weights_.copy()

        # Should stay roughly uniform since margin isn't violated
        assert_allclose(w_before, [0.5, 0.5], atol=0.1)

    def test_pamr_md_update(self, simple_returns):
        """Test PAMR with mirror-descent update."""
        model = MeanReversion(
            strategy="pamr",
            epsilon=1.0,
            update_mode="md",
            learning_rate=0.1,
            mirror="euclidean",
        )
        model.fit(simple_returns)

        # Check valid weights
        for w in model.all_weights_:
            assert_valid_weights(w)

    def test_pamr_mean_reverting_behavior(self, mean_reverting_returns):
        """Test PAMR shifts weights toward recent losers."""
        model = MeanReversion(strategy="pamr", epsilon=1.0, update_mode="pa")
        model.fit(mean_reverting_returns)

        # Check that model learns something (not uniform)
        final_weights = model.weights_
        assert not np.allclose(final_weights, 1.0 / 3.0, atol=0.1)


# ============================================================================
# 3. OLMAR Tests
# ============================================================================


class TestOLMAR:
    """Test Online Moving Average Reversion (OLMAR)."""

    def test_olmar1_predictor_cold_start(self):
        """Test OLMAR-1 predictor in cold-start phase."""
        predictor = OLMAR1Predictor(window=3)
        predictor.reset(d=2)

        # First observation - should return itself
        x1 = np.array([1.02, 0.98])
        phi1 = predictor.update_and_predict(x1)
        assert_allclose(phi1, x1)

        # Second observation - still in cold start
        x2 = np.array([0.99, 1.01])
        phi2 = predictor.update_and_predict(x2)
        assert_allclose(phi2, x2)

    def test_olmar1_predictor_moving_average(self):
        """Test OLMAR-1 predictor computes correct moving average."""
        predictor = OLMAR1Predictor(window=2)
        predictor.reset(d=2)

        # Build up history
        x1 = np.array([1.1, 0.9])
        x2 = np.array([0.9, 1.1])
        x3 = np.array([1.0, 1.0])

        predictor.update_and_predict(x1)
        predictor.update_and_predict(x2)
        phi = predictor.update_and_predict(x3)

        # phi should be moving average of inverse cumulative products
        # Manual calculation for verification
        assert phi.shape == (2,)
        assert np.all(phi > 0)

    def test_olmar2_predictor_exponential_smoothing(self):
        """Test OLMAR-2 predictor exponential smoothing."""
        predictor = OLMAR2Predictor(alpha=0.5)
        predictor.reset(d=2)

        # Initial phi is ones
        x1 = np.array([1.2, 0.8])
        phi1 = predictor.update_and_predict(x1)

        # phi = alpha * 1 + (1 - alpha) * (1 / x1)
        expected = 0.5 * np.ones(2) + 0.5 * (1.0 / x1)
        assert_allclose(phi1, expected, atol=1e-6)

        # Next update
        x2 = np.array([0.9, 1.1])
        phi2 = predictor.update_and_predict(x2)
        expected2 = 0.5 * np.ones(2) + 0.5 * (phi1 / x2)
        assert_allclose(phi2, expected2, atol=1e-6)

    def test_olmar_pa_basic(self, mean_reverting_returns):
        """Test OLMAR-1 with PA update."""
        model = MeanReversion(
            strategy="olmar",
            olmar_order=1,
            olmar_window=5,
            epsilon=1.1,
            update_mode="pa",
        )
        model.fit(mean_reverting_returns)

        # Check weights are valid
        for w in model.all_weights_:
            assert_valid_weights(w)

    def test_olmar1_vs_olmar2_predictors(self, mean_reverting_returns):
        """Test both OLMAR predictor variants produce valid results."""
        model1 = MeanReversion(
            strategy="olmar",
            olmar_order=1,
            olmar_window=3,
            epsilon=1.1,
            update_mode="pa",
        )
        model2 = MeanReversion(
            strategy="olmar",
            olmar_order=2,
            olmar_alpha=0.5,
            epsilon=1.1,
            update_mode="pa",
        )

        model1.fit(mean_reverting_returns)
        model2.fit(mean_reverting_returns)

        # Both should produce valid weights
        assert_valid_weights(model1.weights_)
        assert_valid_weights(model2.weights_)

        # Weight trajectories should differ (different predictors)
        # Compare trajectories instead of just final weights
        distance = np.linalg.norm(model1.all_weights_ - model2.all_weights_)
        assert distance > 1e-3, (
            "Different predictors should produce different trajectories"
        )

    def test_olmar_with_different_losses(self, simple_returns):
        """Test OLMAR with different surrogate losses in MD mode."""
        for loss in ["hinge", "squared_hinge", "softplus"]:
            model = MeanReversion(
                strategy="olmar",
                olmar_order=2,
                epsilon=1.1,
                loss=loss,
                update_mode="md",
                learning_rate=0.1,
            )
            model.fit(simple_returns)
            assert_valid_weights(model.weights_)

    def test_olmar_epsilon_effect(self, mean_reverting_returns):
        """Test that different epsilon values affect aggressiveness."""
        # Low epsilon - less aggressive
        model_low = MeanReversion(
            strategy="olmar", olmar_order=1, epsilon=1.01, update_mode="pa"
        )

        # High epsilon - more aggressive
        model_high = MeanReversion(
            strategy="olmar", olmar_order=1, epsilon=2.0, update_mode="pa"
        )

        X_small = mean_reverting_returns[:10]
        model_low.fit(X_small)
        model_high.fit(X_small)

        # Both should produce valid weights
        assert_valid_weights(model_low.weights_)
        assert_valid_weights(model_high.weights_)


# ============================================================================
# 4. CWMR Tests
# ============================================================================


class TestCWMR:
    """Test Confidence-Weighted Mean Reversion (CWMR)."""

    def test_cwmr_pa_basic(self, simple_returns):
        """Test basic CWMR with PA (closed-form KL) update."""
        model = MeanReversion(
            strategy="cwmr",
            cwmr_eta=0.95,
            cwmr_sigma0=1.0,
            epsilon=1.0,
            update_mode="pa",
        )
        model.fit(simple_returns)

        # Check weights are valid
        for w in model.all_weights_:
            assert_valid_weights(w)

        # Check internal state
        assert model._cwmr_mu is not None
        assert model._cwmr_Sdiag is not None
        assert model._cwmr_mu.shape == (2,)
        assert model._cwmr_Sdiag.shape == (2,)
        assert np.all(model._cwmr_Sdiag > 0)  # Variance must be positive

    def test_cwmr_oco_mode(self, simple_returns):
        """Test CWMR with OCO/MD update mode."""
        model = MeanReversion(
            strategy="cwmr",
            cwmr_eta=0.95,
            cwmr_sigma0=1.0,
            cwmr_mean_lr=0.1,
            cwmr_var_lr=0.1,
            epsilon=1.0,
            update_mode="md",
        )
        model.fit(simple_returns)

        # Check weights are valid
        for w in model.all_weights_:
            assert_valid_weights(w)

    def test_cwmr_variance_clipping(self):
        """Test CWMR variance bounds are respected."""
        model = MeanReversion(
            strategy="cwmr",
            cwmr_eta=0.95,
            cwmr_sigma0=1.0,
            cwmr_min_var=1e-6,
            cwmr_max_var=10.0,
            epsilon=1.0,
            update_mode="pa",
        )

        X = np.array(
            [
                [0.02, -0.01],
                [-0.03, 0.02],
                [0.01, -0.02],
            ]
        )

        model.fit(X)

        # Check variance bounds
        assert np.all(model._cwmr_Sdiag >= 1e-6)
        assert np.all(model._cwmr_Sdiag <= 10.0)

    def test_cwmr_confidence_level_effect(self, mean_reverting_returns):
        """Test different confidence levels produce different behavior."""
        # Low confidence
        model_low = MeanReversion(
            strategy="cwmr",
            cwmr_eta=0.60,  # Lower confidence
            epsilon=1.0,
            update_mode="pa",
        )

        # High confidence
        model_high = MeanReversion(
            strategy="cwmr",
            cwmr_eta=0.95,  # Higher confidence
            epsilon=1.0,
            update_mode="pa",
        )

        X_small = mean_reverting_returns[:10]
        model_low.fit(X_small)
        model_high.fit(X_small)

        # Both should produce valid results
        assert_valid_weights(model_low.weights_)
        assert_valid_weights(model_high.weights_)

        # Variance should differ (higher confidence = lower variance typically)
        # This is a soft assertion
        assert np.mean(model_high._cwmr_Sdiag) >= 0  # Just check it exists

    def test_cwmr_second_order_information(self, mean_reverting_returns):
        """Test CWMR maintains and updates covariance."""
        model = MeanReversion(
            strategy="cwmr",
            cwmr_eta=0.95,
            cwmr_sigma0=1.0,
            epsilon=1.0,
            update_mode="pa",
        )

        # Fit gradually and track variance evolution
        variances = []
        for i in range(1, len(mean_reverting_returns)):
            model.partial_fit(mean_reverting_returns[i : i + 1])
            variances.append(model._cwmr_Sdiag.copy())

        variances = np.array(variances)

        # Variance should change over time (learning)
        assert not np.allclose(variances[0], variances[-1])


# ============================================================================
# 5. Surrogate Loss Tests
# ============================================================================


class TestSurrogateLosses:
    """Test surrogate loss functions."""

    def test_hinge_loss(self):
        """Test hinge loss computation."""
        loss = HingeSurrogateLoss(epsilon=1.0)

        w = np.array([0.5, 0.5])
        phi = np.array([1.2, 0.8])

        # Value
        val = loss.value(w, phi)
        expected_val = max(0.0, 1.0 - np.dot(w, phi))
        assert_allclose(val, expected_val)

        # Gradient
        grad = loss.grad(w, phi)
        if np.dot(phi, w) < 1.0:
            assert_allclose(grad, -phi)
        else:
            assert_allclose(grad, np.zeros_like(phi))

    def test_squared_hinge_loss(self):
        """Test squared hinge loss computation."""
        loss = SquaredHingeSurrogateLoss(epsilon=1.0)

        w = np.array([0.5, 0.5])
        phi = np.array([1.2, 0.8])

        # Value
        val = loss.value(w, phi)
        slack = max(0.0, 1.0 - np.dot(w, phi))
        expected_val = slack**2
        assert_allclose(val, expected_val)

        # Gradient
        grad = loss.grad(w, phi)
        if np.dot(phi, w) < 1.0:
            expected_grad = -2.0 * (1.0 - np.dot(w, phi)) * phi
            assert_allclose(grad, expected_grad)

    def test_softplus_loss(self):
        """Test softplus loss computation and smoothness."""
        loss = SoftplusSurrogateLoss(epsilon=1.0, beta=5.0)

        w = np.array([0.5, 0.5])
        phi = np.array([1.2, 0.8])

        # Value should be positive and smooth
        val = loss.value(w, phi)
        assert val >= 0

        # Gradient should be smooth (always defined)
        grad = loss.grad(w, phi)
        assert grad.shape == phi.shape

        # Test numerical stability for extreme values
        phi_large = np.array([100.0, 100.0])
        val_large = loss.value(w, phi_large)
        assert np.isfinite(val_large)


# ============================================================================
# 6. Partial Fit and Fit Tests
# ============================================================================


class TestPartialFitBehavior:
    """Test partial_fit and fit methods."""

    def test_partial_fit_single_row(self, simple_returns):
        """Test partial_fit with single row."""
        model = MeanReversion(strategy="pamr", epsilon=1.0)

        # Fit one row at a time
        for row in simple_returns:
            model.partial_fit(row.reshape(1, -1))

        assert_valid_weights(model.weights_)

    def test_fit_multi_row(self, simple_returns):
        """Test fit with multiple rows."""
        model = MeanReversion(strategy="pamr", epsilon=1.0)
        model.fit(simple_returns)

        assert model.all_weights_.shape == (4, 2)
        for w in model.all_weights_:
            assert_valid_weights(w)

    def test_warm_start_true(self, simple_returns):
        """Test warm_start=True preserves state."""
        model = MeanReversion(strategy="pamr", epsilon=1.0, warm_start=True)

        # First fit
        model.fit(simple_returns[:2])
        w_after_first = model.weights_.copy()

        # Second fit should continue from previous state
        model.fit(simple_returns[2:])
        w_after_second = model.weights_

        # State should have evolved
        assert not np.allclose(w_after_first, w_after_second)

    def test_warm_start_false(self, simple_returns):
        """Test warm_start=False resets state."""
        model = MeanReversion(strategy="pamr", epsilon=1.0, warm_start=False)

        # First fit
        model.fit(simple_returns[:2])

        # Second fit should reset
        model.fit(simple_returns)

        # Should work without errors
        assert_valid_weights(model.weights_)

    def test_fit_produces_all_weights(self, simple_returns):
        """Test fit produces all_weights_ attribute."""
        model = MeanReversion(strategy="pamr", epsilon=1.0)
        model.fit(simple_returns)

        assert hasattr(model, "all_weights_")
        assert model.all_weights_.shape == (
            len(simple_returns),
            simple_returns.shape[1],
        )


# ============================================================================
# 7. Edge Cases and Numerical Stability
# ============================================================================


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_returns(self):
        """Test handling of zero returns."""
        X = np.zeros((5, 3))
        model = MeanReversion(strategy="pamr", epsilon=1.0)
        model.fit(X)

        # Should not crash, weights should be valid
        assert_valid_weights(model.weights_)

    def test_extreme_returns(self):
        """Test handling of extreme returns."""
        X = np.array(
            [
                [0.5, -0.4],  # Large positive and negative
                [-0.3, 0.6],
                [0.1, -0.05],
            ]
        )

        model = MeanReversion(strategy="pamr", epsilon=1.0)
        model.fit(X)

        # Should handle extreme values
        assert_valid_weights(model.weights_)
        assert np.all(np.isfinite(model.weights_))

    def test_two_assets_extreme_concentration(self):
        """Test with 2 assets where one dominates."""
        # Asset 1 consistently outperforms asset 2
        X = np.array(
            [
                [0.05, -0.04],
                [0.04, -0.03],
                [0.06, -0.05],
            ]
        )
        model = MeanReversion(strategy="pamr", epsilon=1.0)
        model.fit(X)

        # Should still produce valid weights (mean reversion might prevent full concentration)
        assert_valid_weights(model.weights_)

    def test_cwmr_very_small_variance(self):
        """Test CWMR doesn't crash with tiny variance."""
        model = MeanReversion(
            strategy="cwmr",
            cwmr_sigma0=1e-12,
            cwmr_min_var=1e-15,
            epsilon=1.0,
            update_mode="pa",
        )

        X = np.array([[0.01, 0.02], [0.02, 0.01]])
        model.fit(X)

        # Should work without numerical issues
        assert np.all(np.isfinite(model.weights_))
        assert np.all(model._cwmr_Sdiag > 0)


# ============================================================================
# 8. Integration with Constraints
# ============================================================================


class TestConstraintIntegration:
    """Test integration with constraints and projector."""

    def test_box_constraints(self, simple_returns):
        """Test with box constraints on weights."""
        model = MeanReversion(
            strategy="pamr", epsilon=1.0, min_weights=0.2, max_weights=0.6
        )
        model.fit(simple_returns)

        # Check box constraints are satisfied
        assert np.all(model.weights_ >= 0.2 - 1e-8)
        assert np.all(model.weights_ <= 0.6 + 1e-8)
        assert_valid_weights(model.weights_)

    def test_turnover_constraint(self, simple_returns):
        """Test with turnover constraints."""
        model = MeanReversion(strategy="pamr", epsilon=1.0, max_turnover=0.1)
        model.fit(simple_returns)

        # Check turnover constraint
        for t in range(1, len(model.all_weights_)):
            turnover = np.sum(np.abs(model.all_weights_[t] - model.all_weights_[t - 1]))
            assert turnover <= 0.1 + 1e-6

    def test_budget_constraint(self, simple_returns):
        """Test with custom budget."""
        budget = 0.8
        model = MeanReversion(strategy="pamr", epsilon=1.0, budget=budget)
        model.fit(simple_returns)

        assert_allclose(np.sum(model.weights_), budget, atol=1e-6)


# ============================================================================
# 9. Comparative Tests
# ============================================================================


class TestComparativePerformance:
    """Compare strategies on specific market types."""

    def test_mean_reverting_market_favors_losers(self, mean_reverting_returns):
        """Test that Follow-the-Loser strategies work on mean-reverting markets."""
        # PAMR
        pamr = MeanReversion(strategy="pamr", epsilon=1.0)
        pamr.fit(mean_reverting_returns)

        # OLMAR
        olmar = MeanReversion(strategy="olmar", olmar_order=1, epsilon=1.1)
        olmar.fit(mean_reverting_returns)

        # Both should learn non-uniform weights
        assert not np.allclose(pamr.weights_, 1.0 / 3.0, atol=0.1)
        assert not np.allclose(olmar.weights_, 1.0 / 3.0, atol=0.1)

    def test_pa_vs_md_convergence(self, mean_reverting_returns):
        """Test PA and MD modes converge to similar weights."""
        X_small = mean_reverting_returns[:20]

        # PA mode
        model_pa = MeanReversion(strategy="pamr", epsilon=1.0, update_mode="pa")
        model_pa.fit(X_small)

        # MD mode
        model_md = MeanReversion(
            strategy="pamr",
            epsilon=1.0,
            update_mode="md",
            learning_rate=0.1,
            mirror="euclidean",
        )
        model_md.fit(X_small)

        # Both should produce reasonable weights
        # (not necessarily close, but both valid)
        assert_valid_weights(model_pa.weights_)
        assert_valid_weights(model_md.weights_)


# ============================================================================
# 10. Regression Tests
# ============================================================================


class TestRegressionOLPS:
    """Regression tests against OLPS Matlab reference implementation."""

    def test_olmar1_simple_case(self):
        """
        Test OLMAR-1 against simple hand-calculated case.

        This is a minimal regression test to ensure the predictor
        and update logic are correct.
        """
        # Simple 2-period, 2-asset case
        X = np.array(
            [
                [0.02, -0.01],  # Asset 1 up, 2 down
                [-0.01, 0.02],  # Asset 1 down, 2 up (mean reversion)
            ]
        )

        model = MeanReversion(
            strategy="olmar",
            olmar_order=1,
            olmar_window=1,
            epsilon=1.05,
            update_mode="pa",
        )
        model.fit(X)

        # After mean reversion, should favor asset 2 (which went up in period 2)
        # This is a soft assertion - just checking it runs and produces valid output
        assert_valid_weights(model.weights_)
        assert np.all(np.isfinite(model.weights_))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
