"""Tests for benchmark portfolio classes (BCRP, CRP, UCRP, BestStock).

This module tests the refactored BCRP class that supports multiple optimization objectives:
- Log-wealth maximization (Kelly criterion)
- Risk minimization (variance, CVaR, CDaR, etc.)
- Mean-risk utility maximization
"""

import numpy as np
import pytest

from skfolio.datasets import load_sp500_dataset
from skfolio.measures import PerfMeasure, RiskMeasure
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.optimization.online import BCRP
from skfolio.optimization.online._benchmark import CRP, UCRP, BestStock
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def X_small():
    """Small dataset for quick tests."""
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    return X.iloc[:50, :5]


@pytest.fixture(scope="module")
def X_medium():
    """Medium dataset for more comprehensive tests."""
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    return X.iloc[:100, :10]


class TestCRP:
    """Test Constant Rebalanced Portfolio."""

    def test_crp_with_custom_weights(self, X_small):
        """Test CRP with custom weights."""
        n_assets = X_small.shape[1]
        weights = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        crp = CRP(weights=weights)
        crp.fit(X_small)
        np.testing.assert_array_almost_equal(crp.weights_, weights)

    def test_ucrp_uniform_weights(self, X_small):
        """Test UCRP produces uniform weights."""
        n_assets = X_small.shape[1]
        ucrp = UCRP()
        ucrp.fit(X_small)
        expected = np.ones(n_assets) / n_assets
        np.testing.assert_array_almost_equal(ucrp.weights_, expected)

    def test_crp_none_defaults_to_uniform(self, X_small):
        """Test CRP with weights=None defaults to uniform."""
        n_assets = X_small.shape[1]
        crp = CRP(weights=None)
        crp.fit(X_small)
        expected = np.ones(n_assets) / n_assets
        np.testing.assert_array_almost_equal(crp.weights_, expected)


class TestBestStock:
    """Test BestStock benchmark."""

    def test_best_stock_picks_best_performer(self, X_small):
        """Test BestStock picks the asset with highest cumulative log return."""
        bs = BestStock()
        bs.fit(X_small)

        # Verify one-hot vector
        assert np.sum(bs.weights_) == pytest.approx(1.0)
        assert np.sum(bs.weights_ > 0) == 1

        # Verify it picks the best asset
        relatives = 1.0 + X_small.values
        log_returns = np.sum(np.log(relatives), axis=0)
        best_idx = np.argmax(log_returns)
        assert bs.weights_[best_idx] == pytest.approx(1.0)


class TestBCRPLogWealth:
    """Test BCRP with default log-wealth maximization."""

    def test_bcrp_default_is_log_wealth(self, X_small):
        """Test BCRP defaults to log-wealth maximization."""
        bcrp = BCRP()
        bcrp.fit(X_small)

        assert bcrp.objective_measure == PerfMeasure.LOG_WEALTH
        assert hasattr(bcrp, "weights_")
        assert bcrp.weights_.shape == (X_small.shape[1],)
        assert np.sum(bcrp.weights_) == pytest.approx(1.0)
        assert np.all(bcrp.weights_ >= 0)

    def test_bcrp_log_wealth_explicit(self, X_small):
        """Test explicit LOG_WEALTH specification."""
        bcrp = BCRP(objective_measure=PerfMeasure.LOG_WEALTH)
        bcrp.fit(X_small)

        assert bcrp.objective_measure == PerfMeasure.LOG_WEALTH
        assert np.sum(bcrp.weights_) == pytest.approx(1.0)

    def test_bcrp_log_wealth_outperforms_uniform(self, X_medium):
        """Test BCRP log-wealth outperforms uniform portfolio."""
        # BCRP should find better weights than uniform
        bcrp = BCRP()
        bcrp.fit(X_medium)

        # Compute log wealth for BCRP
        relatives = 1.0 + X_medium.values
        bcrp_log_wealth = np.sum(np.log(relatives @ bcrp.weights_))

        # Compute log wealth for uniform
        n = X_medium.shape[1]
        uniform_weights = np.ones(n) / n
        uniform_log_wealth = np.sum(np.log(relatives @ uniform_weights))

        # BCRP should be at least as good (allowing for numerical tolerance)
        assert bcrp_log_wealth >= uniform_log_wealth - 1e-6

    def test_bcrp_log_wealth_with_constraints(self, X_small):
        """Test BCRP log-wealth with box constraints."""
        bcrp = BCRP(
            objective_measure=PerfMeasure.LOG_WEALTH,
            min_weights=0.1,
            max_weights=0.5,
        )
        bcrp.fit(X_small)

        assert np.all(bcrp.weights_ >= 0.1 - 1e-6)
        assert np.all(bcrp.weights_ <= 0.5 + 1e-6)
        assert np.sum(bcrp.weights_) == pytest.approx(1.0)


class TestBCRPVariance:
    """Test BCRP with variance minimization."""

    def test_bcrp_variance_minimization(self, X_small):
        """Test BCRP with variance minimization."""
        bcrp = BCRP(
            objective_measure=RiskMeasure.VARIANCE,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
        )
        bcrp.fit(X_small)

        assert bcrp.objective_measure == RiskMeasure.VARIANCE
        assert np.sum(bcrp.weights_) == pytest.approx(1.0)
        assert np.all(bcrp.weights_ >= 0)

    def test_bcrp_variance_lower_than_uniform(self, X_medium):
        """Test variance minimization produces lower variance than uniform."""
        bcrp = BCRP(
            objective_measure=RiskMeasure.VARIANCE,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
        )
        bcrp.fit(X_medium)

        # Compute portfolio variance
        cov = np.cov(X_medium.values.T)
        bcrp_variance = bcrp.weights_ @ cov @ bcrp.weights_

        # Uniform variance
        n = X_medium.shape[1]
        uniform_weights = np.ones(n) / n
        uniform_variance = uniform_weights @ cov @ uniform_weights

        # BCRP should have lower or equal variance
        assert bcrp_variance <= uniform_variance + 1e-6

    def test_bcrp_mean_variance_utility(self, X_small):
        """Test BCRP with mean-variance utility maximization."""
        bcrp = BCRP(
            objective_measure=RiskMeasure.VARIANCE,
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            risk_aversion=2.0,
        )
        bcrp.fit(X_small)

        assert np.sum(bcrp.weights_) == pytest.approx(1.0)
        assert np.all(bcrp.weights_ >= 0)


class TestBCRPCVaR:
    """Test BCRP with CVaR minimization."""

    def test_bcrp_cvar_minimization(self, X_small):
        """Test BCRP with CVaR minimization."""
        bcrp = BCRP(
            objective_measure=RiskMeasure.CVAR,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            cvar_beta=0.95,
        )
        bcrp.fit(X_small)

        assert bcrp.objective_measure == RiskMeasure.CVAR
        assert np.sum(bcrp.weights_) == pytest.approx(1.0)
        assert np.all(bcrp.weights_ >= 0)

    def test_bcrp_cvar_different_betas(self, X_small):
        """Test BCRP CVaR with different confidence levels."""
        bcrp_90 = BCRP(
            objective_measure=RiskMeasure.CVAR,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            cvar_beta=0.90,
        )
        bcrp_99 = BCRP(
            objective_measure=RiskMeasure.CVAR,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            cvar_beta=0.99,
        )

        bcrp_90.fit(X_small)
        bcrp_99.fit(X_small)

        # Both should produce valid weights
        assert np.sum(bcrp_90.weights_) == pytest.approx(1.0)
        assert np.sum(bcrp_99.weights_) == pytest.approx(1.0)

        # Weights may differ due to different risk aversion
        # (90% CVaR is less conservative than 99% CVaR)


class TestBCRPCDaR:
    """Test BCRP with CDaR (Conditional Drawdown at Risk) minimization."""

    def test_bcrp_cdar_minimization(self, X_small):
        """Test BCRP with CDaR minimization."""
        bcrp = BCRP(
            objective_measure=RiskMeasure.CDAR,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            cdar_beta=0.95,
        )
        bcrp.fit(X_small)

        assert bcrp.objective_measure == RiskMeasure.CDAR
        assert np.sum(bcrp.weights_) == pytest.approx(1.0)
        assert np.all(bcrp.weights_ >= 0)


class TestBCRPOtherRiskMeasures:
    """Test BCRP with other risk measures."""

    @pytest.mark.parametrize(
        "risk_measure",
        [
            RiskMeasure.STANDARD_DEVIATION,
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.SEMI_VARIANCE,
        ],
    )
    def test_bcrp_various_risk_measures(self, risk_measure, X_small):
        """Test BCRP with various risk measures."""
        bcrp = BCRP(
            objective_measure=risk_measure,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
        )
        bcrp.fit(X_small)

        assert bcrp.objective_measure == risk_measure
        assert np.sum(bcrp.weights_) == pytest.approx(1.0)
        assert np.all(bcrp.weights_ >= 0)


class TestBCRPConstraints:
    """Test BCRP with various constraints."""

    def test_bcrp_with_box_constraints(self, X_small):
        """Test BCRP respects min/max weight constraints."""
        bcrp = BCRP(
            objective_measure=RiskMeasure.VARIANCE,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            min_weights=0.05,
            max_weights=0.40,
        )
        bcrp.fit(X_small)

        assert np.all(bcrp.weights_ >= 0.05 - 1e-6)
        assert np.all(bcrp.weights_ <= 0.40 + 1e-6)
        assert np.sum(bcrp.weights_) == pytest.approx(1.0)

    def test_bcrp_with_transaction_costs(self, X_small):
        """Test BCRP with transaction costs."""
        previous_weights = np.ones(X_small.shape[1]) / X_small.shape[1]

        bcrp = BCRP(
            objective_measure=PerfMeasure.LOG_WEALTH,
            transaction_costs=0.001,
            previous_weights=previous_weights,
        )
        bcrp.fit(X_small)

        assert np.sum(bcrp.weights_) == pytest.approx(1.0)


class TestBCRPValidation:
    """Test BCRP parameter validation."""

    def test_bcrp_rejects_invalid_measure(self, X_small):
        """Test BCRP raises error for invalid measure type."""
        with pytest.raises(ValueError, match="objective_measure must be RiskMeasure"):
            bcrp = BCRP(objective_measure="invalid")
            bcrp.fit(X_small)

    def test_bcrp_accepts_valid_measures(self, X_small):
        """Test BCRP accepts both RiskMeasure and LOG_WEALTH."""
        # Should not raise
        bcrp1 = BCRP(objective_measure=PerfMeasure.LOG_WEALTH)
        bcrp1.fit(X_small)

        bcrp2 = BCRP(objective_measure=RiskMeasure.VARIANCE)
        bcrp2.fit(X_small)


class TestBCRPFitDynamic:
    """Test BCRP fit_dynamic method for dynamic regret computation."""

    def test_bcrp_fit_dynamic_produces_all_weights(self, X_small):
        """Test fit_dynamic produces weights for each time step."""
        bcrp = BCRP()
        bcrp.fit_dynamic(X_small)

        assert hasattr(bcrp, "all_weights_")
        assert bcrp.all_weights_.shape == X_small.shape

    def test_bcrp_fit_dynamic_with_variance(self, X_small):
        """Test fit_dynamic works with variance objective."""
        bcrp = BCRP(
            objective_measure=RiskMeasure.VARIANCE,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
        )
        bcrp.fit_dynamic(X_small)

        assert hasattr(bcrp, "all_weights_")
        assert bcrp.all_weights_.shape == X_small.shape

    def test_bcrp_fit_dynamic_first_row_is_best_stock(self, X_small):
        """Test fit_dynamic uses BestStock for t=1."""
        bcrp = BCRP()
        bcrp.fit_dynamic(X_small)

        # First weight should be one-hot (BestStock)
        first_weights = bcrp.all_weights_[0, :]
        assert np.sum(first_weights) == pytest.approx(1.0)
        assert np.sum(first_weights > 0) == 1


class TestBCRPComparison:
    """Test comparing different BCRP objectives."""

    def test_log_wealth_vs_variance_produces_different_weights(self, X_medium):
        """Test log-wealth and variance minimization produce different portfolios."""
        bcrp_log = BCRP(objective_measure=PerfMeasure.LOG_WEALTH)
        bcrp_var = BCRP(
            objective_measure=RiskMeasure.VARIANCE,
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
        )

        bcrp_log.fit(X_medium)
        bcrp_var.fit(X_medium)

        # Weights should generally be different
        # (unless data is very special)
        diff = np.linalg.norm(bcrp_log.weights_ - bcrp_var.weights_)
        assert diff > 0.01  # Should have meaningful difference

    def test_variance_utility_interpolates(self, X_small):
        """Test variance utility with different risk aversions."""
        bcrp_conservative = BCRP(
            objective_measure=RiskMeasure.VARIANCE,
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            risk_aversion=5.0,
        )
        bcrp_aggressive = BCRP(
            objective_measure=RiskMeasure.VARIANCE,
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            risk_aversion=0.5,
        )

        bcrp_conservative.fit(X_small)
        bcrp_aggressive.fit(X_small)

        # Both should produce valid portfolios
        assert np.sum(bcrp_conservative.weights_) == pytest.approx(1.0)
        assert np.sum(bcrp_aggressive.weights_) == pytest.approx(1.0)


class TestBCRPBackwardCompatibility:
    """Test backward compatibility with original BCRP behavior."""

    def test_default_behavior_unchanged(self, X_small):
        """Test default BCRP() still does log-wealth maximization."""
        bcrp = BCRP()
        bcrp.fit(X_small)

        # Should default to LOG_WEALTH
        assert bcrp.objective_measure == PerfMeasure.LOG_WEALTH
        assert np.sum(bcrp.weights_) == pytest.approx(1.0)
        assert np.all(bcrp.weights_ >= 0)

    def test_partial_fit_works(self, X_small):
        """Test partial_fit still works."""
        bcrp = BCRP()
        result = bcrp.partial_fit(X_small)

        assert result is bcrp
        assert hasattr(bcrp, "weights_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
