"""Integration tests for FTRL with custom objective functions."""

import numpy as np
import pytest

from skfolio.measures._enums import PerfMeasure, RiskMeasure
from skfolio.optimization.online import FollowTheWinner


@pytest.fixture
def sample_returns():
    """Generate sample return data."""
    np.random.seed(42)
    n_obs = 50
    n_assets = 3
    # Generate returns with slight positive drift
    returns = np.random.randn(n_obs, n_assets) * 0.02 + 0.001
    return returns


class TestFTRLWithObjectives:
    """Test FTRL with different objective functions."""

    def test_default_log_wealth(self, sample_returns):
        """Test FTRL with default log-wealth objective."""
        model = FollowTheWinner(strategy="eg", learning_rate=0.1)
        model.fit(sample_returns)

        assert model.weights_.shape == (3,)
        assert np.isclose(model.weights_.sum(), 1.0)
        assert np.all(model.weights_ >= 0)

    def test_variance_objective(self, sample_returns):
        """Test FTRL with variance risk objective."""
        model = FollowTheWinner(
            strategy="ogd",
            objective=RiskMeasure.VARIANCE,
            learning_rate=0.05,
        )
        model.fit(sample_returns)

        assert model.weights_.shape == (3,)
        assert np.isclose(model.weights_.sum(), 1.0)
        assert np.all(model.weights_ >= 0)

    def test_cvar_objective(self, sample_returns):
        """Test FTRL with CVaR risk objective."""
        model = FollowTheWinner(
            strategy="eg",
            objective=RiskMeasure.CVAR,
            learning_rate=0.1,
        )
        model.fit(sample_returns)

        assert model.weights_.shape == (3,)
        assert np.isclose(model.weights_.sum(), 1.0)
        assert np.all(model.weights_ >= 0)

    def test_mean_objective(self, sample_returns):
        """Test FTRL with mean performance objective."""
        model = FollowTheWinner(
            strategy="ogd",
            objective=PerfMeasure.MEAN,
            learning_rate=0.05,
        )
        model.fit(sample_returns)

        assert model.weights_.shape == (3,)
        assert np.isclose(model.weights_.sum(), 1.0)
        assert np.all(model.weights_ >= 0)

    def test_partial_fit_with_objective(self, sample_returns):
        """Test partial_fit with custom objective."""
        model = FollowTheWinner(
            strategy="eg",
            objective=RiskMeasure.STANDARD_DEVIATION,
            learning_rate=0.1,
        )

        # Fit incrementally (one row at a time)
        for returns_t in sample_returns:
            model.partial_fit(returns_t)

        assert model.weights_.shape == (3,)
        assert np.isclose(model.weights_.sum(), 1.0)
        assert np.all(model.weights_ >= 0)

    def test_different_strategies_with_objectives(self, sample_returns):
        """Test different FTRL strategies with custom objectives."""
        strategies = ["eg", "ogd", "adagrad"]

        for strategy in strategies:
            model = FollowTheWinner(
                strategy=strategy,
                objective=RiskMeasure.VARIANCE,
                learning_rate=0.1,
            )
            model.fit(sample_returns)

            assert model.weights_.shape == (3,)
            assert np.isclose(model.weights_.sum(), 1.0, atol=1e-6)
            assert np.all(model.weights_ >= -1e-6)

    def test_warm_start_with_objective(self, sample_returns):
        """Test warm start with custom objective."""
        model = FollowTheWinner(
            strategy="eg",
            objective=RiskMeasure.VARIANCE,
            learning_rate=0.1,
            warm_start=True,
        )

        # First fit
        model.fit(sample_returns[:30])
        weights_first = model.weights_.copy()

        # Continue fitting (warm start)
        model.fit(sample_returns[30:])
        weights_second = model.weights_.copy()

        # Weights should have changed
        assert not np.allclose(weights_first, weights_second)

    def test_no_warm_start_with_objective(self, sample_returns):
        """Test without warm start resets the model."""
        model = FollowTheWinner(
            strategy="eg",
            objective=RiskMeasure.VARIANCE,
            learning_rate=0.1,
            warm_start=False,
        )

        # First fit
        model.fit(sample_returns[:30])

        # Second fit should reset
        model.fit(sample_returns[30:])
        weights = model.weights_.copy()

        assert weights.shape == (3,)


class TestObjectiveCompatibility:
    """Test compatibility of objectives with FTRL features."""

    def test_transaction_costs_with_objective(self, sample_returns):
        """Test transaction costs work with custom objectives."""
        model = FollowTheWinner(
            strategy="eg",
            objective=RiskMeasure.VARIANCE,
            learning_rate=0.1,
            transaction_costs=0.001,
        )
        model.fit(sample_returns)

        assert model.weights_.shape == (3,)
        assert np.isclose(model.weights_.sum(), 1.0)

    def test_management_fees_with_objective(self, sample_returns):
        """Test management fees work with custom objectives."""
        model = FollowTheWinner(
            strategy="eg",
            objective=RiskMeasure.VARIANCE,
            learning_rate=0.1,
            management_fees=0.0001,
        )
        model.fit(sample_returns)

        assert model.weights_.shape == (3,)
        assert np.isclose(model.weights_.sum(), 1.0)

    def test_constraints_with_objective(self, sample_returns):
        """Test weight constraints work with custom objectives."""
        model = FollowTheWinner(
            strategy="eg",
            objective=RiskMeasure.CVAR,
            learning_rate=0.1,
            min_weights=0.1,
            max_weights=0.6,
        )
        model.fit(sample_returns)

        assert model.weights_.shape == (3,)
        assert np.all(model.weights_ >= 0.1 - 1e-6)
        assert np.all(model.weights_ <= 0.6 + 1e-6)


class TestObjectivePerformance:
    """Test that objectives lead to sensible portfolio behavior."""

    def test_variance_minimization(self):
        """Test that variance objective leads to lower variance portfolios."""
        np.random.seed(42)
        # Create returns where asset 0 has high variance, assets 1,2 have low variance
        n_obs = 100
        returns = np.zeros((n_obs, 3))
        returns[:, 0] = np.random.randn(n_obs) * 0.05  # High variance
        returns[:, 1] = np.random.randn(n_obs) * 0.01  # Low variance
        returns[:, 2] = np.random.randn(n_obs) * 0.01  # Low variance

        model = FollowTheWinner(
            strategy="ogd",
            objective=RiskMeasure.VARIANCE,
            learning_rate=0.05,
        )
        model.fit(returns)

        # Variance objective should avoid the high-variance asset
        assert model.weights_[0] < 0.5  # Asset 0 should have lower weight

    def test_mean_maximization(self):
        """Test that mean objective favors assets with higher returns."""
        np.random.seed(42)
        # Create returns where asset 0 has high mean, others have low mean
        n_obs = 100
        returns = np.zeros((n_obs, 3))
        returns[:, 0] = np.random.randn(n_obs) * 0.02 + 0.01  # High mean
        returns[:, 1] = np.random.randn(n_obs) * 0.02 - 0.005  # Low mean
        returns[:, 2] = np.random.randn(n_obs) * 0.02 - 0.005  # Low mean

        model = FollowTheWinner(
            strategy="ogd",
            objective=PerfMeasure.MEAN,
            learning_rate=0.1,
        )
        model.fit(returns)

        # Mean objective should favor the high-return asset
        assert model.weights_[0] > 0.4  # Asset 0 should have higher weight
