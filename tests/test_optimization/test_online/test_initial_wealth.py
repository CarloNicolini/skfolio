"""Tests for initial_wealth tracking in online portfolio selection."""

import numpy as np
import pytest

from skfolio.optimization.online import FollowTheLoser, FollowTheWinner


class TestInitialWealthScalar:
    """Test scalar initial_wealth tracking."""

    def test_scalar_initial_wealth_winner(self):
        """Test scalar initial_wealth with FollowTheWinner."""
        initial_wealth = 100000.0
        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            learning_rate=0.05,
        )

        # Sample data: 3 assets, 5 periods
        X = np.array(
            [
                [0.01, 0.02, -0.01],
                [0.02, -0.01, 0.01],
                [-0.01, 0.01, 0.02],
                [0.01, 0.01, 0.01],
                [0.02, 0.00, -0.01],
            ]
        )

        est.fit(X)

        # Check initial wealth is set
        assert hasattr(est, "wealth_")
        assert hasattr(est, "all_wealth_")

        # Check all_wealth_ shape (T+1: initial + 5 periods)
        assert est.all_wealth_.shape == (6,)

        # Check first wealth is initial
        assert est.all_wealth_[0] == initial_wealth

        # Check wealth is positive throughout
        assert np.all(est.all_wealth_ > 0)

        # Check final wealth is reasonable
        assert est.wealth_ > 0

    def test_scalar_initial_wealth_loser(self):
        """Test scalar initial_wealth with FollowTheLoser."""
        initial_wealth = 50000.0
        est = FollowTheLoser(
            strategy="olmar",
            initial_wealth=initial_wealth,
            olmar_window=3,
            epsilon=1.0,
        )

        # Sample data
        X = np.array(
            [
                [0.01, 0.02, -0.01],
                [0.02, -0.01, 0.01],
                [-0.01, 0.01, 0.02],
                [0.01, 0.01, 0.01],
            ]
        )

        est.fit(X)

        assert est.all_wealth_[0] == initial_wealth
        assert est.all_wealth_.shape == (5,)
        assert np.all(est.all_wealth_ > 0)

    def test_wealth_compounds_correctly(self):
        """Test wealth compounds correctly with known returns."""
        initial_wealth = 1000.0
        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            learning_rate=0.05,
        )

        # Simple scenario: uniform weights, known returns
        X = np.array(
            [
                [0.10, 0.10, 0.10],  # 10% return across all assets
            ]
        )

        est.fit(X)

        # With uniform weights (1/3 each) and 10% return: wealth = 1000 * 1.10 = 1100
        expected_final = initial_wealth * 1.10
        assert np.isclose(est.wealth_, expected_final, rtol=0.01)


class TestInitialWealthArray:
    """Test array initial_wealth tracking."""

    def test_array_initial_wealth_computes_weights(self):
        """Test array initial_wealth computes initial_weights proportionally."""
        initial_wealth = np.array([10000.0, 20000.0, 30000.0])
        total = 60000.0

        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            learning_rate=0.05,
        )

        # Before fitting, check that weights were computed proportionally
        # We need to trigger initialization first
        X = np.array([[0.01, 0.02, -0.01]])

        # Do a single partial_fit to trigger initialization
        est.partial_fit(X)

        # Check total wealth was set correctly
        # Note: all_wealth_ is only populated by fit(), not partial_fit
        # So let's check the wealth_ attribute directly
        # But we need to account for the one period of returns already applied
        # So let's just verify the initial computation differently

        # Verify by creating a fresh estimator and checking weights before any trading
        est2 = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            learning_rate=0.05,
        )

        # Initialize wealth first (which will also initialize weights from initial_wealth array)
        est2._initialize_wealth(3)

        # Check total wealth
        assert est2.wealth_ == total

        # Check weights are proportional (accounting for budget=1.0)
        expected_weights = initial_wealth / total
        assert np.allclose(est2.weights_, expected_weights, rtol=1e-6)

    def test_array_initial_wealth_with_default_budget(self):
        """Test array initial_wealth with default budget."""
        initial_wealth = np.array([10000.0, 20000.0, 30000.0])

        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            learning_rate=0.05,
        )

        X = np.array([[0.01, 0.02, -0.01]])

        est.fit(X)

        # Weights should sum to default budget (1.0)
        assert np.isclose(np.sum(est.all_weights_[0]), 1.0, rtol=1e-6)

    def test_array_initial_wealth_loser(self):
        """Test array initial_wealth with FollowTheLoser."""
        initial_wealth = np.array([5000.0, 10000.0, 15000.0])

        est = FollowTheLoser(
            strategy="pamr",
            initial_wealth=initial_wealth,
            epsilon=1.0,
        )

        X = np.array(
            [
                [0.01, 0.02, -0.01],
                [0.02, -0.01, 0.01],
            ]
        )

        est.fit(X)

        assert est.all_wealth_[0] == 30000.0
        assert est.all_wealth_.shape == (3,)


class TestInitialWealthValidation:
    """Test validation of initial_wealth parameter."""

    def test_array_weights_and_array_wealth_raises_error(self):
        """Test that both array initial_weights and initial_wealth raises ValueError."""
        with pytest.raises(ValueError, match="cannot both be arrays"):
            est = FollowTheWinner(
                strategy="eg",
                initial_weights=np.array([0.3, 0.3, 0.4]),
                initial_wealth=np.array([10000.0, 20000.0, 30000.0]),
                learning_rate=0.05,
            )
            X = np.array([[0.01, 0.02, -0.01]])
            est.fit(X)

    def test_scalar_wealth_and_array_weights_compatible(self):
        """Test scalar initial_wealth with array initial_weights is allowed."""
        initial_wealth = 100000.0
        initial_weights = np.array([0.2, 0.3, 0.5])

        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            initial_weights=initial_weights,
            learning_rate=0.05,
        )

        X = np.array([[0.01, 0.02, -0.01]])
        est.fit(X)

        # Should work without error
        assert est.all_wealth_[0] == initial_wealth
        assert np.allclose(est.all_weights_[0], initial_weights)

    def test_negative_wealth_raises_error(self):
        """Test negative initial_wealth raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            est = FollowTheWinner(
                strategy="eg",
                initial_wealth=-1000.0,
                learning_rate=0.05,
            )
            X = np.array([[0.01, 0.02, -0.01]])
            est.fit(X)

    def test_zero_wealth_raises_error(self):
        """Test zero initial_wealth raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            est = FollowTheWinner(
                strategy="eg",
                initial_wealth=0.0,
                learning_rate=0.05,
            )
            X = np.array([[0.01, 0.02, -0.01]])
            est.fit(X)

    def test_array_wealth_shape_mismatch(self):
        """Test array initial_wealth with wrong shape raises error."""
        with pytest.raises(ValueError, match="must have shape"):
            est = FollowTheWinner(
                strategy="eg",
                initial_wealth=np.array([10000.0, 20000.0]),  # Wrong shape
                learning_rate=0.05,
            )
            X = np.array([[0.01, 0.02, -0.01]])  # 3 assets
            est.fit(X)


class TestSklearnConvention:
    """Test sklearn convention: learned attributes only after fit."""

    def test_wealth_not_exists_before_fit(self):
        """Test that wealth_ doesn't exist before fitting (sklearn convention)."""
        est_winner = FollowTheWinner(
            strategy="eg",
            initial_wealth=10000.0,
            learning_rate=0.05,
        )

        est_loser = FollowTheLoser(
            strategy="olmar",
            initial_wealth=50000.0,
            olmar_window=3,
            epsilon=1.0,
        )

        # wealth_ should not exist before fitting
        assert not hasattr(est_winner, "wealth_")
        assert not hasattr(est_loser, "wealth_")

        # After fitting, wealth_ should exist
        X = np.array([[0.01, 0.02, -0.01]])
        est_winner.fit(X)
        est_loser.fit(X)

        assert hasattr(est_winner, "wealth_")
        assert hasattr(est_loser, "wealth_")
        assert est_winner.wealth_ > 0
        assert est_loser.wealth_ > 0


class TestDefaultBehavior:
    """Test default behavior when initial_wealth is not provided."""

    def test_default_wealth_is_one(self):
        """Test default initial_wealth is 1.0."""
        est = FollowTheWinner(strategy="eg", learning_rate=0.05)

        X = np.array(
            [
                [0.01, 0.02, -0.01],
                [0.02, -0.01, 0.01],
            ]
        )

        est.fit(X)

        # Default wealth should be 1.0
        assert est.all_wealth_[0] == 1.0
        assert est.all_wealth_.shape == (3,)

    def test_default_wealth_loser(self):
        """Test default initial_wealth for FollowTheLoser."""
        est = FollowTheLoser(strategy="pamr", epsilon=1.0)

        X = np.array([[0.01, 0.02, -0.01]])

        est.fit(X)

        assert est.all_wealth_[0] == 1.0


class TestWealthWithFeesAndCosts:
    """Test wealth tracking with transaction costs and management fees."""

    def test_wealth_with_transaction_costs(self):
        """Test wealth accounts for transaction costs."""
        initial_wealth = 10000.0
        transaction_costs = 0.001  # 0.1% per trade

        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            transaction_costs=transaction_costs,
            learning_rate=0.05,
        )

        X = np.array(
            [
                [0.02, 0.02, 0.02],
                [0.02, 0.02, 0.02],
            ]
        )

        est.fit(X)

        # Wealth should be less than if there were no costs
        # With 2% return and no costs: 10000 * 1.02 * 1.02 = 10404
        # With costs, should be less
        assert est.wealth_ < 10404
        assert est.wealth_ > 9000  # Still positive

    def test_wealth_with_management_fees(self):
        """Test wealth accounts for management fees."""
        initial_wealth = 10000.0
        management_fees = 0.0001  # 0.01% per period

        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            management_fees=management_fees,
            learning_rate=0.05,
        )

        X = np.array(
            [
                [0.02, 0.02, 0.02],
                [0.02, 0.02, 0.02],
            ]
        )

        est.fit(X)

        # Wealth should account for fees
        assert est.wealth_ > 9000
        assert est.wealth_ < 10500

    def test_wealth_with_both_costs_and_fees(self):
        """Test wealth with both transaction costs and management fees."""
        initial_wealth = 10000.0

        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            transaction_costs=0.001,
            management_fees=0.0001,
            learning_rate=0.05,
        )

        X = np.array(
            [
                [0.02, 0.02, 0.02],
                [0.02, 0.02, 0.02],
                [0.02, 0.02, 0.02],
            ]
        )

        est.fit(X)

        # Both costs and fees should reduce final wealth
        assert est.wealth_ > 9000
        assert est.wealth_ < 10700


class TestWarmStart:
    """Test wealth tracking with warm_start=False."""

    def test_warm_start_false_resets_wealth(self):
        """Test warm_start=False resets wealth."""
        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=10000.0,
            warm_start=False,
            learning_rate=0.05,
        )

        X = np.array([[0.01, 0.02, -0.01]])

        # First fit
        est.fit(X)
        first_wealth = est.wealth_

        # Second fit with warm_start=False should reset
        est.fit(X)
        second_wealth = est.wealth_

        # Wealth should be reset and follow same trajectory
        assert np.isclose(first_wealth, second_wealth, rtol=1e-6)

    def test_warm_start_true_preserves_wealth(self):
        """Test warm_start=True continues wealth tracking."""
        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=10000.0,
            warm_start=True,
            learning_rate=0.05,
        )

        X1 = np.array([[0.01, 0.02, -0.01]])
        X2 = np.array([[0.02, 0.01, 0.00]])

        # First fit
        est.fit(X1)
        wealth_after_first = est.wealth_

        # Second fit with warm_start=True
        est.fit(X2)
        wealth_after_second = est.wealth_

        # Wealth should continue from first fit
        assert wealth_after_second != wealth_after_first


class TestPartialFit:
    """Test wealth tracking with partial_fit."""

    def test_partial_fit_updates_wealth(self):
        """Test partial_fit updates wealth correctly."""
        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=10000.0,
            learning_rate=0.05,
        )

        X = np.array([[0.02, 0.02, 0.02]])

        # Partial fit
        est.partial_fit(X)

        # Wealth should be updated
        assert est.wealth_ > 10000.0
        assert est.wealth_ < 11000.0

    def test_sequential_partial_fits(self):
        """Test sequential partial_fit calls update wealth correctly."""
        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=10000.0,
            learning_rate=0.05,
        )

        X1 = np.array([[0.01, 0.01, 0.01]])
        X2 = np.array([[0.02, 0.02, 0.02]])

        # wealth_ should not exist before fitting (sklearn convention)
        assert not hasattr(est, "wealth_")

        est.partial_fit(X1)
        wealth_1 = est.wealth_

        est.partial_fit(X2)
        wealth_2 = est.wealth_

        # Wealth should increase with positive returns
        assert wealth_1 > 10000.0  # Compare to initial_wealth
        assert wealth_2 > wealth_1


class TestCrossValidation:
    """Test wealth tracking against manual calculations."""

    def test_manual_wealth_calculation(self):
        """Test wealth matches manual calculation for simple scenario."""
        initial_wealth = 1000.0
        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            learning_rate=0.05,
        )

        # Simple scenario with uniform weights
        X = np.array(
            [
                [0.10, 0.10, 0.10],  # 10% return
                [0.05, 0.05, 0.05],  # 5% return
            ]
        )

        est.fit(X)

        # Manual calculation (assuming approximately uniform weights)
        # Period 1: 1000 * 1.10 = 1100
        # Period 2: 1100 * 1.05 = 1155
        expected_final = 1000.0 * 1.10 * 1.05

        # Allow some tolerance due to non-uniform weights from EG
        assert np.isclose(est.wealth_, expected_final, rtol=0.05)

    def test_wealth_evolution_monotonic_with_positive_returns(self):
        """Test wealth is monotonically increasing with positive returns."""
        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=10000.0,
            learning_rate=0.05,
        )

        # All positive returns
        X = np.array(
            [
                [0.01, 0.01, 0.01],
                [0.02, 0.02, 0.02],
                [0.01, 0.01, 0.01],
            ]
        )

        est.fit(X)

        # Wealth should be monotonically increasing
        for i in range(len(est.all_wealth_) - 1):
            assert est.all_wealth_[i + 1] >= est.all_wealth_[i]


class TestBothEstimators:
    """Test consistency between FollowTheWinner and FollowTheLoser."""

    def test_winner_and_loser_both_track_wealth(self):
        """Test both estimators track wealth correctly."""
        initial_wealth = 10000.0

        winner = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            learning_rate=0.05,
        )

        loser = FollowTheLoser(
            strategy="olmar",
            initial_wealth=initial_wealth,
            olmar_window=3,
            epsilon=1.0,
        )

        X = np.array(
            [
                [0.01, 0.02, -0.01],
                [0.02, -0.01, 0.01],
                [-0.01, 0.01, 0.02],
                [0.01, 0.01, 0.01],
            ]
        )

        winner.fit(X)
        loser.fit(X)

        # Both should track wealth
        assert hasattr(winner, "wealth_")
        assert hasattr(winner, "all_wealth_")
        assert hasattr(loser, "wealth_")
        assert hasattr(loser, "all_wealth_")

        # Both should have same shape
        assert winner.all_wealth_.shape == loser.all_wealth_.shape

        # Both should start at initial_wealth
        assert winner.all_wealth_[0] == initial_wealth
        assert loser.all_wealth_[0] == initial_wealth
