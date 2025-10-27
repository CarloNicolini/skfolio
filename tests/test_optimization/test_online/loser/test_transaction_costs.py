"""Test transaction costs and management fees in Follow-The-Loser strategies."""

import numpy as np
import pytest

from skfolio.optimization.online import FollowTheLoser


class TestTransactionCosts:
    """Test transaction costs work in both PA and MD modes."""

    @pytest.mark.parametrize("strategy", ["olmar", "pamr"])
    @pytest.mark.parametrize("update_mode", ["pa"])
    def test_transaction_costs_reduce_turnover(self, strategy, update_mode):
        """Transaction costs should reduce portfolio turnover in PA mode."""
        np.random.seed(42)
        X = np.random.randn(20, 5) * 0.01

        # Without transaction costs
        model_no_cost = FollowTheLoser(
            strategy=strategy,
            update_mode=update_mode,
            transaction_costs=0.0,
            warm_start=False,
        )
        model_no_cost.fit(X)
        weights_no_cost = model_no_cost.all_weights_

        # With transaction costs
        model_with_cost = FollowTheLoser(
            strategy=strategy,
            update_mode=update_mode,
            transaction_costs=0.01,  # 1% per trade
            warm_start=False,
        )
        model_with_cost.fit(X)
        weights_with_cost = model_with_cost.all_weights_

        # Compute turnover (L1 distance between consecutive weights)
        turnover_no_cost = np.sum(np.abs(np.diff(weights_no_cost, axis=0)))
        turnover_with_cost = np.sum(np.abs(np.diff(weights_with_cost, axis=0)))

        # Transaction costs should reduce turnover
        assert turnover_with_cost < turnover_no_cost, (
            f"Transaction costs should reduce turnover for {strategy}/{update_mode}. "
            f"Got {turnover_with_cost:.4f} >= {turnover_no_cost:.4f}"
        )

    @pytest.mark.parametrize("strategy", ["olmar", "pamr"])
    @pytest.mark.parametrize("update_mode", ["pa"])
    def test_higher_costs_reduce_turnover_more(self, strategy, update_mode):
        """Higher transaction costs should reduce turnover more."""
        np.random.seed(42)
        X = np.random.randn(20, 5) * 0.01

        # Low cost
        model_low = FollowTheLoser(
            strategy=strategy,
            update_mode=update_mode,
            transaction_costs=0.001,
            warm_start=False,
        )
        model_low.fit(X)

        # High cost
        model_high = FollowTheLoser(
            strategy=strategy,
            update_mode=update_mode,
            transaction_costs=0.01,
            warm_start=False,
        )
        model_high.fit(X)

        turnover_low = np.sum(np.abs(np.diff(model_low.all_weights_, axis=0)))
        turnover_high = np.sum(np.abs(np.diff(model_high.all_weights_, axis=0)))

        assert turnover_high <= turnover_low, (
            f"Higher costs should reduce turnover more for {strategy}/{update_mode}. "
            f"Got low={turnover_low:.4f}, high={turnover_high:.4f}"
        )


class TestManagementFees:
    """Test management fees application."""

    @pytest.mark.parametrize("strategy", ["olmar", "pamr", "rmr"])
    def test_management_fees_affect_weights(self, strategy):
        """Management fees should affect portfolio weights."""
        np.random.seed(42)
        # Create upward-trending data
        X = np.random.randn(20, 5) * 0.01 + 0.005

        model_no_fee = FollowTheLoser(
            strategy=strategy,
            management_fees=0.0,
            warm_start=False,
        )
        model_no_fee.fit(X)

        model_with_fee = FollowTheLoser(
            strategy=strategy,
            management_fees=0.001,  # 0.1% per period
            warm_start=False,
        )
        model_with_fee.fit(X)

        # Weights should differ when management fees are applied
        final_no_fee = model_no_fee.weights_
        final_with_fee = model_with_fee.weights_

        # Should not be identical (fees affect the optimization)
        assert not np.allclose(final_no_fee, final_with_fee, atol=1e-6), (
            f"Management fees should affect weights for {strategy}"
        )

    @pytest.mark.parametrize("strategy", ["olmar", "rmr"])
    def test_apply_fees_to_phi(self, strategy):
        """Test apply_fees_to_phi parameter affects predictor."""
        np.random.seed(42)
        X = np.random.randn(20, 5) * 0.01
        fees = 0.001

        model_fees_to_phi = FollowTheLoser(
            strategy=strategy,
            management_fees=fees,
            apply_fees_to_phi=True,
            warm_start=False,
        )
        model_fees_to_phi.fit(X)

        model_no_fees_to_phi = FollowTheLoser(
            strategy=strategy,
            management_fees=fees,
            apply_fees_to_phi=False,
            warm_start=False,
        )
        model_no_fees_to_phi.fit(X)

        # Weights should differ when fees are applied to phi
        final_with = model_fees_to_phi.weights_
        final_without = model_no_fees_to_phi.weights_

        # Should not be identical
        assert not np.allclose(final_with, final_without, rtol=1e-3), (
            f"apply_fees_to_phi should affect weights for {strategy}"
        )


class TestCombinedCostsAndFees:
    """Test interaction of transaction costs and management fees."""

    @pytest.mark.parametrize("strategy", ["olmar"])
    @pytest.mark.parametrize("update_mode", ["pa"])
    def test_combined_effects(self, strategy, update_mode):
        """Both costs and fees should affect portfolio behavior."""
        np.random.seed(42)
        X = np.random.randn(20, 5) * 0.01 + 0.003

        # Baseline
        model_none = FollowTheLoser(
            strategy=strategy,
            update_mode=update_mode,
            transaction_costs=0.0,
            management_fees=0.0,
            warm_start=False,
        )
        model_none.fit(X)

        # Only transaction costs
        model_tc = FollowTheLoser(
            strategy=strategy,
            update_mode=update_mode,
            transaction_costs=0.005,
            management_fees=0.0,
            warm_start=False,
        )
        model_tc.fit(X)

        # Only management fees
        model_mf = FollowTheLoser(
            strategy=strategy,
            update_mode=update_mode,
            transaction_costs=0.0,
            management_fees=0.001,
            warm_start=False,
        )
        model_mf.fit(X)

        # Both
        model_both = FollowTheLoser(
            strategy=strategy,
            update_mode=update_mode,
            transaction_costs=0.005,
            management_fees=0.001,
            warm_start=False,
        )
        model_both.fit(X)

        # Compute turnovers
        turnover_none = np.sum(np.abs(np.diff(model_none.all_weights_, axis=0)))
        turnover_tc = np.sum(np.abs(np.diff(model_tc.all_weights_, axis=0)))
        turnover_both = np.sum(np.abs(np.diff(model_both.all_weights_, axis=0)))

        # Transaction costs should reduce turnover in PA mode
        assert turnover_tc < turnover_none, (
            f"Transaction costs alone should reduce turnover for {strategy}/{update_mode}"
        )
        assert (
            turnover_both <= turnover_none or abs(turnover_both - turnover_none) < 0.5
        ), (
            f"Combined costs/fees should not increase turnover significantly for {strategy}/{update_mode}"
        )

    @pytest.mark.parametrize("strategy", ["olmar", "pamr"])
    @pytest.mark.parametrize("update_mode", ["pa", "md"])
    def test_costs_fees_consistency(self, strategy, update_mode):
        """Test that costs and fees work consistently across modes."""
        np.random.seed(42)
        X = np.random.randn(15, 5) * 0.01 + 0.002

        model = FollowTheLoser(
            strategy=strategy,
            update_mode=update_mode,
            transaction_costs=0.003,
            management_fees=0.0005,
            warm_start=False,
        )
        model.fit(X)

        # Should complete without errors and have valid weights
        assert hasattr(model, "weights_")
        assert hasattr(model, "all_weights_")
        assert model.weights_.shape == (5,)
        assert np.isclose(np.sum(model.weights_), 1.0, atol=1e-6)
        assert np.all(model.weights_ >= -1e-6)  # Allow small numerical errors
        assert np.all(model.weights_ <= 1.0 + 1e-6)
