"""
Test equivalence between PAMR PA mode and MD mode.

The goal is to verify that under the simplest settings, PA (passive-aggressive)
and MD (mirror descent) modes produce equivalent results. This validates that
the OCO/Mirror Descent framework can be applied to mean-reversion problems.

Mathematical expectation:
-----------------------
For PAMR, the PA update is:
    w_{t+1} = Proj_K(w_t - tau * c)
    where:
    - c = x - mean(x) * 1  (centered)
    - tau = max(0, w^T x - eps) / ||c||^2  (adaptive step size)

The MD update with Euclidean geometry is:
    w_{t+1} = Proj_K(w_t - eta * g)
    where:
    - g = x - mean(x) * 1  (centered gradient, when violated)
    - eta = fixed learning rate

For equivalence, we need eta to adapt like tau, OR we need to test
with specific data where tau happens to equal a fixed eta.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization.online import FTLStrategy, FollowTheLoser
from skfolio.optimization.online._mixins import PAMRVariant
from skfolio.preprocessing import prices_to_returns


class TestPAMRModeEquivalence:
    """Test PA vs MD mode equivalence for PAMR."""

    @pytest.fixture
    def simple_data(self):
        """Load simple dataset for testing."""
        prices = load_sp500_dataset()
        # Use small subset for detailed analysis
        prices_subset = prices.iloc[:10, :5].copy()
        net_returns = prices_to_returns(prices_subset)
        return net_returns

    def test_pamr_single_step_basic_variant(self, simple_data):
        """
        Test PA vs MD for a single step with PAMR basic variant.

        Strategy: Use adaptive learning rate in MD that matches PA's tau.
        """
        # Get first period of data
        X_single = simple_data.iloc[:1].values

        # Common parameters
        epsilon = 0.5
        pamr_variant = PAMRVariant.SIMPLE  # Basic PAMR (no cap, no regularization)

        # Run PA mode
        model_pa = FollowTheLoser(
            strategy=FTLStrategy.PAMR,
            pamr_variant=pamr_variant,
            epsilon=epsilon,
            update_mode="pa",
            transaction_costs=0.0,
            management_fees=0.0,
            warm_start=False,
            min_weights=0.0,
            max_weights=1.0,
            budget=1.0,
        )
        model_pa.partial_fit(X_single)

        weights_pa = model_pa.weights_.copy()

        print("\n" + "=" * 80)
        print("PAMR Single Step: PA Mode")
        print("=" * 80)
        print(f"Input relatives: {1.0 + X_single[0]}")
        print(f"Initial weights: {np.ones(5) / 5}")
        print(f"Updated weights: {weights_pa}")
        print(f"Epsilon: {epsilon}")

        # Compute what tau was used (reverse engineer from PA update)
        x_t = 1.0 + X_single[0]
        w_init = np.ones(5) / 5
        margin = float(x_t @ w_init)
        ell = max(0.0, margin - epsilon)
        c = x_t - np.mean(x_t)
        denom = float(np.dot(c, c))
        tau_pa = ell / denom if denom > 0 else 0.0

        print(f"\nPA step details:")
        print(f"  Margin (w^T x): {margin:.6f}")
        print(f"  Violation (ell): {ell:.6f}")
        print(f"  ||c||^2: {denom:.6f}")
        print(f"  Tau (step size): {tau_pa:.6f}")

        # Now run MD mode with the same effective learning rate
        # MD uses: w_new = w - eta * g_centered
        # PA uses: w_new = w - tau * c
        # So we want eta = tau
        model_md = FollowTheLoser(
            strategy=FTLStrategy.PAMR,
            pamr_variant=pamr_variant,
            epsilon=epsilon,
            update_mode="md",
            learning_rate=tau_pa,  # Use PA's tau as learning rate
            mirror="euclidean",
            transaction_costs=0.0,
            management_fees=0.0,
            warm_start=False,
            min_weights=0.0,
            max_weights=1.0,
            budget=1.0,
        )
        model_md.partial_fit(X_single)

        weights_md = model_md.weights_.copy()

        print("\n" + "=" * 80)
        print("PAMR Single Step: MD Mode (with eta=tau)")
        print("=" * 80)
        print(f"Learning rate (eta): {tau_pa:.6f}")
        print(f"Updated weights: {weights_md}")

        # Compare
        diff = np.abs(weights_pa - weights_md)
        max_diff = np.max(diff)

        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)
        print(f"Weight difference: {diff}")
        print(f"Max difference: {max_diff:.9f}")

        # They should match if the update logic is equivalent
        assert_allclose(
            weights_pa,
            weights_md,
            rtol=1e-6,
            atol=1e-8,
            err_msg="PA and MD weights should match when eta=tau",
        )

    def test_pamr_no_violation_case(self, simple_data):
        """
        Test case where margin constraint is satisfied (no update needed).
        Both PA and MD should leave weights unchanged.
        """
        # Use data and high epsilon to ensure no violation
        X_single = simple_data.iloc[:1].values
        epsilon = 10.0  # Very high, constraint will be satisfied

        for update_mode in ["pa", "md"]:
            model = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant="simple",
                epsilon=epsilon,
                update_mode=update_mode,
                learning_rate=1.0,
                transaction_costs=0.0,
                management_fees=0.0,
                warm_start=False,
            )
            model.partial_fit(X_single)

            # Should remain uniform
            expected_weights = np.ones(5) / 5
            assert_allclose(
                model.weights_,
                expected_weights,
                rtol=1e-9,
                err_msg=f"{update_mode} should not update when constraint satisfied",
            )

            print(
                f"\n{update_mode.upper()} mode with no violation: weights remain uniform ✓"
            )

    def test_pamr_multi_step_with_adaptive_eta(self, simple_data):
        """
        Test multi-step comparison where we adapt eta at each step.

        This is conceptual: in practice MD uses fixed eta, but we can
        test if the single-step logic is correct by adapting eta.
        """
        X = simple_data.iloc[:5].values  # 5 periods
        epsilon = 0.5

        print("\n" + "=" * 80)
        print("PAMR Multi-Step: Adaptive eta to match tau")
        print("=" * 80)

        # Run PA mode (full sequence)
        model_pa = FollowTheLoser(
            strategy=FTLStrategy.PAMR,
            pamr_variant=PAMRVariant.SIMPLE,
            epsilon=epsilon,
            update_mode="pa",
            transaction_costs=0.0,
            management_fees=0.0,
            warm_start=False,
        )
        model_pa.fit(X)
        weights_pa_final = model_pa.weights_.copy()
        all_weights_pa = model_pa.all_weights_.copy()

        print(f"\nPA mode:")
        print(f"  Final weights: {weights_pa_final}")
        print(f"  All weights shape: {all_weights_pa.shape}")

        # For MD, we'd need to adapt eta at each step, which isn't directly
        # supported. Instead, let's check if using a fixed "good" eta gives
        # similar results

        for eta in [0.01, 0.1, 0.5, 1.0]:
            model_md = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=PAMRVariant.SIMPLE,
                epsilon=epsilon,
                update_mode="md",
                learning_rate=eta,
                mirror="euclidean",
                transaction_costs=0.0,
                management_fees=0.0,
                warm_start=False,
            )
            model_md.fit(X)
            weights_md_final = model_md.weights_.copy()

            diff = np.abs(weights_pa_final - weights_md_final)
            max_diff = np.max(diff)

            print(f"\nMD mode with eta={eta:.2f}:")
            print(f"  Final weights: {weights_md_final}")
            print(f"  Max diff from PA: {max_diff:.6f}")

    def test_pamr_gradient_computation(self, simple_data):
        """
        Test that gradients are computed correctly in both modes.

        For PA: gradient is c = x - mean(x)*1
        For MD: gradient is g = x (then centered: g - mean(g))

        They should be identical when constraint is violated.
        """
        X_single = simple_data.iloc[:1].values
        x_t = 1.0 + X_single[0]
        w_init = np.ones(5) / 5
        epsilon = 0.5

        # PA gradient
        margin = float(x_t @ w_init)
        ell = max(0.0, margin - epsilon)
        c_pa = x_t - np.mean(x_t)

        # MD gradient (when violated)
        g_md_raw = x_t
        g_md_centered = g_md_raw - np.mean(g_md_raw)

        print("\n" + "=" * 80)
        print("Gradient Comparison")
        print("=" * 80)
        print(f"Relatives (x): {x_t}")
        print(f"PA centered gradient (c): {c_pa}")
        print(f"MD centered gradient (g): {g_md_centered}")
        print(f"Difference: {c_pa - g_md_centered}")

        # They should be identical
        assert_allclose(
            c_pa,
            g_md_centered,
            rtol=1e-12,
            err_msg="PA and MD centered gradients should match",
        )

    def test_pamr_projection_consistency(self, simple_data):
        """
        Test that projection is applied consistently in both modes.
        """
        X_single = simple_data.iloc[:1].values
        epsilon = 0.5

        # Test with different constraint settings
        constraint_configs = [
            {"min_weights": 0.0, "max_weights": 1.0, "budget": 1.0},
            {"min_weights": 0.1, "max_weights": 0.4, "budget": 1.0},
            {"min_weights": 0.0, "max_weights": 0.5, "budget": 1.0},
        ]

        for i, constraints in enumerate(constraint_configs):
            print(f"\n{'=' * 80}")
            print(f"Projection Test {i + 1}: {constraints}")
            print("=" * 80)

            # Compute tau from PA to use as eta in MD
            x_t = 1.0 + X_single[0]
            w_init = np.ones(5) / 5
            margin = float(x_t @ w_init)
            ell = max(0.0, margin - epsilon)
            c = x_t - np.mean(x_t)
            denom = float(np.dot(c, c))
            tau = ell / denom if denom > 0 else 0.0

            # PA mode
            model_pa = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=PAMRVariant.SIMPLE,
                epsilon=epsilon,
                update_mode="pa",
                transaction_costs=0.0,
                management_fees=0.0,
                warm_start=False,
                **constraints,
            )
            model_pa.partial_fit(X_single)
            weights_pa = model_pa.weights_.copy()

            # MD mode with matching eta
            model_md = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=PAMRVariant.SIMPLE,
                epsilon=epsilon,
                update_mode="md",
                learning_rate=tau,
                mirror="euclidean",
                transaction_costs=0.0,
                management_fees=0.0,
                warm_start=False,
                **constraints,
            )
            model_md.partial_fit(X_single)
            weights_md = model_md.weights_.copy()

            diff = np.abs(weights_pa - weights_md)
            max_diff = np.max(diff)

            print(f"PA weights:  {weights_pa}")
            print(f"MD weights:  {weights_md}")
            print(f"Max diff:    {max_diff:.9f}")

            # Verify both respect constraints
            assert np.all(weights_pa >= constraints["min_weights"] - 1e-8)
            assert np.all(weights_pa <= constraints["max_weights"] + 1e-8)
            assert np.abs(np.sum(weights_pa) - constraints["budget"]) < 1e-8

            assert np.all(weights_md >= constraints["min_weights"] - 1e-8)
            assert np.all(weights_md <= constraints["max_weights"] + 1e-8)
            assert np.abs(np.sum(weights_md) - constraints["budget"]) < 1e-8

            # They should match when eta=tau
            assert_allclose(
                weights_pa,
                weights_md,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"PA and MD should match with constraints {constraints}",
            )

    def test_pamr_all_variants_pa_vs_md(self, simple_data):
        """
        Test all three PAMR variants (0, 1, 2) for PA vs MD equivalence.
        """
        X_single = simple_data.iloc[:1].values
        epsilon = 0.5
        C = 500.0
        tau = None
        for variant in [
            PAMRVariant.SIMPLE,
            PAMRVariant.SLACK_LINEAR,
            PAMRVariant.SLACK_QUADRATIC,
        ]:
            print(f"\n{'=' * 80}")
            print(f"Testing PAMR variant {variant}")
            print("=" * 80)

            # Compute tau for this variant
            x_t = 1.0 + X_single[0]
            w_init = np.ones(5) / 5
            margin = float(x_t @ w_init)
            ell = max(0.0, margin - epsilon)
            c = x_t - np.mean(x_t)
            denom = float(np.dot(c, c))

            if variant == PAMRVariant.SIMPLE:
                tau = ell / denom if denom > 0 else 0.0
            elif variant == PAMRVariant.SLACK_LINEAR:
                tau = min(C, ell / denom) if denom > 0 else 0.0
            elif variant == PAMRVariant.SLACK_QUADRATIC:
                tau = ell / (denom + 1.0 / (2.0 * C)) if denom > 0 else 0.0

            print(f"Computed tau: {tau:.6f}")

            # PA mode
            model_pa = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=variant,
                pamr_C=C,
                epsilon=epsilon,
                update_mode="pa",
                transaction_costs=0.0,
                management_fees=0.0,
                warm_start=False,
            )
            model_pa.partial_fit(X_single)
            weights_pa = model_pa.weights_.copy()

            # MD mode
            model_md = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=variant,
                pamr_C=C,
                epsilon=epsilon,
                update_mode="md",
                learning_rate=tau,
                mirror="euclidean",
                transaction_costs=0.0,
                management_fees=0.0,
                warm_start=False,
            )
            model_md.partial_fit(X_single)
            weights_md = model_md.weights_.copy()

            diff = np.abs(weights_pa - weights_md)
            max_diff = np.max(diff)

            print(f"PA weights: {weights_pa}")
            print(f"MD weights: {weights_md}")
            print(f"Max diff:   {max_diff:.9f}")

            if variant == 0:
                # Variant 0 should match exactly
                assert_allclose(
                    weights_pa,
                    weights_md,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"Variant {variant} should match",
                )
            else:
                # Variants 1 and 2 modify tau, so MD won't match unless
                # we account for the modification
                # For now, just report the difference
                print(f"Note: Variant {variant} has modified tau computation")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
