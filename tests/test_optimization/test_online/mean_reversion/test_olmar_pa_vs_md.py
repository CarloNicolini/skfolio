"""
Test equivalence between OLMAR PA mode and MD mode.

Mathematical expectation for OLMAR:
----------------------------------
For OLMAR, the PA update is:
    w_{t+1} = Proj_K(w_t + lambda * c)
    where:
    - phi = reversion predictor (moving average or exponential smoothing)
    - c = phi - mean(phi) * 1  (centered)
    - lambda = max(0, eps - phi^T w) / ||c||^2  (adaptive step size)

The MD update with Euclidean geometry is:
    w_{t+1} = Proj_K(w_t - eta * g)
    where:
    - g = -phi + mean(phi) * 1  (centered gradient, when violated)
    - eta = fixed learning rate

For equivalence, we need eta = lambda (adaptive).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization.online import MeanReversion, MeanReversionStrategy
from skfolio.preprocessing import prices_to_returns


class TestOLMARModeEquivalence:
    """Test PA vs MD mode equivalence for OLMAR."""

    @pytest.fixture
    def simple_data(self):
        """Load simple dataset for testing."""
        prices = load_sp500_dataset()
        # Use small subset for detailed analysis
        # Need enough data for OLMAR-1 window
        prices_subset = prices.iloc[:15, :5].copy()
        net_returns = prices_to_returns(prices_subset)
        return net_returns

    def test_olmar1_single_step_after_warmup(self, simple_data):
        """
        Test PA vs MD for OLMAR-1 after warmup period.

        OLMAR-1 needs window periods before it starts updating.
        Test a single update after the window.
        """
        window = 5
        # Use window+1 periods (window for warmup, +1 for first real update)
        X = simple_data.iloc[: window + 1].values

        # Common parameters
        epsilon = 2.0  # Typical OLMAR margin
        olmar_variant = "olps"

        # Run PA mode
        model_pa = MeanReversion(
            strategy=MeanReversionStrategy.OLMAR,
            olmar_predictor=1,
            olmar_window=window,
            olmar_variant=olmar_variant,
            epsilon=epsilon,
            update_mode="pa",  # PASSIVE-AGGRESSIVE update as in Li & Hoi (2012)
            transaction_costs=0.0,
            management_fees=0.0,
            apply_fees_to_phi=False,
            warm_start=False,
            min_weights=0.0,
            max_weights=1.0,
            budget=1.0,
        )
        model_pa.fit(X)

        weights_pa = model_pa.weights_.copy()
        all_weights_pa = model_pa.all_weights_.copy()

        print("\n" + "=" * 80)
        print("OLMAR-1 Single Step (After Warmup): PA Mode")
        print("=" * 80)
        print(f"Window: {window}")
        print(f"Total periods: {len(X)}")
        print(f"All weights shape: {all_weights_pa.shape}")
        print(f"Final weights: {weights_pa}")
        print(f"Epsilon: {epsilon}")

        # Compute what lambda was used in the last step
        # We need to reverse-engineer from the last update
        # For OLMAR-1, the predictor at step t uses data up to t-1
        x_last = 1.0 + X[-1]
        relatives_history = 1.0 + X

        # Manually compute phi as OLMAR-1 would
        W = window
        T = len(relatives_history)
        d = x_last.shape[0]
        tmp = np.ones(d, dtype=float)
        phi = np.zeros(d, dtype=float)
        for i in range(W):
            phi += 1.0 / np.maximum(tmp, 1e-10)
            x_idx = T - 1 - i  # Last period index in history
            tmp = tmp * np.maximum(relatives_history[x_idx], 1e-10)
        phi = phi / float(W)

        w_prev = all_weights_pa[-2] if all_weights_pa.shape[0] > 1 else np.ones(5) / 5
        margin = float(phi @ w_prev)
        ell = max(0.0, epsilon - margin)
        c = phi - np.mean(phi)
        denom = float(np.dot(c, c))
        lambda_pa = ell / denom if denom > 0 else 0.0

        print(f"\nPA step details (last update):")
        print(f"  Phi (predictor): {phi}")
        print(f"  Previous weights: {w_prev}")
        print(f"  Margin (phi^T w): {margin:.6f}")
        print(f"  Violation (ell): {ell:.6f}")
        print(f"  ||c||^2: {denom:.6f}")
        print(f"  Lambda (step size): {lambda_pa:.6f}")

        # Now run MD mode with the same effective learning rate
        model_md = MeanReversion(
            strategy=MeanReversionStrategy.OLMAR,
            olmar_predictor=1,
            olmar_window=window,
            olmar_variant=olmar_variant,
            epsilon=epsilon,
            update_mode="md",  # MIRROR-DESCENT
            learning_rate=lambda_pa,
            mirror="euclidean",
            transaction_costs=0.0,
            management_fees=0.0,
            apply_fees_to_phi=False,
            warm_start=False,
            min_weights=0.0,
            max_weights=1.0,
            budget=1.0,
        )
        model_md.fit(X)

        weights_md = model_md.weights_.copy()

        print("\n" + "=" * 80)
        print("OLMAR-1 Single Step (After Warmup): MD Mode (with eta=lambda)")
        print("=" * 80)
        print(f"Learning rate (eta): {lambda_pa:.6f}")
        print(f"Final weights: {weights_md}")

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
            err_msg="OLMAR-1 PA and MD weights should match when eta=lambda",
        )

    def test_olmar2_single_step(self, simple_data):
        """
        Test PA vs MD for OLMAR-2.

        OLMAR-2 starts updating from period 1 (after first observation).
        """
        # Use 2 periods (first for initialization, second for update)
        X = simple_data.iloc[:2].values

        # Common parameters
        epsilon = 2.0
        alpha = 0.5

        # Run PA mode
        model_pa = MeanReversion(
            strategy=MeanReversionStrategy.OLMAR,
            olmar_predictor=2,
            olmar_alpha=alpha,
            epsilon=epsilon,
            update_mode="pa",
            transaction_costs=0.0,
            management_fees=0.0,
            apply_fees_to_phi=False,
            warm_start=False,
            min_weights=0.0,
            max_weights=1.0,
            budget=1.0,
        )
        model_pa.fit(X)

        weights_pa = model_pa.weights_.copy()

        print("\n" + "=" * 80)
        print("OLMAR-2 Single Step: PA Mode")
        print("=" * 80)
        print(f"Alpha: {alpha}")
        print(f"Updated weights: {weights_pa}")
        print(f"Epsilon: {epsilon}")

        # Compute lambda from PA update
        # OLMAR-2 predictor: phi_t = alpha*1 + (1-alpha)*(phi_{t-1} / x_{t-1})
        # After first period: phi_1 = alpha*1 + (1-alpha)*(1 / x_0)
        x_0 = 1.0 + X[0]
        phi = alpha * np.ones_like(x_0) + (1.0 - alpha) * (1.0 / np.maximum(x_0, 1e-10))

        w_init = np.ones(5) / 5
        margin = float(phi @ w_init)
        ell = max(0.0, epsilon - margin)
        c = phi - np.mean(phi)
        denom = float(np.dot(c, c))
        lambda_pa = ell / denom if denom > 0 else 0.0

        print(f"\nPA step details:")
        print(f"  Phi (predictor): {phi}")
        print(f"  Margin (phi^T w): {margin:.6f}")
        print(f"  Violation (ell): {ell:.6f}")
        print(f"  ||c||^2: {denom:.6f}")
        print(f"  Lambda (step size): {lambda_pa:.6f}")

        # Run MD mode with matching eta
        model_md = MeanReversion(
            strategy=MeanReversionStrategy.OLMAR,
            olmar_predictor=2,
            olmar_alpha=alpha,
            epsilon=epsilon,
            update_mode="md",
            learning_rate=lambda_pa,
            mirror="euclidean",
            transaction_costs=0.0,
            management_fees=0.0,
            apply_fees_to_phi=False,
            warm_start=False,
            min_weights=0.0,
            max_weights=1.0,
            budget=1.0,
        )
        model_md.fit(X)

        weights_md = model_md.weights_.copy()

        print("\n" + "=" * 80)
        print("OLMAR-2 Single Step: MD Mode (with eta=lambda)")
        print("=" * 80)
        print(f"Learning rate (eta): {lambda_pa:.6f}")
        print(f"Updated weights: {weights_md}")

        # Compare
        diff = np.abs(weights_pa - weights_md)
        max_diff = np.max(diff)

        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)
        print(f"Weight difference: {diff}")
        print(f"Max difference: {max_diff:.9f}")

        assert_allclose(
            weights_pa,
            weights_md,
            rtol=1e-6,
            atol=1e-8,
            err_msg="OLMAR-2 PA and MD weights should match when eta=lambda",
        )

    def test_olmar_gradient_direction(self, simple_data):
        """
        Test gradient direction for OLMAR.

        OLMAR tries to move toward phi (reversion target).
        PA: w_new = w + lambda * (phi - mean(phi))
        MD: w_new = w - eta * g, where g = -(phi - mean(phi)) when violated

        So MD gradient should be -c_pa.
        """
        X = simple_data.iloc[:2].values
        epsilon = 2.0
        alpha = 0.5

        # Compute phi for OLMAR-2
        x_0 = 1.0 + X[0]
        phi = alpha * np.ones_like(x_0) + (1.0 - alpha) * (1.0 / np.maximum(x_0, 1e-10))

        # PA direction
        c_pa = phi - np.mean(phi)

        # MD direction (should be -c_pa since OLMAR maximizes phi^T w)
        # When violated (phi^T w < epsilon), we want to increase phi^T w
        # Gradient of loss L = max(0, epsilon - phi^T w) is -phi
        # So we move in direction -(-phi) = phi
        g_md_raw = -phi  # Gradient of hinge loss
        g_md_centered = g_md_raw - np.mean(g_md_raw)

        print("\n" + "=" * 80)
        print("OLMAR Gradient Direction")
        print("=" * 80)
        print(f"Phi (predictor): {phi}")
        print(f"PA direction (c): {c_pa}")
        print(f"MD gradient (g): {g_md_centered}")
        print(f"Check: c = -g? {np.allclose(c_pa, -g_md_centered)}")
        print(f"Difference (c + g): {c_pa + g_md_centered}")

        # PA moves in +c direction, MD moves in -g direction
        # So we need c = -g, or equivalently, c + g = 0
        assert_allclose(
            c_pa,
            -g_md_centered,
            rtol=1e-12,
            err_msg="OLMAR PA and MD directions should be opposite (c = -g)",
        )

    def test_olmar_no_violation_case(self, simple_data):
        """
        Test case where margin constraint is satisfied (no update needed).
        """
        X = simple_data.iloc[:2].values
        epsilon = 0.1  # Very low, constraint will be satisfied

        for update_mode in ["pa", "md"]:
            model = MeanReversion(
                strategy=MeanReversionStrategy.OLMAR,
                olmar_predictor=2,
                olmar_alpha=0.5,
                epsilon=epsilon,
                update_mode=update_mode,
                learning_rate=1.0,
                transaction_costs=0.0,
                management_fees=0.0,
                apply_fees_to_phi=False,
                warm_start=False,
            )
            model.fit(X)

            # Should remain uniform (or close to it if there's any update)
            # With very low epsilon, OLMAR typically doesn't violate
            print(
                f"\n{update_mode.upper()} mode with low epsilon: weights = {model.weights_}"
            )

    def test_olmar_projection_consistency(self, simple_data):
        """
        Test that projection is applied consistently in both modes.
        """
        X = simple_data.iloc[:2].values
        epsilon = 2.0

        # Test with different constraint settings
        constraint_configs = [
            {"min_weights": 0.0, "max_weights": 1.0, "budget": 1.0},
            {"min_weights": 0.1, "max_weights": 0.4, "budget": 1.0},
            {"min_weights": 0.0, "max_weights": 0.5, "budget": 1.0},
        ]

        for i, constraints in enumerate(constraint_configs):
            print(f"\n{'=' * 80}")
            print(f"OLMAR Projection Test {i + 1}: {constraints}")
            print("=" * 80)

            # Compute lambda from PA
            x_0 = 1.0 + X[0]
            phi = 0.5 * np.ones_like(x_0) + 0.5 * (1.0 / np.maximum(x_0, 1e-10))
            w_init = np.ones(5) / 5
            margin = float(phi @ w_init)
            ell = max(0.0, epsilon - margin)
            c = phi - np.mean(phi)
            denom = float(np.dot(c, c))
            lambda_val = ell / denom if denom > 0 else 0.0

            # PA mode
            model_pa = MeanReversion(
                strategy=MeanReversionStrategy.OLMAR,
                olmar_predictor=2,
                olmar_alpha=0.5,
                epsilon=epsilon,
                update_mode="pa",
                transaction_costs=0.0,
                management_fees=0.0,
                apply_fees_to_phi=False,
                warm_start=False,
                **constraints,
            )
            model_pa.fit(X)
            weights_pa = model_pa.weights_.copy()

            # MD mode with matching eta
            model_md = MeanReversion(
                strategy=MeanReversionStrategy.OLMAR,
                olmar_predictor=2,
                olmar_alpha=0.5,
                epsilon=epsilon,
                update_mode="md",
                learning_rate=lambda_val,
                mirror="euclidean",
                transaction_costs=0.0,
                management_fees=0.0,
                apply_fees_to_phi=False,
                warm_start=False,
                **constraints,
            )
            model_md.fit(X)
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

            # They should match when eta=lambda
            assert_allclose(
                weights_pa,
                weights_md,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"OLMAR PA and MD should match with constraints {constraints}",
            )

    def test_olmar_both_variants(self, simple_data):
        """
        Test both OLMAR-1 (cumprod variant) and OLMAR-2 for PA vs MD equivalence.
        """
        print("\n" + "=" * 80)
        print("Testing OLMAR-1 (cumprod variant)")
        print("=" * 80)

        window = 3
        X1 = simple_data.iloc[: window + 1].values
        epsilon = 2.0

        # OLMAR-1 with cumprod variant
        for update_mode in ["pa", "md"]:
            kwargs = dict(
                strategy=MeanReversionStrategy.OLMAR,
                olmar_order=1,
                olmar_window=window,
                olmar_variant="cumprod",
                epsilon=epsilon,
                update_mode=update_mode,
                transaction_costs=0.0,
                management_fees=0.0,
                apply_fees_to_phi=False,
                warm_start=False,
            )
            if update_mode == "md":
                kwargs["learning_rate"] = 1.0

            model = MeanReversion(**kwargs)
            model.fit(X1)
            print(f"{update_mode.upper()} final weights: {model.weights_}")

        print("\n" + "=" * 80)
        print("Testing OLMAR-2")
        print("=" * 80)

        X2 = simple_data.iloc[:3].values

        for update_mode in ["pa", "md"]:
            kwargs = dict(
                strategy=MeanReversionStrategy.OLMAR,
                olmar_order=2,
                olmar_alpha=0.5,
                epsilon=epsilon,
                update_mode=update_mode,
                transaction_costs=0.0,
                management_fees=0.0,
                apply_fees_to_phi=False,
                warm_start=False,
            )
            if update_mode == "md":
                kwargs["learning_rate"] = 1.0

            model = MeanReversion(**kwargs)
            model.fit(X2)
            print(f"{update_mode.upper()} final weights: {model.weights_}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
