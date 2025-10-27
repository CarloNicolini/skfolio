"""Tests for automatic learning rate estimation."""

import warnings

import numpy as np
import pytest

from skfolio.measures._enums import PerfMeasure, RiskMeasure
from skfolio.optimization.online._learning_rate import (
    compute_adagrad_base_learning_rate,
    compute_eg_learning_rate,
    compute_ogd_learning_rate,
    compute_prod_learning_rate,
    compute_sword_base_learning_rate,
    estimate_gradient_bound,
    estimate_simplex_diameter,
    get_auto_learning_rate,
)


class TestSimplexDiameter:
    """Test simplex diameter estimation."""

    def test_unit_budget(self):
        """Test diameter = sqrt(2) for unit budget."""
        diameter = estimate_simplex_diameter(
            n_assets=2, min_weights=0, max_weights=1, budget=1.0
        )
        assert np.isclose(diameter, np.sqrt(2.0))

    def test_half_budget(self):
        """Test diameter scales linearly with budget."""
        diameter = estimate_simplex_diameter(
            n_assets=2, min_weights=0, max_weights=1, budget=0.5
        )
        assert np.isclose(diameter, 0.5 * np.sqrt(2.0))


class TestGradientBoundEstimation:
    """Test gradient bound estimation for different objectives."""

    def test_logwealth_default(self):
        """Test G≈0.8 for logwealth with default ±40% returns."""
        G = estimate_gradient_bound(None)  # defaults to logwealth
        # With [0.6, 1.4] relatives: max(|0.6-1|, |1.4-1|) * 2 = 0.4 * 2 = 0.8
        assert np.isclose(G, 0.8)

    def test_logwealth_enum(self):
        """Test logwealth with enum."""
        G = estimate_gradient_bound(PerfMeasure.LOG_WEALTH)
        assert np.isclose(G, 0.8)

    def test_logwealth_string(self):
        """Test logwealth with string."""
        G = estimate_gradient_bound("logwealth")
        assert np.isclose(G, 0.8)

    def test_variance_no_data(self):
        """Test variance uses conservative default without data."""
        G = estimate_gradient_bound(RiskMeasure.VARIANCE)
        assert G == 0.01  # conservative default

    def test_variance_with_data(self):
        """Test variance computes G = 2||Σ||_op with historical data."""
        np.random.seed(42)
        returns = np.random.randn(100, 5) * 0.01  # 1% daily vol
        G = estimate_gradient_bound(RiskMeasure.VARIANCE, historical_returns=returns)
        # Should be small (much smaller than logwealth)
        assert 0 < G < 0.1
        # Should be roughly 2 * max eigenvalue of cov matrix
        cov = np.cov(returns, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        expected = 2.0 * np.max(np.abs(eigvals))
        assert np.isclose(G, expected)

    def test_mean_no_data(self):
        """Test mean uses conservative default without data."""
        G = estimate_gradient_bound(PerfMeasure.MEAN)
        assert G == 0.001  # conservative default

    def test_mean_with_data(self):
        """Test mean computes G = ||E[X]||_2 with historical data."""
        np.random.seed(42)
        returns = np.random.randn(100, 5) * 0.001  # small returns
        G = estimate_gradient_bound(PerfMeasure.MEAN, historical_returns=returns)
        mean_ret = np.mean(returns, axis=0)
        expected = np.linalg.norm(mean_ret, ord=2)
        assert np.isclose(G, expected)

    def test_convex_default(self):
        """Test general convex measures use G=1.0."""
        G = estimate_gradient_bound(RiskMeasure.CVAR)
        assert G == 1.0

    def test_unknown_objective_raises(self):
        """Test unknown objective raises ValueError."""
        with pytest.raises(ValueError, match="not found in MEASURE_PROPERTIES"):
            estimate_gradient_bound("unknown_objective")

    def test_variance_much_smaller_than_logwealth(self):
        """Test variance gradient is much smaller than logwealth."""
        np.random.seed(42)
        returns = np.random.randn(100, 5) * 0.01
        G_var = estimate_gradient_bound(
            RiskMeasure.VARIANCE, historical_returns=returns
        )
        G_log = estimate_gradient_bound(None)
        assert G_var < G_log / 10  # variance gradient at least 10x smaller


class TestStrategyLearningRates:
    """Test strategy-specific learning rate computation."""

    def test_ogd_decay(self):
        """Test OGD learning rate decays as 1/sqrt(t) with default (empirical) scale."""
        diameter = np.sqrt(2.0)
        gradient_bound = 0.8

        eta_1 = compute_ogd_learning_rate(1, diameter, gradient_bound)
        eta_4 = compute_ogd_learning_rate(4, diameter, gradient_bound)
        eta_100 = compute_ogd_learning_rate(100, diameter, gradient_bound)

        # Should decay as 1/sqrt(t)
        assert eta_4 < eta_1
        assert eta_100 < eta_4
        # With empirical boost (6.4x), ratios remain same
        assert np.isclose(eta_4 / eta_1, np.sqrt(2.0 / 5.0))  # sqrt((1+1)/(4+1))
        assert np.isclose(eta_100 / eta_1, np.sqrt(2.0 / 101.0))  # sqrt((1+1)/(100+1))

    def test_eg_decay_moderate_scale(self):
        """Test EG learning rate decays as sqrt(8*log(n)/t) with moderate scale."""
        n_assets = 10

        eta_1 = compute_eg_learning_rate(1, n_assets, scale="moderate")
        eta_100 = compute_eg_learning_rate(100, n_assets, scale="moderate")

        # Should decay
        assert eta_100 < eta_1
        # Check formula (8*log(n)/t with t→t+1)
        expected_1 = np.sqrt(8.0 * np.log(10) / 2)  # t=1 → t+1=2
        expected_100 = np.sqrt(8.0 * np.log(10) / 101)  # t=100 → t+1=101
        assert np.isclose(eta_1, expected_1)
        assert np.isclose(eta_100, expected_100)

    def test_prod_same_as_eg(self):
        """Test PROD uses same formula as EG."""
        n_assets = 10
        t = 5

        eta_eg = compute_eg_learning_rate(t, n_assets)
        eta_prod = compute_prod_learning_rate(t, n_assets)

        assert np.isclose(eta_eg, eta_prod)

    def test_adagrad_base_rate(self):
        """Test AdaGrad base rate = D/sqrt(n)."""
        diameter = np.sqrt(2.0)
        n_assets = 10

        eta = compute_adagrad_base_learning_rate(diameter, n_assets)
        expected = diameter / np.sqrt(n_assets)
        assert np.isclose(eta, expected)

    def test_sword_base_rate(self):
        """Test SWORD base rate = 1/sqrt(n)."""
        n_assets = 10

        eta = compute_sword_base_learning_rate(n_assets)
        expected = 1.0 / np.sqrt(n_assets)
        assert np.isclose(eta, expected)


class TestAutoLearningRate:
    """Test get_auto_learning_rate integration."""

    def test_ogd_returns_callable(self):
        """Test OGD auto returns time-varying callable."""
        lr_fn = get_auto_learning_rate("ogd", n_assets=10)
        assert callable(lr_fn)

        # Test decay
        eta_1 = lr_fn(1)
        eta_10 = lr_fn(10)
        eta_100 = lr_fn(100)

        assert eta_10 < eta_1
        assert eta_100 < eta_10

    def test_eg_returns_callable(self):
        """Test EG auto returns n/sqrt(t) with default empirical scaling."""
        lr_fn = get_auto_learning_rate("eg", n_assets=10)
        assert callable(lr_fn)

        eta_5 = lr_fn(5)
        # Default is empirical: n/sqrt(t+1) = 10/sqrt(6)
        expected = 10.0 / np.sqrt(6.0)
        assert np.isclose(eta_5, expected)

    def test_eg_theory_mode(self):
        """Test EG with theory scale returns sqrt(log(n)/t)."""
        lr_fn = get_auto_learning_rate("eg", n_assets=10, scale="theory")
        assert callable(lr_fn)

        eta_5 = lr_fn(5)
        # Theory: sqrt(log(n)/(t+1)) = sqrt(log(10)/6)
        expected = np.sqrt(np.log(10) / 6.0)
        assert np.isclose(eta_5, expected)

    def test_prod_returns_callable(self):
        """Test PROD auto returns n/sqrt(t) with default empirical scaling."""
        lr_fn = get_auto_learning_rate("prod", n_assets=10)
        assert callable(lr_fn)

        eta_5 = lr_fn(5)
        # Default is empirical: n/sqrt(t+1) = 10/sqrt(6)
        expected = 10.0 / np.sqrt(6.0)
        assert np.isclose(eta_5, expected)

    def test_adagrad_returns_callable_constant(self):
        """Test AdaGrad auto returns callable (constant)."""
        lr_fn = get_auto_learning_rate("adagrad", n_assets=10)
        assert callable(lr_fn)

        # Should return constant
        eta_1 = lr_fn(1)
        eta_100 = lr_fn(100)
        assert np.isclose(eta_1, eta_100)

    def test_sword_variants_return_callable(self):
        """Test all SWORD variants return callables."""
        for strategy in ["sword", "sword_var", "sword_small", "sword_best", "sword_pp"]:
            lr_fn = get_auto_learning_rate(strategy, n_assets=10)
            assert callable(lr_fn)

            # Should return constant
            eta_1 = lr_fn(1)
            eta_100 = lr_fn(100)
            assert np.isclose(eta_1, eta_100)

    def test_adabarrons_returns_callable(self):
        """Test AdaBARRONS returns callable."""
        lr_fn = get_auto_learning_rate("adabarrons", n_assets=10)
        assert callable(lr_fn)

        eta = lr_fn(1)
        assert np.isclose(eta, 0.1)

    def test_unknown_strategy_raises(self):
        """Test unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_auto_learning_rate("unknown_strategy", n_assets=10)

    def test_custom_budget(self):
        """Test custom budget affects diameter."""
        lr_fn_1 = get_auto_learning_rate("ogd", n_assets=10, budget=1.0)
        lr_fn_05 = get_auto_learning_rate("ogd", n_assets=10, budget=0.5)

        # With budget 0.5, diameter is half, so learning rate should be half
        eta_1 = lr_fn_1(1)
        eta_05 = lr_fn_05(1)
        assert np.isclose(eta_05, eta_1 * 0.5)

    def test_explicit_gradient_bound(self):
        """Test explicit gradient_bound overrides estimation."""
        # With explicit bound
        lr_fn_explicit = get_auto_learning_rate("ogd", n_assets=10, gradient_bound=5.0)

        # With default bound
        lr_fn_default = get_auto_learning_rate("ogd", n_assets=10)

        eta_explicit = lr_fn_explicit(1)
        eta_default = lr_fn_default(1)

        # Different gradient bounds should give different learning rates
        assert not np.isclose(eta_explicit, eta_default)

    def test_warns_on_custom_objective(self):
        """Test warning when using auto with non-logwealth objective."""
        with pytest.warns(UserWarning, match="custom objectives"):
            get_auto_learning_rate("ogd", n_assets=10, objective=RiskMeasure.VARIANCE)

    def test_no_warning_on_logwealth(self):
        """Test no warning for logwealth objective."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            # Should not raise any warning
            get_auto_learning_rate("ogd", n_assets=10, objective=None)

    def test_no_warning_with_explicit_gradient_bound(self):
        """Test no warning when explicit gradient_bound is provided."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            # Should not raise any warning even with custom objective
            get_auto_learning_rate(
                "ogd",
                n_assets=10,
                objective=RiskMeasure.VARIANCE,
                gradient_bound=1.0,
            )

    def test_eg_independent_of_objective(self):
        """Test EG learning rate is independent of objective."""
        lr_fn_log = get_auto_learning_rate("eg", n_assets=10, objective=None)

        # Suppress warning for variance
        with pytest.warns(UserWarning):
            lr_fn_var = get_auto_learning_rate(
                "eg", n_assets=10, objective=RiskMeasure.VARIANCE
            )

        # Should be the same (EG doesn't use gradient bound)
        assert np.isclose(lr_fn_log(5), lr_fn_var(5))


class TestIntegrationWithFollowTheWinner:
    """Test integration with FollowTheWinner class."""

    def test_ftw_accepts_auto(self):
        """Test FollowTheWinner accepts 'auto' learning rate."""
        from skfolio.optimization.online import FollowTheWinner

        # Should not raise
        ftw = FollowTheWinner(strategy="ogd", learning_rate="auto")
        assert ftw.learning_rate == "auto"

    def test_ftw_auto_initializes(self):
        """Test FollowTheWinner with auto learning rate initializes correctly."""
        from skfolio.optimization.online import FollowTheWinner

        ftw = FollowTheWinner(strategy="eg", learning_rate="auto")

        # Fit with some data
        X = np.random.randn(10, 5) * 0.01
        ftw.fit(X)

        # Should have initialized and fitted
        assert ftw._is_initialized
        assert hasattr(ftw, "weights_")

    def test_ftw_auto_different_strategies(self):
        """Test auto learning rate with different strategies."""
        from skfolio.optimization.online import FollowTheWinner

        X = np.random.randn(10, 5) * 0.01

        for strategy in ["ogd", "eg", "prod", "adagrad"]:
            ftw = FollowTheWinner(strategy=strategy, learning_rate="auto")
            ftw.fit(X)

            assert ftw._is_initialized
            assert hasattr(ftw, "weights_")


class TestFitPredictWithAuto:
    """Test fit_predict with auto learning rate to catch NaN issues."""

    def test_eg_fit_predict_no_nan(self):
        """Test EG with auto learning rate doesn't produce NaN weights."""
        from skfolio.optimization.online import FollowTheWinner

        np.random.seed(42)
        X = np.random.randn(20, 5) * 0.01

        ftw = FollowTheWinner(strategy="eg", learning_rate="auto")
        portfolio = ftw.fit_predict(X)

        # Check no NaN in weights
        assert not np.any(np.isnan(ftw.weights_))
        assert not np.any(np.isnan(portfolio.weights))

        # Check valid simplex constraints
        assert np.isclose(np.sum(ftw.weights_), 1.0)
        assert np.all(ftw.weights_ >= 0)
        assert np.all(ftw.weights_ <= 1)

    def test_ogd_fit_predict_no_nan(self):
        """Test OGD with auto learning rate doesn't produce NaN weights."""
        from skfolio.optimization.online import FollowTheWinner

        np.random.seed(42)
        X = np.random.randn(20, 5) * 0.01

        ftw = FollowTheWinner(strategy="ogd", learning_rate="auto")
        portfolio = ftw.fit_predict(X)

        # Check no NaN in weights
        assert not np.any(np.isnan(ftw.weights_))
        assert not np.any(np.isnan(portfolio.weights))

        # Check valid simplex constraints
        assert np.isclose(np.sum(ftw.weights_), 1.0)
        assert np.all(ftw.weights_ >= 0)
        assert np.all(ftw.weights_ <= 1)

    def test_prod_fit_predict_no_nan(self):
        """Test PROD with auto learning rate doesn't produce NaN weights."""
        from skfolio.optimization.online import FollowTheWinner

        np.random.seed(42)
        X = np.random.randn(20, 5) * 0.01

        ftw = FollowTheWinner(strategy="prod", learning_rate="auto")
        portfolio = ftw.fit_predict(X)

        # Check no NaN in weights
        assert not np.any(np.isnan(ftw.weights_))
        assert not np.any(np.isnan(portfolio.weights))

        # Check valid simplex constraints
        assert np.isclose(np.sum(ftw.weights_), 1.0, atol=1e-6)
        assert np.all(ftw.weights_ >= -1e-10)  # Allow small numerical errors
        assert np.all(ftw.weights_ <= 1.0 + 1e-10)

    def test_adagrad_fit_predict_no_nan(self):
        """Test AdaGrad with auto learning rate doesn't produce NaN weights."""
        from skfolio.optimization.online import FollowTheWinner

        np.random.seed(42)
        X = np.random.randn(20, 5) * 0.01

        ftw = FollowTheWinner(strategy="adagrad", learning_rate="auto")
        portfolio = ftw.fit_predict(X)

        # Check no NaN in weights
        assert not np.any(np.isnan(ftw.weights_))
        assert not np.any(np.isnan(portfolio.weights))

        # Check valid simplex constraints
        assert np.isclose(np.sum(ftw.weights_), 1.0)
        assert np.all(ftw.weights_ >= 0)
        assert np.all(ftw.weights_ <= 1)

    def test_all_strategies_fit_predict_no_nan(self):
        """Test all strategies with auto learning rate work correctly."""
        from skfolio.optimization.online import FollowTheWinner

        np.random.seed(42)
        X = np.random.randn(15, 5) * 0.01

        strategies = ["ogd", "eg", "prod", "adagrad", "sword_var"]

        for strategy in strategies:
            ftw = FollowTheWinner(strategy=strategy, learning_rate="auto")
            portfolio = ftw.fit_predict(X)

            # Check no NaN
            assert not np.any(np.isnan(ftw.weights_)), (
                f"{strategy} produced NaN weights"
            )
            assert not np.any(np.isnan(portfolio.weights)), (
                f"{strategy} produced NaN portfolio"
            )

            # Check constraints (with tolerance for numerical precision)
            weight_sum = np.sum(ftw.weights_)
            assert 0.99 <= weight_sum <= 1.01, f"{strategy} weights sum to {weight_sum}"
            assert np.all(ftw.weights_ >= -1e-8), f"{strategy} has negative weights"
            assert np.all(ftw.weights_ <= 1.0 + 1e-8), f"{strategy} has weights > 1"

    def test_auto_learning_rate_with_small_dataset(self):
        """Test auto learning rate works with very small datasets (edge case)."""
        from skfolio.optimization.online import FollowTheWinner

        # Single period (minimal case)
        X = np.array([[0.01, 0.02, -0.01, 0.00, 0.03]])

        ftw = FollowTheWinner(strategy="eg", learning_rate="auto")
        _ = ftw.fit_predict(X)

        assert not np.any(np.isnan(ftw.weights_))
        assert np.isclose(np.sum(ftw.weights_), 1.0)

    def test_auto_learning_rate_handles_large_moves(self):
        """Test auto learning rate handles large return movements."""
        from skfolio.optimization.online import FollowTheWinner

        np.random.seed(42)
        # Simulate volatile market with ±30% moves
        X = np.random.randn(20, 5) * 0.15

        ftw = FollowTheWinner(strategy="eg", learning_rate="auto")
        _ = ftw.fit_predict(X)

        assert not np.any(np.isnan(ftw.weights_))
        assert not np.any(np.isinf(ftw.weights_))
        assert 0.99 <= np.sum(ftw.weights_) <= 1.01

    def test_t_zero_safe_in_learning_rate(self):
        """Test that t=0 is handled safely in learning rate functions."""
        # Direct test of the fix
        from skfolio.optimization.online._learning_rate import (
            compute_eg_learning_rate,
            compute_ogd_learning_rate,
            compute_prod_learning_rate,
        )

        # These should not raise or return NaN/inf
        eta_eg = compute_eg_learning_rate(0, 10)
        assert not np.isnan(eta_eg)
        assert not np.isinf(eta_eg)
        assert eta_eg > 0

        eta_ogd = compute_ogd_learning_rate(0, np.sqrt(2), 0.8)
        assert not np.isnan(eta_ogd)
        assert not np.isinf(eta_ogd)
        assert eta_ogd > 0

        eta_prod = compute_prod_learning_rate(0, 10)
        assert not np.isnan(eta_prod)
        assert not np.isinf(eta_prod)
        assert eta_prod > 0

        # t=0 and t=1 should be different (t→t+1: 0→1, 1→2)
        eta_eg_0 = compute_eg_learning_rate(0, 10)
        eta_eg_1 = compute_eg_learning_rate(1, 10)
        assert eta_eg_0 > eta_eg_1  # η(0) = 10/√1 > η(1) = 10/√2

        eta_prod_0 = compute_prod_learning_rate(0, 10)
        eta_prod_1 = compute_prod_learning_rate(1, 10)
        assert eta_prod_0 > eta_prod_1  # η(0) = 10/√1 > η(1) = 10/√2


class TestScaleParameter:
    """Test the new scale parameter for empirically-validated learning rates."""

    def test_eg_empirical_vs_theory(self):
        """Test EG empirical scaling (n/sqrt(t)) vs theory (sqrt(log(n)/t))."""
        n_assets = 64
        t = 99  # t=99 → t+1=100

        eta_empirical = compute_eg_learning_rate(t, n_assets, scale="empirical")
        eta_theory = compute_eg_learning_rate(t, n_assets, scale="theory")
        eta_moderate = compute_eg_learning_rate(t, n_assets, scale="moderate")

        # Empirical: n/sqrt(t+1) = 64/sqrt(100) = 6.4
        assert np.isclose(eta_empirical, 64.0 / np.sqrt(100.0))

        # Theory: sqrt(log(n)/t+1) = sqrt(log(64)/100) ≈ 0.203
        assert np.isclose(eta_theory, np.sqrt(np.log(64.0) / 100.0))

        # Moderate: sqrt(8*log(n)/t+1) ≈ sqrt(8*4.159/100) = 0.577
        assert np.isclose(eta_moderate, np.sqrt(8.0 * np.log(64.0) / 100.0))

        # Empirical should be much larger than theory
        assert eta_empirical > eta_theory * 10
        assert eta_empirical > eta_moderate

    def test_prod_follows_eg_scaling(self):
        """Test PROD follows same scaling as EG."""
        n_assets = 88
        t = 0

        for scale in ["theory", "moderate", "empirical"]:
            eta_eg = compute_eg_learning_rate(t, n_assets, scale=scale)
            eta_prod = compute_prod_learning_rate(t, n_assets, scale=scale)
            assert np.isclose(eta_eg, eta_prod)

    def test_ogd_empirical_boost(self):
        """Test OGD empirical mode applies sqrt(n) boost."""
        diameter = np.sqrt(2.0)
        gradient_bound = 0.8
        t = 0

        eta_theory = compute_ogd_learning_rate(
            t, diameter, gradient_bound, scale="theory"
        )
        eta_empirical = compute_ogd_learning_rate(
            t, diameter, gradient_bound, scale="empirical"
        )

        # Empirical should have 6.4x boost
        assert np.isclose(eta_empirical, eta_theory * 6.4)

    def test_scale_parameter_validation(self):
        """Test invalid scale parameter raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scale"):
            compute_eg_learning_rate(0, 10, scale="invalid")

        with pytest.raises(ValueError, match="Unknown scale"):
            compute_prod_learning_rate(0, 10, scale="invalid")

        with pytest.raises(ValueError, match="Unknown scale"):
            compute_ogd_learning_rate(0, np.sqrt(2), 0.8, scale="invalid")

    def test_get_auto_learning_rate_with_scale(self):
        """Test get_auto_learning_rate passes scale parameter correctly."""
        n_assets = 64

        # Test empirical (default)
        lr_fn_empirical = get_auto_learning_rate(
            "eg", n_assets=n_assets, scale="empirical"
        )
        # t=0 → η = 64/√1 = 64
        assert np.isclose(lr_fn_empirical(0), 64.0)

        # Test theory
        lr_fn_theory = get_auto_learning_rate("eg", n_assets=n_assets, scale="theory")
        # t=0 → η = √(log(64)/1) ≈ 2.04
        assert np.isclose(lr_fn_theory(0), np.sqrt(np.log(64.0)))

        # Test moderate
        lr_fn_moderate = get_auto_learning_rate(
            "eg", n_assets=n_assets, scale="moderate"
        )
        # t=0 → η = √(8*log(64)/1) ≈ 5.77
        assert np.isclose(lr_fn_moderate(0), np.sqrt(8.0 * np.log(64.0)))

    def test_empirical_achieves_high_bcrp_percentage(self):
        """Test empirical scaling rationale: n/sqrt(t) >> sqrt(log(n)/t)."""
        # Test case from user's validation: ftse100 with 64 assets
        n_assets = 64

        # At t=99 (day 100):
        eta_empirical = compute_eg_learning_rate(99, n_assets, scale="empirical")
        eta_theory = compute_eg_learning_rate(99, n_assets, scale="theory")

        # Empirical should be much larger (enables 90-95% of BCRP)
        # vs theory's 70-80% of BCRP
        ratio = eta_empirical / eta_theory
        # For n=64: 64/sqrt(100) / sqrt(log(64)/100) ≈ 6.4 / 0.203 ≈ 31.5x
        assert ratio > 20  # At least 20x larger

    def test_scale_affects_all_strategies(self):
        """Test scale parameter works for all strategies that support it."""
        n_assets = 20

        # EG
        lr_eg_emp = get_auto_learning_rate("eg", n_assets, scale="empirical")
        lr_eg_theory = get_auto_learning_rate("eg", n_assets, scale="theory")
        assert lr_eg_emp(0) > lr_eg_theory(0)

        # PROD
        lr_prod_emp = get_auto_learning_rate("prod", n_assets, scale="empirical")
        lr_prod_theory = get_auto_learning_rate("prod", n_assets, scale="theory")
        assert lr_prod_emp(0) > lr_prod_theory(0)

        # OGD
        lr_ogd_emp = get_auto_learning_rate("ogd", n_assets, scale="empirical")
        lr_ogd_theory = get_auto_learning_rate("ogd", n_assets, scale="theory")
        assert lr_ogd_emp(0) > lr_ogd_theory(0)

    def test_t_plus_one_indexing(self):
        """Test that t→t+1 conversion is applied correctly."""
        n_assets = 10

        # At t=0 (0-indexed), we should use t_theory=1 (1-indexed)
        eta_0 = compute_eg_learning_rate(0, n_assets, scale="theory")
        expected_0 = np.sqrt(np.log(n_assets) / 1.0)  # t_theory = 0 + 1 = 1
        assert np.isclose(eta_0, expected_0)

        # At t=99 (0-indexed), we should use t_theory=100 (1-indexed)
        eta_99 = compute_eg_learning_rate(99, n_assets, scale="empirical")
        expected_99 = n_assets / np.sqrt(100.0)  # t_theory = 99 + 1 = 100
        assert np.isclose(eta_99, expected_99)
