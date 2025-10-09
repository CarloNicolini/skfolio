import numpy as np
import pytest
from tests.test_optimization.test_online.utils import (
    assert_simplex_trajectory,
    make_stationary_returns,
)

from skfolio.optimization.online import FTRLProximal, FTRLStrategy
from skfolio.optimization.online._benchmark import CRP
from skfolio.optimization.online._regret import RegretType, regret

from .utils import assert_box_budget, group_sum


@pytest.fixture
def X_small_single(X_small):
    return X_small.iloc[[0], :]


def test_partial_fit(X_small_single):
    est = FTRLProximal()
    ptf = est.partial_fit(X_small_single)
    assert ptf.weights_.shape == (X_small_single.shape[1],)
    assert_box_budget(est.weights_, 0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "method",
    [
        FTRLStrategy.EG,
        FTRLStrategy.OGD,
        FTRLStrategy.ADAGRAD,
        FTRLStrategy.ADABARRONS,
    ],
)
def test_methods_basic_validity_fit(method, X_small):
    # Keep runtime low for heavier methods
    est = FTRLProximal(strategy=method)
    est.fit(X_small)
    assert_box_budget(est.weights_, 0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "method",
    [
        FTRLStrategy.EG,
        FTRLStrategy.OGD,
        FTRLStrategy.ADAGRAD,
        FTRLStrategy.ADABARRONS,
    ],
)
def test_methods_basic_validity_partial_fit(method, X_small_single):
    # Keep runtime low for heavier methods
    est = FTRLProximal(strategy=method)
    est.partial_fit(X_small_single)
    assert_box_budget(est.weights_, 0.0, 1.0, 1.0)


def test_smooth_prediction(X_small):
    # Test that smooth prediction runs and produces different weights from vanilla
    est_vanilla = FTRLProximal(strategy=FTRLStrategy.EG, learning_rate=0.1)
    est_smooth = FTRLProximal(
        strategy=FTRLStrategy.EG, learning_rate=0.1, grad_predictor=True
    )

    est_vanilla.fit(X_small)
    est_smooth.fit(X_small)

    assert_box_budget(est_vanilla.weights_, 0.0, 1.0, 1.0)
    assert_box_budget(est_smooth.weights_, 0.0, 1.0, 1.0)

    # Weights should be different due to the optimistic term
    assert not np.allclose(est_vanilla.weights_, est_smooth.weights_), (
        "Smooth prediction weights are identical to vanilla"
    )


def test_turnover_projection(X_small):
    max_turnover = 1
    n = X_small.shape[1]
    prev = np.ones(n) / n
    est = FTRLProximal(previous_weights=prev, max_turnover=max_turnover).fit(
        X_small.iloc[:1]
    )
    l1 = np.abs(est.weights_ - prev).sum()
    assert l1 <= max_turnover + 1e-8
    assert_box_budget(est.weights_, 0.0, 1.0, 1.0)


def test_convex_fallback_groups_linear(X_small_single, groups, linear_constraints):
    # Force convex path via groups/linear constraints
    budget = 0.9
    est = FTRLProximal(
        strategy=FTRLStrategy.EG,
        min_weights=0.0,
        max_weights=0.8,
        budget=budget,
        groups=groups,
        linear_constraints=linear_constraints,
    )
    est.partial_fit(X_small_single)
    w = est.weights_

    # Box + budget
    assert_box_budget(w, 0.0, 0.8, budget)

    # Check a subset of linear constraints semantics
    eq_sum = group_sum(w, groups, "Equity", 0)
    bond_sum = group_sum(w, groups, "Bond", 0)
    assert eq_sum <= 0.5 * bond_sum + 1e-6

    us_sum = group_sum(w, groups, "US", 1)
    assert us_sum >= 0.1 - 1e-6

    europe_sum = group_sum(w, groups, "Europe", 1)
    fund_sum = group_sum(w, groups, "Fund", 0)
    assert europe_sum >= 0.5 * fund_sum - 1e-6


@pytest.mark.parametrize(
    "lower,upper,budget",
    [(0.0, 0.5, 0.8)],
)
def test_bounds_and_budget(lower, upper, budget, X_small_single):
    est = FTRLProximal(min_weights=lower, max_weights=upper, budget=budget)
    est.partial_fit(X_small_single)
    assert_box_budget(est.weights_, lower, upper, budget)


def test_warm_start_and_initial_weights(X_small_single):
    n = X_small_single.shape[1]
    init = np.random.rand(n)
    init /= np.sum(init)

    # With warm_start=False, weights should reset to a deterministic state
    est_no_warm = FTRLProximal(
        strategy=FTRLStrategy.EG, initial_weights=init, warm_start=False
    )
    est_no_warm.partial_fit(X_small_single)  # First fit uses init
    first_weights = est_no_warm.weights_.copy()
    est_no_warm.fit(X_small_single)  # Second fit should reset and give same result
    assert np.allclose(first_weights, est_no_warm.weights_)

    # With warm_start=True, weights should persist and continue updating
    est_warm = FTRLProximal(
        strategy=FTRLStrategy.EG, initial_weights=init, warm_start=True
    )
    est_warm.partial_fit(X_small_single)
    first_weights_warm = est_warm.weights_.copy()
    est_warm.partial_fit(X_small_single)
    second_weights_warm = est_warm.weights_.copy()
    assert not np.allclose(init, first_weights_warm)  # weights should have been updated
    assert not np.allclose(
        first_weights_warm, second_weights_warm
    )  # second update should differ


# def test_universal_custom_experts(X_small_single):
#     n = X_small_single.shape[1]
#     # Experts: equal-weight and first asset only
#     ew = np.ones((n, 1)) / n
#     e1 = np.zeros((n, 1))
#     e1[0, 0] = 1.0
#     M = np.concatenate([ew, e1], axis=1)
#     est = OPS(method=OnlineMethod.UNIVERSAL, experts=M)
#     est.partial_fit(X_small_single)
#     # Internals should use provided experts
#     assert est._loss._experts.shape == M.shape
#     # Result must satisfy box/budget
#     check_box_budget(est.weights_, 0.0, 1.0, 1.0)


def test_partial_fit_streaming_equivalence(X_small):
    est_batch = FTRLProximal(strategy=FTRLStrategy.EG, learning_rate=0.3)
    est_batch.fit(X_small)

    est_stream = FTRLProximal(strategy=FTRLStrategy.EG, learning_rate=0.3)
    for i in range(len(X_small)):
        est_stream.partial_fit(X_small.iloc[[i], :])

    # Expect the same final weights (same updates in order)
    np.testing.assert_allclose(est_stream.weights_, est_batch.weights_, atol=1e-10)


def test_convex_variance_bound(X_small):
    Sigma = np.cov(X_small.to_numpy().T)
    # Loose bound to ensure feasibility
    var_bound = float(np.trace(Sigma)) / Sigma.shape[0]
    est = FTRLProximal(
        strategy=FTRLStrategy.EG,
        covariance=Sigma,
        variance_bound=var_bound * 2.0,
        min_weights=0.0,
        max_weights=0.5,
        budget=0.9,
    )
    est.fit(X_small)
    w = est.weights_
    assert_box_budget(w, 0.0, 0.5, 0.9)
    quad = float(w @ Sigma @ w)
    assert quad <= var_bound * 2.0 + 1e-6


def test_eg_tilde_implementation(X_small):
    """Test EG-Tilde mixing step."""
    # With alpha=1, result should be uniform portfolio
    est_uniform = FTRLProximal(
        strategy=FTRLStrategy.EG, eg_tilde=True, eg_tilde_alpha=1.0
    ).fit(X_small)
    n_assets = X_small.shape[1]
    uniform = np.ones(n_assets) / n_assets
    np.testing.assert_allclose(est_uniform.weights_, uniform, atol=1e-8)

    # With alpha=0, result should be same as standard EG
    est_eg = FTRLProximal(strategy=FTRLStrategy.EG, eg_tilde=False).fit(X_small)
    est_no_mix = FTRLProximal(
        strategy=FTRLStrategy.EG, eg_tilde=True, eg_tilde_alpha=0.0
    ).fit(X_small)
    np.testing.assert_allclose(est_eg.weights_, est_no_mix.weights_, atol=1e-8)

    # Test with a callable alpha
    alpha_schedule = lambda t: 1.0 / t if t > 0 else 1.0
    est_callable = FTRLProximal(
        strategy=FTRLStrategy.EG, eg_tilde=True, eg_tilde_alpha=alpha_schedule
    ).fit(X_small)
    assert_box_budget(est_callable.weights_, 0.0, 1.0, 1.0)
    assert not np.allclose(est_callable.weights_, est_eg.weights_)


@pytest.mark.parametrize(
    "method",
    [
        FTRLStrategy.EG,
        FTRLStrategy.OGD,
        FTRLStrategy.ADAGRAD,
        FTRLStrategy.ADABARRONS,
    ],
)
def test_ftrl_vs_omd_mode(method, X_small):
    """Test that FTRL and OMD modes run and produce different results."""
    # OMD mode (default)
    est_omd = FTRLProximal(strategy=method, update_mode=False)
    est_omd.fit(X_small)
    assert_box_budget(est_omd.weights_, 0.0, 1.0, 1.0)

    # FTRL mode
    est_ftrl = FTRLProximal(strategy=method, update_mode=True)
    est_ftrl.fit(X_small)
    assert_box_budget(est_ftrl.weights_, 0.0, 1.0, 1.0)

    # The algorithms are different, so their weights should not be close
    assert not np.allclose(est_omd.weights_, est_ftrl.weights_), (
        f"FTRL and OMD weights are identical for method {method.value}"
    )


def test_learning_rate_callable(X_small):
    """Test that a callable learning_rate runs correctly."""
    # Constant learning_rate
    est_const = FTRLProximal(strategy=FTRLStrategy.EG, learning_rate=0.1).fit(X_small)
    assert_box_budget(est_const.weights_, 0.0, 1.0, 1.0)

    # Equivalent callable
    est_callable_equiv = FTRLProximal(
        strategy=FTRLStrategy.EG, learning_rate=lambda t: 0.1
    ).fit(X_small)
    np.testing.assert_allclose(
        est_const.weights_, est_callable_equiv.weights_, atol=1e-8
    )

    # Time-varying callable
    lr_fn = lambda t: 1.0 / (t + 10)
    est_callable_varied = FTRLProximal(
        strategy=FTRLStrategy.EG, learning_rate=lr_fn
    ).fit(X_small)
    assert_box_budget(est_callable_varied.weights_, 0.0, 1.0, 1.0)

    # Result should be different from constant regularization
    assert not np.allclose(est_const.weights_, est_callable_varied.weights_)


@pytest.mark.parametrize(
    "objective, ftrl_flag, min_final_weight",
    [
        (FTRLStrategy.EG, False, 0.95),
        (FTRLStrategy.EG, True, 0.95),
        (FTRLStrategy.OGD, False, 0.80),
        (FTRLStrategy.ADAGRAD, False, 0.85),
        (FTRLStrategy.ADABARRONS, True, 0.85),
    ],
)
def test_convergence_to_best_asset_under_stationary_env(
    objective, ftrl_flag, min_final_weight
):
    # In a stationary environment with one dominant asset, OPS should converge most mass to it.
    X = make_stationary_returns(T=250, gap=0.01, n=2)
    est = FTRLProximal(
        strategy=objective,
        update_mode=ftrl_flag,
        learning_rate=0.2,
        warm_start=False,
    )
    est.fit(X)
    W = est.all_weights_
    assert_simplex_trajectory(W)
    # last weights put most mass on the dominant asset
    assert W[-1, 0] >= min_final_weight


def test_static_regret_against_best_crp():
    # Compare against a fixed CRP comparator [1, 0] instead of solving BCRP (avoids cvxpy solver here)
    X = make_stationary_returns(T=200, gap=0.01, n=2)
    est = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode=True,
        learning_rate=0.2,
        warm_start=False,
        portfolio_params={},
    )
    comp = CRP(weights=np.array([1.0, 0.0]), portfolio_params={})
    r = regret(
        estimator=est,
        X=X,
        comparator=comp,
        regret_type=RegretType.STATIC,
        average="final",
    )
    # Final average regret should be small in this easy environment
    assert r[-1] <= 0.01  # 1% average regret is a conservative bound


def test_smooth_prediction_accelerates_initial_adaptation():
    # Optimistic (last-gradient) prediction should speed up the initial allocation shift
    T = 12
    X = make_stationary_returns(T=T, gap=0.02, n=2)
    base = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode=False,
        learning_rate=0.3,
        grad_predictor=False,
        warm_start=False,
    )
    opti = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode=False,
        learning_rate=0.3,
        grad_predictor=True,
        warm_start=False,
    )

    base.fit(X)
    opti.fit(X)
    # Compare weight on asset 0 at an early time (e.g., t=3). Optimistic should be larger or equal.
    assert opti.all_weights_[2, 0] >= base.all_weights_[2, 0] - 1e-12


def test_eg_tilde_mixing_with_uniform():
    # EG-tilde mixes the EG step with the uniform portfolio
    X = make_stationary_returns(T=1, gap=0.01, n=3)
    pure = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode=False,
        learning_rate=0.5,
        eg_tilde=False,
        warm_start=False,
    )
    mix = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode=False,
        learning_rate=0.5,
        eg_tilde=True,
        eg_tilde_alpha=0.5,
        warm_start=False,
    )

    pure.fit(X)
    mix.fit(X)

    w_pure = pure.all_weights_[-1]
    w_mix = mix.all_weights_[-1]
    uniform = np.ones_like(w_pure) / w_pure.size
    w_expected = 0.5 * w_pure + 0.5 * uniform
    # Projection preserves the simplex so we expect exact match up to numerical error
    np.testing.assert_allclose(w_mix, w_expected, atol=1e-12, rtol=0)


def test_management_fees_flip_preference():
    # High management fee on asset 0 can flip the preference to asset 1
    T = 150
    X = make_stationary_returns(T=T, gap=0.01, n=2)
    # apply 2% fee on asset 0 every period -> effective 1.01 * (1 - 0.02) ~ 0.9898 < 1.0
    est = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode=False,
        learning_rate=0.2,
        warm_start=False,
        management_fees=np.array([0.02, 0.0]),
    )
    est.fit(X)
    W = est.all_weights_
    assert_simplex_trajectory(W)
    # Now asset 1 is more attractive net-of-fees
    assert W[-1, 1] >= 0.8


def test_partial_fit_input_and_warnings_on_sample_weight_and_nonpositive_return():
    # partial_fit must accept a single row; multiple rows should raise
    X = make_stationary_returns(T=5, gap=0.01, n=2)
    est = FTRLProximal(
        strategy=FTRLStrategy.EG, update_mode=False, learning_rate=0.2, warm_start=False
    )

    # sample_weight warning
    with pytest.warns(UserWarning, match="sample_weight is ignored"):
        est.partial_fit(X[0:1, :], sample_weight=np.ones(1))

    # Multi-row to partial_fit should error
    with pytest.raises(ValueError, match="expects a single row"):
        est.partial_fit(X[0:2, :])


def test_objective_not_implemented_raises_value_error():
    # Not supported strategy Enums should raise.
    X = make_stationary_returns(T=5, gap=0.01, n=2)
    est = FTRLProximal(
        strategy="INVALID_OBJECTIVE",
        update_mode=False,
        learning_rate=0.2,
        warm_start=False,
    )
    with pytest.raises(ValueError, match="Unknown objective"):
        est.fit(X)


@pytest.mark.xfail(
    reason="AutoProjector configuration captures previous_weights only at init and does not update per round."
)
def test_turnover_cap_enforced_each_round():
    # Expect L1 distance between consecutive weights to be capped by max_turnover.
    # Current implementation passes previous_weights to ProjectionConfig only once;
    # if not updated each round inside AutoProjector, this constraint may not be enforced.
    T = 20
    X = make_stationary_returns(T=T, gap=0.02, n=3)
    prev = np.array([1 / 3] * 3)
    est = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode=False,
        learning_rate=0.5,
        warm_start=False,
        previous_weights=prev,
        max_turnover=0.10,  # cap L1 change per round
    )
    est.fit(X)
    W = est.all_weights_
    l1_changes = np.sum(np.abs(W[1:] - W[:-1]), axis=1)
    assert np.all(l1_changes <= 0.10 + 1e-9)


def test_weights_respect_min_max_and_budget_constraints():
    # Enforce min/max weights using projector; all weights must lie in [min, max] and sum==budget
    T = 50
    X = make_stationary_returns(T=T, gap=0.01, n=3)
    est = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode=True,
        learning_rate=0.2,
        warm_start=False,
        min_weights=np.array([0.20, 0.0, 0.0]),
        max_weights=np.array([0.8, 0.8, 0.8]),
        budget=1.0,
    )
    est.fit(X)
    W = est.all_weights_
    sums = np.sum(W, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert np.all(W[:, 0] >= 0.20 - 1e-12)
    assert np.all(W <= 0.8 + 1e-12)


def test_warm_start_reset_vs_non_warm_behavior():
    # When warm_start=False, calling fit twice should not accumulate state
    X = make_stationary_returns(T=30, gap=0.02, n=2)

    est_cold = FTRLProximal(
        strategy=FTRLStrategy.EG, update_mode=False, learning_rate=0.2, warm_start=False
    )
    est_cold.fit(X)
    W1 = est_cold.all_weights_.copy()
    est_cold.fit(X)  # re-fit from scratch
    W2 = est_cold.all_weights_.copy()
    np.testing.assert_allclose(W1, W2, atol=1e-12, rtol=0)

    # With warm_start=True, second fit continues from previous state, yielding different path
    est_warm = FTRLProximal(
        strategy=FTRLStrategy.EG, update_mode=False, learning_rate=0.2, warm_start=True
    )
    est_warm.fit(X)
    W3 = est_warm.all_weights_.copy()
    est_warm.fit(X)
    W4 = est_warm.all_weights_.copy()
    # Paths differ since state carries over
    assert not np.allclose(W3, W4, atol=1e-12, rtol=0)
