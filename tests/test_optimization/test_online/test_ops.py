import numpy as np
import pytest

from skfolio.optimization.online import OPS, OnlineMethod, UpdateRule


@pytest.fixture
def X_small_single(X_small):
    return X_small.iloc[[0], :]


def _check_box_budget(
    w: np.ndarray, lower: float, upper: float, budget: float, atol=1e-8
):
    assert np.isfinite(w).all(), "Infinite weights!"
    assert np.all(abs(np.sum(w) - budget) <= 1e-6), "budget constraint not respected"
    assert np.all(np.max(w) - upper <= atol), "box constraint (upper) not respected"
    assert np.all(np.min(w) - lower >= -atol), "box constraint (lower) not respected"


def _group_sum(
    weights: np.ndarray, groups: list[list[str]], group_name: str, axis: int
) -> float:
    labels = groups[axis]
    mask = np.array([g == group_name for g in labels], dtype=bool)
    return float(np.sum(weights[mask]))


def test_partial_fit(X_small_single):
    est = OPS()
    ptf = est.partial_fit(X_small_single)
    assert ptf.weights_.shape == (X_small_single.shape[1],)
    _check_box_budget(est.weights_, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("rule", list(UpdateRule))
def test_update_rules_constraints(rule, X_small_single):
    est = OPS(update_rule=rule)
    est.partial_fit(X_small_single)
    _check_box_budget(est.weights_, 0.0, 1.0, 1.0)


@pytest.mark.parametrize("method", [OnlineMethod.HEDGE, OnlineMethod.BUY_AND_HOLD])
def test_methods_basic_validity(method, X_small_single):
    # Keep runtime low for heavier methods
    est = OPS(method=method, universal_n_samples=50)
    est.partial_fit(X_small_single)
    _check_box_budget(est.weights_, 0.0, 1.0, 1.0)

    # UNIVERSAL internals sanity: w == M @ alpha (then projected)
    if method == OnlineMethod.UNIVERSAL:
        M = est._loss_method._experts
        alpha = est._loss_method._alpha
        w_raw = M @ alpha
        # May be identical if only box+sum fast path is used
        assert np.allclose(np.sum(w_raw), 1.0, atol=1e-6)
        assert est.weights_.shape == w_raw.shape


def test_turnover_projection(X_small):
    max_turnover = 1
    n = X_small.shape[1]
    prev = np.ones(n) / n
    est = OPS(previous_weights=prev, max_turnover=max_turnover).fit(X_small.iloc[:1])
    l1 = np.abs(est.weights_ - prev).sum()
    assert l1 <= max_turnover + 1e-8
    _check_box_budget(est.weights_, 0.0, 1.0, 1.0)


def test_convex_fallback_groups_linear(X_small_single, groups, linear_constraints):
    # Force convex path via groups/linear constraints
    budget = 0.9
    est = OPS(
        method=OnlineMethod.HEDGE,
        min_weights=0.0,
        max_weights=0.8,
        budget=budget,
        groups=groups,
        linear_constraints=linear_constraints,
    )
    est.partial_fit(X_small_single)
    w = est.weights_

    # Box + budget
    _check_box_budget(w, 0.0, 0.8, budget)

    # Check a subset of linear constraints semantics
    eq_sum = _group_sum(w, groups, "Equity", 0)
    bond_sum = _group_sum(w, groups, "Bond", 0)
    assert eq_sum <= 0.5 * bond_sum + 1e-6

    us_sum = _group_sum(w, groups, "US", 1)
    assert us_sum >= 0.1 - 1e-6

    europe_sum = _group_sum(w, groups, "Europe", 1)
    fund_sum = _group_sum(w, groups, "Fund", 0)
    assert europe_sum >= 0.5 * fund_sum - 1e-6


@pytest.mark.parametrize(
    "lower,upper,budget",
    [(0.0, 0.5, 0.8)],
)
def test_bounds_and_budget(lower, upper, budget, X_small_single):
    est = OPS(min_weights=lower, max_weights=upper, budget=budget)
    est.partial_fit(X_small_single)
    _check_box_budget(est.weights_, lower, upper, budget)


@pytest.mark.parametrize("rule", [UpdateRule.EMD, UpdateRule.OGD])
def test_warm_start_and_initial_weights_buy_and_hold(rule, X_small_single):
    n = X_small_single.shape[1]
    init = np.ones(n) / n
    est = OPS(
        method=OnlineMethod.BUY_AND_HOLD,
        update_rule=rule,
        initial_weights=init,
        warm_start=False,
    )
    est.partial_fit(X_small_single)
    # With zero gradients and EMD/OGD, Buy-and-Hold should keep weights
    assert np.allclose(est.weights_, init, atol=1e-10)


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
#     _check_box_budget(est.weights_, 0.0, 1.0, 1.0)


def test_partial_fit_streaming_equivalence(X_small):
    est_batch = OPS(method=OnlineMethod.HEDGE, update_rule=UpdateRule.EMD, eta0=0.3)
    est_batch.fit(X_small)

    est_stream = OPS(method=OnlineMethod.HEDGE, update_rule=UpdateRule.EMD, eta0=0.3)
    for i in range(len(X_small)):
        est_stream.partial_fit(X_small.iloc[[i], :])

    # Expect the same final weights (same updates in order)
    np.testing.assert_allclose(est_stream.weights_, est_batch.weights_, atol=1e-10)


def test_convex_variance_bound(X_small):
    Sigma = np.cov(X_small.to_numpy().T)
    # Loose bound to ensure feasibility
    var_bound = float(np.trace(Sigma)) / Sigma.shape[0]
    est = OPS(
        method=OnlineMethod.HEDGE,
        covariance=Sigma,
        variance_bound=var_bound * 2.0,
        min_weights=0.0,
        max_weights=0.5,
        budget=0.9,
    )
    est.fit(X_small)
    w = est.weights_
    _check_box_budget(w, 0.0, 0.5, 0.9)
    quad = float(w @ Sigma @ w)
    assert quad <= var_bound * 2.0 + 1e-6
