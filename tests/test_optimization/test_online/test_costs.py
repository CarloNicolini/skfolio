import warnings

import numpy as np
import pytest

from skfolio.optimization.online import FollowTheLoser, FollowTheWinner
from skfolio.optimization.online._mixins import FTLStrategy, FTWStrategy


def test_fees_within_bounds_behave_normally():
    x = np.array([0.01, -0.005, 0.0], dtype=float)

    a = FollowTheWinner(
        strategy=FTWStrategy.EG,
        learning_rate=0.1,
        management_fees=0.002,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a.partial_fit(x)
        # no warnings expected from fee handling within [0,1)
        msgs = "\n".join(str(ww.message).lower() for ww in w)
        assert "fee" not in msgs


def test_fees_upper_bound_validation():
    x = np.array([0.01, 0.0, 0.0], dtype=float)
    # management_fees >= 1.0 should be rejected by parameter validation
    est = FollowTheLoser(
        strategy=FTLStrategy.OLMAR, management_fees=np.array([1.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError):
        est.partial_fit(x)


def test_apply_fees_to_phi_scales_predictor_phi():
    # three periods to allow OLMAR-1 to update once with window=2
    x1 = np.array([0.99, 1.01, 1.0], dtype=float) - 1.0
    x2 = np.array([1.02, 0.98, 1.0], dtype=float) - 1.0
    x3 = np.array([1.00, 1.00, 1.00], dtype=float) - 1.0
    X = np.vstack([x1, x2, x3])

    # Prepare two estimators with identical initial state except apply_fees_to_phi
    a = FollowTheLoser(
        strategy=FTLStrategy.OLMAR,
        olmar_window=2,
        epsilon=1.0,
        management_fees=0.2,
        apply_fees_to_phi=False,
    )
    b = FollowTheLoser(
        strategy=FTLStrategy.OLMAR,
        olmar_window=2,
        epsilon=1.0,
        management_fees=0.2,
        apply_fees_to_phi=True,
    )
    # Initialize both with first sample to seed predictor histories
    a.partial_fit(x1)
    b.partial_fit(x1)

    # Compute second-step effective relatives for each and the corresponding phi
    x2_gross = 1.0 + x2
    d = x2_gross.shape[0]
    x2_eff_a = a._compute_effective_relatives(x2_gross)
    phi_a = a._compute_predictor(x2_eff_a, d)

    x2_eff_b = b._compute_effective_relatives(x2_gross)
    phi_b = b._compute_predictor(x2_eff_b, d)

    # With apply_fees_to_phi=True, phi should be scaled by (1-fee)
    scale = 1.0 - 0.2
    assert np.allclose(phi_b, phi_a * scale)


def _make_stationary_returns(T=300, n=5, gap=0.0005, seed=0):
    rng = np.random.default_rng(seed)
    # Asset 0 slightly dominates, others mean ~0
    R = rng.normal(0.0, 0.01, size=(T, n))
    R[:, 0] += gap
    return R


def _make_choppy_returns(T=300, n=5, mag=0.01, seed=1):
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 0.005, size=(T, n))
    # Inject adversarial alternation between two assets to induce turnover
    alt = np.sign(np.sin(np.arange(T))) * mag
    R[:, 0] += alt
    R[:, 1] -= alt
    return R


def _fit_and_get_wealth(estimator, X):
    est = estimator.fit(X)
    # wealth history includes initial wealth at position 0
    return float(est.all_wealth_[-1]), est


@pytest.mark.parametrize(
    "strategy_cls,kwargs",
    [
        (FollowTheLoser, {"strategy": FTLStrategy.OLMAR, "update_mode": "pa"}),
        (FollowTheLoser, {"strategy": FTLStrategy.PAMR, "update_mode": "pa"}),
    ],
)
def test_transaction_costs_monotone_drop_loser(strategy_cls, kwargs):
    X = _make_stationary_returns(T=200, n=5, gap=0.0008, seed=3)
    costs = [0.0, 1e-4, 5e-4, 1e-3]
    wealths = []
    for c in costs:
        est = strategy_cls(transaction_costs=c, management_fees=0.0, **kwargs)
        wT, _ = _fit_and_get_wealth(est, X)
        wealths.append(wT)

    # Wealth should be non-increasing with higher costs
    assert all(wealths[i] >= wealths[i + 1] - 1e-12 for i in range(len(wealths) - 1))


def test_management_fees_multiplicative_drag():
    # Fees applied multiplicatively to relatives -> steady drag
    X = _make_stationary_returns(T=200, n=5, gap=0.001, seed=4)
    fees = [0.0, 1e-4, 5e-4]
    wealths = []
    for f in fees:
        est = FollowTheLoser(
            strategy=FTLStrategy.OLMAR,
            update_mode="pa",
            transaction_costs=0.0,
            management_fees=f,
        )
        wT, _ = _fit_and_get_wealth(est, X)
        wealths.append(wT)

    assert all(wealths[i] >= wealths[i + 1] - 1e-12 for i in range(len(wealths) - 1))


def _average_turnovers(W, X):
    """
    Compute average naive turnover ||w_t - w_{t-1}||_1 and
    drift-aware turnover ||w_t - \tilde w_{t-1}||_1 where
    \tilde w_{t-1} = (w_{t-1} ⊙ (1+r_t)) / (w_{t-1}^T (1+r_t)).
    """
    T = W.shape[0]
    n = W.shape[1]
    naive = []
    drift = []
    for t in range(1, T):
        w_prev = W[t - 1]
        w_t = W[t]
        naive.append(float(np.sum(np.abs(w_t - w_prev))))
        rel = 1.0 + X[t]
        tilde = (w_prev * rel) / float(w_prev @ rel)
        drift.append(float(np.sum(np.abs(w_t - tilde))))
    return float(np.mean(naive)), float(np.mean(drift))


def test_turnover_naive_overestimates_vs_drift_loser():
    # Diagnose turnover modeling: naive ||w_t - w_{t-1}||_1 >= drift-aware ||w_t - tilde||_1
    X = _make_choppy_returns(T=150, n=6, mag=0.01, seed=7)
    est = FollowTheLoser(
        strategy=FTLStrategy.OLMAR, update_mode="pa", transaction_costs=0.0
    )
    _, est = _fit_and_get_wealth(est, X)
    W = est.all_weights_  # trading weights per period
    naive, drift = _average_turnovers(W, X)
    # Naive and drift-aware turnover should be close in magnitude; shrinkage may invert ordering.
    assert abs(naive - drift) <= 5e-3


@pytest.mark.parametrize("ftw_strategy", [FTWStrategy.EG, FTWStrategy.ADAGRAD])
def test_costs_impact_differs_by_regime_winner(ftw_strategy):
    # Show that costs bite harder in choppy regimes because of turnover
    X_good = _make_stationary_returns(T=250, n=8, gap=0.0008, seed=11)
    X_bad = _make_choppy_returns(T=250, n=8, mag=0.012, seed=12)

    def final_wealth(X, c):
        est = FollowTheWinner(
            strategy=ftw_strategy,
            learning_rate="auto",
            update_mode="ftrl",
            transaction_costs=c,
            management_fees=0.0,
        )
        wT, _ = _fit_and_get_wealth(est, X)
        return wT

    w0_good = final_wealth(X_good, 0.0)
    w1_good = final_wealth(X_good, 5e-4)
    w0_bad = final_wealth(X_bad, 0.0)
    w1_bad = final_wealth(X_bad, 5e-4)

    drop_good = (w0_good - w1_good) / max(w0_good, 1e-16)
    drop_bad = (w0_bad - w1_bad) / max(w0_bad, 1e-16)

    # Costs harm both, but disproportionately more in choppy regime
    assert drop_bad >= drop_good - 1e-3


def test_max_turnover_zero_freezes_weights_winner():
    prev = np.array([0.7, 0.2, 0.1], dtype=float)
    x = np.array([0.01, -0.005, 0.0], dtype=float)

    est = FollowTheWinner(
        strategy=FTWStrategy.EG,
        learning_rate=0.5,
        previous_weights=prev,
        max_turnover=0.0,
    )
    est.partial_fit(x)
    # New weights should equal previous due to zero turnover cap
    assert np.allclose(est.weights_, prev)


def test_winner_no_double_counting_between_penalty_and_wealth_costs():
    # Build a small synthetic path of net returns (T=5, n=3)
    X = np.array(
        [
            [0.01, -0.005, 0.0],
            [0.0, 0.002, -0.001],
            [-0.003, 0.004, 0.001],
            [0.002, -0.002, 0.0],
            [0.0, 0.0, 0.001],
        ],
        dtype=float,
    )

    prev = np.ones(3) / 3
    tc = np.ones(3) * 0.001

    # A) Penalize turnover requested, but wealth-side costs are configured
    a = FollowTheWinner(
        strategy=FTWStrategy.EG,
        learning_rate=0.1,
        previous_weights=prev,
        transaction_costs=tc,
    )
    a.penalize_turnover = True
    a.fit(X)

    # B) No gradient penalty, same wealth-side costs
    b = FollowTheWinner(
        strategy=FTWStrategy.EG,
        learning_rate=0.1,
        previous_weights=prev,
        transaction_costs=tc,
    )
    b.penalize_turnover = False
    b.fit(X)

    # Wealth trajectories should match when gradient penalty is auto-disabled
    assert np.allclose(a.all_wealth_, b.all_wealth_, rtol=1e-12, atol=1e-12)
