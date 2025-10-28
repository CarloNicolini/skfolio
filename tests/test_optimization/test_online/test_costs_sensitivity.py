import numpy as np
import pytest

from skfolio.optimization.online import FollowTheWinner, FollowTheLoser
from skfolio.optimization.online._mixins import FTWStrategy, FTLStrategy


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
    # In choppy regimes, naive significantly exceeds drift-aware on average
    assert naive >= drift - 1e-12
    assert naive - drift > 1e-3  # visible gap


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
