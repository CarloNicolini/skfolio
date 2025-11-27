import numpy as np

from skfolio.optimization.online._loser import FTLStrategy, FollowTheLoser


def test_loser_no_double_counting_between_penalty_and_wealth_costs():
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

    prev = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    tc = np.array([0.001, 0.001, 0.001], dtype=float)

    # A) Penalize turnover requested, but wealth-side costs are configured
    a = FollowTheLoser(
        strategy=FTLStrategy.OLMAR,
        previous_weights=prev,
        transaction_costs=tc,
        epsilon=1.5,
    )
    # Initialize strategy
    a.partial_fit(X[0])
    a._strategy_impl.penalize_turnover = True
    a.fit(X[1:])

    # B) No gradient penalty, same wealth-side costs
    b = FollowTheLoser(
        strategy=FTLStrategy.OLMAR,
        previous_weights=prev,
        transaction_costs=tc,
        epsilon=1.5,
    )
    b.fit(X)

    # Final wealth should match when gradient penalty is auto-disabled
    assert np.isclose(a.all_wealth_[-1], b.all_wealth_[-1], rtol=1e-12, atol=1e-12)
