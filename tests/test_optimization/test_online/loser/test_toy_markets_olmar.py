import numpy as np
import pytest

from skfolio.optimization.online._benchmark import BCRP
from skfolio.optimization.online._ftloser import FTLoser


def build_market(pattern, n):
    P = np.asarray(pattern, dtype=float)
    reps = int(np.ceil(n / P.shape[0]))
    R = np.vstack([P] * reps)[:n, :]
    return R


def to_net(R):
    return R - 1.0


def cumulative_wealth_from_weights(R, W):
    dots = (W * R).sum(axis=1)
    return float(np.prod(dots, dtype=float))


def expected_bcrp(n):
    return (9.0 / 8.0) ** (n / 2.0)


def expected_olmar(name, n):
    if name == "A":
        return 9.0 / 8.0
    if name == "B":
        return (9.0 / 16.0) * (2.0 ** ((n - 4) / 2.0))
    if name == "C":
        return (9.0 / 8.0) * (2.0 ** ((n - 5) / 6.0))
    if name == "D":
        return (9.0 / 4.0) * (2.0 ** ((n - 6) / 8.0))
    if name == "E":
        return (9.0 / 2.0) * (2.0 ** ((n - 7) / 10.0))
    raise ValueError


@pytest.mark.parametrize(
    "name,pattern,k,period",
    [
        ("A", [[1, 2], [1, 0.5]], 1, 2),
        ("B", [[1, 2], [1, 2], [1, 0.5], [1, 0.5]], 2, 4),
        ("C", [[1, 2]] * 3 + [[1, 0.5]] * 3, 3, 6),
        ("D", [[1, 2]] * 4 + [[1, 0.5]] * 4, 4, 8),
        ("E", [[1, 2]] * 5 + [[1, 0.5]] * 5, 5, 10),
    ],
)
def test_bcrp_toy_markets(name, pattern, k, period):
    n = 10 * period
    R = build_market(pattern, n)
    X = to_net(R)
    bcrp = BCRP().fit(X)
    w = np.asarray(bcrp.weights_, dtype=float)
    W = np.repeat(w[None, :], n, axis=0)
    wealth = cumulative_wealth_from_weights(R, W)
    expw = expected_bcrp(n)
    assert np.isclose(wealth, expw, rtol=1e-8, atol=0)


@pytest.mark.parametrize(
    "name,pattern,k,period",
    [
        ("A", [[1, 2], [1, 0.5]], 1, 2),
        ("B", [[1, 2], [1, 2], [1, 0.5], [1, 0.5]], 2, 4),
        ("C", [[1, 2]] * 3 + [[1, 0.5]] * 3, 3, 6),
        ("D", [[1, 2]] * 4 + [[1, 0.5]] * 4, 4, 8),
        ("E", [[1, 2]] * 5 + [[1, 0.5]] * 5, 5, 10),
    ],
)
def test_olmar_toy_markets(name, pattern, k, period):
    n = 10 * period
    R = build_market(pattern, n)
    X = to_net(R)
    model = FTLoser(
        strategy="olmar1",
        epsilon=2.0,
        strategy_params={"window": k, "variant": "olps"},
        update_mode="pa",
        warm_start=False,
    )
    model.fit(X)
    W_trade = np.asarray(model.all_weights_, dtype=float)
    assert W_trade.shape == R.shape
    wealth = cumulative_wealth_from_weights(R, W_trade)
    expw = expected_olmar(name, n)
    assert np.isclose(wealth, expw, rtol=1e-6, atol=1e-10)
