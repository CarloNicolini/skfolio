import numpy as np
import pytest

from skfolio.optimization.online import FollowTheWinner
from skfolio.optimization.online._ftrl import SwordMeta, FirstOrderOCO
from skfolio.optimization.online._mirror_maps import (
    AdaptiveMahalanobisMap,
)
from skfolio.optimization.online._mixins import FTWStrategy
from skfolio.optimization.online._projection import IdentityProjector


def test_sword_var_simplex_and_zero_grad_stability():
    # Two-asset toy dataset with small returns; include a zero-gradient scenario
    X = np.array(
        [
            [0.00, 0.00],
            [0.01, -0.01],
            [0.02, -0.02],
            [-0.01, 0.01],
        ],
        dtype=float,
    )

    model = FollowTheWinner(
        strategy=FTWStrategy.SWORD_VAR,
        update_mode="omd",
        learning_rate=0.1,
        warm_start=False,
    )
    model.fit(X)

    W = model.all_weights_
    assert W.shape == (X.shape[0], X.shape[1])
    # Feasibility on [0,1], sum to 1 within projection tolerance
    assert np.all(W >= -1e-12)
    assert np.all(W <= 1 + 1e-12)
    s = np.sum(W, axis=1)
    assert np.allclose(s, 1.0, atol=1e-8)


def test_sword_meta_weights_shift_to_better_expert():
    # Build two FTRL experts with fixed current iterates, and check that SwordMeta shifts weight
    d = 3
    proj = IdentityProjector()
    # Use Euclidean maps; we'll manually set current iterations
    e1 = FirstOrderOCO(
        mirror_map=AdaptiveMahalanobisMap(),
        projector=proj,
        eta=0.1,
        predictor=None,
        mode="omd",
    )
    e2 = FirstOrderOCO(
        mirror_map=AdaptiveMahalanobisMap(),
        projector=proj,
        eta=0.1,
        predictor=None,
        mode="omd",
    )
    # Manually set experts' current decisions x_t
    e1._x_t = np.array([1.0, 0.0, 0.0])  # focus asset 0
    e2._x_t = np.array([0.0, 1.0, 0.0])  # focus asset 1

    meta = SwordMeta(experts=[e1, e2], projector=proj, eta_meta=1.0)
    # Gradient favors asset 0: <g, e1> < <g, e2>
    g = np.array([0.1, 10.0, 0.0])
    # Before step alpha uniform
    assert np.allclose(
        meta._alpha if meta._alpha is not None else [0.5, 0.5], [0.5, 0.5], atol=1e-12
    )
    _ = meta.step(g)
    # After one update, alpha should tilt towards expert 1 (asset 0 specialist),
    # because <g, e1> = 0.1 < <g, e2> = 10.0
    assert meta._alpha[0] > meta._alpha[1]


@pytest.mark.parametrize("objective", [FTWStrategy.SWORD_BEST, FTWStrategy.SWORD_PP])
def test_ops_integration_sword_best_and_pp(objective):
    rng = np.random.default_rng(0)
    T, n = 20, 5
    # Toy returns with mild nonstationarity
    base = rng.normal(0.0005, 0.01, size=(T, n))
    trend = np.linspace(-0.01, 0.01, T)[:, None]
    X = base + np.concatenate([trend, -trend, np.zeros((T, n - 2))], axis=1)

    model = FollowTheWinner(
        strategy=objective,
        update_mode="omd",
        learning_rate=0.1,
        warm_start=False,
    )
    model.fit(X)
    W = model.all_weights_
    assert W.shape == (T, n)
    # basic feasibility
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(W >= -1e-10)
    assert np.all(W <= 1 + 1e-10)
