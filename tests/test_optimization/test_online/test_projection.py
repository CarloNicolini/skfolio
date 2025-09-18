import numpy as np

from skfolio.optimization.online._projection import (
    AutoProjector,
    ProjectionConfig,
    project_box_and_sum,
    project_with_turnover,
)


def test_project_box_and_sum_identity():
    y = np.array([0.9, 0.1])
    w = project_box_and_sum(y, lower=0.0, upper=1.0, budget=1.0)
    assert np.allclose(w, y)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= -1e-12) and np.all(w <= 1.0 + 1e-12)


def test_project_box_and_sum_symmetric_bisection():
    y = np.array([0.6, 0.6])
    # With budget=1 and [0,1] box, should project to [0.5, 0.5]
    w = project_box_and_sum(y, lower=0.0, upper=1.0, budget=1.0)
    assert np.allclose(w, np.array([0.5, 0.5]), atol=1e-12)


def test_project_box_and_sum_with_clipping():
    y = np.array([-0.1, 1.1])
    # Already sums to 1, projection should clip to [0, 1]
    w = project_box_and_sum(y, lower=0.0, upper=1.0, budget=1.0)
    assert np.allclose(w, np.array([0.0, 1.0]), atol=1e-12)


def test_project_with_turnover_zero_cap_returns_previous():
    """
    Controls that having a max turnover of 0 just returns the previous weights
    even if projected
    """
    prev = np.array([0.7, 0.3])
    w_raw = np.array([0.2, 0.8])
    w = project_with_turnover(
        w_raw=w_raw,
        previous_weights=prev,
        max_turnover=0.0,
        lower=0.0,
        upper=1.0,
        budget=1.0,
    )
    assert np.allclose(w, prev)


def test_project_with_turnover_partial_move():
    prev = np.array([0.7, 0.3])
    w_raw = np.array([0.2, 0.8])
    l1 = float(np.sum(np.abs(w_raw - prev)))  # 1.0
    cap = 0.2
    factor = cap / l1
    expected = prev + factor * (w_raw - prev)
    w = project_with_turnover(
        w_raw=w_raw,
        previous_weights=prev,
        max_turnover=cap,
        lower=0.0,
        upper=1.0,
        budget=1.0,
    )
    assert np.allclose(w, expected, atol=1e-12)
    assert np.isclose(np.sum(np.abs(w - prev)), cap, atol=1e-12)
    assert np.isclose(w.sum(), 1.0)


def test_auto_projector_fast_path():
    cfg = ProjectionConfig()
    projector = AutoProjector(cfg)
    # Projects [0.6, 0.6] to [0.5, 0.5] with default [0,1] and budget=1
    w = projector.project(np.array([0.6, 0.6]))
    assert np.allclose(w, np.array([0.5, 0.5]), atol=1e-12)
