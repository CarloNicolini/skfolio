# tests/test_optimization/test_online/test_olmar1_phi_regression.py
import numpy as np

from skfolio.optimization.online._mean_reversion import MeanReversion, OLMAR1Predictor
from skfolio.optimization.online._utils import CLIP_EPSILON


def _olps_matlab_like_phi(history, W):
    # Reference implementation mimicking olmar1_expert.m
    T = len(history)
    if T < W + 1:
        return history[-1].copy()
    d = history[-1].shape[0]
    tmp = np.ones(d, dtype=float)
    phi = np.zeros(d, dtype=float)
    for i in range(W):
        phi += 1.0 / np.maximum(tmp, CLIP_EPSILON)
        x_idx = T - i - 1
        tmp = tmp * np.maximum(history[x_idx], CLIP_EPSILON)
    return phi * (1.0 / float(W))


def test_olmar1_phi_matches_olps_reference():
    W = 3
    # synthetic history (gross relatives)
    H = [
        np.array([1.2, 0.9]),
        np.array([0.8, 1.1]),
        np.array([1.5, 0.7]),
        np.array([0.9, 1.4]),
    ]
    pred = OLMAR1Predictor(window=W)
    pred.reset(d=2)
    out = []
    for x in H:
        out.append(pred.update_and_predict(x))

    # last phi should match OLPS-style computation
    ref = _olps_matlab_like_phi(H, W)
    np.testing.assert_allclose(out[-1], ref, rtol=0, atol=1e-12)


def test_olmar1_phi_cold_start_uses_last_relative_until_W_plus_1():
    W = 4
    H = [
        np.array([1.1, 0.9]),
        np.array([1.2, 1.0]),
        np.array([0.95, 1.05]),
        np.array([1.3, 0.85]),  # T == W
    ]
    pred = OLMAR1Predictor(window=W)
    pred.reset(d=2)
    out = []
    for x in H:
        out.append(pred.update_and_predict(x))

    # T=4, W=4 -> T < W+1, must return last observed
    np.testing.assert_allclose(out[-1], H[-1], rtol=0, atol=1e-12)

    # Add one more (T=5 == W+1): now should use the moving-average
    x5 = np.array([0.8, 1.1])
    out.append(pred.update_and_predict(x5))
    ref = _olps_matlab_like_phi(H + [x5], W)
    np.testing.assert_allclose(out[-1], ref, rtol=0, atol=1e-12)


def test_all_weights_are_trading_weights():
    X = np.array([[0.0, 0.0], [0.1, -0.1]], dtype=float)  # net returns
    model = MeanReversion(
        strategy="olmar",
        olmar_order=1,
        olmar_window=2,
        epsilon=2.0,
        update_mode="pa",
        warm_start=False,
    )
    model.fit(X)
    W = model.all_weights_
    # day-1 is uniform
    np.testing.assert_allclose(W[0], np.array([0.5, 0.5]), atol=1e-12)
    # day-2 is a valid simplex vector (it's what was decided after seeing day-1)
    np.testing.assert_allclose(W[1].sum(), 1.0, atol=1e-12)
    assert np.all(W[1] >= -1e-12)


def _olps_phi_reference(history, W):
    T = len(history)
    if T < W + 1:
        return history[-1].copy()
    d = history[-1].shape[0]
    tmp = np.ones(d, dtype=float)
    phi = np.zeros(d, dtype=float)
    for i in range(W):
        phi += 1.0 / np.maximum(tmp, CLIP_EPSILON)
    x_idx = T - i - 1
    tmp = tmp * np.maximum(history[x_idx], CLIP_EPSILON)
    return phi * (1.0 / float(W))


def test_olmar1_phi_matches_olps_reference():
    W = 4
    H = [
        np.array([1.2, 0.9]),
        np.array([0.8, 1.1]),
        np.array([1.5, 0.7]),
        np.array([0.9, 1.4]),
        np.array([1.1, 0.95]),
    ]
    pred = OLMAR1Predictor(window=W)
    pred.reset(d=2)
    out = []
    for x in H:
        out.append(pred.update_and_predict(x))
    np.testing.assert_allclose(out[-1], _olps_phi_reference(H, W), atol=1e-12)


def test_olmar1_phi_cold_start_uses_last_relative_until_W_plus_1():
    W = 4
    H = [np.array([1.1, 0.9]), np.array([1.2, 1.0]), np.array([0.95, 1.05])]
    pred = OLMAR1Predictor(window=W)
    pred.reset(d=2)
    # Until T < W+1 -> return last relative
    for x in H:
        out = pred.update_and_predict(x)
    np.testing.assert_allclose(out, H[-1], atol=1e-12)
    # Add one more (T = W+1) -> moving-average mode
    out2 = pred.update_and_predict(np.array([0.9, 1.2]))
    ref = _olps_phi_reference(H + [np.array([0.9, 1.2])], W)
    np.testing.assert_allclose(out2, ref, atol=1e-12)
