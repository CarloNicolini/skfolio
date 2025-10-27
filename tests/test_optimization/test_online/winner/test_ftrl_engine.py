import numpy as np
import pytest

from skfolio.optimization.online._ftrl import FirstOrderOCO
from skfolio.optimization.online._mirror_maps import (
    AdaptiveLogBarrierMap,
    AdaptiveMahalanobisMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
)
from skfolio.optimization.online._prediction import LastGradPredictor
from skfolio.optimization.online._projection import (
    AutoProjector,
    IdentityProjector,
    ProjectionConfig,
)


def rng_grad_sequence(d=5, T=10, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=d).astype(float) for _ in range(T)]


def assert_simplex(w: np.ndarray, atol=1e-10):
    assert np.all(w >= -1e-12)
    assert np.all(w <= 1 + 1e-12)
    s = float(np.sum(w))
    assert abs(s - 1.0) <= atol


@pytest.mark.parametrize("d,T", [(4, 7), (5, 12)])
def test_entropy_omd_ftrl_equivalence_constant_eta(d, T):
    # Entropy mirror map: OMD and FTRL are equivalent with constant eta and simplex projection
    # when starting from uniform initialization.
    g_seq = rng_grad_sequence(d=d, T=T, seed=42)
    projector = AutoProjector(ProjectionConfig(lower=0.0, upper=1.0, budget=1.0))
    eta = 0.1

    omd = FirstOrderOCO(
        mirror_map=EntropyMirrorMap(), projector=projector, eta=eta, mode="omd"
    )
    ftr = FirstOrderOCO(
        mirror_map=EntropyMirrorMap(), projector=projector, eta=eta, mode="ftrl"
    )

    W_omd = []
    W_ftr = []
    for g in g_seq:
        W_omd.append(omd.step(g).copy())
        W_ftr.append(ftr.step(g).copy())

    for w1, w2 in zip(W_omd, W_ftr):
        assert_simplex(w1)
        assert_simplex(w2)
        # Entropy map equivalence up to numerical error
        np.testing.assert_allclose(w1, w2, rtol=1e-10, atol=1e-12)


def test_euclidean_omd_step_matches_formula_identity_projector():
    # With Euclidean map and identity projector, OMD step is x_{t+1} = x_t - eta_t (g_t + m_t)
    d = 3
    g0 = np.array([1.0, -2.0, 0.5])
    g1 = np.array([-0.3, 0.7, 1.2])
    eta = 0.2
    omd = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        mode="omd",
    )

    # x0 initialized uniformly
    x0 = np.ones(d) / d
    x1 = omd.step(g0)
    np.testing.assert_allclose(x1, x0 - eta * g0, rtol=0, atol=1e-12)

    x2 = omd.step(g1)
    np.testing.assert_allclose(x2, x1 - eta * g1, rtol=0, atol=1e-12)


def test_euclidean_ftrl_matches_dual_averaging_identity_projector():
    # With Euclidean map and identity projector, FTRL step is x_{t+1} = - eta_t * sum_{s<=t} g_s
    d = 4
    g0 = np.array([1.0, 2.0, -3.0, 0.5])
    g1 = np.array([0.0, -1.0, 1.0, -0.5])
    eta = 0.1
    ftr = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        mode="ftrl",
    )

    x1 = ftr.step(g0)
    np.testing.assert_allclose(x1, -eta * g0, rtol=0, atol=1e-12)

    x2 = ftr.step(g1)
    np.testing.assert_allclose(x2, -eta * (g0 + g1), rtol=0, atol=1e-12)


def test_predictor_effect_and_shape_check():
    # OMD with LastGradPredictor: at t=1 predictor is zero; at t=2 predictor equals g0
    d = 2
    eta = 0.3
    g0 = np.array([1.0, -1.0])
    g1 = np.array([2.0, 0.5])
    pred = LastGradPredictor()
    omd_pred = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        predictor=pred,
        mode="omd",
    )
    omd_base = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        predictor=None,
        mode="omd",
    )

    # Step 1 identical (predictor returns zeros)
    x1_pred = omd_pred.step(g0)
    x1_base = omd_base.step(g0)
    np.testing.assert_allclose(x1_pred, x1_base, rtol=0, atol=1e-12)

    # Step 2: predictor returns last_grad = g0, effect equals an extra -eta*g0 step
    x2_pred = omd_pred.step(g1)
    x2_base = omd_base.step(g1)
    np.testing.assert_allclose(x2_pred, x2_base - eta * g0, rtol=0, atol=1e-12)

    # Wrong-shape predictor must raise
    class BadPredictor:
        def __call__(self, t, last_x, last_g):
            return np.zeros(d + 1)  # wrong shape

    omd_bad = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        predictor=BadPredictor(),
        mode="omd",
    )
    with pytest.raises(ValueError, match="wrong shape"):
        omd_bad.step(g0)


def test_eta_schedule_array_and_callable():
    d = 3
    g = np.array([1.0, 0.5, -2.0])

    # Array schedule
    eta_arr = np.array([0.3, 0.2, 0.1])
    omd_a = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta_arr,
        mode="omd",
    )
    x0 = np.ones(d) / d
    x1 = omd_a.step(g)
    np.testing.assert_allclose(x1, x0 - eta_arr[0] * g, atol=1e-12)
    x2 = omd_a.step(g)
    np.testing.assert_allclose(x2, x1 - eta_arr[1] * g, atol=1e-12)
    x3 = omd_a.step(g)
    np.testing.assert_allclose(x3, x2 - eta_arr[2] * g, atol=1e-12)
    # beyond length, last value is repeated
    x4 = omd_a.step(g)
    np.testing.assert_allclose(x4, x3 - eta_arr[-1] * g, atol=1e-12)

    # Callable schedule
    seq = [0.05, 0.07, 0.11]

    def eta_fn(t):  # t=0,1,2... uses seq[t] if in range else last
        return seq[t] if t < len(seq) else seq[-1]

    omd_c = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta_fn,
        mode="omd",
    )
    y0 = np.ones(d) / d
    y1 = omd_c.step(g)
    np.testing.assert_allclose(y1, y0 - seq[0] * g, atol=1e-12)
    y2 = omd_c.step(g)
    np.testing.assert_allclose(y2, y1 - seq[1] * g, atol=1e-12)
    y3 = omd_c.step(g)
    np.testing.assert_allclose(y3, y2 - seq[2] * g, atol=1e-12)
    y4 = omd_c.step(g)
    np.testing.assert_allclose(y4, y3 - seq[-1] * g, atol=1e-12)


def test_dynamic_mirror_maps_update_internals():
    # Check that dynamic maps update state (AdaGrad, AdaBARRONS)
    d = 3
    g0 = np.array([2.0, 0.0, 0.0])
    g1 = np.array([2.0, 0.0, 0.0])

    # AdaGrad-like: H grows with sum of squares, step in coord 0 diminishes compared to Euclidean
    am = AdaptiveMahalanobisMap(eps=1e-8)
    omd_adagrad = FirstOrderOCO(
        mirror_map=am, projector=IdentityProjector(), eta=1.0, mode="omd"
    )
    euc = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=1.0,
        mode="omd",
    )

    x1_ad = omd_adagrad.step(g0)  # after first step, both similar direction
    x1_eu = euc.step(g0)
    # second step: AdaGrad scales down coord 0 more
    x2_ad = omd_adagrad.step(g1)
    x2_eu = euc.step(g1)

    # movement along coord 0 is smaller in AdaGrad than Euclidean on second step
    move_ad = abs(x2_ad[0] - x1_ad[0])
    move_eu = abs(x2_eu[0] - x1_eu[0])
    assert move_ad <= move_eu + 1e-12

    # AdaBARRONS: keep positivity and normalization even with extreme grads
    alb = AdaptiveLogBarrierMap(eps=1e-12)
    proj = AutoProjector(ProjectionConfig(lower=0.0, upper=1.0, budget=1.0))
    ftrl_alb = FirstOrderOCO(mirror_map=alb, projector=proj, eta=0.5, mode="ftrl")
    w = ftrl_alb.step(np.array([-10.0, -10.0, -10.0]))  # adversarial sign
    assert_simplex(w, atol=1e-9)


def test_euclidean_omd_matches_projected_step():
    """
    With ψ(w)=½‖w‖² the OMD step is Proj(w_t - η g_t); _composite_update should be identity plus projector (src/.../_mirror_maps.py:328-340).
    """
    engine = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=0.1,
        mode="omd",
    )
    engine._x_t = np.array([0.4, 0.6])
    grad = np.array([0.3, -0.2])
    w_next = engine.step(grad)
    expected = np.array([0.4, 0.6]) - 0.1 * grad
    assert np.allclose(w_next, expected)
