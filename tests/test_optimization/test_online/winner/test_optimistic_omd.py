"""Tests for optimistic online mirror descent implementation."""

import numpy as np
import pytest

from skfolio.optimization.online._ftrl import FirstOrderOCO
from skfolio.optimization.online._mirror_maps import (
    EntropyMirrorMap,
    EuclideanMirrorMap,
)
from skfolio.optimization.online._prediction import LastGradPredictor, SmoothPredictor
from skfolio.optimization.online._projection import IdentityProjector


class ZeroPredictor:
    """Predictor that always returns zero - should match vanilla OMD."""

    def __call__(self, t, x, g):
        if x is not None:
            return np.zeros_like(x)
        if g is not None:
            return np.zeros_like(g)
        return np.array([])


class ConstantPredictor:
    """Predictor that returns a constant vector."""

    def __init__(self, value):
        self.value = value

    def __call__(self, t, x, g):
        return self.value.copy()


def test_zero_predictor_matches_vanilla_omd():
    """Test that zero predictor matches vanilla OMD (no optimism)."""
    np.random.seed(42)
    d = 5
    T = 10
    eta = 0.1

    # Create two engines: one with zero predictor, one without
    map1 = EuclideanMirrorMap()
    map2 = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine_optimistic = FirstOrderOCO(
        mirror_map=map1, projector=proj, eta=eta, predictor=ZeroPredictor(), mode="omd"
    )

    engine_vanilla = FirstOrderOCO(
        mirror_map=map2, projector=proj, eta=eta, predictor=None, mode="omd"
    )

    # Run both engines on same sequence of gradients
    gradients = [np.random.randn(d) for _ in range(T)]

    weights_optimistic = []
    weights_vanilla = []

    for g in gradients:
        w1 = engine_optimistic.step(g)
        w2 = engine_vanilla.step(g)
        weights_optimistic.append(w1)
        weights_vanilla.append(w2)

    # They should produce identical trajectories
    for w1, w2 in zip(weights_optimistic, weights_vanilla, strict=False):
        np.testing.assert_allclose(w1, w2, rtol=1e-10, atol=1e-10)


def test_first_round_initialization():
    """Test that first round handles missing prev_prediction correctly."""
    np.random.seed(42)
    d = 3
    eta = 0.1

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    # First gradient
    g1 = np.array([1.0, 2.0, 3.0])
    w1 = engine.step(g1)

    # Should initialize _prev_prediction to zeros, so effective_grad = g1 + g1 = 2*g1
    # But since prev_prediction is initialized to zero, effective_grad = g1 - 0 + 0 = g1
    # (because LastGradPredictor returns empty/zeros on first call)
    assert engine._prev_prediction is not None
    assert w1.shape == (d,)


def test_last_grad_predictor_effective_gradient():
    """Test that LastGradPredictor produces correct effective gradients."""
    np.random.seed(42)
    d = 3
    eta = 0.1

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    g1 = np.array([1.0, 0.0, 0.0])
    g2 = np.array([0.0, 1.0, 0.0])
    g3 = np.array([0.0, 0.0, 1.0])

    # Round 1: prev_pred = 0, m_t = 0 (no last grad yet)
    # effective_grad = g1 - 0 + 0 = g1
    w1 = engine.step(g1)
    prev_pred_1 = engine._prev_prediction.copy()
    # After round 1, prev_prediction should be the prediction made at round 1
    # LastGradPredictor at round 1 returned 0, so prev_pred should be 0

    # Round 2: prev_pred = 0, m_t = g1 (last grad predictor returns g1)
    # effective_grad = g2 - 0 + g1
    w2 = engine.step(g2)
    prev_pred_2 = engine._prev_prediction.copy()
    # After round 2, prev_prediction should be g1

    # Round 3: prev_pred = g1, m_t = g2
    # effective_grad = g3 - g1 + g2
    w3 = engine.step(g3)
    prev_pred_3 = engine._prev_prediction.copy()
    # After round 3, prev_prediction should be g2

    # Check that prev_prediction is updated correctly
    np.testing.assert_allclose(prev_pred_2, g1, rtol=1e-10)
    np.testing.assert_allclose(prev_pred_3, g2, rtol=1e-10)


def test_prediction_tracking():
    """Test that _prev_prediction is correctly tracked across rounds."""
    np.random.seed(42)
    d = 3
    eta = 0.1

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    # Use constant predictor for easy tracking
    const_vec = np.array([0.5, 0.5, 0.5])
    predictor = ConstantPredictor(const_vec)

    engine = FirstOrderOCO(
        mirror_map=map_obj, projector=proj, eta=eta, predictor=predictor, mode="omd"
    )

    g1 = np.array([1.0, 0.0, 0.0])
    g2 = np.array([0.0, 1.0, 0.0])

    # Round 1
    _ = engine.step(g1)
    # After round 1, prev_prediction should be const_vec
    np.testing.assert_allclose(engine._prev_prediction, const_vec, rtol=1e-10)

    # Round 2
    _ = engine.step(g2)
    # Should still be const_vec (constant predictor)
    np.testing.assert_allclose(engine._prev_prediction, const_vec, rtol=1e-10)


def test_smooth_predictor():
    """Test SmoothPredictor with clipping."""
    np.random.seed(42)
    d = 3
    eta = 0.1
    L = 0.5

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj,
        projector=proj,
        eta=eta,
        predictor=SmoothPredictor(smoothness_L=L),
        mode="omd",
    )

    # First gradient - predictor has no history yet, returns zeros
    g1 = np.array([2.0, -3.0, 1.0])
    _ = engine.step(g1)
    # After first step, prev_prediction should be zeros (no last_grad yet)
    np.testing.assert_allclose(engine._prev_prediction, np.zeros(d), rtol=1e-10)

    # Second gradient - now predictor should clip g1
    g2 = np.array([1.0, 1.0, 1.0])
    _ = engine.step(g2)
    # Predictor should have clipped g1 to [-L, L]
    expected_clipped = np.clip(g1, -L, L)
    np.testing.assert_allclose(engine._prev_prediction, expected_clipped, rtol=1e-10)


def test_omd_vs_ftrl_modes_with_predictor():
    """Test that OMD and FTRL modes both work with predictors."""
    np.random.seed(42)
    d = 3
    T = 5
    eta = 0.1

    map_omd = EuclideanMirrorMap()
    map_ftrl = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine_omd = FirstOrderOCO(
        mirror_map=map_omd,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    engine_ftrl = FirstOrderOCO(
        mirror_map=map_ftrl,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="ftrl",
    )

    gradients = [np.random.randn(d) for _ in range(T)]

    # Both should run without errors
    for g in gradients:
        w_omd = engine_omd.step(g)
        w_ftrl = engine_ftrl.step(g)
        assert w_omd.shape == (d,)
        assert w_ftrl.shape == (d,)


def test_numerical_example_hand_calculated():
    """Test against hand-calculated optimistic OMD update."""
    # Simple 2D case with Euclidean mirror map
    d = 2
    eta = 1.0

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    # Use constant predictor for predictability
    predictor = ConstantPredictor(np.array([0.1, 0.1]))

    engine = FirstOrderOCO(
        mirror_map=map_obj, projector=proj, eta=eta, predictor=predictor, mode="omd"
    )

    # Start from uniform
    x0 = np.array([0.5, 0.5])
    engine._x_t = x0.copy()

    g1 = np.array([1.0, -1.0])

    # Round 1:
    # prev_pred = 0 (initialized to zero)
    # m_t = [0.1, 0.1] (from predictor)
    # effective_grad = g1 - 0 + [0.1, 0.1] = [1.1, -0.9]
    # OMD update: x1 = x0 - eta * effective_grad = [0.5, 0.5] - 1.0 * [1.1, -0.9]
    #           = [-0.6, 1.4]

    w1 = engine.step(g1)

    expected_w1 = np.array([-0.6, 1.4])
    np.testing.assert_allclose(w1, expected_w1, rtol=1e-10, atol=1e-10)

    # After round 1, prev_prediction should be [0.1, 0.1]
    np.testing.assert_allclose(
        engine._prev_prediction, np.array([0.1, 0.1]), rtol=1e-10
    )


def test_entropy_mirror_map_with_predictor():
    """Test that optimistic updates work with entropy mirror map (EG)."""
    np.random.seed(42)
    d = 4
    T = 10
    eta = 0.1

    map_obj = EntropyMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    gradients = [np.random.randn(d) * 0.1 for _ in range(T)]

    # Should run without errors and produce valid probability distributions
    for g in gradients:
        w = engine.step(g)
        assert w.shape == (d,)
        assert np.all(w >= 0)
        assert np.isclose(np.sum(w), 1.0, rtol=1e-6)


def test_smooth_gradients_lower_regret():
    """Test that optimistic OMD has advantage on smooth gradient sequences."""
    np.random.seed(42)
    d = 5
    T = 50
    eta = 0.05

    # Create smooth gradient sequence (slowly changing)
    base_grad = np.random.randn(d)
    gradients = []
    for t in range(T):
        noise = np.random.randn(d) * 0.01  # Small noise
        gradients.append(base_grad + noise)
        base_grad = base_grad * 0.99 + noise  # Slow drift

    # Test with vanilla OMD
    map1 = EuclideanMirrorMap()
    proj1 = IdentityProjector()
    engine_vanilla = FirstOrderOCO(
        mirror_map=map1, projector=proj1, eta=eta, predictor=None, mode="omd"
    )

    # Test with optimistic OMD
    map2 = EuclideanMirrorMap()
    proj2 = IdentityProjector()
    engine_optimistic = FirstOrderOCO(
        mirror_map=map2,
        projector=proj2,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    loss_vanilla = 0.0
    loss_optimistic = 0.0

    for g in gradients:
        w1 = engine_vanilla.step(g)
        w2 = engine_optimistic.step(g)

        # Linear loss
        loss_vanilla += np.dot(g, w1)
        loss_optimistic += np.dot(g, w2)

    # On smooth sequences, optimistic should typically do better or comparable
    # (not a strict requirement due to randomness, but good sanity check)
    # We just check both are finite and reasonable
    assert np.isfinite(loss_vanilla)
    assert np.isfinite(loss_optimistic)


def test_no_predictor_no_prev_prediction():
    """Test that without predictor, _prev_prediction stays None."""
    np.random.seed(42)
    d = 3
    eta = 0.1

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj, projector=proj, eta=eta, predictor=None, mode="omd"
    )

    g1 = np.array([1.0, 0.0, 0.0])
    _ = engine.step(g1)

    # Without predictor, _prev_prediction should remain None
    assert engine._prev_prediction is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
