import numpy as np
import pytest
from scipy.special import softmax

from skfolio.optimization.online._base import FTRLProximal, FTRLStrategy
from skfolio.optimization.online._mirror_maps import (
    EntropyMirrorMap,
)
from skfolio.optimization.online._projection import (
    AutoProjector,
    ProjectionConfig,
)

from ..utils import assert_box_budget


def test_entropy_omd_matches_exponentiated_gradient():
    """
    For ψ entropy and OMD mode, Orabona Eq. (8.13) gives wₜ₊₁ ∝ wₜ·exp(-η gₜ). _composite_update implements softmax(log wₜ - ηgₜ).
    Weights from FTRLProximal coincide with exponentiated-gradient update, validating OMD pathway.
    """
    model = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode="omd",
        learning_rate=0.5,
        warm_start=False,
        initial_weights=np.array([0.6, 0.4]),
    )
    X = np.array([[0.02, -0.01]])
    model.fit(X)
    w0 = np.array([0.6, 0.4])
    rel = 1.0 + X[0]
    grad = -rel / np.dot(w0, rel)
    expected = w0 * np.exp(-0.5 * grad)
    expected /= expected.sum()
    assert np.allclose(model.weights_, expected, atol=1e-10)


def test_entropy_ftrl_matches_dual_averaging():
    """Dual averaging with entropy gives wₜ₊₁ ∝ exp(-η Σ_{s≤t} g_s) (Orabona Lemma 8.6).
    _FTRLEngine’s FTRL branch uses Σg.
    Expected outcome: final weights align with dual averaging recursion, confirming cumulative-gradient handling."""
    model = FTRLProximal(
        strategy=FTRLStrategy.EG,
        update_mode="ftrl",
        learning_rate=0.3,
        warm_start=False,
        initial_weights=np.array([0.5, 0.5]),
    )
    X = np.array([[0.05, -0.02], [-0.01, 0.03]])
    model.fit(X)
    w = np.array([0.5, 0.5])
    G = np.zeros_like(w)
    for row in X:
        rel = 1.0 + row
        grad = -rel / np.dot(w, rel)
        G += grad
        w = softmax(-0.3 * G)
    assert np.allclose(model.weights_, w, atol=1e-10)
