import numpy as np

from skfolio.optimization.online._ftrl import _FTRLEngine
from skfolio.optimization.online._mirror_maps import AdaptiveMahalanobisMap
from skfolio.optimization.online._projection import IdentityProjector


class SpyMahalanobis(AdaptiveMahalanobisMap):
    def __init__(self, eps=1e-8):
        super().__init__(d=None, eps=eps)
        self.h_calls = []

    def grad_psi(self, w: np.ndarray) -> np.ndarray:
        h = self.H_diag().copy()
        self.h_calls.append(("grad_psi", h))
        return super().grad_psi(w)

    def grad_psi_star(self, z: np.ndarray) -> np.ndarray:
        h = self.H_diag().copy()
        self.h_calls.append(("grad_psi_star", h))
        return super().grad_psi_star(z)


def gen_repeated_single_coord_grads(d=3, T=3, coord=0, val=2.0):
    g = np.zeros(d, dtype=float)
    g[coord] = val
    return [g.copy() for _ in range(T)]


def test_adagrad_off_by_one_accumulation_order_is_previous_only():
    # FTRL-Proximal vs FTRL-Centered differ by an off-by-one in the norm used in the bound.
    # Our engine updates the adaptive map after the step, so the geometry at round t uses g_{1:t-1}.
    # This test asserts that the per-round diagonal used BEFORE the update does not yet include g_t.
    d = 3
    spy = SpyMahalanobis(eps=1e-10)
    eng = _FTRLEngine(
        mirror_map=spy, projector=IdentityProjector(), eta=1.0, mode="omd"
    )
    grads = gen_repeated_single_coord_grads(d=d, T=2, coord=0, val=3.0)

    # Step 1: before update, h should be sqrt(eps) on all coords
    x1 = eng.step(grads[0])
    h_gpsi_t1 = [h for name, h in spy.h_calls if name == "grad_psi"][0]
    assert np.allclose(h_gpsi_t1, np.sqrt(np.full(d, spy.eps)), atol=1e-12)

    # Step 2: now h should include g_1^2 in coord 0
    spy.h_calls.clear()
    x2 = eng.step(grads[1])
    h_gpsi_t2 = [h for name, h in spy.h_calls if name == "grad_psi"][0]
    assert h_gpsi_t2[0] > h_gpsi_t1[0]
    assert np.allclose(h_gpsi_t2[1:], h_gpsi_t1[1:], atol=1e-12)


def test_adagrad_scale_free_trajectory_under_scalar_rescaling():
    # Data-dependent learning rates produce scale-free behavior: scaling gradients and inversely scaling η
    # should leave the trajectory nearly unchanged (up to epsilon effects).
    d, T = 5, 30
    rng = np.random.default_rng(7)
    grads = [rng.normal(size=d) for _ in range(T)]

    # Base run
    map_a = AdaptiveMahalanobisMap(eps=1e-10)
    eng_a = _FTRLEngine(
        mirror_map=map_a, projector=IdentityProjector(), eta=0.5, mode="omd"
    )
    Wa = [eng_a.step(g).copy() for g in grads]
    Wa = np.vstack(Wa)

    # Scaled run: gradients * c, same eta
    c = 10.0
    map_b = AdaptiveMahalanobisMap(eps=1e-10)
    eng_b = _FTRLEngine(
        mirror_map=map_b, projector=IdentityProjector(), eta=0.5, mode="omd"
    )
    Wb = [eng_b.step(c * g).copy() for g in grads]
    Wb = np.vstack(Wb)
    # Trajectories should be very close
    np.testing.assert_allclose(Wa, Wb, rtol=1e-6, atol=1e-8)


def test_adagrad_extreme_coordinate_hits_do_not_numerically_blow_up():
    # Very large gradient spikes on one coordinate should shrink the step along that coordinate
    # without causing NaNs in the adaptive geometry.
    d, T = 4, 10
    grads = [np.array([1e6, 0.0, 0.0, 0.0])] + [np.zeros(d)] * (T - 1)
    eng = _FTRLEngine(
        mirror_map=AdaptiveMahalanobisMap(eps=1e-12),
        projector=IdentityProjector(),
        eta=1.0,
        mode="omd",
    )
    for g in grads:
        w = eng.step(g)
        assert np.all(np.isfinite(w))
