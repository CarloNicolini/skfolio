import numpy as np
import pytest
from scipy.special import softmax as softmax_stable

from skfolio.optimization.online._ftrl import FTRL, LastGradPredictor
from skfolio.optimization.online._mirror_maps import (
    EntropyMirrorMap,
)
from skfolio.optimization.online._projection import (
    AutoProjector,
    ProjectionConfig,
)


def mult_weights_update(
    w: np.ndarray, g: np.ndarray, m: np.ndarray, eta: float
) -> np.ndarray:
    # Entropic OMD with predictor m corresponds to: w' ∝ w ⊙ exp(-η (g + m))
    z = np.log(np.maximum(w, 1e-16)) - eta * (g + m)
    return softmax_stable(z)


def gen_low_variation_grads(d=10, T=200, seed=0, drift=0.01):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=d)
    seq = []
    for _ in range(T):
        seq.append(g.copy())
        g = g + drift * rng.normal(size=d)
    return seq


def gen_high_variation_grads(d=10, T=200, seed=1, mag=1.0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=d)
    seq = []
    for t in range(T):
        sign = -1.0 if (t % 2) else 1.0
        seq.append(sign * mag * base + 0.1 * rng.normal(size=d))
    return seq


def cumulative_post_update_linear_loss(engine: FTRL, grads: list[np.ndarray]) -> float:
    # We accumulate ⟨x_{t+1}, g_t⟩ after each update (same metric across methods).
    cum = 0.0
    for g in grads:
        x_next = engine.step(g)
        cum += float(np.dot(x_next, g))
    return cum


def assert_simplex(w: np.ndarray, atol=1e-12):
    assert np.all(w >= -1e-12)
    assert np.allclose(np.sum(w), 1.0, atol=atol)


def new_entropy_engine(eta, predictor=None, projector=None, mode="omd") -> FTRL:
    proj = projector or AutoProjector(
        ProjectionConfig(lower=0.0, upper=1.0, budget=1.0)
    )
    return FTRL(
        mirror_map=EntropyMirrorMap(),
        projector=proj,
        eta=eta,
        predictor=predictor,
        mode=mode,
    )


def test_eg_omd_matches_multiplicative_weights_with_optimism():
    # Exact identity: Entropic OMD with predictor m_t is multiplicative weights w' ∝ w * exp(-η(g + m))
    d = 5
    eta = 0.3
    g0 = np.array([0.5, -0.1, 0.2, -0.2, 0.0])
    g1 = np.array([-0.4, 0.3, -0.1, 0.2, -0.3])
    pred = LastGradPredictor()
    eng = new_entropy_engine(
        eta=eta, predictor=pred, projector=AutoProjector(ProjectionConfig())
    )

    # x0 uniform (engine creates it lazily on first step)
    x0 = np.ones(d) / d

    # Step 1: m_0 = 0 (predictor has no last_grad yet)
    x1_eng = eng.step(g0)
    x1_exp = mult_weights_update(w=x0, g=g0, m=np.zeros_like(g0), eta=eta)
    np.testing.assert_allclose(x1_eng, x1_exp, atol=1e-12, rtol=0)

    # Step 2: m_1 = g0 (LastGradPredictor)
    x2_eng = eng.step(g1)
    x2_exp = mult_weights_update(w=x1_exp, g=g1, m=g0, eta=eta)
    np.testing.assert_allclose(x2_eng, x2_exp, atol=1e-12, rtol=0)


def test_optimism_helps_under_low_gradient_variation_and_can_hurt_when_wrong():
    # Literature: optimistic algorithms enjoy problem-dependent bounds in terms of gradient variation;
    # they help when gradients vary slowly and can be worse when predictions are inaccurate.
    # See Chiang et al. (2012) gradient-variation bounds and the optimistic Hedge/FTRL analysis
    # that scales with sum ||g_t - m_t||^2 (cf. meta-bounds in the optimistic Hedge/FTRL derivations).
    d, T, eta = 10, 200, 0.3
    low_var = gen_low_variation_grads(d=d, T=T, seed=0, drift=0.01)
    high_var = gen_high_variation_grads(d=d, T=T, seed=1, mag=1.0)

    # OMD with and without optimism (last-gradient prediction)
    base = new_entropy_engine(
        eta=eta, predictor=None, projector=AutoProjector(ProjectionConfig())
    )
    opti = new_entropy_engine(
        eta=eta,
        predictor=LastGradPredictor(),
        projector=AutoProjector(ProjectionConfig()),
    )

    L_base = cumulative_post_update_linear_loss(base, low_var)
    L_opti = cumulative_post_update_linear_loss(opti, low_var)
    assert (
        L_opti <= L_base + 1e-10
    )  # optimistic should not be worse on low-variation seq

    # this predictor is maximally wrong for alternating signs
    class WrongPredictor:
        def __call__(self, t, last_x, last_grad):
            return last_grad if last_grad is not None else np.zeros_like(last_x)

    anti = new_entropy_engine(
        eta=eta, predictor=WrongPredictor(), projector=AutoProjector(ProjectionConfig())
    )
    L_anti = cumulative_post_update_linear_loss(anti, high_var)
    base2 = new_entropy_engine(
        eta=eta, predictor=None, projector=AutoProjector(ProjectionConfig())
    )
    L_base2 = cumulative_post_update_linear_loss(base2, high_var)
    assert L_anti >= L_base2 - 1e-10


@pytest.mark.xfail(
    reason="OMD with negative-entropy and time-varying LR can suffer linear regret in worst case without stabilization; prefer FTRL form."
)
def test_time_varying_lr_negative_entropy_pathology():
    # With time-varying learning rate and entropic regularizer, OMD can be unstable,
    # whereas the FTRL formulation is recommended. See remarks on Optimistic Hedge with time-varying rates.
    # We craft an adversarial alternating gradient sequence and increasing step sizes.
    d, T = 2, 150
    grads = gen_high_variation_grads(d=d, T=T, seed=5, mag=1.0)
    eta_seq = np.linspace(0.05, 1.0, T)  # increasing LR

    # OMD, time-varying LR
    omd = new_entropy_engine(
        eta=eta_seq,
        predictor=None,
        projector=AutoProjector(ProjectionConfig()),
        mode="omd",
    )
    loss_omd = cumulative_post_update_linear_loss(omd, grads)

    # FTRL, time-varying LR (same eta schedule)
    ftrl = new_entropy_engine(
        eta=eta_seq,
        predictor=None,
        projector=AutoProjector(ProjectionConfig()),
        mode="ftrl",
    )
    loss_ftrl = cumulative_post_update_linear_loss(ftrl, grads)

    # Expect OMD to degrade vs. FTRL in an adversarial setting when eta grows aggressively
    assert loss_omd >= loss_ftrl - 1e-10


def test_entropy_ftrl_and_omd_with_extreme_eta_remain_in_simplex():
    # Numerical stability check: very large learning rates must not cause NaNs and must respect the simplex.
    d, T = 5, 50
    rng = np.random.default_rng(123)
    grads = [rng.normal(size=d) for _ in range(T)]
    eta = 1e6

    omd = new_entropy_engine(eta=eta, predictor=LastGradPredictor())
    ftr = new_entropy_engine(eta=eta, predictor=LastGradPredictor(), mode="ftrl")

    for g in grads:
        w1 = omd.step(g)
        w2 = ftr.step(g)
        assert np.all(np.isfinite(w1)) and np.all(np.isfinite(w2))
        assert_simplex(w1)
        assert_simplex(w2)
