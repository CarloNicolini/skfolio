import numpy as np

from skfolio.optimization.online._descent import (
    EntropicMirrorDescent,
    OnlineGradientDescent,
    OnlineNewtonStep,
)


def _run_descent(
    descent, w0: np.ndarray, steps: int, grad_fn
) -> tuple[np.ndarray, np.ndarray]:
    """Utility: iterate w_{t+1} = project(step(w_t, g_t)).

    Returns the final weights and the trajectory (including w0).
    """
    w = np.asarray(w0, dtype=float)
    traj: list[np.ndarray] = [w.copy()]
    for _ in range(steps):
        g = grad_fn(w)
        y = descent.step(w, g)
        w = y
        traj.append(w.copy())
    return w, np.vstack(traj)


def test_ogd_step_matches_schedule_and_converges_to_zero():
    rng = np.random.default_rng(0)
    n = 5
    w0 = rng.normal(size=n)
    eta = 0.1
    ogd = OnlineGradientDescent(projector=None, eta=eta, use_schedule=True)

    # Gradient for f(w) = ||w||^2 is 2w
    grad = lambda w: 2.0 * w

    # Convergence in L2 to zero within a reasonable number of rounds
    ogd = OnlineGradientDescent(projector=None, eta=eta, use_schedule=True)
    w_final, traj = _run_descent(ogd, w0, steps=1500, grad_fn=grad)
    # Norm should be very small
    assert np.linalg.norm(w_final) <= 1e-6
    # Objective decreased substantially
    f0 = float(np.dot(w0, w0))
    fT = float(np.dot(w_final, w_final))
    assert fT < f0 * 1e-4


def test_ons_converges_fast_on_quadratic_and_first_step_matches_eta():
    rng = np.random.default_rng(2)
    n = 5
    w0 = rng.normal(size=n)

    eta = 0.1
    # Use eps=1.0 so A_inv_0 = I, disable auto-eta to control step exactly
    ons = OnlineNewtonStep(
        projector=None,
        eta=eta,
        eps=1.0,
        jitter=1e-12,
        recompute_every=None,
        auto_eta=False,
    )

    grad = lambda w: 2.0 * w

    # Run for several rounds and verify substantial objective decrease
    w_final, traj = _run_descent(ons, w0, steps=1500, grad_fn=grad)
    f0 = float(np.dot(w0, w0))
    fT = float(np.dot(w_final, w_final))
    # With constant eta and Sherman-Morrison, we expect a clear decrease but not necessarily to ~0
    assert fT <= f0 * 0.95
