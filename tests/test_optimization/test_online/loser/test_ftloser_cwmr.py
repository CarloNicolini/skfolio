# tests/test_ftloser_cwmr.py
import numpy as np
import pytest
from scipy.stats import norm

from skfolio.optimization.online._ftloser import FTLoser


def _make_cwmr(
    *,
    epsilon: float = 1.0,
    update_mode: str = "pa",
    cwmr_eta: float = 0.95,
    cwmr_sigma0: float = 0.5,
    cwmr_min_var: float | None = 1e-12,
    cwmr_max_var: float | None = None,
    cwmr_mean_lr: float = 1.0,
    cwmr_var_lr: float = 1.0,
    **kwargs,
) -> FTLoser:
    return FTLoser(
        strategy="cwmr",
        epsilon=epsilon,
        update_mode=update_mode,
        cwmr_eta=cwmr_eta,
        cwmr_sigma0=cwmr_sigma0,
        cwmr_min_var=cwmr_min_var,
        cwmr_max_var=cwmr_max_var,
        cwmr_mean_lr=cwmr_mean_lr,
        cwmr_var_lr=cwmr_var_lr,
        warm_start=False,
        **kwargs,
    )


def _simple_market():
    return np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [0.08, -0.05],
            [-0.06, 0.06],
            [-0.02, 0.02],
        ],
        dtype=float,
    )


def _assert_simplex(paths: np.ndarray) -> None:
    assert np.all(paths >= -1e-12)
    assert np.allclose(paths.sum(axis=1), 1.0, atol=1e-9)


def _violation(mu: np.ndarray, sigma: np.ndarray, x: np.ndarray, eps: float, phi: float) -> float:
    s = float(np.dot(sigma * x, x))
    return float(np.dot(mu, x)) + phi * np.sqrt(max(s, 0.0)) - eps


def test_cwmr_pa_runs_and_is_feasible():
    model = _make_cwmr(cwmr_eta=0.97, cwmr_sigma0=0.3)
    X = _simple_market()
    model.fit(X)
    W = model.all_weights_
    assert W.shape == X.shape
    _assert_simplex(W)


def test_cwmr_pa_no_update_when_constraint_satisfied():
    model = _make_cwmr(epsilon=10.0)
    X = np.array([[0.0, 0.0]], dtype=float)
    model.fit(X)
    np.testing.assert_allclose(model.all_weights_[0], np.array([0.5, 0.5]), atol=1e-12)


def test_cwmr_pa_mean_reversion_directionality():
    X = np.array([[0.0, 0.0], [0.2, -0.1]], dtype=float)
    model = _make_cwmr(cwmr_eta=0.95, cwmr_sigma0=0.4)
    model.fit(X[:1])
    w_prev = model.weights_.copy()
    model.partial_fit(X[1:2])
    w_new = model.weights_.copy()
    assert w_new[1] >= w_prev[1] - 1e-12


def test_cwmr_pa_variance_shrinkage_matches_loading():
    X = np.array([[0.0, 0.0], [0.3, -0.1]], dtype=float)
    model = _make_cwmr(cwmr_eta=0.99, cwmr_sigma0=1.0)
    model.fit(X[:1])
    sigma_before = model._cwmr_Sdiag.copy()
    model.partial_fit(X[1:2])
    sigma_after = model._cwmr_Sdiag.copy()
    assert sigma_after[0] <= sigma_before[0] + 1e-12
    assert np.any(sigma_after < sigma_before - 1e-14)


def test_cwmr_pa_eta_controls_update_strength():
    X = np.array([[0.0, 0.0], [0.25, -0.2]], dtype=float)
    slow = _make_cwmr(cwmr_eta=0.90, cwmr_sigma0=0.5)
    fast = _make_cwmr(cwmr_eta=0.99, cwmr_sigma0=0.5)
    slow.fit(X[:1])
    baseline = slow.weights_.copy()
    slow.partial_fit(X[1:2])
    fast.fit(X[:1])
    fast.partial_fit(X[1:2])
    l1_slow = np.sum(np.abs(slow.weights_ - baseline))
    l1_fast = np.sum(np.abs(fast.weights_ - baseline))
    assert l1_fast >= l1_slow - 1e-12


def test_cwmr_param_validation_guardrails():
    model = _make_cwmr(cwmr_eta=0.5)
    X = np.array([[0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError):
        model.fit(X)


def test_cwmr_md_runs_and_stays_in_simplex():
    model = _make_cwmr(update_mode="md", cwmr_eta=0.96, cwmr_sigma0=0.4)
    X = _simple_market()
    model.fit(X)
    W = model.all_weights_
    assert W.shape == X.shape
    _assert_simplex(W)


def test_cwmr_md_reduces_violation():
    model = _make_cwmr(
        update_mode="md",
        cwmr_eta=0.97,
        cwmr_sigma0=0.6,
        cwmr_mean_lr=0.8,
        cwmr_var_lr=0.3,
    )
    X = np.array([[0.0, 0.0], [0.2, -0.1]], dtype=float)
    model.fit(X[:1])
    mu_prev = model.weights_.copy()
    sigma_prev = model._cwmr_Sdiag.copy()
    x = 1.0 + X[1]
    phi = norm.ppf(model.cwmr_eta)
    violation_before = _violation(mu_prev, sigma_prev, x, model.epsilon, phi)
    model.partial_fit(X[1:2])
    mu_new = model.weights_.copy()
    sigma_new = model._cwmr_Sdiag.copy()
    violation_after = _violation(mu_new, sigma_new, x, model.epsilon, phi)
    assert violation_after <= violation_before + 1e-10


def test_cwmr_md_learning_rates_control_update_and_variance():
    X = np.array([[0.0, 0.0], [0.25, -0.15]], dtype=float)
    mean_slow = _make_cwmr(update_mode="md", cwmr_mean_lr=0.2, cwmr_var_lr=0.5)
    mean_fast = _make_cwmr(update_mode="md", cwmr_mean_lr=1.0, cwmr_var_lr=0.5)
    var_slow = _make_cwmr(update_mode="md", cwmr_mean_lr=0.5, cwmr_var_lr=0.1)
    var_fast = _make_cwmr(update_mode="md", cwmr_mean_lr=0.5, cwmr_var_lr=0.8)

    mean_slow.fit(X[:1])
    baseline = mean_slow.weights_.copy()
    mean_slow.partial_fit(X[1:2])
    mean_fast.fit(X[:1])
    mean_fast.partial_fit(X[1:2])
    l1_slow = np.sum(np.abs(mean_slow.weights_ - baseline))
    l1_fast = np.sum(np.abs(mean_fast.weights_ - baseline))
    assert l1_fast >= l1_slow - 1e-12

    var_slow.fit(X[:1])
    var_slow.partial_fit(X[1:2])
    var_fast.fit(X[:1])
    var_fast.partial_fit(X[1:2])
    sigma_slow = var_slow._cwmr_Sdiag.copy()
    sigma_fast = var_fast._cwmr_Sdiag.copy()
    assert np.all(sigma_fast <= sigma_slow + 1e-12)


def test_cwmr_state_persists_between_modes():
    model = _make_cwmr(update_mode="pa", cwmr_eta=0.96, cwmr_sigma0=0.4)
    X = np.array([[0.0, 0.0], [0.18, -0.09]], dtype=float)
    model.fit(X[:1])
    mu_pa = model.weights_.copy()
    sigma_pa = model._cwmr_Sdiag.copy()
    # Switch to OCO update while keeping learned state
    model.update_mode = "md"
    model.partial_fit(X[1:2])
    mu_md = model.weights_.copy()
    sigma_md = model._cwmr_Sdiag.copy()
    assert np.allclose(mu_pa.shape, mu_md.shape)
    assert np.all(sigma_md > 0.0)
