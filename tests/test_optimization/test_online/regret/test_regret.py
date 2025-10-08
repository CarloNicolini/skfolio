# ruff: noqa: E402
import numpy as np
import pytest

from skfolio.optimization.online import BCRP, FTRLProximal, FTRLStrategy
from skfolio.optimization.online._mixins import RegretType
from skfolio.optimization.online._regret import _running_regret, regret


@pytest.mark.parametrize(
    "avg_flag",
    [False, True, "running", "final", "none"],
)
@pytest.mark.parametrize("regret_type", [RegretType.STATIC, RegretType.DYNAMIC])
def test_shapes_and_finiteness(X_small, avg_flag, regret_type):
    est = FTRLProximal(strategy=FTRLStrategy.EG, eta=0.05, warm_start=False)
    r = regret(
        estimator=est,
        X=X_small,
        comparator=BCRP(),
        average=avg_flag,
        regret_type=regret_type,
    )
    assert isinstance(r, np.ndarray)
    assert r.ndim == 1
    assert r.size == X_small.shape[0]
    assert np.all(np.isfinite(r))


def test_running_vs_final_average_agreement_at_T(X_small):
    est = FTRLProximal(strategy=FTRLStrategy.EG, eta=0.05, warm_start=False)
    # cumulative
    r_cum = regret(est, X_small, comparator=BCRP(), average=False)
    # running average (R_t / t)
    r_run = regret(est, X_small, comparator=BCRP(), average="running")
    # final average constant curve
    r_final = regret(est, X_small, comparator=BCRP(), average="final")

    T = X_small.shape[0]
    assert np.isclose(r_run[-1], r_cum[-1] / T, atol=1e-9)
    assert np.allclose(r_final, r_cum[-1] / T, atol=1e-9)


def test_windowed_running_and_final_average(X_small):
    est = FTRLProximal(strategy=FTRLStrategy.EG, eta=0.05, warm_start=False)
    w = 10
    r_win = regret(est, X_small, comparator=BCRP(), average=False, window=w)
    r_win_run = regret(est, X_small, comparator=BCRP(), average="running", window=w)
    r_win_final = regret(est, X_small, comparator=BCRP(), average="final", window=w)

    # Before window-1, the windowed curve should be zero (by construction in our implementation)
    assert np.allclose(r_win[: w - 1], 0.0)
    # Running average divides by w
    assert np.allclose(r_win_run[w - 1 :], r_win[w - 1 :] / float(w))
    # Final returns a constant equal to last window average
    expected_final = r_win[-1] / float(w)
    assert np.allclose(r_win_final, expected_final)


def test_comparator_instance_required(X_small):
    est = FTRLProximal(strategy=FTRLStrategy.EG, eta=0.05, warm_start=False)
    # Should accept instance or None (defaults to BCRP())
    r1 = regret(est, X_small, comparator=BCRP())
    r2 = regret(est, X_small, comparator=None)
    assert r1.shape == r2.shape


def test_invalid_window_raises(X_small):
    est = FTRLProximal(strategy=FTRLStrategy.EG, eta=0.05, warm_start=False)
    with pytest.raises(ValueError):
        regret(est, X_small, comparator=BCRP(), window=0)
    with pytest.raises(ValueError):
        regret(est, X_small, comparator=BCRP(), window=X_small.shape[0] + 1)


def test_dynamic_without_fit_dynamic_raises(X_small):
    class DummyComparator(BCRP.__class__):  # create a dummy type without fit_dynamic
        pass

    # Build a simple dummy instance with no fit_dynamic attribute
    comp = object()
    est = FTRLProximal(strategy=FTRLStrategy.EG, eta=0.05, warm_start=False)
    with pytest.raises(ValueError):
        regret(est, X_small, comparator=comp, regret_type=RegretType.DYNAMIC)


from skfolio.optimization.online._base import BaseOptimization


class DummyEstimator(BaseOptimization):
    def __init__(self, weights_seq):
        super().__init__()
        self._weights_seq = np.asarray(weights_seq)

    def fit(self, X):
        self.all_weights_ = self._weights_seq
        self.weights_ = self._weights_seq[-1]
        return self


class DummyComparator(BaseOptimization):
    def __init__(self, w_star):
        super().__init__()
        self._w = np.asarray(w_star)

    def fit(self, X):
        self.weights_ = self._w
        return self


def test_static_regret_matches_manual_curve():
    """
    Returned curve equals manual cumulative sums; ensures static branch correctness.
    """
    X = np.array([[0.1, -0.05], [-0.02, 0.04]])
    est = DummyEstimator(weights_seq=[[0.5, 0.5], [0.6, 0.4]])
    comp = DummyComparator([0.55, 0.45])
    reg = regret(est, X, comparator=comp, regret_type=RegretType.STATIC, average=False)
    rel = 1.0 + X
    loss_est = -np.log(np.sum(rel * est.all_weights_, axis=1))
    loss_comp = -np.log(rel @ comp.weights_)
    manual = np.cumsum(loss_est - loss_comp)
    assert np.allclose(reg, manual)


def test_dynamic_worst_case_uses_argmax_weights():
    """Expected outcome: regrets identical, validating worst-case comparator logic."""
    X = np.array([[0.1, -0.05], [0.03, 0.2], [-0.02, 0.01]])
    est = DummyEstimator(
        weights_seq=np.array(
            [
                [0.5, 0.5],
                [0.4, 0.6],
                [0.3, 0.7],
            ]
        )
    )
    reg = regret(est, X, regret_type=RegretType.DYNAMIC_WORST_CASE, average=False)
    rel = 1.0 + X
    wc = np.zeros_like(rel)
    wc[np.arange(rel.shape[0]), np.argmax(rel, axis=1)] = 1.0
    manual = np.cumsum(
        -np.log(np.sum(rel * est.all_weights_, axis=1))
        + np.log(np.sum(rel * wc, axis=1))
    )
    assert np.allclose(reg, manual)


def test_universal_dynamic_path_length_zero():
    """Expected outcome: universal dynamic regret with PT=0 matches static regret, confirming special-case routing."""
    X = np.array([[0.05, 0.0], [0.02, 0.01]])
    est = DummyEstimator(weights_seq=np.array([[0.5, 0.5], [0.6, 0.4]]))
    reg_uni = regret(
        est,
        X,
        regret_type=RegretType.DYNAMIC_UNIVERSAL,
        dynamic_config={"path_length": 0.0, "solver": "SCS"},
    )
    reg_static = regret(est, X, regret_type=RegretType.STATIC)
    assert np.allclose(reg_uni, reg_static)


def test_running_average_and_window_modes():
    """Expected outcome: _running_regret divides by t for average=True and handles window rolling sums; ensures theoretical interpretation of average regret."""
    losses_algo = np.array([0.3, 0.2, 0.4])
    losses_comp = np.array([0.1, 0.25, 0.2])
    rr = _running_regret(losses_algo, losses_comp, average=True)
    expected = np.cumsum(losses_algo - losses_comp) / np.arange(1, 4)
    assert np.allclose(rr, expected)
    windowed = _running_regret(losses_algo, losses_comp, window=2, average=True)
    manual = np.array(
        [
            0.0,
            np.sum(losses_algo[:2] - losses_comp[:2]) / 2,
            np.sum(losses_algo[1:] - losses_comp[1:]) / 2,
        ]
    )
    assert np.allclose(windowed, manual)
