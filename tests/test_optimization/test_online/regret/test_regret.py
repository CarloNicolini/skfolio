# ruff: noqa: E402
import numpy as np
import pytest

from skfolio.optimization.online import BCRP, FTRLProximal, FTRLStrategy
from skfolio.optimization.online._mixins import RegretType
from skfolio.optimization.online._regret import regret


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
