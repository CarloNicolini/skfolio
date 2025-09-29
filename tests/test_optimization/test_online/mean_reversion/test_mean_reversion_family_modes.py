import numpy as np

from skfolio.optimization.online._mean_reversion import MeanReversion


def test_olmar_pamr_cwmr_all_run_under_pa():
    X = np.array([[0.0, 0.0], [0.1, -0.1]], dtype=float)
    m_olmar = MeanReversion(
        strategy="olmar",
        olmar_order=1,
        olmar_window=2,
        epsilon=2.0,
        update_mode="pa",
        warm_start=False,
    )
    m_pamr = MeanReversion(
        strategy="pamr", epsilon=1.0, update_mode="pa", warm_start=False
    )
    m_cwmr = MeanReversion(
        strategy="cwmr",
        epsilon=1.0,
        cwmr_eta=0.95,
        cwmr_sigma0=0.5,
        update_mode="pa",
        warm_start=False,
    )

    for m in (m_olmar, m_pamr, m_cwmr):
        m.fit(X)
        assert m.all_weights_.shape == X.shape
        assert np.allclose(m.all_weights_.sum(axis=1), 1.0, atol=1e-9)


def test_cwmr_md_runs_and_produces_portfolios():
    X = np.array([[0.0, 0.0], [0.05, -0.03]], dtype=float)
    m = MeanReversion(
        strategy="cwmr",
        epsilon=1.0,
        cwmr_eta=0.95,
        cwmr_sigma0=0.5,
        update_mode="md",
        warm_start=False,
    )
    m.fit(X)
    assert m.all_weights_.shape == X.shape
    assert np.allclose(m.all_weights_.sum(axis=1), 1.0, atol=1e-9)
