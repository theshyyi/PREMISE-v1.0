import numpy as np

from premise.metrics import bias, mae, rmse, corr, nse, pod, far, csi, hss


def test_continuous_metrics_basic():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    sim = np.array([1.0, 2.0, 3.0, 5.0])
    assert bias(obs, sim) == 0.25
    assert mae(obs, sim) == 0.25
    assert np.isclose(rmse(obs, sim), 0.5)
    assert np.isclose(corr(obs, sim), np.corrcoef(obs, sim)[0, 1])
    assert np.isfinite(nse(obs, sim))


def test_event_metrics_basic():
    obs = np.array([0.0, 2.0, 0.0, 4.0])
    sim = np.array([0.0, 3.0, 1.0, 0.0])
    assert np.isclose(pod(obs, sim, threshold=1.0), 0.5)
    assert np.isclose(far(obs, sim, threshold=1.0), 0.5)
    assert np.isclose(csi(obs, sim, threshold=1.0), 1 / 3)
    assert np.isfinite(hss(obs, sim, threshold=1.0))
