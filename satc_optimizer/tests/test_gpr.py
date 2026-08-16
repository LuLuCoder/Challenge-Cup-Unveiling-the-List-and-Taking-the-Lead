"""GPR 代理模型测试。"""

import numpy as np
import pytest

from satc import config
from satc.gpr import SurrogateModel


@pytest.fixture(scope="module")
def model():
    m = SurrogateModel()
    m.fit(config.X_REAL, config.Y_REAL)
    return m


def test_predict_shapes(model):
    X_all = np.vstack([
        [a, b, c, d]
        for a in config.LEVELS["A"]
        for b in config.LEVELS["B"]
        for c in config.LEVELS["C"]
        for d in config.LEVELS["D"]
    ])
    mean = model.predict(X_all)
    mean2, std = model.predict(X_all, return_std=True)

    assert mean.shape == (81, 3)
    assert mean2.shape == (81, 3)
    assert std.shape == (81, 3)
    assert np.all(np.isfinite(mean))
    assert np.all(std >= 0.0)


def test_predict_deterministic(model):
    mean1, _ = model.predict(config.X_REAL, return_std=True)
    mean2, _ = model.predict(config.X_REAL, return_std=True)
    assert np.allclose(mean1, mean2)


def test_training_point_interpolation(model):
    """训练点上预测值应接近真实值（白噪声很小，允许小偏差）。"""
    pred, _ = model.predict(config.X_REAL, return_std=True)
    assert np.allclose(pred, config.Y_REAL, atol=2.0)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SurrogateModel().predict(config.X_REAL)
