"""数据模块测试。"""

import numpy as np
import pytest

from satc import config
from satc.data import (
    find_real_experiment_index,
    generate_full_space,
    get_paper_real_value,
    validate_real_data,
)


def test_validate_real_data():
    assert validate_real_data() is True


def test_generate_full_space():
    X = generate_full_space()
    assert X.shape == (81, 4)
    assert len(np.unique(X, axis=0)) == 81
    for i, name in enumerate(config.PARAMETER_NAMES):
        assert set(np.round(X[:, i], 10)) == set(
            np.round(config.LEVELS[name], 10)
        )


def test_find_real_experiment_index():
    for i in range(len(config.X_REAL)):
        assert find_real_experiment_index(config.X_REAL[i]) == i
    assert find_real_experiment_index([0.16, 0.20, 200.0, 40.0]) == -1


def test_get_paper_real_value():
    index, y = get_paper_real_value()
    assert index == find_real_experiment_index(config.PAPER_OPTIMAL)
    assert np.allclose(y, config.Y_REAL[index])


def test_invalid_data_rejected(monkeypatch):
    monkeypatch.setattr(config, "X_REAL", np.zeros((8, 4)))
    with pytest.raises(ValueError):
        validate_real_data()
