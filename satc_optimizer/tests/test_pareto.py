"""Pareto 与折中选解测试。"""

import numpy as np
import pytest

from satc.pareto import (
    compromise_scores,
    constrained_pareto,
    dominates,
    pareto_front,
    percent_score,
    select_compromise,
)


def test_dominates():
    assert dominates([1.0, 2.0], [2.0, 2.0])
    assert not dominates([1.0, 2.0], [1.0, 2.0])
    assert not dominates([2.0, 2.0], [1.0, 2.0])


def test_pareto_front():
    F = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [1.5, 1.5]])
    front = pareto_front(F)
    assert set(front.tolist()) == {0, 1, 3}


def test_constrained_pareto_all_feasible():
    X = np.array([
        [0.15, 0.15, 180.0, 35.0],
        [0.15, 0.20, 200.0, 40.0],
        [0.25, 0.20, 180.0, 45.0],
    ])
    F = np.array([[5.0, 5.0], [3.0, 3.0], [4.0, 4.0]])
    front, violations = constrained_pareto(X, F)
    assert list(front) == [1]
    assert np.all(violations == 0.0)


def test_select_compromise():
    X = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[10.0, 1.0], [5.0, 5.0], [1.0, 10.0]])
    best_x, best_f, score, idx = select_compromise(X, F)
    assert idx == 1
    assert np.allclose(best_x, [2.0])
    assert np.allclose(best_f, [5.0, 5.0])
    assert score == pytest.approx(4.0 / 9.0)


def test_compromise_scores():
    F = np.array([[10.0, 1.0], [5.0, 5.0], [1.0, 10.0]])
    scores = compromise_scores(F)
    assert scores[0] == pytest.approx(0.5)
    assert scores[1] == pytest.approx(4.0 / 9.0)
    assert scores[2] == pytest.approx(0.5)


def test_compromise_scores_external_baseline():
    """用前沿的 min/max 给前沿外的点打分，结果可能超出 [0, 1]。"""
    F = np.array([[10.0, 1.0], [1.0, 10.0]])
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    external = np.array([[30.0, 30.0]])
    scores = compromise_scores(external, f_min=f_min, f_max=f_max)
    assert scores[0] == pytest.approx(29.0 / 9.0)


def test_percent_score():
    assert percent_score(0.0) == pytest.approx(100.0)
    assert percent_score(0.5) == pytest.approx(50.0)
    assert percent_score(1.0) == pytest.approx(0.0)
    # 前沿外方案允许超出 [0, 100]
    assert percent_score(1.2) == pytest.approx(-20.0)
    assert percent_score(-0.1) == pytest.approx(110.0)


def test_compromise_scores_weighted():
    F = np.array([[10.0, 1.0], [5.0, 5.0], [1.0, 10.0]])
    # 只关注 ΔT（第一目标）：归一化后分别为 1, 4/9, 0
    scores = compromise_scores(F, weights=[1.0, 0.0])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(4.0 / 9.0)
    assert scores[2] == pytest.approx(0.0)
    # 0.5/0.5 与等权结果一致（内部自动归一化）
    assert np.allclose(
        compromise_scores(F, weights=[0.5, 0.5]),
        compromise_scores(F),
    )


def test_select_compromise_weighted():
    X = np.array([[1.0], [2.0], [3.0]])
    F = np.array([[10.0, 1.0], [5.0, 5.0], [1.0, 10.0]])
    best_x, _, _, idx = select_compromise(X, F, weights=[1.0, 0.0])
    assert idx == 2
    assert np.allclose(best_x, [3.0])


def test_compromise_scores_invalid_weights():
    F = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError):
        compromise_scores(F, weights=[1.0])
    with pytest.raises(ValueError):
        compromise_scores(F, weights=[0.0, 0.0])
