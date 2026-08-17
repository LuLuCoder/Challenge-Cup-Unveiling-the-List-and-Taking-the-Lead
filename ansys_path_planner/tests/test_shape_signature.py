"""形状签名与相似度评判测试。"""

import numpy as np
import pytest

from path_planner.analysis.shape_signature import (
    compute_signature,
    signature_similarity,
)


def _cube(n, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, 3)) * scale


def _sphere(n, scale=1.0, seed=1):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts * (rng.random(n) ** (1.0 / 3.0))[:, None] * scale


def test_cube_cube_high_similarity():
    """两个立方体（尺寸、密度不同）应高度相似。"""
    a = compute_signature(_cube(3000, scale=2.0, seed=0))
    b = compute_signature(_cube(1500, scale=1.0, seed=1))
    assert signature_similarity(a, b) > 0.9


def test_cube_sphere_low_similarity():
    """立方体与球体相似度应明显更低。"""
    cube = compute_signature(_cube(3000, seed=0))
    sphere = compute_signature(_sphere(3000, seed=1))
    assert signature_similarity(cube, sphere) < 0.8


def test_signature_rotation_invariance():
    """同一零件旋转后签名仍应高度相似（主轴对齐）。"""
    pts = _cube(2000, seed=3)
    theta = np.deg2rad(37)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])
    a = compute_signature(pts)
    b = compute_signature(pts @ rot.T)
    assert signature_similarity(a, b) > 0.9
