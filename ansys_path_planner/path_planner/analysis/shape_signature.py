"""零件形状签名与相似度评判。

用途：把已仿真零件（模板）的形状特征存入模板库，对新零件自动判断
“是否与模板相似”，相似则直接映射模板路径，避免重复 ANSYS 仿真。

签名方案：点云规范化（质心归零 + 主轴对齐 + 等比例缩放）后，
统计三个主轴方向的坐标直方图 + 径向距离直方图，用直方图 L1 距离
衡量相似度。该方案对点云密度、旋转、镜像不敏感，且能区分不同外形。
"""

import numpy as np


_AXIS_BINS = 16
_RADIAL_BINS = 24
_AXIS_RANGE = (-0.8, 0.8)
_RADIAL_RANGE = (0.0, 1.2)


def canonical_points(points):
    """把点云规范到统一坐标系：质心归零 + 主轴对齐 + 等比例缩放。

    返回：
        normalized  (N,3) 归一化坐标（包围盒最长边 = 1）
        transform   (centroid, axes, scale)，供路径/应力映射复用同一变换
    """
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    centered = points - centroid

    # 主轴（PCA）：特征值升序，按降序取列作为主轴
    cov = centered.T @ centered / max(len(centered), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]

    # 主轴符号确定性：让该轴坐标和为正（消除翻转歧义）
    for i in range(3):
        if (centered @ axes[:, i]).sum() < 0:
            axes[:, i] *= -1

    aligned = centered @ axes
    scale = float(np.max(np.ptp(aligned, axis=0))) if len(aligned) else 1.0
    if scale < 1e-12:
        scale = 1.0
    normalized = aligned / scale
    return normalized, (centroid, axes, scale)


def _hist(points, bins, range_, axis=None):
    if axis is None:
        values = np.linalg.norm(points, axis=1)
    else:
        values = points[:, axis]
    hist, _ = np.histogram(values, bins=bins, range=range_, density=True)
    return hist.astype(float)


def compute_signature(points, axis_bins=_AXIS_BINS, radial_bins=_RADIAL_BINS):
    """计算零件形状签名（用于相似度对比与模板入库）。"""
    points = np.asarray(points, dtype=float)
    normalized, _transform = canonical_points(points)

    blocks = []
    for ax in range(3):
        blocks.append(_hist(normalized, axis_bins, _AXIS_RANGE, axis=ax))
    blocks.append(_hist(normalized, radial_bins, _RADIAL_RANGE))

    return {
        "node_count": int(len(points)),
        "blocks": [b.tolist() for b in blocks],
        "axis_bins": int(axis_bins),
        "radial_bins": int(radial_bins),
    }


def _normalize(block):
    block = np.asarray(block, dtype=float)
    total = float(block.sum())
    return block / total if total > 1e-12 else block


def _block_similarity(a, b):
    """两个一维直方图块的相似度：1 - 归一化 L1 距离/2。"""
    na = _normalize(a)
    nb = _normalize(b)
    return 1.0 - float(np.abs(na - nb).sum()) / 2.0


def signature_similarity(sig_a, sig_b):
    """两个零件形状签名的相似度（0~1，越大越相似）。"""
    blocks_a = sig_a["blocks"]
    blocks_b = sig_b["blocks"]
    if len(blocks_a) != len(blocks_b):
        raise ValueError("两个签名的直方图块数量不一致。")
    similarities = [
        _block_similarity(x, y)
        for x, y in zip(blocks_a, blocks_b)
    ]
    return float(np.mean(similarities))
