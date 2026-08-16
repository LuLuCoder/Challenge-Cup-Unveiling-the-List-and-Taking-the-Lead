"""真实实验数据校验、查找与离散参数空间生成。"""

from itertools import product

import numpy as np

from satc import config


def validate_real_data(logger=None):
    """检查论文真实实验数据是否完整，返回 True 或抛出 ValueError。"""
    errors = []
    if config.X_REAL.shape != (9, 4):
        errors.append(f"X_REAL 尺寸错误：{config.X_REAL.shape}，应为 (9, 4)")
    if config.Y_REAL.shape != (9, 3):
        errors.append(f"Y_REAL 尺寸错误：{config.Y_REAL.shape}，应为 (9, 3)")
    if len(config.X_REAL) != len(config.Y_REAL):
        errors.append("X_REAL 与 Y_REAL 数量不一致")
    if errors:
        raise ValueError("；".join(errors))

    if logger is not None:
        logger.write(f"论文真实实验样本数：{len(config.X_REAL)}")
    return True


def find_real_experiment_index(x):
    """在论文真实实验数据中查找参数组合，找不到返回 -1。"""
    x = np.asarray(x, dtype=float)
    for i in range(len(config.X_REAL)):
        if np.allclose(config.X_REAL[i], x, atol=1e-12):
            return i
    return -1


def get_paper_real_value():
    """返回论文最优方案在真实实验数据中的下标与真实目标值。"""
    index = find_real_experiment_index(config.PAPER_OPTIMAL)
    if index < 0:
        raise ValueError("论文最优参数没有在 9 组论文真实实验数据中找到。")
    return index, config.Y_REAL[index].copy()


def generate_full_space():
    """生成完整离散参数空间：3 × 3 × 3 × 3 = 81 组。"""
    candidates = [
        [a, b, c, d]
        for a, b, c, d in product(
            config.LEVELS["A"],
            config.LEVELS["B"],
            config.LEVELS["C"],
            config.LEVELS["D"],
        )
    ]
    return np.asarray(candidates, dtype=float)
