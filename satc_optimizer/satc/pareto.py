"""Pareto 支配、前沿与折中选解。"""

import numpy as np

from satc.constraints import thermal_constraint


def dominates(f1, f2):
    """f1 是否支配 f2：全部不劣且至少一项严格更优（最小化问题）。"""
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    return bool(np.all(f1 <= f2) and np.any(f1 < f2))


def pareto_front(F):
    """返回非支配点索引（最小化问题）。"""
    F = np.asarray(F, dtype=float)
    result = []
    for i in range(len(F)):
        dominated = any(
            dominates(F[j], F[i])
            for j in range(len(F))
            if j != i
        )
        if not dominated:
            result.append(i)
    return np.asarray(result, dtype=int)


def constrained_pareto(X, F, violations=None):
    """在满足热约束的解中计算 Pareto 前沿；无可行解时退回违规最小 10 个。"""
    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=float)
    if violations is None:
        violations = np.array(
            [thermal_constraint(x) for x in X], dtype=float
        )

    feasible_mask = violations <= 1e-12
    feasible_indices = np.where(feasible_mask)[0]

    if len(feasible_indices) == 0:
        best = np.argsort(violations)
        return best[:10], violations

    local_front = pareto_front(F[feasible_indices])
    return feasible_indices[local_front], violations


def compromise_scores(F, f_min=None, f_max=None, weights=None):
    """
    计算每个解的折中评分：各目标按 [f_min, f_max] 归一化后加权平均。

    0 = 全部目标均为最优，越小表示越均衡/越接近最优。
    传入 f_min/f_max 时，可对前沿之外的方案（如论文方案）
    使用与前沿解相同的基准做公平对比。
    weights：各目标权重（长度 = 目标数），默认等权；内部自动归一化。
    """
    F = np.asarray(F, dtype=float)
    if f_min is None:
        f_min = F.min(axis=0)
    if f_max is None:
        f_max = F.max(axis=0)

    denominator = np.where(f_max - f_min < 1e-12, 1.0, f_max - f_min)
    normalized = (F - f_min) / denominator

    if weights is None:
        return normalized.mean(axis=1)

    weights = np.asarray(weights, dtype=float)
    if weights.shape != (F.shape[1],):
        raise ValueError(f"权重长度必须等于目标数 {F.shape[1]}。")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("权重之和必须大于 0。")
    return normalized @ (weights / total)


def percent_score(score):
    """
    把 0~1 折中评分（越小越好）转换为百分制（100 分最好）。

    前沿内方案评分在 [0, 1]，转换后在 [0, 100]；
    前沿外的方案可能得到 <0 或 >100，表示其劣于/优于前沿整体范围。
    """
    return 100.0 * (1.0 - float(score))


def select_compromise(X, F, weights=None):
    """在 Pareto 前沿上选折中解：各目标归一化后加权平均分最小。"""
    X = np.asarray(X, dtype=float)
    F = np.asarray(F, dtype=float)

    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    scores = compromise_scores(
        F, f_min=f_min, f_max=f_max, weights=weights
    )

    idx = int(np.argmin(scores))
    return X[idx].copy(), F[idx].copy(), float(scores[idx]), idx
