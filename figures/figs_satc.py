"""5.3 代理辅助多目标工艺参数优化算法配图。"""

import sys
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import save

PROJECT = Path(__file__).resolve().parent.parent
SATC = PROJECT / "satc_optimizer"
DATA = PROJECT / "data"
sys.path.insert(0, str(SATC))

from satc import config
from satc.data import (
    find_real_experiment_index,
    generate_full_space,
    get_paper_real_value,
)
from satc.gpr import SurrogateModel
from satc.mechanics import STRESS_FILE_MAP, suggest_weights
from satc.pareto import (
    constrained_pareto,
    select_compromise,
)
from satc.validation import loo_validation


@lru_cache(maxsize=1)
def _model():
    m = SurrogateModel()
    m.fit(config.X_REAL, config.Y_REAL)
    return m


@lru_cache(maxsize=1)
def _space():
    X = generate_full_space()
    F_pred = _model().predict(X)
    F_opt = F_pred.copy()
    for i, x in enumerate(X):
        ri = find_real_experiment_index(x)
        if ri >= 0:
            F_opt[i] = config.Y_REAL[ri]
    return X, F_pred, F_opt


def _front():
    X, _Fp, F_opt = _space()
    from satc.constraints import thermal_constraint

    violations = np.array([thermal_constraint(x) for x in X], dtype=float)
    idx, _ = constrained_pareto(X, F_opt, violations=violations)
    return X[idx], F_opt[idx], idx


def fig_5_11_parameter_space():
    """图5-11 81 组参数空间与 9 组正交布点（4×4 关系矩阵）。"""
    X, _Fp, _Fo = _space()
    real = np.array([find_real_experiment_index(x) >= 0 for x in X])
    names = config.PARAMETER_NAMES
    units = config.PARAMETER_UNITS
    jit = [0.005, 0.005, 1.0, 0.3]
    cmap = plt.get_cmap("Blues")
    rng = np.random.default_rng(0)
    n = 4

    fig, axes = plt.subplots(n, n, figsize=(11, 10))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # 对角：各参数水平分布
                vals = X[:, i]
                levels = np.unique(vals)
                counts = [int((vals == lv).sum()) for lv in levels]
                colors = cmap(np.linspace(0.35, 0.9, len(levels)))
                ax.bar(range(len(levels)), counts, color=colors, width=0.55)
                ax.set_xticks(range(len(levels)))
                ax.set_xticklabels([f"{lv:g}" for lv in levels], fontsize=7)
                ax.set_title(f"{names[i]} ({units[i]})", fontsize=11)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.2, axis="y")
                ax.spines[["top", "right"]].set_visible(False)
            elif i < j:
                xi = X[:, i] + rng.normal(0, jit[i], len(X))
                xj = X[:, j] + rng.normal(0, jit[j], len(X))
                ax.scatter(xi[~real], xj[~real], s=20, c="#B0BEC5",
                           alpha=0.75, edgecolors="none", zorder=2)
                ax.scatter(xi[real], xj[real], s=60, marker="*",
                           c="#D62728", edgecolors="black", linewidths=0.5,
                           zorder=3)
                ax.tick_params(labelsize=7)
                ax.grid(alpha=0.2)
                if i == 0:
                    ax.set_title(f"{names[j]}", fontsize=11)
                if j == i + 1:
                    ax.set_ylabel(f"{names[i]}", fontsize=10)
            else:
                # 下三角：预测点密度热图（hexbin）+ 真实实验点
                xi = X[:, i] + rng.normal(0, jit[i], len(X))
                xj = X[:, j] + rng.normal(0, jit[j], len(X))
                ax.hexbin(xi[~real], xj[~real], gridsize=16, cmap="Blues",
                          alpha=0.8, mincnt=1, linewidths=0)
                ax.scatter(xi[real], xj[real], s=55, marker="*",
                           c="#D62728", edgecolors="black", linewidths=0.5,
                           zorder=3)
                ax.tick_params(labelsize=7)
                if j == 0:
                    ax.set_ylabel(f"{names[i]}", fontsize=10)

    handles = [
        plt.Line2D([0], [0], marker="o", ls="", ms=7, c="#B0BEC5",
                   label="GPR 预测点（72 组）"),
        plt.Line2D([0], [0], marker="*", ls="", ms=12, c="#D62728",
                   label="真实实验点（9 组）"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=2, fontsize=10, frameon=False)
    fig.suptitle("81 组离散参数空间与 9 组正交实验布点", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig5_11_parameter_space")


def fig_5_12_loo():
    """图5-12 GPR 留一法验证：预测对比 + 残差分析（带 ±2σ 不确定性）。"""
    pred, std, metrics = loo_validation(config.X_REAL, config.Y_REAL)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    cmap = plt.get_cmap("Blues")
    for j in range(3):
        yt = config.Y_REAL[:, j]
        yp = pred[:, j]
        sd = std[:, j]
        res = yp - yt
        ab = np.abs(res)

        # 上排：实测 vs 预测（±2σ 误差棒 + ±10% 容差带）
        ax = axes[0, j]
        lo = min(yt.min(), yp.min()) - 2
        hi = max(yt.max(), yp.max()) + 2
        xb = np.linspace(lo, hi, 60)
        ax.fill_between(xb, 0.9 * xb, 1.1 * xb, color="#EAF2FB", alpha=0.85,
                        label="±10% 容差带")
        ax.errorbar(yt, yp, yerr=2 * sd, fmt="none", ecolor="#9AA5B1",
                    elinewidth=1.0, capsize=3, zorder=1)
        ax.scatter(yt, yp, c=ab, cmap=cmap, s=72, edgecolors="white",
                   linewidths=1.2, zorder=3, vmin=0, vmax=float(ab.max()))
        ax.plot([lo, hi], [lo, hi], "--", color="#D62728", lw=1.4,
                label="45° 理想线")
        m = metrics[j]
        ax.text(0.04, 0.97, f"MAE = {m['MAE']:.2f}\n"
                            f"RMSE = {m['RMSE']:.2f}\n"
                            f"R²  = {m['R2']:.3f}",
                transform=ax.transAxes, fontsize=9.5, va="top",
                bbox=dict(fc="white", ec="#CCCCCC", alpha=0.92, pad=4))
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("实测值", fontsize=10)
        ax.set_ylabel("LOO 预测值", fontsize=10)
        ax.set_title(f"({chr(97 + j)}) {config.OBJECTIVE_NAMES[j]}",
                     fontsize=12)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, loc="lower right")

        # 下排：预测残差（±2σ 误差棒 + 残差带）
        ax2 = axes[1, j]
        mean_r = float(res.mean())
        std_r = float(res.std())
        ax2.fill_between([0.5, 9.5], mean_r - 2 * std_r, mean_r + 2 * std_r,
                         color="#EAF2FB", alpha=0.85, label="±2σ 残差带")
        ax2.axhline(0, color="#D62728", lw=1.2, ls="--")
        ax2.errorbar(np.arange(9) + 1, res, yerr=2 * sd, fmt="none",
                     ecolor="#9AA5B1", elinewidth=1.0, capsize=3, zorder=1)
        ax2.scatter(np.arange(9) + 1, res, c=ab, cmap=cmap, s=58,
                    edgecolors="white", linewidths=1.0, zorder=3,
                    vmin=0, vmax=float(ab.max()))
        ax2.set_xlim(0.5, 9.5)
        ax2.set_xlabel("实验编号", fontsize=10)
        ax2.set_ylabel("预测残差", fontsize=10)
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("GPR 代理模型留一法验证（9 组真实实验）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig5_12_loo")


def fig_5_13_gpr_uncertainty():
    """图5-13 GPR 预测均值 ±2σ 置信带（扫描打印速度）。"""
    model = _model()
    x0 = config.PAPER_OPTIMAL.copy()
    D = np.linspace(30, 50, 120)
    Xs = np.tile(x0, (len(D), 1))
    Xs[:, 3] = D
    mean, std = model.predict(Xs, return_std=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    for ax, j, name in zip(axes, range(3), config.OBJECTIVE_NAMES):
        ax.plot(D, mean[:, j], color="#2C7FB8", lw=2, label="预测均值")
        ax.fill_between(D, mean[:, j] - 2 * std[:, j],
                        mean[:, j] + 2 * std[:, j], color="#72B7B2",
                        alpha=0.45, label="±2σ 置信带")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("打印速度 D (mm/s)", fontsize=10)
        ax.set_ylabel("预测值", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("固定 A/B/C（论文方案），扫描打印速度的代理预测", fontsize=14)
    fig.tight_layout()
    save(fig, "fig5_13_gpr_uncertainty")


def fig_5_14_all_predictions():
    """图5-14 81 组全空间预测结果（真实/预测区分）。"""
    X, _Fp, F_opt = _space()
    real = np.array([find_real_experiment_index(x) >= 0 for x in X])
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(F_opt[~real, 0], F_opt[~real, 1], F_opt[~real, 2],
               c="#B0BEC5", s=24, alpha=0.7, label="GPR 预测（72 组）")
    ax.scatter(F_opt[real, 0], F_opt[real, 1], F_opt[real, 2],
               c="#D62728", s=80, marker="*", label="真实实验（9 组）")
    ax.set_xlabel("ΔT 拉伸偏差"); ax.set_ylabel("ΔB 弯曲偏差")
    ax.set_zlabel("ΔS 层间剪切")
    ax.set_title("81 组参数空间三目标分布", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "fig5_14_all_predictions")


def fig_5_15_pareto():
    """图5-15 Pareto 前沿与推荐方案。"""
    X, _Fp, F_opt = _space()
    X_f, F_f, idx = _front()
    best_x, best_f, score_raw, _li = select_compromise(X_f, F_f)
    from satc.pareto import percent_score

    _paper_idx, paper_f = get_paper_real_value()
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(F_opt[:, 0], F_opt[:, 1], F_opt[:, 2],
               c="#D6DDE6", s=22, alpha=0.75, label="81 组解")
    ax.scatter(F_f[:, 0], F_f[:, 1], F_f[:, 2],
               c="#2C7FB8", s=50, alpha=0.95, label="Pareto 前沿")
    ax.scatter(*best_f, c="#D62728", s=150, marker="^", label="SATC 推荐")
    ax.scatter(*paper_f, c="#F2CF5B", s=140, marker="s",
               edgecolors="black", linewidths=0.8, label="论文方案")
    ax.set_xlabel("ΔT"); ax.set_ylabel("ΔB"); ax.set_zlabel("ΔS")
    ax.set_title(f"约束 Pareto 前沿（推荐评分 "
                 f"{percent_score(score_raw):.1f} 分）", fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "fig5_15_pareto")


def fig_5_16_weight_sensitivity():
    """图5-16 权重对折中推荐的影响（ΔT–ΔS 投影）。"""
    X_f, F_f, _idx = _front()
    combos = [
        ("等权", (1, 1, 1)),
        ("强度优先", (0.5, 0.25, 0.25)),
        ("效率优先", (0.2, 0.2, 0.6)),
    ]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.scatter(F_f[:, 0], F_f[:, 2], c="#2C7FB8", s=45, label="Pareto 前沿")
    markers = ["^", "s", "D"]
    for (lbl, w), mk in zip(combos, markers):
        _x, f, _s, _i = select_compromise(X_f, F_f, weights=np.array(w))
        ax.scatter(f[0], f[2], marker=mk, s=120, edgecolors="black",
                   linewidths=0.8, label=lbl)
    ax.set_xlabel("ΔT 拉伸偏差"); ax.set_ylabel("ΔS 层间剪切")
    ax.set_title("不同目标权重下的折中推荐位置", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "fig5_16_weight_sensitivity")


def fig_5_17_exposure_weights():
    """图5-17 力学暴露度与自动权重（渐变条形 + 环形图 + 权重含义）。"""
    res = suggest_weights(
        str(DATA / "node_coordinates_s.csv"),
        stress_files={
            comp: str(DATA / name)
            for name, comp in STRESS_FILE_MAP.items()
        },
    )
    exps = res["exposures"]
    keys = ["tensile", "bending", "shear"]
    short = ["拉伸", "弯曲", "层间剪切"]
    meaning = ["σ1 正分量（拉应力）", "挠度 / 分层 σ1 变化", "τxz / τyz 层间剪应力"]
    vals = [exps[k] if exps[k] is not None else 0.0 for k in keys]
    w = res["weights"]

    # 蓝紫冷色系淡色：每根柱子一个颜色
    light_colors = ["#CEBEEA", "#AFCBE3", "#8FBEDC"]
    donut_colors = ["#AE98D8", "#7DA6CB", "#5594C4"]

    fig = plt.figure(figsize=(14.5, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.9, 1.15],
                          wspace=0.4)

    # (a) 暴露度：淡色柱状图
    ax = fig.add_subplot(gs[0])
    vmax = max(max(vals), 1e-6) * 1.25
    ax.bar(range(3), vals, color=light_colors, width=0.55)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, vmax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(short, fontsize=10)
    ax.set_yticks([])
    for k, v in enumerate(vals):
        ax.text(k, v + vmax * 0.03, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_title("(a) 三类失效风险暴露度", fontsize=12)
    ax.set_ylabel(r"暴露度 $e=\mathrm{mean}((v/v_{max})^2)$", fontsize=10)
    ax.grid(alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # (b) 权重环形图
    ax2 = fig.add_subplot(gs[1])
    colors = donut_colors
    wedges, _t, autotexts = ax2.pie(
        w, colors=colors, startangle=90,
        autopct=lambda p: f"{p:.1f}%", pctdistance=0.78,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(9.5)
    ax2.text(0, 0.08, "目标权重", ha="center", fontsize=11)
    ax2.text(0, -0.16, "归一化", ha="center", fontsize=9, color="#666666")
    ax2.legend(
        wedges, [f"ΔT {short[0]}", f"ΔB {short[1]}", f"ΔS {short[2]}"],
        loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8.5,
        frameon=False,
    )
    ax2.set_title("(b) 归一化目标权重", fontsize=12)

    # (c) 权重横向淡色条 + 物理含义
    ax3 = fig.add_subplot(gs[2])
    ax3.barh(range(3), w, color=light_colors, height=0.55)
    for k in range(3):
        ax3.text(w[k] + 0.01, k, f"{w[k]:.3f}", va="center", fontsize=10)
    ax3.set_xlim(0, max(w) + 0.12)
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_yticks(range(3))
    ax3.set_yticklabels(
        [f"ΔT 拉伸\n{meaning[0]}", f"ΔB 弯曲\n{meaning[1]}",
         f"ΔS 层剪\n{meaning[2]}"],
        fontsize=8.8,
    )
    ax3.set_xlabel("权重", fontsize=10)
    ax3.set_title("(c) 权重数值与物理含义", fontsize=12)
    ax3.grid(alpha=0.2, axis="x")
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("基于 ANSYS 力学结果的自动权重", fontsize=15)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.20,
                        wspace=0.4)
    save(fig, "fig5_17_exposure_weights")


def run_all():
    """生成 5.3 节全部可自动绘制的图（界面截图除外）。"""
    fig_5_11_parameter_space()
    fig_5_12_loo()
    fig_5_13_gpr_uncertainty()
    fig_5_14_all_predictions()
    fig_5_15_pareto()
    fig_5_16_weight_sensitivity()
    fig_5_17_exposure_weights()
    print("5.3 配图完成（图5-18 为软件界面截图，需人工截取）")


if __name__ == "__main__":
    run_all()
