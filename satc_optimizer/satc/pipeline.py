"""SATC-NSGA-II 主流程编排。"""

import time
import traceback

import numpy as np

from satc import config
from satc.constraints import thermal_constraint
from satc.data import (
    find_real_experiment_index,
    generate_full_space,
    get_paper_real_value,
    validate_real_data,
)
from satc.gpr import SurrogateModel
from satc.io import (
    save_all_predictions,
    save_loo,
    save_pareto_results,
    save_summary,
)
from satc.logging_utils import Logger
from satc.pareto import (
    compromise_scores,
    constrained_pareto,
    dominates,
    percent_score,
    select_compromise,
)
from satc.validation import loo_validation


def normalize_weights(weights, n=3):
    """校验并归一化目标权重；None 表示等权。"""
    if weights is None:
        return np.ones(n, dtype=float) / n
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"权重数量必须为 {n} 个，当前为 {w.shape}。")
    if not np.all(np.isfinite(w)):
        raise ValueError("权重必须为有限数值。")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("权重之和必须大于 0。")
    return w / total


def build_optimization_objectives(X_ALL, GPR_F):
    """
    构造 81 组优化目标值：
    9 个论文真实实验点使用真实目标值，其余 72 组使用 GPR 预测值。
    """
    F_OPT = np.asarray(GPR_F, dtype=float).copy()
    data_types = []
    real_indices = []

    for i, x in enumerate(X_ALL):
        real_index = find_real_experiment_index(x)
        if real_index >= 0:
            F_OPT[i] = config.Y_REAL[real_index]
            data_types.append("论文真实实验数据")
        else:
            data_types.append("GPR代理预测")
        real_indices.append(real_index)

    return F_OPT, data_types, np.asarray(real_indices, dtype=int)


def compare_with_paper(logger, optimized_x, optimized_f, optimized_std):
    """SATC 推荐方案与论文方案比较。"""
    logger.section("SATC推荐方案与论文方案比较")
    paper_index, paper_f = get_paper_real_value()

    logger.write(f"论文方案：{config.PAPER_OPTIMAL_NAME}")
    logger.write(f"论文真实实验编号：{paper_index + 1}")
    logger.write()

    logger.write("参数比较")
    logger.write("-" * 90)
    for i, name in enumerate(config.PARAMETER_NAMES):
        logger.write(
            f"{name} = "
            f"论文 {config.PAPER_OPTIMAL[i]:>8.3f} "
            f"{config.PARAMETER_UNITS[i]:<5}"
            f"SATC {optimized_x[i]:>8.3f} "
            f"{config.PARAMETER_UNITS[i]}"
        )

    logger.write()
    logger.write("目标值比较")
    logger.write("-" * 90)
    for i, name in enumerate(config.OBJECTIVE_NAMES):
        if abs(paper_f[i]) > 1e-12:
            improvement = (
                (paper_f[i] - optimized_f[i]) / paper_f[i] * 100.0
            )
        else:
            improvement = np.nan
        logger.write(
            f"{name:<18}"
            f"论文真实 = {paper_f[i]:>9.4f}    "
            f"SATC = {optimized_f[i]:>9.4f}    "
            f"Std = {optimized_std[i]:>9.4f}    "
            f"变化 = {improvement:+9.2f}%"
        )

    logger.write()
    logger.write("Pareto支配关系")
    logger.write("-" * 90)
    if dominates(optimized_f, paper_f):
        logger.write("SATC推荐方案的目标预测值支配论文方案真实实验值。")
        logger.write("注意：该结论仅代表代理模型预测，必须通过后续真实实验验证。")
    elif dominates(paper_f, optimized_f):
        logger.write("论文方案真实实验值支配SATC推荐方案。")
    else:
        logger.write("两种方案互不支配，属于不同Pareto权衡方案。")


def run_pipeline(output_dir=None, logger=None, weights=None):
    """
    运行完整优化流程，返回结果字典。

    output_dir：输出目录（默认 results/）。
    weights：三个目标的权重（默认等权）。
    """
    out_dir = config.resolve_output_dir(output_dir)
    if logger is None:
        logger = Logger(out_dir / config.LOG_FILENAME)

    start_time = time.time()

    try:
        # 0. 权重校验
        w_norm = normalize_weights(weights)
        logger.write(
            f"目标权重（归一化）："
            f"ΔT={w_norm[0]:.3f}, ΔB={w_norm[1]:.3f}, ΔS={w_norm[2]:.3f}"
        )

        # 1. 真实数据检查
        logger.section("1. 论文真实实验数据检查")
        validate_real_data(logger)

        # 2. LOO 验证
        logger.section("2. 代理模型 Leave-One-Out 验证")
        loo_predictions, loo_std, loo_metrics = loo_validation(
            config.X_REAL, config.Y_REAL
        )
        for m in loo_metrics:
            logger.write(
                f"{m['objective']:<18}"
                f"MAE = {m['MAE']:10.4f}    "
                f"RMSE = {m['RMSE']:10.4f}    "
                f"R² = {m['R2']:10.4f}"
            )
        save_loo(
            config.X_REAL, config.Y_REAL,
            loo_predictions, loo_std, output_dir=out_dir,
        )

        # 3. 使用全部 9 组数据训练最终 GPR
        logger.section("3. 使用全部9组论文真实数据训练最终GPR")
        surrogate = SurrogateModel()
        surrogate.fit(config.X_REAL, config.Y_REAL)

        # 4. 生成 81 组完整参数空间
        logger.section("4. 生成完整离散参数空间")
        X_ALL = generate_full_space()
        logger.write(f"完整离散参数空间：{len(X_ALL)} 组（3×3×3×3）")

        # 5. GPR 预测
        logger.section("5. 对81组参数组合进行GPR代理预测")
        prediction_start = time.perf_counter()
        GPR_F_ALL, GPR_STD_ALL = surrogate.predict(
            X_ALL, return_std=True
        )
        prediction_elapsed = time.perf_counter() - prediction_start
        logger.write(f"GPR预测矩阵：{GPR_F_ALL.shape}")
        logger.write(f"GPR预测总耗时：{prediction_elapsed:.6f} s")

        # 6. 构造优化目标
        logger.section("6. 构造81组优化目标值")
        F_OPT_ALL, DATA_TYPES, REAL_INDICES = (
            build_optimization_objectives(X_ALL, GPR_F_ALL)
        )
        real_count = int(np.sum(REAL_INDICES >= 0))
        logger.write(f"论文真实实验点：{real_count}")
        logger.write(f"GPR代理预测点：{len(X_ALL) - real_count}")
        if real_count != 9:
            raise RuntimeError("论文真实实验点数量不是9。")

        # 7. 保存 81 组
        logger.section("7. 保存81组参数空间结果")
        save_all_predictions(
            X_ALL, GPR_F_ALL, GPR_STD_ALL, F_OPT_ALL,
            DATA_TYPES, REAL_INDICES, output_dir=out_dir,
        )

        # 8. 热约束检查
        logger.section("8. 热约束检查")
        violations = np.array(
            [thermal_constraint(x) for x in X_ALL], dtype=float
        )
        feasible_count = int(np.sum(violations <= 1e-12))
        logger.write(f"全部组合：{len(X_ALL)}")
        logger.write(f"热约束可行组合：{feasible_count}")
        logger.write(f"热约束不可行组合：{len(X_ALL) - feasible_count}")

        # 9. Pareto 前沿
        logger.section("9. SATC代理辅助Pareto Front")
        front_indices, _ = constrained_pareto(
            X_ALL, F_OPT_ALL, violations=violations
        )
        X_FRONT = X_ALL[front_indices]
        F_FRONT = F_OPT_ALL[front_indices]
        GPR_STD_FRONT = GPR_STD_ALL[front_indices]
        DATA_TYPES_FRONT = [DATA_TYPES[i] for i in front_indices]
        REAL_INDICES_FRONT = [int(REAL_INDICES[i]) for i in front_indices]
        logger.write(f"Pareto解数量：{len(X_FRONT)}")

        # 10. 保存 Pareto
        logger.section("10. 保存Pareto结果")
        save_pareto_results(
            X_FRONT, F_FRONT, GPR_STD_FRONT,
            DATA_TYPES_FRONT, REAL_INDICES_FRONT, output_dir=out_dir,
        )

        # 11. 综合推荐
        logger.section("11. SATC综合推荐方案")
        best_x, best_f, score_raw, best_local_index = select_compromise(
            X_FRONT, F_FRONT, weights=w_norm
        )
        best_global_index = int(front_indices[best_local_index])
        best_std = GPR_STD_ALL[best_global_index]
        best_data_type = DATA_TYPES[best_global_index]
        best_real_index = int(REAL_INDICES[best_global_index])

        # 论文方案折中评分：与前沿解使用相同的归一化基准，便于公平对比
        _, paper_f = get_paper_real_value()
        front_f_min = F_FRONT.min(axis=0)
        front_f_max = F_FRONT.max(axis=0)
        paper_score_raw = float(compromise_scores(
            paper_f[None, :], f_min=front_f_min, f_max=front_f_max,
            weights=w_norm,
        )[0])

        # 对外统一使用百分制：100 分最好
        score = percent_score(score_raw)
        paper_score = percent_score(paper_score_raw)

        logger.write("SATC推荐参数：")
        for i, name in enumerate(config.PARAMETER_NAMES):
            logger.write(
                f"{name} = {best_x[i]:.3f} {config.PARAMETER_UNITS[i]}"
            )
        logger.write()
        logger.write("推荐方案数据来源：" + best_data_type)
        logger.write()
        logger.write("用于Pareto优化的目标值：")
        for i, name in enumerate(config.OBJECTIVE_NAMES):
            logger.write(f"{name} = {best_f[i]:.6f}")
        logger.write()
        logger.write(f"综合评分（百分制，100 分最好）：{score:.2f}")
        logger.write(f"论文方案综合评分（百分制）：{paper_score:.2f}")
        logger.write(f"GPR预测标准差：{best_std}")
        logger.write()
        if best_real_index >= 0:
            logger.write(f"推荐点类型：论文真实实验点（编号 {best_real_index + 1}）")
        else:
            logger.write("推荐点类型：论文未实验参数组合（GPR预测），建议后续真实实验验证。")

        # 12. 与论文比较
        compare_with_paper(logger, best_x, best_f, best_std)

        # 13. 保存摘要
        logger.section("12. 保存最终摘要")
        save_summary(
            best_x, best_f, best_std, score,
            best_data_type, paper_score, output_dir=out_dir,
        )

        # 14. 完成
        elapsed = time.time() - start_time
        logger.section("13. 程序运行完成")
        logger.write(f"总运行时间：{elapsed:.2f} s")
        logger.write(f"输出目录：{out_dir}")

        return {
            "best_x": best_x,
            "best_f": best_f,
            "best_std": best_std,
            "score": score,
            "best_data_type": best_data_type,
            "best_real_index": best_real_index,
            "best_global_index": best_global_index,
            "paper_score": paper_score,
            "weights": w_norm,
            "front_indices": front_indices,
            "loo_metrics": loo_metrics,
            "n_pareto": len(X_FRONT),
            "output_dir": out_dir,
        }

    except Exception as e:
        logger.section("程序异常终止")
        logger.write(f"异常类型：{type(e).__name__}")
        logger.write(f"异常信息：{str(e)}")
        logger.write()
        logger.write("完整Traceback：")
        logger.write(traceback.format_exc())
        raise
