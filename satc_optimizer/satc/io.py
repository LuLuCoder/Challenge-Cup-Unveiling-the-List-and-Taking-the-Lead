"""CSV 结果保存。"""

import numpy as np
import pandas as pd

from satc import config
from satc.data import get_paper_real_value


def _save(df, filename, output_dir):
    out_dir = config.resolve_output_dir(output_dir)
    path = out_dir / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_loo(X, Y, predictions, std_predictions, output_dir=None):
    """保存 LOO 验证结果。"""
    rows = []
    for i in range(len(X)):
        rows.append({
            "Experiment": i + 1,
            "A": X[i, 0],
            "B": X[i, 1],
            "C": X[i, 2],
            "D": X[i, 3],
            "Real_DeltaT": Y[i, 0],
            "Pred_DeltaT": predictions[i, 0],
            "Std_DeltaT": std_predictions[i, 0],
            "Real_DeltaB": Y[i, 1],
            "Pred_DeltaB": predictions[i, 1],
            "Std_DeltaB": std_predictions[i, 1],
            "Real_DeltaS": Y[i, 2],
            "Pred_DeltaS": predictions[i, 2],
            "Std_DeltaS": std_predictions[i, 2],
        })
    return _save(
        pd.DataFrame(rows), config.LOO_FILENAME, output_dir
    )


def save_all_predictions(X_ALL, GPR_F, GPR_STD, F_OPT,
                         data_types, real_indices, output_dir=None):
    """保存 81 组参数空间的结果（9 组真实 + 72 组 GPR 预测）。"""
    rows = []
    for i in range(len(X_ALL)):
        real_index = int(real_indices[i])
        rows.append([
            i + 1,
            X_ALL[i, 0], X_ALL[i, 1], X_ALL[i, 2], X_ALL[i, 3],
            data_types[i],
            real_index + 1 if real_index >= 0 else np.nan,
            F_OPT[i, 0], F_OPT[i, 1], F_OPT[i, 2],
            GPR_F[i, 0], GPR_STD[i, 0],
            GPR_F[i, 1], GPR_STD[i, 1],
            GPR_F[i, 2], GPR_STD[i, 2],
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Index",
            "LayerThickness_mm",
            "FirstLayerThickness_mm",
            "NozzleTemperature_C",
            "PrintingSpeed_mm_s",
            "DataType",
            "PaperExperiment",
            "Optimization_DeltaT",
            "Optimization_DeltaB",
            "Optimization_DeltaS",
            "GPR_DeltaT",
            "GPR_Std_DeltaT",
            "GPR_DeltaB",
            "GPR_Std_DeltaB",
            "GPR_DeltaS",
            "GPR_Std_DeltaS",
        ],
    )
    return _save(df, config.ALL_PREDICTIONS_FILENAME, output_dir)


def save_pareto_results(X_FRONT, F_FRONT, F_STD_FRONT,
                        data_types_front, real_indices_front,
                        output_dir=None):
    """保存 Pareto 前沿结果。"""
    rows = []
    for i in range(len(X_FRONT)):
        real_index = int(real_indices_front[i])
        rows.append([
            i + 1,
            X_FRONT[i, 0], X_FRONT[i, 1], X_FRONT[i, 2], X_FRONT[i, 3],
            data_types_front[i],
            real_index + 1 if real_index >= 0 else np.nan,
            F_FRONT[i, 0], F_FRONT[i, 1], F_FRONT[i, 2],
            F_STD_FRONT[i, 0],
            F_STD_FRONT[i, 1],
            F_STD_FRONT[i, 2],
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "ParetoIndex",
            "LayerThickness_mm",
            "FirstLayerThickness_mm",
            "NozzleTemperature_C",
            "PrintingSpeed_mm_s",
            "DataType",
            "PaperExperiment",
            "DeltaT",
            "DeltaB",
            "DeltaS",
            "GPR_Std_DeltaT",
            "GPR_Std_DeltaB",
            "GPR_Std_DeltaS",
        ],
    )
    return _save(df, config.PARETO_FILENAME, output_dir)


def save_summary(best_x, best_f, best_std, score,
                 best_data_type, paper_score, output_dir=None):
    """保存最终摘要：论文方案 vs SATC 推荐方案。"""
    _, paper_f = get_paper_real_value()

    rows = [
        [
            "论文方案",
            config.PAPER_OPTIMAL[0],
            config.PAPER_OPTIMAL[1],
            config.PAPER_OPTIMAL[2],
            config.PAPER_OPTIMAL[3],
            "论文真实实验数据",
            paper_f[0], paper_f[1], paper_f[2],
            np.nan, np.nan, np.nan,
            paper_score,
        ],
        [
            "SATC推荐方案",
            best_x[0], best_x[1], best_x[2], best_x[3],
            best_data_type,
            best_f[0], best_f[1], best_f[2],
            best_std[0], best_std[1], best_std[2],
            score,
        ],
    ]

    df = pd.DataFrame(
        rows,
        columns=[
            "方案",
            "A_mm",
            "B_mm",
            "C_C",
            "D_mm_s",
            "DataType",
            "DeltaT",
            "DeltaB",
            "DeltaS",
            "Std_DeltaT",
            "Std_DeltaB",
            "Std_DeltaS",
            "CompromiseScore_100",
        ],
    )
    return _save(df, config.SUMMARY_FILENAME, output_dir)
