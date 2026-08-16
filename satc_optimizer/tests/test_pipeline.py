"""主流程集成测试。"""

import pandas as pd
import pytest
import numpy as np

from satc import config
from satc.pipeline import run_pipeline


def test_run_pipeline(tmp_path):
    result = run_pipeline(output_dir=str(tmp_path))

    # 输出文件齐全
    for filename in (
        config.LOG_FILENAME,
        config.LOO_FILENAME,
        config.ALL_PREDICTIONS_FILENAME,
        config.PARETO_FILENAME,
        config.SUMMARY_FILENAME,
    ):
        assert (tmp_path / filename).exists()

    # 推荐参数必须落在离散水平内
    for i, name in enumerate(config.PARAMETER_NAMES):
        assert result["best_x"][i] in config.LEVELS[name]

    # Pareto 前沿非空
    assert result["n_pareto"] >= 1

    # 百分制评分：推荐方案（前沿最低均衡分）应不劣于论文方案
    assert 0.0 <= result["score"] <= 100.0
    assert np.isfinite(result["paper_score"])
    assert result["score"] >= result["paper_score"]

    # LOO 指标结构
    assert len(result["loo_metrics"]) == 3
    for metric in result["loo_metrics"]:
        assert set(metric) == {"objective", "MAE", "RMSE", "R2"}

    # 81 组全预测 CSV：9 组真实 + 72 组 GPR
    all_df = pd.read_csv(tmp_path / config.ALL_PREDICTIONS_FILENAME)
    assert len(all_df) == 81
    assert (all_df["DataType"] == "论文真实实验数据").sum() == 9
    assert (all_df["DataType"] == "GPR代理预测").sum() == 72

    # 真实实验点的优化目标值 = 真实值
    real_rows = all_df[all_df["DataType"] == "论文真实实验数据"]
    for _, row in real_rows.iterrows():
        paper_idx = int(row["PaperExperiment"]) - 1
        assert row["Optimization_DeltaT"] == pytest.approx(
            config.Y_REAL[paper_idx, 0]
        )


def test_run_pipeline_weighted(tmp_path):
    result = run_pipeline(output_dir=str(tmp_path), weights=[1.0, 0.0, 0.0])

    assert result["weights"][0] == pytest.approx(1.0)
    assert result["weights"][1] == pytest.approx(0.0)
    assert result["weights"][2] == pytest.approx(0.0)

    # 只优化 ΔT 时，推荐方案的 ΔT 应等于 81 组中的最小值
    all_df = pd.read_csv(tmp_path / config.ALL_PREDICTIONS_FILENAME)
    assert result["best_f"][0] == pytest.approx(
        all_df["Optimization_DeltaT"].min()
    )


def test_run_pipeline_invalid_weights(tmp_path):
    with pytest.raises(ValueError):
        run_pipeline(output_dir=str(tmp_path), weights=[1.0, 0.0])
