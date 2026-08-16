"""留一法（LOO）验证代理模型。"""

import numpy as np

from satc import config
from satc.gpr import SurrogateModel


def loo_validation(X, Y):
    """对每个样本留一训练并预测，返回预测值、标准差和指标列表。"""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(X)

    predictions = np.zeros_like(Y, dtype=float)
    std_predictions = np.zeros_like(Y, dtype=float)

    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False

        model = SurrogateModel()
        model.fit(X[train_mask], Y[train_mask])
        pred, std = model.predict(X[i:i + 1], return_std=True)

        predictions[i] = pred[0]
        std_predictions[i] = std[0]

    metrics = []
    for j, name in enumerate(config.OBJECTIVE_NAMES):
        y_true = Y[:, j]
        y_pred = predictions[:, j]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        metrics.append({
            "objective": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

    return predictions, std_predictions, metrics
