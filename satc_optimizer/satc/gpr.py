"""手动实现的高斯过程回归（GPR）代理模型。

不调用 sklearn 的 GaussianProcessRegressor.predict()，
按标准 GPR 解析公式计算预测均值与标准差，避免
Windows + scipy/BLAS 下小样本预测偶发阻塞。
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from satc import config


class SurrogateModel:
    """RBF 核 GPR，每个优化目标单独训练一个模型。"""

    def __init__(self):
        self.x_scaler = StandardScaler()
        self.models = []
        self.fitted = False

    # ---------- 核函数 ----------

    @staticmethod
    def rbf_kernel(X1, X2, length_scale=None):
        """RBF 核：exp(-0.5 * ||x1-x2||² / length_scale²)。"""
        if length_scale is None:
            length_scale = config.GPR_LENGTH_SCALE
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)

        sq_dist = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1, keepdims=True).T
            - 2.0 * X1 @ X2.T
        )
        sq_dist = np.maximum(sq_dist, 0.0)
        return np.exp(-0.5 * sq_dist / (length_scale ** 2))

    # ---------- 训练 ----------

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        self._validate_shape(X, Y)

        Xs = self.x_scaler.fit_transform(X)
        self.models = [
            self._fit_single(Xs, Y[:, j])
            for j in range(Y.shape[1])
        ]
        self.fitted = True
        return self

    @staticmethod
    def _validate_shape(X, Y):
        if X.ndim != 2:
            raise ValueError("X 必须为二维数组。")
        if Y.ndim != 2:
            raise ValueError("Y 必须为二维数组。")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X 和 Y 样本数量不一致。")
        if Y.shape[1] != 3:
            raise ValueError("当前程序必须有 3 个优化目标。")

    def _fit_single(self, Xs, y):
        """训练单个目标：RBF 核 + 白噪声 + Cholesky 求解。"""
        y = np.asarray(y, dtype=np.float64)

        # 输出归一化（等价 normalize_y=True）
        y_mean = float(np.mean(y))
        y_std = float(np.std(y))
        if y_std < 1e-12:
            y_std = 1.0
        yn = (y - y_mean) / y_std

        # 协方差矩阵：RBF + WhiteKernel 噪声 + 额外 jitter
        K = self.rbf_kernel(Xs, Xs)
        K = K + config.GPR_NOISE_LEVEL * np.eye(len(Xs))
        K = K + config.GPR_EXTRA_JITTER * np.eye(len(Xs))

        L, jitter = self._cholesky_with_jitter(K)
        if L is None:
            raise RuntimeError("GPR 协方差矩阵 Cholesky 分解失败。")

        # alpha = K^-1 y：两次三角求解，不显式求逆
        temp = np.linalg.solve(L, yn)
        alpha = np.linalg.solve(L.T, temp)

        return {
            "X_train": Xs.copy(),
            "y_mean": y_mean,
            "y_std": y_std,
            "L": L,
            "alpha": alpha,
            "noise_level": config.GPR_NOISE_LEVEL,
            "jitter": jitter,
        }

    @staticmethod
    def _cholesky_with_jitter(K):
        """Cholesky 分解；失败时逐次加大 jitter，最多尝试 N 次。"""
        jitter = 0.0
        for attempt in range(config.GPR_JITTER_ATTEMPTS):
            K_try = K.copy()
            if attempt > 0:
                jitter = config.GPR_JITTER_BASE ** (
                    config.GPR_JITTER_START_EXP + attempt
                )
                K_try += jitter * np.eye(len(K))
            try:
                L = np.linalg.cholesky(K_try)
                return L, jitter
            except np.linalg.LinAlgError:
                continue
        return None, jitter

    # ---------- 预测 ----------

    def predict(self, X, return_std=False):
        """预测均值（可附带预测标准差）。"""
        if not self.fitted:
            raise RuntimeError("代理模型尚未训练。")

        X = np.asarray(X, dtype=np.float64)
        Xs = self.x_scaler.transform(X)

        predictions = []
        stds = []
        for model in self.models:
            K_star = self.rbf_kernel(Xs, model["X_train"])
            mean = model["y_mean"] + model["y_std"] * (K_star @ model["alpha"])
            predictions.append(np.asarray(mean, dtype=np.float64))

            if return_std:
                # var = k(x,x) - k* K^-1 k*^T，RBF 下 k(x,x)=1
                v = np.linalg.solve(model["L"], K_star.T)
                variance = 1.0 - np.sum(v ** 2, axis=0)
                variance = np.maximum(variance, 0.0)
                std = np.sqrt(variance) * model["y_std"]
                stds.append(np.asarray(std, dtype=np.float64))

        mean_all = np.column_stack(predictions)
        if return_std:
            return mean_all, np.column_stack(stds)
        return mean_all
