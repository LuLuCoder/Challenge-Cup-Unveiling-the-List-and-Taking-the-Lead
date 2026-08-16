"""BLAS 线程配置。

必须在导入 numpy / scipy / sklearn 之前调用 configure_blas_threads()，
避免 Windows 下小样本 GPR 预测偶发阻塞。
"""

import os

BLAS_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def configure_blas_threads():
    """把 BLAS 相关线程数统一设置为 1（已设置的不覆盖）。"""
    for key, value in BLAS_THREAD_ENV.items():
        os.environ.setdefault(key, value)
