"""统一主界面入口：python main.py"""

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _ROOT.parent

for _dir in (
    _ROOT,
    _PROJECT_ROOT / "ansys_path_planner",
    _PROJECT_ROOT / "satc_optimizer",
):
    _path = str(_dir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# BLAS 线程配置必须在导入 numpy / scipy / sklearn 之前执行，
# 否则 Windows 下小样本 GPR 预测偶发阻塞。
# from satc.threads import configure_blas_threads  # noqa: E402

# configure_blas_threads()

import tkinter as tk  # noqa: E402

from workbench.ui.main_window import MainWindow  # noqa: E402


def main():
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
