"""GUI 前端入口：python app.py（命令行版本仍为 python main.py）。"""

import tkinter as tk

from satc.threads import configure_blas_threads

# 必须在导入 numpy / sklearn 之前调用，避免 Windows BLAS 线程阻塞
configure_blas_threads()

from satc.ui.main_window import SATCOptimizerApp  # noqa: E402


def main():
    root = tk.Tk()
    SATCOptimizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
