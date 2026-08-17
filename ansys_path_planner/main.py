"""程序入口：python main.py 启动图形界面。"""
import tkinter as tk

import matplotlib

# 强制统一 Tk 后端，避免环境默认 qtagg 与 Tk 画布混用导致白屏
matplotlib.use("TkAgg")

from path_planner.ui.main_window import ANSYSPathPlannerApp


def main():
    root = tk.Tk()
    ANSYSPathPlannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
