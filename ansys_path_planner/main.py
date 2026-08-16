"""程序入口：python main.py 启动图形界面。"""
import tkinter as tk

from path_planner.ui.main_window import ANSYSPathPlannerApp


def main():
    root = tk.Tk()
    ANSYSPathPlannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
