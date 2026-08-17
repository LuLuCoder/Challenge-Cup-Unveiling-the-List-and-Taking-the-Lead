"""相似零件模板映射入口：python main_template.py"""

import tkinter as tk

import matplotlib

matplotlib.use("TkAgg")

from path_planner.ui.template_window import TemplateApp


def main():
    root = tk.Tk()
    TemplateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
