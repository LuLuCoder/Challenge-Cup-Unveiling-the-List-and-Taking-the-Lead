"""统一主界面：把两个子软件整合进同一个窗口。"""

import importlib
import tkinter as tk
from tkinter import messagebox, ttk

from workbench.ui.embed import EmbeddableRoot


WINDOW_TITLE = "连续碳纤维复合材料 3D 打印智能制造平台"
WINDOW_GEOMETRY = "1680x1000"

# 模块注册表：新增子软件时，在这里追加一项即可。
# (标签页标题, 子应用模块路径, 子应用类名)
MODULES = [
    ("① ANSYS 仿真驱动路径规划", "path_planner.ui.main_window", "ANSYSPathPlannerApp"),
    ("② SATC 参数优化", "satc.ui.main_window", "SATCOptimizerApp"),
    ("③ 相似零件模板映射", "path_planner.ui.template_window", "TemplateApp"),
]


class MainWindow:
    """主窗口：顶部标题栏 + 标签页（每个标签页承载一个子软件）。"""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._apps = []
        self._app_labels = {}
        self._busy_apps = set()
        self.setup_ui()

    def setup_ui(self):
        self._build_header()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        for label, module, cls_name in MODULES:
            self._add_module_tab(label, module, cls_name)

    def _build_header(self):
        header = ttk.Frame(self.root)
        header.pack(fill="x", padx=14, pady=(12, 8))

        ttk.Label(
            header,
            text=WINDOW_TITLE,
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(side="left")

        ttk.Label(
            header,
            text="路径规划 · 参数优化 一体化工作台",
            foreground="#666666",
        ).pack(side="left", padx=14)

        self.header_status = ttk.Label(
            header,
            text="空闲",
            foreground="#2C7FB8",
        )
        self.header_status.pack(side="right", padx=8)

        for index, (label, _module, _cls) in enumerate(MODULES):
            ttk.Button(
                header,
                text=label,
                command=lambda i=index: self.notebook.select(i),
            ).pack(side="right", padx=4)

    def _add_module_tab(self, label, module, cls_name):
        container = EmbeddableRoot(self.notebook)
        self.notebook.add(container, text=label)
        try:
            app_class = self._load_app_class(module, cls_name)
            app = app_class(container)
        except Exception as e:
            messagebox.showerror(
                "模块加载失败",
                f"「{label}」无法启动：\n{repr(e)}",
            )
            return
        self._apps.append(app)
        self._app_labels[app] = label

        setter = getattr(app, "set_task_hook", None)
        if setter is not None:
            setter(self._on_module_busy)

    @staticmethod
    def _load_app_class(module, cls_name):
        mod = importlib.import_module(module)
        return getattr(mod, cls_name)

    def _on_module_busy(self, app, busy):
        """任一模块正在计算时，禁用其他模块的重任务按钮，避免并发挤内存。"""
        if busy:
            self._busy_apps.add(app)
        else:
            self._busy_apps.discard(app)

        any_busy = len(self._busy_apps) > 0
        for other in self._apps:
            if other in self._busy_apps:
                continue
            for attr in (
                "import_button", "plan_button", "run_button",
                "process_button", "save_button", "library_button",
            ):
                btn = getattr(other, attr, None)
                if btn is not None:
                    btn.config(state="disabled" if any_busy else "normal")

        if any_busy:
            names = "、".join(
                self._app_labels[a] for a in self._busy_apps
            )
            self.header_status.config(
                text=f"正在计算：{names}…", foreground="#D62728"
            )
        else:
            self.header_status.config(text="空闲", foreground="#2C7FB8")

    def _on_tab_changed(self, event=None):
        """切走标签页时暂停该页动画，切回时恢复，避免后台持续重绘。"""
        try:
            current = self.notebook.index(self.notebook.select())
        except tk.TclError:
            return

        for index, app in enumerate(self._apps):
            pause = getattr(app, "pause_animation", None)
            resume = getattr(app, "resume_animation", None)
            if pause is None or resume is None:
                continue
            if index == current:
                resume()
            else:
                pause()

    def _on_close(self):
        """关闭主窗口前，先停掉子应用里可能仍在运行的动画定时器。"""
        for app in self._apps:
            animation = getattr(app, "_animation", None)
            if animation is not None:
                try:
                    animation.event_source.stop()
                except Exception:
                    pass
        self.root.destroy()
