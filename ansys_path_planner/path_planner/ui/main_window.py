"""ANSYS 仿真驱动路径规划系统主窗口。"""
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from path_planner import config
from path_planner.analysis.path_planning import generate_layer_path
from path_planner.analysis.shape_signature import compute_signature
from path_planner.analysis.stress import merge_ansys_files_data
from path_planner.analysis.template_library import (
    delete_template,
    find_best_template,
    list_templates,
    map_from_template,
    save_template,
)
from path_planner.parsers.auto_classify import classify_ansys_files
from path_planner.parsers.coordinates import parse_coordinate_file
from path_planner.parsers.step_mesh import mesh_step_file
from path_planner.ui.step_inspect_window import show_step_inspection
from path_planner.visualization.plots import attach_colorbar, plot_density_path
from path_planner.visualization.virtual_arm import (
    forward_virtual_arm,
    solve_virtual_arm,
)


def _configure_matplotlib():
    plt.rcParams["font.sans-serif"] = config.MATPLOTLIB_FONTS
    plt.rcParams["axes.unicode_minus"] = False


class ANSYSPathPlannerApp:
    """主窗口：数据导入、路径规划、可视化与保存。"""

    def __init__(self, root):
        _configure_matplotlib()

        self.root = root
        self.root.title(config.APP_TITLE)
        self.root.geometry(config.APP_GEOMETRY)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.selected_ansys_files = []

        self.data = None          # 融合后的仿真数据 DataFrame
        self.path_data = None     # 规划后的路径 DataFrame
        self.result_name = None
        self._cbar_stress = None  # 应力面板 colorbar，刷新前先移除
        self._animation = None    # 机械臂动画对象，需持有引用防止被回收
        self.anim_running = False
        self._anim_paused = False
        self._busy = False
        self.result_queue = queue.Queue()
        self._task_hook = None  # 统一平台注入的忙碌状态回调（可选）
        self._last_real_data = None  # 最近一次真实仿真融合数据（供存模板）
        self._last_step_mesh = None  # 最近一次 STEP 网格结果（供检查窗口）
        self._last_step_entry = None  # 最近一次 STEP 命中的模板条目

        self.setup_ui()
        self._check_gmsh()

    def _check_gmsh(self):
        """启动自检：提示当前 Python 环境是否具备 STEP 网格化能力。"""
        try:
            import gmsh  # noqa: F401

            gmsh_ok = True
        except Exception:
            gmsh_ok = False
        if gmsh_ok:
            self.status_var.set(
                "就绪：已启用 gmsh 网格化，STEP 文件可直接导入。"
            )
        else:
            self.status_var.set(
                "警告：当前 Python 环境未安装 gmsh，STEP 导入将无法网格化；"
                "请用 pytorch 环境启动或执行 pip install gmsh"
            )

    def _on_real_import(self, merged, skipped):
        """真实仿真数据导入完成后的扩展钩子（子类可覆写）。"""

    def set_task_hook(self, callback):
        """供统一平台注入忙碌状态回调：callback(app, busy)。"""
        self._task_hook = callback

    # --------------------------------------------------------
    # UI 搭建
    # --------------------------------------------------------

    def setup_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=10)

        ttk.Label(
            top,
            text="ANSYS 仿真驱动三维路径规划",
            font=("Microsoft YaHei", 18, "bold"),
        ).pack(side="left")

        self._build_file_section()
        self._build_param_section()
        self._build_status_bar()
        self._build_plot_area()
        self._build_table()

    def _build_file_section(self):
        file_frame = ttk.LabelFrame(self.root, text="① ANSYS 数据导入")
        file_frame.pack(fill="x", padx=12, pady=5)

        ttk.Label(file_frame, text="ANSYS 数据文件：").grid(
            row=0, column=0, padx=8, pady=8
        )
        self.ansys_files_var = tk.StringVar()
        ttk.Entry(
            file_frame, textvariable=self.ansys_files_var, width=85
        ).grid(row=0, column=1, padx=5)
        ttk.Button(
            file_frame, text="选择文件", command=self.select_ansys_files
        ).grid(row=0, column=2, padx=5)
        self.import_button = ttk.Button(
            file_frame, text="自动导入并计算",
            command=self.import_and_merge,
        )
        self.import_button.grid(row=0, column=3, padx=15)
        self.inspect_button = ttk.Button(
            file_frame, text="STEP 检查",
            command=self._open_step_inspection, state="disabled",
        )
        self.inspect_button.grid(row=0, column=4, padx=5)

        ttk.Label(file_frame, text="相似度阈值：").grid(
            row=1, column=0, padx=8, pady=8
        )
        self.similarity_var = tk.DoubleVar(
            value=config.DEFAULT_SIMILARITY_THRESHOLD
        )
        ttk.Spinbox(
            file_frame,
            from_=config.SIMILARITY_MIN,
            to=config.SIMILARITY_MAX,
            increment=0.01,
            textvariable=self.similarity_var,
            width=6,
        ).grid(row=1, column=1, sticky="w", padx=5)
        self.save_template_button = ttk.Button(
            file_frame, text="存入模板库", command=self.save_as_template
        )
        self.save_template_button.grid(row=1, column=2, padx=5)
        self.template_lib_button = ttk.Button(
            file_frame, text="模板库", command=self.open_template_library
        )
        self.template_lib_button.grid(row=1, column=3, padx=5)

        ttk.Label(
            file_frame,
            text=(
                "一次多选：节点坐标文件 + X/Y/Z/XY/YZ/XZ 六个应力分量文件"
                "（Ctrl/Shift 多选，自动按文件名与内容识别）"
            ),
            foreground="#666666",
        ).grid(row=2, column=1, sticky="w", padx=5, pady=(0, 4))
        ttk.Label(
            file_frame,
            text=(
                "只选节点坐标文件时：自动与模板库比对，相似度 ≥ 阈值"
                "则直接映射模板路径，无需重新仿真"
            ),
            foreground="#666666",
        ).grid(row=3, column=1, sticky="w", padx=5, pady=(0, 7))
        ttk.Label(
            file_frame,
            text=(
                "也可直接导入 STEP(.step/.stp) 文件：自动转换为点云"
                "并与模板库对比，无需 ANSYS 导出 CSV"
            ),
            foreground="#2C7FB8",
        ).grid(row=4, column=1, sticky="w", padx=5, pady=(0, 7))

    def _build_param_section(self):
        param_frame = ttk.LabelFrame(self.root, text="② 路径规划参数")
        param_frame.pack(fill="x", padx=12, pady=5)

        ttk.Label(param_frame, text="高优先级分位数：").pack(
            side="left", padx=8
        )
        self.percentile_var = tk.DoubleVar(value=config.DEFAULT_PERCENTILE)
        ttk.Spinbox(
            param_frame,
            from_=config.PERCENTILE_MIN,
            to=config.PERCENTILE_MAX,
            increment=1,
            textvariable=self.percentile_var,
            width=8,
        ).pack(side="left")

        ttk.Label(param_frame, text="分层数：").pack(
            side="left", padx=(15, 4)
        )
        self.layers_var = tk.IntVar(value=config.DEFAULT_LAYERS)
        ttk.Spinbox(
            param_frame,
            from_=config.LAYER_MIN,
            to=config.LAYER_MAX,
            increment=1,
            textvariable=self.layers_var,
            width=6,
        ).pack(side="left")

        self.plan_button = ttk.Button(
            param_frame, text="生成路径", command=self.plan_path
        )
        self.plan_button.pack(side="left", padx=15)

        self.animation_button = ttk.Button(
            param_frame, text="▶ 播放/暂停演示",
            command=self.toggle_animation,
        )
        self.animation_button.pack(side="left", padx=(0, 5))

        ttk.Label(
            param_frame,
            text="（低应力区域不删除，仅降低路径优先级）",
            foreground="#666666",
        ).pack(side="left", padx=8)

        self.save_merged_button = ttk.Button(
            param_frame, text="保存融合 CSV", command=self.save_merged
        )
        self.save_merged_button.pack(side="left", padx=5)
        self.save_path_button = ttk.Button(
            param_frame, text="保存路径 CSV", command=self.save_path
        )
        self.save_path_button.pack(side="left", padx=5)

    def _build_status_bar(self):
        self.status_var = tk.StringVar(value="等待导入 ANSYS 数据")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 11),
        ).pack(anchor="w", padx=15, pady=5)

    def _set_busy(self, busy, text=None):
        """进入/退出忙碌状态：禁用操作按钮，避免重复触发任务。"""
        self._busy = busy
        if text is not None:
            self.status_var.set(text)
        state = "disabled" if busy else "normal"
        for btn in (
            getattr(self, "import_button", None),
            getattr(self, "plan_button", None),
            getattr(self, "save_merged_button", None),
            getattr(self, "save_path_button", None),
            getattr(self, "animation_button", None),
            getattr(self, "save_template_button", None),
            getattr(self, "template_lib_button", None),
        ):
            if btn is not None:
                btn.config(state=state)
        hook = self._task_hook
        if hook is not None:
            try:
                hook(self, busy)
            except Exception:
                pass
        self.root.update_idletasks()

    def _build_plot_area(self):
        # 外层滚动容器：图比可视区域大时，用滑块滚动查看完整图形
        scroll_frame = ttk.Frame(self.root)
        scroll_frame.pack(fill="both", expand=True, padx=12, pady=5)

        self.plot_container = tk.Canvas(
            scroll_frame, highlightthickness=0
        )
        vbar = ttk.Scrollbar(
            scroll_frame, orient="vertical",
            command=self.plot_container.yview,
        )
        hbar = ttk.Scrollbar(
            scroll_frame, orient="horizontal",
            command=self.plot_container.xview,
        )
        self.plot_container.configure(
            xscrollcommand=hbar.set, yscrollcommand=vbar.set
        )

        hbar.pack(side="bottom", fill="x")
        vbar.pack(side="right", fill="y")
        self.plot_container.pack(side="left", fill="both", expand=True)

        # 四个独立画布：几何/应力/路径只在数据更新时绘制一次，
        # 机械臂画布单独做实时动画，避免每帧重绘整张四联图
        inner = ttk.Frame(self.plot_container)
        self.plot_container.create_window(
            (0, 0), window=inner, anchor="nw"
        )
        self.plot_container.bind(
            "<Configure>", self._update_scroll_region
        )

        self._make_3d_canvas(inner, "geom")
        self._make_3d_canvas(inner, "stress")
        self._make_3d_canvas(inner, "path")
        self._make_3d_canvas(inner, "robot")

        # 路径图右侧预留图例空间
        self.fig_path.subplots_adjust(
            left=0.02, right=0.78, top=0.96, bottom=0.02
        )

        self.canvas_geom.get_tk_widget().grid(
            row=0, column=0, padx=4, pady=4
        )
        self.canvas_stress.get_tk_widget().grid(
            row=0, column=1, padx=4, pady=4
        )
        self.canvas_path.get_tk_widget().grid(
            row=1, column=0, padx=4, pady=4
        )
        self.canvas_robot.get_tk_widget().grid(
            row=1, column=1, padx=4, pady=4
        )

        # 双击任意图 -> 弹出放大窗口
        for _name in ("geom", "stress", "path", "robot"):
            widget = getattr(self, f"canvas_{_name}").get_tk_widget()
            widget.bind(
                "<Double-Button-1>",
                lambda e, n=_name: self._open_zoom(n),
            )

    def _make_3d_canvas(self, master, name):
        """创建单个 3D 视图画布，并把 fig/ax/canvas 存为属性。"""
        fig = plt.Figure(figsize=(6.2, 4.4), dpi=100)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.96, bottom=0.02)
        ax = fig.add_subplot(111, projection="3d")
        canvas = FigureCanvasTkAgg(fig, master=master)
        setattr(self, f"fig_{name}", fig)
        setattr(self, f"ax_{name}", ax)
        setattr(self, f"canvas_{name}", canvas)
        return canvas

    def _build_table(self):
        table_frame = ttk.LabelFrame(self.root, text="③ 路径溯源")
        table_frame.pack(fill="x", padx=12, pady=8)

        columns = (
            "Step", "Layer", "Node", "X", "Y", "Z",
            "SimulationValue", "Priority", "PathWeight",
        )
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=4
        )
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=130)
        self.tree.pack(side="left", fill="x", expand=True)

        scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.tree.yview
        )
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<Double-1>", self.on_table_click)

    # --------------------------------------------------------
    # 文件选择
    # --------------------------------------------------------

    def select_ansys_files(self):
        paths = filedialog.askopenfilenames(
            title=(
                "选择 ANSYS 数据文件（可按住 Ctrl/Shift 多选）："
                "节点坐标 + X/Y/Z/XY/YZ/XZ 六个应力分量文件"
            ),
            filetypes=[
                ("CSV/TXT", "*.csv *.txt"),
                ("STEP", "*.step *.stp"),
                ("CSV", "*.csv"),
                ("TXT", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self.selected_ansys_files = list(paths)
            self.ansys_files_var.set("；".join(paths))

    # --------------------------------------------------------
    # 数据导入与融合
    # --------------------------------------------------------

    def import_and_merge(self):
        """后台线程导入：完整 ANSYS 数据走真实仿真；仅节点坐标走模板映射。

        解析/融合/主应力计算都在后台线程执行（均为向量化或 C 实现，
        不长时间占用 GIL），界面保持可操作；统一平台会在此期间禁止
        另一模块同时启动重任务。
        """
        paths = getattr(self, "selected_ansys_files", None) or []
        if not paths:
            messagebox.showerror("导入失败", "请先选择 ANSYS 数据文件。")
            return
        if self._busy:
            messagebox.showinfo("提示", "正在处理中，请稍候……")
            return

        self._set_busy(True, "正在导入并计算（后台执行，界面可继续操作）……")
        self.root.update()
        threshold = float(self.similarity_var.get())

        def worker():
            try:
                step_paths = [
                    p for p in paths
                    if str(p).lower().endswith((".step", ".stp"))
                ]
                if step_paths:
                    # STEP 直接导入：gmsh 四面体网格 -> 节点点云 -> 与模板对比 -> 映射
                    mesh = mesh_step_file(
                        step_paths[0],
                        progress_cb=lambda m: self.result_queue.put(
                            ("step_progress", m)
                        ),
                    )
                    self.result_queue.put(("step", mesh, threshold))
                else:
                    node_path, stress_files, skipped = classify_ansys_files(
                        paths, require_stress=False
                    )
                    if len(stress_files) == len(config.STRESS_COMPONENTS):
                        merged = merge_ansys_files_data(node_path, stress_files)
                        self.result_queue.put(("real", merged, skipped, None))
                    else:
                        node_df = parse_coordinate_file(node_path)
                        self._queue_template_mapping(node_df, skipped, threshold)
            except Exception as exc:
                self.result_queue.put(("error", repr(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_import)

    def _queue_template_mapping(self, node_df, skipped, threshold,
                                entry=None, sim=None):
        """后台线程内执行模板检索与映射，结果放入队列。

        entry/sim 可预传入（STEP 流程已先检索以便展示检查窗口），
        否则在此自动检索。
        """
        def worker():
            try:
                signature = compute_signature(
                    node_df[["X", "Y", "Z"]].to_numpy(float)
                )
                e, s = entry, sim
                if e is None or s is None:
                    # threshold=0 使 find_best_template 始终返回最相似模板
                    e, s = find_best_template(signature, threshold=0.0)
                if e is None or s < threshold:
                    self.result_queue.put(
                        ("no_match", node_df, s, threshold, skipped)
                    )
                    return
                mapped_geom, _ = map_from_template(e["path"], node_df)
                n_layers = int(
                    e.get("n_layers") or config.DEFAULT_LAYERS
                )
                path_data, _ = generate_layer_path(
                    mapped_geom,
                    "Maximum_Principal",
                    percentile=config.DEFAULT_PERCENTILE,
                    n_layers=n_layers,
                )
                self.result_queue.put(
                    ("mapped", mapped_geom, path_data, (e, s, skipped))
                )
            except Exception as exc:
                self.result_queue.put(("error", repr(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_import(self):
        """主线程轮询导入结果，成功后再更新数据与绘图。"""
        try:
            item = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_import)
            return
        status = item[0]
        payload = list(item[1:])

        self._set_busy(False)
        if status == "error":
            messagebox.showerror("导入失败", payload[0])
            self.status_var.set("导入失败")
            return
        if status == "step_progress":
            self.status_var.set(payload[0])
            self.root.after(100, self._poll_import)
            return
        if status == "step":
            mesh, threshold = payload
            if not mesh["ok"]:
                self._set_busy(False)
                reason = mesh.get("error") or "未知原因"
                messagebox.showerror(
                    "STEP 网格化失败",
                    "无法用 gmsh 对 STEP 文件划分四面体网格，已停止导入"
                    "（避免使用不准确的点云）。\n\n"
                    f"原因：{reason}\n\n"
                    "请确认：\n"
                    "1. 使用已安装 gmsh 的 Python 环境启动本软件\n"
                    "   （pytorch 环境：C:\\Users\\29384\\.conda\\envs\\"
                    "pytorch\\python.exe）；\n"
                    "2. 若缺失，执行：pip install gmsh",
                )
                self.status_var.set("STEP 网格化失败（gmsh 不可用）")
                return
            self._set_busy(
                True, "网格划分完成，正在与模板库比对（后台执行）…"
            )
            node_df = mesh["node_df"]

            def match_worker():
                try:
                    signature = compute_signature(mesh["points"])
                    entry, sim = find_best_template(
                        signature, threshold=0.0
                    )
                    self.result_queue.put(
                        ("step_ready", mesh, node_df, entry, sim, threshold)
                    )
                except Exception as e:
                    self.result_queue.put(("error", repr(e)))

            threading.Thread(target=match_worker, daemon=True).start()
            self.root.after(100, self._poll_import)
            return
        if status == "step_ready":
            mesh, node_df, entry, sim, threshold = payload
            self._set_busy(False)
            self._last_step_mesh = mesh
            self._last_step_entry = entry
            if hasattr(self, "inspect_button"):
                self.inspect_button.configure(state="normal")
            # 检查窗口：模板 / 线框 / 四面体网格 / 点云 + 质量报告
            try:
                show_step_inspection(self.root, mesh, entry)
            except Exception as exc:  # noqa: BLE001 - 展示失败不应阻断路径映射
                messagebox.showwarning(
                    "检查窗口显示失败", str(exc)
                )
            self._set_busy(True, "模板比对完成，正在映射应力场与路径…")
            self._queue_template_mapping(
                node_df, [], threshold, entry=entry, sim=sim
            )
            self.root.after(100, self._poll_import)
            return

        if status == "no_match":
            node_df, sim, threshold, _skipped = payload
            # 未命中模板也显示零件点云（例如来自 STEP 的几何）
            self.data = node_df
            self.path_data = None
            self.result_name = "Maximum_Principal"
            self._last_real_data = None
            self.status_var.set(
                f"未命中模板：最高相似度 {sim:.3f} < 阈值 {threshold:.2f}；"
                "已显示零件点云。请先建立匹配模板或调低相似度阈值。"
            )
            self.plot_geometry()
            messagebox.showinfo(
                "未命中模板",
                f"模板库中最高相似度 = {sim:.3f}，低于阈值 {threshold:.2f}。\n\n"
                "已显示该零件的点云（来自 STEP/节点文件）。\n"
                "可选：调低相似度阈值后重试；或先导入真实仿真数据"
                "建立与该零件匹配的模板。",
            )
            return
        if status == "real":
            merged, skipped, _ = payload
            self.data = merged
            self.path_data = None
            self.result_name = "Maximum_Principal"
            self._last_real_data = merged.copy()
            note = (
                f"；已忽略 {len(skipped)} 个辅助文件（如变形结果）"
                if skipped else ""
            )
            self.status_var.set(
                f"导入成功：{len(merged)} 个匹配节点；自动识别 "
                "1 个坐标文件 + 6 个应力分量文件，"
                "已完成应力张量、主应力和主方向计算。"
                + note
            )
            self.plot_geometry()
            self.plot_simulation()
            self._on_real_import(merged, skipped)
            return
        if status == "mapped":
            mapped_geom, path_data, (entry, sim, skipped) = payload
            self.data = mapped_geom
            self.path_data = path_data
            self.result_name = "Maximum_Principal"
            self._last_real_data = None
            note = (
                f"；已忽略 {len(skipped)} 个辅助文件" if skipped else ""
            )
            self.status_var.set(
                f"命中模板「{entry['name']}」（相似度 {sim:.2f}）："
                f"已通过模板映射应力场并按层规划路径"
                f"（{len(path_data)} 个路径点，共 "
                f"{int(path_data['Layer'].max())} 层），"
                "未重新进行 ANSYS 仿真。" + note
            )
            self.plot_geometry()
            self.plot_simulation()
            self.plot_path()
            self.update_table()
            return

    def _open_step_inspection(self):
        """重新打开最近一次 STEP 的网格检查窗口（想看就看）。"""
        if self._last_step_mesh is None:
            messagebox.showinfo(
                "提示", "请先导入 STEP 文件并完成网格化。"
            )
            return
        try:
            show_step_inspection(
                self.root, self._last_step_mesh, self._last_step_entry
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showwarning("检查窗口显示失败", str(exc))

    # --------------------------------------------------------
    # 模板库管理
    # --------------------------------------------------------

    def save_as_template(self):
        """把最近一次真实仿真 + 规划路径存入模板库。"""
        if self._last_real_data is None:
            messagebox.showinfo(
                "提示",
                "请先导入真实仿真数据（节点坐标 + 六应力分量）。",
            )
            return
        if self.path_data is None:
            messagebox.showinfo(
                "提示", "请先生成路径，再存入模板库。"
            )
            return

        name = simpledialog.askstring(
            "存入模板库", "模板名称：", parent=self.root
        )
        if not name:
            return
        try:
            folder = save_template(
                self._last_real_data, self.path_data, name
            )
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
            return
        messagebox.showinfo("已存入模板库", f"模板已保存：\n{folder}")

    def open_template_library(self):
        """弹出模板库窗口：查看 / 删除模板。"""
        entries = list_templates()
        if not entries:
            messagebox.showinfo(
                "模板库为空",
                "暂无模板。请先导入真实仿真数据并生成路径，"
                "再点击「存入模板库」。",
            )
            return

        win = tk.Toplevel(self.root)
        win.title("模板库")
        win.geometry("680x360")
        tree = ttk.Treeview(
            win,
            columns=("name", "nodes", "size", "path", "time"),
            show="headings",
            height=12,
        )
        for col, text, width in (
            ("name", "名称", 180),
            ("nodes", "节点数", 90),
            ("size", "尺寸X×Y×Z(mm)", 160),
            ("path", "路径点数", 90),
            ("time", "创建时间", 150),
        ):
            tree.heading(col, text=text)
            tree.column(col, width=width)
        tree.pack(fill="both", expand=True, padx=8, pady=8)

        entry_map = {}
        for e in entries:
            iid = tree.insert(
                "",
                "end",
                values=(
                    e["name"],
                    e["node_count"],
                    " × ".join(str(v) for v in e["size_mm"]),
                    e["path_points"],
                    e["created_at"],
                ),
            )
            entry_map[iid] = e

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(
            btns,
            text="删除选中",
            command=lambda: self._delete_selected_template(
                tree, entry_map, win
            ),
        ).pack(side="left")
        ttk.Button(btns, text="关闭", command=win.destroy).pack(side="right")

    def _delete_selected_template(self, tree, entry_map, win):
        sel = tree.selection()
        if not sel:
            return
        entry = entry_map[sel[0]]
        if not messagebox.askyesno(
            "确认删除", f"删除模板「{entry['name']}」？"
        ):
            return
        try:
            delete_template(entry["path"])
        except Exception as e:
            messagebox.showerror("删除失败", str(e))
            return
        win.destroy()
        messagebox.showinfo("已删除", "模板已删除。")

    # --------------------------------------------------------
    # 仿真结果可视化
    # --------------------------------------------------------

    @staticmethod
    def _plot_sample_indices(n):
        """节点云绘图降采样：超过上限时均匀随机抽 PLOT_MAX_POINTS 个点。"""
        if n <= config.PLOT_MAX_POINTS:
            return np.arange(n)
        rng = np.random.default_rng(0)
        return np.sort(
            rng.choice(n, size=config.PLOT_MAX_POINTS, replace=False)
        )

    def plot_geometry(self):
        """左上视图：实际模型几何形状（节点云）。"""
        if self.data is None:
            return

        self.ax_geom.clear()
        self._draw_geometry(self.ax_geom, self.data)
        self.canvas_geom.draw()

    def _draw_geometry(self, ax, data):
        """把几何节点云画到指定 3D 坐标轴（主视图/放大窗口共用）。"""
        d = data.iloc[self._plot_sample_indices(len(data))]
        ax.scatter(
            d["X"], d["Y"], d["Z"],
            c="#4C78A8", s=4, alpha=0.85, depthshade=True,
        )
        ax.set_title("模型几何形状（节点云）", fontsize=13)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    def plot_simulation(self):
        """右上视图：应力分析结果。"""
        if self.data is None:
            return

        self.ax_stress.clear()
        sc = self._draw_stress(self.ax_stress, self.fig_stress, self.data)
        self._cbar_stress = attach_colorbar(
            self.fig_stress, self.ax_stress, sc,
            shrink=0.65, pad=0.08, current_cbar=self._cbar_stress,
        )
        self.canvas_stress.draw()

    def _draw_stress(self, ax, fig, data):
        """把应力场画到指定 3D 坐标轴，返回 scatter（供加 colorbar）。"""
        d = data.iloc[self._plot_sample_indices(len(data))]
        vmin, vmax = self._value_range(d)
        sc = ax.scatter(
            d["X"], d["Y"], d["Z"],
            c=d[self.result_name], cmap="turbo",
            vmin=vmin, vmax=vmax, s=10, alpha=0.85, depthshade=True,
        )
        ax.set_title("应力分析结果（最大主应力）", fontsize=13)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return sc

    # --------------------------------------------------------
    # 路径规划
    # --------------------------------------------------------

    def plan_path(self):
        """后台线程执行路径规划，界面保持可操作。"""
        if self.data is None:
            messagebox.showerror(
                "路径规划失败", "请先导入 ANSYS 节点坐标和仿真结果。"
            )
            return
        if self._busy:
            messagebox.showinfo("提示", "正在处理中，请稍候……")
            return

        try:
            percentile = float(self.percentile_var.get())
            n_layers = int(self.layers_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("路径规划失败", "分位数/分层数必须是数字。")
            return

        self._set_busy(True, "正在生成路径（后台执行，界面可继续操作）……")
        self.root.update()
        data = self.data  # 工作线程只读引用，主线程期间不修改

        def worker():
            try:
                path_data, threshold = generate_layer_path(
                    data,
                    "Maximum_Principal",
                    percentile=percentile,
                    n_layers=n_layers,
                )
                self.result_queue.put(("ok", path_data, threshold))
            except Exception as e:
                self.result_queue.put(("error", repr(e)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_plan)

    def _poll_plan(self):
        """主线程轮询路径规划结果，成功后再绘图。"""
        try:
            item = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_plan)
            return
        status = item[0]
        payload = list(item[1:])

        self._set_busy(False)
        if status != "ok":
            messagebox.showerror("路径规划失败", payload[0])
            return

        path_data, threshold = payload
        self.path_data = path_data
        self.plot_path()
        self.update_table()
        self.status_var.set(
            f"路径生成完成：{len(self.path_data)} 个路径点；"
            f"共 {int(self.path_data['Layer'].max())} 层；"
            "全部有效节点均保留；"
            f"高优先级阈值 = {threshold:.6g}"
        )

    # --------------------------------------------------------
    # 路径可视化
    # --------------------------------------------------------

    def plot_path(self):
        """左下视图：规划路径（仅路径线，不含节点散点）。"""
        if self.data is None or self.path_data is None:
            return

        self.ax_path.clear()
        self.ax_robot.clear()
        self._draw_path(self.ax_path, self.path_data, max_segments=25000)

        # ---- 右图：虚拟机械臂 ----
        self._plot_virtual_arm(self.path_data)

        self.canvas_path.draw()
        self.canvas_robot.draw()

    def _draw_path(self, ax, path_data, max_segments=None):
        """把规划路径画到指定 3D 坐标轴（主视图/放大窗口共用）。"""
        p = path_data
        xyz = p[["X", "Y", "Z"]].to_numpy(float)
        # 穿越空区（无节点）的段不绘制，避免路径穿过零件孔洞/空隙
        seg_types = p["Segment_Type"].to_numpy()
        skip_mask = seg_types[1:] == "空区断开"
        plot_density_path(
            ax, xyz, p["Path_Spacing"].to_numpy(float),
            linewidth=1.6, alpha=0.9,
            attach_labels=True, with_legend=False,
            skip_mask=skip_mask, max_segments=max_segments,
        )
        ax.set_title("规划路径（密度着色）", fontsize=13)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 小图例放图外右侧，不遮挡路径
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            frameon=True,
            framealpha=0.7,
            handlelength=1.2,
            borderpad=0.4,
            labelspacing=0.4,
        )

    def _plot_virtual_arm(self, path_df):
        """把路径压缩到虚拟机械臂工作空间，并让机械臂沿路径运动。"""
        # 把路径缩放并平移到零件位置，与机械臂错开：
        # 1) 以路径中心为基准等比缩放，使半径 <= 0.16（零件放大）；
        # 2) 平移到 (0.46, 0, 0.16)，最低层 z=0 与基座平面重合；
        # 3) 机械臂基座在原点，零件在右侧，两者不重叠。
        # 6 自由度关节臂（大臂 0.40 + 小臂 0.36 + 腕/工具），末端贴合路径。
        seg_types = path_df["Segment_Type"].to_numpy()
        normalized, path_center, scale = self._robot_normalized(path_df)

        # 零件节点云（浅色小点），帮助看清机械臂与零件的相对位置
        if self.data is not None:
            d = self.data.iloc[self._plot_sample_indices(len(self.data))]
            pts = d[["X", "Y", "Z"]].to_numpy(float)
            cloud = (pts - path_center) * scale + np.array([0.46, 0.0, 0.16])
            if len(cloud) > 3000:
                rng = np.random.default_rng(1)
                cloud = cloud[rng.choice(len(cloud), 3000, replace=False)]
            self.ax_robot.scatter(
                cloud[:, 0], cloud[:, 1], cloud[:, 2],
                c="#B0BEC5", s=2, alpha=0.35, depthshade=False,
            )

        # 完整轨迹（虚线）：按子路径分段绘制，跳过穿越空区的段
        dashed_segments = self._robot_dashed(normalized, seg_types)
        if dashed_segments:
            # LineCollection 要求所有线段点数一致：
            # 把每条折线拆成等长的“两点段”再批量绘制
            # 同时按总点数降采样，避免动画每帧重画十几万段轨迹
            total = sum(len(poly) for poly in dashed_segments)
            step = max(1, total // 6000)
            flat_segments = []
            for poly in dashed_segments:
                flat_segments.extend(
                    poly[j:j + 2]
                    for j in range(0, len(poly) - 1, step)
                )
            dashed = Line3DCollection(
                flat_segments,
                linestyles="--",
                linewidths=1.5,
                colors=["#999999"],
                alpha=0.7,
            )
            self.ax_robot.add_collection3d(dashed)

        # 已走轨迹高亮 / 机械臂本体 / 末端点（动画中逐帧更新）
        self._trail_line, = self.ax_robot.plot(
            [], [], [], color="#E45756", linewidth=2.0, alpha=0.95,
        )
        self._arm_line, = self.ax_robot.plot(
            [], [], [], marker="o", markersize=7,
            linewidth=5, color="#4C78A8",
        )
        self._tip_marker, = self.ax_robot.plot(
            [], [], [], marker="o", markersize=10, color="#E45756",
        )

        # 限制动画帧数，保证演示时长合适（默认约 400 帧）
        max_frames = 400
        step = max(1, int(np.ceil(len(normalized) / max_frames)))
        self._anim_points = normalized[::step]

        self.ax_robot.set_xlim(-0.95, 0.95)
        self.ax_robot.set_ylim(-0.95, 0.95)
        self.ax_robot.set_zlim(0, 1.1)
        self.ax_robot.set_title("虚拟机械臂运动轨迹", fontsize=13)
        self.ax_robot.set_xlabel("X")
        self.ax_robot.set_ylabel("Y")
        self.ax_robot.set_zlabel("Z")

        self._start_arm_animation()

    def _robot_normalized(self, path_df):
        """把路径压缩到机械臂工作空间，返回 (归一化坐标, 路径中心, 缩放比)。"""
        xyz = path_df[["X", "Y", "Z"]].to_numpy(float)
        path_center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
        path_radius = float(np.max(np.linalg.norm(xyz - path_center, axis=1)))
        part_radius = 0.16
        scale = part_radius / max(path_radius, 1e-12)
        normalized = (xyz - path_center) * scale
        return (
            normalized + np.array([0.46, 0.0, 0.16]),
            path_center,
            scale,
        )

    def _robot_dashed(self, normalized, seg_types):
        """把归一化路径按子路径切成折线列表（跳过空区段）。"""
        skip = seg_types[1:] == "空区断开"
        dashed = []
        run_start = 0
        for i in range(len(normalized) - 1):
            if skip[i]:
                if i > run_start:
                    dashed.append(normalized[run_start:i + 1])
                run_start = i + 1
        if run_start < len(normalized):
            dashed.append(normalized[run_start:])
        return dashed

    def _open_robot_zoom(self):
        """双击机械臂图：左=机械臂动画（无零件），右=零件+路径完成进度。"""
        if self.path_data is None:
            messagebox.showinfo("提示", "请先生成路径。")
            return

        win = tk.Toplevel(self.root)
        win.title("机械臂动画 · 放大视图（左：机械臂 / 右：零件进度）")
        win.geometry("1280x680")
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_zoom(win))

        fig = plt.Figure(figsize=(12.5, 6.4), dpi=100)
        fig.subplots_adjust(
            left=0.03, right=0.99, top=0.96, bottom=0.03, wspace=0.25
        )
        ax_arm = fig.add_subplot(121, projection="3d")
        ax_part = fig.add_subplot(122, projection="3d")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        path_df = self.path_data
        seg_types = path_df["Segment_Type"].to_numpy()
        normalized, path_center, scale = self._robot_normalized(path_df)

        # ---- 左：机械臂（无零件），虚线为运动轨迹 ----
        dashed = self._robot_dashed(normalized, seg_types)
        if dashed:
            total = sum(len(poly) for poly in dashed)
            step_d = max(1, total // 6000)
            flat = []
            for poly in dashed:
                flat.extend(
                    poly[j:j + 2]
                    for j in range(0, len(poly) - 1, step_d)
                )
            if flat:
                ax_arm.add_collection3d(Line3DCollection(
                    flat, linestyles="--", linewidths=1.2,
                    colors=["#999999"], alpha=0.6,
                ))
        arm_line, = ax_arm.plot(
            [], [], [], marker="o", markersize=7,
            linewidth=5, color="#4C78A8",
        )
        tip_marker, = ax_arm.plot(
            [], [], [], marker="o", markersize=10, color="#E45756",
        )
        ax_arm.set_xlim(-0.95, 0.95)
        ax_arm.set_ylim(-0.95, 0.95)
        ax_arm.set_zlim(0, 1.1)
        ax_arm.set_title("机械臂沿路径运动", fontsize=12)
        ax_arm.set_xlabel("X")
        ax_arm.set_ylabel("Y")
        ax_arm.set_zlabel("Z")

        # ---- 右：零件放大 + 路径完成进度 ----
        if self.data is not None:
            d = self.data.iloc[self._plot_sample_indices(len(self.data))]
            pts = d[["X", "Y", "Z"]].to_numpy(float)
            cloud = (pts - path_center) * scale + np.array([0.46, 0.0, 0.16])
            if len(cloud) > 4000:
                rng = np.random.default_rng(2)
                cloud = cloud[rng.choice(len(cloud), 4000, replace=False)]
            ax_part.scatter(
                cloud[:, 0], cloud[:, 1], cloud[:, 2],
                c="#B0BEC5", s=3, alpha=0.5, depthshade=False,
            )

        max_frames = 400
        step_pts = max(1, int(np.ceil(len(normalized) / max_frames)))
        anim_pts = normalized[::step_pts]

        full_line, = ax_part.plot(
            anim_pts[:, 0], anim_pts[:, 1], anim_pts[:, 2],
            color="#90A4AE", linewidth=1.0, alpha=0.7,
        )
        done_line, = ax_part.plot(
            [], [], [], color="#E45756", linewidth=2.8, alpha=0.95,
        )
        ax_part.set_xlim(0.24, 0.70)
        ax_part.set_ylim(-0.25, 0.25)
        ax_part.set_zlim(-0.02, 0.40)
        ax_part.set_title("零件与路径完成进度", fontsize=12)
        ax_part.set_xlabel("X")
        ax_part.set_ylabel("Y")
        ax_part.set_zlabel("Z")

        def update(frame):
            target = anim_pts[frame]
            q = solve_virtual_arm(target)
            arm = forward_virtual_arm(q)
            arm_line.set_data(arm[:, 0], arm[:, 1])
            arm_line.set_3d_properties(arm[:, 2])
            tip_marker.set_data([target[0]], [target[1]])
            tip_marker.set_3d_properties([target[2]])
            done = anim_pts[:frame + 1]
            done_line.set_data(done[:, 0], done[:, 1])
            done_line.set_3d_properties(done[:, 2])
            return arm_line, tip_marker, done_line

        anim = FuncAnimation(
            fig, update, frames=len(anim_pts),
            interval=100, blit=False, repeat=True,
        )
        win._zoom_anim = anim
        canvas.draw()

    def _close_zoom(self, win):
        """关闭放大窗口前停止其动画定时器。"""
        anim = getattr(win, "_zoom_anim", None)
        if anim is not None:
            try:
                anim.event_source.stop()
            except Exception:
                pass
        win.destroy()

    def _open_zoom(self, name):
        """双击图后弹出放大窗口。"""
        if name == "robot":
            self._open_robot_zoom()
            return

        win = tk.Toplevel(self.root)
        win.title(f"{name} · 放大视图")
        win.geometry("980x740")
        fig = plt.Figure(figsize=(9.4, 6.8), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        try:
            if name == "geom":
                if self.data is None:
                    raise ValueError("暂无数据，请先导入。")
                self._draw_geometry(ax, self.data)
            elif name == "stress":
                if self.data is None:
                    raise ValueError("暂无数据，请先导入。")
                sc = self._draw_stress(ax, fig, self.data)
                attach_colorbar(fig, ax, sc, shrink=0.7, pad=0.08)
            elif name == "path":
                if self.path_data is None:
                    raise ValueError("请先生成路径。")
                fig.subplots_adjust(
                    left=0.03, right=0.78, top=0.96, bottom=0.03
                )
                self._draw_path(ax, self.path_data, max_segments=60000)
        except ValueError as e:
            messagebox.showinfo("提示", str(e), parent=win)
            win.destroy()
            return

        canvas.draw()

    def _start_arm_animation(self):
        """创建并启动机械臂沿路径运动的动画。"""
        if getattr(self, "_animation", None) is not None:
            self._animation.event_source.stop()

        pts = self._anim_points

        def update(frame):
            target = pts[frame]
            q = solve_virtual_arm(target)
            arm = forward_virtual_arm(q)

            self._arm_line.set_data(arm[:, 0], arm[:, 1])
            self._arm_line.set_3d_properties(arm[:, 2])

            self._tip_marker.set_data([target[0]], [target[1]])
            self._tip_marker.set_3d_properties([target[2]])

            self._trail_line.set_data(
                pts[:frame + 1, 0], pts[:frame + 1, 1]
            )
            self._trail_line.set_3d_properties(pts[:frame + 1, 2])

            return self._arm_line, self._tip_marker, self._trail_line

        self._animation = FuncAnimation(
            self.fig_robot, update, frames=len(pts),
            interval=100, blit=False, repeat=True,
        )
        self.anim_running = True
        self._anim_paused = False
        self.canvas_robot.draw_idle()

    def toggle_animation(self):
        """播放/暂停机械臂演示。"""
        if getattr(self, "_animation", None) is None:
            messagebox.showinfo("提示", "请先生成路径。")
            return

        if self.anim_running:
            self._animation.event_source.stop()
            self.anim_running = False
            self._anim_paused = False
        else:
            self._animation.event_source.start()
            self.anim_running = True
            self._anim_paused = False

    def pause_animation(self):
        """暂停机械臂动画（供统一平台切换标签页时调用）。"""
        if (
            getattr(self, "_animation", None) is not None
            and self.anim_running
        ):
            self._animation.event_source.stop()
            self.anim_running = False
            self._anim_paused = True

    def resume_animation(self):
        """恢复机械臂动画（仅当切走前正在播放）。"""
        if (
            getattr(self, "_animation", None) is not None
            and getattr(self, "_anim_paused", False)
        ):
            self._animation.event_source.start()
            self.anim_running = True
            self._anim_paused = False

    def _on_close(self):
        """关闭窗口前停止动画，避免后台定时器报错。"""
        if getattr(self, "_animation", None) is not None:
            try:
                self._animation.event_source.stop()
            except Exception:
                pass
        self.root.destroy()

    # --------------------------------------------------------
    # 表格
    # --------------------------------------------------------

    def update_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.path_data is None:
            return

        show = self.path_data.head(300)
        for _, row in show.iterrows():
            self.tree.insert(
                "",
                "end",
                values=(
                    int(row["Step"]),
                    int(row["Layer"]),
                    int(row["Node"]),
                    f"{row['X']:.6g}",
                    f"{row['Y']:.6g}",
                    f"{row['Z']:.6g}",
                    f"{row[self.result_name]:.6g}",
                    row["Path_Priority"],
                    f"{row['Path_Weight']:.3f}",
                ),
            )

    def on_table_click(self, event):
        item = self.tree.focus()
        if not item:
            return

        values = self.tree.item(item, "values")
        if not values:
            return

        step = int(values[0])
        if self.path_data is None:
            return

        row = self.path_data[self.path_data["Step"] == step]
        if row.empty:
            return
        row = row.iloc[0]

        messagebox.showinfo(
            "路径点溯源",
            f"Step：{int(row['Step'])}\n"
            f"分层：第 {int(row['Layer'])} 层\n"
            f"ANSYS Node：{int(row['Node'])}\n"
            f"X：{row['X']}\n"
            f"Y：{row['Y']}\n"
            f"Z：{row['Z']}\n"
            f"{self.result_name}：{row[self.result_name]}\n"
            f"归一化值：{row['Value_Normalized']:.4f}\n"
            f"路径权重：{row['Path_Weight']:.4f}\n"
            f"区域等级：{row['Path_Priority']}",
        )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    def save_merged(self):
        if self.data is None:
            messagebox.showwarning("提示", "请先导入数据。")
            return

        path = filedialog.asksaveasfilename(
            title="保存 ANSYS 融合数据",
            defaultextension=".csv",
            initialfile=config.DEFAULT_MERGED_FILE,
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        self.data.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("保存完成", f"已保存：\n{path}")

    def save_path(self):
        if self.path_data is None:
            messagebox.showwarning("提示", "请先生成路径。")
            return

        path = filedialog.asksaveasfilename(
            title="保存规划路径",
            defaultextension=".csv",
            initialfile=config.DEFAULT_PATH_FILE,
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        self.path_data.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("保存完成", f"已保存：\n{path}")

    # --------------------------------------------------------
    # 内部工具
    # --------------------------------------------------------

    def _value_range(self, df):
        vmin = float(df[self.result_name].min())
        vmax = float(df[self.result_name].max())
        if abs(vmax - vmin) < 1e-15:
            vmax = vmin + 1.0
        return vmin, vmax

    def _update_scroll_region(self, event=None):
        """画布尺寸变化时刷新滚动区域，保证能滚到完整图形。"""
        self.plot_container.configure(
            scrollregion=self.plot_container.bbox("all")
        )
