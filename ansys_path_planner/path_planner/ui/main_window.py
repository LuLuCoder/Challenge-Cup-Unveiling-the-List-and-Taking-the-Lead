"""ANSYS 仿真驱动路径规划系统主窗口。"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from path_planner import config
from path_planner.analysis.path_planning import generate_layer_path
from path_planner.analysis.stress import merge_ansys_files_data
from path_planner.parsers.auto_classify import classify_ansys_files
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
        self._task_hook = None  # 统一平台注入的忙碌状态回调（可选）

        self.setup_ui()

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

        ttk.Label(
            file_frame,
            text=(
                "一次多选：节点坐标文件 + X/Y/Z/XY/YZ/XZ 六个应力分量文件"
                "（Ctrl/Shift 多选，自动按文件名与内容识别）"
            ),
            foreground="#666666",
        ).grid(row=1, column=1, sticky="w", padx=5, pady=(0, 7))

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

        # 四个 3D 视图：几何 / 应力 / 路径 / 虚拟机械臂
        self.fig = plt.Figure(figsize=(15, 9), dpi=100)
        self.fig.subplots_adjust(
            left=0.03, right=0.97, top=0.97,
            bottom=0.03, wspace=0.15, hspace=0.15,
        )
        self.ax_geom = self.fig.add_subplot(221, projection="3d")
        self.ax_stress = self.fig.add_subplot(222, projection="3d")
        self.ax_path = self.fig.add_subplot(223, projection="3d")
        self.ax_robot = self.fig.add_subplot(224, projection="3d")

        self.canvas = FigureCanvasTkAgg(
            self.fig, master=self.plot_container
        )
        canvas_widget = self.canvas.get_tk_widget()
        self.plot_container.create_window(
            (0, 0), window=canvas_widget, anchor="nw"
        )
        self.plot_container.bind(
            "<Configure>", self._update_scroll_region
        )

    def _build_table(self):
        table_frame = ttk.LabelFrame(self.root, text="③ 路径溯源")
        table_frame.pack(fill="x", padx=12, pady=8)

        columns = (
            "Step", "Layer", "Node", "X", "Y", "Z",
            "SimulationValue", "Priority", "PathWeight",
        )
        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=7
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
        """同步执行文件识别与数据融合。

        计算量大时界面会短暂无响应；统一平台会在此期间禁止另一模块
        同时启动重任务，避免双任务并发卡死/闪退。
        """
        paths = getattr(self, "selected_ansys_files", None) or []
        if not paths:
            messagebox.showerror("导入失败", "请先选择 ANSYS 数据文件。")
            return
        if self._busy:
            messagebox.showinfo("提示", "正在处理中，请稍候……")
            return

        self._set_busy(True, "正在识别并读取 ANSYS 文件……")
        self.root.update()
        try:
            node_path, stress_files, skipped = classify_ansys_files(paths)
            merged = merge_ansys_files_data(node_path, stress_files)
        except Exception as e:
            self._set_busy(False)
            messagebox.showerror("导入失败", str(e))
            self.status_var.set("导入失败")
            return

        self._set_busy(False)
        self.data = merged
        # 后续路径规划统一使用最大主应力
        self.result_name = "Maximum_Principal"

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

    # --------------------------------------------------------
    # 仿真结果可视化
    # --------------------------------------------------------

    def plot_geometry(self):
        """左上视图：实际模型几何形状（节点云）。"""
        if self.data is None:
            return

        self.ax_geom.clear()
        d = self.data
        self.ax_geom.scatter(
            d["X"], d["Y"], d["Z"],
            c="#4C78A8", s=4, alpha=0.85, depthshade=True,
        )
        self.ax_geom.set_title("模型几何形状（节点云）", fontsize=13)
        self.ax_geom.set_xlabel("X")
        self.ax_geom.set_ylabel("Y")
        self.ax_geom.set_zlabel("Z")
        self.canvas.draw()

    def plot_simulation(self):
        """右上视图：应力分析结果。"""
        if self.data is None:
            return

        self.ax_stress.clear()
        d = self.data
        vmin, vmax = self._value_range(d)

        sc = self.ax_stress.scatter(
            d["X"], d["Y"], d["Z"],
            c=d[self.result_name], cmap="turbo",
            vmin=vmin, vmax=vmax, s=10, alpha=0.85, depthshade=True,
        )
        self.ax_stress.set_title("应力分析结果（最大主应力）", fontsize=13)
        self.ax_stress.set_xlabel("X")
        self.ax_stress.set_ylabel("Y")
        self.ax_stress.set_zlabel("Z")

        self._cbar_stress = attach_colorbar(
            self.fig, self.ax_stress, sc,
            shrink=0.65, pad=0.08, current_cbar=self._cbar_stress,
        )
        self.canvas.draw()

    # --------------------------------------------------------
    # 路径规划
    # --------------------------------------------------------

    def plan_path(self):
        """同步执行路径规划（计算量大时界面短暂无响应）。"""
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

        self._set_busy(True, "正在生成路径……")
        self.root.update()
        try:
            path_data, threshold = generate_layer_path(
                self.data,
                "Maximum_Principal",
                percentile=percentile,
                n_layers=n_layers,
            )
        except Exception as e:
            self._set_busy(False)
            messagebox.showerror("路径规划失败", str(e))
            return

        self._set_busy(False)
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
        if self.data is None:
            return

        self.ax_path.clear()
        self.ax_robot.clear()

        p = self.path_data

        # 路径按密集程度分色：稀疏 -> 冷色，高密度 -> 暖色
        xyz = p[["X", "Y", "Z"]].to_numpy(float)
        # 穿越空区（无节点）的段不绘制，避免路径穿过零件孔洞/空隙
        seg_types = p["Segment_Type"].to_numpy()
        skip_mask = seg_types[1:] == "空区断开"
        plot_density_path(
            self.ax_path, xyz, p["Path_Spacing"].to_numpy(float),
            linewidth=2.2, alpha=0.95,
            attach_labels=True, with_legend=False,
            skip_mask=skip_mask,
        )

        self.ax_path.set_title("规划路径（密度着色）", fontsize=13)
        self.ax_path.set_xlabel("X")
        self.ax_path.set_ylabel("Y")
        self.ax_path.set_zlabel("Z")

        # 小图例放图外右侧，不遮挡路径
        self.ax_path.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=7,
            frameon=True,
            framealpha=0.7,
            handlelength=1.2,
            borderpad=0.4,
            labelspacing=0.4,
        )

        # ---- 右图：虚拟机械臂 ----
        self._plot_virtual_arm(p)

        self.canvas.draw()

    def _plot_virtual_arm(self, path_df):
        """把路径压缩到虚拟机械臂工作空间，并让机械臂沿路径运动。"""
        xyz = path_df[["X", "Y", "Z"]].to_numpy(float)

        # 把路径缩放并平移到零件位置，与机械臂错开：
        # 1) 以路径中心为基准等比缩放，使半径 <= 0.12；
        # 2) 平移到 (0.34, 0, 0.12)，最低层 z=0 与基座平面重合；
        # 3) 机械臂基座在原点，零件在右侧，两者不重叠。
        # 大臂=小臂=0.28，伸展半径 0.56，零件任意点都在可达范围内。
        path_center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
        path_radius = float(np.max(np.linalg.norm(xyz - path_center, axis=1)))
        part_radius = 0.12
        scale = part_radius / max(path_radius, 1e-12)
        normalized = (xyz - path_center) * scale
        normalized = normalized + np.array([0.34, 0.0, 0.12])

        # 完整轨迹（虚线）：按子路径分段绘制，跳过穿越空区的段
        seg_types = path_df["Segment_Type"].to_numpy()
        skip = seg_types[1:] == "空区断开"
        run_start = 0
        for i in range(len(normalized) - 1):
            if skip[i]:
                if i > run_start:
                    self._plot_dashed_trajectory(
                        normalized[run_start:i + 1]
                    )
                run_start = i + 1
        if run_start < len(normalized):
            self._plot_dashed_trajectory(normalized[run_start:])

        # 已走轨迹高亮 / 机械臂本体 / 末端点（动画中逐帧更新）
        self._trail_line, = self.ax_robot.plot(
            [], [], [], color="#E45756", linewidth=2.0, alpha=0.95,
        )
        self._arm_line, = self.ax_robot.plot(
            [], [], [], marker="o", linewidth=4, color="#4C78A8",
        )
        self._tip_marker, = self.ax_robot.plot(
            [], [], [], marker="o", markersize=7, color="#E45756",
        )

        # 限制动画帧数，保证演示时长合适（默认约 400 帧）
        max_frames = 400
        step = max(1, int(np.ceil(len(normalized) / max_frames)))
        self._anim_points = normalized[::step]

        self.ax_robot.set_xlim(-0.62, 0.62)
        self.ax_robot.set_ylim(-0.62, 0.62)
        self.ax_robot.set_zlim(0, 0.78)
        self.ax_robot.set_title("虚拟机械臂运动轨迹", fontsize=13)
        self.ax_robot.set_xlabel("X")
        self.ax_robot.set_ylabel("Y")
        self.ax_robot.set_zlabel("Z")

        self._start_arm_animation()

    def _plot_dashed_trajectory(self, points):
        """绘制一段虚线轨迹。"""
        self.ax_robot.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            linestyle="--", linewidth=1.5, alpha=0.7,
        )

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
            self.fig, update, frames=len(pts),
            interval=60, blit=False, repeat=True,
        )
        self.anim_running = True
        self._anim_paused = False
        self.canvas.draw_idle()

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
