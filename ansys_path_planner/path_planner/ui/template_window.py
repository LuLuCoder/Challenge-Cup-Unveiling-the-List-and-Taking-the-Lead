"""相似零件模板映射界面（路径规划增强版）。

使用方式：`python main_template.py`
- 建模板：导入真实 ANSYS 数据（节点坐标 + 六应力分量）-> 自动规划路径
  -> 点击「存入模板库」；
- 用模板：新零件只导入节点坐标文件 -> 自动评判与模板的相似度
  -> 命中（>= 阈值）则直接映射模板路径，无需重新 ANSYS 仿真。
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import matplotlib

if matplotlib.get_backend().lower() != "tkagg":
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
from path_planner.parsers.coordinates import parse_coordinate_file
from path_planner.visualization.plots import plot_density_path


plt.rcParams["font.sans-serif"] = config.MATPLOTLIB_FONTS
plt.rcParams["axes.unicode_minus"] = False

_LETTER_MAP = {
    "x": "SX", "y": "SY", "z": "SZ",
    "xy": "SXY", "yz": "SYZ", "xz": "SXZ",
}


def _component_from_filename(path):
    name = os.path.basename(str(path)).strip().lower()
    low_map = {k.lower(): v for k, v in config.STRESS_FILE_MAP.items()}
    if name in low_map:
        return low_map[name]
    return _LETTER_MAP.get(name.split(".")[0].strip())


def _classify(paths):
    """本地轻量分类：按文件名识别六应力分量，按内容识别节点坐标文件。"""
    stress_files = {}
    others = []
    for p in paths:
        comp = _component_from_filename(p)
        if comp is not None:
            stress_files[comp] = str(p)
        else:
            others.append(str(p))

    node_path = None
    for p in others:
        try:
            df = parse_coordinate_file(p)
            if len(df):
                node_path = p
                break
        except Exception:
            continue
    if node_path is None:
        raise ValueError("未识别到节点坐标文件（需包含 Node,X,Y,Z 数据）。")
    return node_path, stress_files


class TemplateApp:
    """模板库主窗口：真实仿真建模板 / 相似零件直接映射模板路径。"""

    def __init__(self, root):
        self.root = root
        self.root.title("相似零件模板映射（路径规划增强版）")
        self.root.geometry("1280x860")

        self.selected_files = []
        self.data = None
        self.path_data = None
        self.result_name = "Maximum_Principal"
        self._last_real_data = None
        self._task_hook = None  # 统一平台注入的忙碌状态回调（可选）

        self.setup_ui()

    def set_task_hook(self, callback):
        """供统一平台注入忙碌状态回调：callback(app, busy)。"""
        self._task_hook = callback

    def _notify_busy(self, busy):
        hook = self._task_hook
        if hook is not None:
            try:
                hook(self, busy)
            except Exception:
                pass

    def setup_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=(10, 4))
        ttk.Label(
            top,
            text="相似零件模板映射 · 免重复仿真",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(side="left")

        file_frame = ttk.LabelFrame(self.root, text="① ANSYS 数据")
        file_frame.pack(fill="x", padx=12, pady=5)

        ttk.Label(file_frame, text="数据文件：").grid(
            row=0, column=0, padx=8, pady=8
        )
        self.files_var = tk.StringVar()
        ttk.Entry(
            file_frame, textvariable=self.files_var, width=72
        ).grid(row=0, column=1, padx=5)
        ttk.Button(
            file_frame, text="选择文件", command=self.select_files
        ).grid(row=0, column=2, padx=5)
        self.process_button = ttk.Button(
            file_frame, text="导入并处理", command=self.process
        )
        self.process_button.grid(row=0, column=3, padx=15)

        ttk.Label(file_frame, text="相似度阈值：").grid(
            row=1, column=0, padx=8, pady=8
        )
        self.threshold_var = tk.DoubleVar(
            value=config.DEFAULT_SIMILARITY_THRESHOLD
        )
        ttk.Spinbox(
            file_frame,
            from_=config.SIMILARITY_MIN,
            to=config.SIMILARITY_MAX,
            increment=0.01,
            textvariable=self.threshold_var,
            width=6,
        ).grid(row=1, column=1, sticky="w", padx=5)
        self.save_button = ttk.Button(
            file_frame, text="存入模板库", command=self.save_as_template
        )
        self.save_button.grid(row=1, column=2, padx=5)
        self.library_button = ttk.Button(
            file_frame, text="模板库", command=self.open_library
        )
        self.library_button.grid(row=1, column=3, padx=5)

        ttk.Label(
            file_frame,
            text=(
                "建模板：节点坐标 + 六应力分量 -> 自动规划 -> 存入模板库；\n"
                "用模板：只选节点坐标文件 -> 自动查库，相似度 ≥ 阈值则直接映射模板路径。"
            ),
            foreground="#666666",
        ).grid(row=2, column=1, columnspan=3, sticky="w", padx=5, pady=(0, 7))

        self.status_var = tk.StringVar(value="等待导入数据")
        ttk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 11),
        ).pack(anchor="w", padx=15, pady=5)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.fig = plt.Figure(figsize=(12, 7.5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------
    # 文件选择与处理
    # ------------------------------------------------------------

    def select_files(self):
        paths = filedialog.askopenfilenames(
            title=(
                "选择 ANSYS 数据文件（可多选）："
                "节点坐标文件；建模板时再加 X/Y/Z/XY/YZ/XZ 六个应力分量文件"
            ),
            filetypes=[
                ("CSV/TXT", "*.csv *.txt"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self.selected_files = list(paths)
            self.files_var.set("；".join(paths))

    def process(self):
        if not self.selected_files:
            messagebox.showerror("导入失败", "请先选择 ANSYS 数据文件。")
            return
        try:
            node_path, stress_files = _classify(self.selected_files)
        except Exception as e:
            messagebox.showerror("导入失败", str(e))
            return

        self._notify_busy(True)
        self.status_var.set("正在识别并处理……")
        self.root.update()
        try:
            if len(stress_files) == len(config.STRESS_COMPONENTS):
                self._process_real(node_path, stress_files)
            else:
                self._process_mapping(node_path)
        except Exception as e:
            self.status_var.set("处理失败")
            messagebox.showerror("处理失败", str(e))
            return
        finally:
            self._notify_busy(False)
        self.plot()

    def _process_real(self, node_path, stress_files):
        merged = merge_ansys_files_data(node_path, stress_files)
        path_data, _threshold = generate_layer_path(
            merged,
            "Maximum_Principal",
            percentile=config.DEFAULT_PERCENTILE,
            n_layers=int(config.DEFAULT_LAYERS),
        )
        self.data = merged
        self.path_data = path_data
        self._last_real_data = merged.copy()
        self.status_var.set(
            f"真实仿真数据：{len(merged)} 个节点，已规划 "
            f"{len(path_data)} 个路径点。可点击「存入模板库」建立模板。"
        )

    def _process_mapping(self, node_path):
        node_df = parse_coordinate_file(node_path)
        signature = compute_signature(
            node_df[["X", "Y", "Z"]].to_numpy(float)
        )
        threshold = float(self.threshold_var.get())
        entry, sim = find_best_template(signature, threshold)
        if entry is None:
            raise ValueError(
                f"模板库中最高相似度 = {sim:.3f}，低于阈值 {threshold:.2f}。\n"
                "请提供该零件的六应力分量文件进行真实仿真，"
                "或调低相似度阈值后重试。"
            )
        mapped_geom, _mapped_path = map_from_template(entry["path"], node_df)
        # 用映射得到的应力场按层重新规划，保证路径是 3D 打印的一层一层结构
        n_layers = int(entry.get("n_layers") or config.DEFAULT_LAYERS)
        path_data, _threshold = generate_layer_path(
            mapped_geom,
            "Maximum_Principal",
            percentile=config.DEFAULT_PERCENTILE,
            n_layers=n_layers,
        )
        self.data = mapped_geom
        self.path_data = path_data
        self._last_real_data = None
        self.status_var.set(
            f"命中模板「{entry['name']}」（相似度 {sim:.2f}），"
            f"已通过模板映射应力场并按层规划路径"
            f"（{len(path_data)} 个路径点，共 "
            f"{int(path_data['Layer'].max())} 层），"
            "未重新进行 ANSYS 仿真。"
        )

    # ------------------------------------------------------------
    # 模板库管理
    # ------------------------------------------------------------

    def save_as_template(self):
        if self._last_real_data is None or self.path_data is None:
            messagebox.showinfo(
                "提示",
                "请先导入真实仿真数据并生成路径，再存入模板库。",
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

    def open_library(self):
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
            columns=("name", "nodes", "path", "time"),
            show="headings",
            height=12,
        )
        for col, text, width in (
            ("name", "名称", 220),
            ("nodes", "节点数", 90),
            ("path", "路径点数", 90),
            ("time", "创建时间", 170),
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
            command=lambda: self._delete_selected(tree, entry_map, win),
        ).pack(side="left")
        ttk.Button(btns, text="关闭", command=win.destroy).pack(side="right")

    def _delete_selected(self, tree, entry_map, win):
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

    # ------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------

    def plot(self):
        if self.data is None:
            return
        self.ax.clear()
        d = self.data
        self.ax.scatter(
            d["X"], d["Y"], d["Z"],
            c="#4C78A8", s=3, alpha=0.55, depthshade=True,
        )
        if self.path_data is not None:
            p = self.path_data
            xyz = p[["X", "Y", "Z"]].to_numpy(float)
            seg = p["Segment_Type"].to_numpy()
            skip = seg[1:] == "空区断开"
            plot_density_path(
                self.ax,
                xyz,
                p["Path_Spacing"].to_numpy(float),
                linewidth=2.2,
                alpha=0.95,
                attach_labels=True,
                with_legend=False,
                skip_mask=skip,
            )
            self.ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=7,
                framealpha=0.7,
            )
        self.ax.set_title("模型节点 + 打印路径", fontsize=13)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw()
