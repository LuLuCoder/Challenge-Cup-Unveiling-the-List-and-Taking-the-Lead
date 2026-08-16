"""SATC-NSGA-II 图形界面前端。"""

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from satc import config
from satc.data import get_paper_real_value
from satc.mechanics import classify_ansys_files, suggest_weights
from satc.pareto import dominates
from satc.pipeline import run_pipeline


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


class SATCOptimizerApp:
    """主窗口：运行优化并展示结果。"""

    SUMMARY_COLUMNS = [
        "方案", "A (mm)", "B (mm)", "C (°C)", "D (mm/s)",
        "数据来源", "ΔT", "ΔB", "ΔS", "综合评分",
    ]
    LOO_COLUMNS = ["目标", "MAE", "RMSE", "R²"]
    FILTER_OPTIONS = ["全部", "论文真实实验数据", "GPR代理预测"]

    def __init__(self, root):
        self.root = root
        self.root.title("SATC-NSGA-II 代理辅助多目标优化系统")
        self.root.geometry("1280x840")

        self.output_dir_var = tk.StringVar(
            value=str(config.DEFAULT_OUTPUT_DIR)
        )
        self.status_var = tk.StringVar(value="等待运行优化")
        self.filter_var = tk.StringVar(value="全部")
        self.weight_vars = [
            tk.DoubleVar(value=config.DEFAULT_WEIGHTS[i])
            for i in range(3)
        ]

        self.result = None
        self.all_df = None
        self.pareto_df = None
        self.result_queue = queue.Queue()
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
        top.pack(fill="x", padx=12, pady=(10, 4))

        ttk.Label(
            top,
            text="SATC-NSGA-II 代理辅助热约束多目标优化",
            font=("Microsoft YaHei", 16, "bold"),
        ).pack(side="left")

        ttk.Label(
            top,
            text="（命令行版本：python main.py）",
            foreground="#666666",
        ).pack(side="left", padx=12)

        # 输出目录 + 运行按钮
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=12, pady=4)

        ttk.Label(ctrl, text="输出目录：").pack(side="left")
        ttk.Entry(
            ctrl, textvariable=self.output_dir_var, width=50
        ).pack(side="left", padx=4)
        ttk.Button(
            ctrl, text="选择", command=self.choose_output_dir
        ).pack(side="left", padx=2)

        self.run_button = ttk.Button(
            ctrl, text="▶ 运行优化", command=self.run_optimization
        )
        self.run_button.pack(side="left", padx=18)

        ttk.Label(ctrl, text="权重 ΔT/ΔB/ΔS：").pack(
            side="left", padx=(12, 4)
        )
        for i in range(3):
            ttk.Spinbox(
                ctrl,
                from_=0.0,
                to=10.0,
                increment=0.1,
                textvariable=self.weight_vars[i],
                width=5,
            ).pack(side="left", padx=2)
        ttk.Button(
            ctrl, text="重置等权", command=self.reset_weights
        ).pack(side="left", padx=4)
        ttk.Button(
            ctrl, text="从ANSYS自动设权重",
            command=self.auto_set_weights,
        ).pack(side="left", padx=6)

        ttk.Label(
            self.root,
            text="权重物理意义：ΔT＝拉伸偏差（拉伸性能稳定）｜"
                 "ΔB＝弯曲偏差（抗弯性能稳定）｜"
                 "ΔS＝层间剪切偏差 ILSS（层间结合，抗分层）。"
                 "权重越大 = 越优先保证该项性能。",
            foreground="#666666",
        ).pack(anchor="w", padx=12, pady=(0, 2))

        ttk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Microsoft YaHei", 11),
        ).pack(anchor="w", padx=12, pady=2)

        # 标签页
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        self._build_summary_tab()
        self._build_loo_tab()
        self._build_all_tab()
        self._build_pareto_tab()
        self._build_plot_tab()
        self._build_log_tab()

    def _make_tree(self, parent, columns, widths=None, height=12):
        """带双向滚动条的表格。"""
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=6, pady=6)

        tree = ttk.Treeview(
            frame, columns=columns, show="headings", height=height
        )
        for i, col in enumerate(columns):
            tree.heading(col, text=col)
            tree.column(
                col,
                width=(widths[i] if widths else 110),
                anchor="center",
            )

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _build_summary_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="① 结果摘要")

        self.summary_tree = self._make_tree(
            tab, self.SUMMARY_COLUMNS, widths=[110, 70, 70, 80, 80, 140, 70, 70, 70, 90], height=4
        )

        text_frame = ttk.Frame(tab)
        text_frame.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.summary_text = tk.Text(text_frame, wrap="word", height=10)
        vsb = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.summary_text.yview
        )
        self.summary_text.configure(yscrollcommand=vsb.set)
        self.summary_text.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.summary_text.config(state="disabled")

    def _build_loo_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="② LOO 验证")
        self.loo_tree = self._make_tree(
            tab, self.LOO_COLUMNS, widths=[180, 120, 120, 120], height=6
        )

    def _build_all_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="③ 81 组预测")

        filter_frame = ttk.Frame(tab)
        filter_frame.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(filter_frame, text="筛选：").pack(side="left")
        combo = ttk.Combobox(
            filter_frame,
            textvariable=self.filter_var,
            state="readonly",
            values=self.FILTER_OPTIONS,
            width=18,
        )
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", lambda e: self.refill_all())

        self.all_tree = self._make_tree(tab, self._all_columns(), height=14)

    def _build_pareto_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="④ Pareto 前沿")
        self.pareto_tree = self._make_tree(tab, self._pareto_columns(), height=14)

    def _build_plot_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="⑤ 可视化")

        plot_frame = ttk.Frame(tab)
        plot_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.fig = plt.Figure(figsize=(11.5, 5.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_log_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="⑥ 运行日志")

        frame = ttk.Frame(tab)
        frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.log_text = tk.Text(frame, wrap="none", state="disabled")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.log_text.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    @staticmethod
    def _all_columns():
        return [
            "Index",
            "LayerThickness_mm",
            "FirstLayerThickness_mm",
            "NozzleTemperature_C",
            "PrintingSpeed_mm_s",
            "DataType",
            "PaperExperiment",
            "Optimization_DeltaT",
            "Optimization_DeltaB",
            "Optimization_DeltaS",
            "GPR_DeltaT",
            "GPR_Std_DeltaT",
            "GPR_DeltaB",
            "GPR_Std_DeltaB",
            "GPR_DeltaS",
            "GPR_Std_DeltaS",
        ]

    @staticmethod
    def _pareto_columns():
        return [
            "ParetoIndex",
            "LayerThickness_mm",
            "FirstLayerThickness_mm",
            "NozzleTemperature_C",
            "PrintingSpeed_mm_s",
            "DataType",
            "PaperExperiment",
            "DeltaT",
            "DeltaB",
            "DeltaS",
            "GPR_Std_DeltaT",
            "GPR_Std_DeltaB",
            "GPR_Std_DeltaS",
        ]

    # --------------------------------------------------------
    # 运行
    # --------------------------------------------------------

    def choose_output_dir(self):
        path = filedialog.askdirectory(
            initialdir=config.DEFAULT_OUTPUT_DIR,
            title="选择输出目录",
        )
        if path:
            self.output_dir_var.set(path)

    def run_optimization(self):
        try:
            weights = [float(v.get()) for v in self.weight_vars]
        except tk.TclError:
            messagebox.showerror("权重输入无效", "权重必须是数字。")
            return
        if sum(weights) <= 0:
            messagebox.showerror("权重输入无效", "权重之和必须大于 0。")
            return

        self.status_var.set("优化运行中……（后台线程，界面可继续操作）")
        self.run_button.config(state="disabled")
        hook = self._task_hook
        if hook is not None:
            try:
                hook(self, True)
            except Exception:
                pass
        output_dir = self.output_dir_var.get().strip() or None

        def worker():
            try:
                result = run_pipeline(
                    output_dir=output_dir, weights=weights
                )
                self.result_queue.put(("ok", result))
            except Exception as e:
                self.result_queue.put(("error", repr(e)))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(100, self._poll_result)

    def reset_weights(self):
        for i, value in enumerate(config.DEFAULT_WEIGHTS):
            self.weight_vars[i].set(value)

    def auto_set_weights(self):
        """多选 ANSYS 数据文件，自动分类并设置权重。"""
        paths = filedialog.askopenfilenames(
            title="选择 ANSYS 数据文件（可按住 Ctrl/Shift 多选）："
                  "节点坐标 + X~XZ 六个应力文件 + 变形文件",
            filetypes=[("CSV/TXT", "*.csv *.txt"), ("All files", "*.*")],
        )
        if not paths:
            return

        try:
            node_path, stress_files, deform_path, notes = (
                classify_ansys_files(paths)
            )
        except ValueError as e:
            messagebox.showerror("文件识别失败", str(e))
            return

        try:
            info = suggest_weights(
                node_path,
                stress_files=stress_files,
                deformation_path=deform_path or None,
            )
        except Exception as e:
            messagebox.showerror("自动权重计算失败", str(e))
            return

        for i, w in enumerate(info["weights"]):
            self.weight_vars[i].set(round(float(w), 4))

        message = info["explanation"]
        if notes:
            message += "\n\n提示：\n" + "\n".join(notes)
        messagebox.showinfo("已自动设置权重", message)

    def _poll_result(self):
        try:
            status, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_result)
            return

        self.run_button.config(state="normal")
        hook = self._task_hook
        if hook is not None:
            try:
                hook(self, False)
            except Exception:
                pass
        if status == "ok":
            self.show_result(payload)
            self.status_var.set(
                f"优化完成：Pareto 前沿 {payload['n_pareto']} 个解，"
                "推荐方案已生成"
            )
        else:
            messagebox.showerror("优化失败", payload)
            self.status_var.set("优化失败")

    # --------------------------------------------------------
    # 结果展示
    # --------------------------------------------------------

    def show_result(self, result):
        self.result = result
        out_dir = result["output_dir"]

        self.populate_summary(result)
        self.populate_loo(result)

        self.all_df = pd.read_csv(
            out_dir / config.ALL_PREDICTIONS_FILENAME
        )
        self.pareto_df = pd.read_csv(
            out_dir / config.PARETO_FILENAME
        )
        self.refill_all()
        self.populate_pareto()
        self.plot_results()
        self.load_log(out_dir)

    def populate_summary(self, result):
        paper_idx, paper_f = get_paper_real_value()
        best_x, best_f = result["best_x"], result["best_f"]

        rows = [
            (
                "论文方案",
                *config.PAPER_OPTIMAL,
                "论文真实实验数据",
                *paper_f,
                result["paper_score"],
            ),
            (
                "SATC推荐方案",
                *best_x,
                result["best_data_type"],
                *best_f,
                result["score"],
            ),
        ]
        self.summary_tree.delete(*self.summary_tree.get_children())
        for row in rows:
            self.summary_tree.insert("", "end", values=row)

        if dominates(best_f, paper_f):
            dominance = "SATC推荐方案（代理预测值）支配论文方案真实实验值，仍需真实实验验证。"
        elif dominates(paper_f, best_f):
            dominance = "论文方案真实实验值支配SATC推荐方案。"
        else:
            dominance = "两种方案互不支配，属于不同 Pareto 权衡方案。"

        lines = [
            f"论文方案 {config.PAPER_OPTIMAL_NAME}（真实实验第 {paper_idx + 1} 组）："
            f"ΔT={paper_f[0]:.4f}，ΔB={paper_f[1]:.4f}，ΔS={paper_f[2]:.4f}",
            f"SATC推荐方案："
            f"ΔT={best_f[0]:.4f}，ΔB={best_f[1]:.4f}，ΔS={best_f[2]:.4f}",
            f"GPR预测标准差："
            f"ΔT={result['best_std'][0]:.4f}，"
            f"ΔB={result['best_std'][1]:.4f}，"
            f"ΔS={result['best_std'][2]:.4f}",
            f"推荐点数据来源：{result['best_data_type']}",
            f"目标权重（归一化）："
            f"ΔT={result['weights'][0]:.3f}，"
            f"ΔB={result['weights'][1]:.3f}，"
            f"ΔS={result['weights'][2]:.3f}",
            "权重含义：ΔT＝拉伸性能权重，ΔB＝抗弯性能权重，"
            "ΔS＝层间结合/抗分层权重；权重越大越优先保证该项性能。",
            f"综合评分（百分制，100 分最好，越接近 100 越均衡）："
            f"论文方案 = {result['paper_score']:.4f}，"
            f"SATC推荐方案 = {result['score']:.4f}",
            "注：前沿外的方案可能出现 <0 或 >100 的评分，属正常现象。",
            dominance,
            "注意：GPR 预测结果不能替代真实实验，未实验参数组合建议先做实验验证。",
        ]
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(lines))
        self.summary_text.config(state="disabled")

    def populate_loo(self, result):
        self.loo_tree.delete(*self.loo_tree.get_children())
        for m in result["loo_metrics"]:
            self.loo_tree.insert(
                "",
                "end",
                values=(
                    m["objective"],
                    f"{m['MAE']:.4f}",
                    f"{m['RMSE']:.4f}",
                    f"{m['R2']:.4f}",
                ),
            )

    def refill_all(self):
        if self.all_df is None:
            return
        flt = self.filter_var.get()
        df = (
            self.all_df
            if flt == "全部"
            else self.all_df[self.all_df["DataType"] == flt]
        )
        self.all_tree.delete(*self.all_tree.get_children())
        for _, row in df.iterrows():
            self.all_tree.insert("", "end", values=list(row))

    def populate_pareto(self):
        self.pareto_tree.delete(*self.pareto_tree.get_children())
        if self.pareto_df is None:
            return
        for _, row in self.pareto_df.iterrows():
            self.pareto_tree.insert("", "end", values=list(row))

    # --------------------------------------------------------
    # 可视化
    # --------------------------------------------------------

    def plot_results(self):
        if self.result is None or self.all_df is None or self.pareto_df is None:
            return

        self.fig.clear()
        ax3d = self.fig.add_subplot(121, projection="3d")
        ax2d = self.fig.add_subplot(122)

        # 左图：Pareto 前沿
        all_df = self.all_df
        ax3d.scatter(
            all_df["Optimization_DeltaT"],
            all_df["Optimization_DeltaB"],
            all_df["Optimization_DeltaS"],
            c="#B0BEC5", s=10, alpha=0.55, depthshade=True,
            label="81 组参数空间",
        )
        p = self.pareto_df
        ax3d.scatter(
            p["DeltaT"], p["DeltaB"], p["DeltaS"],
            c="#F58518", s=35, alpha=0.9, depthshade=True,
            label=f"Pareto 前沿（{len(p)} 个）",
        )

        best_f = self.result["best_f"]
        ax3d.scatter(
            [best_f[0]], [best_f[1]], [best_f[2]],
            c="#D62728", marker="*", s=280,
            label="SATC 推荐方案",
        )
        _, paper_f = get_paper_real_value()
        ax3d.scatter(
            [paper_f[0]], [paper_f[1]], [paper_f[2]],
            c="#2CA02C", marker="^", s=170,
            label="论文方案",
        )

        ax3d.set_xlabel("ΔT")
        ax3d.set_ylabel("ΔB")
        ax3d.set_zlabel("ΔS")
        ax3d.set_title("Pareto 前沿（三目标最小化）", fontsize=12)
        ax3d.legend(loc="upper right", fontsize=8, framealpha=0.7)

        # 右图：LOO 预测 vs 实际
        loo_df = pd.read_csv(
            self.result["output_dir"] / config.LOO_FILENAME
        )
        colors = ["#4C78A8", "#F58518", "#54A24B"]
        tags = ["DeltaT", "DeltaB", "DeltaS"]
        for j, (name, tag) in enumerate(
            zip(config.OBJECTIVE_NAMES, tags)
        ):
            real = loo_df[f"Real_{tag}"]
            pred = loo_df[f"Pred_{tag}"]
            ax2d.scatter(
                real, pred, s=45, alpha=0.85,
                color=colors[j], label=name,
            )

        all_vals = np.concatenate(
            [loo_df[f"Real_{t}"].to_numpy() for t in tags]
        )
        lo, hi = float(all_vals.min()), float(all_vals.max())
        ax2d.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
        ax2d.set_xlabel("真实值")
        ax2d.set_ylabel("LOO 预测值")
        ax2d.set_title("LOO 留一验证：预测 vs 实际", fontsize=12)
        ax2d.legend(fontsize=8, framealpha=0.7)

        self.fig.tight_layout()
        self.canvas.draw()

    def load_log(self, out_dir):
        log_path = out_dir / config.LOG_FILENAME
        text = (
            log_path.read_text(encoding="utf-8")
            if log_path.exists()
            else "（无日志文件）"
        )
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", text)
        self.log_text.config(state="disabled")
