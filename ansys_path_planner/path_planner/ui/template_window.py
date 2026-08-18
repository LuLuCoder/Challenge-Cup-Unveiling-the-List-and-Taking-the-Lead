"""相似零件模板映射界面（复用主窗口四画布架构）。

使用方式：`python main_template.py`（或统一主界面第③个标签页）
- 建模板：导入真实 ANSYS 数据（节点坐标 + 六应力分量）-> 自动规划路径
  -> 点击「存入模板库」；
- 用模板：新零件只导入节点坐标文件（或直接导入 STEP 自动网格化取点云）
  -> 自动评判与模板的相似度 -> 命中则直接映射模板路径，无需重新 ANSYS 仿真。

与路径规划主窗口共用同一套架构：四个独立 3D 画布（模型几何 / 应力分析 /
规划路径 / 虚拟机械臂动画）、后台线程计算、节点云自动降采样，
大量点云数据导入不再卡界面。
"""

from path_planner import config
from path_planner.ui.main_window import ANSYSPathPlannerApp


class TemplateApp(ANSYSPathPlannerApp):
    """模板映射页 = 路径规划主窗口 + "真实数据导入后自动规划路径"。"""

    def __init__(self, root):
        super().__init__(root)
        try:
            self.root.title("相似零件模板映射 · 免重复仿真")
            self.root.geometry("1650x1000")
        except Exception:
            # 嵌入到统一主界面时 title/geometry 无意义，静默忽略
            pass

    # ------------------------------------------------------------
    # 模板页行为：真实仿真数据导入后直接生成路径
    # ------------------------------------------------------------

    def _on_real_import(self, merged, skipped):
        self.plan_path()

    # ------------------------------------------------------------
    # 兼容旧接口（run_template.py 等外部入口可能调用）
    # ------------------------------------------------------------

    def _process_real(self, node_path, stress_files):
        """同步处理真实仿真数据并生成路径（旧接口；新流程走后台线程）。"""
        from path_planner.analysis.path_planning import generate_layer_path
        from path_planner.analysis.stress import merge_ansys_files_data

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
        self.plot_geometry()
        self.plot_simulation()
        self.plot_path()
        self.update_table()
