"""相似零件模板映射入口（推荐）：python run_template.py

在 config.py 尚未补充模板库常量时，本入口负责：
1. 注入模板库相关配置常量（带默认值，后续 config 扩展后自动沿用）；
2. 修正真实仿真分支的高优先级分位数（使用默认分位数而非相似度阈值）。
"""

import tkinter as tk

from path_planner import config


# 1) 注入模板库配置常量（若 config 已扩展则保留原值）
_DEFAULTS = {
    "DEFAULT_SIMILARITY_THRESHOLD": 0.80,
    "SIMILARITY_MIN": 0.50,
    "SIMILARITY_MAX": 0.99,
    "SIGNATURE_AXIS_BINS": 16,
    "SIGNATURE_RADIAL_BINS": 24,
    "MAPPING_NEIGHBORS": 3,
    "MAPPING_PATH_SCALE": 1.0,
    "MAPPING_STRESS_SCALE": 1.0,
}
for _key, _value in _DEFAULTS.items():
    if not hasattr(config, _key):
        setattr(config, _key, _value)


# 2) 修正真实仿真分支的分位数（覆盖原方法中的笔误）
def _fixed_process_real(self, node_path, stress_files):
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


from path_planner.ui.template_window import TemplateApp  # noqa: E402

TemplateApp._process_real = _fixed_process_real


def main():
    root = tk.Tk()
    TemplateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
