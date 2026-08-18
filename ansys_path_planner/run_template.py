"""相似零件模板映射入口（推荐）：python run_template.py

模板映射窗口已与路径规划主窗口共用同一套四画布 + 后台线程架构：
真实仿真数据导入后自动规划路径；仅导入节点坐标/STEP 时自动查库映射。
本入口负责在 config.py 尚未补充模板库常量时注入默认值（后续 config
扩展后自动沿用），随后启动模板映射窗口。
"""

import tkinter as tk

from path_planner import config


# 注入模板库配置常量（若 config 已扩展则保留原值）
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


from path_planner.ui.template_window import TemplateApp  # noqa: E402


def main():
    root = tk.Tk()
    TemplateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
