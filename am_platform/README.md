# 连续碳纤维复合材料 3D 打印智能制造平台（统一主界面）

把两个独立子软件整合到同一个 Tk 主窗口中，以标签页切换：

1. **① ANSYS 仿真驱动路径规划**（原 `ansys_path_planner`）
2. **② SATC 参数优化**（原 `satc_optimizer`）

两个子项目保持原样不动，本平台通过"嵌入适配器"把每个子应用
加载进各自的标签页，因此可以随时单独运行原来的入口。

## 启动

```bash
python main.py
```

也可以直接双击 `run.bat`。

依赖（与两个子项目相同）：

```bash
pip install -r requirements.txt
```

## 目录结构

```text
am_platform/
├─ main.py                    # 统一入口
├─ run.bat                    # Windows 双击启动
├─ requirements.txt
├─ README.md
└─ workbench/
   ├─ __init__.py
   └─ ui/
      ├─ __init__.py
      ├─ embed.py             # 子应用嵌入适配器（屏蔽 title/geometry/protocol）
      └─ main_window.py       # 主窗口：标题栏 + 标签页 + 模块注册表
```

## 如何新增模块（例如缺陷识别）

在 `workbench/ui/main_window.py` 的 `MODULES` 注册表里追加一项：

```python
MODULES = [
    ("① ANSYS 仿真驱动路径规划", "path_planner.ui.main_window", "ANSYSPathPlannerApp"),
    ("② SATC 参数优化", "satc.ui.main_window", "SATCOptimizerApp"),
    ("③ 在线缺陷识别", "defect_detection.ui.main_window", "DefectDetectionApp"),
]
```

并把对应项目目录加入 `main.py` 的搜索路径列表即可。

## 说明

- 统一入口会在导入 numpy 之前调用 `configure_blas_threads()`，
  避免 Windows 下 GPR 预测偶发阻塞；
- 参数优化在后台线程执行；路径规划的数据融合/路径生成计算量大，
  同步执行（期间界面短暂无响应，属正常现象）；
- 同一时间只允许一个模块运行重任务：某个模块计算中，另一个模块的
  运行按钮会自动禁用，顶部状态栏会显示"正在计算"，防止两个大计算
  同时挤占 CPU/内存导致卡死或闪退；
- 切换标签页时自动暂停机械臂动画，切回时恢复，避免隐藏页面在后台
  持续重绘拖慢整体界面；
- 关闭主窗口时会先停止子应用里的机械臂动画定时器，避免后台报错。
