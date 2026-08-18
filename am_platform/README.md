# 连续碳纤维复合材料 3D 打印智能制造平台（统一主界面）

把子软件整合到同一个 Tk 主窗口中，以标签页切换：

1. **① ANSYS 仿真驱动路径规划**（原 `ansys_path_planner`）
2. **② SATC 参数优化**（原 `satc_optimizer`）
3. **③ 相似零件模板映射**（`ansys_path_planner` 的模板映射窗口）

两个子项目保持原样不动，本平台通过"嵌入适配器"把每个子应用
加载进各自的标签页，因此可以随时单独运行原来的入口。

模板映射标签页与路径规划标签页共用同一套界面架构：四个独立 3D 画布
（模型几何 / 应力分析 / 规划路径 / 虚拟机械臂动画），导入与路径生成
均在后台线程执行并自动降采样，大量点云数据不再卡界面；真实仿真数据
导入后自动规划路径，便于直接存入模板库。

路径规划标签页支持直接导入 STEP(.step/.stp)：调用 gmsh 自动划分
四面体网格并以网格节点作为点云，弹出四视图检查窗口（模板形状 /
STEP 线框 / 四面体网格 / 点云 + 网格质量报告），并可用「STEP 检查」
按钮随时重新打开最近一次网格结果。

注意：启动入口会强制 `matplotlib.use("TkAgg")`——本机 matplotlib 默认
后端可能是 `qtagg`（安装了 PyQt），与 Tk 界面混用会导致窗口白屏。

## 启动

```bash
python main.py
```

也可以直接双击 `run.bat`。

依赖（与两个子项目相同）：

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含 `gmsh`（STEP 网格化必需）。

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
    ("③ 相似零件模板映射", "path_planner.ui.template_window", "TemplateApp"),
]
```

并把对应项目目录加入 `main.py` 的搜索路径列表即可。

## 说明

- 路径规划的数据融合/路径生成、参数优化均在后台线程执行，界面不冻结；
- 同一时间只允许一个模块运行重任务：某个模块计算中，另一个模块的
  运行按钮会自动禁用，顶部状态栏会显示"正在计算"，防止两个大计算
  同时挤占 CPU/内存导致卡死或闪退；
- 切换标签页时自动暂停机械臂动画，切回时恢复，避免隐藏页面在后台
  持续重绘拖慢整体界面；
- 路径规划的 STEP 导入在后台线程执行网格化（gmsh 关闭信号注册，
  兼容后台线程），界面不冻结；
- 启动时强制 matplotlib 的 TkAgg 后端，避免环境默认 qtagg 与 Tk
  画布混用导致窗口白屏/黑屏；
- 关闭主窗口时会先停止子应用里的机械臂动画定时器，避免后台报错。
