# ANSYS 仿真驱动路径规划系统

对原单文件脚本
`../ANSYS_仿真驱动路径规划_应力梯度增强版.py`
的工程化重构：拆分为分层包结构，算法行为保持一致，
并修复了原文件无法运行的问题。

## 功能

- 一次多选导入 ANSYS 文件（节点坐标 + X/Y/Z/XY/YZ/XZ 六个应力分量），
  按文件名与内容自动识别分类（与 SATC 参数优化器一致）
- 按 Node 融合数据，计算主应力、主方向与 von-Mises 应力
- 3D 打印层式路径：沿切片轴分层，层内生成应力自适应
  间距的锯齿扫描线（高应力区更密），层间顺序连接
- GUI：四个 3D 视图（模型几何节点云、应力分析结果、
  仿真场与规划路径、虚拟机械臂轨迹）+ 路径溯源表格；
  机械臂（2 连杆逆解）末端精确贴合规划路径实时运动演示，
  支持播放/暂停
- 保存融合 CSV 与路径 CSV

## 目录结构

```text
ansys_path_planner/
├── main.py                        # 入口：启动 GUI
├── requirements.txt               # 运行依赖
├── requirements-dev.txt           # 测试依赖
├── README.md
├── path_planner/
│   ├── config.py                  # 全部常量与可调参数
│   ├── utils/
│   │   ├── text.py                # 多编码文本读取
│   │   └── numeric.py             # 数字提取工具
│   ├── parsers/
│   │   ├── coordinates.py         # 节点坐标解析
│   │   ├── results.py             # 单文件仿真结果解析
│   │   ├── stress_components.py   # 六应力分量加载（文件夹 / 多选文件）
│   │   └── auto_classify.py       # 多选文件自动分类（识别坐标与六分量）
│   ├── analysis/
│   │   ├── stress.py              # 应力张量、主应力、数据融合
│   │   └── path_planning.py       # 路径规划算法（PathPlanner）
│   ├── visualization/
│   │   ├── plots.py               # 密度配色、colorbar 管理
│   │   └── virtual_arm.py         # 虚拟机械臂演示运动学
│   └── ui/
│       └── main_window.py         # tkinter 主窗口
└── tests/                         # pytest 单元测试
```

## 安装与运行

```bash
pip install -r requirements.txt
python main.py
```

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

## 算法说明

1. 六个应力分量组装为 3x3 对称应力张量，特征分解得到主应力与主方向；
2. 最大主应力归一化后控制扫描线间距（高应力 -> 小间距，更密）；
3. 按切片轴（默认 Z）把节点云分成若干层；
4. 每一层内沿分条轴用自适应间距分条，条内沿扫描轴排序，
   相邻条方向交替，形成连续锯齿扫描线（zigzag）；
5. 层与层按顺序连接，输出带 Layer / Segment_Type（层内路径、
   层间过渡）标记的连续路径表；
6. 空区检测：相邻路径段沿直线采样，若中途远离所有节点
   （无零件实体），该段标记为"空区断开"并拆分为新子路径，
   可视化时不再绘制；实体内低应力区域仍全部保留路径。

> 旧版"沿最大主应力方向追踪流线"的算法保留在
> `generate_surface_path()` / `PathPlanner` 中，GUI 默认改用层式规划。

## 与原脚本的差异

- 修复 `IndentationError`：原脚本 `plot_path` 后多出一段模块级缩进代码，
  导致整个文件无法解析运行；重构时将其功能并入密度路径绘制逻辑。
- 修复 colorbar 堆积：多次生成路径时旧色条会残留在图中，现改为先移除再添加。
- 界面扩展为 2x2 四个 3D 视图：左上模型几何（节点云）、
  右上应力分析结果、左下规划路径（仅路径线，密度着色，
  图例放图外右侧不遮挡）、右下虚拟机械臂轨迹。
- 绘图区带横向/纵向滚动条：图形大于可视区域时可用滑块
  滚动查看完整内容（替代原先的滚轮缩放）。
- 删除死代码：`select_result`、未使用的 `target` 变量、多余导入等。
- 所有魔法数字集中到 `path_planner/config.py`，便于统一调参。
- 保留 `merge_ansys_data`（单文件结果模式）作为批处理接口。
