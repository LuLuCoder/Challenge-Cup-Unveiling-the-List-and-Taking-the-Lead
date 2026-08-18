# 第五章配图生成

本目录提供第五章（《第五章--算法（实际实现版）》）配图的自动生成代码，
数据全部取自仓库真实数据（`data/`、`ansys/`、模板库、`satc_optimizer`
内置实验数据）。

## 生成方式

```bash
cd figures
python make_figures.py            # 生成 5.2 + 5.3 全部图（输出到 output/）
python make_figures.py --path     # 仅 5.2 路径规划图
python make_figures.py --satc     # 仅 5.3 参数优化图
```

运行环境：`pytorch` conda 环境（已装 numpy / pandas / matplotlib / scipy /
gmsh）。如需自定义输出目录，先设置环境变量 `FIG_OUT`。

## 已生成图（16 张，300 dpi）

### 5.2 路径规划

| 文件 | 对应图 | 说明 |
| --- | --- | --- |
| `fig5_1_framework.png` | 图5-1 | 算法总体框架流程（自绘） |
| `fig5_2_stress_field.png` | 图5-2 | von Mises 应力云图 + 主应力方向场 |
| `fig5_3_direction_continuity.png` | 图5-3 | 方向场连续性处理前后对比（示意） |
| `fig5_4_spacing_curve.png` | 图5-4 | 应力自适应间距映射曲线 |
| `fig5_5_zigzag.png` | 图5-5 | 分层与层内锯齿扫描示意 |
| `fig5_6_void_detection.png` | 图5-6 | 空区检测与子路径切分（零件1 实路径） |
| `fig5_7_path_density.png` | 图5-7 | 规划路径整体效果（密度等级着色） |
| `fig5_8_signature.png` | 图5-8 | 点云规范化 + 主轴/径向直方图 |
| `fig5_9_template_mapping.png` | 图5-9 | 模板路径 vs 映射至目标零件 |

图5-10（软件主界面/STEP 检查窗口）为界面截图，需人工截取。

### 5.3 参数优化

| 文件 | 对应图 | 说明 |
| --- | --- | --- |
| `fig5_11_parameter_space.png` | 图5-11 | 81 组参数空间与 9 组正交布点 |
| `fig5_12_loo.png` | 图5-12 | GPR 留一法验证散点（ΔT/ΔB/ΔS） |
| `fig5_13_gpr_uncertainty.png` | 图5-13 | GPR 预测均值 ±2σ 置信带 |
| `fig5_14_all_predictions.png` | 图5-14 | 81 组全空间三目标分布 |
| `fig5_15_pareto.png` | 图5-15 | Pareto 前沿 + 推荐方案 + 论文方案 |
| `fig5_16_weight_sensitivity.png` | 图5-16 | 权重对折中推荐位置的影响 |
| `fig5_17_exposure_weights.png` | 图5-17 | 力学暴露度与自动权重 |

图5-18（参数优化界面）为界面截图，需人工截取。

## 代码结构

```text
figures/
├── make_figures.py   # 入口
├── common.py         # 字体、保存、路径批量绘制等公共工具
├── figs_path.py      # 5.2 路径规划图
├── figs_satc.py      # 5.3 参数优化图
└── output/           # 生成的 PNG（300 dpi）
```

修改任何一张图后，重跑 `python make_figures.py` 即可整体刷新输出。
