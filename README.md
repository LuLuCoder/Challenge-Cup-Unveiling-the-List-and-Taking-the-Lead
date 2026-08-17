# 连续碳纤维复合材料 3D 打印智能制造系统

面向**挑战杯·揭榜挂帅 3D 打印赛题**的完整软件体系，围绕连续碳纤维增强聚合物
（cCFRP）3D 打印构建"**仿真驱动路径规划 → 代理辅助参数优化 → 在线缺陷识别**"
闭环，覆盖打印前（设计/规划）、打印中（工艺参数）、打印后/过程中（质量检测）
三个环节。

本仓库包含三个功能模块、一个统一主界面，以及 ANSYS 仿真工程与示例数据：

| 文件夹                 | 模块                      | 作用                                     |
| ---------------------- | ------------------------- | ---------------------------------------- |
| `ansys_path_planner` | ① ANSYS 仿真驱动路径规划 | 按应力场生成 3D 打印层式路径             |
| `satc_optimizer`     | ② SATC-NSGA-II 参数优化  | 代理模型 + Pareto 推荐打印参数           |
| `defect_detection`   | ③ 在线缺陷识别           | YOLOv8 逐层视觉缺陷检测闭环              |
| `am_platform`        | 统一主界面                | 将 ①② 整合进同一窗口（③ 可扩展接入）  |
| `ansys`              | ANSYS Workbench 工程      | 本赛题零件的力学仿真工程（analyse.wbpj） |
| `data`               | 示例数据                  | 节点坐标 + 六应力分量，可直接导入 ①②   |

---

## 1. 系统总体架构

系统由三个功能模块和一个统一主界面组成，覆盖打印全流程：

| 环节     | 模块            | 输入                        | 输出                                            | 关键技术                                  |
| -------- | --------------- | --------------------------- | ----------------------------------------------- | ----------------------------------------- |
| 打印前   | ① 路径规划     | ANSYS 节点坐标 + 六应力分量 | 层式 3D 打印路径                                | 应力自适应间距、空区检测、模板映射免仿真、6 自由度机械臂演示 |
| 打印前   | ② 参数优化     | 9 组论文真实实验数据        | 推荐打印参数（层厚/首层层厚/喷嘴温度/打印速度） | RBF-GPR 代理模型、Pareto 前沿、百分制评分 |
| 打印中   | ③ 缺陷识别     | 打印过程逐层图像            | NG/OK 判定、缺陷类别与位置                      | YOLOv8、在线检测与 FPS 统计               |
| 统一入口 | `am_platform` | —                          | ①② 集成于同一界面，③ 可扩展接入              | 标签页集成、并发任务保护                  |

三个模块构成闭环：ANSYS 仿真结果驱动路径规划；实验数据训练代理模型得到推荐
参数；打印过程中的逐层图像经 YOLOv8 在线检测，缺陷判定结果可反馈回参数优化
环节形成闭环调整。各模块也可独立运行。

---

## 2. 模块说明

### 2.1 ① ANSYS 仿真驱动路径规划（`ansys_path_planner/`）

把 ANSYS 力学仿真结果转化为面向 3D 打印的**层式路径**：

- **数据导入**：一次多选节点坐标文件与 X/Y/Z/XY/YZ/XZ 六个应力分量文件，
  按文件名与内容自动识别分类（与参数优化模块一致的导入方式）；
- **应力分析**：由应力张量计算最大/中间/最小主应力、主应力方向与 von-Mises
  应力（批量矩阵特征分解，大数据快速）；
- **路径规划**：沿切片轴（默认 Z）分层，层内生成应力自适应间距的锯齿扫描线
  （高应力区更密），层间顺序连接；无节点（空区）区域自动断开、不绘制路径，
  实体内低应力区域全部保留；
- **可视化**：四个独立 3D 画布（模型几何、应力分析、规划路径、虚拟机械臂），
  只有机械臂画布实时动画；**6 自由度关节臂**末端精确贴合路径、零件放大显示；
  双击任意图可弹出放大窗口（机械臂放大窗口为"左：机械臂动画 / 右：零件
  路径完成进度"双窗）；节点云全量显示，路径用批量绘制不糊图；
- **性能**：导入与路径生成在后台线程执行，界面不冻结；13.8 万节点
  导入约 2 秒、规划约 2 秒。
- **相似零件模板映射（免重复仿真）**：真实仿真 + 路径规划结果可存入模板库；
  新零件只导入节点坐标文件，系统自动评判与模板的形状相似度（主轴坐标/径向
  直方图 L1 距离，阈值可调），相似度达标则映射模板应力场并在新零件上按层
  重规划路径（仍是层式 3D 打印结构），无需重新做 ANSYS 分析。

```bash
cd ansys_path_planner
pip install -r requirements.txt
python main.py
```

依赖：`numpy pandas matplotlib scipy`（测试：`pytest`）。

### 2.2 ② SATC-NSGA-II 参数优化（`satc_optimizer/`）

基于论文真实实验数据的小样本代理辅助多目标优化：

- **数据**：内置 9 组 cCFRP 3D 打印真实实验数据（Xie 2024）；
- **代理模型**：手写 RBF 高斯过程回归（GPR），对 3 层 4 参数离散空间（81 组）
  预测三个目标 ΔT（拉伸偏差）/ ΔB（弯曲偏差）/ ΔS（层间剪切偏差 ILSS）；
- **寻优**：热约束过滤 → Pareto 前沿 → 按权重折中推荐；
- **评分**：百分制综合评分（100 分最好），论文方案与推荐方案同基准对比；
- **权重**：可自定义 ΔT/ΔB/ΔS 权重（物理意义已在界面标注），也可从 ANSYS
  力学结果自动设置权重（点云 + 六应力分量 + 变形文件，一次多选自动分类）。

```bash
cd satc_optimizer
pip install -r requirements.txt
python app.py        # 图形界面
python main.py       # 命令行（可加 --weights / --auto-weights）
```

依赖：`numpy pandas scikit-learn matplotlib`（测试：`pytest`）。

### 2.3 ③ 在线缺陷识别（`defect_detection/`）

面向打印过程逐层图像的 YOLOv8 在线视觉缺陷识别闭环：

- **类别**：`misalignment`（纤维错位）/ `gap`（间隙）/ `buildup`（堆积）；
- **数据**：内置合成演示数据生成器；支持 Roboflow 导出包（YOLOv8 格式）与
  COCO JSON 导入；`yolov8n.pt` 预训练权重已下载；
- **流水线**：数据准备 → 预处理（ROI/尺寸统一/归一化）→ 训练（迁移学习）→
  评估（Precision/Recall/F1/mAP + 混淆矩阵 + 论文对照表）→ 批量预测 →
  在线检测（置信度判定 NG/OK，输出 FPS 与逐层日志）；
- **论文参考**：Zubayer et al., 2024, Composites Part C: Open Access,
  DOI: 10.1016/j.jcomc.2024.100451（YOLOv8，错位识别准确率约 94%）。

```bash
cd defect_detection
pip install -r requirements.txt
python dataset/synthetic.py --n-train 120 --n-val 30 --n-test 30
python train.py --epochs 100
python evaluate.py
python predict.py --source data/images/test
python online.py --source data/images/test
```

依赖：`ultralytics opencv-python numpy pandas` 等（需 PyTorch 环境）。

### 2.4 统一主界面（`am_platform/`）

把 ①②③ 整合进同一个 Tk 主窗口（标签页切换），并做了并发保护：

- 任一模块计算中，另一模块的重任务按钮自动禁用（避免双任务并发卡死/闪退）；
- 切换标签页自动暂停/恢复机械臂动画；关闭窗口自动清理动画定时器；
- 启动时强制 matplotlib 的 TkAgg 后端，避免环境默认 qtagg 导致白屏；
- 新增模块只需在 `workbench/ui/main_window.py` 的 `MODULES` 注册表中追加一项。

```bash
cd am_platform
pip install -r requirements.txt
python main.py        # 或双击 run.bat
```

---

## 3. 数据说明

### 3.1 ANSYS 仿真文件（路径规划 / 自动权重共用）

仓库已附带一套可直接运行的示例数据：

- `ansys/`：ANSYS Workbench 仿真工程（`analyse.wbpj`），对应赛题零件的静力学分析；
- `data/`：从该工程导出的数据——`node_coordinates.csv`（节点坐标）
  与 `X.txt / Y.txt / Z.txt / XY.txt / YZ.txt / XZ.txt`（六应力分量），
  在路径规划或参数优化界面中一次多选即可导入。

文件格式约定：

- **节点坐标文件**：`Node, X, Y, Z`（CSV/TXT，兼容 ANSYS 导出异常格式）；
- **六应力分量文件**：`X.txt / Y.txt / Z.txt / XY.txt / YZ.txt / XZ.txt`
  （节点号 + 数值，文件名不区分大小写）；
- **变形结果文件**（可选，自动权重用）：节点号 + 变形值。

### 3.2 参数优化实验数据（内置）

9 组论文真实实验（参数：层厚 A、首层层厚 B、喷嘴温度 C、打印速度 D；
目标：ΔT / ΔB / ΔS），定义在 `satc_optimizer/satc/config.py`。

### 3.3 缺陷识别数据集

论文原文声明数据 "available on request"（需向作者索取）。可用的公开替代：

- Roboflow · 疑似论文原数据集（上海交大，yolo v8 CFRP-misalignment defect
  inspection and distance measurement）：
  https://universe.roboflow.com/shanghai-jiaotong-university-bfxbs/yolo-v8-cfrp-misalignment-defect-inspection-and-distance-measurement
- 其他相近数据集（连续碳纤维在线检测 / 碳纤维表面缺陷 / 3D 打印过程缺陷）：
  见 `defect_detection/README.md` 的"真实数据来源"章节。

> 注意：论文配图版权归原论文所有，用于学术验证需注明出处；Roboflow 数据集
> 使用前请确认其许可协议。

---

## 4. 运行环境

- Python 3.9+（建议使用 conda 环境）；
- 各模块依赖可在各自目录 `pip install -r requirements.txt`，或统一安装：

```bash
pip install numpy pandas matplotlib scipy scikit-learn ultralytics opencv-python
```

- 测试：各工程根目录执行 `pytest`。

---

## 5. 目录结构

```text
Challenge-Cup-Unveiling-the-List-and-Taking-the-Lead/
├── README.md                     # 本文件（系统总览）
├── ansys/                        # ANSYS Workbench 仿真工程（analyse.wbpj）
├── data/                         # 示例 ANSYS 数据（节点坐标 + 六应力分量）
├── ansys_path_planner/           # ① 路径规划
│   ├── main.py
│   ├── main_template.py / run_template.py   # 模板映射独立入口
│   ├── path_planner/
│   │   ├── config.py             # 全局常量与算法参数
│   │   ├── parsers/              # 坐标/结果/应力分量/自动分类解析
│   │   ├── analysis/             # 应力分析 + 层式规划 + 形状签名 + 模板库
│   │   ├── visualization/        # 绘图 + 6 自由度虚拟机械臂
│   │   └── ui/                   # tkinter 主窗口 + 模板映射窗口
│   └── tests/
├── satc_optimizer/               # ② 参数优化
│   ├── app.py                    # GUI 入口
│   ├── main.py                   # 命令行入口
│   ├── satc/
│   │   ├── config.py             # 论文数据、参数水平、权重
│   │   ├── data.py / gpr.py      # 数据与 GPR 代理模型
│   │   ├── pareto.py / pipeline.py
│   │   ├── mechanics.py          # ANSYS 自动权重
│   │   └── ui/                   # tkinter 主窗口
│   └── tests/
├── defect_detection/             # ③ 在线缺陷识别
│   ├── config.py / train.py / evaluate.py / predict.py / online.py
│   ├── dataset/                  # 合成数据 / Roboflow / COCO 导入
│   ├── data/                     # 数据集与论文 PDF
│   ├── results/                  # 训练与评估输出
│   └── yolov8n.pt                # 预训练权重
└── am_platform/                  # 统一主界面
    ├── main.py / run.bat
    └── workbench/ui/             # 主窗口 + 嵌入适配器
```

---

## 6. 与赛题/报告章节的对应

| 赛题环节     | 系统模块                  | 对应报告章节 |
| ------------ | ------------------------- | ------------ |
| 打印路径设计 | ① ANSYS 仿真驱动路径规划 | 5.1          |
| 打印参数优选 | ② SATC-NSGA-II 参数优化  | 5.2          |
| 过程质量监控 | ③ 在线缺陷识别（YOLOv8） | 5.4          |
