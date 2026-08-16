# 5.4 在线缺陷识别方法（YOLOv8）

面向连续碳纤维增强复合材料（cCFRP）3D 打印的在线视觉缺陷识别，
对应论文/报告 5.4 节。方法框架：

```text
5.4.1 在线图像采集（逐层拍照 / 摄像头）
5.4.2 图像预处理（ROI / 尺寸统一 / 归一化）
5.4.3 缺陷识别网络（YOLOv8）
5.4.4 在线判定（置信度 + 位置 + 缺陷面积）
5.4.5 实验验证（P/R/F1/mAP/FPS，与论文对照）
```

## 目录结构

```text
defect_detection/
├── config.py              # 类别、路径、训练/在线参数、论文参考
├── requirements.txt
├── dataset/
│   ├── synthetic.py       # 生成合成缺陷数据集（离线演示）
│   ├── prepare.py         # Roboflow 导出包 / COCO JSON → YOLO 格式
│   └── preprocess.py      # 5.4.2 图像预处理示例
├── train.py               # 5.4.3 YOLOv8 训练（迁移学习）
├── evaluate.py            # 5.4.5 测试集指标 + 混淆矩阵 + 论文对照表
├── predict.py             # 批量推理 + 检测框可视化 + 明细 CSV
├── online.py              # 5.4.1+5.4.4 在线检测（逐层流/摄像头）+ FPS
├── data/                  # 数据集（自动生成/导入）
└── results/               # 训练、评估、预测、在线输出
```

## 快速开始（合成数据演示，无需外部数据）

```bash
pip install -r requirements.txt

# 1. 生成合成缺陷数据集（正常/错位/间隙/堆积）
python dataset/synthetic.py --n-train 120 --n-val 30 --n-test 30

# 2. 训练（首次自动下载 yolov8n.pt 预训练权重）
python train.py --epochs 100

# 3. 测试集评估（输出指标 CSV + 混淆矩阵 + 论文对照表）
python evaluate.py

# 4. 批量推理并画框
python predict.py --source data/images/test

# 5. 在线检测（逐层图片流，模拟逐层拍照）
python online.py --source data/images/test
```

> 合成数据仅用于跑通流程与代码演示；正式验证请使用真实数据
> （见下）。合成图是程序绘制的模拟打印层，不是真实缺陷照片。

## 真实数据来源（论文验证）

推荐验证参考文献：

> Zubayer et al., 2024, *Enhancing additive manufacturing precision:
> Intelligent inspection and optimization for defect-free continuous
> carbon fiber-reinforced polymer*, Composites Part C: Open Access,
> DOI: 10.1016/j.jcomc.2024.100451

论文基于 YOLOv8 对逐层图像做实时缺陷检测（每参数组 50 张图，
错位缺陷识别准确率约 94%，mAP50 约 0.88/0.90/0.94）。
数据接入方式：

1. **Roboflow 导出包（推荐）**：若论文数据或同类 cCFRP 缺陷数据
   在 Roboflow 公开，导出 YOLOv8 格式 zip 后执行：
   ```bash
   python dataset/prepare.py --roboflow-zip roboflow_export.zip
   ```
2. **COCO 标注**：
   ```bash
   python dataset/prepare.py --coco-json annotations.json --image-dir images/
   ```
3. **论文配图**：从开放获取页面（ScienceDirect，DOI 见上）下载
   缺陷示例图，用标注工具（如 labelImg / Roboflow 标注）标成
   YOLO 格式后放入 `data/images/{train,val,test}` +
   `data/labels/...`。

导入后运行 `python dataset/prepare.py --check` 检查类别分布，
再按"快速开始"第 2~5 步执行。

## 与论文对照

运行 `python evaluate.py` 后生成 `results/eval/paper_comparison.md`，
包含本方法的 mAP50 / Precision / Recall / F1 / mAP50-95 与
论文参考指标的对照表，可直接放入 5.4.5 节。

## 在线判定逻辑（5.4.4）

每一层图像推理后：

- 输出缺陷类别、置信度、检测框坐标与面积占比（`predict.py` 的 CSV）；
- `confidence >= alert-conf` 判定为 NG（缺陷），否则 OK；
- 记录逐层判定与 FPS，输出 `results/online/online_log.csv`；
- 可扩展为报警 / 暂停打印 / 反馈参数调整（与 5.2 参数优化闭环）。

## 注意事项

- 训练与推理需要可用的 PyTorch 环境；首次训练会自动下载
  `yolov8n.pt` 预训练权重（需联网）；
- 论文配图版权归原论文所有，用于学术验证请注明出处；
  Roboflow 数据集使用前请确认其许可协议；
- 类别默认：`misalignment`（纤维错位）/ `gap`（间隙）/
  `buildup`（堆积），可在 `config.py` 中按论文实际类别调整。
