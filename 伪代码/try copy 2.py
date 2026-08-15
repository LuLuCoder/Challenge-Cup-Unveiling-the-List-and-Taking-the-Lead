#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第五章 5.4节 -- 在线缺陷识别方法 配图生成脚本
生成5张学术风格矢量图，300 DPI输出
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# ============================================================
# 中文字体自动检测与设置
# ============================================================
# 按优先级尝试常见中文字体（Windows / Linux / macOS）
candidates = [
    'SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun',
    'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
    'Noto Sans CJK SC', 'Noto Sans CJK JP',
    'Source Han Sans SC', 'Source Han Serif SC',
    'PingFang SC', 'Heiti SC', 'STHeiti',
]

available = set(f.name for f in fm.fontManager.ttflist)
chosen = None
for c in candidates:
    if c in available:
        chosen = c
        break

if chosen:
    plt.rcParams['font.family'] = chosen
    print(f"[INFO] 使用中文字体: {chosen}")
else:
    print("[WARN] 未找到已知中文字体，尝试加载系统默认字体")
    # 回退：遍历查找任何带 CJK / SC / CN 的字体
    for f in available:
        if any(k in f for k in ['CJK', 'SC', 'CN', 'Hei', 'Song', 'Kai', 'Fang']):
            plt.rcParams['font.family'] = f
            print(f"[INFO] 回退字体: {f}")
            break

plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

OUTPUT_DIR = './'


# ============================================================
# 图1：多传感器协同检测体系架构图
# ============================================================
def plot_sensor_architecture():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('图5-X  多传感器协同检测体系架构', fontsize=14, fontweight='bold', pad=20)

    # 缺陷类型（左侧）
    defects = ['铺放偏移', '成型厚度异常', '送料量失配', '断丝', '纤维浸润不足']
    defect_colors = ['#E8F4FD', '#FFF3E0', '#E8F5E9', '#FFEBEE', '#F3E5F5']
    response_levels = ['2级：限幅补偿', '2级：限幅补偿', '2级：限幅补偿', '4级：安全停机', '3级：风险标记']

    for i, (defect, color, level) in enumerate(zip(defects, defect_colors, response_levels)):
        y = 6.5 - i * 1.3
        rect = FancyBboxPatch((0.3, y - 0.35), 2.2, 0.7, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(1.4, y, defect, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(1.4, y - 0.2, level, ha='center', va='center', fontsize=8, color='#666666')

    # 传感器层（中间）
    sensors = ['工业相机', '激光位移' + chr(10) + '传感器', '张力传感器', '送料编码器', '红外热像仪']
    sensor_colors = ['#BBDEFB', '#C8E6C9', '#FFCCBC', '#D1C4E9', '#B2EBF2']
    sensor_x = [4.8, 6.0, 7.2, 8.4, 9.6]

    for x, sensor, color in zip(sensor_x, sensors, sensor_colors):
        rect = FancyBboxPatch((x - 0.5, 6.8), 1.0, 0.8, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, 7.2, sensor, ha='center', va='center', fontsize=9, fontweight='bold')

    # 核心算法层（右侧）
    algorithms = ['边缘提取' + chr(10) + '中心线比对', '轮廓三维重建', '宽度偏差率' + chr(10) + '长度校验',
                  '多源逻辑表决', '温度梯度' + chr(10) + '异常检测']
    alg_y = [6.5, 5.2, 3.9, 2.6, 1.3]
    alg_colors = ['#E3F2FD', '#E8F5E9', '#FFF8E1', '#FFEBEE', '#F3E5F5']

    for y, alg, color in zip(alg_y, algorithms, alg_colors):
        rect = FancyBboxPatch((8.5, y - 0.35), 2.8, 0.7, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(9.9, y, alg, ha='center', va='center', fontsize=9)

    # 缺陷->传感器 连接
    connections = [
        (0, [0]),
        (1, [0, 1]),
        (2, [0, 3]),
        (3, [0, 2, 3]),
        (4, [4]),
    ]
    for def_idx, sensor_indices in connections:
        y_start = 6.5 - def_idx * 1.3
        for si in sensor_indices:
            x_end = sensor_x[si]
            ax.annotate('', xy=(x_end, 6.8), xytext=(2.5, y_start),
                       arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8,
                                      connectionstyle="arc3,rad=0.1"))

    # 传感器->算法 连接
    alg_connections = [
        (0, 0), (1, 1), (0, 2), (3, 2),
        (2, 3), (0, 3), (3, 3), (4, 4),
    ]
    for si, ai in alg_connections:
        ax.annotate('', xy=(8.5, alg_y[ai]), xytext=(sensor_x[si] + 0.5, 6.8),
                   arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8,
                                  connectionstyle="arc3,rad=0.1"))

    ax.text(6.0, 0.3,
            '注：箭头表示数据流向；响应等级2级为限幅补偿，3级为风险标记，4级为安全停机',
            ha='center', va='center', fontsize=9, style='italic', color='#555555')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig5_sensor_architecture.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] fig5_sensor_architecture.png")


# ============================================================
# 图2：断丝多模态检测三源融合逻辑表决机制
# ============================================================
def plot_filament_breakage_logic():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('图5-X  断丝多模态检测三源融合逻辑表决机制', fontsize=14, fontweight='bold', pad=20)

    sources = [
        ('视觉检测', '帧间差分' + chr(10) + '轨迹连续性突变', 1.5, 5.5, '#E3F2FD'),
        ('张力监测', '百Hz频率采样' + chr(10) + '张力跌落至近零', 5.0, 5.5, '#FFEBEE'),
        ('编码器校验', '实际vs理论' + chr(10) + '送料长度比对', 8.5, 5.5, '#FFF8E1'),
    ]

    for name, desc, x, y, color in sources:
        rect = FancyBboxPatch((x - 1.0, y - 0.6), 2.0, 1.2, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y + 0.15, name, ha='center', va='center', fontsize=11, fontweight='bold')
        ax.text(x, y - 0.25, desc, ha='center', va='center', fontsize=8, color='#555555')
        ax.annotate('', xy=(x, 3.8), xytext=(x, 4.9),
                   arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # 逻辑表决模块
    logic_rect = FancyBboxPatch((2.5, 2.5), 5.0, 1.3, boxstyle="round,pad=0.1",
                                 facecolor='#F5F5F5', edgecolor='#333333', linewidth=2)
    ax.add_patch(logic_rect)
    ax.text(5.0, 3.4, '逻辑表决机制', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5.0, 2.9, '任意两源同时报警 -> 触发断丝判定', ha='center', va='center',
            fontsize=10, color='#C62828')

    # 真值表
    table_data = [
        ['视觉', '张力', '编码器', '判定结果'],
        ['0', '0', '0', '正常'],
        ['1', '0', '0', '正常'],
        ['0', '1', '0', '正常'],
        ['0', '0', '1', '正常'],
        ['1', '1', '0', '断丝（报警）'],
        ['1', '0', '1', '断丝（报警）'],
        ['0', '1', '1', '断丝（报警）'],
        ['1', '1', '1', '断丝（报警）'],
    ]

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='bottom', bbox=[0.15, 0.02, 0.7, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for i in range(4):
        table[(0, i)].set_facecolor('#E0E0E0')
        table[(0, i)].set_text_props(fontweight='bold')

    for row in [4, 5, 6, 7]:
        for col in range(4):
            table[(row, col)].set_facecolor('#FFEBEE')
        table[(row, 3)].set_text_props(color='#C62828', fontweight='bold')

    # 处置动作
    action_rect = FancyBboxPatch((7.5, 2.7), 2.2, 0.9, boxstyle="round,pad=0.05",
                                  facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
    ax.add_patch(action_rect)
    ax.text(8.6, 3.3, '4级响应', ha='center', va='center', fontsize=11, fontweight='bold', color='#C62828')
    ax.text(8.6, 2.95, '停止送料 · 抬升喷嘴', ha='center', va='center', fontsize=9, color='#C62828')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig5_filament_breakage_logic.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] fig5_filament_breakage_logic.png")


# ============================================================
# 图3：缺陷分级闭环处置流程图
# ============================================================
def plot_defect_closed_loop():
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('图5-X  缺陷分级闭环处置流程', fontsize=14, fontweight='bold', pad=20)

    # 检测输入
    detect_rect = FancyBboxPatch((4.5, 8.0), 3.0, 0.7, boxstyle="round,pad=0.05",
                                  facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(detect_rect)
    ax.text(6.0, 8.35, '多源缺陷检测', ha='center', va='center', fontsize=12, fontweight='bold', color='#1565C0')

    # 分级判定菱形
    diamond = plt.Polygon([[6.0, 7.2], [7.5, 6.5], [6.0, 5.8], [4.5, 6.5]],
                           facecolor='#FFF8E1', edgecolor='#F57C00', linewidth=2)
    ax.add_patch(diamond)
    ax.text(6.0, 6.5, '缺陷严重程度' + chr(10) + '分级判定', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(6.0, 7.2), xytext=(6.0, 8.0),
               arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # 四个等级分支
    levels = [
        ('1级' + chr(10) + '轻微偏差', '限幅内自动补偿', 1.0, 4.8, '#E8F5E9', '#2E7D32', 0.3),
        ('2级' + chr(10) + '几何偏差', '工艺参数优化' + chr(10) + '限幅补偿', 3.5, 4.8, '#E3F2FD', '#1565C0', 0.5),
        ('3级' + chr(10) + '浸润风险', '风险热力图标记' + chr(10) + '层间再加热补偿', 6.5, 4.8, '#FFF3E0', '#E65100', 0.5),
        ('4级' + chr(10) + '严重故障', '安全停机保护' + chr(10) + '停止送料·抬升喷嘴', 9.5, 4.8, '#FFEBEE', '#C62828', 0.5),
    ]

    for level, action, x, y, facecolor, edgecolor, severity in levels:
        rect = FancyBboxPatch((x - 1.1, y + 0.3), 2.2, 0.8, boxstyle="round,pad=0.05",
                               facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y + 0.7, level, ha='center', va='center', fontsize=10, fontweight='bold', color=edgecolor)
        ax.annotate('', xy=(x, y + 1.1), xytext=(6.0, 5.8),
                   arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2,
                                  connectionstyle="arc3,rad=" + str(severity)))

        action_rect = FancyBboxPatch((x - 1.1, y - 1.0), 2.2, 1.0, boxstyle="round,pad=0.05",
                                      facecolor='#FAFAFA', edgecolor=edgecolor, linewidth=1.2, linestyle='--')
        ax.add_patch(action_rect)
        ax.text(x, y - 0.5, action, ha='center', va='center', fontsize=9, color='#333333')
        ax.annotate('', xy=(x, 2.8), xytext=(x, y - 1.0),
                   arrowprops=dict(arrowstyle='->', color='#666666', lw=1.0))

    # 质量追溯数据库
    trace_rect = FancyBboxPatch((2.0, 1.8), 8.0, 1.0, boxstyle="round,pad=0.1",
                                 facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=2)
    ax.add_patch(trace_rect)
    ax.text(6.0, 2.55, '质量追溯数据库', ha='center', va='center', fontsize=12, fontweight='bold', color='#6A1B9A')
    ax.text(6.0, 2.15, '缺陷记录 · 时间戳 · 构件坐标 · 层号 · 可复现档案',
            ha='center', va='center', fontsize=9, color='#555555')

    # 闭环反馈
    ax.annotate('', xy=(1.5, 8.35), xytext=(1.5, 2.3),
               arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.5,
                              connectionstyle="arc3,rad=-0.3"))
    ax.text(0.8, 5.5, '闭环反馈', ha='center', va='center', fontsize=10,
            color='#6A1B9A', fontweight='bold', rotation=90)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig5_defect_closed_loop.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] fig5_defect_closed_loop.png")


# ============================================================
# 图4：红外热像温度异常检测（三子图）
# ============================================================
def plot_thermal_inspection():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('图5-X  纤维浸润质量热辐射评估', fontsize=14, fontweight='bold', y=1.02)

    np.random.seed(42)
    x = np.linspace(0, 10, 200)
    y = np.linspace(0, 10, 200)
    X, Y = np.meshgrid(x, y)

    # (a) 正常浸润
    T_normal = 180 + 5 * np.sin(X / 2) * np.cos(Y / 2) + np.random.normal(0, 0.5, X.shape)
    T_normal = np.clip(T_normal, 175, 190)
    im1 = axes[0].imshow(T_normal, extent=[0, 10, 0, 10], origin='lower', cmap='coolwarm', vmin=170, vmax=200)
    axes[0].set_title('(a) 正常浸润区域', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('X / mm')
    axes[0].set_ylabel('Y / mm')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04).set_label('温度 / °C', fontsize=9)

    # (b) 异常区域
    T_abnormal = T_normal.copy()
    mask1 = (X - 3)**2 + (Y - 7)**2 < 2.5
    T_abnormal[mask1] -= 8 + np.random.normal(0, 1, T_abnormal[mask1].shape)
    mask2 = (X - 7)**2 + (Y - 3)**2 < 2.0
    T_abnormal[mask2] += 7 + np.random.normal(0, 1, T_abnormal[mask2].shape)
    T_abnormal = np.clip(T_abnormal, 165, 205)

    im2 = axes[1].imshow(T_abnormal, extent=[0, 10, 0, 10], origin='lower', cmap='coolwarm', vmin=170, vmax=200)
    axes[1].set_title('(b) 浸润异常区域', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('X / mm')
    axes[1].set_ylabel('Y / mm')
    axes[1].annotate('干斑' + chr(10) + '(过冷)', xy=(3, 7), xytext=(1, 9),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    fontsize=9, color='blue', fontweight='bold')
    axes[1].annotate('树脂富集' + chr(10) + '(过热)', xy=(7, 3), xytext=(8.5, 1.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=9, color='red', fontweight='bold')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04).set_label('温度 / °C', fontsize=9)

    # (c) 背景减除异常检测
    T_diff = T_abnormal - T_normal
    T_diff = np.clip(T_diff, -12, 12)
    im3 = axes[2].imshow(T_diff, extent=[0, 10, 0, 10], origin='lower', cmap='RdBu_r', vmin=-10, vmax=10)
    axes[2].set_title('(c) 温度梯度异常检测', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('X / mm')
    axes[2].set_ylabel('Y / mm')
    axes[2].contour(X, Y, T_diff, levels=[-5, 5], colors=['blue', 'red'], linewidths=2)
    axes[2].text(0.5, 9.3, '阈值: ±5°C', fontsize=9, color='black', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04).set_label('温度偏差 / °C', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig5_thermal_inspection.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] fig5_thermal_inspection.png")


# ============================================================
# 图5：轨迹形貌视觉检测指标
# ============================================================
def plot_visual_inspection():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('图5-X  轨迹形貌视觉检测指标', fontsize=14, fontweight='bold', y=1.02)

    # (a) 铺放偏移与宽度偏差
    ax1 = axes[0]
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 8)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a) 铺放偏移与宽度偏差检测', fontsize=11, fontweight='bold')

    theta = np.linspace(0, 2 * np.pi, 100)
    r = 3.0
    theory_x = 6 + r * np.cos(theta)
    theory_y = 4 + r * np.sin(theta)
    ax1.plot(theory_x, theory_y, 'k--', linewidth=1.5, label='理论路径')

    r_inner = 2.7 + 0.2 * np.sin(3 * theta)
    r_outer = 3.3 + 0.2 * np.sin(3 * theta)
    aix = 6.3 + r_inner * np.cos(theta)
    aiy = 4.2 + r_inner * np.sin(theta)
    aox = 6.3 + r_outer * np.cos(theta)
    aoy = 4.2 + r_outer * np.sin(theta)

    ax1.fill(np.concatenate([aox, aix[::-1]]),
             np.concatenate([aoy, aiy[::-1]]),
             color='#1565C0', alpha=0.4, label='实际轨迹')
    ax1.plot(aix, aiy, 'b-', linewidth=1.2)
    ax1.plot(aox, aoy, 'b-', linewidth=1.2)

    ax1.annotate('', xy=(6.3, 4.2), xytext=(6, 4),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax1.text(6.15, 4.1, 'Δd', fontsize=11, color='red', fontweight='bold')

    angle_idx = 25
    ax1.annotate('', xy=(aox[angle_idx], aoy[angle_idx]),
                xytext=(aix[angle_idx], aiy[angle_idx]),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax1.text(aox[angle_idx] + 0.3, aoy[angle_idx], 'w', fontsize=11, color='green', fontweight='bold')

    ax1.text(1, 7.2, '判定阈值：', fontsize=10, fontweight='bold')
    ax1.text(1, 6.7, '· 中心偏移 > 0.3 mm -> 触发补偿', fontsize=9, color='#C62828')
    ax1.text(1, 6.3, '· 宽度偏差 > 10% -> 触发补偿', fontsize=9, color='#C62828')
    ax1.text(1, 5.9, '· 连续3帧异常 -> 确认报警', fontsize=9, color='#C62828')
    ax1.legend(loc='lower right', fontsize=9)

    # (b) 成型厚度轮廓三维重建
    ax2 = axes[1]
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 8)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('(b) 成型厚度轮廓三维重建', fontsize=11, fontweight='bold')

    base = Rectangle((1, 1), 10, 0.5, facecolor='#BDBDBD', edgecolor='#424242', linewidth=1.5)
    ax2.add_patch(base)
    ax2.text(6, 0.7, '打印基板', ha='center', va='center', fontsize=9, color='#424242')

    tpx = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    tpy = np.array([1.5, 2.0, 2.2, 2.1, 2.0, 2.1, 2.2, 2.0, 1.5])
    ax2.plot(tpx, tpy, 'k--', linewidth=1.5, label='理论轮廓')
    ax2.fill_between(tpx, 1.5, tpy, alpha=0.1, color='black')

    apx = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    apy = np.array([1.5, 2.3, 2.8, 2.4, 2.2, 2.5, 2.9, 2.3, 1.5])
    ax2.plot(apx, apy, 'b-', linewidth=2, label='实测轮廓')
    ax2.fill_between(apx, 1.5, apy, alpha=0.3, color='#1565C0')

    for xi, yi in zip([3.5, 6.5, 9], [2.55, 2.1, 2.4]):
        ax2.annotate('', xy=(xi, yi), xytext=(xi, 1.5),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
        ax2.text(xi + 0.15, (yi + 1.5) / 2, 'h', fontsize=10, color='red', fontweight='bold')

    for xi in [3, 5, 7, 9]:
        idx = list(apx).index(xi)
        ax2.plot([xi, xi], [apy[idx] + 0.3, 5.5], 'g-', linewidth=0.8, alpha=0.6)
        ax2.plot(xi, 5.5, 'go', markersize=4)
    ax2.text(6, 5.8, '激光位移传感器扫描线', ha='center', va='center', fontsize=9, color='green')
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}fig5_visual_inspection.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("[OK] fig5_visual_inspection.png")


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("第五章 5.4节 配图生成开始")
    print("=" * 50)
    plot_sensor_architecture()
    plot_filament_breakage_logic()
    plot_defect_closed_loop()
    plot_thermal_inspection()
    plot_visual_inspection()
    print("=" * 50)
    print("全部5张图生成完毕，输出目录: " + OUTPUT_DIR)
    print("=" * 50)