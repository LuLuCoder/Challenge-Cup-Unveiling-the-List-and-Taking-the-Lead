
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图5-X  纤维浸润质量热辐射评估（红外热像风格）
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# 中文字体自动检测
# ============================================================
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
    for f in available:
        if any(k in f for k in ['CJK', 'SC', 'CN', 'Hei', 'Song', 'Kai', 'Fang']):
            plt.rcParams['font.family'] = f
            print(f"[INFO] 回退字体: {f}")
            break

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

OUTPUT_DIR = './'

# ============================================================
# 生成红外热像数据
# ============================================================
np.random.seed(42)
H, W = 120, 160  # 模拟红外热像仪分辨率

def generate_fractal_noise(shape, octaves=4, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for i in range(octaves):
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, frequency, shape[1]),
            np.linspace(0, frequency, shape[0])
        )
        layer = np.sin(grid_x * np.pi * 2) * np.cos(grid_y * np.pi * 2)
        layer += 0.5 * np.sin(grid_x * np.pi * 4) * np.cos(grid_y * np.pi * 4)
        noise += layer * amplitude
        frequency *= 2
        amplitude *= persistence
    return (noise - noise.min()) / (noise.max() - noise.min())

# 正常浸润区域
base_temp = 180
T_normal = base_temp + 3 * generate_fractal_noise((H, W), octaves=5, persistence=0.4)
T_normal += np.random.normal(0, 0.3, T_normal.shape)

# 异常区域
T_abnormal = T_normal.copy()
yy, xx = np.mgrid[0:H, 0:W]

# 干斑1（过冷）
mask1 = np.exp(-((xx - 45)**2 + (yy - 35)**2) / (2 * 18**2))
T_abnormal -= mask1 * 9

# 干斑2（过冷，边缘小缺陷）
mask2 = np.exp(-((xx - 120)**2 + (yy - 85)**2) / (2 * 10**2))
T_abnormal -= mask2 * 6

# 树脂富集（过热）
mask3 = np.exp(-((xx - 95)**2 + (yy - 75)**2) / (2 * 15**2))
T_abnormal += mask3 * 8

# 扫描噪声
scan_noise = np.random.normal(0, 0.4, (H, W))
scan_noise += 0.3 * np.sin(yy[:, 0] * 0.8)[:, None] * np.ones((1, W))
T_abnormal += scan_noise

T_diff = T_abnormal - T_normal

T_normal = np.clip(T_normal, 170, 195)
T_abnormal = np.clip(T_abnormal, 168, 198)
T_diff = np.clip(T_diff, -12, 12)

# ============================================================
# 自定义红外铁红调色板 (Ironbow)
# ============================================================
ironbow_colors = [
    (0.0, 0.0, 0.0),      # 黑
    (0.2, 0.0, 0.4),      # 深紫
    (0.6, 0.0, 0.2),      # 深红
    (1.0, 0.0, 0.0),      # 红
    (1.0, 0.4, 0.0),      # 橙红
    (1.0, 0.8, 0.0),      # 黄
    (1.0, 1.0, 0.6),      # 浅黄
    (1.0, 1.0, 1.0),      # 白
]
ironbow_cmap = LinearSegmentedColormap.from_list('ironbow', ironbow_colors)

# ============================================================
# 绘图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
fig.suptitle('图5-X  纤维浸润质量热辐射评估（红外热像）', fontsize=14, fontweight='bold', y=1.02)

# (a) 正常浸润
im1 = axes[0].imshow(T_normal, cmap=ironbow_cmap, vmin=172, vmax=192, interpolation='nearest')
axes[0].set_title('(a) 正常浸润区域', fontsize=11, fontweight='bold', pad=8)
axes[0].set_xlabel('像素 X')
axes[0].set_ylabel('像素 Y')
axes[0].text(5, 8, f'MAX: {T_normal.max():.1f}°C\nMIN: {T_normal.min():.1f}°C',
            fontsize=8, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
axes[0].axhline(H // 2, color='lime', linewidth=0.5, alpha=0.5)
axes[0].axvline(W // 2, color='lime', linewidth=0.5, alpha=0.5)
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04).set_label('温度 / °C', fontsize=9)

# (b) 异常区域
im2 = axes[1].imshow(T_abnormal, cmap=ironbow_cmap, vmin=172, vmax=192, interpolation='nearest')
axes[1].set_title('(b) 浸润异常区域', fontsize=11, fontweight='bold', pad=8)
axes[1].set_xlabel('像素 X')
axes[1].set_ylabel('像素 Y')
axes[1].annotate('干斑\n(过冷)', xy=(45, 35), xytext=(20, 15),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5),
                fontsize=9, color='cyan', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
axes[1].annotate('树脂富集\n(过热)', xy=(95, 75), xytext=(120, 50),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5),
                fontsize=9, color='yellow', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))
axes[1].text(5, 8, f'MAX: {T_abnormal.max():.1f}°C\nMIN: {T_abnormal.min():.1f}°C',
            fontsize=8, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
axes[1].axhline(H // 2, color='lime', linewidth=0.5, alpha=0.5)
axes[1].axvline(W // 2, color='lime', linewidth=0.5, alpha=0.5)
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04).set_label('温度 / °C', fontsize=9)

# (c) 差分检测
diff_cmap = LinearSegmentedColormap.from_list('thermal_diff',
    ['#00008B', '#4169E1', '#87CEEB', '#F0F0F0', '#FFD700', '#FF4500', '#8B0000'])
im3 = axes[2].imshow(T_diff, cmap=diff_cmap, vmin=-10, vmax=10, interpolation='nearest')
axes[2].set_title('(c) 背景减除异常检测', fontsize=11, fontweight='bold', pad=8)
axes[2].set_xlabel('像素 X')
axes[2].set_ylabel('像素 Y')
axes[2].contour(xx, yy, T_diff, levels=[-5, 5], colors=['cyan', 'yellow'], linewidths=1.5)
axes[2].contourf(xx, yy, T_diff, levels=[-10, -5], colors=['cyan'], alpha=0.15)
axes[2].contourf(xx, yy, T_diff, levels=[5, 10], colors=['yellow'], alpha=0.15)
axes[2].text(5, 8, '阈值: ±5°C\n蓝色: 干斑\n黄色: 树脂富集',
            fontsize=8, color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
axes[2].axhline(H // 2, color='white', linewidth=0.5, alpha=0.4)
axes[2].axvline(W // 2, color='white', linewidth=0.5, alpha=0.4)
fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04).set_label('温度偏差 / °C', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fig5_thermal_inspection.png', bbox_inches='tight',
            facecolor='white', edgecolor='none', dpi=300)
plt.show()
print("[OK] 红外热像图已生成")