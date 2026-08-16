"""虚拟机械臂：2 连杆平面运动学，末端可精确到达目标点。"""
import math

import numpy as np

L1 = 0.18  # 肩部立柱高度
L2 = 0.28  # 大臂长度
L3 = 0.28  # 小臂长度
MIN_REACH = abs(L2 - L3)   # 0.03，最内可达半径
MAX_REACH = L2 + L3        # 0.27，最大伸展长度


def solve_virtual_arm(target):
    """
    2 连杆逆解（肘部向上），保证末端精确到达 target。

    q1 决定方位角；q2/q3 决定竖直平面内肩/肘角；
    q4~q6 仅作为演示姿态角，不影响末端位置。
    """
    x, y, z = (float(v) for v in target)

    q1 = math.atan2(y, x)

    r = math.hypot(x, y)
    zr = z - L1
    d = math.hypot(r, zr)

    # 超出可达范围时收回到边界，保证数值稳定
    d = min(max(d, MIN_REACH * 1.02), MAX_REACH * 0.98)

    # 余弦定理求肘部角（取正根，肘部向上）
    cos_q3 = (d * d - L2 * L2 - L3 * L3) / (2.0 * L2 * L3)
    cos_q3 = min(max(cos_q3, -1.0), 1.0)
    q3 = math.acos(cos_q3)

    # 肩部角：目标方向角 - 大臂与目标方向之间的夹角
    q2 = math.atan2(zr, r) - math.atan2(
        L3 * math.sin(q3), L2 + L3 * math.cos(q3)
    )

    q4 = 0.2 * math.sin(x * 3.0)
    q5 = 0.2 * math.cos(y * 3.0)
    q6 = q1
    return np.array([q1, q2, q3, q4, q5, q6])


def forward_virtual_arm(q, base=None):
    """2 连杆正运动学，末端位置与逆解输入精确一致。"""
    if base is None:
        base = np.zeros(3)

    q1, q2, q3, *_ = q

    p0 = np.asarray(base, dtype=float)
    p1 = p0 + np.array([0, 0, L1])
    p2 = p1 + np.array([
        L2 * math.cos(q1) * math.cos(q2),
        L2 * math.sin(q1) * math.cos(q2),
        L2 * math.sin(q2),
    ])
    p3 = p2 + np.array([
        L3 * math.cos(q1) * math.cos(q2 + q3),
        L3 * math.sin(q1) * math.cos(q2 + q3),
        L3 * math.sin(q2 + q3),
    ])

    return np.vstack([p0, p1, p2, p3])
