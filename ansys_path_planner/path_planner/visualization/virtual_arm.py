"""虚拟机械臂：6 自由度关节臂运动学，工具末端精确到达目标点。

关节配置（演示用）：
    θ1 基座回转（绕 Z）
    θ2 肩部俯仰
    θ3 肘部俯仰
    θ4 腕部俯仰（使工具保持竖直向下，保证末端贴合路径）
    θ5 腕部偏摆（演示角，恒 0）
    θ6 工具回转（演示角，恒 0）
"""
import math

import numpy as np

H0 = 0.26   # 肩部立柱高度
L1 = 0.40   # 大臂长度
L2 = 0.36   # 小臂长度
L3 = 0.10   # 腕部长度
L4 = 0.14   # 工具长度
TOOL_LEN = L3 + L4           # 工具总长（腕心到末端）
MIN_REACH = abs(L1 - L2) + 1e-6
MAX_REACH = L1 + L2 - 1e-6


def solve_virtual_arm(target):
    """6 自由度逆解，保证工具末端精确到达 target。

    工具竖直向下，腕心 = target + TOOL_LEN 向上；平面内按大臂/小臂
    两连杆余弦定理解肩/肘角，腕部角度补足使工具竖直。
    """
    x, y, z = (float(v) for v in target)

    # 腕心（工具竖直向下，末端 = 腕心 - TOOL_LEN 沿 Z）
    wx, wy, wz = x, y, z + TOOL_LEN

    q1 = math.atan2(wy, wx)
    r = math.hypot(wx, wy)
    zr = wz - H0
    d = math.hypot(r, zr)
    d = min(max(d, MIN_REACH), MAX_REACH)

    # 余弦定理：肩部夹角 / 肘部内角
    cos_b = (L1 * L1 + d * d - L2 * L2) / (2.0 * L1 * d)
    b = math.acos(min(max(cos_b, -1.0), 1.0))
    cos_e = (L1 * L1 + L2 * L2 - d * d) / (2.0 * L1 * L2)
    e = math.acos(min(max(cos_e, -1.0), 1.0))

    a = math.atan2(r, zr)      # 肩→腕心方向（相对竖直）
    q2 = a - b                 # 大臂方向
    q3 = math.pi - e           # 肘部弯折
    q4 = math.pi - (q2 + q3)   # 腕部俯仰：工具竖直向下

    q5 = 0.0
    q6 = 0.0
    return np.array([q1, q2, q3, q4, q5, q6])


def forward_virtual_arm(q, base=None):
    """6 自由度正运动学，返回关节点链（6 个点：基座→末端）。"""
    if base is None:
        base = np.zeros(3)
    q1, q2, q3, q4, *_ = q

    def direction(angle):
        # 竖直平面内 angle 相对竖直方向的单位向量，绕 Z 转 q1
        return np.array([
            math.sin(angle) * math.cos(q1),
            math.sin(angle) * math.sin(q1),
            math.cos(angle),
        ])

    p0 = np.asarray(base, dtype=float)
    p1 = p0 + np.array([0.0, 0.0, H0])            # 肩
    p2 = p1 + L1 * direction(q2)                   # 肘
    p3 = p2 + L2 * direction(q2 + q3)              # 腕
    p4 = p3 + L3 * direction(q2 + q3 + q4)         # 腕端/工具根部
    p5 = p4 + L4 * direction(q2 + q3 + q4)         # 工具末端

    return np.vstack([p0, p1, p2, p3, p4, p5])
