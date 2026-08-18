"""STEP 文件 -> 零件点云（轻量实现，不依赖任何 CAD 库）。

解析 ISO 10303-21（AP203/AP214）B-rep 实体：
    - 收集所有 CARTESIAN_POINT（顶点/控制点）作为基础点集；
    - 对 ADVANCED_FACE 的解析曲面（平面/圆柱/球/圆锥/圆环），
      用其边界顶点估计参数域并均匀采样，得到零件表面点云；
    - 解析失败或无面数据时退化为顶点点集。
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


MAX_POINTS = 60000        # 输出点云数量上限
FACE_SAMPLE_MAX = 1200    # 单个面的采样点数上限

_ENTITY_RE = re.compile(r"#\s*(\d+)\s*=\s*([A-Z_0-9]+)\s*\(")
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")


# ---------- 文本 / 实体切分 ----------

def _read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin1", errors="replace")


def _tokenize(text):
    """返回 {实体 id: (类型, 参数原始串)}。"""
    entities = {}
    for m in _ENTITY_RE.finditer(text):
        eid = int(m.group(1))
        etype = m.group(2)
        # 从 '(' 开始，括号配平后遇到 ';' 结束
        i = m.end() - 1
        depth = 0
        start = i
        end = -1
        while i < len(text):
            c = text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
            i += 1
        if end < 0:
            continue
        j = end + 1
        while j < len(text) and text[j] not in ";\r\n":
            j += 1
        entities[eid] = (etype, text[start:end + 1])
    return entities


# ---------- 参数解析 ----------

def _parse_value(s, i):
    """解析一个 STEP 值，返回 (值, 下一个下标)。"""
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    if i >= len(s):
        return None, i
    c = s[i]
    if c == "'":  # 字符串（'' 转义）
        j = i + 1
        buf = []
        while j < len(s):
            if s[j] == "'":
                if j + 1 < len(s) and s[j + 1] == "'":
                    buf.append("'")
                    j += 2
                else:
                    break
            else:
                buf.append(s[j])
                j += 1
        return "".join(buf), j + 1
    if c == "#":  # 引用
        j = i + 1
        while j < len(s) and s[j].isdigit():
            j += 1
        return int(s[i + 1:j]), j
    if c == "(":  # 列表
        items = []
        j = i + 1
        while True:
            while j < len(s) and s[j] in " \t\r\n,":
                j += 1
            if j < len(s) and s[j] == ")":
                j += 1
                break
            v, j = _parse_value(s, j)
            items.append(v)
        return items, j
    if c == ".":  # 枚举
        j = s.find(".", i + 1)
        return s[i:j + 1], j + 1
    if c == "$" or c == "*":
        return None, i + 1
    m = _NUMBER_RE.match(s, i)
    if m:
        return float(m.group(0)), m.end()
    m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", s[i:])
    if m:
        return m.group(0), i + m.end()
    return None, i + 1


def _parse_params(raw):
    raw = raw.strip()
    if not raw.startswith("("):
        raise ValueError("STEP 参数必须以 ( 开头")
    vals, _ = _parse_value(raw, 0)
    return vals


# ---------- 实体解析 ----------

def _as_points(values):
    """把 (x,y,z) 元组转成 float 数组；失败返回 None。"""
    try:
        return np.array([float(values[0]), float(values[1]), float(values[2])])
    except (TypeError, ValueError, IndexError):
        return None


def _direction(values):
    v = _as_points(values)
    if v is None:
        return None
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def _placement_from(loc_vals, axis_vals, ref_vals):
    """由已解析的 CARTESIAN_POINT / DIRECTION 参数构造 (origin,x,y,z)。"""
    origin = _as_points(loc_vals[1]) if loc_vals and len(loc_vals) > 1 else None
    z = _direction(axis_vals[1]) if axis_vals and len(axis_vals) > 1 else None
    x = _direction(ref_vals[1]) if ref_vals and len(ref_vals) > 1 else None
    if origin is None:
        return None
    if z is None:
        z = np.array([0.0, 0.0, 1.0])
    if x is None:
        x = (np.array([1.0, 0.0, 0.0]) if abs(z[2]) < 0.999
             else np.array([0.0, 1.0, 0.0]))
    x = x - np.dot(x, z) * z
    xn = float(np.linalg.norm(x))
    if xn < 1e-12:
        return None
    x = x / xn
    y = np.cross(z, x)
    return origin, x, y, z


def _collect_loops(face_vals, entities, parsed, cp_map):
    """解析 ADVANCED_FACE 的边界环，返回 [(outer, 顶点列表, 边曲线列表)]。"""
    loops = []

    def resolve(ref):
        if isinstance(ref, int):
            if ref in parsed:
                return parsed[ref]
            if ref not in entities:
                return None
            typ, raw = entities[ref]
            parsed[ref] = (typ, _parse_params(raw))
            return parsed[ref]
        return None

    def vertex_point(vals):
        # VERTEX_POINT(名字, #CARTESIAN_POINT)
        ref = vals[1] if vals and len(vals) > 1 and isinstance(vals[1], int) else None
        return cp_map.get(ref)

    # ADVANCED_FACE(名字, (#FACE_BOUND...), #曲面, 方向)
    bounds = face_vals[1] if len(face_vals) > 1 and isinstance(face_vals[1], list) else []
    for b in bounds:
        if not isinstance(b, int):
            continue
        bv = resolve(b)
        if not bv or not isinstance(bv[0], str):
            continue
        btype, bvals = bv
        if btype not in ("FACE_BOUND", "FACE_OUTER_BOUND"):
            continue
        loop_ref = bvals[1] if len(bvals) > 1 and isinstance(bvals[1], int) else None
        lv = resolve(loop_ref)
        if not lv or lv[0] != "EDGE_LOOP":
            continue
        edge_refs = lv[1][1] if len(lv[1]) > 1 and isinstance(lv[1][1], list) else []
        pts = []
        edges = []
        for oe in edge_refs:
            if not isinstance(oe, int):
                continue
            ov = resolve(oe)
            if not ov or ov[0] != "ORIENTED_EDGE":
                continue
            edge_ref = ov[1][3] if len(ov[1]) > 3 else None
            ev = resolve(edge_ref)
            if not ev or ev[0] != "EDGE_CURVE":
                continue
            curve_ref = ev[1][3] if len(ev[1]) > 3 else None
            cv = resolve(curve_ref)
            v1 = resolve(ev[1][1]) if len(ev[1]) > 1 else None
            v2 = resolve(ev[1][2]) if len(ev[1]) > 2 else None
            p1 = vertex_point(v1[1]) if v1 and v1[0] == "VERTEX_POINT" else None
            p2 = vertex_point(v2[1]) if v2 and v2[0] == "VERTEX_POINT" else None
            if p1 is not None:
                pts.append(p1)
            edges.append((cv[0] if cv else None, cv[1] if cv else None, p1, p2))
        if pts:
            loops.append({
                "outer": btype == "FACE_OUTER_BOUND",
                "points": pts,
                "edges": edges,
            })
    return loops


def _point_in_polygon(x, y, poly):
    """射线法判断点是否在多边形内。"""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x2 - (x2 - x1) * (y2 - y) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def _sample_plane_mesh(placement, loops, fallback_boundary):
    """平面面网格化：外环内、内环（孔）外，均匀取网格节点。"""
    origin, x, y, z = placement

    def to_uv(p):
        return (float(np.dot(p - origin, x)), float(np.dot(p - origin, y)))

    outer_pts = next(
        (loop["points"] for loop in loops if loop["outer"]),
        fallback_boundary,
    )
    outer_poly = [to_uv(p) for p in outer_pts]
    if len(outer_poly) < 3:
        return []
    inner_polys = [
        [to_uv(p) for p in loop["points"]]
        for loop in loops if not loop["outer"]
    ]
    inner_polys = [p for p in inner_polys if len(p) >= 3]

    us = [p[0] for p in outer_poly]
    vs = [p[1] for p in outer_poly]
    u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
    if abs(u1 - u0) < 1e-9 and abs(v1 - v0) < 1e-9:
        return []
    nu = int(math.ceil(math.sqrt(FACE_SAMPLE_MAX * max(abs(u1 - u0), 1e-9)
                                 / max(abs(u1 - u0) + abs(v1 - v0), 1e-9))))
    nu = max(2, min(nu, 80))
    nv = max(2, min(FACE_SAMPLE_MAX // max(nu, 1), 80))

    pts = []
    for i in range(nu):
        for j in range(nv):
            u = u0 + (u1 - u0) * i / (nu - 1)
            v = v0 + (v1 - v0) * j / (nv - 1)
            if not _point_in_polygon(u, v, outer_poly):
                continue
            if any(_point_in_polygon(u, v, poly) for poly in inner_polys):
                continue  # 孔内不取点
            pts.append(origin + u * x + v * y)
    return pts


def _sample_cylinder(placement, radius, boundary):
    origin, x, y, z = placement
    if radius <= 0:
        return []
    angles = []
    heights = []
    for p in boundary:
        d = p - origin
        angles.append(math.atan2(float(np.dot(d, y)), float(np.dot(d, x))))
        heights.append(float(np.dot(d, z)))
    angles = sorted(angles)
    unwrapped = np.unwrap(np.array(angles))
    a0, a1 = float(unwrapped.min()), float(unwrapped.max())
    h0, h1 = min(heights), max(heights)
    span = a1 - a0
    if span <= 1e-9 and h1 - h0 <= 1e-9:
        return []
    na = max(2, min(int(math.ceil(40 * span / (2 * math.pi))), 90))
    nh = max(2, min(int(math.ceil(math.sqrt(FACE_SAMPLE_MAX / max(na, 1)))), 40))
    pts = []
    for i in range(na):
        for j in range(nh):
            a = a0 + span * i / (na - 1)
            h = h0 + (h1 - h0) * j / (nh - 1)
            pts.append(origin + radius * (math.cos(a) * x + math.sin(a) * y) + h * z)
    return pts


def _circle_arc_points(cvals, p1, p2, entities, parsed):
    """CIRCLE 边两点之间的短弧折线；失败返回 None。"""
    def resolve(ref):
        if isinstance(ref, int):
            if ref in parsed:
                return parsed[ref]
            if ref not in entities:
                return None
            typ, raw = entities[ref]
            parsed[ref] = (typ, _parse_params(raw))
            return parsed[ref]
        return None

    try:
        av = resolve(cvals[1])
        if not av or av[0] != "AXIS2_PLACEMENT_3D" or not av[1]:
            return None
        pvals = av[1]
        loc = resolve(pvals[1]) if len(pvals) > 1 else None
        axis = resolve(pvals[2]) if len(pvals) > 2 else None
        refd = resolve(pvals[3]) if len(pvals) > 3 else None
        origin = _as_points(loc[1]) if loc else None
        zaxis = _direction(axis[1]) if axis else None
        xaxis = _direction(refd[1]) if refd else None
        radius = float(cvals[2]) if len(cvals) > 2 else 0.0
        if origin is None or radius <= 0:
            return None
        if zaxis is None:
            zaxis = np.array([0.0, 0.0, 1.0])
        if xaxis is None:
            xaxis = (np.array([1.0, 0.0, 0.0]) if abs(zaxis[2]) < 0.999
                     else np.array([0.0, 1.0, 0.0]))
        xaxis = xaxis - np.dot(xaxis, zaxis) * zaxis
        xn = float(np.linalg.norm(xaxis))
        if xn < 1e-12:
            return None
        xaxis = xaxis / xn
        yaxis = np.cross(zaxis, xaxis)

        def _angle(p):
            d = p - origin
            return math.atan2(float(np.dot(d, yaxis)), float(np.dot(d, xaxis)))

        a1 = _angle(p1)
        a2 = _angle(p2)
        delta = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi  # 短弧
        n = max(6, int(abs(delta) / (2 * math.pi) * 36))
        arc = []
        for k in range(n + 1):
            a = a1 + delta * k / n
            arc.append(origin + radius * (
                math.cos(a) * xaxis + math.sin(a) * yaxis
            ))
        return arc
    except Exception:
        return None


def _sample_edges(edges, entities, parsed):
    """沿边界曲线采样：CIRCLE 圆弧按段取点，其余曲线只取端点。"""
    pts = []
    for ctype, cvals, p1, p2 in edges:
        if ctype != "CIRCLE" or not cvals or p1 is None or p2 is None:
            continue
        arc = _circle_arc_points(cvals, p1, p2, entities, parsed)
        if arc:
            pts.extend(arc)
    return pts


def _sample_sphere(placement, radius, boundary):
    origin, x, y, z = placement
    if radius <= 0:
        return []
    thetas, phis = [], []
    for p in boundary:
        d = (p - origin) / radius
        thetas.append(math.acos(max(-1.0, min(1.0, float(np.dot(d, z))))))
        phis.append(math.atan2(float(np.dot(d, y)), float(np.dot(d, x))))
    phis = np.unwrap(np.sort(np.array(phis)))
    t0, t1 = min(thetas), max(thetas)
    p0, p1 = float(phis.min()), float(phis.max())
    span_t = max(t1 - t0, 0.2)
    span_p = max(p1 - p0, 0.2)
    nt = max(3, min(int(math.ceil(20 * span_t / math.pi)), 40))
    np_ = max(3, min(int(math.ceil(40 * span_p / (2 * math.pi))), 60))
    pts = []
    for i in range(nt):
        for j in range(np_):
            t = t0 + span_t * i / (nt - 1)
            p = p0 + span_p * j / (np_ - 1)
            pts.append(origin + radius * (
                math.sin(t) * math.cos(p) * x
                + math.sin(t) * math.sin(p) * y
                + math.cos(t) * z
            ))
    return pts


def _sample_cone(placement, radius, semi_angle, boundary):
    origin, x, y, z = placement
    if radius <= 0 or semi_angle <= 0:
        return []
    angles, heights = [], []
    for p in boundary:
        d = p - origin
        angles.append(math.atan2(float(np.dot(d, y)), float(np.dot(d, x))))
        heights.append(float(np.dot(d, z)))
    angles = np.unwrap(np.sort(np.array(angles)))
    a0, a1 = float(angles.min()), float(angles.max())
    h0, h1 = min(heights), max(heights)
    span = max(a1 - a0, 0.2)
    na = max(3, min(int(math.ceil(30 * span / (2 * math.pi))), 60))
    nh = max(2, min(int(math.ceil(30 * max(h1 - h0, 0.05))), 30))
    pts = []
    for i in range(na):
        for j in range(nh):
            a = a0 + span * i / (na - 1)
            h = h0 + (h1 - h0) * j / (nh - 1)
            rr = radius + h * math.tan(semi_angle)
            pts.append(origin + rr * (math.cos(a) * x + math.sin(a) * y) + h * z)
    return pts


def _sample_torus(placement, major_r, minor_r, boundary):
    origin, x, y, z = placement
    if major_r <= 0 or minor_r <= 0:
        return []
    us, vs = [], []
    for p in boundary:
        d = p - origin
        r = math.hypot(float(np.dot(d, x)), float(np.dot(d, y)))
        us.append(math.atan2(float(np.dot(d, z)), r - major_r))
        vs.append(math.atan2(float(np.dot(d, y)), float(np.dot(d, x))))
    us = np.unwrap(np.sort(np.array(us)))
    vs = np.unwrap(np.sort(np.array(vs)))
    u0, u1 = float(us.min()), float(us.max())
    v0, v1 = float(vs.min()), float(vs.max())
    nu = max(3, min(int(math.ceil(24 * max(u1 - u0, 0.2) / math.pi)), 36))
    nv = max(3, min(int(math.ceil(24 * max(v1 - v0, 0.2) / math.pi)), 36))
    pts = []
    for i in range(nu):
        for j in range(nv):
            u = u0 + (u1 - u0) * i / (nu - 1)
            v = v0 + (v1 - v0) * j / (nv - 1)
            rr = major_r + minor_r * math.cos(u)
            pts.append(origin + rr * (math.cos(v) * x + math.sin(v) * y)
                       + minor_r * math.sin(u) * z)
    return pts


# ---------- 主入口 ----------

def parse_step_points(path, max_points=MAX_POINTS):
    """把 STEP 文件解析为 (N,3) 点云。"""
    text = _read_text(path)
    entities = _tokenize(text)
    if not entities:
        raise ValueError("未在 STEP 文件中识别到任何实体。")

    parsed = {}

    def resolve(ref):
        if isinstance(ref, int):
            if ref in parsed:
                return parsed[ref]
            if ref not in entities:
                return None
            typ, raw = entities[ref]
            try:
                parsed[ref] = (typ, _parse_params(raw))
            except Exception:
                parsed[ref] = (typ, None)
            return parsed[ref]
        return None

    # 1) 收集全部 CARTESIAN_POINT
    cp_map = {}
    for eid, (typ, raw) in entities.items():
        if typ != "CARTESIAN_POINT":
            continue
        try:
            vals = _parse_params(raw)
        except Exception:
            continue
        coords = vals[1] if len(vals) > 1 and isinstance(vals[1], (tuple, list)) else None
        p = _as_points(coords)
        if p is not None:
            cp_map[eid] = p

    all_points = list(cp_map.values())

    # 2) 解析曲面并按面采样
    face_ids = [eid for eid, (typ, _r) in entities.items() if typ == "ADVANCED_FACE"]
    sampled = []
    for eid in face_ids:
        vals = resolve(eid)
        if not vals or vals[0] != "ADVANCED_FACE" or not vals[1]:
            continue
        surf_ref = vals[1][2] if len(vals[1]) > 2 else None
        sv = resolve(surf_ref)
        if not sv or not sv[1]:
            continue
        stype, svalues = sv
        if stype not in ("PLANE", "CYLINDRICAL_SURFACE", "SPHERICAL_SURFACE",
                         "CONICAL_SURFACE", "TOROIDAL_SURFACE"):
            continue
        loops = _collect_loops(vals[1], entities, parsed, cp_map)
        boundary = [p for loop in loops for p in loop["points"]]
        if not boundary:
            continue
        placement = None
        axis2_ref = (
            svalues[1] if len(svalues) > 1 and isinstance(svalues[1], int)
            else None
        )
        av = resolve(axis2_ref)
        if av and av[0] == "AXIS2_PLACEMENT_3D" and isinstance(av[1], list):
            pvals = av[1]

            def _ref_vals(idx):
                r = pvals[idx] if len(pvals) > idx and isinstance(pvals[idx], int) else None
                rv = resolve(r)
                return rv[1] if rv else None

            placement = _placement_from(
                _ref_vals(1), _ref_vals(2), _ref_vals(3)
            )
        if placement is None:
            continue
        edge_points = _sample_edges(
            [e for loop in loops for e in loop["edges"]],
            entities, parsed,
        )
        try:
            if stype == "PLANE":
                pts = _sample_plane_mesh(placement, loops, boundary)
            elif stype == "CYLINDRICAL_SURFACE":
                radius = float(svalues[1]) if len(svalues) > 1 else 0.0
                pts = _sample_cylinder(placement, radius, boundary)
            elif stype == "SPHERICAL_SURFACE":
                radius = float(svalues[1]) if len(svalues) > 1 else 0.0
                pts = _sample_sphere(placement, radius, boundary)
            elif stype == "CONICAL_SURFACE":
                radius = float(svalues[1]) if len(svalues) > 1 else 0.0
                angle = float(svalues[2]) if len(svalues) > 2 else 0.0
                pts = _sample_cone(placement, radius, angle, boundary)
            else:
                major = float(svalues[1]) if len(svalues) > 1 else 0.0
                minor = float(svalues[2]) if len(svalues) > 2 else 0.0
                pts = _sample_torus(placement, major, minor, boundary)
        except Exception:
            pts = []
        sampled.extend(pts)
        sampled.extend(edge_points)

    if sampled:
        # 只用曲面采样点：CARTESIAN_POINT 里常混有圆心/轴点等辅助点
        # （大半径圆角圆心可在零件外数百毫米），混入会污染包围盒与形状签名。
        all_points = sampled

    if not all_points:
        raise ValueError("STEP 文件中没有提取到任何点。")

    arr = np.asarray(all_points, dtype=float)
    # 去重（6 位小数容差）
    rounded = np.round(arr, 6)
    _u, idx = np.unique(rounded, axis=0, return_index=True)
    arr = arr[np.sort(idx)]

    if len(arr) > max_points:
        step = max(1, len(arr) // max_points)
        arr = arr[::step][:max_points]
    return arr


def parse_step_file(path):
    """STEP -> DataFrame(Node, X, Y, Z)，供模板匹配与映射使用。"""
    points = parse_step_points(path)
    return pd.DataFrame({
        "Node": np.arange(1, len(points) + 1),
        "X": points[:, 0],
        "Y": points[:, 1],
        "Z": points[:, 2],
    })


# ============================================================
# 共享面解析 / 线框 / 体素四面体网格 / 质量评判
# ============================================================


def _collect_faces(path):
    """解析 STEP 中所有可处理的解析曲面面，返回 (faces, cp_map)。"""
    text = _read_text(path)
    entities = _tokenize(text)
    if not entities:
        raise ValueError("未在 STEP 文件中识别到任何实体。")

    parsed = {}

    def resolve(ref):
        if isinstance(ref, int):
            if ref in parsed:
                return parsed[ref]
            if ref not in entities:
                return None
            typ, raw = entities[ref]
            try:
                parsed[ref] = (typ, _parse_params(raw))
            except Exception:
                parsed[ref] = (typ, None)
            return parsed[ref]
        return None

    cp_map = {}
    for eid, (typ, raw) in entities.items():
        if typ != "CARTESIAN_POINT":
            continue
        try:
            vals = _parse_params(raw)
        except Exception:
            continue
        coords = vals[1] if len(vals) > 1 and isinstance(vals[1], (tuple, list)) else None
        p = _as_points(coords)
        if p is not None:
            cp_map[eid] = p

    faces = []
    for eid in [e for e, (t, _r) in entities.items() if t == "ADVANCED_FACE"]:
        vals = resolve(eid)
        if not vals or vals[0] != "ADVANCED_FACE" or not vals[1]:
            continue
        loops = _collect_loops(vals[1], entities, parsed, cp_map)
        boundary = [p for loop in loops for p in loop["points"]]
        if not boundary:
            continue
        surf_ref = vals[1][2] if len(vals[1]) > 2 else None
        sv = resolve(surf_ref)
        if not sv or not sv[1]:
            continue
        stype, svalues = sv
        if stype not in ("PLANE", "CYLINDRICAL_SURFACE", "SPHERICAL_SURFACE",
                         "CONICAL_SURFACE", "TOROIDAL_SURFACE"):
            continue
        placement = None
        axis2_ref = (
            svalues[1] if len(svalues) > 1 and isinstance(svalues[1], int)
            else None
        )
        av = resolve(axis2_ref)
        if av and av[0] == "AXIS2_PLACEMENT_3D" and isinstance(av[1], list):
            pvals = av[1]

            def _ref_vals(idx):
                r = pvals[idx] if len(pvals) > idx and isinstance(pvals[idx], int) else None
                rv = resolve(r)
                return rv[1] if rv else None

            placement = _placement_from(
                _ref_vals(1), _ref_vals(2), _ref_vals(3)
            )
        if placement is None:
            continue
        faces.append({
            "stype": stype,
            "svalues": svalues,
            "placement": placement,
            "loops": loops,
            "boundary": boundary,
        })
    return faces, cp_map


def _face_polygons(f):
    """平面面在参数域的 外环/内环 多边形。"""
    origin, x, y, z = f["placement"]

    def to_uv(p):
        return (float(np.dot(p - origin, x)), float(np.dot(p - origin, y)))

    outer = next(
        (loop["points"] for loop in f["loops"] if loop["outer"]),
        f["boundary"],
    )
    outer_poly = [to_uv(p) for p in outer]
    inner_polys = [
        [to_uv(p) for p in loop["points"]]
        for loop in f["loops"] if not loop["outer"]
    ]
    return outer_poly, [p for p in inner_polys if len(p) >= 3]


def _param_bounds_cyl(f):
    origin, x, y, z = f["placement"]
    angles, heights = [], []
    for p in f["boundary"]:
        d = p - origin
        angles.append(math.atan2(float(np.dot(d, y)), float(np.dot(d, x))))
        heights.append(float(np.dot(d, z)))
    unw = np.unwrap(np.sort(np.array(angles)))
    return float(unw.min()), float(unw.max()), min(heights), max(heights)


def _quadratic(a, b, c):
    if abs(a) < 1e-14:
        return []
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    s = math.sqrt(disc)
    return [(-b + s) / (2 * a), (-b - s) / (2 * a)]


def _ray_crossings(point, faces):
    """沿 +X 方向的射线与所有解析面求交，返回穿越次数（用于内外判定）。"""
    dvec = np.array([1.0, 0.0, 0.0])
    count = 0
    for f in faces:
        stype = f["stype"]
        origin, x, y, z = f["placement"]
        try:
            if stype == "PLANE":
                n = z
                denom = float(n[0])
                if abs(denom) < 1e-12:
                    continue
                t = float(np.dot(n, origin - point)) / denom
                if t <= 1e-9:
                    continue
                p = point + t * dvec
                u = float(np.dot(p - origin, x))
                v = float(np.dot(p - origin, y))
                outer, inners = _face_polygons(f)
                if not _point_in_polygon(u, v, outer):
                    continue
                if any(_point_in_polygon(u, v, poly) for poly in inners):
                    continue
                count += 1
            elif stype == "CYLINDRICAL_SURFACE":
                radius = float(f["svalues"][1]) if len(f["svalues"]) > 1 else 0.0
                if radius <= 0:
                    continue
                w = point - origin
                wz = float(np.dot(w, z))
                dz = float(np.dot(dvec, z))
                wr = w - wz * z
                dr = dvec - dz * z
                A = float(np.dot(dr, dr))
                B = 2.0 * float(np.dot(wr, dr))
                C = float(np.dot(wr, wr)) - radius ** 2
                a0, a1, h0, h1 = _param_bounds_cyl(f)
                for t in _quadratic(A, B, C):
                    if t <= 1e-9:
                        continue
                    p = point + t * dvec
                    ang = math.atan2(float(np.dot(p - origin, y)),
                                     float(np.dot(p - origin, x)))
                    h = float(np.dot(p - origin, z))
                    # 角度与高度是否落在面的参数域内（角度 ±2π 折叠）
                    ok = h0 - 1e-6 <= h <= h1 + 1e-6
                    if ok:
                        span = a1 - a0
                        if span >= 2 * math.pi - 1e-3:
                            ok = True
                        else:
                            mid = (a0 + a1) / 2
                            half = span / 2
                            diff = (ang - mid + math.pi) % (2 * math.pi) - math.pi
                            ok = abs(diff) <= half + 1e-6
                    if ok:
                        count += 1
            elif stype == "SPHERICAL_SURFACE":
                radius = float(f["svalues"][1]) if len(f["svalues"]) > 1 else 0.0
                if radius <= 0:
                    continue
                w = point - origin
                A = float(np.dot(dvec, dvec))
                B = 2.0 * float(np.dot(w, dvec))
                C = float(np.dot(w, w)) - radius ** 2
                for t in _quadratic(A, B, C):
                    if t <= 1e-9:
                        continue
                    count += 1
        except Exception:
            continue
    return count


def _inside_solid(point, faces):
    return _ray_crossings(point, faces) % 2 == 1


def parse_step_wireframe(path):
    """STEP -> 边界线框（折线列表），用于查看原始模型。"""
    text = _read_text(path)
    entities = _tokenize(text)
    parsed = {}

    def resolve(ref):
        if isinstance(ref, int):
            if ref in parsed:
                return parsed[ref]
            if ref not in entities:
                return None
            typ, raw = entities[ref]
            parsed[ref] = (typ, _parse_params(raw))
            return parsed[ref]
        return None

    polylines = []
    seen = set()
    for eid in [e for e, (t, _r) in entities.items() if t == "ADVANCED_FACE"]:
        vals = resolve(eid)
        if not vals or vals[0] != "ADVANCED_FACE" or not vals[1]:
            continue
        loops = _collect_loops(vals[1], entities, parsed, {})
        for loop in loops:
            for ctype, cvals, p1, p2 in loop["edges"]:
                key = (round(float(p1[0]), 5), round(float(p1[1]), 5),
                       round(float(p1[2]), 5), round(float(p2[0]), 5),
                       round(float(p2[1]), 5), round(float(p2[2]), 5))
                if key in seen:
                    continue
                seen.add(key)
                if ctype == "CIRCLE" and cvals and p1 is not None and p2 is not None:
                    arc = _circle_arc_points(cvals, p1, p2, entities, parsed)
                    if arc:
                        polylines.append(np.asarray(arc, dtype=float))
                elif p1 is not None and p2 is not None:
                    polylines.append(np.array([p1, p2], dtype=float))
    return polylines


# 每个体素剖成 5 个四面体（Freudenthal，体对角线 000-111）
_TET5_LOCAL = [
    (0, 1, 4, 7),
    (0, 2, 4, 7),
    (0, 2, 6, 7),
    (0, 3, 6, 7),
    (0, 3, 1, 7),
]
_VOXEL_OFFSETS = [
    (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
]


def _tet_quality(v, tets):
    """四面体质量：6√2·V / (Σ棱长²)^1.5，正四面体=1，范围 (0,1]。"""
    q = np.empty(len(tets), dtype=float)
    verts = v[tets]
    e = verts[:, 1:] - verts[:, :1]          # (M,3,3)
    e0, e1, e2 = e[:, 0], e[:, 1], e[:, 2]
    cross = np.cross(e1, e2)
    vol = np.abs(np.einsum("ij,ij->i", e0, cross)) / 6.0
    edges = np.stack([
        np.linalg.norm(e0, axis=1), np.linalg.norm(e1, axis=1),
        np.linalg.norm(e2, axis=1),
        np.linalg.norm(verts[:, 1] - verts[:, 2], axis=1),
        np.linalg.norm(verts[:, 1] - verts[:, 3], axis=1),
        np.linalg.norm(verts[:, 2] - verts[:, 3], axis=1),
    ], axis=1)
    l2 = np.sum(edges ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 6.0 * math.sqrt(2.0) * vol / np.power(l2, 1.5)
    q[~np.isfinite(q)] = 0.0
    return np.clip(q, 0.0, 1.0)


def parse_step_mesh(path, max_tets=40000):
    """STEP -> 体素四面体网格。

    返回 dict：
        vertices  (N,3) 四面体网格顶点
        tets      (M,4) 四面体索引
        voxel_size / grid_dims
        quality   {"min","mean","p25","verdict"}
        volume    网格总体积
    """
    faces, _ = _collect_faces(path)
    if not faces:
        raise ValueError("STEP 中没有可处理的解析曲面。")

    pts = np.asarray([p for f in faces for p in f["boundary"]], dtype=float)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    ext = hi - lo
    if ext.max() < 1e-9:
        raise ValueError("STEP 包围盒退化为一点。")

    # 体素尺寸：让总四面体数不超过预算
    target_voxels = max(64, max_tets // 5)
    voxel = ext.max() / 28.0
    while True:
        n = np.maximum(np.ceil(ext / max(voxel, 1e-12)).astype(int), 1)
        n[2] = max(int(n[2]), 1)
        if int(n[0]) * int(n[1]) * int(n[2]) <= target_voxels:
            break
        voxel *= 1.35

    n = np.maximum(n, 1)
    centers = []
    for i in range(int(n[0])):
        for j in range(int(n[1])):
            for k in range(int(n[2])):
                c = lo + (np.array([i, j, k]) + 0.5) * voxel
                if _inside_solid(c, faces):
                    centers.append((i, j, k))

    if not centers:
        raise ValueError("体素化后没有内部单元（请检查 STEP 是否闭合）。")

    vertex_map = {}
    vertices = []
    tets = []

    def add_vertex(i, j, k, dx, dy, dz):
        key = (i + dx, j + dy, k + dz)
        if key in vertex_map:
            return vertex_map[key]
        idx = len(vertices)
        vertex_map[key] = idx
        vertices.append(lo + np.array([key[0], key[1], key[2]]) * voxel)
        return idx

    for i, j, k in centers:
        local_idx = [add_vertex(i, j, k, *off) for off in _VOXEL_OFFSETS]
        for t in _TET5_LOCAL:
            tets.append([local_idx[a] for a in t])

    vertices = np.asarray(vertices, dtype=float)
    tets = np.asarray(tets, dtype=int)

    quality = _tet_quality(vertices, tets)
    mean_q = float(np.mean(quality))
    min_q = float(np.min(quality))
    p25 = float(np.percentile(quality, 25))
    if mean_q >= 0.7:
        verdict = "优（接近正四面体，适合有限元）"
    elif mean_q >= 0.5:
        verdict = "良（可接受，建议加密体素）"
    else:
        verdict = "差（单元畸变，需减小体素或检查 STEP）"

    # 顶点体积 = 各四面体体积之和
    verts = vertices[tets]
    e0 = verts[:, 1] - verts[:, 0]
    e1 = verts[:, 2] - verts[:, 0]
    e2 = verts[:, 3] - verts[:, 0]
    volume = float(np.abs(np.einsum("ij,ij->i", e0, np.cross(e1, e2))).sum() / 6.0)

    return {
        "vertices": vertices,
        "tets": tets,
        "voxel_size": float(voxel),
        "grid_dims": [int(n[0]), int(n[1]), int(n[2])],
        "quality": {
            "min": round(min_q, 4),
            "mean": round(mean_q, 4),
            "p25": round(p25, 4),
            "verdict": verdict,
        },
        "volume": volume,
    }


def parse_step_mesh_points(path, max_points=MAX_POINTS):
    """把四面体网格的顶点作为点云（有限元节点分布）。"""
    mesh = parse_step_mesh(path)
    arr = mesh["vertices"]
    rounded = np.round(arr, 6)
    _u, idx = np.unique(rounded, axis=0, return_index=True)
    arr = arr[np.sort(idx)]
    if len(arr) > max_points:
        step = max(1, len(arr) // max_points)
        arr = arr[::step][:max_points]
    return arr
