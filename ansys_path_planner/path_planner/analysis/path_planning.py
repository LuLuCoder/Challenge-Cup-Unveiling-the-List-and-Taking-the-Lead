"""基于最大主应力方向场的三维路径规划。"""
import math

import numpy as np
import pandas as pd

from path_planner import config


def normalize(values):
    """Min-Max 归一化到 [0, 1]；常量输入返回全 0。"""
    v = np.asarray(values, dtype=float)
    mn, mx = np.nanmin(v), np.nanmax(v)
    if abs(mx - mn) < 1e-15:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)


class PathPlanner:
    """
    在 ANSYS 节点云中沿最大主应力方向贪心追踪流线，生成覆盖式路径。

    核心思想：
        1. 全部有效节点参与规划；
        2. Maximum_Principal 控制路径密度（高应力区更密）；
        3. Principal_VX/VY/VZ 控制路径方向；
        4. cKDTree 在当前点附近寻找沿主方向的下一个节点；
        5. 低应力区域仍保持覆盖，仅降低优先级。
    """

    def __init__(self, df, result_name="Maximum_Principal",
                 percentile=config.DEFAULT_PERCENTILE, spacing=0.0):
        required = [result_name, "Principal_VX", "Principal_VY", "Principal_VZ"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"缺少字段：{missing}")

        self.result_name = result_name
        self.percentile = percentile
        self.spacing = spacing
        self.work = df.copy()
        self._prepare()

    # ---------- 准备 ----------

    def _prepare(self):
        w = self.work
        stress = w[self.result_name].to_numpy(float)
        w["Value_Normalized"] = normalize(stress)

        self.threshold = float(np.percentile(stress, self.percentile))
        w["Priority"] = np.where(stress >= self.threshold, 1, 0)

        self.points = w[["X", "Y", "Z"]].to_numpy(float)
        self.characteristic = self._characteristic_length()

        self._compute_adaptive_spacing()
        w["Path_Weight"] = (
            config.PATH_WEIGHT_BASE
            + config.PATH_WEIGHT_STRESS_FACTOR * w["Value_Normalized"]
        )

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("需要 scipy：pip install scipy") from None
        self.tree = cKDTree(self.points)

    def _characteristic_length(self):
        bbox = np.ptp(self.points, axis=0)
        return max(float(np.max(bbox)), 1e-12)

    def _compute_adaptive_spacing(self):
        """高应力 -> 小步长，低应力 -> 大步长。"""
        if self.spacing <= 0:
            d_max = self.characteristic * config.SPACING_MAX_RATIO
            d_min = self.characteristic * config.SPACING_MIN_RATIO
        else:
            d_min = self.spacing
            d_max = self.spacing * config.SPACING_USER_MAX_MULTIPLIER

        self.work["Path_Spacing"] = (
            d_max
            - (d_max - d_min)
            * np.power(
                self.work["Value_Normalized"].to_numpy(float),
                config.SPACING_GAMMA,
            )
        )

    # ---------- 追踪 ----------

    def _trace(self, start, direction_sign=1):
        """从 start 沿最大主应力方向追踪一条流线，返回节点索引列表。"""
        local_path = []
        current = start
        previous_direction = None

        for _ in range(len(self.points)):
            if not self.unvisited[current]:
                break

            local_path.append(current)
            self.unvisited[current] = False

            direction = np.array([
                self.work.iloc[current]["Principal_VX"],
                self.work.iloc[current]["Principal_VY"],
                self.work.iloc[current]["Principal_VZ"],
            ], dtype=float)

            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                break

            direction /= norm
            direction *= direction_sign

            if previous_direction is not None and np.dot(
                direction, previous_direction
            ) < 0:
                direction *= -1
            previous_direction = direction.copy()

            best = self._pick_next(current, direction)
            if best is None:
                break
            current = best

        return local_path

    def _pick_next(self, current, direction):
        """在当前点邻域内按打分选择下一个节点；无合适候选返回 None。"""
        current_point = self.points[current]
        local_spacing = float(self.work.iloc[current]["Path_Spacing"])

        radius = max(
            local_spacing * config.SEARCH_RADIUS_SPACING_MULTIPLIER,
            self.characteristic * config.SEARCH_RADIUS_MIN_RATIO,
        )

        candidates = self.tree.query_ball_point(current_point, radius)
        best, best_score = None, -np.inf

        for j in candidates:
            if j == current or not self.unvisited[j]:
                continue

            delta = self.points[j] - current_point
            dist = np.linalg.norm(delta)
            if dist < 1e-12:
                continue

            unit_delta = delta / dist
            direction_score = float(np.dot(unit_delta, direction))
            if direction_score < config.DIRECTION_MIN_DOT:
                continue

            distance_score = math.exp(
                -abs(dist - local_spacing) / max(local_spacing, 1e-12)
            )
            stress_score = 0.5 + 0.5 * self.work.iloc[j]["Value_Normalized"]

            score = (
                config.SCORE_DIRECTION_WEIGHT * direction_score
                + config.SCORE_DISTANCE_WEIGHT * distance_score
                + config.SCORE_STRESS_WEIGHT * stress_score
            )
            if score > best_score:
                best, best_score = j, score

        return best

    # ---------- 规划主流程 ----------

    def plan(self):
        """
        生成覆盖全部节点的路径。

        返回：
            path_df    带 Step / Path_Priority / Density 等列的路径表
            threshold  高优先级应力阈值
        """
        self.unvisited = np.ones(len(self.points), dtype=bool)
        path_indices = []

        # 第一条路径：从最大应力节点双向沿主方向追踪
        start_idx = int(np.argmax(self.work[self.result_name].to_numpy(float)))

        forward = self._trace(start_idx, direction_sign=1)
        if start_idx < len(self.unvisited):
            self.unvisited[start_idx] = True
        backward = self._trace(start_idx, direction_sign=-1)[::-1]

        combined = []
        for idx in backward[:-1] + forward:
            if idx not in combined:
                combined.append(idx)
        path_indices.extend(combined)

        # 剩余未覆盖节点：按应力从高到低作为种子继续追线
        remaining = sorted(
            np.where(self.unvisited)[0],
            key=lambda i: self.work[self.result_name].iloc[i],
            reverse=True,
        )
        for seed in remaining:
            if not self.unvisited[seed]:
                continue
            line = self._trace(int(seed), direction_sign=1)
            if line:
                path_indices.extend(line)

        if not path_indices:
            raise ValueError("没有生成有效路径。")

        return self._build_path_table(path_indices), self.threshold

    def _build_path_table(self, path_indices):
        path_df = self.work.iloc[path_indices].copy()
        path_df["Step"] = np.arange(1, len(path_df) + 1)
        path_df["Path_Priority"] = np.where(
            path_df["Priority"] == 1, "高应力/高密度", "常规覆盖"
        )

        spacing = path_df["Path_Spacing"].to_numpy(float)
        s_min, s_max = float(np.min(spacing)), float(np.max(spacing))
        path_df["Density"] = (
            0.5 if abs(s_max - s_min) < 1e-12
            else (s_max - spacing) / (s_max - s_min)
        )
        path_df["Density_Level"] = np.clip(
            (path_df["Density"].to_numpy(float) * 5).astype(int), 0, 4
        )

        xyz = path_df[["X", "Y", "Z"]].to_numpy(float)
        distances = np.zeros(len(xyz))
        if len(xyz) > 1:
            distances[1:] = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
        path_df["Segment_Length"] = distances

        return path_df


def generate_surface_path(df, result_name="Maximum_Principal",
                          percentile=config.DEFAULT_PERCENTILE, spacing=0.0):
    """便捷接口：等价于 PathPlanner(df, ...).plan()。"""
    return PathPlanner(
        df, result_name=result_name,
        percentile=percentile, spacing=spacing,
    ).plan()


class LayerPathPlanner:
    """
    面向 3D 打印的层式路径规划。

    沿切片轴把节点云分成若干层；每一层内部按应力自适应间距
    分条生成锯齿扫描线（zigzag），高应力区域扫描线更密、
    低应力区域更疏，但所有节点均被覆盖；层与层按顺序连接，
    形成一条连续的整体路径。
    """

    AXES = ["X", "Y", "Z"]

    def __init__(self, df, result_name="Maximum_Principal",
                 percentile=config.DEFAULT_PERCENTILE,
                 n_layers=config.DEFAULT_LAYERS,
                 spacing=0.0, slice_axis=config.SLICE_AXIS):
        required = [result_name, "X", "Y", "Z"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"缺少字段：{missing}")
        if slice_axis not in self.AXES:
            raise ValueError(f"切片轴必须是 X/Y/Z 之一，当前为：{slice_axis}")

        self.result_name = result_name
        self.percentile = percentile
        self.n_layers = max(1, int(n_layers))
        self.spacing = spacing
        self.slice_axis = slice_axis

        # 层内平面坐标：切片轴之外的另外两轴
        in_plane = [a for a in self.AXES if a != slice_axis]
        self.raster_axis = in_plane[0]   # 扫描方向（条内排序轴）
        self.strip_axis = in_plane[1]    # 分条方向（扫描线间距轴）

        self.work = df.copy()
        self._prepare()

    # ---------- 准备 ----------

    def _prepare(self):
        w = self.work
        stress = w[self.result_name].to_numpy(float)
        w["Value_Normalized"] = normalize(stress)

        self.threshold = float(np.percentile(stress, self.percentile))
        w["Priority"] = np.where(stress >= self.threshold, 1, 0)

        self.points = w[["X", "Y", "Z"]].to_numpy(float)
        self.characteristic = self._characteristic_length()
        self._compute_spacing_bounds()

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("需要 scipy：pip install scipy") from None
        self.tree = cKDTree(self.points)

        # 典型节点间距：用于空区检测半径（粗网格适当放大，细网格不受影响）
        if len(self.points) >= 2:
            nn_dist, _ = self.tree.query(self.points, k=2)
            self.typical_spacing = float(np.median(nn_dist[:, 1]))
        else:
            self.typical_spacing = self.characteristic * 0.02

        w["Path_Spacing"] = self._spacing_from_normalized(
            w["Value_Normalized"].to_numpy(float)
        )
        w["Path_Weight"] = (
            config.PATH_WEIGHT_BASE
            + config.PATH_WEIGHT_STRESS_FACTOR * w["Value_Normalized"]
        )

    def _characteristic_length(self):
        bbox = np.ptp(self.points, axis=0)
        return max(float(np.max(bbox)), 1e-12)

    def _compute_spacing_bounds(self):
        """自适应扫描线间距范围：高应力 -> 小间距（密），低应力 -> 大间距。"""
        if self.spacing <= 0:
            self.d_max = self.characteristic * config.SPACING_MAX_RATIO
            self.d_min = self.characteristic * config.SPACING_MIN_RATIO
        else:
            self.d_min = self.spacing
            self.d_max = self.spacing * config.SPACING_USER_MAX_MULTIPLIER

    def _spacing_from_normalized(self, values):
        return (
            self.d_max
            - (self.d_max - self.d_min)
            * np.power(values, config.SPACING_GAMMA)
        )

    # ---------- 分层 ----------

    def _slice_layers(self):
        """把节点按切片轴坐标切成若干层，返回每层的节点索引列表。"""
        slice_index = self.AXES.index(self.slice_axis)
        coords = self.points[:, slice_index]
        zmin, zmax = float(coords.min()), float(coords.max())

        if zmax - zmin < 1e-12:
            return [np.arange(len(self.points))]

        thickness = (zmax - zmin) / self.n_layers
        layers = []
        for k in range(self.n_layers):
            lo = zmin + k * thickness
            hi = zmin + (k + 1) * thickness
            if k == self.n_layers - 1:
                mask = (coords >= lo) & (coords <= hi)
            else:
                mask = (coords >= lo) & (coords < hi)
            idx = np.where(mask)[0]
            if len(idx) > 0:
                layers.append(idx)
        return layers

    # ---------- 层内扫描 ----------

    def _raster_layer(self, idx):
        """
        层内锯齿扫描：沿分条轴用自适应间距分条，
        条内沿扫描轴排序，相邻条方向交替形成连续 zigzag。
        """
        idx = np.asarray(idx, dtype=int)
        if len(idx) == 1:
            return [int(idx[0])]

        raster_index = self.AXES.index(self.raster_axis)
        strip_index = self.AXES.index(self.strip_axis)
        xs = self.points[idx, raster_index]
        ys = self.points[idx, strip_index]

        order = np.argsort(ys, kind="stable")
        sorted_idx = idx[order]
        sorted_ys = ys[order]

        bins = []
        pos = 0
        y = float(sorted_ys[0])
        ymax = float(sorted_ys[-1])

        while y <= ymax + 1e-12 and pos < len(sorted_idx):
            # 推进到当前条起始位置
            while pos < len(sorted_idx) and sorted_ys[pos] < y - 1e-12:
                pos += 1
            if pos >= len(sorted_idx):
                break

            # 条宽 = 当前位置应力对应的自适应间距
            s = self.work.iloc[sorted_idx[pos]]["Value_Normalized"]
            pitch = float(self._spacing_from_normalized(s))
            if pitch < 1e-12:
                pitch = self.characteristic * config.SPACING_MIN_RATIO

            band = []
            j = pos
            while j < len(sorted_idx) and sorted_ys[j] < y + pitch:
                band.append(sorted_idx[j])
                j += 1

            if band:
                band.sort(key=lambda i: self.points[i, raster_index])
                bins.append(band)
                pos = j

            y += pitch

        line = []
        for k, band in enumerate(bins):
            if k % 2 == 1:
                band = band[::-1]
            line.extend(band)
        return line

    def _segment_supported(self, a, b):
        """
        判断 a->b 的直线段是否全程经过有节点的区域。

        沿线段均匀采样，若某个采样点附近（检测半径内）没有任何节点，
        说明该处没有零件实体，路径段属于穿越空区，应断开。
        """
        a_pt = self.points[a]
        b_pt = self.points[b]
        delta = b_pt - a_pt
        dist = float(np.linalg.norm(delta))
        if dist < 1e-12:
            return True

        check_radius = max(
            self.characteristic * config.VOID_CHECK_RADIUS_RATIO,
            self.d_max,
            self.typical_spacing * config.VOID_CHECK_SPACING_MULTIPLIER,
        )
        n_samples = config.VOID_SAMPLE_MIN + int(
            np.ceil(dist / max(check_radius, 1e-12))
        )
        for t in np.linspace(0.0, 1.0, n_samples):
            point = a_pt + t * delta
            if not self.tree.query_ball_point(point, check_radius):
                return False
        return True

    # ---------- 规划主流程 ----------

    def plan(self):
        """
        生成层式路径。

        返回：
            path_df    带 Step / Layer / Segment_Type 等列的路径表
            threshold  高优先级应力阈值
        """
        layers = self._slice_layers()

        path_indices = []
        layer_numbers = []
        for k, idx in enumerate(layers, start=1):
            line = self._raster_layer(idx)
            path_indices.extend(line)
            layer_numbers.extend([k] * len(line))

        if not path_indices:
            raise ValueError("没有生成有效路径。")

        return self._build_path_table(path_indices, layer_numbers), self.threshold

    def _build_path_table(self, path_indices, layer_numbers):
        path_df = self.work.iloc[path_indices].copy()
        path_df["Step"] = np.arange(1, len(path_df) + 1)
        path_df["Layer"] = np.asarray(layer_numbers, dtype=int)
        path_df["Path_Priority"] = np.where(
            path_df["Priority"] == 1, "高应力/高密度", "常规覆盖"
        )

        # 段类型与子路径：
        #   层内路径    —— 同一层内的扫描段
        #   层间过渡    —— 相邻层之间的连接段（实体内）
        #   空区断开    —— 穿越无节点区域的段，不参与打印路径
        # 子路径按空区断开切分，实体内低应力区域仍全部保留
        layers_arr = np.asarray(layer_numbers, dtype=int)
        seg_type = ["起点"]
        sub_path = [1]
        for i in range(1, len(path_indices)):
            if not self._segment_supported(
                path_indices[i - 1], path_indices[i]
            ):
                seg_type.append("空区断开")
                sub_path.append(sub_path[-1] + 1)
            elif layers_arr[i] != layers_arr[i - 1]:
                seg_type.append("层间过渡")
                sub_path.append(sub_path[-1])
            else:
                seg_type.append("层内路径")
                sub_path.append(sub_path[-1])

        path_df["Segment_Type"] = seg_type
        path_df["SubPath"] = np.asarray(sub_path, dtype=int)

        spacing = path_df["Path_Spacing"].to_numpy(float)
        s_min, s_max = float(np.min(spacing)), float(np.max(spacing))
        path_df["Density"] = (
            0.5 if abs(s_max - s_min) < 1e-12
            else (s_max - spacing) / (s_max - s_min)
        )
        path_df["Density_Level"] = np.clip(
            (path_df["Density"].to_numpy(float) * 5).astype(int), 0, 4
        )

        xyz = path_df[["X", "Y", "Z"]].to_numpy(float)
        distances = np.zeros(len(xyz))
        if len(xyz) > 1:
            distances[1:] = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
        path_df["Segment_Length"] = distances

        return path_df


def generate_layer_path(df, result_name="Maximum_Principal",
                        percentile=config.DEFAULT_PERCENTILE,
                        n_layers=config.DEFAULT_LAYERS,
                        spacing=0.0, slice_axis=config.SLICE_AXIS):
    """层式 3D 打印路径规划便捷接口。"""
    return LayerPathPlanner(
        df, result_name=result_name, percentile=percentile,
        n_layers=n_layers, spacing=spacing, slice_axis=slice_axis,
    ).plan()
