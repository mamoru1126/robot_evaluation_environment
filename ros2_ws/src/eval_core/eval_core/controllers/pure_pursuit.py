"""Regulated Pure Pursuit and Dynamic Window Pure Pursuit (DWPP).

Both follow the structure of Nav2's ``nav2_regulated_pure_pursuit_controller``:

1. velocity-scaled lookahead and carrot point on the path
2. curvature ``kappa = 2 y / (x^2 + y^2)`` of the arc to the carrot
3. regulated linear velocity (curvature / obstacle proximity / goal approach)
4. ``w = kappa * v`` (RPP)  -- or --  DWPP velocity selection inside the
   dynamic window (Ohnishi, "DWPP: Dynamic Window Pure Pursuit Considering
   Velocity and Acceleration Constraints", arXiv:2601.15006)
5. forward collision check along the commanded arc; when a collision is
   imminent the robot stops and waits (Nav2 throws NoValidControl here).

The DWPP functions are a direct port of
``dynamic_window_pure_pursuit_functions.hpp`` (Apache-2.0, (c) 2025 Fumiya
Ohnishi) from the Nav2 main branch, which is not available in ROS 2 Jazzy.
Neither controller *avoids* obstacles: they track the path and stop.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ..geometry import to_robot_frame
from .base import BaseController, ControlOutput, ControllerInput, Param, rollout_constant


# ----------------------------------------------------------------------------
# DWPP functions (port of nav2 dynamic_window_pure_pursuit_functions.hpp)
# ----------------------------------------------------------------------------
def compute_dynamic_window(cur_v, cur_w, max_v, min_v, max_w, min_w,
                           max_acc, max_dec, max_alpha_acc, max_alpha_dec, dt):
    """Returns (max_v_dw, min_v_dw, max_w_dw, min_w_dw).

    ``max_dec`` / ``max_alpha_dec`` are *negative* numbers like in Nav2.
    """
    eps = 1e-3

    def window(cur, vmax, vmin, acc, dec):
        if cur > eps:
            cmax = cur + acc * dt
            cmin = cur + dec * dt
        elif cur < -eps:
            cmax = cur - dec * dt
            cmin = cur - acc * dt
        else:
            cmax = cur + acc * dt
            cmin = cur - acc * dt
        return min(cmax, vmax), max(cmin, vmin)

    v_hi, v_lo = window(cur_v, max_v, min_v, max_acc, max_dec)
    w_hi, w_lo = window(cur_w, max_w, min_w, max_alpha_acc, max_alpha_dec)
    return [v_hi, v_lo, w_hi, w_lo]


def apply_regulation_to_dynamic_window(regulated_v, dw):
    v_reg_min = min(0.0, regulated_v)
    v_reg_max = max(0.0, regulated_v)
    dw[1] = max(dw[1], v_reg_min)
    dw[0] = min(dw[0], v_reg_max)
    if dw[1] > dw[0]:
        if dw[1] > v_reg_max:
            dw[0] = dw[1]
        else:
            dw[1] = dw[0]
    return dw


def compute_optimal_velocity_within_dynamic_window(dw, curvature, sign) -> Tuple[float, float]:
    max_v, min_v, max_w, min_w = dw
    if abs(curvature) < 1e-3:
        v = max_v if sign >= 0.0 else min_v
        if min_w <= 0.0 <= max_w:
            w = 0.0
        else:
            w = min_w if abs(min_w) <= abs(max_w) else max_w
        return v, w

    big = 1.7976931348623157e308
    best_v = -big * sign
    best_w = 0.0
    candidates = [
        (min_v, curvature * min_v),
        (max_v, curvature * max_v),
        (min_w / curvature, min_w),
        (max_w / curvature, max_w),
    ]
    for cv, cw in candidates:
        if min_v <= cv <= max_v and min_w <= cw <= max_w:
            if cv * sign > best_v * sign:
                best_v, best_w = cv, cw
    if best_v != -big * sign:
        return best_v, best_w

    corners = [(min_v, min_w), (min_v, max_w), (max_v, min_w), (max_v, max_w)]
    denom = math.sqrt(curvature * curvature + 1.0)
    closest = big
    best_v = -big * sign
    best_w = 0.0
    for cv, cw in corners:
        dist = abs(curvature * cv - cw) / denom
        if dist < closest or (abs(dist - closest) <= 1e-3 and cv * sign > best_v * sign):
            closest = dist
            best_v, best_w = cv, cw
    return best_v, best_w


def compute_dynamic_window_velocities(cur_v, cur_w, max_v, min_v, max_w, min_w, max_acc, max_dec,
                                      max_alpha_acc, max_alpha_dec, regulated_v, curvature, sign, dt):
    dw = compute_dynamic_window(cur_v, cur_w, max_v, min_v, max_w, min_w,
                                max_acc, max_dec, max_alpha_acc, max_alpha_dec, dt)
    dw = apply_regulation_to_dynamic_window(regulated_v, dw)
    return compute_optimal_velocity_within_dynamic_window(dw, curvature, sign)


# ----------------------------------------------------------------------------
# Regulated pure pursuit
# ----------------------------------------------------------------------------
class RegulatedPurePursuit(BaseController):
    name = "pure_pursuit"
    label = "Pure Pursuit (Regulated)"
    description = ("Nav2 Regulated Pure Pursuit 相当。経路を追従するだけで回避はしない。"
                   "前方の円弧上に衝突が迫ると停止して待つ。")
    dynamic_window = False

    PARAMS = {
        "target_speed": Param(0.8, 0.05, 2.0, 0.05, "目標速度 [m/s]"),
        "lookahead_time": Param(1.5, 0.1, 5.0, 0.1, "前方注視時間 [s]",
                                "注視距離 = 速度 × 時間 (min/max でクリップ)"),
        "min_lookahead": Param(0.3, 0.05, 3.0, 0.05, "最小注視距離 [m]"),
        "max_lookahead": Param(0.9, 0.1, 5.0, 0.05, "最大注視距離 [m]"),
        "use_regulated_linear_velocity_scaling": Param(True, label="曲率で減速"),
        "regulated_min_radius": Param(0.9, 0.1, 5.0, 0.05, "減速開始の旋回半径 [m]"),
        "regulated_min_speed": Param(0.25, 0.0, 1.0, 0.05, "減速時の最低速度 [m/s]"),
        "use_cost_regulated_scaling": Param(True, label="障害物近接で減速"),
        "cost_scaling_dist": Param(0.6, 0.05, 3.0, 0.05, "近接減速を始める距離 [m]",
                                   "ロボット表面から障害物までの距離"),
        "cost_scaling_gain": Param(1.0, 0.1, 2.0, 0.05, "近接減速ゲイン"),
        "approach_dist": Param(0.6, 0.0, 3.0, 0.05, "ゴール接近減速距離 [m]"),
        "min_approach_speed": Param(0.05, 0.0, 1.0, 0.01, "ゴール接近時の最低速度 [m/s]"),
        "use_rotate_to_heading": Param(True, label="大きくずれたらその場回転"),
        "rotate_to_heading_min_angle": Param(0.785, 0.1, 3.14, 0.05, "その場回転する角度 [rad]"),
        "rotate_to_heading_speed": Param(1.8, 0.1, 4.0, 0.1, "その場回転速度 [rad/s]"),
        "use_collision_detection": Param(True, label="衝突予測で停止"),
        "collision_time": Param(1.0, 0.1, 5.0, 0.1, "衝突予測時間 [s]",
                                "コマンド円弧を注視点までこの時間だけ先読みして衝突判定"),
        "safety_margin": Param(0.05, 0.0, 1.0, 0.01, "衝突判定の余裕 [m]"),
    }

    def reset(self) -> None:
        self.blocked = False

    # ------------------------------------------------------------------
    def _lookahead(self, v: float) -> float:
        L = abs(v) * self.p["lookahead_time"]
        return min(max(L, self.p["min_lookahead"]), self.p["max_lookahead"])

    def _regulate(self, curvature: float, clearance: float, remaining: float, carrot_goal_dist: float) -> float:
        v = self.p["target_speed"]
        curv_v, cost_v = v, v
        if self.p["use_regulated_linear_velocity_scaling"] and abs(curvature) > 1e-9:
            radius = abs(1.0 / curvature)
            rmin = self.p["regulated_min_radius"]
            if radius < rmin:
                curv_v = v * (1.0 - abs(radius - rmin) / rmin)
        if self.p["use_cost_regulated_scaling"]:
            d = max(clearance, 0.0)
            if d < self.p["cost_scaling_dist"]:
                cost_v = v * (self.p["cost_scaling_gain"] * d / self.p["cost_scaling_dist"])
        v = min(curv_v, cost_v)
        v = max(v, self.p["regulated_min_speed"])
        # approach to the end of the path
        ad = self.p["approach_dist"]
        if ad > 0 and remaining < ad:
            scale = carrot_goal_dist / ad
            approach_v = max(v * scale, self.p["min_approach_speed"])
            v = min(v, approach_v)
        return min(max(v, 0.0), self.robot.max_v)

    def _collision_imminent(self, inp: ControllerInput, v: float, w: float, carrot_dist: float) -> bool:
        st = inp.state
        obs = inp.obstacles
        margin = self.p["safety_margin"]
        if obs.clearance_now(st.x, st.y) <= 0.0:
            return True
        if abs(v) < 1e-3:
            return False
        dt = 0.05
        xy, t = rollout_constant(st.x, st.y, st.yaw, v, w, self.p["collision_time"], dt)
        dist = np.hypot(xy[:, 0] - st.x, xy[:, 1] - st.y)
        keep = dist <= max(carrot_dist, self.robot.radius + 0.1)
        if not np.any(keep):
            return False
        c = obs.clearance(xy[keep], t[keep])
        return bool(np.any(c <= margin))

    # ------------------------------------------------------------------
    def compute(self, inp: ControllerInput) -> ControlOutput:
        st = inp.state
        path = inp.path
        L = self._lookahead(st.v)  # nav2 uses the odometry speed for the lookahead
        carrot, _ = path.lookahead_point(st.x, st.y, L)
        cx, cy = to_robot_frame(carrot[0], carrot[1], st.x, st.y, st.yaw)
        cx, cy = float(cx), float(cy)
        d2 = cx * cx + cy * cy
        curvature = 2.0 * cy / d2 if d2 > 1e-3 else 0.0
        carrot_dist = math.sqrt(d2)

        remaining = path.remaining(st.x, st.y)
        goal = path.goal
        goal_dist = math.hypot(goal[0] - st.x, goal[1] - st.y)
        clearance = inp.obstacles.clearance_now(st.x, st.y)

        rotating = False
        angle_to_carrot = math.atan2(cy, cx)
        if self.p["use_rotate_to_heading"] and abs(angle_to_carrot) > self.p["rotate_to_heading_min_angle"]:
            # rotate in place towards the path (angular acceleration limited)
            rotating = True
            sign = 1.0 if angle_to_carrot > 0 else -1.0
            w_des = sign * self.p["rotate_to_heading_speed"]
            dw = self.robot.max_alpha * inp.dt
            w = min(max(w_des, inp.last_cmd[1] - dw), inp.last_cmd[1] + dw)
            v = 0.0
        else:
            v_reg = self._regulate(curvature, clearance, remaining, goal_dist)
            if not self.dynamic_window:
                v = v_reg
                w = v * curvature
            else:
                v, w = compute_dynamic_window_velocities(
                    inp.last_cmd[0], inp.last_cmd[1],
                    self.robot.max_v, self.robot.min_v, self.robot.max_w, -self.robot.max_w,
                    self.robot.max_acc, -self.robot.max_dec, self.robot.max_alpha, -self.robot.max_alpha,
                    v_reg, curvature, 1.0, inp.dt)

        blocked = False
        if self.p["use_collision_detection"] and not rotating:
            if self._collision_imminent(inp, v, w, carrot_dist):
                blocked = True
                v, w = 0.0, 0.0
        self.blocked = blocked
        traj, _ = rollout_constant(st.x, st.y, st.yaw, v, w, 1.5, 0.1)
        return ControlOutput(v, w, trajectory=traj.tolist(),
                             info={"carrot": [float(carrot[0]), float(carrot[1])], "curvature": curvature,
                                   "blocked": blocked, "rotating": rotating, "lookahead": L})


class DynamicWindowPurePursuit(RegulatedPurePursuit):
    name = "dwpp"
    label = "DWPP (Dynamic Window Pure Pursuit)"
    description = ("Pure Pursuit の曲率を保ちつつ、速度・加速度制約 (Dynamic Window) 内で"
                   "曲率直線に最も近い指令を選ぶ。Nav2 main の DWPP 実装を移植。回避はせず停止で対応。")
    dynamic_window = True
