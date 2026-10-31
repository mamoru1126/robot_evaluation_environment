"""Dynamic Window Approach (Fox, Burgard & Thrun 1997).

Velocity pairs (v, w) reachable within one control period (the dynamic
window) are sampled, each is rolled out as a constant-curvature arc for
``sim_time`` seconds and scored with DWB-like critics:

* path   : lateral distance of the arc end point to the path
* goal   : remaining path length from the arc end point (progress)
* heading: heading error w.r.t. the path direction at the end point
* obstacle: closeness of the arc to obstacles (arcs that collide are rejected)
* speed  : preference for high forward velocity

With ``perception = ground_truth`` the humans are predicted with a
constant-velocity model along the arc time.
"""
from __future__ import annotations

import math

import numpy as np

from ..geometry import wrap_angle
from .base import BaseController, ControlOutput, ControllerInput, Param


class DWAController(BaseController):
    name = "dwa"
    label = "DWA (Dynamic Window Approach)"
    description = ("速度空間の動的ウィンドウ内で (v, ω) をサンプリングし、等速円弧を評価して最良の指令を選ぶ。"
                   "障害物を避けながら経路に戻る。")

    PARAMS = {
        "target_speed": Param(0.8, 0.05, 2.0, 0.05, "最大(目標)速度 [m/s]"),
        "sim_time": Param(3.0, 0.5, 6.0, 0.1, "予測時間 [s]"),
        "sim_dt": Param(0.1, 0.02, 0.5, 0.01, "予測刻み [s]"),
        "v_samples": Param(9, 2, 31, 1, "v サンプル数"),
        "w_samples": Param(21, 3, 61, 1, "ω サンプル数"),
        "w_path": Param(1.0, 0.0, 20.0, 0.1, "重み: 経路からの横ずれ"),
        "w_goal": Param(1.0, 0.0, 20.0, 0.1, "重み: 進捗 (残り距離)"),
        "w_heading": Param(0.3, 0.0, 20.0, 0.1, "重み: 経路方向との角度差"),
        "w_obstacle": Param(4.0, 0.0, 50.0, 0.1, "重み: 障害物への近さ"),
        "w_speed": Param(0.5, 0.0, 20.0, 0.1, "重み: 速度"),
        "obstacle_influence": Param(0.6, 0.05, 5.0, 0.05, "障害物コストが効く距離 [m]"),
        "safety_margin": Param(0.15, 0.0, 1.0, 0.01, "衝突判定の余裕 [m]"),
        "use_measured_velocity": Param(True, label="窓の中心に計測速度を使う",
                                       help="False なら前回指令値を中心にする"),
    }

    def reset(self) -> None:
        self.last_best = None

    def compute(self, inp: ControllerInput) -> ControlOutput:
        st = inp.state
        r = self.robot
        T = max(inp.dt, 1e-3)
        v0, w0 = (st.v, st.w) if self.p["use_measured_velocity"] else inp.last_cmd
        v_hi = min(self.p["target_speed"], r.max_v, v0 + r.max_acc * T)
        v_lo = max(r.min_v, v0 - r.max_dec * T)
        if v_lo > v_hi:
            v_lo = v_hi
        w_hi = min(r.max_w, w0 + r.max_alpha * T)
        w_lo = max(-r.max_w, w0 - r.max_alpha * T)
        if w_lo > w_hi:
            w_lo = w_hi

        vs = np.linspace(v_lo, v_hi, int(self.p["v_samples"]))
        ws = np.linspace(w_lo, w_hi, int(self.p["w_samples"]))
        if w_lo < 0.0 < w_hi:
            ws = np.unique(np.concatenate([ws, [0.0]]))
        V, W = np.meshgrid(vs, ws, indexing="ij")
        V = V.ravel()
        W = W.ravel()

        dt = self.p["sim_dt"]
        n = max(1, int(round(self.p["sim_time"] / dt)))
        t = np.arange(1, n + 1) * dt                          # (n,)
        th = st.yaw + W[:, None] * t[None, :]                 # (K, n)
        small = np.abs(W) < 1e-6
        Wsafe = np.where(small, 1.0, W)
        xs_arc = st.x + (V / Wsafe)[:, None] * (np.sin(th) - math.sin(st.yaw))
        ys_arc = st.y - (V / Wsafe)[:, None] * (np.cos(th) - math.cos(st.yaw))
        xs_lin = st.x + V[:, None] * t[None, :] * math.cos(st.yaw)
        ys_lin = st.y + V[:, None] * t[None, :] * math.sin(st.yaw)
        xs = np.where(small[:, None], xs_lin, xs_arc)
        ys = np.where(small[:, None], ys_lin, ys_arc)
        traj = np.stack([xs, ys], -1)                         # (K, n, 2)

        clear = inp.obstacles.clearance(traj, t[None, :])     # (K, n)
        min_clear = clear.min(axis=1)
        now_clear = inp.obstacles.clearance_now(st.x, st.y)
        margin = self.p["safety_margin"]
        # a trajectory is admissible if it never gets closer than the margin
        # (when we already are inside the margin, require not getting closer)
        if now_clear > margin:
            ok = min_clear > margin
        else:
            ok = min_clear >= now_clear - 1e-3

        end = traj[:, -1, :]
        path = inp.path
        lat = path.lateral_errors(end)
        s_end = path.progress(end)
        remaining = path.length - s_end
        head_err = np.abs(wrap_angle(th[:, -1] - path.heading_at(s_end)))
        infl = self.p["obstacle_influence"]
        obs_cost = np.clip((infl - min_clear) / infl, 0.0, 1.0) ** 2
        speed_cost = (self.p["target_speed"] - V) / max(self.p["target_speed"], 1e-3)

        cost = (self.p["w_path"] * lat + self.p["w_goal"] * remaining + self.p["w_heading"] * head_err
                + self.p["w_obstacle"] * obs_cost + self.p["w_speed"] * speed_cost)
        cost = np.where(ok, cost, np.inf)

        info = {"num_candidates": int(len(V)), "num_valid": int(np.count_nonzero(ok))}
        if not np.any(np.isfinite(cost)):
            # no admissible arc: brake as hard as possible
            v = max(v0 - r.max_dec * T, 0.0)
            w = 0.0
            info["blocked"] = True
            best_traj = traj[0].tolist() if len(traj) else None
            return ControlOutput(v, w, trajectory=best_traj, info=info)

        k = int(np.argmin(cost))
        v, w = float(V[k]), float(W[k])
        # a few candidate arcs for visualisation
        sel = np.linspace(0, len(V) - 1, min(15, len(V))).astype(int)
        cands = [traj[i, :: max(1, n // 10)].tolist() for i in sel if ok[i]]
        info.update({"blocked": False, "min_clearance": float(min_clear[k]), "cost": float(cost[k])})
        return ControlOutput(v, w, trajectory=traj[k].tolist(), candidates=cands, info=info)
