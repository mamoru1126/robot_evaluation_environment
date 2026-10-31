"""Nonlinear Model Predictive Control (single shooting, SLSQP).

* model     : unicycle  x+ = x + v cos(th) dt,  y+ = y + v sin(th) dt,  th+ = th + w dt
* decision  : v_0..v_{N-1}, w_0..w_{N-1}
* bounds    : velocity limits (box) and acceleration limits (linear inequalities,
              coupled to the current velocity)
* cost      : contouring / lag error w.r.t. a reference point that moves along
              the path at the target speed, heading error, speed tracking,
              control rate, and a smooth penalty that keeps a safety distance
              from obstacles (humans predicted with constant velocity when
              ground-truth perception is used; LiDAR points otherwise).

The gradient is computed analytically with a backward (adjoint) pass so that
scipy's SLSQP converges in a few milliseconds.  The previous solution is
shifted and used as a warm start (receding horizon).
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..geometry import wrap_angle
from .base import BaseController, ControlOutput, ControllerInput, Param

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


class MPCController(BaseController):
    name = "mpc"
    label = "MPC (Nonlinear MPC / SLSQP)"
    description = ("ユニサイクルモデルで N ステップ先まで予測し、経路追従誤差・速度・障害物との距離の"
                   "コストを速度/加速度制約付きで最適化する非線形 MPC (SLSQP, 解析勾配)。")

    PARAMS = {
        "target_speed": Param(0.8, 0.05, 2.0, 0.05, "目標速度 [m/s]"),
        "horizon_steps": Param(20, 5, 60, 1, "予測ステップ数 N"),
        "dt": Param(0.15, 0.02, 0.5, 0.01, "予測刻み [s]"),
        "q_lateral": Param(4.0, 0.0, 100.0, 0.1, "重み: 横方向誤差²"),
        "q_lag": Param(1.0, 0.0, 100.0, 0.1, "重み: 進行方向誤差²"),
        "q_heading": Param(0.5, 0.0, 100.0, 0.1, "重み: 方位誤差²"),
        "q_terminal": Param(3.0, 1.0, 50.0, 0.5, "終端コストの倍率"),
        "r_speed": Param(0.5, 0.0, 100.0, 0.1, "重み: 目標速度との差²"),
        "r_w": Param(0.05, 0.0, 10.0, 0.01, "重み: ω²"),
        "r_dv": Param(1.0, 0.0, 100.0, 0.1, "重み: Δv²"),
        "r_dw": Param(0.5, 0.0, 100.0, 0.1, "重み: Δω²"),
        "q_obstacle": Param(100.0, 0.0, 5000.0, 5.0, "重み: 安全距離の侵害²"),
        "safety_distance": Param(0.5, 0.0, 3.0, 0.05, "人・障害物との目標安全距離 [m]",
                                 "ロボット表面からの距離。これより近づくとコストが掛かる"),
        "q_collision": Param(1000.0, 0.0, 1e5, 10.0, "重み: 接触の侵害²"),
        "obstacle_radius": Param(3.0, 0.5, 10.0, 0.5, "考慮する静的障害物点の範囲 [m]"),
        "max_static_points": Param(10, 0, 50, 1, "考慮する静的障害物点の最大数"),
        "max_iter": Param(30, 1, 200, 1, "SLSQP 最大反復回数"),
        "multi_start": Param(True, label="左右回避の初期解も試す (多点スタート)",
                             help="障害物が近いとき、直進・右回避・左回避の3つの初期解から最適化して最良を採用"),
        "pass_side": Param("right", label="すれ違う側の優先", type="choice", choices=["right", "left", "none"]),
        "w_side": Param(1.0, 0.0, 50.0, 0.1, "重み: 優先しない側への横ずれ²"),
    }

    def reset(self) -> None:
        self.z_prev = None

    # ------------------------------------------------------------------
    def _setup(self, inp: ControllerInput):
        st = inp.state
        p = self.p
        N = int(p["horizon_steps"])
        dt = float(p["dt"])
        path = inp.path
        s0, _, _ = path.project(st.x, st.y)
        s_ref = s0 + p["target_speed"] * dt * np.arange(1, N + 1)
        ref = path.points_at(s_ref)
        psi = path.heading_at(s_ref)
        # obstacles: dynamic agents (predicted) + representative static points
        obs_pos = []
        obs_rad = []
        tt = dt * np.arange(1, N + 1)
        for a in inp.obstacles.agents:
            if not inp.obstacles.use_prediction:
                obs_pos.append(np.tile([a.x, a.y], (N, 1)))
            else:
                obs_pos.append(np.stack([a.x + a.vx * tt, a.y + a.vy * tt], -1))
            obs_rad.append(a.radius)
        if p["max_static_points"] > 0:
            pts = inp.obstacles.nearby_static_points(st.x, st.y, p["obstacle_radius"],
                                                     int(p["max_static_points"]), 0.25)
            for q in pts:
                obs_pos.append(np.tile(q, (N, 1)))
                obs_rad.append(inp.obstacles.point_radius)
        obs_pos = np.asarray(obs_pos).reshape(-1, N, 2)
        obs_rad = np.asarray(obs_rad, float)
        return N, dt, ref, psi, obs_pos, obs_rad

    def _cost_grad(self, z, x0, y0, th0, v_init, w_init, N, dt, ref, psi, obs_pos, obs_rad):
        p = self.p
        v = z[:N]
        w = z[N:]
        # forward pass
        xs = np.empty(N + 1)
        ys = np.empty(N + 1)
        ths = np.empty(N + 1)
        xs[0], ys[0], ths[0] = x0, y0, th0
        for k in range(N):
            xs[k + 1] = xs[k] + v[k] * math.cos(ths[k]) * dt
            ys[k + 1] = ys[k] + v[k] * math.sin(ths[k]) * dt
            ths[k + 1] = ths[k] + w[k] * dt
        X = xs[1:]
        Y = ys[1:]
        TH = ths[1:]
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        ex = X - ref[:, 0]
        ey = Y - ref[:, 1]
        e_lag = cpsi * ex + spsi * ey
        e_lat = -spsi * ex + cpsi * ey
        e_th = wrap_angle(TH - psi)
        wk = np.ones(N)
        wk[-1] = p["q_terminal"]
        J = np.sum(wk * (p["q_lateral"] * e_lat ** 2 + p["q_lag"] * e_lag ** 2 + p["q_heading"] * e_th ** 2))
        # gradient of the state cost w.r.t. states (k = 1..N)
        dlat = wk * 2 * p["q_lateral"] * e_lat
        if p["pass_side"] != "none" and p["w_side"] > 0:
            sgn = 1.0 if p["pass_side"] == "right" else -1.0     # penalise deviations to the other side
            hinge = np.maximum(0.0, sgn * e_lat)
            J += p["w_side"] * np.sum(hinge ** 2)
            dlat = dlat + 2 * p["w_side"] * hinge * sgn
        gx = dlat * (-spsi) + wk * 2 * p["q_lag"] * e_lag * cpsi
        gy = dlat * cpsi + wk * 2 * p["q_lag"] * e_lag * spsi
        gth = wk * 2 * p["q_heading"] * e_th
        # obstacle penalty
        if len(obs_rad):
            dx = X[None, :] - obs_pos[:, :, 0]
            dy = Y[None, :] - obs_pos[:, :, 1]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-9
            d = dist - obs_rad[:, None] - self.robot.radius
            viol_s = np.maximum(0.0, p["safety_distance"] - d)
            viol_c = np.maximum(0.0, 0.02 - d)
            J += p["q_obstacle"] * np.sum(viol_s ** 2) + p["q_collision"] * np.sum(viol_c ** 2)
            dJdd = -2.0 * p["q_obstacle"] * viol_s - 2.0 * p["q_collision"] * viol_c
            gx += np.sum(dJdd * dx / dist, axis=0)
            gy += np.sum(dJdd * dy / dist, axis=0)
        # control cost
        vt = p["target_speed"]
        dv = np.diff(np.concatenate([[v_init], v]))
        dw = np.diff(np.concatenate([[w_init], w]))
        J += p["r_speed"] * np.sum((v - vt) ** 2) + p["r_w"] * np.sum(w ** 2) \
            + p["r_dv"] * np.sum(dv ** 2) + p["r_dw"] * np.sum(dw ** 2)
        gv = 2 * p["r_speed"] * (v - vt) + 2 * p["r_dv"] * dv
        gv[:-1] -= 2 * p["r_dv"] * dv[1:]
        gw = 2 * p["r_w"] * w + 2 * p["r_dw"] * dw
        gw[:-1] -= 2 * p["r_dw"] * dw[1:]
        # backward (adjoint) pass through the dynamics
        lx = ly = lth = 0.0
        for k in range(N - 1, -1, -1):
            # lambda_{k+1} includes the stage cost of state k+1
            lx += gx[k]
            ly += gy[k]
            lth += gth[k]
            c, s = math.cos(ths[k]), math.sin(ths[k])
            gv[k] += (lx * c + ly * s) * dt
            gw[k] += lth * dt
            # propagate to state k
            lth = lth + (-lx * v[k] * s + ly * v[k] * c) * dt
        return float(J), np.concatenate([gv, gw]), np.stack([X, Y], -1)

    # ------------------------------------------------------------------
    def _obstacle_ahead(self, st, ref, obs_pos, obs_rad) -> bool:
        """Is any (predicted) obstacle close to the reference corridor within the horizon?"""
        reach = self.robot.radius + self.p["safety_distance"] + 0.3
        d = np.hypot(obs_pos[:, :, 0] - ref[None, :, 0], obs_pos[:, :, 1] - ref[None, :, 1]) - obs_rad[:, None]
        return bool(np.any(d < reach))

    def _sidestep_seed(self, N, dt, v_init, w_init, side) -> np.ndarray:
        """S-shaped manoeuvre (turn, counter-turn, straight) to the given side (+1 = left)."""
        r = self.robot
        v = np.minimum(self.p["target_speed"], v_init + r.max_acc * dt * np.arange(1, N + 1)) * 0.8
        n1 = max(2, N // 4)
        w = np.zeros(N)
        w[:n1] = side * 1.2
        w[n1:2 * n1] = -side * 1.2
        # respect the angular acceleration limit from the current yaw rate
        out = np.empty(N)
        prev = w_init
        for k in range(N):
            out[k] = np.clip(w[k], prev - r.max_alpha * dt, prev + r.max_alpha * dt)
            prev = out[k]
        return np.concatenate([v, out])

    # ------------------------------------------------------------------
    def compute(self, inp: ControllerInput) -> ControlOutput:
        if minimize is None:  # pragma: no cover
            raise RuntimeError("scipy is required for the MPC controller")
        st = inp.state
        r = self.robot
        p = self.p
        N, dt, ref, psi, obs_pos, obs_rad = self._setup(inp)
        v_init, w_init = inp.last_cmd
        vmax = min(p["target_speed"], r.max_v)

        if self.z_prev is None or len(self.z_prev) != 2 * N:
            z0 = np.concatenate([np.full(N, v_init), np.full(N, w_init)])
        else:
            shift = max(1, int(round(inp.dt / dt)))
            vprev, wprev = self.z_prev[:N], self.z_prev[N:]
            z0 = np.concatenate([np.concatenate([vprev[shift:], np.repeat(vprev[-1], shift)])[:N],
                                 np.concatenate([wprev[shift:], np.repeat(wprev[-1], shift)])[:N]])
        lo = np.concatenate([np.full(N, r.min_v), np.full(N, -r.max_w)])
        hi = np.concatenate([np.full(N, vmax), np.full(N, r.max_w)])
        z0 = np.clip(z0, lo, hi)

        # acceleration limits as G z + h >= 0
        D = np.eye(N) - np.eye(N, k=-1)            # (z_k - z_{k-1})
        e0 = np.zeros(N)
        e0[0] = 1.0
        Z = np.zeros((N, N))
        G = np.vstack([np.hstack([-D, Z]), np.hstack([D, Z]), np.hstack([Z, -D]), np.hstack([Z, D])])
        h = np.concatenate([r.max_acc * dt + e0 * v_init,     # acc dt - (v_k - v_{k-1}) >= 0
                            r.max_dec * dt - e0 * v_init,     # dec dt + (v_k - v_{k-1}) >= 0
                            r.max_alpha * dt + e0 * w_init,
                            r.max_alpha * dt - e0 * w_init])

        args = (st.x, st.y, st.yaw, v_init, w_init, N, dt, ref, psi, obs_pos, obs_rad)
        bounds = list(zip(lo, hi))
        cons = [{"type": "ineq", "fun": lambda z: G @ z + h, "jac": lambda z: G}]
        t0 = time.perf_counter()
        seeds = [z0]
        if p["multi_start"] and len(obs_rad) and self._obstacle_ahead(st, ref, obs_pos, obs_rad):
            for side in (-1.0, 1.0):   # right, left
                seeds.append(np.clip(self._sidestep_seed(N, dt, v_init, w_init, side), lo, hi))
        best = None
        nit = 0
        ok = False
        for k, seed in enumerate(seeds):
            res = minimize(lambda z: self._cost_grad(z, *args)[:2], seed, jac=True, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           options={"maxiter": int(p["max_iter"]) if k == 0 else max(5, int(p["max_iter"]) // 3),
                                    "ftol": 1e-4})
            z = res.x if np.all(np.isfinite(res.x)) else seed
            z = np.clip(z, lo, hi)
            Jz = self._cost_grad(z, *args)[0]
            nit += int(getattr(res, "nit", 0))
            if best is None or Jz < best[0] - 1e-6:
                best = (Jz, z, k)
                ok = bool(res.success)
        solve_ms = (time.perf_counter() - t0) * 1e3
        z = best[1]
        self.z_prev = z
        J, _, traj = self._cost_grad(z, *args)
        v = float(np.clip(z[0], max(r.min_v, v_init - r.max_dec * dt), min(vmax, v_init + r.max_acc * dt)))
        w = float(np.clip(z[N], max(-r.max_w, w_init - r.max_alpha * dt), min(r.max_w, w_init + r.max_alpha * dt)))
        info = {"cost": J, "success": ok, "iterations": nit, "seeds": len(seeds),
                "chosen_seed": ["warm", "right", "left"][best[2]], "solve_ms": solve_ms,
                "num_obstacles": int(len(obs_rad)), "blocked": False}
        return ControlOutput(v, w, trajectory=traj.tolist(),
                             candidates=[ref.tolist()], info=info)
