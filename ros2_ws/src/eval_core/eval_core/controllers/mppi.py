"""Model Predictive Path Integral control (Williams et al., 2017).

Gaussian perturbations of the nominal control sequence are rolled out with a
unicycle model (velocity / acceleration limits enforced during the rollout),
each rollout gets a cost, and the nominal sequence is updated with the
soft-min (exp(-S/lambda)) weighted average of the perturbations.  The first
control is applied and the sequence is shifted for warm-starting.
Critics are similar in spirit to Nav2's MPPI controller (path follow /
path angle / prefer speed / obstacle / goal progress / smoothness).
"""
from __future__ import annotations

import numpy as np

from ..geometry import wrap_angle
from .base import BaseController, ControlOutput, ControllerInput, Param


class MPPIController(BaseController):
    name = "mppi"
    label = "MPPI (Model Predictive Path Integral)"
    description = ("ランダムにばらつかせた制御系列を多数ロールアウトし、コストの指数重み付き平均で"
                   "制御系列を更新するサンプリング型 MPC。障害物回避と経路追従を同時に行う。")

    PARAMS = {
        "target_speed": Param(0.8, 0.05, 2.0, 0.05, "目標速度 [m/s]"),
        "num_samples": Param(800, 50, 5000, 50, "サンプル数"),
        "horizon_steps": Param(30, 5, 80, 1, "予測ステップ数"),
        "model_dt": Param(0.1, 0.02, 0.3, 0.01, "予測刻み [s]"),
        "temperature": Param(0.05, 0.005, 10.0, 0.005, "温度 λ", "小さいほど低コストのサンプルに集中する"),
        "noise_v": Param(0.2, 0.01, 1.0, 0.01, "ノイズ σv [m/s]"),
        "noise_w": Param(0.6, 0.01, 3.0, 0.01, "ノイズ σω [rad/s]"),
        "noise_smoothing": Param(0.8, 0.0, 0.98, 0.02, "ノイズの時間相関 (0=白色)",
                                 "大きいほど持続的な旋回 (回り込み) をサンプルしやすい"),
        "w_path": Param(1.0, 0.0, 50.0, 0.1, "重み: 経路からの横ずれ²"),
        "w_heading": Param(0.5, 0.0, 50.0, 0.1, "重み: 経路方向との角度差²"),
        "w_speed": Param(1.0, 0.0, 50.0, 0.1, "重み: 目標速度との差²"),
        "w_goal": Param(1.0, 0.0, 50.0, 0.1, "重み: 終端での残り距離"),
        "w_obstacle": Param(8.0, 0.0, 200.0, 0.5, "重み: 障害物への近さ"),
        "w_smooth": Param(0.1, 0.0, 50.0, 0.05, "重み: ω の変化²"),
        "collision_cost": Param(100.0, 1.0, 1e4, 1.0, "衝突ステップのコスト"),
        "obstacle_influence": Param(1.0, 0.05, 5.0, 0.05, "障害物コストが効く距離 [m]"),
        "safety_margin": Param(0.05, 0.0, 1.0, 0.01, "衝突とみなす余裕 [m]"),
        "pass_side": Param("right", label="すれ違う側の優先", type="choice", choices=["right", "left", "none"],
                           help="左右対称な状況で回避方向を決めるための小さなバイアス"),
        "w_side": Param(0.5, 0.0, 10.0, 0.1, "重み: 優先しない側への横ずれ"),
        "seed": Param(0, 0, 1_000_000, 1, "乱数シード (0=毎回ランダム)"),
    }

    def reset(self) -> None:
        self.U = None
        seed = int(self.p.get("seed", 0))
        self.rng = np.random.default_rng(seed if seed > 0 else None)

    # ------------------------------------------------------------------
    def _rollout(self, x0, y0, th0, v0, w0, U, dt):
        """U: (K, T, 2) -> clamped controls (K, T, 2), states (K, T, 3)."""
        r = self.robot
        K, T, _ = U.shape
        Uc = np.empty_like(U)
        X = np.empty((K, T, 3))
        x = np.full(K, x0)
        y = np.full(K, y0)
        th = np.full(K, th0)
        vp = np.full(K, v0)
        wp = np.full(K, w0)
        vmax = min(self.p["target_speed"], r.max_v)
        for k in range(T):
            v = np.clip(U[:, k, 0], np.maximum(r.min_v, vp - r.max_dec * dt), np.minimum(vmax, vp + r.max_acc * dt))
            w = np.clip(U[:, k, 1], np.maximum(-r.max_w, wp - r.max_alpha * dt),
                        np.minimum(r.max_w, wp + r.max_alpha * dt))
            Uc[:, k, 0] = v
            Uc[:, k, 1] = w
            x = x + v * np.cos(th) * dt
            y = y + v * np.sin(th) * dt
            th = th + w * dt
            X[:, k, 0] = x
            X[:, k, 1] = y
            X[:, k, 2] = th
            vp, wp = v, w
        return Uc, X

    def compute(self, inp: ControllerInput) -> ControlOutput:
        st = inp.state
        p = self.p
        K = int(p["num_samples"])
        T = int(p["horizon_steps"])
        dt = float(p["model_dt"])
        v0, w0 = inp.last_cmd

        if self.U is None or self.U.shape[0] != T:
            self.U = np.zeros((T, 2))
            self.U[:, 0] = p["target_speed"]
        else:
            # warm start: shift by the elapsed control periods
            shift = max(1, int(round(inp.dt / dt)))
            self.U = np.concatenate([self.U[shift:], np.repeat(self.U[-1:], min(shift, T), axis=0)])[:T]

        white = self.rng.standard_normal((K, T, 2))
        a = float(p["noise_smoothing"])
        if a > 0.0:
            noise = np.empty_like(white)
            noise[:, 0] = white[:, 0]
            c = np.sqrt(1.0 - a * a)
            for k in range(1, T):
                noise[:, k] = a * noise[:, k - 1] + c * white[:, k]
        else:
            noise = white
        noise = noise * np.array([p["noise_v"], p["noise_w"]])
        noise[0] = 0.0                                  # nominal sequence
        Usamp = self.U[None, :, :] + noise
        Usamp[1, :, 0] = 0.0                            # a braking sample
        Usamp[1, :, 1] = 0.0
        Uc, X = self._rollout(st.x, st.y, st.yaw, v0, w0, Usamp, dt)
        eps = Uc - self.U[None]

        xy = X[..., :2]
        t = np.arange(1, T + 1) * dt
        path = inp.path
        slat = path.signed_lateral(xy)                   # (K, T), left positive
        lat = np.abs(slat)
        s = path.progress(xy)
        head = wrap_angle(X[..., 2] - path.heading_at(s))
        clear = inp.obstacles.clearance(xy, t[None, :])
        infl = p["obstacle_influence"]
        obs = p["w_obstacle"] * np.clip((infl - clear) / infl, 0.0, None) ** 2
        coll = clear <= p["safety_margin"]
        obs = np.where(coll, p["collision_cost"], obs)
        dw = np.diff(np.concatenate([np.full((K, 1), w0), Uc[..., 1]], axis=1), axis=1)
        step_cost = (p["w_path"] * lat ** 2 + p["w_heading"] * head ** 2
                     + p["w_speed"] * (Uc[..., 0] - p["target_speed"]) ** 2
                     + obs + p["w_smooth"] * dw ** 2)
        if p["pass_side"] != "none" and p["w_side"] > 0:
            sgn = 1.0 if p["pass_side"] == "right" else -1.0   # penalise the other side
            step_cost = step_cost + p["w_side"] * np.maximum(0.0, sgn * slat)
        S = step_cost.sum(axis=1) * dt + p["w_goal"] * (path.length - s[:, -1])

        Smin = float(S.min())
        weights = np.exp(-(S - Smin) / max(p["temperature"], 1e-6))
        weights /= weights.sum()
        U_new = self.U + np.einsum("k,ktc->tc", weights, eps)
        # light smoothing of the control sequence
        if T >= 3:
            sm = U_new.copy()
            sm[1:-1] = 0.25 * U_new[:-2] + 0.5 * U_new[1:-1] + 0.25 * U_new[2:]
            U_new = sm
        self.U = U_new

        Uopt, Xopt = self._rollout(st.x, st.y, st.yaw, v0, w0, U_new[None], dt)
        v, w = float(Uopt[0, 0, 0]), float(Uopt[0, 0, 1])
        opt_clear = inp.obstacles.clearance(Xopt[0, :, :2], t)
        blocked = bool(np.all(coll[:, : max(1, T // 3)].any(axis=1)))
        # show a handful of the best samples
        best = np.argsort(S)[:12]
        cands = [X[i, ::2, :2].tolist() for i in best]
        info = {"min_cost": Smin, "min_clearance": float(opt_clear.min()), "blocked": blocked,
                "ess": float(1.0 / np.sum(weights ** 2))}
        return ControlOutput(v, w, trajectory=Xopt[0, :, :2].tolist(), candidates=cands, info=info)
