"""Evaluation metrics computed from the recorded time series of one trial.

Distances to humans are evaluated with ground-truth positions:

* center distance  : ||p_robot - p_human||
* clearance        : center distance - r_robot - r_human  (<= 0 means contact)
* personal / intimate space : time during which the center distance is below
  the thresholds of the scenario (Hall's proxemics, 1.2 m / 0.45 m default)
* TTC              : time to collision assuming both keep their current velocity
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

METRIC_INFO = {
    "travel_time": ("移動時間 START→GOAL", "s"),
    "time_to_goal": ("試行開始から GOAL まで", "s"),
    "mean_speed": ("平均速度 (START→GOAL)", "m/s"),
    "max_speed": ("最大速度", "m/s"),
    "path_length": ("走行距離 (START→GOAL)", "m"),
    "max_lateral_deviation": ("最大横ずれ", "m"),
    "rms_lateral_deviation": ("横ずれ RMS", "m"),
    "min_center_distance": ("人との最小距離 (中心間)", "m"),
    "min_clearance": ("人との最小クリアランス (表面間)", "m"),
    "personal_space_time": ("パーソナルスペース侵入時間", "s"),
    "intimate_space_time": ("密接距離侵入時間", "s"),
    "min_ttc": ("最小 TTC", "s"),
    "collisions": ("接触回数", "回"),
    "stops": ("停止回数", "回"),
    "stop_time": ("停止時間合計", "s"),
    "blocked_time": ("衝突回避のための停止時間", "s"),
    "rms_acc": ("加速度 RMS", "m/s²"),
    "rms_jerk": ("ジャーク RMS", "m/s³"),
    "rms_ang_acc": ("角加速度 RMS", "rad/s²"),
    "mean_abs_w": ("平均 |ω|", "rad/s"),
    "compute_ms_mean": ("制御計算時間 平均", "ms"),
    "compute_ms_max": ("制御計算時間 最大", "ms"),
}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return v if math.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _crossing(t: np.ndarray, x: np.ndarray, line: float) -> Optional[float]:
    if len(x) == 0:
        return None
    if x[0] >= line:
        return float(t[0])
    idx = np.nonzero((x[:-1] < line) & (x[1:] >= line))[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    a = (line - x[i]) / max(x[i + 1] - x[i], 1e-12)
    return float(t[i] + a * (t[i + 1] - t[i]))


def _duration(t: np.ndarray, cond: np.ndarray) -> float:
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    return float(np.sum(dt * cond[:-1]))


def _episodes(t: np.ndarray, cond: np.ndarray, min_len: float) -> int:
    n = 0
    start = None
    for ti, c in zip(t, cond):
        if c and start is None:
            start = ti
        elif not c and start is not None:
            if ti - start >= min_len:
                n += 1
            start = None
    if start is not None and t[-1] - start >= min_len:
        n += 1
    return n


def ttc_series(p: np.ndarray, u: np.ndarray, r: float, cap: float = 10.0) -> np.ndarray:
    """Time to collision for relative positions p (n,2) and velocities u (n,2)."""
    a = np.sum(u * u, axis=1)
    b = 2.0 * np.sum(p * u, axis=1)
    c = np.sum(p * p, axis=1) - r * r
    out = np.full(len(p), cap)
    inside = c <= 0
    out[inside] = 0.0
    disc = b * b - 4 * a * c
    ok = (~inside) & (a > 1e-9) & (disc >= 0)
    tau = np.full(len(p), np.inf)
    tau[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2 * a[ok])
    good = ok & (tau >= 0)
    out[good] = np.minimum(tau[good], cap)
    return out


def compute(data: np.ndarray, columns: Sequence[str], scenario: Dict[str, Any], robot_radius: float,
            outcome: str, collisions: int = 0) -> Dict[str, Any]:
    col = {c: i for i, c in enumerate(columns)}
    out: Dict[str, Any] = {"outcome": outcome, "collisions": int(collisions),
                           "success": outcome == "goal" and collisions == 0, "num_samples": int(len(data))}
    if len(data) < 2:
        return out
    t = data[:, col["t"]]
    x, y = data[:, col["x"]], data[:, col["y"]]
    yaw, v, w = data[:, col["yaw"]], data[:, col["v"]], data[:, col["w"]]
    c = scenario["course"]
    t_start = _crossing(t, x, c["start_x"])
    t_goal = _crossing(t, x, c["goal_x"])
    out["t_start_cross"] = t_start
    out["t_goal_cross"] = t_goal
    out["travel_time"] = (t_goal - t_start) if (t_start is not None and t_goal is not None) else None
    out["time_to_goal"] = t_goal
    t0 = t_start if t_start is not None else t[0]
    t1 = t_goal if t_goal is not None else t[-1]
    win = (t >= t0) & (t <= t1)
    if np.count_nonzero(win) < 2:
        win = np.ones(len(t), bool)
    xw, yw, tw, vw, ww = x[win], y[win], t[win], v[win], w[win]
    out["path_length"] = float(np.sum(np.hypot(np.diff(xw), np.diff(yw))))
    if out["travel_time"]:
        out["mean_speed"] = (c["goal_x"] - c["start_x"]) / out["travel_time"]
    out["max_speed"] = float(np.max(np.abs(v)))
    lat = yw - c["y"]
    out["max_lateral_deviation"] = float(np.max(np.abs(lat)))
    out["rms_lateral_deviation"] = float(np.sqrt(np.mean(lat ** 2)))
    out["mean_abs_w"] = float(np.mean(np.abs(ww)))
    if len(tw) >= 3:
        dt = np.diff(tw)
        dt[dt <= 1e-6] = 1e-6
        acc = np.diff(vw) / dt
        out["rms_acc"] = float(np.sqrt(np.mean(acc ** 2)))
        out["rms_ang_acc"] = float(np.sqrt(np.mean((np.diff(ww) / dt) ** 2)))
        if len(acc) >= 2:
            jerk = np.diff(acc) / dt[1:]
            out["rms_jerk"] = float(np.sqrt(np.mean(jerk ** 2)))
    stopped = np.abs(vw) < 0.05
    out["stops"] = _episodes(tw, stopped, 0.5)
    out["stop_time"] = _duration(tw, stopped)
    if "blocked" in col:
        out["blocked_time"] = _duration(t, data[:, col["blocked"]] > 0.5)
    if "compute_ms" in col:
        cm = data[:, col["compute_ms"]]
        cm = cm[np.isfinite(cm)]
        if len(cm):
            out["compute_ms_mean"] = float(np.mean(cm))
            out["compute_ms_max"] = float(np.max(cm))

    # ---- humans --------------------------------------------------------
    ev = scenario.get("evaluation", {})
    ps = float(ev.get("personal_space", 1.2))
    isp = float(ev.get("intimate_space", 0.45))
    humans_out: List[Dict[str, Any]] = []
    any_personal = np.zeros(len(t), bool)
    any_intimate = np.zeros(len(t), bool)
    min_center = math.inf
    min_clear = math.inf
    min_ttc = math.inf
    vr = np.stack([v * np.cos(yaw), v * np.sin(yaw)], -1)
    for i, h in enumerate(scenario["humans"]):
        key = f"h{i + 1}_x"
        if key not in col or not h.get("enabled", True):
            continue
        hx = data[:, col[f"h{i + 1}_x"]]
        hy = data[:, col[f"h{i + 1}_y"]]
        hvx = data[:, col[f"h{i + 1}_vx"]]
        hvy = data[:, col[f"h{i + 1}_vy"]]
        ok = np.isfinite(hx) & np.isfinite(hy)
        if not np.any(ok):
            continue
        d = np.full(len(t), np.inf)
        d[ok] = np.hypot(hx[ok] - x[ok], hy[ok] - y[ok])
        clear = d - h["radius"] - robot_radius
        k = int(np.argmin(d))
        p = np.stack([hx - x, hy - y], -1)
        u = np.stack([np.nan_to_num(hvx), np.nan_to_num(hvy)], -1) - vr
        ttc = ttc_series(p[ok], u[ok], h["radius"] + robot_radius)
        personal = d < ps
        intimate = d < isp
        any_personal |= personal
        any_intimate |= intimate
        hd = {
            "id": h.get("id", f"human_{i + 1}"),
            "pattern": h["pattern"],
            "min_center_distance": float(d[k]),
            "min_clearance": float(clear[k]),
            "t_min_distance": float(t[k]),
            "robot_at_min": [float(x[k]), float(y[k])],
            "human_at_min": [float(hx[k]), float(hy[k])],
            "personal_space_time": _duration(t, personal),
            "intimate_space_time": _duration(t, intimate),
            "min_ttc": float(np.min(ttc)) if len(ttc) else None,
        }
        humans_out.append({k2: _clean(v2) if not isinstance(v2, list) else v2 for k2, v2 in hd.items()})
        min_center = min(min_center, float(d[k]))
        min_clear = min(min_clear, float(clear[k]))
        if len(ttc):
            min_ttc = min(min_ttc, float(np.min(ttc)))
    out["humans"] = humans_out
    out["min_center_distance"] = min_center if math.isfinite(min_center) else None
    out["min_clearance"] = min_clear if math.isfinite(min_clear) else None
    out["min_ttc"] = min_ttc if math.isfinite(min_ttc) else None
    out["personal_space_time"] = _duration(t, any_personal)
    out["intimate_space_time"] = _duration(t, any_intimate)
    out["personal_space"] = ps
    out["intimate_space"] = isp
    return {k: (_clean(v) if not isinstance(v, (list, dict)) else v) for k, v in out.items()}


def aggregate(summaries: List[Dict[str, Any]], group_key: str = "controller") -> List[Dict[str, Any]]:
    """Mean / std / min / max of numeric metrics per group."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in summaries:
        groups.setdefault(str(s.get(group_key, "?")), []).append(s)
    out = []
    for g, items in groups.items():
        row: Dict[str, Any] = {group_key: g, "n": len(items),
                               "success_rate": float(np.mean([1.0 if s.get("success") else 0.0 for s in items])),
                               "label": items[0].get("controller_label", g) if group_key == "controller" else g}
        for m in METRIC_INFO:
            vals = [s.get(m) for s in items]
            vals = np.array([float(v) for v in vals if isinstance(v, (int, float)) and v is not None
                             and math.isfinite(float(v))])
            if len(vals):
                row[m] = {"mean": float(vals.mean()), "std": float(vals.std()), "min": float(vals.min()),
                          "max": float(vals.max()), "n": int(len(vals))}
        out.append(row)
    return out
