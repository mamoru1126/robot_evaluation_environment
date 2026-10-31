"""Server-side figures (matplotlib, PNG) for single runs and comparisons.

Labels are in English so that the figures render identically on any system
(no CJK font required) and can be pasted into papers / reports directly.
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from .metrics import METRIC_INFO  # noqa: E402

CONTROLLER_COLORS = {
    "pure_pursuit": "#1f77b4", "dwpp": "#ff7f0e", "dwa": "#2ca02c", "mppi": "#d62728", "mpc": "#9467bd",
}
EXTRA_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
HUMAN_COLORS = ["#e6550d", "#3182bd", "#31a354", "#756bb1"]

METRIC_EN = {
    "travel_time": ("Travel time START->GOAL", "s"),
    "time_to_goal": ("Time to GOAL", "s"),
    "mean_speed": ("Mean speed", "m/s"),
    "max_speed": ("Max speed", "m/s"),
    "path_length": ("Path length", "m"),
    "max_lateral_deviation": ("Max lateral deviation", "m"),
    "rms_lateral_deviation": ("RMS lateral deviation", "m"),
    "min_center_distance": ("Min distance to humans (center)", "m"),
    "min_clearance": ("Min clearance to humans", "m"),
    "personal_space_time": ("Time in personal space", "s"),
    "intimate_space_time": ("Time in intimate space", "s"),
    "min_ttc": ("Min time-to-collision", "s"),
    "collisions": ("Collisions", "count"),
    "stops": ("Stops", "count"),
    "stop_time": ("Stop time", "s"),
    "blocked_time": ("Blocked time", "s"),
    "rms_acc": ("RMS acceleration", "m/s^2"),
    "rms_jerk": ("RMS jerk", "m/s^3"),
    "rms_ang_acc": ("RMS angular acceleration", "rad/s^2"),
    "mean_abs_w": ("Mean |omega|", "rad/s"),
    "compute_ms_mean": ("Compute time (mean)", "ms"),
    "compute_ms_max": ("Compute time (max)", "ms"),
}


def controller_color(name: str, order: Sequence[str] = ()) -> str:
    if name in CONTROLLER_COLORS:
        return CONTROLLER_COLORS[name]
    extras = [n for n in order if n not in CONTROLLER_COLORS]
    i = extras.index(name) if name in extras else abs(hash(name))
    return EXTRA_COLORS[i % len(EXTRA_COLORS)]


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return buf.getvalue()


def _col(run: Dict[str, Any], name: str) -> np.ndarray:
    return run["data"][:, run["columns"].index(name)]


def _humans(run: Dict[str, Any]) -> List[int]:
    out = []
    sc = run["config"]["scenario"]
    for i, h in enumerate(sc["humans"]):
        if h.get("enabled", True) and f"h{i + 1}_x" in run["columns"]:
            out.append(i)
    return out


def _run_label(run: Dict[str, Any]) -> str:
    s = run["summary"]
    return f"{s.get('controller')} / {s.get('scenario')} #{s.get('repeat', 1)}"


# ----------------------------------------------------------------------------
# single run
# ----------------------------------------------------------------------------
def trajectory_figure(runs: List[Dict[str, Any]], title: str = "Trajectories") -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    names = [r["summary"].get("controller", "") for r in runs]
    course = runs[0]["config"]["scenario"]["course"] if runs else {"start_x": 0, "goal_x": 10, "y": 0}
    ymin, ymax = -3.0, 3.0
    for r in runs:
        for i in _humans(r):
            hx, hy = _col(r, f"h{i + 1}_x"), _col(r, f"h{i + 1}_y")
            ok = np.isfinite(hx)
            if np.any(ok):
                ymin, ymax = min(ymin, np.nanmin(hy) - 0.5), max(ymax, np.nanmax(hy) + 0.5)
    for k, r in enumerate(runs):
        x, y = _col(r, "x"), _col(r, "y")
        name = r["summary"].get("controller", "")
        color = controller_color(name, names)
        ax.plot(x, y, color=color, lw=2, label=_run_label(r) if len(runs) > 1 else f"robot ({name})")
        rad = r["config"].get("robot_radius", 0.25)
        t = _col(r, "t")
        for tk in np.arange(0, t[-1] if len(t) else 0, 2.0):
            j = int(np.searchsorted(t, tk))
            if j < len(t):
                ax.add_patch(Circle((x[j], y[j]), rad, fill=False, ec=color, lw=0.8, alpha=0.6))
        for i in _humans(r):
            hx, hy = _col(r, f"h{i + 1}_x"), _col(r, f"h{i + 1}_y")
            ax.plot(hx, hy, color=HUMAN_COLORS[i % len(HUMAN_COLORS)], lw=1.2, ls="--",
                    alpha=0.9 if len(runs) == 1 else 0.35,
                    label=f"human {i + 1}" if k == 0 else None)
            hs = r["summary"].get("humans") or []
            for hd in hs:
                if hd.get("id") == f"human_{i + 1}" and hd.get("robot_at_min") and len(runs) == 1:
                    (rx, ry), (px, py) = hd["robot_at_min"], hd["human_at_min"]
                    ax.plot([rx, px], [ry, py], color="k", lw=0.8)
                    ax.annotate(f"{hd['min_center_distance']:.2f} m", ((rx + px) / 2, (ry + py) / 2),
                                fontsize=8, ha="left")
                    ax.add_patch(Circle((px, py), r["config"]["scenario"]["humans"][i]["radius"], fill=False,
                                        ec=HUMAN_COLORS[i % len(HUMAN_COLORS)], lw=1))
    for xl, lab, c in ((course["start_x"], "START", "#2ca02c"), (course["goal_x"], "GOAL", "#d62728")):
        ax.axvline(xl, color=c, lw=2, alpha=0.7)
        ax.text(xl, ymax, lab, color=c, ha="center", va="bottom", fontsize=9)
    ax.axhline(course["y"], color="0.6", lw=0.8, ls=":")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    return _png(fig)


def distance_figure(runs: List[Dict[str, Any]], mode: str = "clearance") -> bytes:
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    names = [r["summary"].get("controller", "") for r in runs]
    ps = isp = None
    for r in runs:
        t = _col(r, "t")
        x, y = _col(r, "x"), _col(r, "y")
        sc = r["config"]["scenario"]
        rad = r["config"].get("robot_radius", 0.25)
        ps = sc["evaluation"]["personal_space"]
        isp = sc["evaluation"]["intimate_space"]
        color = controller_color(r["summary"].get("controller", ""), names)
        for i in _humans(r):
            d = np.hypot(_col(r, f"h{i + 1}_x") - x, _col(r, f"h{i + 1}_y") - y)
            if mode == "clearance":
                d = d - rad - sc["humans"][i]["radius"]
            ls = ["-", "--", ":", "-."][i % 4]
            lab = (f"{_run_label(r)} h{i + 1}" if len(runs) > 1 else f"human {i + 1}")
            ax.plot(t, d, color=color if len(runs) > 1 else HUMAN_COLORS[i % len(HUMAN_COLORS)], ls=ls,
                    lw=1.6, label=lab)
    if mode == "clearance":
        ax.axhline(0.0, color="r", lw=1, label="contact")
        ax.set_ylabel("clearance to human [m]  (surface distance)")
    else:
        if ps:
            ax.axhline(ps, color="orange", lw=1, ls="--", label=f"personal space {ps} m")
        if isp:
            ax.axhline(isp, color="r", lw=1, ls="--", label=f"intimate space {isp} m")
        ax.set_ylabel("distance to human [m]  (center to center)")
    ax.set_xlabel("time [s]")
    ax.set_ylim(bottom=min(-0.2, ax.get_ylim()[0]), top=min(ax.get_ylim()[1], 8.0))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Distance to humans")
    return _png(fig)


def velocity_figure(runs: List[Dict[str, Any]]) -> bytes:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, layout="constrained")
    names = [r["summary"].get("controller", "") for r in runs]
    for r in runs:
        t = _col(r, "t")
        color = controller_color(r["summary"].get("controller", ""), names)
        lab = _run_label(r) if len(runs) > 1 else "measured"
        a1.plot(t, _col(r, "v"), color=color, lw=1.6, label=lab)
        a2.plot(t, _col(r, "w"), color=color, lw=1.6, label=lab)
        if len(runs) == 1:
            a1.plot(t, _col(r, "cmd_v"), color="0.4", lw=1, ls="--", label="command")
            a2.plot(t, _col(r, "cmd_w"), color="0.4", lw=1, ls="--", label="command")
    a1.set_ylabel("v [m/s]")
    a2.set_ylabel("omega [rad/s]")
    a2.set_xlabel("time [s]")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=7, ncol=2)
    a1.set_title("Velocity")
    return _png(fig)


# ----------------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------------
def comparison_figure(groups: List[Dict[str, Any]], metrics: Sequence[str] = (
        "travel_time", "min_clearance", "personal_space_time", "success_rate"),
        name_key: str = "group", color_key: Optional[str] = "controller") -> bytes:
    """Bar charts (mean +- std) of the given metrics for each group."""
    n = len(metrics)
    cols = 2 if n > 1 else 1
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), layout="constrained", squeeze=False)
    ckeys = [str(g.get(color_key, "")) for g in groups] if color_key else []
    palette = list(CONTROLLER_COLORS.values()) + EXTRA_COLORS
    for ax, m in zip(axes.ravel(), metrics):
        means, stds, colors, labels = [], [], [], []
        for k, g in enumerate(groups):
            if m == "success_rate":
                means.append(100.0 * g.get("success_rate", 0.0))
                stds.append(0.0)
            else:
                st = g.get(m)
                means.append(st["mean"] if st else np.nan)
                stds.append(st["std"] if st else 0.0)
            colors.append(controller_color(ckeys[k], ckeys) if color_key else palette[k % len(palette)])
            labels.append(f"{g.get(name_key)}\n(n={g['n']})")
        xs = np.arange(len(groups))
        bars = ax.bar(xs, means, yerr=stds, color=colors, capsize=4, alpha=0.9)
        for b, v in zip(bars, means):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), ha="center",
                            va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8)
        if m == "success_rate":
            ax.set_title("Success rate [%]")
            ax.set_ylim(0, 110)
        else:
            name, unit = METRIC_EN.get(m, (m, ""))
            ax.set_title(f"{name} [{unit}]")
        ax.grid(axis="y", alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return _png(fig)


def figure_for(kind: str, runs: List[Dict[str, Any]]) -> bytes:
    if kind == "trajectory":
        return trajectory_figure(runs, "Trajectory" if len(runs) == 1 else "Trajectories")
    if kind == "distance":
        return distance_figure(runs, "center")
    if kind == "clearance":
        return distance_figure(runs, "clearance")
    if kind == "velocity":
        return velocity_figure(runs)
    raise KeyError(kind)
