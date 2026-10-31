"""Figure / report generation entry points.

These functions take only picklable arguments so the web server can run them
in a separate process (matplotlib is CPU heavy and would otherwise compete
with the ROS executor for the GIL).
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
import time
import zipfile
from pathlib import Path as FsPath
from typing import Any, Dict, List, Sequence, Tuple

from . import metrics as metrics_mod
from . import plots
from .storage import ResultStore, summaries_to_csv

RunRef = Tuple[str, str]   # (batch_id, run_id)


def _load(root: str, refs: Sequence[RunRef]) -> List[Dict[str, Any]]:
    store = ResultStore(FsPath(root))
    out = []
    for b, r in refs:
        try:
            out.append(store.load_run(b, r))
        except (OSError, ValueError, FileNotFoundError):
            continue
    return out


def group_value(summary: Dict[str, Any], group_by: str) -> str:
    if group_by == "scenario":
        return str(summary.get("scenario"))
    if group_by == "controller_scenario":
        return f"{summary.get('controller')} | {summary.get('scenario')}"
    if group_by == "controller_perception":
        return f"{summary.get('controller')} | {summary.get('perception')}"
    return str(summary.get("controller"))


def analysis(root: str, refs: Sequence[RunRef], group_by: str = "controller") -> Dict[str, Any]:
    store = ResultStore(FsPath(root))
    summaries = []
    for b, r in refs:
        try:
            s = json.loads((store.run_dir(b, r) / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s["group"] = group_value(s, group_by)
        summaries.append(s)
    groups = metrics_mod.aggregate(summaries, "group")
    order = []
    for s in summaries:
        if s["group"] not in order:
            order.append(s["group"])
    groups.sort(key=lambda g: order.index(g["group"]) if g["group"] in order else 0)
    for g in groups:
        first = next((s for s in summaries if s["group"] == g["group"]), {})
        g["controller"] = first.get("controller")
    return {"summaries": summaries, "groups": groups, "group_by": group_by}


def figure_png(root: str, refs: Sequence[RunRef], kind: str) -> bytes:
    runs = _load(root, refs)
    if not runs:
        raise FileNotFoundError("no runs")
    return plots.figure_for(kind, runs)


def compare_png(root: str, refs: Sequence[RunRef], metrics: Sequence[str], group_by: str = "controller") -> bytes:
    a = analysis(root, refs, group_by)
    if not a["groups"]:
        raise FileNotFoundError("no runs")
    mets = list(metrics) or ["travel_time", "min_clearance", "personal_space_time", "success_rate"]
    color_key = "controller" if group_by in ("controller", "controller_scenario", "controller_perception") else None
    return plots.comparison_figure(a["groups"], mets, name_key="group", color_key=color_key)


def build_export(root: str, refs: Sequence[RunRef], group_by: str = "controller") -> bytes:
    """ZIP with summary CSV, per-run time series, figures and an HTML report."""
    a = analysis(root, refs, group_by)
    runs = _load(root, refs)
    buf = io.BytesIO()
    figs: List[Tuple[str, str]] = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.csv", summaries_to_csv(a["summaries"]))
        # aggregated table
        lines = ["group,n,success_rate," + ",".join(f"{m}_mean,{m}_std" for m in metrics_mod.METRIC_INFO)]
        for g in a["groups"]:
            row = [g["group"], str(g["n"]), f"{g['success_rate']:.3f}"]
            for m in metrics_mod.METRIC_INFO:
                st = g.get(m)
                row += [f"{st['mean']:.4g}", f"{st['std']:.4g}"] if st else ["", ""]
            lines.append(",".join(f'"{c}"' if "," in c else c for c in row))
        z.writestr("groups.csv", "\n".join(lines) + "\n")
        for run in runs:
            s = run["summary"]
            name = f"timeseries/{s.get('batch_id')}__{s.get('run_id')}.csv"
            cols = run["columns"]
            body = [",".join(cols)]
            for r in run["data"]:
                body.append(",".join("nan" if not math.isfinite(v) else f"{v:.6g}" for v in r))
            z.writestr(name, "\n".join(body) + "\n")
        if runs:
            figs.append(("figures/compare.png", "Comparison"))
            z.writestr("figures/compare.png", compare_png(root, refs, ["travel_time", "min_clearance",
                                                                       "personal_space_time", "success_rate"],
                                                          group_by))
            figs.append(("figures/compare_more.png", "Comparison (more)"))
            z.writestr("figures/compare_more.png", compare_png(root, refs, ["min_ttc", "max_lateral_deviation",
                                                                            "rms_jerk", "compute_ms_mean"],
                                                               group_by))
            if len(runs) <= 12:
                for kind, title in (("trajectory", "Trajectories"), ("clearance", "Clearance to humans"),
                                    ("velocity", "Velocity")):
                    fn = f"figures/all_{kind}.png"
                    z.writestr(fn, plots.figure_for(kind, runs))
                    figs.append((fn, title))
            for run in runs[:40]:
                s = run["summary"]
                stem = f"{s.get('run_id')}"
                for kind in ("trajectory", "clearance"):
                    fn = f"figures/runs/{stem}_{kind}.png"
                    z.writestr(fn, plots.figure_for(kind, [run]))
                    figs.append((fn, f"{stem} {kind}"))
        z.writestr("report.html", _report_html(a, figs))
    return buf.getvalue()


def _report_html(a: Dict[str, Any], figs: List[Tuple[str, str]]) -> str:
    h = html.escape
    rows = []
    keys = ["travel_time", "min_clearance", "min_center_distance", "personal_space_time", "min_ttc",
            "max_lateral_deviation", "collisions", "compute_ms_mean"]
    head = "".join(f"<th>{h(metrics_mod.METRIC_INFO[k][0])}<br><small>[{h(metrics_mod.METRIC_INFO[k][1])}]</small></th>"
                   for k in keys)
    for g in a["groups"]:
        cells = []
        for k in keys:
            st = g.get(k)
            cells.append(f"<td>{st['mean']:.3f} ± {st['std']:.3f}</td>" if st else "<td>-</td>")
        rows.append(f"<tr><td>{h(str(g['group']))}</td><td>{g['n']}</td>"
                    f"<td>{100 * g['success_rate']:.0f}%</td>{''.join(cells)}</tr>")
    run_rows = []
    for s in a["summaries"]:
        def f(k):
            v = s.get(k)
            return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else h(str(v)))
        run_rows.append("<tr>" + "".join(f"<td>{x}</td>" for x in (
            h(s.get("run_id", "")), h(str(s.get("controller"))), h(str(s.get("scenario"))), h(str(s.get("perception"))),
            h(str(s.get("outcome"))), f("travel_time"), f("min_clearance"), f("personal_space_time"),
            f("collisions"))) + "</tr>")
    fig_html = "".join(f'<figure><img src="{h(fn)}" alt="{h(t)}"><figcaption>{h(t)}</figcaption></figure>'
                       for fn, t in figs)
    return f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>評価レポート</title>
<style>body{{font-family:system-ui,'Noto Sans JP',sans-serif;margin:24px;color:#1d2433}}
table{{border-collapse:collapse;margin:12px 0;font-size:13px}}td,th{{border:1px solid #ccd;padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}th{{background:#eef1f6}}figure{{margin:16px 0}}
img{{max-width:100%;border:1px solid #dde}}figcaption{{font-size:12px;color:#556}}</style></head><body>
<h1>ロボット評価レポート</h1><p>生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')} / グループ: {h(a['group_by'])}</p>
<h2>集計 (平均 ± 標準偏差)</h2><table><tr><th>グループ</th><th>試行数</th><th>成功率</th>{head}</tr>{''.join(rows)}</table>
<h2>試行一覧</h2><table><tr><th>run</th><th>制御器</th><th>シナリオ</th><th>認識</th><th>結果</th><th>移動時間[s]</th>
<th>最小クリアランス[m]</th><th>PS侵入[s]</th><th>接触</th></tr>{''.join(run_rows)}</table>
<h2>グラフ</h2>{fig_html}</body></html>"""
