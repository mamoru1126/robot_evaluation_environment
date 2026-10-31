"""Result storage on disk.

results/
  <batch_id>/
    batch.json                 request + list of trials + status
    <run_id>/
      config.json              controller / scenario / perception of the trial
      timeseries.csv           20 Hz samples (robot, commands, humans)
      summary.json             metrics (see eval_core.metrics)
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import threading
import time
from pathlib import Path as FsPath
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_SAFE = re.compile(r"[^0-9A-Za-z_\-.]+")


def safe(s: str) -> str:
    return (_SAFE.sub("_", s or "").strip("_.") or "x")[:80]


def results_dir() -> FsPath:
    env = os.environ.get("EVAL_RESULTS_DIR")
    if env:
        return FsPath(env)
    here = FsPath(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "scenarios").is_dir():
            return parent / "results"
    return FsPath("/workspace/results")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _sanitize(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def dump_json(path: FsPath, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_sanitize(data), f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def load_json(path: FsPath) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ResultStore:
    def __init__(self, root: Optional[FsPath] = None):
        self.root = FsPath(root) if root else results_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    # ------------------------------------------------------------------
    def create_batch(self, name: str, spec: Dict[str, Any], trials: List[Dict[str, Any]],
                     backend: str = "") -> Dict[str, Any]:
        with self.lock:
            base = time.strftime("%Y%m%d-%H%M%S") + "_" + safe(name)
            bid = base
            k = 2
            while (self.root / bid).exists():
                bid = f"{base}_{k}"
                k += 1
            (self.root / bid).mkdir(parents=True)
            batch = {
                "id": bid, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running", "backend": backend, "request": spec,
                "trials": [t["id"] for t in trials], "completed": [],
            }
            dump_json(self.root / bid / "batch.json", batch)
            return batch

    def update_batch(self, batch: Dict[str, Any]) -> None:
        with self.lock:
            d = self.root / batch["id"]
            if d.is_dir():
                dump_json(d / "batch.json", batch)

    def save_run(self, batch_id: str, trial: Dict[str, Any], columns: Sequence[str],
                 rows: List[List[float]], summary: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            d = self.root / batch_id / safe(trial["id"])
            d.mkdir(parents=True, exist_ok=True)
            dump_json(d / "config.json", trial)
            with open(d / "timeseries.csv", "w", encoding="utf-8", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(columns)
                for r in rows:
                    wr.writerow([("nan" if not math.isfinite(v) else f"{v:.6g}") for v in r])
            summary = dict(summary)
            summary["run_id"] = d.name
            summary["batch_id"] = batch_id
            dump_json(d / "summary.json", summary)
            return {"batch_id": batch_id, "run_id": d.name, "summary": summary}

    # ------------------------------------------------------------------
    def list_batches(self) -> List[Dict[str, Any]]:
        out = []
        with self.lock:
            for d in sorted(self.root.iterdir(), reverse=True) if self.root.is_dir() else []:
                bj = d / "batch.json"
                if not bj.is_file():
                    continue
                try:
                    batch = load_json(bj)
                except (OSError, ValueError):
                    continue
                runs = []
                for rd in sorted(d.iterdir()):
                    sj = rd / "summary.json"
                    if rd.is_dir() and sj.is_file():
                        try:
                            runs.append(load_json(sj))
                        except (OSError, ValueError):
                            continue
                batch = {k: v for k, v in batch.items() if k != "request"}
                batch["runs"] = runs
                out.append(batch)
        return out

    def load_run(self, batch_id: str, run_id: str) -> Dict[str, Any]:
        d = self.root / safe(batch_id) / safe(run_id)
        if not d.is_dir():
            raise FileNotFoundError(f"{batch_id}/{run_id}")
        config = load_json(d / "config.json")
        summary = load_json(d / "summary.json")
        with open(d / "timeseries.csv", "r", encoding="utf-8") as f:
            rd = csv.reader(f)
            columns = next(rd)
            rows = [[float(v) if v not in ("", "nan") else float("nan") for v in r] for r in rd]
        return {"config": config, "summary": summary, "columns": columns,
                "data": np.asarray(rows, float).reshape(-1, len(columns))}

    def run_dir(self, batch_id: str, run_id: str) -> FsPath:
        return self.root / safe(batch_id) / safe(run_id)

    def delete_batch(self, batch_id: str) -> bool:
        d = self.root / safe(batch_id)
        with self.lock:
            if d.is_dir() and (d / "batch.json").is_file():
                shutil.rmtree(d)
                return True
        return False


def timeseries_to_json(run: Dict[str, Any], max_points: int = 1500) -> Dict[str, Any]:
    data = run["data"]
    step = max(1, int(math.ceil(len(data) / max_points)))
    sub = data[::step]
    if len(data) and step > 1 and not np.array_equal(sub[-1], data[-1]):
        sub = np.vstack([sub, data[-1:]])
    cols = {}
    for i, c in enumerate(run["columns"]):
        cols[c] = [None if not math.isfinite(v) else round(float(v), 4) for v in sub[:, i]]
    return cols


def summaries_to_csv(summaries: List[Dict[str, Any]]) -> str:
    from .metrics import METRIC_INFO
    keys = ["batch_id", "run_id", "controller", "scenario", "repeat", "perception", "backend", "outcome",
            "success"] + list(METRIC_INFO.keys())
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(keys)
    for s in summaries:
        wr.writerow(["" if s.get(k) is None else s.get(k) for k in keys])
    return buf.getvalue()
