"""Web server (Starlette + uvicorn): evaluation control, live view,
results, analysis and export — everything from the browser.

REST API (JSON):

  GET  /api/info                      static information (robot, world, enums, defaults)
  GET  /api/status                    backend availability + current batch / trial
  GET  /api/stream                    Server-Sent Events, live state at 10 Hz
  GET  /api/controllers               available controllers (built-in + custom) and load errors
  GET  /api/scenarios                 scenario presets
  POST /api/scenarios                 save a preset            DELETE /api/scenarios/{name}
  POST /api/scenario/preview          normalised scenario + human routes for drawing
  POST /api/run                       start a batch            POST /api/stop
  GET  /api/results                   batches with run summaries
  GET  /api/results/{batch}/{run}     one run incl. time series
  DELETE /api/results/{batch}
  POST /api/analysis                  aggregated statistics of selected runs
  GET  /api/figure.png?kind=..&runs=  matplotlib figure of runs (trajectory/distance/clearance/velocity)
  GET  /api/compare.png?runs=..&metrics=..&group_by=..
  POST /api/export                    ZIP (CSV + PNG figures + HTML report)
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import json
import math
import multiprocessing
import os
import threading
import time
from pathlib import Path as FsPath
from typing import Any, Dict, List, Optional, Tuple

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import controllers as ctrl_registry
from .. import report
from ..metrics import METRIC_INFO
from ..planner_core import PERCEPTION_MODES
from ..scenario import (DEFAULT_SCENARIO, PATTERN_LABELS, TRIGGER_LABELS, delete_preset, evaluation_path,
                        human_route, list_presets, normalize_scenario, robot_start_pose, save_preset)
from ..specs import RobotSpec, WorldSpec
from ..storage import ResultStore, _sanitize, safe, timeseries_to_json

STATIC_DIR = FsPath(__file__).resolve().parent / "static"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-3.0.1.min.js"


def _json(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(_sanitize(data), status_code=status)


def find_plotly_js() -> Optional[FsPath]:
    env = os.environ.get("EVAL_PLOTLY_JS")
    if env and FsPath(env).is_file():
        return FsPath(env)
    spec = importlib.util.find_spec("plotly")
    if spec and spec.origin:
        cand = FsPath(spec.origin).parent / "package_data" / "plotly.min.js"
        if cand.is_file():
            return cand
    for cand in (FsPath("/usr/share/javascript/plotly/plotly.min.js"),):
        if cand.is_file():
            return cand
    return None


class EvalService:
    def __init__(self, backends: Dict[str, Any], default_backend: str, store: ResultStore,
                 robot: RobotSpec, world: WorldSpec):
        self.backends = backends
        self.default_backend = default_backend
        self.store = store
        self.robot = robot
        self.world = world
        self.active = default_backend
        self.lock = threading.RLock()
        ctx = multiprocessing.get_context("spawn")
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
        self.plotly_js = find_plotly_js()

    # ------------------------------------------------------------------
    def backend(self, name: Optional[str] = None):
        return self.backends.get(name or self.active) or self.backends[self.default_backend]

    def availability(self) -> List[Dict[str, Any]]:
        out = []
        for name, be in self.backends.items():
            try:
                av = be.available()
            except Exception as exc:  # noqa: BLE001
                av = {"ok": False, "message": str(exc)}
            out.append({"name": name, "label": getattr(be, "label", name), **av,
                        "busy": be.manager.busy})
        return out

    def busy_backend(self) -> Optional[str]:
        for name, be in self.backends.items():
            if be.manager.busy:
                return name
        return None

    def live(self) -> Dict[str, Any]:
        be = self.backend()
        snap = be.snapshot()
        return {"snap": snap, "status": be.manager.status(snap.get("time")), "scene": be.manager.live_scene(),
                "backend": self.active, "wall_time": time.time()}

    async def run_in_pool(self, fn, *args):
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self.pool, fn, *args)
        except concurrent.futures.process.BrokenProcessPool:
            ctx = multiprocessing.get_context("spawn")
            self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
            return await loop.run_in_executor(self.pool, fn, *args)

    def shutdown(self) -> None:
        for be in self.backends.values():
            try:
                be.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.pool.shutdown(wait=False, cancel_futures=True)


def _parse_runs(text: str) -> List[Tuple[str, str]]:
    refs = []
    for item in (text or "").split(","):
        item = item.strip()
        if "/" in item:
            b, r = item.split("/", 1)
            refs.append((safe(b), safe(r)))
    return refs


def _refs_from_body(body: Dict[str, Any]) -> List[Tuple[str, str]]:
    refs = []
    for r in body.get("runs") or []:
        if isinstance(r, dict):
            refs.append((safe(str(r.get("batch_id", ""))), safe(str(r.get("run_id", "")))))
        elif isinstance(r, str) and "/" in r:
            b, rr = r.split("/", 1)
            refs.append((safe(b), safe(rr)))
    return refs


def create_app(service: EvalService) -> Starlette:
    S = service

    async def index(request: Request):
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    async def plotly_js(request: Request):
        if S.plotly_js is not None:
            return FileResponse(S.plotly_js, media_type="application/javascript",
                                headers={"Cache-Control": "max-age=86400"})
        return RedirectResponse(PLOTLY_CDN)

    async def info(request: Request):
        return _json({
            "robot": S.robot.to_dict(), "world": S.world.to_dict(),
            "backends": S.availability(), "default_backend": S.default_backend,
            "perception_modes": PERCEPTION_MODES, "patterns": PATTERN_LABELS, "triggers": TRIGGER_LABELS,
            "default_scenario": normalize_scenario(DEFAULT_SCENARIO, S.world.max_humans),
            "metric_info": METRIC_INFO, "results_dir": str(S.store.root),
        })

    async def status(request: Request):
        be = S.backend()
        snap = be.snapshot()
        return _json({"backend": S.active, "backends": S.availability(),
                      "status": be.manager.status(snap.get("time"))})

    async def stream(request: Request):
        async def gen():
            last_av = 0.0
            av = S.availability()
            while True:
                if await request.is_disconnected():
                    break
                payload = S.live()
                now = time.time()
                if now - last_av > 1.0:
                    av = S.availability()
                    last_av = now
                payload["backends"] = av
                yield "data: " + json.dumps(_sanitize(payload), separators=(",", ":")) + "\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def controllers(request: Request):
        found, errors = ctrl_registry.discover()
        return _json({"controllers": ctrl_registry.describe(found), "errors": errors,
                      "custom_dir": str(ctrl_registry.custom_dir())})

    async def scenarios(request: Request):
        if request.method == "POST":
            body = await request.json()
            name = save_preset(body.get("scenario") or body)
            return _json({"ok": True, "name": name})
        return _json({"presets": list_presets()})

    async def scenario_delete(request: Request):
        ok = delete_preset(request.path_params["name"])
        return _json({"ok": ok}, 200 if ok else 404)

    async def scenario_preview(request: Request):
        body = await request.json()
        sc = normalize_scenario(body.get("scenario") or body, S.world.max_humans)
        routes = [human_route(h, sc["course"]) if h["enabled"] else None for h in sc["humans"]]
        return _json({"scenario": sc, "routes": routes, "robot_start": robot_start_pose(sc),
                      "path": evaluation_path(sc)})

    async def run(request: Request):
        body = await request.json()
        name = body.get("backend") or S.default_backend
        if name not in S.backends:
            return _json({"ok": False, "error": f"unknown backend '{name}'"}, 400)
        busy = S.busy_backend()
        if busy:
            return _json({"ok": False, "error": f"評価実行中です ({busy})。停止してから開始してください。"}, 409)
        be = S.backends[name]
        av = be.available()
        if not av.get("ok"):
            return _json({"ok": False, "error": f"{be.label}: {av.get('message')}"}, 409)
        found, _ = ctrl_registry.discover()
        ctrls = body.get("controllers") or []
        if not ctrls:
            return _json({"ok": False, "error": "制御器を1つ以上選択してください"}, 400)
        for c in ctrls:
            obj = found.get(c.get("name"))
            if obj is None:
                return _json({"ok": False, "error": f"unknown controller {c.get('name')}"}, 400)
            if isinstance(obj, dict) and name == "sim2d":
                return _json({"ok": False, "error": f"外部制御器 {c.get('name')} は Gazebo バックエンドでのみ使えます"}, 400)
            c["label"] = obj.get("label", c["name"]) if isinstance(obj, dict) else obj.label
        try:
            batch = be.start_batch(body)
        except RuntimeError as exc:
            return _json({"ok": False, "error": str(exc)}, 409)
        with S.lock:
            S.active = name
        return _json({"ok": True, "batch": {k: v for k, v in batch.items() if k != "request"}})

    async def stop(request: Request):
        for be in S.backends.values():
            be.abort()
        return _json({"ok": True})

    async def results(request: Request):
        return _json({"batches": S.store.list_batches()})

    async def result_one(request: Request):
        b, r = request.path_params["batch"], request.path_params["run"]
        try:
            run = S.store.load_run(b, r)
        except (FileNotFoundError, OSError, ValueError):
            return _json({"error": "not found"}, 404)
        return _json({"config": run["config"], "summary": run["summary"], "timeseries": timeseries_to_json(run)})

    async def result_delete(request: Request):
        b = request.path_params["batch"]
        for be in S.backends.values():
            if be.manager.busy and be.manager.batch and be.manager.batch.get("id") == b:
                return _json({"ok": False, "error": "running"}, 409)
        return _json({"ok": S.store.delete_batch(b)})

    async def analysis(request: Request):
        body = await request.json()
        refs = _refs_from_body(body)
        res = report.analysis(str(S.store.root), refs, body.get("group_by", "controller"))
        return _json(res)

    async def figure(request: Request):
        refs = _parse_runs(request.query_params.get("runs", ""))
        kind = request.query_params.get("kind", "trajectory")
        if not refs or kind not in ("trajectory", "distance", "clearance", "velocity"):
            return _json({"error": "bad request"}, 400)
        try:
            png = await S.run_in_pool(report.figure_png, str(S.store.root), refs, kind)
        except FileNotFoundError:
            return _json({"error": "not found"}, 404)
        return Response(png, media_type="image/png")

    async def compare(request: Request):
        refs = _parse_runs(request.query_params.get("runs", ""))
        mets = [m for m in request.query_params.get("metrics", "").split(",") if m in METRIC_INFO or m == "success_rate"]
        group_by = request.query_params.get("group_by", "controller")
        if not refs:
            return _json({"error": "bad request"}, 400)
        try:
            png = await S.run_in_pool(report.compare_png, str(S.store.root), refs, mets, group_by)
        except FileNotFoundError:
            return _json({"error": "not found"}, 404)
        return Response(png, media_type="image/png")

    async def export(request: Request):
        body = await request.json()
        refs = _refs_from_body(body)
        if not refs:
            return _json({"error": "no runs selected"}, 400)
        data = await S.run_in_pool(report.build_export, str(S.store.root), refs, body.get("group_by", "controller"))
        fname = time.strftime("eval_report_%Y%m%d-%H%M%S.zip")
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    routes = [
        Route("/", index),
        Route("/vendor/plotly.min.js", plotly_js),
        Route("/api/info", info),
        Route("/api/status", status),
        Route("/api/stream", stream),
        Route("/api/controllers", controllers),
        Route("/api/scenarios", scenarios, methods=["GET", "POST"]),
        Route("/api/scenarios/{name}", scenario_delete, methods=["DELETE"]),
        Route("/api/scenario/preview", scenario_preview, methods=["POST"]),
        Route("/api/run", run, methods=["POST"]),
        Route("/api/stop", stop, methods=["POST"]),
        Route("/api/results", results),
        Route("/api/results/{batch}/{run}", result_one),
        Route("/api/results/{batch}", result_delete, methods=["DELETE"]),
        Route("/api/analysis", analysis, methods=["POST"]),
        Route("/api/figure.png", figure),
        Route("/api/compare.png", compare),
        Route("/api/export", export, methods=["POST"]),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ]
    return Starlette(routes=routes)


def serve(service: EvalService, host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn

    app = create_app(service)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    try:
        server.run()
    finally:
        service.shutdown()
