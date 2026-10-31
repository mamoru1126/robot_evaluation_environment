"""End-to-end test of the web API with the built-in 2D simulator backend."""
import json
import socket
import threading
import time
import urllib.request

import pytest

starlette = pytest.importorskip("starlette")
uvicorn = pytest.importorskip("uvicorn")

from eval_core.backends.sim2d_backend import Sim2DBackend  # noqa: E402
from eval_core.scenario import list_presets  # noqa: E402
from eval_core.specs import load_robot_spec, load_world_spec  # noqa: E402
from eval_core.storage import ResultStore  # noqa: E402
from eval_core.web.server import EvalService, create_app  # noqa: E402


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _req(base, path, body=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(base + path, data=data, method=method or ("POST" if body is not None else "GET"),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
    return json.loads(raw) if "json" in ctype else raw


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    robot, world = load_robot_spec(), load_world_spec()
    store = ResultStore(tmp_path_factory.mktemp("results"))
    service = EvalService({"sim2d": Sim2DBackend(robot, world, store)}, "sim2d", store, robot, world)
    port = _free_port()
    cfg = uvicorn.Config(create_app(service), host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(cfg)
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            _req(base, "/api/info")
            break
        except OSError:
            time.sleep(0.1)
    yield base
    srv.should_exit = True
    th.join(5)
    service.shutdown()


def test_info_and_controllers(server):
    info = _req(server, "/api/info")
    assert info["default_backend"] == "sim2d"
    assert set(info["perception_modes"]) == {"scan", "scan_tracking", "ground_truth"}
    ctrls = _req(server, "/api/controllers")
    names = {c["name"] for c in ctrls["controllers"]}
    assert {"dwa", "pure_pursuit", "mpc", "mppi", "dwpp"} <= names
    html = _req(server, "/")
    assert b"<canvas" in html


def test_run_results_analysis_export(server):
    presets = {p["name"]: p for p in _req(server, "/api/scenarios")["presets"]}
    sc = presets["crossing_and_head_on"]
    r = _req(server, "/api/run", {"backend": "sim2d", "name": "api", "scenarios": [sc],
                                  "controllers": [{"name": "dwa"}, {"name": "dwpp"}], "repeats": 1,
                                  "perception": "ground_truth"})
    assert r["ok"]
    for _ in range(600):
        if not _req(server, "/api/status")["status"]["busy"]:
            break
        time.sleep(0.2)
    batches = _req(server, "/api/results")["batches"]
    b = next(x for x in batches if x["id"] == r["batch"]["id"])
    assert b["status"] == "finished"
    assert len(b["runs"]) == 2
    refs = [{"batch_id": b["id"], "run_id": x["run_id"]} for x in b["runs"]]
    a = _req(server, "/api/analysis", {"runs": refs, "group_by": "controller"})
    assert {g["group"] for g in a["groups"]} == {"dwa", "dwpp"}
    one = _req(server, f"/api/results/{b['id']}/{b['runs'][0]['run_id']}")
    assert len(one["timeseries"]["t"]) > 10
    q = ",".join(f"{x['batch_id']}/{x['run_id']}" for x in refs)
    png = _req(server, f"/api/figure.png?kind=trajectory&runs={q}")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    z = _req(server, "/api/export", {"runs": refs})
    assert z[:2] == b"PK"
