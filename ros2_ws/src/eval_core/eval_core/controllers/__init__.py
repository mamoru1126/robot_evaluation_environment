"""Controller registry: built-in controllers + user controllers.

User controllers are discovered from ``$EVAL_CUSTOM_CONTROLLER_DIR``
(``/workspace/custom_controllers`` in Docker, ``<repo>/custom_controllers``
in a source checkout):

* ``*.py``   : every subclass of :class:`BaseController` defined in the file
* ``*.yaml`` : an *external* controller = any ROS 2 node started with a shell
               command (C++ / Nav2 / other languages), see docs.

Discovery re-imports the files every time, so edits are picked up at the next
trial without restarting anything.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import traceback
from pathlib import Path as FsPath
from typing import Any, Dict, List, Optional, Tuple, Type

from .base import BaseController, ControlOutput, ControllerInput, Param, RobotState  # noqa: F401
from .dwa import DWAController
from .mpc import MPCController
from .mppi import MPPIController
from .pure_pursuit import DynamicWindowPurePursuit, RegulatedPurePursuit

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

BUILTIN: Dict[str, Type[BaseController]] = {
    c.name: c for c in (DWAController, RegulatedPurePursuit, MPCController, MPPIController,
                        DynamicWindowPurePursuit)
}


def custom_dir() -> FsPath:
    env = os.environ.get("EVAL_CUSTOM_CONTROLLER_DIR")
    if env:
        return FsPath(env)
    here = FsPath(__file__).resolve()
    for parent in here.parents:
        cand = parent / "custom_controllers"
        if cand.is_dir():
            return cand
    return FsPath("/workspace/custom_controllers")


def _load_module(path: FsPath):
    mod_name = f"eval_custom_{path.stem}_{abs(hash(str(path))) % 10**8}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(mod_name, None)
    return module


def discover(directory: Optional[FsPath] = None) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Return ({name: class_or_external_dict}, errors)."""
    found: Dict[str, Any] = dict(BUILTIN)
    errors: List[Dict[str, str]] = []
    d = FsPath(directory) if directory else custom_dir()
    if not d.is_dir():
        return found, errors
    for path in sorted(d.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = _load_module(path)
        except Exception:  # noqa: BLE001 - report any error of user code
            errors.append({"file": path.name, "error": traceback.format_exc(limit=3)})
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseController) and obj is not BaseController \
                    and obj.__module__ == module.__name__ and getattr(obj, "name", "base") != "base":
                if obj.name in BUILTIN:
                    errors.append({"file": path.name,
                                   "error": f"name '{obj.name}' collides with a built-in controller"})
                    continue
                obj._source_file = path.name  # type: ignore[attr-defined]
                found[obj.name] = obj
    if yaml is not None:
        for path in sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if data.get("type") != "external" or not data.get("name") or not data.get("command"):
                    raise ValueError("external controller yaml needs: type: external, name, command")
                data = dict(data)
                data["_source_file"] = path.name
                found[str(data["name"])] = data
            except Exception as exc:  # noqa: BLE001
                errors.append({"file": path.name, "error": str(exc)})
    return found, errors


def describe(found: Dict[str, Any]) -> List[Dict[str, Any]]:
    """JSON-friendly list for the web UI."""
    out = []
    for name, obj in found.items():
        if isinstance(obj, dict):
            params = []
            for k, v in (obj.get("params") or {}).items():
                params.append(Param(v, label=k).to_dict(k))
            out.append({"name": name, "label": obj.get("label", name), "kind": "external",
                        "builtin": False, "description": obj.get("description", ""),
                        "source": obj.get("_source_file", ""), "params": params})
        else:
            out.append({"name": name, "label": obj.label, "kind": "python",
                        "builtin": name in BUILTIN, "description": obj.description,
                        "source": getattr(obj, "_source_file", "built-in"),
                        "params": obj.param_schema()})
    return out


def create(name: str, robot, params: Optional[Dict[str, Any]] = None,
           directory: Optional[FsPath] = None) -> BaseController:
    found, errors = discover(directory) if name not in BUILTIN else (BUILTIN, [])
    obj = found.get(name)
    if obj is None:
        msg = f"unknown controller '{name}'"
        if errors:
            msg += "; errors while loading custom controllers: " + "; ".join(e["file"] for e in errors)
        raise KeyError(msg)
    if isinstance(obj, dict):
        raise TypeError(f"controller '{name}' is an external controller")
    return obj(robot, params or {})
