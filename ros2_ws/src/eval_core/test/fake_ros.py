"""A tiny in-process stand-in for rclpy and the message packages.

It lets the ROS glue code (local_planner_node, gazebo_backend, scan_relay)
run in unit tests without ROS 2 / Gazebo.  Messages use __slots__ so that a
typo in a field name raises AttributeError exactly like the real classes.
Only the API surface used by eval_core is implemented.
"""
from __future__ import annotations

import sys
import types
from collections import deque


# ----------------------------------------------------------------------------- messages
class Msg:
    __slots__ = ()
    _defaults = {}

    def __init__(self, **kw):
        for k, f in self._defaults.items():
            object.__setattr__(self, k, f() if callable(f) else f)
        for k, v in kw.items():
            setattr(self, k, v)


def msg_class(_cls_name, **fields):
    return type(_cls_name, (Msg,), {"__slots__": tuple(fields), "_defaults": fields})


Time = msg_class("Time", sec=0, nanosec=0)
Header = msg_class("Header", stamp=Time, frame_id="")
Vector3 = msg_class("Vector3", x=0.0, y=0.0, z=0.0)
Point = msg_class("Point", x=0.0, y=0.0, z=0.0)
Quaternion = msg_class("Quaternion", x=0.0, y=0.0, z=0.0, w=1.0)
Pose = msg_class("Pose", position=Point, orientation=Quaternion)
PoseWithCovariance = msg_class("PoseWithCovariance", pose=Pose, covariance=list)
Twist = msg_class("Twist", linear=Vector3, angular=Vector3)
TwistWithCovariance = msg_class("TwistWithCovariance", twist=Twist, covariance=list)
Odometry = msg_class("Odometry", header=Header, child_frame_id="", pose=PoseWithCovariance, twist=TwistWithCovariance)
LaserScan = msg_class("LaserScan", header=Header, angle_min=0.0, angle_max=0.0, angle_increment=0.0,
                      time_increment=0.0, scan_time=0.0, range_min=0.0, range_max=0.0, ranges=list, intensities=list)
String = msg_class("String", data="")
PoseStamped = msg_class("PoseStamped", header=Header, pose=Pose)
PathMsg = msg_class("Path", header=Header, poses=list)
ColorRGBA = msg_class("ColorRGBA", r=0.0, g=0.0, b=0.0, a=0.0)
Marker = msg_class("Marker", header=Header, ns="", id=0, type=0, action=0, pose=Pose, scale=Vector3,
                   color=ColorRGBA, points=list)
Marker.LINE_STRIP = 4
Marker.ADD = 0
MarkerArray = msg_class("MarkerArray", markers=list)
Clock = msg_class("Clock", clock=Time)
Entity = msg_class("Entity", id=0, name="", type=0)
for _i, _n in enumerate(["NONE", "LIGHT", "MODEL", "LINK", "VISUAL", "COLLISION", "SENSOR", "JOINT"]):
    setattr(Entity, _n, _i)
SetEntityPoseRequest = msg_class("SetEntityPose_Request", entity=Entity, pose=Pose)
SetEntityPoseResponse = msg_class("SetEntityPose_Response", success=False)


class SetEntityPose:
    Request = SetEntityPoseRequest
    Response = SetEntityPoseResponse


# ----------------------------------------------------------------------------- bus / clock
class Bus:
    def __init__(self):
        self.subs = {}
        self.services = {}
        self.queue = deque()
        self.sim_time = 0.0
        self.nodes = []
        self.published = {}
        self.latched = {}

    def publish(self, topic, msg):
        self.published[topic] = self.published.get(topic, 0) + 1
        for typ, cb in self.subs.get(topic, []):
            if typ is not type(msg):
                raise TypeError(f"topic {topic}: publishing {type(msg).__name__} but subscriber expects {typ.__name__}")
            self.queue.append((cb, msg))

    def spin_once(self, max_items=10000):
        n = 0
        while self.queue and n < max_items:
            cb, msg = self.queue.popleft()
            cb(msg)
            n += 1
        for node in list(self.nodes):
            for t in node._timers:
                if self.sim_time + 1e-9 >= t.next:
                    t.next = max(t.next + t.period, self.sim_time)
                    t.cb()
        while self.queue and n < max_items:
            cb, msg = self.queue.popleft()
            cb(msg)
            n += 1


BUS = Bus()


class _Stamp:
    def __init__(self, t):
        self.nanoseconds = int(round(t * 1e9))

    def to_msg(self):
        return Time(sec=int(self.nanoseconds // 10 ** 9), nanosec=int(self.nanoseconds % 10 ** 9))


class _Clock:
    def now(self):
        return _Stamp(BUS.sim_time)


class _Logger:
    def __init__(self, name):
        self.name = name
        self.lines = []

    def _log(self, lvl, msg):
        self.lines.append((lvl, msg))

    def info(self, m):
        self._log("info", m)

    def warn(self, m):
        self._log("warn", m)

    def warning(self, m):
        self._log("warn", m)

    def error(self, m):
        self._log("error", m)


def _is_latched(qos):
    return isinstance(qos, dict) and qos.get("durability") == 1


class _Publisher:
    def __init__(self, typ, topic, latched=False):
        self.typ = typ
        self.topic = topic
        self.latched = latched

    def publish(self, msg):
        if type(msg) is not self.typ:
            raise TypeError(f"{self.topic}: expected {self.typ.__name__}, got {type(msg).__name__}")
        if self.latched:
            BUS.latched[self.topic] = msg
        BUS.publish(self.topic, msg)

    def get_subscription_count(self):
        return len(BUS.subs.get(self.topic, []))


class _Timer:
    def __init__(self, period, cb):
        self.period = float(period)
        self.cb = cb
        self.next = BUS.sim_time + self.period


class _Future:
    def __init__(self, result):
        self._r = result

    def done(self):
        return True

    def result(self):
        return self._r


class _Client:
    def __init__(self, typ, name):
        self.typ = typ
        self.name = name

    def service_is_ready(self):
        return self.name in BUS.services

    def wait_for_service(self, timeout_sec=None):
        return self.service_is_ready()

    def call_async(self, req):
        if type(req) is not self.typ.Request:
            raise TypeError("wrong request type")
        handler = BUS.services.get(self.name)
        if handler is None:
            return _Future(None)
        resp = self.typ.Response()
        return _Future(handler(req, resp))


class Parameter:
    def __init__(self, name, type_=None, value=None):
        self.name = name
        self.value = value


class _ParamValue:
    def __init__(self, v):
        self.value = v


class Node:
    def __init__(self, node_name, parameter_overrides=None, **kw):
        self._name = node_name
        self._timers = []
        self._params = {p.name: p.value for p in (parameter_overrides or [])}
        self._logger = _Logger(node_name)
        BUS.nodes.append(self)

    def get_name(self):
        return self._name

    def declare_parameter(self, name, value=None, descriptor=None, **kw):
        self._params.setdefault(name, value)
        return _ParamValue(self._params[name])

    def get_parameter(self, name):
        return _ParamValue(self._params.get(name))

    def get_clock(self):
        return _Clock()

    def get_logger(self):
        return self._logger

    def create_publisher(self, typ, topic, qos):
        return _Publisher(typ, topic, _is_latched(qos))

    def create_subscription(self, typ, topic, cb, qos):
        BUS.subs.setdefault(topic, []).append((typ, cb))
        if _is_latched(qos) and topic in BUS.latched:
            BUS.queue.append((cb, BUS.latched[topic]))
        return (topic, cb)

    def create_timer(self, period, cb):
        t = _Timer(period, cb)
        self._timers.append(t)
        return t

    def create_client(self, typ, name):
        return _Client(typ, name)

    def create_service(self, typ, name, handler):
        BUS.services[name] = handler
        return name

    def destroy_node(self):
        if self in BUS.nodes:
            BUS.nodes.remove(self)


class SingleThreadedExecutor:
    def add_node(self, node):
        pass

    def spin(self):
        return None

    def shutdown(self, timeout_sec=None):
        return True


class ExternalShutdownException(Exception):
    pass


# ----------------------------------------------------------------------------- install
def install():
    """Register fake modules in sys.modules.  Returns the previous entries."""
    names = ["rclpy", "rclpy.node", "rclpy.qos", "rclpy.executors", "rclpy.parameter",
             "geometry_msgs", "geometry_msgs.msg", "nav_msgs", "nav_msgs.msg", "sensor_msgs", "sensor_msgs.msg",
             "std_msgs", "std_msgs.msg", "visualization_msgs", "visualization_msgs.msg",
             "rosgraph_msgs", "rosgraph_msgs.msg", "ros_gz_interfaces", "ros_gz_interfaces.msg",
             "ros_gz_interfaces.srv", "rcl_interfaces", "rcl_interfaces.msg"]
    saved = {n: sys.modules.get(n) for n in names}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    state = {"ok": False}
    rclpy = mod("rclpy", init=lambda args=None: state.update(ok=True), ok=lambda: state["ok"],
                shutdown=lambda: state.update(ok=False),
                create_node=lambda name, **kw: Node(name, **kw), spin=lambda node: None)
    rclpy.node = mod("rclpy.node", Node=Node)
    rclpy.qos = mod("rclpy.qos", QoSProfile=lambda **kw: kw, ReliabilityPolicy=types.SimpleNamespace(RELIABLE=1, BEST_EFFORT=2),
                    DurabilityPolicy=types.SimpleNamespace(TRANSIENT_LOCAL=1, VOLATILE=2),
                    qos_profile_sensor_data={"sensor": True})
    rclpy.executors = mod("rclpy.executors", SingleThreadedExecutor=SingleThreadedExecutor,
                          ExternalShutdownException=ExternalShutdownException)
    rclpy.parameter = mod("rclpy.parameter", Parameter=Parameter)
    mod("geometry_msgs")
    mod("geometry_msgs.msg", Twist=Twist, Point=Point, PoseStamped=PoseStamped, Pose=Pose, Vector3=Vector3,
        Quaternion=Quaternion)
    mod("nav_msgs")
    mod("nav_msgs.msg", Odometry=Odometry, Path=PathMsg)
    mod("sensor_msgs")
    mod("sensor_msgs.msg", LaserScan=LaserScan)
    mod("std_msgs")
    mod("std_msgs.msg", String=String, Header=Header)
    mod("visualization_msgs")
    mod("visualization_msgs.msg", Marker=Marker, MarkerArray=MarkerArray)
    mod("rosgraph_msgs")
    mod("rosgraph_msgs.msg", Clock=Clock)
    mod("ros_gz_interfaces")
    mod("ros_gz_interfaces.msg", Entity=Entity)
    mod("ros_gz_interfaces.srv", SetEntityPose=SetEntityPose)
    mod("rcl_interfaces")
    mod("rcl_interfaces.msg", ParameterDescriptor=lambda **kw: kw)
    return saved


def uninstall(saved):
    for n, m in saved.items():
        if m is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = m
    for n in [k for k in sys.modules if k.startswith("eval_core.ros")]:
        sys.modules.pop(n, None)


def reset_bus():
    global BUS
    BUS.__init__()
    return BUS
