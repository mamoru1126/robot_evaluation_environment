"""自作制御器のサンプル (Python プラグイン).

このファイルのように ``BaseController`` を継承したクラスを置くだけで、
Web UI の制御器一覧に自動で追加されます (再ビルド・再起動は不要。
「制御器」タブの「再読込」か、次の試行開始時に読み直されます)。

入力 ``inp`` (ControllerInput):
  inp.state      : RobotState(t, x, y, yaw, v, w)   ロボットの真値姿勢と計測速度
  inp.path       : Path  START->GOAL の直線経路 (project / lookahead_point / heading_at ...)
  inp.obstacles  : ObstacleSet  障害物
                   clearance(xy, tau) -> ロボット表面から障害物までの距離 (<=0 で接触)
                   agents            -> 人 (位置・速度, 認識モードが追跡/真値のとき)
  inp.dt         : 制御周期 [s]
  inp.last_cmd   : 前回の指令 (v, w)
出力: ControlOutput(v, w, trajectory=[[x, y], ...], info={...})
     trajectory はライブビューに緑の線で表示されます。
"""
import math

import numpy as np

from eval_core.controllers.base import BaseController, ControlOutput, Param, rollout_constant


class StopAndGo(BaseController):
    name = "example_stop_and_go"
    label = "サンプル: Stop & Go (自作例)"
    description = ("経路からの横ずれと方位誤差を P 制御で補正しながら進み、"
                   "前方の監視範囲に人や障害物が入ったら停止して待つ最小構成の自作例。")

    PARAMS = {
        "target_speed": Param(0.6, 0.1, 1.5, 0.05, "目標速度 [m/s]"),
        "k_lat": Param(1.5, 0.0, 10.0, 0.1, "横ずれゲイン"),
        "k_head": Param(2.0, 0.0, 10.0, 0.1, "方位ゲイン"),
        "stop_distance": Param(1.2, 0.2, 5.0, 0.1, "停止する前方距離 [m]"),
        "watch_half_width": Param(0.5, 0.2, 2.0, 0.05, "前方監視範囲の半幅 [m]"),
    }

    def compute(self, inp):
        st = inp.state
        s, lateral, _ = inp.path.project(st.x, st.y)
        heading = float(inp.path.heading_at(s))
        e_th = math.atan2(math.sin(st.yaw - heading), math.cos(st.yaw - heading))
        w = -self.p["k_lat"] * lateral - self.p["k_head"] * e_th
        v = self.p["target_speed"]

        # 前方の監視範囲 (ロボット前方の帯) にサンプル点を並べて障害物を調べる
        ahead = np.linspace(0.0, self.p["stop_distance"], 10)
        side = np.linspace(-1.0, 1.0, 5) * max(self.p["watch_half_width"] - self.robot.radius, 0.0)
        A, S = np.meshgrid(ahead, side)
        c, s_ = math.cos(st.yaw), math.sin(st.yaw)
        pts = np.stack([st.x + A * c - S * s_, st.y + A * s_ + S * c], -1)
        blocked = bool(np.any(inp.obstacles.clearance(pts, 0.0) <= 0.05))
        if blocked:
            v, w = 0.0, 0.0

        traj, _ = rollout_constant(st.x, st.y, st.yaw, v, w, 1.5, 0.1)
        return ControlOutput(v, w, trajectory=traj.tolist(), info={"blocked": blocked})
