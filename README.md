# robot_evaluation_environment

差動二輪の円柱型配膳ロボットを **ROS 2 Jazzy + Gazebo Harmonic** 上で START ラインから GOAL ラインまで走らせ、
人の横切り・対向などに対する **移動時間** と **人との距離** を、**ブラウザだけで** 評価・解析する Docker 環境です。

- 制御器: **DWA / Pure Pursuit (Regulated) / MPC (非線形 MPC) / MPPI / DWPP** + `custom_controllers/` に置いた自作 (.py / 外部 ROS 2 ノード)
- 人 (既定 2 人): 横切る (交差角・出会い頭タイミング) / 対向 / 同方向 / 静止 / ウェイポイント、歩き出すトリガー、ロボット回避の有無
- 評価: 移動時間、人との最小距離・クリアランス、パーソナルスペース侵入時間、TTC、接触、滑らかさ、計算時間
- Web UI: 評価の開始・停止、ライブビュー、結果一覧、複数試行の比較解析、グラフ (PNG)・CSV・HTML レポート出力

**ドキュメント (構成図・実行方法・各設定の説明): [docs/index.html](docs/index.html)**

## クイックスタート

```bash
git clone https://github.com/mamoru1126/robot_evaluation_environment.git
cd robot_evaluation_environment
./scripts/build.sh
./scripts/start.sh
```

ブラウザで http://localhost:8080 を開きます。NVIDIA GPU / Gazebo GUI を使う場合:

```bash
./scripts/start.sh --gpu
./scripts/start.sh --gpu --gui
```

| コマンド | 内容 |
|---|---|
| `./scripts/logs.sh` | ログ表示 (`./scripts/logs.sh sim` などサービス指定可) |
| `./scripts/stop.sh` | 停止 (results/ は残る) |
| `./scripts/diagnose.sh` | トピック・レート・サービスの確認 |
| `./scripts/test.sh` | テスト (Gazebo 不要) |
| `./scripts/start.sh --2d-only` | Gazebo なしの簡易2Dシミュレータだけで起動 |

## 構成

```
docker/ , docker-compose.yml        sim (Gazebo + bridge) / eval (制御器 + Web) / gui / rviz
config/                             robot.yaml, world.yaml, scenarios/*.json
custom_controllers/                 自作制御器 (サンプルあり)
ros2_ws/src/eval_core               制御器・人の動き・試行管理・指標・Web UI・ROS ノード
ros2_ws/src/eval_gazebo             ワールド/モデル生成・ブリッジ・launch
docs/index.html                     ドキュメント
results/                            評価結果 (自動生成)
```

DWPP は Nav2 (main ブランチ) の `dynamic_window_pure_pursuit_functions.hpp` (Apache-2.0, © 2025 Fumiya Ohnishi) を Python に移植したものです。
