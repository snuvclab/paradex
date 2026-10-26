# RobustDexGrasp policy on the real xArm6 + Allegro v5 (mainpc / paradex)

Files on mainpc: `~/paradex/robustdex_deploy/` (package + `export_v5_3650/` = policy.npz, meta.json, fk_xarm_allegro.xml).
`src/capture/camera/stereo_track.py` is patched with `--cloud-pub` (backup: `stereo_track.py.bak_robustdex`).

## 0. One command (recommended): tracker + policy runner + shared Viser viewer
    cd ~/paradex && ~/anaconda3/bin/python robustdex_deploy/launch.py --dry-run
    cd ~/paradex && ~/anaconda3/bin/python robustdex_deploy/launch.py --allow-live --scale 0.5 --release
    # other object:  ... launch.py --prompt red_cup --dry-run        (SAM3 init + segment prompt; default yellow_mustard_bottle)

`launch.py` starts `stereo_track.py` in the background (base env, the usual fast-foundation-trt / SAM3 / Cutie
settings, `--frames 100000`, Viser on 8080, plus `--cloud-pub` and `--policy-sub`), waits for the first object cloud,
then runs the policy runner in the foreground inside the ROS2 `robot` env (confirmation prompts and Esc work there).
`--prompt NAME` sets the tracker's SAM3 init/segment prompt; other arguments before `--` go to the runner, anything
after `--` is appended to the stereo_track command (its usual flags are filled in by default, see DEFAULT_TRACKER_ARGS
in launch.py). Ctrl-C / runner exit stops the tracker (`--keep-tracker` to leave it running). Logs: tracker `logs/robustdex_deploy/stereo_track_*.log`, runner terminal output `logs/robustdex_deploy/runner_*.log`
(tee), per-step data `logs/robustdex_deploy/run_*.jsonl`. Shutdown test: `python robustdex_deploy/tests/launch_ctrlc_test.py`.

`--dry-run` is a **virtual rollout**: the robot is not commanded; instead the runner integrates its own PD targets
(perfect-tracking assumption) from the pre-grasp pose through the 70 policy steps and the lift, against the live
tracked object cloud (assumed static). A blue ghost robot (paradex URDF, same kinematics as the policy's FK model:
0.0 mm wrist difference) replays that motion in the viewer, so you see where the hand would go before enabling
`--allow-live`.

The Viser page (http://mainpc:8080) shows the tracked object plus the policy state at 5 Hz: ghost robot (blue, policy's q), hand probe frames
(red = contact flag the policy sees), frame -> nearest object point vectors (yellow), predicted frames of the commanded
target (blue), pre-grasp palm target (green) and a "Policy" status line. The two processes only exchange ZMQ messages
(cloud 5571, policy state 5572), so the ~3 Hz tracker and the 5 Hz policy never block each other.

Shutdown (verified with mock processes): Ctrl-C hits launcher + runner together; the runner holds the robot and closes
its ROS controllers (the xArm stays in servo mode holding the last target, the Allegro holds its last command) while
the launcher waits for it (20 s, then SIGTERM, which the runner maps to the same cleanup); afterwards the tracker's
process group (SAM3 worker included) gets SIGINT, 20 s, then SIGKILL. The same happens on any error exit. Esc in the
runner = hold + abort episode. No `--save-pair-masks` / `--save-frame-clouds` / `--grasp-index` /
`--enable-grasp-execution` are passed to the tracker.

Manual two-terminal operation (sections 1-2) still works.

## 1. Object tracking (base anaconda env, as before) + cloud publisher
Add `--cloud-pub tcp://127.0.0.1:5571` to the usual stereo_track command (keep `--grasp-index`, drop
`--enable-grasp-execution` so nothing else commands the robot):

    cd ~/paradex && OMP_NUM_THREADS=4 ~/anaconda3/bin/python src/capture/camera/stereo_track.py \
      --init-prompt yellow_mustard_bottle --segment-prompt yellow_mustard_bottle --local-four-pair \
      --stereo-backend fast-foundation-trt --fast-root output/stereo_tracking/optimization_ab_20260922/assets/Fast-FoundationStereo \
      --fast-engine-dir output/stereo_tracking/fast_trt_ab_20260923/twostage_trt103 --fast-trt-cuda-graph \
      --device 1 --cutie-device 0 --sam3-device 1 --tracking-model stable --icp-method robust_multiscale --fusion-every 3 \
      --grasp-index output/grasp_retrieval/hrdex_allegro_full --grasp-auto-interval 0 \
      --frames 10000 --visualize --viser-port 8080 --keep-viewer-open \
      --cloud-pub tcp://127.0.0.1:5571

Every frame publishes the fused object model at its tracked pose (robot base frame, m) + the ICP transform.

## 2. Policy runner (ROS env: `sis` first, then `~/anaconda3/envs/robot/bin/python`)
Needs the xArm ROS2 driver (`/right/xarm/*`) and the Allegro v5 driver (`/right/allegroHand_0/*`) running, nothing
else publishing to `/right/allegroHand_0/joint_cmd`.

    cd ~/paradex
    # a) dry run: full pipeline (cloud, FK, obs, policy, safety checks) -- nothing is sent to the robot
    ~/anaconda3/envs/robot/bin/python -m robustdex_deploy.runner --export robustdex_deploy/export_v5_3650 --backend real --dry-run
    # b) first live trial: half-size steps, confirmation prompt before pre-grasp / policy / lift
    ~/anaconda3/envs/robot/bin/python -m robustdex_deploy.runner --export robustdex_deploy/export_v5_3650 --backend real --allow-live --scale 0.5 --release
    # c) nominal
    ~/anaconda3/envs/robot/bin/python -m robustdex_deploy.runner --export robustdex_deploy/export_v5_3650 --backend real --allow-live --release

Episode: pre-grasp IK from the cloud (palm 25 cm from the visible centre, training rule) -> slow joint move
(<= 0.25 rad/s) -> 70 policy steps at 5 Hz -> scripted 15 cm vertical lift (IK) + 3 s hold -> `--release` opens the
hand after Enter. Esc or Ctrl-C holds the robot immediately.

Options: `--contact dist|proxy|tactile|zero` (default `dist`: contact flag = hand-frame-to-object-point distance
< `--dist-on` 0.02 m from FK + cloud, force = PD-torque proxy |2(target-q)| -- best of the real-computable variants in
sim ablation: 65% vs 63% proxy-only vs 38% none vs 79% sim sensors), `--table-offset DZ` (real table top z in the
robot frame minus 0.018), `--table-margin 0.02` (abort if a predicted hand/arm frame goes deeper below the table top),
`--max-arm-delta 0.03` (hard cap, rad/step), `--log-dir`. Logs: `logs/robustdex_deploy/run_*.jsonl`.

Sim audit (same runner, 40 episodes): arm links / palm never touch the table; only fingertips do, on flat objects
(banana, scissors), <= 19 N (finger torque limit). Upright bottles/cans: no table contact at all -> first live objects.

## 3. Checks before the first live run
1. Robot still, cloud running, `--dry-run`: `min_dist` (hand frame to object) printed per 10 steps should be
   plausible (~0.1-0.3 m at the pre-grasp); pre-grasp `feasible` > 0.
2. Measure the table top z in the robot base frame (cloud of the table or a known point); sim is 0.018. Pass the
   difference as `--table-offset`.
3. Check that the published cloud lies on the table (z of its lowest points ~ table height) in the Viser view.
4. The exported policy (meta.json `contact_mode: sensor`) was trained with sim contact sensors; the default
   `--contact dist` is the closest real-computable substitute (sim: 65% vs 79% with sensors). No fine-tuned export yet.
