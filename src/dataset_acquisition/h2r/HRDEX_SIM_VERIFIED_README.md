# HRDex sim-verified grasp replay (Allegro v5 / Robotiq 2F-85 / Inspire DFTP)

Real-robot path for the HRDex contact-map benchmark: measure the object 6D
pose, plan HOME → pregrasp with CuRobo, then replay a grasp that already passed
the 1000-trial CPU PhysX benchmark.  No contact-map inference or grasp
optimization runs on the robot PC; only object pose + arm planning/IK.

## Data on shared_data

```
/home/capture16/shared_data/capture/h2r/hrdex_sim_verified/
  <benchmark>/                 legacy40 (40 poses) | unified34 (34 poses, paper-arch maps) |
                               unified34_winning (34 poses, per-embodiment recipes with Transferred >= Human,
                               see hrdex_contactmap/WINNING_TRANSFER_RECIPES.md; Allegro sims use corrected inertia)
    pack_summary.json
    <embodiment>/              inspire_dftp | allegro_v5 | robotiq_2f85
      <object>/<scene>/        scene = human capture episode of the held-out pose
        result.npz             DFTP   : <src>_pregrasp_traj (K,25,18), <src>_final_q (K,21)
        result_allegro.npz     Allegro: <src>_traj_q_all (K,25,22), <src>_traj_q, <src>_best_particle=0
        result_robotiq.npz     Robotiq: <src>_pregrasp_traj (K,25,12), <src>_gripper_cmd (K,25)
        candidates.json        ranking, sim metrics of every particle, mesh-frame check, provenance
        <src>__*optimized.npz / <src>__*sim*.json   copied benchmark sources (--copy-sources)
```

`<src>` = `human` (Human-Contact) or `robot` (Transferred-Contact).  `K` = up
to 3 candidates per condition, ranked: passed cumulative **and** force-only
success → smallest settle slip → smallest worst-axis displacement → lowest
energy.  Candidate 0 is the best; `candidates.json` lists the benchmark
`candidate` index so the sim video/JSON can be looked up.

Packing is done on the machine that holds `hrdex_contactmap`:

```bash
python src/dataset_acquisition/h2r/hrdex_pack.py --benchmark legacy40 --copy-sources
python src/dataset_acquisition/h2r/hrdex_pack.py --benchmark unified34 --copy-sources   # once runs/unified34_benchmark is complete
```

Frames (IMPORTANT): the 6D pose RPC (FoundationPose) reports the object in the
**original mesh frame** (`<obj>.obj`), while the optimizer/sim work in the
viser-aligned canonical frame (`<obj>_viser.obj` = `viser_align` applied to the
original mesh; rotations of 60-180 deg for most objects).  Since 2026-09-12 the
packer therefore stores wrists in the **original** frame
(`wrist_frame_id = 'original'`, `viser_align` saved alongside) so
`pose_robot @ wrist` is correct with the RPC pose.  The runners read
`wrist_frame_id`, bridge with `viser_align` when it differs from
`--pose_mesh_frame` (default `original`; use `viser` if the pose server is ever
switched to the aligned mesh), and display/plan with the mesh of that frame.
Symptom of a frame mismatch: the hand approaches from a wrong side (e.g. palm
up under a lying spam_can).  The wrist is the simulator's hand URDF root; Euler
angles are scipy intrinsic `XYZ` exactly like the 0518 packs.

`table_clearance.json` (from `hrdex_contactmap/scripts/check_pack_table_clearance.py`)
lists, per candidate, how far the lowest hand link sits above the table when the
object lies as in the human demo.  The optimizer/sim have no table, so Allegro
candidates on low objects can reach below it (46/180 in unified34); prefer
candidates with positive clearance (`--particle k`).  Pregrasp motion = 10 cm straight retreat away from the object
centroid (rotation held), fingers open for the first 60 % of the path, then
closing linearly to the grasp joints.

## Runners (robot PC)

All runners use the same frame and motion contract: capture image → RPC 6D pose
→ `pose_robot` → deterministic staged approach → IK preview → hand overlay →
Viser preview → `y` confirmation → streaming. Stage A keeps HOME height while
moving in the table plane and completing wrist rotation, then descends along
the table normal to pregrasp (`--table_normal_xyz 0,0,1` by default, so it also
works when the robot-coordinate floor plane is not XY). The prescribed link6 poses are streamed through xArm's
internal Cartesian IK; CuRobo sequential IK supplies the inspectable preview and
rejects unreachable/colliding waypoints rather than choosing an unconstrained
MotionGen route. Do not use `--yes` on first runs.

Default motion limits are deliberately conservative: Stage A planar 0.06 m/s,
wrist rotation 0.35 rad/s, descent 0.03 m/s, replay 15 Hz, and measured-state →
HOME at 10 deg/s. A previewed arm path above 30 deg/s is rejected, as is live
feedback above 60 deg/s or within 0.03 rad of a URDF joint limit. These are
command-line options, but increase them only after an empty-workspace replay.

Inspire DFTP (`capture_object6d.py` with `--hand_type inspire_dftp`):

```bash
python src/dataset_acquisition/h2r/capture_object6d.py --name apple --scene 0 --src robot \
  --hand_type inspire_dftp --pipeline \
  --result_root /home/capture16/shared_data/capture/h2r/hrdex_sim_verified/legacy40/inspire_dftp
```
* URDF `rsc/robot/xarm_inspire_DFTP.urdf`, CuRobo `rsc/curobo/content/configs/robot/xarm_inspire_dftp.yml`
  (generated from the F1 config with `tools/make_curobo_cfg_variant.py --mode filter`; same joint names).
* Hand: `get_hand("inspire_dftp")` (serial InspireController), registers 0..1000 in channel order
  `[little, ring, middle, index, thumb_bend(thumb_2), thumb_rotate(thumb_1)]`, **1000 = open** (controller
  init value), driver joints normalized by the DFTP URDF upper limits.  Squeeze default 22 registers.
* Verify on the hand first: `_hand_qpos_to_dftp_msg` direction and channel order with an open/close dry test.

Allegro v5 (`capture_object6d_allegro_v5.py`):

```bash
python src/dataset_acquisition/h2r/capture_object6d_allegro_v5.py --name apple --scene 0 --src robot --pipeline \
  --result_root /home/capture16/shared_data/capture/h2r/hrdex_sim_verified/legacy40/allegro_v5
```
* Without `--particle` it replays `<src>_traj_q` (= candidate 0); `--particle k` picks another packed candidate.
* Packed joints are the simulator's targets (legacy CEDex controller, 20 % over-closed inner targets);
  the optimizer's own final joints are kept as `<src>_legacy_final_q`.  Consider `--grasp_squeeze 0`.

Robotiq 2F-85 (`capture_object6d_robotiq.py`, new):

```bash
python src/dataset_acquisition/h2r/capture_object6d_robotiq.py --name apple --scene 0 --src robot --pipeline \
  --result_root /home/capture16/shared_data/capture/h2r/hrdex_sim_verified/legacy40/robotiq_2f85
```
* URDF `rsc/robot/xarm_robotiq_2f_85.urdf`, CuRobo `xarm_robotiq_2f85.yml` (spheres generated from the
  URDF meshes by `tools/make_curobo_cfg_variant.py --mode spheres-from-mesh`; coarse, review in the sphere editor).
* The simulator root is `robotiq_85_mount_link`; the runner multiplies each waypoint by Rx(pi) to target
  `robotiq_85_base_link` (`base_link_from_root_rpy` in the pack).
* Gripper: `get_hand("robotiq_2f85")`, one normalized close value (knuckle rad / 0.8), squeeze +0.05.

## Not yet exercised on this machine (do these first on the robot PC)

1. CuRobo load of `xarm_inspire_dftp.yml` / `xarm_robotiq_2f85.yml` (mesh paths, link lists) – run
   `--pipeline` with `--no_record_video` and stop at the Viser preview.
2. Hand direction/order dry tests: DFTP registers, Robotiq 0/1, Allegro joint order.
3. Visual check in Viser that the packed hand sits on the object like the sim video
   (`hrdex_contactmap/runs/.../simulation.json` candidate index is in `candidates.json`).
4. Only then approach-only, then close/lift, one candidate at a time.
