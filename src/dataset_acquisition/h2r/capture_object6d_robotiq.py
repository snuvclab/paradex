from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[3]))
PROJECT_ROOT = Path(__file__).parents[3]

import argparse
import datetime
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import trimesh
import zmq

from paradex.calibration.utils import load_current_C2R, save_current_camparam
from paradex.utils.path import shared_dir
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.visualization.robot import RobotModule
from paradex.visualization.visualizer.viser import ViserViewer

import pose_normalize
from hrdex_replay_common import (
    XArmJointWatchdog,
    arm_link_pose_trajectory,
    assert_arm_joint_limits,
    assert_bounded_joint_steps,
    assert_bounded_pose_steps,
    curobo_active_joint_names,
    densify_timed_commands,
    end_effector_poses_to_command_link,
    interpolate_joint_anchors,
    move_xarm_home_safely,
    pack_frame_to_pose_frame,
    resolve_pose_mesh,
    resolve_scene_arg,
    retime_step_periods,
    select_curobo_active_qpos,
    sparse_waypoint_indices,
    staged_home_to_pregrasp,
    trajectory_phase,
    unwrap_xarm_joints,
    validate_motion_args,
)

# xArm + Robotiq 2F-85 variant of capture_object6d.py.  Replays sim-verified
# HRDex grasps packed by hrdex_pack.py (result_robotiq.npz): wrist trajectory
# of the simulator hand root (robotiq_85_mount_link) + one normalized gripper
# command per waypoint.  Planning targets robotiq_85_base_link, which is the
# mount link rotated by pi about x.
DEFAULT_ROBOT_URDF = str(PROJECT_ROOT / "rsc" / "robot" / "xarm_robotiq_2f_85.urdf")
DEFAULT_CUROBO_ROBOT_CFG = str(
    PROJECT_ROOT / "rsc" / "curobo" / "content" / "configs" / "robot"
    / "xarm_robotiq_2f85.yml"
)
ROBOTIQ_DRIVE_JOINT = "robotiq_85_left_knuckle_joint"
ROBOTIQ_MAX_RAD = 0.8
# simulator root (mount link) -> base_link that curobo/URDF expose
MOUNT_TO_BASE = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)  # Rx(pi)
# Same xarm home pose used by the taeksoo pipeline (HOME_DEG -> rad applied
# only to the 6 arm joints; hand stays at zero).
HOME_DEG = [-84.9, -25.5, -41.1, 118.8, -1.9, 158.1]

# Make sibling helper modules importable.
_SRC_DIR = str(PROJECT_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _send_rpc_once(addr: str, req: dict, timeout_ms: int = 300000) -> dict:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(addr)
        sock.send_json(req)
        return sock.recv_json()
    finally:
        sock.close()
        ctx.term()


def _to_shared_data_path(abs_path: str) -> str:
    norm = os.path.normpath(abs_path)
    marker = f"{os.sep}shared_data{os.sep}"
    if marker in norm:
        tail = norm.split(marker, 1)[1]
        return os.path.join("shared_data", tail).replace(os.sep, "/")
    if norm.endswith(f"{os.sep}shared_data"):
        return "shared_data"
    raise ValueError(f"path is not under shared_data: {abs_path}")


def _to_4x4(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat
        return out
    raise ValueError(f"matrix must be 4x4 or 3x4, got {mat.shape}")


def _extract_pose_response(res: Dict[str, Any]) -> Dict[str, Any]:
    pose = res.get("object_6d", res)
    out = {
        "pose_world": pose.get("pose_world"),
        "pose_left_cam": pose.get("pose_left_cam"),
        "R_world": pose.get("R_world"),
        "t_world": pose.get("t_world"),
    }
    if out["pose_world"] is None:
        raise RuntimeError(f"invalid response, pose_world missing: {res}")
    return out


def _resolve_mesh_path(mesh_name: str, mesh_root_dir: str, frame: str = "original") -> str:
    """Mesh matching the frame the 6D pose RPC reports: raw <obj>.obj (FoundationPose,
    ``original``) or the viser-aligned <obj>_viser.obj (``viser``)."""
    return resolve_pose_mesh(mesh_root_dir, mesh_name, frame)


def _pack_frame_to_pose_frame(d, pose_mesh_frame: str) -> np.ndarray:
    """4x4 that re-expresses packed wrists in the frame of the 6D pose estimate.

    Packs record ``wrist_frame_id`` ('original' = raw <obj>.obj frame, or
    'canonical' = viser-aligned <obj>_viser.obj frame) and ``viser_align``
    (T_viser_from_original).  The pose RPC reports the object in
    ``pose_mesh_frame``; a mismatch is bridged with viser_align.
    """
    pack_frame = str(d["wrist_frame_id"]) if "wrist_frame_id" in d.files else "canonical"
    F = pack_frame_to_pose_frame(d, pose_mesh_frame)
    print(f"[pipe] wrist frame: pack='{pack_frame}', pose='{pose_mesh_frame}' -> "
          f"{'identity' if np.allclose(F, np.eye(4)) else 'viser_align bridge'}")
    return F.astype(np.float64)


def _add_debug_images_to_gui(
    vis,
    debug_dir: str,
    *,
    anchor_xyz=(1.0, 1.0, 0.0),
    plane_width: float = 1.0,
    spacing: float = 0.05,
) -> int:
    """Show RPC-side mask overlay PNGs both in the GUI sidebar AND as large
    floating planes above the scene so they are readable without squinting.

    The 6D pose server writes <abs_save_path>/debug/mask_overlay_*.png on each
    inference call. We add a small thumbnail in the GUI for quick toggling, and
    place full-resolution image planes above the workspace as a billboard the
    user can rotate around."""
    import glob
    import cv2

    if not os.path.isdir(debug_dir):
        return 0
    paths = sorted(glob.glob(os.path.join(debug_dir, "*.png"))
                   + glob.glob(os.path.join(debug_dir, "*.jpg")))
    if not paths:
        return 0

    images = []
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # left-right flip so the plane normal flip done in scene.add_image does
        # not produce a mirrored image when viewed from the front.
        rgb = np.ascontiguousarray(rgb[:, ::-1, :])
        images.append((p, rgb))
    if not images:
        return 0

    # GUI thumbnails (sidebar): keep them as a quick reference / on-off toggle.
    folder = vis.server.gui.add_folder(f"Debug ({len(images)})")
    with folder:
        for p, img in images:
            vis.server.gui.add_image(img, label=os.path.basename(p))

    # Scene billboards: stack horizontally along +X above anchor_xyz. Plane is
    # rotated rot_x(-90deg) so the image faces +Y AND is right-side up from the
    # default front view (the +rot_x(+90) variant was upside-down).
    anchor = np.asarray(anchor_xyz, dtype=float)
    wxyz_face_neg_y = (float(np.cos(np.pi / 4)), float(-np.sin(np.pi / 4)), 0.0, 0.0)
    x_cursor = -0.5 * (len(images) - 1) * (plane_width + spacing)
    for p, img in images:
        h, w = img.shape[:2]
        rw = float(plane_width)
        rh = float(plane_width) * (float(h) / float(max(w, 1)))
        pos = (anchor[0] + x_cursor, anchor[1], anchor[2] + 0.5 * rh)
        vis.server.scene.add_image(
            name=f"/debug_images/{os.path.basename(p)}",
            image=img,
            render_width=rw,
            render_height=rh,
            position=pos,
            wxyz=wxyz_face_neg_y,
        )
        x_cursor += plane_width + spacing

    print(f"[debug] added {len(images)} image(s) from {debug_dir} "
          f"(sidebar + scene planes, plane_width={plane_width}m)")
    return len(images)


def _popup_debug_images(debug_dir: str) -> int:
    """Open the RPC-side mask overlay PNGs in the OS default image viewer as
    standalone popup windows, independent of the viser scene."""
    import glob
    import subprocess

    if not os.path.isdir(debug_dir):
        return 0
    paths = sorted(glob.glob(os.path.join(debug_dir, "*.png"))
                   + glob.glob(os.path.join(debug_dir, "*.jpg")))
    if not paths:
        return 0
    for p in paths:
        try:
            subprocess.Popen(["xdg-open", p],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[debug] popup failed for {p}: {e}")
    print(f"[debug] opened {len(paths)} debug image(s) as OS popups from {debug_dir}")
    return len(paths)


def _load_mesh(mesh_path: str) -> trimesh.Trimesh:
    geom = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(geom, trimesh.Trimesh):
        return geom
    if isinstance(geom, trimesh.Scene):
        geoms = [g for g in geom.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"no mesh geometry in scene: {mesh_path}")
        return trimesh.util.concatenate(geoms)
    if isinstance(geom, list):
        return trimesh.util.concatenate(geom)
    raise ValueError(f"unsupported mesh type: {type(geom)}")


def _solve_ik_chain_curobo(robot_cfg: Dict[str, Any], target_poses: np.ndarray,
                           seed_arm_qpos: np.ndarray, device: str = "cuda:0",
                           num_seeds: int = 32, null_space_weight: float = 1.0,
                           self_collision: bool = True,
                           position_threshold: float = 0.005,
                           rotation_threshold: float = 0.05,
                           sample_stride: int = 10,
                           world_cfg_dict: Optional[Dict[str, Any]] = None):
    """Per-waypoint IK with continuity warm-start. Sets up curobo IKSolver
    once and calls solve_single in a loop, feeding the previous solution as
    the seed for the next.

    target_poses: (W, 4, 4) eef targets in robot frame
    seed_arm_qpos: (active_dof,) initial seed
    returns: (W, active_dof) IK solutions; failures hold the previous solution.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "curobo" / "src"))
    import copy
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from object6d.capture_rcc import _rotmat_to_wxyz

    robot_cfg = copy.deepcopy(robot_cfg)
    cspace = robot_cfg["robot_cfg"]["kinematics"]["cspace"]
    cspace["null_space_weight"] = [float(null_space_weight)] * len(cspace["joint_names"])
    dev = torch.device(device if ("cuda" in str(device) and torch.cuda.is_available()) else "cpu")
    tensor_args = TensorDeviceType(device=dev)
    world_cfg = None
    if world_cfg_dict is not None:
        from curobo.geom.types import WorldConfig
        world_cfg = WorldConfig.from_dict(world_cfg_dict)
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg, world_cfg, tensor_args=tensor_args,
        num_seeds=int(num_seeds), position_threshold=float(position_threshold),
        rotation_threshold=float(rotation_threshold),
        self_collision_check=bool(self_collision), self_collision_opt=bool(self_collision),
        use_cuda_graph=False,
    )
    ik = IKSolver(ik_cfg)

    W = int(target_poses.shape[0])
    solver_joint_names = list(ik.joint_names)
    configured_active_names = curobo_active_joint_names(robot_cfg)
    if solver_joint_names != configured_active_names:
        raise RuntimeError(
            "CuRobo active joint order differs from config: "
            f"solver={solver_joint_names}, config={configured_active_names}"
        )
    last = select_curobo_active_qpos(seed_arm_qpos, robot_cfg, solver_joint_names)
    dof = len(solver_joint_names)
    sample_indices = sparse_waypoint_indices(W, sample_stride)
    anchors = np.zeros((len(sample_indices), dof), dtype=np.float32)

    n_succ = 0
    if W > 0:
        anchors[0] = last
        n_succ = 1
    fallback_count = 0
    progress_every = max(1, len(sample_indices) // 10)
    for anchor_i, i in enumerate(sample_indices[1:], start=1):
        T = target_poses[i]
        wxyz = _rotmat_to_wxyz(T[:3, :3])
        pose = Pose(
            position=torch.tensor(T[:3, 3], device=dev, dtype=torch.float32).view(1, 3),
            quaternion=torch.tensor(wxyz, device=dev, dtype=torch.float32).view(1, 4),
        )
        seed_t = torch.tensor(last, device=dev, dtype=torch.float32).view(1, 1, dof)
        retract_t = torch.tensor(last, device=dev, dtype=torch.float32).view(1, dof)
        result = ik.solve_single(
            pose, retract_config=retract_t, seed_config=seed_t, num_seeds=1
        )
        succ = bool(result.success.flatten()[0].item()) if hasattr(result.success, "flatten") else bool(result.success)
        if not succ and num_seeds > 1:
            fallback_count += 1
            result = ik.solve_single(
                pose,
                retract_config=retract_t,
                seed_config=seed_t,
                num_seeds=int(num_seeds),
            )
            succ = bool(result.success.flatten()[0].item()) if hasattr(result.success, "flatten") else bool(result.success)
        if succ:
            sol = result.solution.detach().cpu().numpy().reshape(-1)[:dof]
            last = sol.astype(np.float32)
            n_succ += 1
        anchors[anchor_i] = last
        if anchor_i % progress_every == 0 or anchor_i == len(sample_indices) - 1:
            print(f"[IK preview] {anchor_i + 1}/{len(sample_indices)} anchors processed, "
                  f"{n_succ} accepted, "
                  f"full-seed retries={fallback_count}")
    out = interpolate_joint_anchors(sample_indices, anchors, W)
    accepted = W if n_succ == len(sample_indices) else n_succ
    print(f"[IK preview] solved {n_succ}/{len(sample_indices)} anchors for {W} frames "
          f"(stride={sample_stride}); full-seed retries={fallback_count}")
    return out, accepted


# Order on the wire (matches /right/joint_states' name list and what the
# ros2_control position_controller actually consumes -- thumb_1 FIRST, then
# thumb_2). NOTE: state_receiver._publish_hand_command reverses the array AND
# swaps thumb_1<->thumb_2, so we publish directly to its cmd_pubs[side]
# instead of going through move().
def _gripper_cmd_from_qpos(full_qpos: np.ndarray, full_joint_names) -> float:
    """URDF knuckle angle (rad) -> normalized close command 0 (open) .. 1 (closed)."""
    q = float(full_qpos[list(full_joint_names).index(ROBOTIQ_DRIVE_JOINT)])
    return float(np.clip(q / ROBOTIQ_MAX_RAD, 0.0, 1.0))


def _wait_xarm_at_target(xarm, target_rad: np.ndarray, tol_rad: float, timeout: float) -> bool:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        data = xarm.get_data()
        cur = data.get("qpos")
        if cur is not None and np.all(np.isfinite(cur)):
            if np.max(np.abs(np.asarray(cur) - target_rad)) < tol_rad:
                return True
        time.sleep(0.05)
    return False


def _execute_on_real_robot(args, pipeline_result: Dict[str, Any], robot_module: RobotModule,
                          abs_save_path: str) -> None:
    """Stream the planned trajectory to the real xarm + Robotiq 2F-85, with
    optional camera recording via CaptureSession. Mirrors taeksoo_send.py's
    safety procedure (home first, wait for arrival, confirm prompt, streaming
    loop, cleanup on any error)."""
    from paradex.io.robot_controller import get_arm, get_hand
    from paradex.calibration.utils import save_current_C2R

    full_traj = np.asarray(pipeline_result["full_traj"], dtype=np.float64)
    Tn = int(full_traj.shape[0])
    full_joint_names = list(robot_module.get_joint_names())

    # arm: joint1..joint6 indices in URDF order
    arm_idx = [full_joint_names.index(f"joint{i}") for i in range(1, 7)]
    arm_traj = full_traj[:, arm_idx]  # (T,6) rad
    arm_pose_traj = np.asarray(pipeline_result["arm_pose_traj"], dtype=np.float64)

    # gripper: one normalized close command per frame
    hand_msgs = np.zeros((Tn, 1), dtype=np.float64)
    for t in range(Tn):
        hand_msgs[t, 0] = _gripper_cmd_from_qpos(full_traj[t], full_joint_names)

    # Grasp squeeze: run the whole grasp sequence (approach + hand_interp +
    # replay) with the planned hand values UNCHANGED. Once the grasp is done we
    # insert an explicit squeeze step -- a gradual hand-only ramp that lowers
    # every joint except thumb_1 by --grasp_squeeze raw -- and then the
    # grasp_hold + lift phases keep that tighter pose.
    n_a = int(pipeline_result.get("n_approach", 0))
    n_b = int(pipeline_result.get("n_hand_interp", 0))
    n_r = int(pipeline_result.get("n_replay", 0))
    squeeze_start = n_a + n_b + n_r
    squeeze_amt = float(args.grasp_squeeze)
    squeeze_steps = int(args.squeeze_steps)
    do_squeeze = squeeze_amt != 0.0 and 0 < squeeze_start < Tn
    if do_squeeze:
        # hold + lift frames keep the squeezed pose (thumb_1 / col 0 untouched).
        hand_msgs[squeeze_start:, 0] = np.clip(hand_msgs[squeeze_start:, 0] + squeeze_amt, 0.0, 1.0)
        print(f"[exec] grasp squeeze +{squeeze_amt:.2f} (normalized close) over {squeeze_steps} "
              f"frames after frame {squeeze_start}")

    # Per-frame streaming period. The pregrasp->grasp replay segment is slowed
    # to --replay_rate_hz so the hand does not close on the object too fast;
    # every other phase runs at --rate_hz.
    periods = np.full(Tn, 1.0 / float(args.rate_hz), dtype=np.float64)
    replay_lo, replay_hi = n_a + n_b, n_a + n_b + n_r
    if replay_hi > replay_lo:
        periods[replay_lo:replay_hi] = 1.0 / float(args.replay_rate_hz)
    nominal_duration = float(np.sum(periods))
    periods, retimed_steps, max_stretch = retime_step_periods(
        arm_traj,
        arm_pose_traj,
        periods,
        max_joint_speed_deg_s=float(args.max_planned_joint_speed_deg_s),
        max_linear_speed_mps=float(args.max_cartesian_linear_speed_mps),
        max_angular_speed_rps=float(args.max_cartesian_angular_speed_rps),
    )
    if retimed_steps:
        print(f"[exec] safety retiming: slowed {retimed_steps} transitions, "
              f"max {max_stretch:.2f}x, duration {nominal_duration:.1f}s "
              f"-> {np.sum(periods):.1f}s")
    assert_bounded_pose_steps(
        arm_pose_traj,
        rate_hz=1.0,
        max_linear_speed_mps=float(args.max_cartesian_linear_speed_mps),
        max_angular_speed_rps=float(args.max_cartesian_angular_speed_rps),
        step_periods=periods,
    )
    assert_bounded_joint_steps(
        arm_traj,
        periods,
        max_joint_speed_deg_s=float(args.max_planned_joint_speed_deg_s),
    )
    print(f"[exec] replay segment frames {replay_lo}..{replay_hi - 1} "
          f"@ {args.replay_rate_hz} Hz (rest @ {args.rate_hz} Hz)")

    home_rad = np.deg2rad(np.asarray(args.home_deg, dtype=np.float64))
    arm0_home_equivalent = home_rad + np.arctan2(
        np.sin(arm_traj[0] - home_rad), np.cos(arm_traj[0] - home_rad)
    )
    if not np.allclose(arm0_home_equivalent, home_rad, atol=1e-3):
        print(f"[exec] WARNING: traj[0] arm != HOME; traj[0]={arm_traj[0]}")
    print(f"[exec] T={Tn}, rate={args.rate_hz} Hz (~{Tn / args.rate_hz:.1f}s)")
    print(f"[exec] arm[0] deg: {np.rad2deg(arm0_home_equivalent).round(2).tolist()}")
    print(f"[exec] gripper[0] close={hand_msgs[0, 0]:.3f}  gripper[{Tn-1}] close={hand_msgs[-1, 0]:.3f}")

    # Optional video capture (CaptureSession) -- lives next to the planning
    # artefacts inside abs_save_path/video.
    cs = None
    capture_dir_abs = None
    if args.record_video:
        from paradex.dataset_acqusition.capture import CaptureSession
        capture_rel_path = os.path.join(_to_shared_data_path(abs_save_path).replace("shared_data/", ""),
                                        "video", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        capture_dir_abs = os.path.join(shared_dir, capture_rel_path)
        cs = CaptureSession(camera=True, realsense=bool(args.realsense))
        cs.start(capture_rel_path)
        print(f"[exec] recording video -> {capture_dir_abs}/raw")
        print("[exec] pre-roll: 3s before motion ...")
        time.sleep(3.0)

    print(f"[exec] init xarm (servo_api=angle_j, "
          f"command_rate={args.arm_servo_rate_hz:.1f} Hz)...")
    xarm = get_arm(
        "xarm",
        servo_api="angle_j",
        fps=float(args.arm_servo_rate_hz),
    )
    print("[exec] init robotiq_2f85 (ROS2 topic controller; action 0=open .. 1=closed)...")
    hand = get_hand("robotiq_2f85")

    def _send_hand(cmd: np.ndarray) -> None:
        hand.move(np.asarray(cmd, dtype=np.float64).reshape(1))

    try:
        # Phase 0: home + open hand, wait for arrival.
        print(f"[exec] commanding HOME deg={list(np.rad2deg(home_rad).round(2))} + hand[0]={hand_msgs[0].tolist()}")
        move_xarm_home_safely(
            xarm,
            home_rad,
            max_joint_speed_deg_s=float(args.home_max_joint_speed_deg_s),
            stream_hz=float(args.home_preposition_hz),
        )
        _send_hand(hand_msgs[0])
        ok = _wait_xarm_at_target(xarm, home_rad,
                                  np.deg2rad(float(args.home_tol_deg)),
                                  float(args.home_timeout))
        if not ok:
            raise RuntimeError(f"arm did not reach HOME within {args.home_timeout}s")
        if xarm.is_error() or (hasattr(hand, "is_error") and hand.is_error()):
            raise RuntimeError("controller in error state before streaming")
        watchdog = XArmJointWatchdog(
            robot_module,
            max_speed_deg_s=float(args.max_measured_joint_speed_deg_s),
            limit_margin_rad=float(args.live_joint_limit_margin_rad),
        )
        watchdog.check(xarm, context="initialization")

        def _stream_joint_range(start: int, stop: int) -> None:
            if stop <= start:
                return
            commands = np.concatenate(
                (arm_traj[start:stop], hand_msgs[start:stop]), axis=1
            )
            dense, dense_periods, source_frames = densify_timed_commands(
                commands,
                periods[start:stop],
                command_rate_hz=float(args.arm_servo_rate_hz),
            )
            source_frames += start
            t0 = time.perf_counter()
            clock = 0.0
            for command, period, frame in zip(dense, dense_periods, source_frames):
                frame = int(frame)
                watchdog.check(
                    xarm,
                    context=(
                        f"frame={frame}, phase={trajectory_phase(frame, pipeline_result)}"
                    ),
                )
                xarm.move(command[:6], is_servo=True)
                _send_hand(command[6:])
                if xarm.is_error() or (hasattr(hand, "is_error") and hand.is_error()):
                    raise RuntimeError(f"controller error at frame {frame}")
                clock += float(period)
                sleep_for = t0 + clock - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

        # Phase 1: grasp sequence -- stream frames up to the end of replay with
        # the planned hand values unchanged. Per-frame period from `periods`.
        grasp_end = squeeze_start if do_squeeze else Tn
        print(f"[exec] streaming grasp sequence: frames 0..{grasp_end - 1} ...")
        _stream_joint_range(0, grasp_end)

        # Phase 1.5: explicit squeeze -- arm frozen at the last grasp pose,
        # hand ramps from the planned grasp pose to the squeezed pose.
        if do_squeeze:
            hand_from = hand_msgs[squeeze_start - 1].copy()
            hand_to = hand_msgs[squeeze_start].copy()
            arm_hold = arm_traj[squeeze_start - 1]
            print(f"[exec] squeezing hand over {squeeze_steps} frames ...")
            for s in range(1, squeeze_steps + 1):
                watchdog.check(xarm, context=f"squeeze_step={s}")
                a = float(s) / float(squeeze_steps)
                cmd = (1.0 - a) * hand_from + a * hand_to
                xarm.move(arm_hold, is_servo=True)
                _send_hand(cmd)
                if xarm.is_error() or (hasattr(hand, "is_error") and hand.is_error()):
                    raise RuntimeError(f"controller error during squeeze step {s}")
                time.sleep(1.0 / float(args.rate_hz))

            # Phase 2: hold + lift with the squeezed hand pose.
            print(f"[exec] streaming hold + lift: frames {squeeze_start}..{Tn - 1} ...")
            _stream_joint_range(squeeze_start, Tn)
        print("[exec] streaming done")
        final_pose = arm_traj[-1]
        final_hand = hand_msgs[-1]
        post_lift_hold_seconds = float(args.post_lift_hold_seconds)
        if post_lift_hold_seconds > 0:
            print(f"[exec] holding lifted grasp for {post_lift_hold_seconds:.1f}s ...")
            hold_deadline = time.perf_counter() + post_lift_hold_seconds
            while time.perf_counter() < hold_deadline:
                watchdog.check(xarm, context="post_lift_hold")
                xarm.move(final_pose, is_servo=True)
                _send_hand(final_hand)
                if xarm.is_error() or (hasattr(hand, "is_error") and hand.is_error()):
                    raise RuntimeError("controller error during post-lift grasp hold")
                time.sleep(1.0 / float(args.rate_hz))

        release_steps = int(args.release_steps)
        if release_steps > 0:
            open_hand = hand_msgs[0]
            print(f"[exec] releasing grasp over {release_steps} frames ...")
            for step in range(1, release_steps + 1):
                watchdog.check(xarm, context=f"release_step={step}")
                alpha = float(step) / float(release_steps)
                _send_hand((1.0 - alpha) * final_hand + alpha * open_hand)
                xarm.move(final_pose, is_servo=True)
                if xarm.is_error() or (hasattr(hand, "is_error") and hand.is_error()):
                    raise RuntimeError(f"controller error during release step {step}")
                time.sleep(1.0 / float(args.rate_hz))
            print("[exec] release done")

        print("[exec] returning xArm to HOME after release ...")
        move_xarm_home_safely(
            xarm,
            home_rad,
            max_joint_speed_deg_s=float(args.return_home_max_joint_speed_deg_s),
            stream_hz=float(args.home_preposition_hz),
        )
        if not _wait_xarm_at_target(
            xarm,
            home_rad,
            np.deg2rad(float(args.home_tol_deg)),
            float(args.home_timeout),
        ):
            raise RuntimeError(f"xArm did not return HOME within {args.home_timeout}s")
        print("[exec] xArm returned HOME")
    finally:
        try: hand.end()
        except Exception as e: print(f"[exec] hand.end error: {e}")
        try: xarm.end(set_break=False)
        except Exception as e: print(f"[exec] xarm.end error: {e}")
        if cs is not None:
            try: cs.stop()
            except Exception as e: print(f"[exec] cs.stop error: {e}")
            try: save_current_C2R(capture_dir_abs)
            except Exception as e: print(f"[exec] save_current_C2R error: {e}")
            try: cs.end()
            except Exception as e: print(f"[exec] cs.end error: {e}")
            print(f"[exec] capture saved to {capture_dir_abs}")


def _run_h2r_pipeline(args, mesh_name: str, pose_robot: np.ndarray,
                     robot_module: RobotModule, abs_save_path: str) -> Dict[str, Any]:
    """End-to-end h2r pipeline (xarm_robotiq_2f85):

       Phase A: HOME-height planar+rotate -> vertical pregrasp descent
       Phase B: per-waypoint IK to track the canonical wrist trajectory while
                replaying finger qpos from result.npz

    The two phases are concatenated into a single (Na+W, n_dof) trajectory in
    URDF joint order. The viewer is intentionally NOT touched here: caller
    receives the full trajectory and registers it with add_traj after viser is
    launched, so the long curobo planning step does not block the GUI."""
    # Curobo builds retract_config on its module-default cuda device (cuda:0)
    # during MotionGenConfig.load_from_robot_config, then forward_bound_pos_warp
    # fails when tensor_args.device is anything else. Constrain
    # CUDA_VISIBLE_DEVICES before importing torch/curobo so cuda:0 always maps
    # to the desired physical GPU, and tell our helpers to use cuda:0.
    curobo_device = str(args.curobo_device)
    if curobo_device.startswith("cuda:"):
        gpu_id = curobo_device.split(":", 1)[1]
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
        print(f"[pipe] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
              f"(curobo will use cuda:0 internally)")
        curobo_device = "cuda:0"

    from object6d.capture_rcc import (
        _run_curobo_plan,
        _normalize_curobo_robot_cfg_paths,
        _normalize_curobo_robot_cfg_paths_with_cfg_file,
        _rotmat_to_wxyz,
    )
    from paradex.utils.file_io import load_yaml
    from scipy.spatial.transform import Rotation

    # ---- 1. load pre-recorded canonical grasp trajectory
    res_path = os.path.join(args.result_root, mesh_name, str(args.scene), "result_robotiq.npz")
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"no packed result: {res_path}")
    d = np.load(res_path, allow_pickle=True)
    pack_to_pose = _pack_frame_to_pose_frame(d, args.pose_mesh_frame)
    key = f"{args.src}_pregrasp_traj"
    if key not in d.files:
        raise KeyError(f"{key} missing in {res_path}")
    traj_canon = d[key][args.particle]  # (W, 6 + 6 mimic joints) wrist + gripper joints
    jnames_traj = [str(x) for x in d["revolute_joint_names"]]
    if "base_link_from_root_rpy" in d.files:
        from scipy.spatial.transform import Rotation as _R
        rpy = np.asarray(d["base_link_from_root_rpy"], dtype=np.float64)
        root_to_base = np.eye(4, dtype=np.float32)
        root_to_base[:3, :3] = _R.from_euler("xyz", rpy).as_matrix().astype(np.float32)
    else:
        root_to_base = MOUNT_TO_BASE
    W = int(traj_canon.shape[0])
    print(f"[pipe] {mesh_name} scene={args.scene} src={args.src} "
          f"particle={args.particle}  W={W} (wrist+finger)")

    # ---- 2. canonical wrist -> robot frame (object already at pose_robot)
    wrist_T_robot = np.zeros((W, 4, 4), dtype=np.float32)
    for i, w in enumerate(traj_canon):
        Wc = np.eye(4, dtype=np.float32)
        Wc[:3, :3] = Rotation.from_euler("XYZ", w[3:6]).as_matrix()
        Wc[:3, 3] = w[:3]
        wrist_T_robot[i] = (pose_robot @ pack_to_pose @ Wc @ root_to_base).astype(np.float32)

    # ---- 3. HOME qpos in full URDF joint order, with the gripper fully open.
    full_joint_names = list(robot_module.get_joint_names())
    n_dof = int(robot_module.get_num_joints())
    name_to_full_idx = {n: i for i, n in enumerate(full_joint_names)}
    name_to_finger_idx = {n: i for i, n in enumerate(jnames_traj)}
    home_qpos_full = np.zeros(n_dof, dtype=np.float32)
    home_qpos_full[:6] = np.deg2rad(np.asarray(args.home_deg, dtype=np.float32))
    matched_finger = sum(jn in name_to_full_idx for jn in name_to_finger_idx)
    print(f"[pipe] finger mapping matched {matched_finger}/{len(jnames_traj)}; Stage A hand is open")

    # ---- 4. curobo robot_cfg (xarm_inspire_f1; cspace=12, active=6 arm)
    cfg_path = os.path.expanduser(args.curobo_robot_cfg)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"curobo cfg not found: {cfg_path}")
    robot_cfg = _normalize_curobo_robot_cfg_paths_with_cfg_file(load_yaml(cfg_path), cfg_path)
    robot_cfg = _normalize_curobo_robot_cfg_paths(robot_cfg)
    # Keep locked hand joints open throughout Stage A.
    lock_joints = robot_cfg["robot_cfg"]["kinematics"].get("lock_joints")
    if isinstance(lock_joints, dict):
        for jn in lock_joints:
            if jn in name_to_full_idx:
                lock_joints[jn] = float(home_qpos_full[name_to_full_idx[jn]])

    # ---- 5. world (table + current object as collision mesh)
    table_size = np.array([float(x) for x in args.table_size_xyz.split(",")], dtype=np.float32)
    table_pos = np.array([float(x) for x in args.table_pos_xyz.split(",")], dtype=np.float32)
    obj_wxyz = _rotmat_to_wxyz(pose_robot[:3, :3])
    world_dict = {
        "cuboid": {
            "table": {
                "dims": table_size.tolist(),
                "pose": [float(table_pos[0]), float(table_pos[1]), float(table_pos[2]),
                         1.0, 0.0, 0.0, 0.0],
            }
        },
        "mesh": {
            "object": {
                "file_path": _resolve_mesh_path(mesh_name, os.path.expanduser(args.mesh_root_dir), args.pose_mesh_frame),
                "pose": [float(pose_robot[0, 3]), float(pose_robot[1, 3]), float(pose_robot[2, 3]),
                         float(obj_wxyz[0]), float(obj_wxyz[1]), float(obj_wxyz[2]), float(obj_wxyz[3])],
            }
        },
    }

    # ---- 6. Phase A: HOME-height planar+rotate, then vertical descend.
    active_jn = curobo_active_joint_names(robot_cfg)
    start_qpos_cspace = np.zeros(len(active_jn), dtype=np.float32)
    for ci, jn in enumerate(active_jn):
        if jn in name_to_full_idx:
            start_qpos_cspace[ci] = home_qpos_full[name_to_full_idx[jn]]
    ee_link = str(robot_cfg["robot_cfg"]["kinematics"]["ee_link"])
    home_eef_pose = robot_module.forward_kinematics(home_qpos_full)[ee_link]
    stage_a_poses, stage_a_ends = staged_home_to_pregrasp(
        home_eef_pose,
        wrist_T_robot[0],
        rate_hz=float(args.rate_hz),
        planar_speed_mps=float(args.stage_a_linear_speed_mps),
        angular_speed_rps=float(args.stage_a_angular_speed_rps),
        raise_speed_mps=float(args.stage_a_raise_speed_mps),
        descent_speed_mps=float(args.stage_a_descent_speed_mps),
        min_hover_clearance_m=float(args.stage_a_hover_clearance_m),
        table_normal=np.asarray([float(x) for x in args.table_normal_xyz.split(",")]),
    )
    print(f"[pipe] Phase A staged waypoints: {stage_a_ends}")
    approach_active, n_stage_a_success = _solve_ik_chain_curobo(
        robot_cfg,
        stage_a_poses,
        start_qpos_cspace,
        device=curobo_device,
        num_seeds=int(args.ik_num_seeds),
        null_space_weight=float(args.ik_null_space_weight),
        position_threshold=float(args.ik_position_threshold),
        rotation_threshold=float(args.ik_rotation_threshold),
        sample_stride=int(args.viser_ik_stride),
        world_cfg_dict=world_dict,
    )
    if n_stage_a_success != len(stage_a_poses):
        print(f"[pipe] Phase A FAILED: {n_stage_a_success}/{len(stage_a_poses)} IK waypoints")
        return None
    approach_active = unwrap_xarm_joints(approach_active, active_jn)
    Na = int(approach_active.shape[0])
    print(f"[pipe] Phase A done: {Na} waypoints, active_dof={approach_active.shape[1]}")

    # ---- 7. Phase B: per-frame IK to follow wrist trajectory
    print(f"[pipe] Phase B: per-waypoint IK across {W} waypoints")
    replay_active, n_replay_success = _solve_ik_chain_curobo(
        robot_cfg=robot_cfg,
        target_poses=wrist_T_robot,
        seed_arm_qpos=approach_active[-1],
        device=curobo_device,
        num_seeds=int(args.ik_num_seeds),
        null_space_weight=float(args.ik_null_space_weight),
        position_threshold=float(args.ik_position_threshold),
        rotation_threshold=float(args.ik_rotation_threshold),
        sample_stride=int(args.viser_ik_stride),
    )
    if n_replay_success != W:
        print(f"[pipe] Phase B FAILED: {n_replay_success}/{W} IK waypoints")
        return None
    replay_active = unwrap_xarm_joints(
        np.concatenate([approach_active[-1:], replay_active], axis=0), active_jn
    )[1:]

    # ---- 8. expand active qpos to full URDF order; overlay hand qpos
    name_to_active_idx = {n: i for i, n in enumerate(active_jn)}

    # Phase A: arm from approach plan, hand stays at the pre-shaped grasp pose.
    approach_full = np.tile(home_qpos_full, (Na, 1))
    for jn, ai in name_to_active_idx.items():
        if jn in name_to_full_idx:
            approach_full[:, name_to_full_idx[jn]] = approach_active[:, ai]

    # Phase B: arm from replay IK, hand from canonical traj by joint name.
    replay_full = np.tile(approach_full[-1], (W, 1))
    for jn, ai in name_to_active_idx.items():
        if jn in name_to_full_idx:
            replay_full[:, name_to_full_idx[jn]] = replay_active[:, ai]
    for jn, fi in name_to_finger_idx.items():
        if jn in name_to_full_idx:
            replay_full[:, name_to_full_idx[jn]] = traj_canon[:, 6 + fi].astype(np.float32)

    # Hand pre-close bridge: after reaching the pregrasp arm pose (end of
    # Phase A), interpolate hand from open -> replay[0] hand pose over
    # ``--hand_interp_steps`` frames before starting the wrist-tracking replay.
    # Arm joints stay constant during the bridge so only the fingers move.
    nh = int(max(0, args.hand_interp_steps))
    bridge_full = np.zeros((0, n_dof), dtype=np.float32)
    q_a_end = approach_full[-1]
    q_b_start = replay_full[0]
    if nh > 0 and not np.allclose(q_a_end, q_b_start, atol=1e-6):
        # Include both endpoints in the linspace, then drop them: endpoint 0 is
        # already approach[-1] and endpoint 1 is replay[0], so emitting the
        # interior (nh-1) points gives a smooth ramp without duplicating either.
        if nh >= 2:
            alphas = np.linspace(0.0, 1.0, nh + 1, dtype=np.float32)[1:-1]
        else:
            alphas = np.array([0.5], dtype=np.float32)
        bridge_full = (1.0 - alphas[:, None]) * q_a_end[None, :] + alphas[:, None] * q_b_start[None, :]

    Nb = int(bridge_full.shape[0])

    # ---- 8bc. Hold the grasp pose still for a few frames so the gripper has
    # time to settle (mostly visual; arm and hand do not move during the dwell).
    Nh = int(max(0, args.grasp_hold_frames))
    hold_full = np.tile(replay_full[-1], (Nh, 1)) if Nh > 0 else np.zeros((0, n_dof), dtype=np.float32)

    # ---- 8c. Phase C: lift +Z after grasp (curobo MotionGen, object removed
    # from collision world since we are holding it). Start qpos is the last
    # replay arm config; hand is held at the last grasp pose throughout.
    lift_h = float(args.lift_height_m)
    Nc = 0
    lift_full = np.zeros((0, n_dof), dtype=np.float32)
    if lift_h > 0:
        print(f"[pipe] Phase C: lift +{lift_h * 100:.1f}cm in robot +Z")
        lift_target = wrist_T_robot[-1].copy()
        lift_target[2, 3] += lift_h

        start_lift_cspace = replay_active[-1].copy()

        # Phase C is a short straight-up motion holding the object; we don't
        # want curobo to reject the start pose because the gripper is touching
        # the object/table. Replace the world with a single tiny dummy cuboid
        # parked 10 m away so curobo's primitive collision checker has at least
        # one obstacle (it errors out on a fully empty world).
        lift_world = {
            "cuboid": {
                "_dummy_far": {
                    "dims": [0.01, 0.01, 0.01],
                    "pose": [10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0],
                }
            },
            "mesh": {},
        }
        lift_plan = _run_curobo_plan(
            robot_cfg=robot_cfg,
            world_cfg_dict=lift_world,
            start_qpos=start_lift_cspace,
            target_eef_pose_robot=lift_target,
            device=curobo_device,
            seeds=int(args.curobo_seeds),
            self_collision=True,
        )
        if lift_plan.get("success"):
            lift_active = np.asarray(lift_plan["trajectory"], dtype=np.float32)
            # Drop the first sample so we don't repeat replay's last frame.
            if lift_active.shape[0] > 1:
                lift_active = lift_active[1:]
            Nc = int(lift_active.shape[0])
            lift_full = np.tile(replay_full[-1], (Nc, 1))
            for jn, ai in name_to_active_idx.items():
                if jn in name_to_full_idx:
                    lift_full[:, name_to_full_idx[jn]] = lift_active[:, ai]
            print(f"[pipe] Phase C done: {Nc} waypoints")
        else:
            print(f"[pipe] Phase C FAILED: {lift_plan.get('status')} {lift_plan.get('error')}")
            return None

    full_traj = np.concatenate([approach_full, bridge_full, replay_full, hold_full, lift_full], axis=0)
    assert_arm_joint_limits(robot_module, full_traj)
    approach_arm_poses = end_effector_poses_to_command_link(
        robot_module,
        stage_a_poses,
        home_qpos_full,
        end_effector_link=ee_link,
    )
    replay_arm_poses = end_effector_poses_to_command_link(
        robot_module,
        wrist_T_robot,
        home_qpos_full,
        end_effector_link=ee_link,
    )
    bridge_arm_poses = np.tile(approach_arm_poses[-1], (Nb, 1, 1))
    hold_arm_poses = np.tile(replay_arm_poses[-1], (Nh, 1, 1))
    lift_arm_poses = (
        arm_link_pose_trajectory(robot_module, lift_full)
        if Nc > 0 else np.zeros((0, 4, 4), dtype=np.float64)
    )
    arm_pose_traj = np.concatenate(
        [approach_arm_poses, bridge_arm_poses, replay_arm_poses,
         hold_arm_poses, lift_arm_poses],
        axis=0,
    )
    if len(arm_pose_traj) != len(full_traj):
        raise RuntimeError("Cartesian command and Viser preview lengths differ")

    # ---- 9. save artefacts next to other capture outputs
    np.save(os.path.join(abs_save_path, "h2r_pipeline_traj.npy"), full_traj)
    np.save(os.path.join(abs_save_path, "h2r_pipeline_link6_poses.npy"), arm_pose_traj)
    with open(os.path.join(abs_save_path, "h2r_pipeline_meta.json"), "w") as f:
        json.dump({"n_approach": Na, "n_hand_interp": Nb, "n_replay": W,
                   "n_grasp_hold": Nh, "n_lift": Nc, "lift_height_m": lift_h,
                   "joint_names": full_joint_names,
                   "active_joint_names": active_jn,
                   "home_deg": list(args.home_deg),
                   "hand_interp_steps_requested": int(args.hand_interp_steps),
                   "grasp_hold_frames_requested": int(args.grasp_hold_frames)}, f, indent=2)
    print(f"[pipe] saved h2r_pipeline_traj.npy shape={full_traj.shape}")

    # ---- 10. caller-facing result; caller adds add_traj after viser starts
    cursor = 0
    ranges = []
    for name, n in [("approach", Na), ("hand_interp", Nb), ("replay", W),
                    ("grasp_hold", Nh), ("lift", Nc)]:
        if n > 0:
            ranges.append((name, cursor, cursor + n - 1))
            cursor += n
    range_str = ", ".join(f"{n}[{lo}..{hi}]" for n, lo, hi in ranges)
    print(f"[pipe] planning done; trajectory shape={full_traj.shape}, "
          f"ranges {range_str}. (viser will open next)")
    return {"full_traj": full_traj, "arm_pose_traj": arm_pose_traj, "range_str": range_str,
            "stage_a_ends": dict(stage_a_ends),
            "n_approach": Na, "n_hand_interp": Nb, "n_replay": W,
            "n_grasp_hold": Nh, "n_lift": Nc}


def _replay_pregrasp(args, vis, mesh_name: str, pose_robot: np.ndarray) -> None:
    """The floating-hand legacy preview is Inspire-specific; use --pipeline."""
    raise NotImplementedError("capture_object6d_robotiq.py supports --pipeline only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "--mesh_name", dest="mesh_name", required=True)
    parser.add_argument("--save_path", default="capture/h2r/object_6d", help="relative path under shared_data")
    parser.add_argument("--rpc_addr", default="tcp://192.168.0.14:5570")
    parser.add_argument("--rpc_timeout_ms", type=int, default=300000)
    parser.add_argument("--rcc_entry", default="image_main.py")
    parser.add_argument("--mesh_root_dir", default=os.path.join(shared_dir, "mesh_new"))
    parser.add_argument("--pose_mesh_frame", default="original", choices=("original", "viser", "canonical"),
                        help="object frame reported by the 6D pose RPC: \'original\' = raw <obj>.obj (FoundationPose, "
                             "default); \'viser\' = <obj>_viser.obj. Packed wrists are converted with viser_align.")
    parser.add_argument("--c2r_path", default=None, help="optional C2R.npy override")
    parser.add_argument("--robot_urdf", default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--viser_addr", default="tcp://localhost:5572",
                        help="ZMQ REQ address of a running viser_server. Push "
                             "the scene + trajectory to it (non-blocking) so the "
                             "execute prompt fires immediately. Empty string => "
                             "fall back to launching a local ViserViewer "
                             "(blocking; prompt fires after Ctrl+C).")
    parser.add_argument("--debug_plane_width", type=float, default=1.0,
                        help="floating debug image plane width (meters)")
    parser.add_argument("--debug_anchor_x", type=float, default=0.5,
                        help="x offset (m) of debug image planes (shift right)")
    parser.add_argument("--debug_anchor_y", type=float, default=-1.0,
                        help="y offset (m) of debug image planes")
    parser.add_argument("--debug_anchor_z", type=float, default=0.0,
                        help="ground height (m) of debug image planes; plane "
                             "center auto-lifted by half its height so the "
                             "bottom edge sits at this z")
    parser.add_argument("--debug_image_mode", default="popup",
                        choices=["popup", "viser", "both", "none"],
                        help="how to show RPC mask overlay PNGs: popup = OS "
                             "image viewer window(s); viser = floating planes "
                             "in the 3D scene; both; none")
    parser.add_argument(
        "--normalize", default="none", choices=pose_normalize.MODES,
        help=("generic pose normalization for objects without registered symmetry; "
              "known mesh_new symmetries are handled automatically and take precedence"))
    parser.add_argument(
        "--no_object_symmetry_normalization",
        dest="object_symmetry_normalization",
        action="store_false",
        help=("disable automatic capture_sil-style continuous-symmetry "
              "canonicalization for objects registered in object_symmetry.py"),
    )
    parser.set_defaults(object_symmetry_normalization=True)
    # --- pre-grasp -> grasp motion replay (our H2R result) ---
    parser.add_argument("--scene", type=int, default=None,
                        help="capture scene/clip number; with --pipeline, omission "
                             "auto-selects the unique packed scene for this object")
    parser.add_argument("--src", default="robot", choices=("robot", "human"),
                        help="which contact source's grasp to replay")
    parser.add_argument("--particle", type=int, default=0,
                        help="packed candidate index; 0 = best-ranked sim-verified grasp")
    parser.add_argument(
        "--result_root",
        default=os.path.join(shared_dir, "capture", "h2r", "hrdex_sim_verified", "legacy40", "robotiq_2f85"),
        help="root holding <obj>/<scene>/result_robotiq.npz (hrdex_pack.py output)")
    parser.add_argument(
        "--hand_urdf",
        default=str(PROJECT_ROOT / "rsc" / "robot" / "inspire_f1" / "URDF-2"
                    / "URDF-2" / "RH56F1_R" / "urdf" / "RH56F1_R.urdf"),
        help="inspire_f1 hand URDF (RH56F1_R) for the legacy motion replay")
    # --- new: full h2r pipeline (HOME -> pregrasp -> grasp replay) ---
    parser.add_argument("--pipeline", action="store_true",
                        help="run the staged h2r pipeline (HOME-height planar+rotate, "
                             "vertical descent, per-waypoint IK to preview the wrist "
                             "trajectory). Requires --scene. Trajectory is "
                             "added to the viewer for playback via add_traj.")
    parser.add_argument("--home_deg", nargs=6, type=float, default=HOME_DEG,
                        help="xarm home joint angles in degrees (6 values)")
    parser.add_argument("--curobo_robot_cfg", default=DEFAULT_CUROBO_ROBOT_CFG,
                        help="curobo robot config yaml (xarm_inspire_f1)")
    parser.add_argument("--curobo_device", default="cuda:1")
    parser.add_argument("--curobo_seeds", type=int, default=128)
    parser.add_argument("--ik_num_seeds", type=int, default=32)
    parser.add_argument("--ik_position_threshold", type=float, default=0.005)
    parser.add_argument("--ik_rotation_threshold", type=float, default=0.05)
    parser.add_argument("--ik_null_space_weight", type=float, default=1.0)
    parser.add_argument("--viser_ik_stride", type=int, default=10,
                        help="solve every Nth waypoint for the Viser-only IK preview; "
                             "real xArm execution uses the dense Cartesian path")
    parser.add_argument("--stage_a_linear_speed_mps", type=float, default=0.12)
    parser.add_argument("--stage_a_angular_speed_rps", type=float, default=0.70)
    parser.add_argument("--stage_a_raise_speed_mps", type=float, default=0.03,
                        help="vertical HOME-to-hover speed")
    parser.add_argument("--stage_a_descent_speed_mps", type=float, default=0.06)
    parser.add_argument("--stage_a_hover_clearance_m", type=float, default=0.12)
    parser.add_argument("--table_normal_xyz", default="0,0,1",
                        help="table-normal direction in robot coordinates")
    parser.add_argument("--symmetry_max_wrist_facing_deg", type=float, default=90.0,
                        help="maximum XY angle between object->wrist and object->xArm "
                             "when selecting a symmetry-equivalent grasp branch")
    parser.add_argument("--table_size_xyz", default="2.4,1.6,0.1")
    parser.add_argument("--table_pos_xyz", default="0.0,0.0,-0.05")
    parser.add_argument("--hand_interp_steps", type=int, default=60,
                        help="bridge frames inserted between Phase A end (arm "
                             "at pregrasp, hand open) and Phase B start (hand "
                             "at first replay finger qpos). Arm is held; only "
                             "fingers ramp. 0 disables the bridge.")
    parser.add_argument("--grasp_hold_frames", type=int, default=30,
                        help="dwell frames inserted after the grasp replay "
                             "(arm + hand held still) before the lift phase. "
                             "0 disables.")
    parser.add_argument("--grasp_squeeze", type=float, default=0.05,
                        help="after the grasp sequence runs, ramp every hand "
                             "joint except thumb_1 down by this raw amount so "
                             "the grasp tightens. 0 disables.")
    parser.add_argument("--squeeze_steps", type=int, default=20,
                        help="number of frames for the explicit squeeze ramp "
                             "(arm frozen, hand-only). At rate_hz this is the "
                             "squeeze duration.")
    parser.add_argument("--lift_height_m", type=float, default=0.07,
                        help="post-grasp lift in robot +Z (meters). 0 disables "
                             "Phase C. Planned via curobo MotionGen with the "
                             "object removed from the collision world.")
    parser.add_argument("--post_lift_hold_seconds", type=float, default=5.0,
                        help="hold the lifted grasp before release")
    parser.add_argument("--release_steps", type=int, default=30,
                        help="fully open the gripper over this many frames after "
                             "the post-lift hold. 0 disables.")
    # --- real-robot execution after the viser session ---
    parser.add_argument("--rate_hz", type=float, default=30.0,
                        help="streaming rate to xarm + inspire_f1 (Hz)")
    parser.add_argument("--replay_rate_hz", type=float, default=15.0,
                        help="streaming rate for the pregrasp->grasp replay "
                             "segment only (Hz). Lower = slower close.")
    parser.add_argument("--home_tol_deg", type=float, default=2.0)
    parser.add_argument("--home_timeout", type=float, default=20.0)
    parser.add_argument("--home_max_joint_speed_deg_s", type=float, default=10.0)
    parser.add_argument("--return_home_max_joint_speed_deg_s", type=float, default=30.0,
                        help="maximum xArm joint speed when returning HOME after release")
    parser.add_argument("--home_preposition_hz", type=float, default=30.0)
    parser.add_argument("--arm_servo_rate_hz", type=float, default=100.0,
                        help="dense xArm joint-servo command rate during replay")
    parser.add_argument("--max_planned_joint_speed_deg_s", type=float, default=30.0)
    parser.add_argument("--max_cartesian_linear_speed_mps", type=float, default=0.10)
    parser.add_argument("--max_cartesian_angular_speed_rps", type=float, default=0.50)
    parser.add_argument("--max_measured_joint_speed_deg_s", type=float, default=60.0)
    parser.add_argument("--live_joint_limit_margin_rad", type=float, default=0.03)
    parser.add_argument("--hand_side", default="right", choices=["right", "left"])
    parser.add_argument("--record_video", action="store_true", default=True,
                        help="record video via CaptureSession during execution")
    parser.add_argument("--no_record_video", dest="record_video",
                        action="store_false")
    parser.add_argument("--realsense", action="store_true",
                        help="also record realsense depth (passed to CaptureSession)")
    parser.add_argument("--yes", action="store_true",
                        help="skip the post-viser execute confirmation prompt")
    args = parser.parse_args()
    validate_motion_args(args)
    args.result_root = os.path.expanduser(args.result_root)
    if args.pipeline and args.scene is None:
        args.scene = resolve_scene_arg(
            args.result_root, args.mesh_name, "result_robotiq.npz", args.scene
        )
        print(f"[pipe] auto-selected scene={args.scene}")

    # mesh existence check (RPC server only needs mesh_name)
    mesh_path = _resolve_mesh_path(args.mesh_name, os.path.expanduser(args.mesh_root_dir), args.pose_mesh_frame)
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh not found: {mesh_path}")

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # When a scene is specified we group runs under it so re-captures of the
    # same scene stay together. The contact source (robot/human) is appended
    # right after the scene number so video + plan artefacts stay separated.
    if args.scene is not None:
        rel_save_path = os.path.join(args.save_path, args.mesh_name,
                                     str(args.scene), args.src, date_str)
    else:
        rel_save_path = os.path.join(args.save_path, args.mesh_name, date_str)
    abs_save_path = os.path.join(shared_dir, rel_save_path)
    os.makedirs(abs_save_path, exist_ok=True)

    # 1) capture image
    rcc = remote_camera_controller(args.rcc_entry)
    save_current_camparam(abs_save_path)
    print(f"Capturing image to {rel_save_path}")
    try:
        rcc.start("image", False, f"shared_data/{rel_save_path}/raw")
        rcc.stop()
    finally:
        rcc.end()
    print(f"image_dir: {abs_save_path}")

    # 2) 6d pose inference via RPC
    req = {
        "command": "infer",
        "image_path": _to_shared_data_path(abs_save_path),
        "mesh_name": args.mesh_name,
    }
    res = _send_rpc_once(args.rpc_addr, req, timeout_ms=args.rpc_timeout_ms)
    pose_out = _extract_pose_response(res)

    object_json_path = os.path.join(abs_save_path, "object_6d.json")
    with open(object_json_path, "w", encoding="utf-8") as f:
        json.dump(pose_out, f, ensure_ascii=False, indent=2)
    print(f"saved: {object_json_path}")
    print(json.dumps(pose_out, ensure_ascii=False, indent=2))

    # 3) coordinate transform: world -> robot frame
    if args.c2r_path is not None:
        c2r = np.load(os.path.expanduser(args.c2r_path))
    else:
        c2r = load_current_C2R()
    c2r = _to_4x4(c2r)
    np.save(os.path.join(abs_save_path, "C2R.npy"), c2r)

    pose_world_raw = _to_4x4(np.asarray(pose_out["pose_world"], dtype=np.float32))
    r2c = np.linalg.inv(c2r)
    pose_robot_raw = r2c @ pose_world_raw
    pose_robot = pose_robot_raw.copy()
    symmetry_info = None
    normalization_applied = "none"
    symmetry_wrist_object = None
    if (
        args.pipeline
        and args.object_symmetry_normalization
        and args.pose_mesh_frame == "original"
        and pose_normalize.has_known_object_symmetry(args.mesh_name)
    ):
        symmetry_result_path = os.path.join(
            args.result_root, args.mesh_name, str(args.scene), "result_robotiq.npz"
        )
        with np.load(symmetry_result_path, allow_pickle=True) as symmetry_packed:
            symmetry_pack_to_pose = pack_frame_to_pose_frame(
                symmetry_packed, args.pose_mesh_frame
            )
        symmetry_wrist_object = pose_normalize.load_packed_pregrasp_wrist(
            symmetry_result_path,
            args.src,
            args.particle,
            symmetry_pack_to_pose,
        )
    if args.object_symmetry_normalization and args.pose_mesh_frame == "original":
        pose_robot, symmetry_info = pose_normalize.normalize_known_object_symmetry(
            args.mesh_name,
            pose_robot_raw,
            wrist_pose_object=symmetry_wrist_object,
            table_normal_robot=np.asarray(
                [float(x) for x in args.table_normal_xyz.split(",")]
            ),
            max_wrist_facing_angle_rad=np.deg2rad(
                args.symmetry_max_wrist_facing_deg
            ),
        )
    if symmetry_info is not None:
        normalization_applied = symmetry_info["normalize_mode"]
        if args.normalize != "none":
            print(f"[normalize] ignoring generic mode={args.normalize} for "
                  f"{args.mesh_name}; object-specific symmetry axis takes precedence")
        if symmetry_info["symmetry_type"] == "continuous_axis":
            print(f"[normalize] {args.mesh_name} continuous symmetry: "
                  f"axis={np.round(symmetry_info['axis_object'], 6).tolist()}, "
                  f"spin={np.rad2deg(symmetry_info['angle_rad']):+.1f} deg, "
                  f"flipped={'yes' if symmetry_info['axis_flipped'] else 'no'}")
        else:
            print(f"[normalize] {args.mesh_name} cuboid symmetry: "
                  f"variant={symmetry_info['variant_index']}/3")
        if symmetry_wrist_object is not None:
            print("[normalize] pregrasp wrist relative height: "
                  f"{symmetry_info['wrist_relative_height_before_m']:+.3f} -> "
                  f"{symmetry_info['wrist_relative_height_after_m']:+.3f} m "
                  f"(robot-facing angle: "
                  f"{np.rad2deg(symmetry_info['wrist_facing_angle_rad']):.1f} deg, "
                  f"constraints={'ok' if symmetry_info['wrist_constraints_satisfied'] else 'miss'})")
    elif args.normalize != "none":
        pose_world_normalized = pose_normalize.normalize_object_pose(
            args.normalize, pose_world_raw, c2r, np.eye(4)
        )
        pose_robot = r2c @ _to_4x4(pose_world_normalized)
        normalization_applied = args.normalize
        print(f"[normalize] mode={args.normalize}")
    pose_robot = _to_4x4(pose_robot)
    pose_world = _to_4x4(c2r @ pose_robot)

    robot_pose_json_path = os.path.join(abs_save_path, "object_6d_robot.json")
    robot_out = {
        "pose_robot": pose_robot.tolist(),
        "pose_robot_raw": pose_robot_raw.tolist(),
        "R_robot": pose_robot[:3, :3].tolist(),
        "t_robot": pose_robot[:3, 3].tolist(),
        "pose_world": pose_world.tolist(),
        "pose_world_raw": pose_world_raw.tolist(),
        "normalize_mode_requested": args.normalize,
        "normalize_mode": normalization_applied,
        "object_symmetry": symmetry_info,
        "mesh_name": args.mesh_name,
        "mesh_path": mesh_path,
    }
    with open(robot_pose_json_path, "w", encoding="utf-8") as f:
        json.dump(robot_out, f, ensure_ascii=False, indent=2)
    print(f"saved: {robot_pose_json_path}")
    print("pose_robot t:", pose_robot[:3, 3].round(4).tolist())

    # 4) visualize robot (qpos=0) + object together in robot frame
    if args.no_vis:
        return

    robot_urdf_path = os.path.expanduser(args.robot_urdf)
    if not os.path.exists(robot_urdf_path):
        raise FileNotFoundError(f"robot urdf not found: {robot_urdf_path}")
    robot_module = RobotModule(robot_urdf_path)

    # Run curobo planning BEFORE creating the viser server so the long planning
    # step doesn't open a half-empty viewer to the user. The pipeline now
    # returns the full trajectory; we hand it to add_traj after viser starts.
    pipeline_result = None
    if args.pipeline:
        pipeline_result = _run_h2r_pipeline(
            args, args.mesh_name, pose_robot, robot_module, abs_save_path,
        )

    mesh = _load_mesh(mesh_path)
    if pipeline_result is not None:
        rest_qpos = pipeline_result["full_traj"][0].astype(np.float32)
    else:
        rest_qpos = np.zeros(robot_module.get_num_joints(), dtype=np.float32)

    # RPC server writes mask overlay PNGs under <abs_save_path>/debug/. Popup
    # works in either viser mode; the in-scene plane variant needs the local
    # ViserViewer instance, so it's only added in the local branch below.
    debug_dir = os.path.join(abs_save_path, "debug")
    if args.debug_image_mode in ("popup", "both"):
        _popup_debug_images(debug_dir)

    if args.viser_addr:
        from paradex.visualization.visualizer.viser_client import ViserClient
        vc = ViserClient(args.viser_addr)
        try:
            vc.ping()
        except Exception as e:
            raise RuntimeError(
                f"viser_server unreachable at {args.viser_addr}: {e}. "
                f"Start it with `python -m paradex.visualization.visualizer.viser_server "
                f"--robot_urdf {robot_urdf_path}` or rerun with --viser_addr ''."
            ) from e
        vc.clear_objects()
        vc.set_object(args.mesh_name, mesh, pose_robot, opacity=1.0)
        vc.set_qpos(rest_qpos, name="robot")
        if pipeline_result is not None:
            vc.set_traj(pipeline_result["full_traj"], name="h2r_pipeline", robot_name="robot")
            print(f"[pipe] pushed {pipeline_result['full_traj'].shape[0]} frames to "
                  f"viser_server; ranges {pipeline_result['range_str']}.")
        else:
            print(f"[viser] pushed object + start qpos to viser_server at {args.viser_addr}")
        vc.close()
    else:
        vis = ViserViewer()
        vis.add_floor(height=0.0)
        vis.add_robot("robot", robot_urdf_path)
        vis.robot_dict["robot"].update_cfg(rest_qpos)
        vis.add_object(args.mesh_name, mesh, pose_robot, opacity=1.0)
        if args.debug_image_mode in ("viser", "both"):
            _add_debug_images_to_gui(
                vis, debug_dir,
                anchor_xyz=(float(args.debug_anchor_x), float(args.debug_anchor_y),
                            float(args.debug_anchor_z)),
                plane_width=float(args.debug_plane_width),
            )
        if pipeline_result is not None:
            vis.add_traj("h2r_pipeline", robot_traj={"robot": pipeline_result["full_traj"]})
            print(f"[pipe] added {pipeline_result['full_traj'].shape[0]} frames to "
                  f"viewer; Playback ranges {pipeline_result['range_str']}.")
        elif args.scene is not None:
            _replay_pregrasp(args, vis, args.mesh_name, pose_robot)
            return
        print("Viser running. Press Ctrl+C to exit.")
        try:
            while True:
                vis.update()
        except KeyboardInterrupt:
            pass

    # After the viser session closes, optionally execute the planned trajectory
    # on the real xarm + inspire_f1. Always confirm first (unless --yes).
    if pipeline_result is not None:
        if args.yes:
            ans = "y"
        else:
            ans = ""
            while ans not in ("y", "n"):
                try:
                    ans = input(
                        f"\n[exec] Execute {pipeline_result['full_traj'].shape[0]}-frame "
                        f"trajectory on REAL robot @ {args.rate_hz} Hz "
                        f"(record_video={args.record_video})? [y/n] "
                    ).strip().lower()
                except EOFError:
                    ans = "n"
                    break
                if ans not in ("y", "n"):
                    print("[exec] please answer 'y' or 'n'")
        if ans == "y":
            _execute_on_real_robot(args, pipeline_result, robot_module, abs_save_path)
        else:
            print("[exec] aborted by user")


if __name__ == "__main__":
    main()
