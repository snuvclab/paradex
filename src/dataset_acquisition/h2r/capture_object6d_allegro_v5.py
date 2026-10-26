"""
h2r capture + execute for the xarm + allegro_v5 stack.

Mirrors capture_object6d.py (xarm + inspire_f1) but swaps the robot to
xarm_allegro_v5 (6-DOF arm + 16-DOF allegro_v5 right hand). The pipeline
shape, RPC handshake, viser preview, and confirm-then-execute flow are the
same; the things that change are:
    - default URDF / curobo cfg
    - hand command path: AllegroController (radian, 16 joints) instead of
      the inspire_f1 raw-motor publisher
    - no raw-motor conversion table; squeeze is an optional radian offset
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[3]))
PROJECT_ROOT = Path(__file__).parents[3]

import argparse
import datetime
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

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
    pack_frame_to_pose_frame,
    pose_segment,
    resolve_pose_mesh,
    resolve_scene_arg,
    retime_step_periods,
    select_curobo_active_qpos,
    sparse_waypoint_indices,
    staged_home_to_pregrasp,
    trajectory_phase,
    validate_motion_args,
)

DEFAULT_ROBOT_URDF = str(PROJECT_ROOT / "rsc" / "robot" / "xarm_allegro_v5.urdf")
DEFAULT_ALLEGRO_HAND_URDF = str(
    PROJECT_ROOT / "rsc" / "robot" / "allegro_v5" / "allegro_right_A.urdf"
)
DEFAULT_CUROBO_ROBOT_CFG = str(
    PROJECT_ROOT / "rsc" / "curobo" / "content" / "configs" / "robot"
    / "xarm_allegro_v5.yml"
)
DEFAULT_RETRIEVAL_ROOT = os.path.join(
    shared_dir, "grasp_retrieval_meshes_20260915_v1", "allegro_old"
)
RETRIEVAL_ACTIONS = ("ours", "retarget", "retarget_opt")
# Same xarm home pose used by the taeksoo / inspire_f1 pipeline.
HOME_DEG = [-84.9, -25.5, -41.1, 118.8, -1.9, 158.1]

# 16 allegro_v5 hand joints in URDF order, identical to the ROS driver order.
ALLEGRO_HAND_JOINTS = [f"joint_{i}_0" for i in range(16)]
OPEN_HAND_QPOS = np.zeros(len(ALLEGRO_HAND_JOINTS), dtype=np.float64)
ALLEGRO_THUMB_MCP_STRETCH = "joint_12_0"
# The physical right-A hand's vendor HOME settles this geared thumb base near
# 1.58 rad. The earlier 0-rad experiment left it motionless at the mechanical
# endpoint while the driver continuously requested negative torque, so use the
# measured hardware HOME convention for planning, preview, and execution.
OPEN_HAND_QPOS[ALLEGRO_HAND_JOINTS.index(ALLEGRO_THUMB_MCP_STRETCH)] = 1.6
# Position-controlled fingers do not settle exactly at the commanded angle.
# 0.15 rad still represents a clearly open finger while avoiding false aborts
# from the small residual error observed after release (for example 0.114 rad).
ALLEGRO_OPEN_TOL_RAD = 0.15
# The geared thumb base has more settling variation than the other joints.
ALLEGRO_THUMB_OPEN_TOL_RAD = 0.2
# Thumb spread/abduction is joint 13. Keep it out of the optional squeeze ramp
# so squeezing does not change which direction the thumb points.
ALLEGRO_THUMB_ABDUCTION = "joint_13_0"
ALLEGRO_THUMB_JOINTS = tuple(ALLEGRO_HAND_JOINTS[12:16])

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


def _phase_a_hand_qpos(traj_canon: np.ndarray) -> np.ndarray:
    """Keep long fingers open while pre-shaping the thumb for replay frame 0."""
    trajectory = np.asarray(traj_canon, dtype=np.float64)
    expected_width = 6 + len(ALLEGRO_HAND_JOINTS)
    if trajectory.ndim != 2 or len(trajectory) == 0 or trajectory.shape[1] != expected_width:
        raise ValueError(
            f"packed Allegro trajectory must have shape (W,{expected_width})"
        )
    phase_a_hand = OPEN_HAND_QPOS.copy()
    thumb_start = ALLEGRO_HAND_JOINTS.index(ALLEGRO_THUMB_JOINTS[0])
    phase_a_hand[thumb_start:] = trajectory[0, 6 + thumb_start:expected_width]
    if not np.all(np.isfinite(phase_a_hand)):
        raise ValueError("packed replay frame 0 contains non-finite Allegro thumb joints")
    return phase_a_hand


def _validate_phase_a_hand_start(
    first_hand_command: np.ndarray,
    expected_phase_a_hand: np.ndarray,
    *,
    atol: float = 1e-4,
) -> np.ndarray:
    """Validate all 16 first-frame joints, including the replay-shaped thumb."""
    actual = np.asarray(first_hand_command, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected_phase_a_hand, dtype=np.float64).reshape(-1)
    if actual.shape != (16,) or expected.shape != (16,):
        raise ValueError("Phase A Allegro hand targets must contain 16 joints")
    errors = np.abs(actual - expected)
    if not np.all(np.isfinite(errors)) or np.any(errors > float(atol)):
        worst = int(np.nanargmax(errors)) if np.any(np.isfinite(errors)) else 0
        raise ValueError(
            "pipeline first-frame Allegro posture does not match the planned "
            f"thumb-pre-shaped Phase A hand: {ALLEGRO_HAND_JOINTS[worst]} "
            f"actual={actual[worst]:.4f}, expected={expected[worst]:.4f}"
        )
    return expected.copy()


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
    """Resolve either a mesh root, an object directory, or an explicit OBJ file."""
    return resolve_pose_mesh(mesh_root_dir, mesh_name, frame)


def _resolve_retrieval_scene(
    retrieval_root: str, mesh_name: str, scene: Optional[int]
) -> int:
    object_root = Path(retrieval_root).expanduser() / mesh_name
    if scene is not None:
        scene_root = object_root / f"scene_{int(scene)}"
        if not scene_root.is_dir():
            raise FileNotFoundError(f"retrieval scene not found: {scene_root}")
        return int(scene)
    candidates = sorted(path for path in object_root.glob("scene_*") if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            f"expected one retrieval scene under {object_root}, found "
            f"{len(candidates)}; pass --scene"
        )
    suffix = candidates[0].name[len("scene_"):]
    try:
        return int(suffix)
    except ValueError as exc:
        raise ValueError(f"invalid retrieval scene directory: {candidates[0]}") from exc


def _load_retrieval_replay(
    args, mesh_name: str, robot_module: RobotModule
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load one evaluated action as object-from-wrist poses plus named hand qpos."""

    action = str(args.retrieval_action)
    path = (
        Path(args.retrieval_root).expanduser()
        / mesh_name
        / f"scene_{int(args.scene)}"
        / "evaluated_actions"
        / action
        / "approach_grasp_trajectory_object.npz"
    )
    if not path.is_file():
        raise FileNotFoundError(f"retrieval trajectory not found: {path}")
    metadata_path = path.with_name("grasp_action_object.json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"retrieval frame metadata required for safe replay: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("coordinate_frame") != "object mesh native frame":
        raise ValueError(
            f"{metadata_path}: expected coordinate_frame='object mesh native frame'"
        )
    if metadata.get("wrist_frame") != "palm_link":
        raise ValueError(f"{metadata_path}: expected wrist_frame='palm_link'")

    with np.load(path, allow_pickle=False) as data:
        required = {"T_object_from_wrist", "qpos", "joint_names"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")
        object_from_palm = np.asarray(data["T_object_from_wrist"], dtype=np.float64)
        hand_qpos = np.asarray(data["qpos"], dtype=np.float64)
        joint_names = [str(value) for value in np.asarray(data["joint_names"]).reshape(-1)]

    count = len(object_from_palm)
    if count == 0 or object_from_palm.shape != (count, 4, 4):
        raise ValueError(f"{path}: T_object_from_wrist must have shape (W,4,4)")
    if hand_qpos.shape != (count, len(ALLEGRO_HAND_JOINTS)):
        raise ValueError(f"{path}: qpos must have shape ({count},16)")
    if not np.all(np.isfinite(object_from_palm)) or not np.all(np.isfinite(hand_qpos)):
        raise ValueError(f"{path}: trajectory contains non-finite values")
    if len(joint_names) != 16 or set(joint_names) != set(ALLEGRO_HAND_JOINTS):
        raise ValueError(f"{path}: joint_names must contain each Allegro V5 joint once")
    for index, transform in enumerate(object_from_palm):
        if (
            not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
        ):
            raise ValueError(f"{path}: frame {index} is not a rigid transform")
    hand_qpos = hand_qpos[:, [joint_names.index(name) for name in ALLEGRO_HAND_JOINTS]]

    # Retrieval metadata calls the stored frame palm_link. Convert it through
    # the combined URDF instead of assuming that palm_link and the planner's
    # wrist end-effector are coincident.
    zero_qpos = np.zeros(robot_module.get_num_joints(), dtype=np.float64)
    fk = robot_module.forward_kinematics(zero_qpos)
    wrist_from_palm = np.linalg.inv(fk["wrist"]) @ fk["palm_link"]
    object_from_wrist = object_from_palm @ np.linalg.inv(wrist_from_palm)
    print(
        f"[pipe] retrieval action={action!r}: loaded {count} palm frames from {path}"
    )
    return object_from_wrist.astype(np.float32), hand_qpos.astype(np.float32), str(path)


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


def _read_current_xarm_qpos(timeout_s: float = 5.0) -> np.ndarray:
    """Read the six-joint planning start without issuing a motion command."""

    from paradex.io.robot_controller import get_arm

    arm = get_arm("xarm", servo_api="cartesian_aa")
    try:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            qpos = np.asarray(arm.get_data().get("qpos"), dtype=np.float64)
            if qpos.shape == (6,) and np.all(np.isfinite(qpos)):
                return qpos.copy()
            time.sleep(0.05)
        raise RuntimeError("timed out waiting for current xArm joint feedback")
    finally:
        arm.end(set_break=False)


def _popup_debug_images(debug_dir: str) -> int:
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


def _unwrap_continuous_joints(qpos: np.ndarray, joint_names, target_names=("joint1", "joint4", "joint6")):
    """For joints with ±2π range, IK may pick equivalent solutions that differ
    by 2π between consecutive waypoints (visually = a 360° flip). Snap each
    frame's chosen-joint value to the 2π-equivalent angle closest to the
    previous frame's, eliminating the wrap-around without changing the EE pose.
    """
    out = qpos.copy()
    idx = [joint_names.index(n) for n in target_names if n in joint_names]
    if not idx:
        return out
    for j in idx:
        for t in range(1, out.shape[0]):
            d = out[t, j] - out[t - 1, j]
            k = int(round(d / (2.0 * np.pi)))
            if k != 0:
                out[t, j] -= 2.0 * np.pi * k
    return out


def _solve_ik_chain_curobo(robot_cfg: Dict[str, Any], target_poses: np.ndarray,
                           seed_arm_qpos: np.ndarray, device: str = "cuda:0",
                           num_seeds: int = 32, null_space_weight: float = 5.0,
                           self_collision: bool = True,
                           position_threshold: float = 0.01,
                           rotation_threshold: float = 0.1,
                           sample_stride: int = 10,
                           world_cfg_dict: Optional[Dict[str, Any]] = None):
    """Per-waypoint IK with strong continuity constraint.

    Continuity is encouraged in two ways:
      1. cspace.null_space_weight is forced to ``null_space_weight`` so curobo
         actively penalises distance from retract_config (= previous solution).
      2. Every new waypoint uses the previous solution as both its explicit
         seed and retract configuration.
    Waypoint 0 is the already-planned Phase A endpoint, so it is copied exactly
    instead of being independently re-solved under potentially different IK
    sampling. A failed later waypoint is temporarily frozen at the previous
    solution, and the success count lets the caller reject the incomplete chain.
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
    n_cs = len(cspace["joint_names"])
    cspace["null_space_weight"] = [float(null_space_weight)] * n_cs

    dev = torch.device(device if ("cuda" in str(device) and torch.cuda.is_available()) else "cpu")
    tensor_args = TensorDeviceType(device=dev)
    world_cfg = None
    if world_cfg_dict is not None:
        from curobo.geom.types import WorldConfig
        world_cfg = WorldConfig.from_dict(world_cfg_dict)
    ik_cfg = IKSolverConfig.load_from_robot_config(
        robot_cfg, world_cfg, tensor_args=tensor_args,
        num_seeds=num_seeds,
        position_threshold=float(position_threshold),
        rotation_threshold=float(rotation_threshold),
        self_collision_check=bool(self_collision),
        self_collision_opt=bool(self_collision),
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

    # Phase A planned this exact target. Reusing its terminal joint state makes
    # the A -> B boundary identical in joint space and avoids asking a second
    # stochastic IK solver to rediscover wp0.
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
        # Adjacent Cartesian samples are very close, so the previous solution
        # is normally sufficient by itself. Avoid generating num_seeds-1
        # random seeds for every sample; retain the full search as a fallback.
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


def _solve_ik_chain_xarm_sdk(
    command_poses: np.ndarray,
    seed_arm_qpos: np.ndarray,
    *,
    xarm_ip: str,
    phase: str,
    joint_bounds: Optional[np.ndarray] = None,
    arm_factory=None,
) -> Tuple[np.ndarray, int]:
    """Query xArm's own IK without issuing motion commands.

    The input is the link6 pose sequence that Cartesian execution sends to
    xArm.  The SDK has no explicit seed argument, so continuous joints are
    unwrapped against HOME/the previous returned solution before the standard
    limit and speed checks consume the result.
    """

    from scipy.spatial.transform import Rotation

    poses = np.asarray(command_poses, dtype=np.float64)
    seed = np.asarray(seed_arm_qpos, dtype=np.float64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("xArm SDK IK expects command poses with shape (N,4,4)")
    if seed.shape != (6,) or not np.all(np.isfinite(seed)):
        raise ValueError("xArm SDK IK seed must be six finite joint angles")
    if joint_bounds is None:
        bounds = np.tile(np.array([-2.0 * np.pi, 2.0 * np.pi]), (6, 1))
    else:
        bounds = np.asarray(joint_bounds, dtype=np.float64)
        if bounds.shape != (6, 2) or not np.all(np.isfinite(bounds)):
            raise ValueError("xArm SDK joint bounds must have shape (6,2)")
    if arm_factory is None:
        from xarm.wrapper import XArmAPI

        arm_factory = lambda ip: XArmAPI(ip, is_radian=True)

    arm = arm_factory(str(xarm_ip))
    solutions = []
    success_count = 0
    last = seed.copy()
    try:
        if hasattr(arm, "connected") and not arm.connected:
            raise RuntimeError(f"xArm SDK could not connect to {xarm_ip}")
        for index, target in enumerate(poses):
            rpy = Rotation.from_matrix(target[:3, :3]).as_euler("xyz")
            sdk_pose = np.concatenate((target[:3, 3] * 1000.0, rpy))
            code, angles = arm.get_inverse_kinematics(
                sdk_pose.tolist(),
                input_is_radian=True,
                return_is_radian=True,
            )
            candidate = np.asarray(angles, dtype=np.float64).reshape(-1)
            if int(code) != 0 or candidate.size < 6 or not np.all(np.isfinite(candidate[:6])):
                print(
                    f"[xarm sdk IK] {phase} waypoint {index}/{len(poses)} failed: "
                    f"code={code}"
                )
                solutions.append(last.copy())
                solutions.extend(last.copy() for _ in range(index + 1, len(poses)))
                break
            candidate = candidate[:6]
            # joint1/4/6 admit 2π-equivalent representations, but unlike the
            # generic visual unwrap we must never walk those equivalents past
            # the physical xArm bounds. Choose the nearest *valid* equivalent.
            for joint in (0, 3, 5):
                equivalents = candidate[joint] + 2.0 * np.pi * np.arange(-3, 4)
                valid = equivalents[
                    (equivalents >= bounds[joint, 0])
                    & (equivalents <= bounds[joint, 1])
                ]
                if valid.size:
                    candidate[joint] = valid[np.argmin(np.abs(valid - last[joint]))]
            solutions.append(candidate)
            last = candidate
            success_count += 1
    finally:
        disconnect = getattr(arm, "disconnect", None)
        if callable(disconnect):
            disconnect()

    result = np.asarray(solutions, dtype=np.float32)
    print(f"[xarm sdk IK] {phase}: {success_count}/{len(poses)} succeeded")
    return result, success_count


def _extract_hand_command(full_qpos: np.ndarray, full_joint_names) -> np.ndarray:
    """URDF rad -> 16 allegro_v5 hand joints in AllegroController wire order
    (joint_0_0 .. joint_15_0). Joints not present in the URDF qpos fall back
    to 0 with a one-time warning."""
    idx = {n: i for i, n in enumerate(full_joint_names)}
    out = np.zeros(16, dtype=np.float64)
    missing = []
    for i, name in enumerate(ALLEGRO_HAND_JOINTS):
        if name in idx:
            out[i] = float(full_qpos[idx[name]])
        else:
            missing.append(name)
    if missing and not getattr(_extract_hand_command, "_warned", False):
        print(f"[exec] WARNING: missing allegro joints in qpos, defaulting to 0: {missing}")
        _extract_hand_command._warned = True
    return out


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


def _wait_hand_connected(hand, timeout: float = 5.0) -> bool:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if hand.connection_event.is_set():
            return True
        time.sleep(0.05)
    return False


def _wait_hand_at_target(
    hand,
    target_rad: np.ndarray,
    *,
    tol_rad,
    timeout: float,
) -> Tuple[bool, np.ndarray]:
    """Wait for measured Allegro joints, not merely command publication."""

    deadline = time.monotonic() + float(timeout)
    target = np.asarray(target_rad, dtype=np.float64)
    tolerance = np.broadcast_to(
        np.asarray(tol_rad, dtype=np.float64), target.shape
    )
    measured = np.full_like(target, np.nan)
    while time.monotonic() < deadline:
        measured = np.asarray(hand.get_data().get("qpos"), dtype=np.float64)
        if (
            measured.shape == target.shape
            and np.all(np.isfinite(measured))
            and np.all(np.abs(measured - target) <= tolerance)
        ):
            return True, measured
        time.sleep(0.05)
    return False, measured


def _open_hand_feedback_tolerances(base_tol_rad: float) -> np.ndarray:
    """Per-joint open feedback tolerances, including thumb hysteresis."""

    tolerances = np.full(len(ALLEGRO_HAND_JOINTS), float(base_tol_rad))
    thumb_idx = ALLEGRO_HAND_JOINTS.index(ALLEGRO_THUMB_MCP_STRETCH)
    tolerances[thumb_idx] = max(tolerances[thumb_idx], ALLEGRO_THUMB_OPEN_TOL_RAD)
    return tolerances


def _move_xarm_to_home_safely(
    xarm,
    target_rad: np.ndarray,
    *,
    max_joint_speed_deg_s: float,
    stream_hz: float,
) -> None:
    """Move from measured qpos to HOME with a bounded joint-space velocity."""
    current = xarm.get_data().get("qpos")
    current = None if current is None else np.asarray(current, dtype=np.float64)
    target_rad = np.asarray(target_rad, dtype=np.float64)
    if current is None or current.shape != (6,) or not np.all(np.isfinite(current)):
        raise RuntimeError("xArm joint feedback unavailable; refusing HOME preposition")
    speed_rad_s = np.deg2rad(float(max_joint_speed_deg_s))
    if not np.isfinite(speed_rad_s) or speed_rad_s <= 0.0:
        raise ValueError("home_max_joint_speed_deg_s must be positive")
    if not np.isfinite(stream_hz) or stream_hz <= 0.0:
        raise ValueError("home_preposition_hz must be positive")
    max_delta = float(np.max(np.abs(target_rad - current)))
    if max_delta <= 1e-6:
        return
    duration = max_delta / speed_rad_s
    n_frames = max(2, int(np.ceil(duration * stream_hz)) + 1)
    positions = np.linspace(current, target_rad, n_frames, dtype=np.float64)
    times = np.linspace(0.0, duration, n_frames, dtype=np.float64)
    print(f"[exec] safe HOME preposition: {duration:.2f}s, {n_frames} frames, "
          f"<= {max_joint_speed_deg_s:.1f} deg/s")
    xarm.move_joint_trajectory(positions, times)


def _densify_timed_cartesian_commands(
    poses: np.ndarray,
    hand_commands: np.ndarray,
    periods: np.ndarray,
    *,
    command_rate_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate synchronized SE(3) and hand commands for Cartesian servoing."""
    from scipy.spatial.transform import Rotation, Slerp

    pose_values = np.asarray(poses, dtype=np.float64)
    hand_values = np.asarray(hand_commands, dtype=np.float64)
    dt = np.asarray(periods, dtype=np.float64).reshape(-1)
    rate = float(command_rate_hz)
    if (
        pose_values.ndim != 3
        or pose_values.shape[1:] != (4, 4)
        or hand_values.ndim != 2
        or len(pose_values) != len(hand_values)
        or len(pose_values) != len(dt)
        or len(pose_values) == 0
    ):
        raise ValueError("Cartesian poses, hand commands, and periods must have matching lengths")
    if (
        not np.all(np.isfinite(pose_values))
        or not np.all(np.isfinite(hand_values))
        or not np.all(np.isfinite(dt))
        or np.any(dt <= 0.0)
        or not np.isfinite(rate)
        or rate <= 0.0
    ):
        raise ValueError("Cartesian commands, periods, and command rate must be finite and positive")

    dense_poses = []
    dense_hands = []
    dense_periods = []
    source_indices = []
    max_period = 1.0 / rate
    for index in range(len(pose_values) - 1):
        subdivisions = max(1, int(np.ceil(dt[index] / max_period)))
        alphas = np.arange(subdivisions, dtype=np.float64) / subdivisions
        rotations = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(
                np.stack((pose_values[index, :3, :3], pose_values[index + 1, :3, :3]))
            ),
        )(alphas).as_matrix()
        translations = (
            (1.0 - alphas[:, None]) * pose_values[index, :3, 3]
            + alphas[:, None] * pose_values[index + 1, :3, 3]
        )
        hands = (
            (1.0 - alphas[:, None]) * hand_values[index]
            + alphas[:, None] * hand_values[index + 1]
        )
        for step in range(subdivisions):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotations[step]
            pose[:3, 3] = translations[step]
            dense_poses.append(pose)
            dense_hands.append(hands[step])
            dense_periods.append(float(dt[index]) / subdivisions)
            source_indices.append(index)
    dense_poses.append(pose_values[-1].copy())
    dense_hands.append(hand_values[-1].copy())
    dense_periods.append(float(dt[-1]))
    source_indices.append(len(pose_values) - 1)
    return (
        np.asarray(dense_poses),
        np.asarray(dense_hands),
        np.asarray(dense_periods),
        np.asarray(source_indices, dtype=np.int64),
    )


def _execute_on_real_robot(args, pipeline_result: Dict[str, Any], robot_module: RobotModule,
                          abs_save_path: str) -> None:
    from paradex.io.robot_controller import get_arm, get_hand
    from paradex.calibration.utils import save_current_C2R

    full_traj = np.asarray(pipeline_result["full_traj"], dtype=np.float64)
    Tn = int(full_traj.shape[0])
    full_joint_names = list(robot_module.get_joint_names())

    arm_idx = [full_joint_names.index(f"joint{i}") for i in range(1, 7)]
    arm_traj = full_traj[:, arm_idx]  # (T,6) rad
    arm_pose_traj = np.asarray(pipeline_result["arm_pose_traj"], dtype=np.float64)

    # hand: 16 allegro joints in wire order, radian
    hand_cmds = np.zeros((Tn, 16), dtype=np.float64)
    for t in range(Tn):
        hand_cmds[t] = _extract_hand_command(full_traj[t], full_joint_names)
    expected_initial_hand = np.asarray(
        pipeline_result["phase_a_hand_qpos"], dtype=np.float64
    )
    initial_hand = _validate_phase_a_hand_start(
        hand_cmds[0], expected_initial_hand
    )
    long_finger_count = ALLEGRO_HAND_JOINTS.index(ALLEGRO_THUMB_JOINTS[0])
    if not np.allclose(
        initial_hand[:long_finger_count],
        OPEN_HAND_QPOS[:long_finger_count],
        atol=1e-4,
    ):
        raise ValueError(
            "pipeline must start with the long Allegro fingers open; replan before executing"
        )

    # Optional grasp squeeze: ramp every hand joint except the thumb abduction
    # by --grasp_squeeze radians AFTER the planned grasp finishes (approach +
    # hand_interp + replay), then hold the squeezed pose for grasp_hold + lift.
    n_a = int(pipeline_result.get("n_approach", 0))
    n_b = int(pipeline_result.get("n_hand_interp", 0))
    n_r = int(pipeline_result.get("n_replay", 0))
    n_release = int(pipeline_result.get("n_release", 0))
    # Release frames sit at the very end of full_traj. They must NOT play
    # before the post-roll video hold, so we stop the main streaming at
    # release_start and stream the release frames separately after the hold.
    release_start = Tn - n_release if n_release > 0 else Tn
    squeeze_start = n_a + n_b + n_r
    squeeze_amt = float(args.grasp_squeeze)
    squeeze_steps = int(args.squeeze_steps)
    do_squeeze = squeeze_amt != 0.0 and 0 < squeeze_start < Tn
    thumb_abd_idx = ALLEGRO_HAND_JOINTS.index(ALLEGRO_THUMB_ABDUCTION)
    squeeze_mask = np.ones(16, dtype=bool)
    squeeze_mask[thumb_abd_idx] = False
    if do_squeeze:
        hand_cmds[squeeze_start:release_start, squeeze_mask] += squeeze_amt
        print(f"[exec] grasp squeeze +{squeeze_amt:.3f} rad over {squeeze_steps} "
              f"frames after frame {squeeze_start} ({ALLEGRO_THUMB_ABDUCTION} excluded)")

    # Per-frame streaming period; replay segment can be slower.
    periods = np.full(Tn, 1.0 / float(args.rate_hz), dtype=np.float64)
    replay_lo, replay_hi = n_a + n_b, n_a + n_b + n_r
    if replay_hi > replay_lo:
        periods[replay_lo:replay_hi] = 1.0 / float(args.replay_rate_hz)
    nominal_duration = float(np.sum(periods))
    arm_execution_mode = str(args.arm_execution_mode)
    sdk_ik_preflight = bool(pipeline_result.get("sdk_ik_preflight", False))
    periods, retimed_steps, max_stretch = retime_step_periods(
        arm_traj if arm_execution_mode == "joint" or sdk_ik_preflight else None,
        arm_pose_traj,
        periods,
        max_joint_speed_deg_s=(
            float(args.max_planned_joint_speed_deg_s)
            if arm_execution_mode == "joint" or sdk_ik_preflight else None
        ),
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
    if arm_execution_mode == "joint" or sdk_ik_preflight:
        assert_bounded_joint_steps(
            arm_traj,
            periods,
            max_joint_speed_deg_s=float(args.max_planned_joint_speed_deg_s),
        )
    print(f"[exec] replay segment frames {replay_lo}..{replay_hi - 1} "
          f"@ {args.replay_rate_hz} Hz (rest @ {args.rate_hz} Hz)")

    home_rad = np.deg2rad(np.asarray(args.home_deg, dtype=np.float64))
    planned_start_arm = np.asarray(
        pipeline_result.get("planning_start_arm_qpos", home_rad), dtype=np.float64
    )
    arm0_start_equivalent = planned_start_arm + np.arctan2(
        np.sin(arm_traj[0] - planned_start_arm),
        np.cos(arm_traj[0] - planned_start_arm),
    )
    if not np.allclose(arm0_start_equivalent, planned_start_arm, atol=1e-3):
        print(f"[exec] WARNING: traj[0] arm != planned START; traj[0]={arm_traj[0]}")
    print(f"[exec] T={Tn}, rate={args.rate_hz} Hz (~{Tn / args.rate_hz:.1f}s)")
    print(f"[exec] arm[0] deg: {np.rad2deg(arm0_start_equivalent).round(2).tolist()}")
    print(f"[exec] hand[0]  rad: {np.round(hand_cmds[0], 3).tolist()}")
    print(f"[exec] hand[-1] rad: {np.round(hand_cmds[-1], 3).tolist()}")

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

    xarm = None
    hand = None
    try:
        servo_api = "angle_j" if arm_execution_mode == "joint" else "cartesian_aa"
        print(f"[exec] init xarm (servo_api={servo_api}, "
              f"command_rate={args.arm_servo_rate_hz:.1f} Hz)...")
        xarm = get_arm(
            "xarm",
            servo_api=servo_api,
            fps=float(args.arm_servo_rate_hz),
        )
        print(f"[exec] init allegro_v5 (hand_side={args.hand_side}, "
              f"namespace={args.allegro_namespace!r})...")
        hand = get_hand(
            "allegro_v5",
            hand_side=args.hand_side,
            namespace=args.allegro_namespace,
        )
        if not _wait_hand_connected(hand, timeout=5.0):
            raise RuntimeError(
                "Allegro state was not received; refusing to execute without hand feedback "
                f"on namespace {args.allegro_namespace!r}"
            )

        print("[exec] resetting Allegro driver to vendor HOME/place mode...")
        home_feedback = hand.run_driver_home(settle_seconds=0.5)
        print(
            "[exec] Allegro HOME/place reset complete: "
            f"joint_12_0={home_feedback[12]:.3f} rad"
        )

        # Phase 0: reproduce the planner start state on hardware before any
        # trajectory sample is streamed. Both feedback checks are mandatory:
        # starting the arm approach with a closed or stale hand is unsafe.
        print(f"[exec] commanding planned START deg="
              f"{list(np.rad2deg(planned_start_arm).round(2))} "
              f"+ thumb-pre-shaped hand={np.round(initial_hand, 3).tolist()}")
        _move_xarm_to_home_safely(
            xarm,
            planned_start_arm,
            max_joint_speed_deg_s=float(args.home_max_joint_speed_deg_s),
            stream_hz=float(args.home_preposition_hz),
        )
        hand_feedback = np.asarray(hand.get_data()["qpos"], dtype=np.float64)
        hand_speed_rad_s = float(args.home_hand_max_speed_rad_s)
        if not np.isfinite(hand_speed_rad_s) or hand_speed_rad_s <= 0.0:
            raise ValueError("home_hand_max_speed_rad_s must be positive")
        hand.set_command_slew_rate(hand_speed_rad_s, initial_action=hand_feedback)
        hand.move(initial_hand)
        hand_command_timeout = max(
            2.0,
            float(np.max(np.abs(initial_hand - hand_feedback)))
            / hand_speed_rad_s
            + 1.0,
        )
        hand_command_ready = hand.wait_for_published_action(
            initial_hand,
            timeout_seconds=hand_command_timeout,
            atol=1e-3,
        )
        hand_tolerances = _open_hand_feedback_tolerances(args.home_hand_tol_rad)
        if args.enable_allegro_start_posture_check:
            hand_feedback_ready, hand_measured = _wait_hand_at_target(
                hand,
                initial_hand,
                tol_rad=hand_tolerances,
                timeout=float(args.home_timeout),
            )
        else:
            hand_feedback_ready = True
            hand_measured = np.asarray(hand.get_data().get("qpos"), dtype=np.float64)
            print(
                "[exec] legacy Allegro start-posture feedback check disabled; "
                f"measured={np.round(hand_measured, 3).tolist()}"
            )
        arm_ready = _wait_xarm_at_target(xarm, planned_start_arm,
                                         np.deg2rad(float(args.home_tol_deg)),
                                         float(args.home_timeout))
        if not arm_ready or not hand_command_ready or not hand_feedback_ready:
            missing = []
            if not arm_ready:
                missing.append("xarm planned START")
            if not hand_command_ready:
                missing.append("Allegro open-hand command publication")
            if not hand_feedback_ready:
                if hand_measured.shape == initial_hand.shape and np.all(np.isfinite(hand_measured)):
                    errors = np.abs(hand_measured - initial_hand)
                    worst = int(np.argmax(errors / hand_tolerances))
                    hand_data = hand.get_data()
                    efforts = np.asarray(hand_data.get("effort"), dtype=np.float64)
                    effort_text = (
                        f", effort={efforts[worst]:.3f}"
                        if efforts.shape == initial_hand.shape
                        and np.isfinite(efforts[worst])
                        else ""
                    )
                    missing.append(
                        "Allegro open-hand feedback "
                        f"(joint_{worst}_0={hand_measured[worst]:.3f} rad, "
                        f"target={initial_hand[worst]:.3f}, "
                        f"error={errors[worst]:.3f}, "
                        f"tol={hand_tolerances[worst]:.3f}{effort_text})"
                    )
                else:
                    missing.append("Allegro open-hand feedback")
            raise RuntimeError(
                "pre-execution initialization timed out: "
                + ", ".join(missing)
            )
        print("[exec] initialization complete: xArm planned START + Allegro thumb pre-shape command")
        hand.set_command_slew_rate(None)
        if xarm.is_error() or hand.is_error():
            raise RuntimeError("controller in error state before streaming")
        watchdog = None
        if args.enable_live_joint_watchdog:
            watchdog = XArmJointWatchdog(
                robot_module,
                max_speed_deg_s=float(args.max_measured_joint_speed_deg_s),
                limit_margin_rad=float(args.live_joint_limit_margin_rad),
            )
            print("[exec] legacy live joint watchdog ENABLED")
            watchdog.check(xarm, context="initialization")
        else:
            print("[exec] legacy live joint watchdog disabled")

        def _check_live_joint_watchdog(context: str) -> None:
            if watchdog is not None:
                watchdog.check(xarm, context=context)

        def _stream_range(start: int, stop: int) -> None:
            if stop <= start:
                return
            if arm_execution_mode == "joint":
                commands = np.concatenate(
                    (arm_traj[start:stop], hand_cmds[start:stop]), axis=1
                )
                dense, dense_periods, source_frames = densify_timed_commands(
                    commands,
                    periods[start:stop],
                    command_rate_hz=float(args.arm_servo_rate_hz),
                )
                dense_arms = dense[:, :6]
                dense_hands = dense[:, 6:]
            else:
                dense_arms, dense_hands, dense_periods, source_frames = (
                    _densify_timed_cartesian_commands(
                        arm_pose_traj[start:stop],
                        hand_cmds[start:stop],
                        periods[start:stop],
                        command_rate_hz=float(args.arm_servo_rate_hz),
                    )
                )
            source_frames += start
            t0 = time.perf_counter()
            clock = 0.0
            for arm_command, hand_command, period, frame in zip(
                dense_arms, dense_hands, dense_periods, source_frames
            ):
                frame = int(frame)
                _check_live_joint_watchdog(
                    f"frame={frame}, phase={trajectory_phase(frame, pipeline_result)}"
                )
                xarm.move(arm_command, is_servo=True)
                hand.move(hand_command)
                if xarm.is_error() or hand.is_error():
                    raise RuntimeError(f"controller error at frame {frame}")
                clock += float(period)
                sleep_for = t0 + clock - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

        # Phase 1: grasp sequence streaming. Stop before the release frames so
        # the post-roll video hold captures the lifted/held pose.
        grasp_end = squeeze_start if do_squeeze else release_start
        print(f"[exec] streaming grasp sequence ({arm_execution_mode} arm): "
              f"frames 0..{grasp_end - 1} ...")
        _stream_range(0, grasp_end)

        # Phase 1.5: explicit squeeze ramp -- arm frozen, hand-only.
        if do_squeeze:
            hand_from = hand_cmds[squeeze_start - 1].copy()
            hand_to = hand_cmds[squeeze_start].copy()
            arm_hold = (
                arm_traj[squeeze_start - 1]
                if arm_execution_mode == "joint"
                else arm_pose_traj[squeeze_start - 1]
            )
            print(f"[exec] squeezing hand over {squeeze_steps} frames ...")
            for s in range(1, squeeze_steps + 1):
                _check_live_joint_watchdog(f"squeeze_step={s}")
                a = float(s) / float(squeeze_steps)
                cmd = (1.0 - a) * hand_from + a * hand_to
                xarm.move(arm_hold, is_servo=True)
                hand.move(cmd)
                if xarm.is_error() or hand.is_error():
                    raise RuntimeError(f"controller error during squeeze step {s}")
                time.sleep(1.0 / float(args.rate_hz))

            # Phase 2: hold + lift with squeezed hand pose (stops before release).
            print(f"[exec] streaming hold + lift: frames {squeeze_start}..{release_start - 1} ...")
            _stream_range(squeeze_start, release_start)
        print("[exec] streaming done (pre-release)")
        post_lift_hold_seconds = float(args.post_lift_hold_seconds)
        if post_lift_hold_seconds > 0:
            hold_pose = (
                arm_traj[release_start - 1]
                if arm_execution_mode == "joint"
                else arm_pose_traj[release_start - 1]
            )
            hold_hand = hand_cmds[release_start - 1]
            print(f"[exec] holding lifted grasp for {post_lift_hold_seconds:.1f}s ...")
            hold_deadline = time.perf_counter() + post_lift_hold_seconds
            while time.perf_counter() < hold_deadline:
                _check_live_joint_watchdog("post_lift_hold")
                xarm.move(hold_pose, is_servo=True)
                hand.move(hold_hand)
                if xarm.is_error() or hand.is_error():
                    raise RuntimeError("controller error during post-lift grasp hold")
                time.sleep(1.0 / float(args.rate_hz))

        # Phase D: release index/middle/ring. Runs AFTER the post-roll hold
        # so the video shows the object held first, then dropped.
        if n_release > 0:
            print(f"[exec] releasing fingers: frames {release_start}..{Tn - 1} ...")
            _stream_range(release_start, Tn)
            print("[exec] release done")

        print("[exec] returning xArm to HOME after release ...")
        _move_xarm_to_home_safely(
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
        try:
            if hand is not None:
                hand.set_command_slew_rate(None)
                hand.end()
        except Exception as e: print(f"[exec] hand.end error: {e}")
        try:
            if xarm is not None:
                xarm.end(set_break=False)
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
                     robot_module: RobotModule, abs_save_path: str,
                     ignore_collisions: bool = False) -> Optional[Dict[str, Any]]:
    """Staged HOME-height planar+rotate and vertical descent to pregrasp,
    followed by per-waypoint IK preview of the canonical wrist trajectory
    while finger qpos is replayed from result.npz by JOINT NAME. Result.npz must
    therefore have revolute_joint_names matching allegro_v5 joint names for the
    fingers to move; otherwise the hand is held at zeros."""
    preview_backend = str(args.path_preview_backend)
    sdk_preflight = preview_backend == "xarm_sdk"
    curobo_device = str(args.curobo_device)
    if not sdk_preflight and curobo_device.startswith("cuda:"):
        gpu_id = curobo_device.split(":", 1)[1]
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_id)
        print(f"[pipe] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
              f"(curobo will use cuda:0 internally)")
        curobo_device = "cuda:0"
    if sdk_preflight:
        print("[pipe] path preview backend=xarm_sdk (CuRobo disabled; "
              "SDK IK does not check collisions)")

    from scipy.spatial.transform import Rotation

    retrieval_action = getattr(args, "retrieval_action", None)
    retrieval_source = retrieval_action is not None
    if retrieval_source:
        wrist_T_object, replay_hand_qpos, source_path = _load_retrieval_replay(
            args, mesh_name, robot_module
        )
        W = int(len(wrist_T_object))
        traj_canon = np.zeros((W, 6 + len(ALLEGRO_HAND_JOINTS)), dtype=np.float32)
        traj_canon[:, 6:] = replay_hand_qpos
        wrist_T_robot = np.asarray(pose_robot, dtype=np.float32)[None] @ wrist_T_object
        print(
            f"[pipe] {mesh_name} scene={args.scene} retrieval={retrieval_action} "
            f"W={W} (object-relative wrist + 16 named Allegro joints)"
        )
    else:
        res_path = os.path.join(
            args.result_root, mesh_name, str(args.scene), "result_allegro.npz"
        )
        if not os.path.exists(res_path):
            raise FileNotFoundError(f"no packed result: {res_path}")
        source_path = res_path
        d = np.load(res_path, allow_pickle=True)
        pack_to_pose = pack_frame_to_pose_frame(d, args.pose_mesh_frame)
        pack_frame = str(d["wrist_frame_id"]) if "wrist_frame_id" in d.files else "canonical"
        print(f"[pipe] wrist frame: pack='{pack_frame}', pose='{args.pose_mesh_frame}' -> "
              f"{'identity' if np.allclose(pack_to_pose, np.eye(4)) else 'viser_align bridge'}")
        # result_allegro.npz layout:
        #   <src>_traj_q_all: particles x waypoints x (wrist[6] + 16 fingers)
        #   <src>_traj_q: selected best-particle trajectory.
        key_all = f"{args.src}_traj_q_all"
        key_best = f"{args.src}_traj_q"
        if args.particle is None:
            if key_best not in d.files:
                raise KeyError(f"{key_best} missing in {res_path}")
            traj_canon = np.asarray(d[key_best])
            particle_used = (int(d[f"{args.src}_best_particle"])
                             if f"{args.src}_best_particle" in d.files else -1)
        else:
            if key_all not in d.files:
                raise KeyError(f"{key_all} missing in {res_path}")
            traj_canon = np.asarray(d[key_all])[int(args.particle)]
            particle_used = int(args.particle)
        W = int(traj_canon.shape[0])
        n_finger_traj = traj_canon.shape[1] - 6
        print(f"[pipe] {mesh_name} scene={args.scene} src={args.src} "
              f"particle={particle_used}  W={W} (wrist + {n_finger_traj} finger joints)")
        if n_finger_traj != len(ALLEGRO_HAND_JOINTS):
            raise ValueError(f"result_allegro.npz finger DOF ({n_finger_traj}) != "
                             f"expected 16 allegro joints")
        wrist_T_robot = np.zeros((W, 4, 4), dtype=np.float32)
        for i, w in enumerate(traj_canon):
            wrist_object = np.eye(4, dtype=np.float32)
            wrist_object[:3, :3] = Rotation.from_euler("XYZ", w[3:6]).as_matrix()
            wrist_object[:3, 3] = w[:3]
            wrist_T_robot[i] = (pose_robot @ pack_to_pose @ wrist_object).astype(np.float32)

    jnames_traj = list(ALLEGRO_HAND_JOINTS)

    full_joint_names = list(robot_module.get_joint_names())
    n_dof = int(robot_module.get_num_joints())
    name_to_full_idx = {n: i for i, n in enumerate(full_joint_names)}
    name_to_finger_idx = {n: i for i, n in enumerate(jnames_traj)}
    # Phase A starts with the long fingers open and the thumb already matching
    # replay frame 0. This avoids delaying the visible thumb pre-shape until the
    # arm has completed its approach and entered the hand bridge.
    phase_a_hand = _phase_a_hand_qpos(traj_canon)
    home_qpos_full = np.zeros(n_dof, dtype=np.float32)
    if retrieval_source:
        planning_start_arm = np.asarray(
            getattr(args, "retrieval_start_arm_qpos", None), dtype=np.float32
        )
        if planning_start_arm.shape != (6,) or not np.all(np.isfinite(planning_start_arm)):
            raise ValueError(
                "retrieval_start_arm_qpos must contain six finite current xArm joints"
            )
    else:
        planning_start_arm = np.deg2rad(
            np.asarray(args.home_deg, dtype=np.float32)
        )
    home_qpos_full[:6] = planning_start_arm
    for jn, open_value in zip(ALLEGRO_HAND_JOINTS, phase_a_hand):
        if jn in name_to_full_idx:
            home_qpos_full[name_to_full_idx[jn]] = float(open_value)
    matched_finger = sum(jn in name_to_full_idx for jn in name_to_finger_idx)
    print(f"[pipe] {Path(source_path).name} -> URDF finger mapping matched "
          f"{matched_finger}/{len(jnames_traj)} joints; "
          "Phase A keeps long fingers open and pre-shapes thumb from replay frame 0 "
          f"({np.round(phase_a_hand[12:16], 3).tolist()})")

    robot_cfg = None
    world_dict = None
    if sdk_preflight:
        active_jn = [f"joint{i}" for i in range(1, 7)]
        ee_link = "wrist"
        plan_label = "xArm SDK IK preflight"
    else:
        from object6d.capture_rcc import (
            _run_curobo_plan,
            _normalize_curobo_robot_cfg_paths,
            _normalize_curobo_robot_cfg_paths_with_cfg_file,
            _rotmat_to_wxyz,
        )
        from paradex.utils.file_io import load_yaml

        cfg_path = os.path.expanduser(args.curobo_robot_cfg)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"curobo cfg not found: {cfg_path}")
        robot_cfg = _normalize_curobo_robot_cfg_paths_with_cfg_file(load_yaml(cfg_path), cfg_path)
        robot_cfg = _normalize_curobo_robot_cfg_paths(robot_cfg)
        lock_joints = robot_cfg["robot_cfg"]["kinematics"].get("lock_joints")
        if isinstance(lock_joints, dict):
            for jn in lock_joints:
                if jn in name_to_full_idx:
                    lock_joints[jn] = float(home_qpos_full[name_to_full_idx[jn]])

        table_size = np.array([float(x) for x in args.table_size_xyz.split(",")], dtype=np.float32)
        table_pos = np.array([float(x) for x in args.table_pos_xyz.split(",")], dtype=np.float32)
        obj_wxyz = _rotmat_to_wxyz(pose_robot[:3, :3])
        collision_world_dict = {
            "cuboid": {"table": {"dims": table_size.tolist(), "pose": [
                float(table_pos[0]), float(table_pos[1]), float(table_pos[2]),
                1.0, 0.0, 0.0, 0.0]}},
            "mesh": {"object": {
                "file_path": _resolve_mesh_path(
                    mesh_name, os.path.expanduser(args.mesh_root_dir), args.pose_mesh_frame
                ),
                "pose": [float(pose_robot[0, 3]), float(pose_robot[1, 3]), float(pose_robot[2, 3]),
                         float(obj_wxyz[0]), float(obj_wxyz[1]), float(obj_wxyz[2]), float(obj_wxyz[3])],
            }},
        }
        collision_free_world_dict = {
            "cuboid": {"_dummy_far": {
                "dims": [0.01, 0.01, 0.01],
                "pose": [10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0]}},
            "mesh": {},
        }
        world_dict = collision_free_world_dict if ignore_collisions else collision_world_dict
        active_jn = curobo_active_joint_names(robot_cfg)
        ee_link = str(robot_cfg["robot_cfg"]["kinematics"]["ee_link"])
        plan_label = "collision-ignored diagnostic" if ignore_collisions else "collision-aware"
    cartesian_execution = str(args.arm_execution_mode) == "cartesian"
    viser_ik_failures = []

    start_qpos_cspace = np.zeros(len(active_jn), dtype=np.float32)
    for ci, jn in enumerate(active_jn):
        if jn in name_to_full_idx:
            start_qpos_cspace[ci] = home_qpos_full[name_to_full_idx[jn]]

    def solve_viser_ik(
        target_poses: np.ndarray,
        seed_qpos: np.ndarray,
        *,
        phase: str,
        phase_world: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, int]:
        try:
            if sdk_preflight:
                command_poses = end_effector_poses_to_command_link(
                    robot_module,
                    target_poses,
                    home_qpos_full,
                    end_effector_link=ee_link,
                )
                limits = robot_module.get_joint_limits()
                sdk_joint_bounds = np.asarray(
                    [limits[f"joint{i}"] for i in range(1, 7)],
                    dtype=np.float64,
                )
                return _solve_ik_chain_xarm_sdk(
                    command_poses,
                    seed_qpos,
                    xarm_ip=str(args.xarm_sdk_ip),
                    phase=phase,
                    joint_bounds=sdk_joint_bounds,
                )
            return _solve_ik_chain_curobo(
                robot_cfg=robot_cfg,
                target_poses=target_poses,
                seed_arm_qpos=seed_qpos,
                device=curobo_device,
                num_seeds=int(args.ik_num_seeds),
                null_space_weight=float(args.ik_null_space_weight),
                self_collision=not ignore_collisions,
                position_threshold=float(args.ik_position_threshold),
                rotation_threshold=float(args.ik_rotation_threshold),
                sample_stride=int(args.viser_ik_stride),
                world_cfg_dict=phase_world,
            )
        except Exception as exc:
            if sdk_preflight:
                print(f"[pipe] {phase} xArm SDK IK ERROR: {exc}")
                frozen = np.asarray(seed_qpos, dtype=np.float32).reshape(1, -1)
                return np.tile(frozen, (len(target_poses), 1)), 0
            if not cartesian_execution:
                raise
            frozen = select_curobo_active_qpos(seed_qpos, robot_cfg, active_jn)
            print(f"[pipe] {phase} Viser IK ERROR: {exc}")
            print("[pipe] Cartesian execution mode: using a frozen xArm mesh preview; "
                  "the link6 pose trajectory remains available.")
            return np.tile(frozen, (len(target_poses), 1)).astype(np.float32), 0

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
    print(f"[pipe] Phase A ({plan_label}) staged waypoints: {stage_a_ends}")
    approach_active, n_stage_a_success = solve_viser_ik(
        stage_a_poses,
        start_qpos_cspace,
        phase="Phase A",
        phase_world=world_dict,
    )
    if n_stage_a_success != len(stage_a_poses):
        print(f"[pipe] Phase A FAILED: {n_stage_a_success}/{len(stage_a_poses)} IK waypoints")
        if not cartesian_execution or sdk_preflight:
            return {"planning_failed": "phase_a"}
        viser_ik_failures.append("phase_a")
        print("[pipe] Cartesian execution mode: Phase A IK is Viser-only; "
              "continuing with the staged link6 pose trajectory.")
    Na = int(approach_active.shape[0])
    print(f"[pipe] Phase A done: {Na} waypoints, active_dof={approach_active.shape[1]}")

    print(f"[pipe] Phase B: per-waypoint IK across {W} waypoints "
          f"(num_seeds={args.ik_num_seeds}, null_space_weight={args.ik_null_space_weight}, "
          f"position_threshold={args.ik_position_threshold}, "
          f"rotation_threshold={args.ik_rotation_threshold})")
    replay_active, n_ik_success = solve_viser_ik(
        wrist_T_robot,
        approach_active[-1],
        phase="Phase B",
    )
    if n_ik_success != W:
        print(f"[pipe] Phase B FAILED: only {n_ik_success}/{W} IK waypoints succeeded")
        if not cartesian_execution or sdk_preflight:
            return {"planning_failed": "phase_b"}
        viser_ik_failures.append("phase_b")
        print("[pipe] Cartesian execution mode: Phase B IK is Viser-only; "
              "continuing with the object-relative wrist pose trajectory.")
    # Include the Phase A endpoint so Phase B's first frame is unwrapped
    # relative to it (avoids a 2π swing across the bridge frames).
    _stitched = np.concatenate([approach_active[-1:], replay_active], axis=0)
    replay_active = _unwrap_continuous_joints(_stitched, active_jn)[1:]

    name_to_active_idx = {n: i for i, n in enumerate(active_jn)}

    approach_full = np.tile(home_qpos_full, (Na, 1))
    for jn, ai in name_to_active_idx.items():
        if jn in name_to_full_idx:
            approach_full[:, name_to_full_idx[jn]] = approach_active[:, ai]

    replay_full = np.tile(approach_full[-1], (W, 1))
    for jn, ai in name_to_active_idx.items():
        if jn in name_to_full_idx:
            replay_full[:, name_to_full_idx[jn]] = replay_active[:, ai]
    for jn, fi in name_to_finger_idx.items():
        if jn in name_to_full_idx:
            replay_full[:, name_to_full_idx[jn]] = traj_canon[:, 6 + fi].astype(np.float32)

    nh = int(max(0, args.hand_interp_steps))
    bridge_full = np.zeros((0, n_dof), dtype=np.float32)
    q_a_end = approach_full[-1]
    q_b_start = replay_full[0]
    if nh > 0 and not np.allclose(q_a_end, q_b_start, atol=1e-6):
        if nh >= 2:
            alphas = np.linspace(0.0, 1.0, nh + 1, dtype=np.float32)[1:-1]
        else:
            alphas = np.array([0.5], dtype=np.float32)
        bridge_full = (1.0 - alphas[:, None]) * q_a_end[None, :] + alphas[:, None] * q_b_start[None, :]
    Nb = int(bridge_full.shape[0])

    Nh = int(max(0, args.grasp_hold_frames))
    hold_full = np.tile(replay_full[-1], (Nh, 1)) if Nh > 0 else np.zeros((0, n_dof), dtype=np.float32)

    lift_h = float(args.lift_height_m)
    Nc = 0
    lift_full = np.zeros((0, n_dof), dtype=np.float32)
    lift_wrist_poses = np.zeros((0, 4, 4), dtype=np.float64)
    if lift_h > 0:
        print(f"[pipe] Phase C: lift +{lift_h * 100:.1f}cm in robot +Z")
        lift_target = wrist_T_robot[-1].copy()
        lift_target[2, 3] += lift_h

        start_lift_cspace = replay_active[-1].copy()
        if cartesian_execution:
            lift_preview_poses, _ = pose_segment(
                wrist_T_robot[-1],
                lift_target,
                linear_speed_mps=float(args.stage_a_descent_speed_mps),
                angular_speed_rps=float(args.stage_a_angular_speed_rps),
                rate_hz=float(args.rate_hz),
            )
            lift_preview_active, n_lift_ik_success = solve_viser_ik(
                lift_preview_poses,
                start_lift_cspace,
                phase="Phase C",
                phase_world=world_dict,
            )
            if n_lift_ik_success != len(lift_preview_poses):
                if sdk_preflight:
                    return {"planning_failed": "phase_c"}
                viser_ik_failures.append("phase_c")
                print("[pipe] Cartesian execution mode: Phase C IK is Viser-only; "
                      "continuing with a vertical link6 lift.")
            lift_wrist_poses = lift_preview_poses[1:]
            lift_active = lift_preview_active[1:]
            Nc = int(len(lift_wrist_poses))
            lift_full = np.tile(replay_full[-1], (Nc, 1))
            for jn, ai in name_to_active_idx.items():
                if jn in name_to_full_idx:
                    lift_full[:, name_to_full_idx[jn]] = lift_active[:, ai]
            print(f"[pipe] Phase C Cartesian lift done: {Nc} waypoints")
        else:
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
                self_collision=not ignore_collisions,
            )
            if lift_plan.get("success"):
                lift_active = np.asarray(lift_plan["trajectory"], dtype=np.float32)
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

    # ---- Phase D: release -- fully open the hand after lift while the arm
    # stays at the lifted pose.
    Nd = int(max(0, args.release_steps))
    release_full = np.zeros((0, n_dof), dtype=np.float32)
    if Nd > 0:
        last = lift_full[-1] if Nc > 0 else (hold_full[-1] if Nh > 0 else replay_full[-1])
        target = last.copy()
        for jn, open_value in zip(ALLEGRO_HAND_JOINTS, OPEN_HAND_QPOS):
            if jn in name_to_full_idx:
                target[name_to_full_idx[jn]] = float(open_value)
        if Nd >= 2:
            alphas = np.linspace(0.0, 1.0, Nd + 1, dtype=np.float32)[1:]
        else:
            alphas = np.array([1.0], dtype=np.float32)
        release_full = (1.0 - alphas[:, None]) * last[None, :] + alphas[:, None] * target[None, :]
        print(f"[pipe] Phase D: fully open hand over {Nd} frames")

    full_traj = np.concatenate(
        [approach_full, bridge_full, replay_full, hold_full, lift_full, release_full], axis=0)
    try:
        assert_arm_joint_limits(robot_module, full_traj)
    except ValueError as exc:
        print(f"[pipe] joint-limit safety check FAILED: {exc}")
        if not cartesian_execution or sdk_preflight:
            return {"planning_failed": "joint_limits", "planning_error": str(exc)}
        if "joint_limits" not in viser_ik_failures:
            viser_ik_failures.append("joint_limits")
        print("[pipe] Cartesian execution mode: joint-limit failure belongs to the "
              "Viser-only IK path; Cartesian execution remains available.")
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
    if Nc > 0 and cartesian_execution:
        lift_arm_poses = end_effector_poses_to_command_link(
            robot_module,
            lift_wrist_poses,
            home_qpos_full,
            end_effector_link=ee_link,
        )
    else:
        lift_arm_poses = (
            arm_link_pose_trajectory(robot_module, lift_full)
            if Nc > 0 else np.zeros((0, 4, 4), dtype=np.float64)
        )
    release_start_pose = (
        lift_arm_poses[-1] if Nc > 0 else replay_arm_poses[-1]
    )
    release_arm_poses = np.tile(release_start_pose, (Nd, 1, 1))
    arm_pose_traj = np.concatenate(
        [approach_arm_poses, bridge_arm_poses, replay_arm_poses,
         hold_arm_poses, lift_arm_poses, release_arm_poses],
        axis=0,
    )
    if len(arm_pose_traj) != len(full_traj):
        raise RuntimeError("Cartesian command and Viser preview lengths differ")
    if cartesian_execution and viser_ik_failures:
        print("[pipe] WARNING: Cartesian execution remains enabled, but CuRobo did "
              "not prove reachability or collision clearance for all phases. "
              "The xArm controller's built-in protections remain active.")

    artifact_suffix = "_collision_ignored" if ignore_collisions else ""
    traj_filename = f"h2r_pipeline_traj{artifact_suffix}.npy"
    meta_filename = f"h2r_pipeline_meta{artifact_suffix}.json"
    np.save(os.path.join(abs_save_path, traj_filename), full_traj)
    np.save(os.path.join(abs_save_path, f"h2r_pipeline_link6_poses{artifact_suffix}.npy"), arm_pose_traj)
    with open(os.path.join(abs_save_path, meta_filename), "w") as f:
        json.dump({"n_approach": Na, "n_hand_interp": Nb, "n_replay": W,
                   "n_grasp_hold": Nh, "n_lift": Nc, "n_release": Nd,
                   "lift_height_m": lift_h,
                   "joint_names": full_joint_names,
                   "active_joint_names": active_jn,
                   "home_deg": list(args.home_deg),
                   "planning_start_arm_qpos": planning_start_arm.tolist(),
                   "replay_source": str(source_path),
                   "retrieval_action": retrieval_action,
                   "phase_a_hand_qpos": phase_a_hand.tolist(),
                   "arm_execution_mode": str(args.arm_execution_mode),
                   "path_preview_backend": preview_backend,
                   "sdk_ik_preflight": bool(sdk_preflight),
                   "collision_verified": bool(not sdk_preflight and not ignore_collisions),
                   "viser_ik_complete": not bool(viser_ik_failures),
                   "viser_ik_failures": list(viser_ik_failures),
                   "diagnostic_only": bool(ignore_collisions),
                   "collisions_ignored": bool(ignore_collisions),
                   "hand_interp_steps_requested": int(args.hand_interp_steps),
                   "grasp_hold_frames_requested": int(args.grasp_hold_frames),
                   "release_steps_requested": int(args.release_steps)}, f, indent=2)
    print(f"[pipe] saved {traj_filename} shape={full_traj.shape}")

    cursor = 0
    ranges = []
    for name, n in [("approach", Na), ("hand_interp", Nb), ("replay", W),
                    ("grasp_hold", Nh), ("lift", Nc), ("release", Nd)]:
        if n > 0:
            ranges.append((name, cursor, cursor + n - 1))
            cursor += n
    range_str = ", ".join(f"{n}[{lo}..{hi}]" for n, lo, hi in ranges)
    print(f"[pipe] planning done; trajectory shape={full_traj.shape}, "
          f"ranges {range_str}.")
    return {"full_traj": full_traj, "arm_pose_traj": arm_pose_traj, "range_str": range_str,
            "diagnostic_only": bool(ignore_collisions),
            "collisions_ignored": bool(ignore_collisions),
            "phase_a_hand_qpos": phase_a_hand.copy(),
            "planning_start_arm_qpos": planning_start_arm.copy(),
            "replay_source": str(source_path),
            "retrieval_action": retrieval_action,
            "arm_execution_mode": str(args.arm_execution_mode),
            "path_preview_backend": preview_backend,
            "sdk_ik_preflight": bool(sdk_preflight),
            "collision_verified": bool(not sdk_preflight and not ignore_collisions),
            "viser_ik_complete": not bool(viser_ik_failures),
            "viser_ik_failures": list(viser_ik_failures),
            "stage_a_ends": dict(stage_a_ends),
            "n_approach": Na, "n_hand_interp": Nb, "n_replay": W,
            "n_grasp_hold": Nh, "n_lift": Nc, "n_release": Nd}


def _build_foundationpose_grasp_scene(
    args,
    mesh_name: str,
    pose_robot: np.ndarray,
    failure_phase: str = "phase_a",
) -> Dict[str, Any]:
    """Build a static, hand-only final grasp scene without solving arm IK."""

    from scipy.spatial.transform import Rotation

    retrieval_action = getattr(args, "retrieval_action", None)
    if retrieval_action is not None:
        wrist_object, hand_trajectory, _ = _load_retrieval_replay(
            args, mesh_name, RobotModule(os.path.expanduser(args.robot_urdf))
        )
        hand_pose_robot = np.asarray(pose_robot, dtype=np.float64) @ wrist_object[-1]
        hand_qpos = hand_trajectory[-1].astype(np.float32)
        source_description = f"retrieval={retrieval_action}"
    else:
        res_path = os.path.join(
            os.path.expanduser(args.result_root),
            mesh_name,
            str(args.scene),
            "result_allegro.npz",
        )
        if not os.path.exists(res_path):
            raise FileNotFoundError(f"no packed result: {res_path}")
        packed = np.load(res_path, allow_pickle=True)
        pack_to_pose = pack_frame_to_pose_frame(packed, args.pose_mesh_frame)
        best_key = f"{args.src}_traj_q"
        all_key = f"{args.src}_traj_q_all"
        if args.particle is None:
            if best_key not in packed.files:
                raise KeyError(f"{best_key} missing in {res_path}")
            trajectory = np.asarray(packed[best_key], dtype=np.float64)
            particle_used = (
                int(packed[f"{args.src}_best_particle"])
                if f"{args.src}_best_particle" in packed.files
                else -1
            )
        else:
            if all_key not in packed.files:
                raise KeyError(f"{all_key} missing in {res_path}")
            trajectory = np.asarray(
                packed[all_key][int(args.particle)], dtype=np.float64
            )
            particle_used = int(args.particle)
        if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] != 22:
            raise ValueError(
                f"expected packed Allegro trajectory shape (W, 22), got {trajectory.shape}"
            )
        final = trajectory[-1]
        wrist_canonical = np.eye(4, dtype=np.float64)
        wrist_canonical[:3, :3] = Rotation.from_euler("XYZ", final[3:6]).as_matrix()
        wrist_canonical[:3, 3] = final[:3]
        hand_pose_robot = (
            np.asarray(pose_robot, dtype=np.float64) @ pack_to_pose @ wrist_canonical
        )
        hand_qpos = final[6:].astype(np.float32)
        source_description = f"src={args.src} particle={particle_used}"
    print(
        f"[pipe] FoundationPose grasp-scene fallback: {mesh_name} scene={args.scene} "
        f"{source_description}, final frame only"
    )
    print("[pipe] No arm IK trajectory exists; REAL execution is disabled.")
    return {
        "diagnostic_only": True,
        "diagnostic_kind": "foundationpose_grasp_scene",
        "failure_phase": str(failure_phase),
        "hand_urdf": DEFAULT_ALLEGRO_HAND_URDF,
        "hand_pose": hand_pose_robot.astype(np.float32),
        "hand_qpos": hand_qpos,
        "range_str": "final_grasp_scene[static]",
    }


def _run_pipeline_with_collision_fallback(
    args,
    mesh_name: str,
    pose_robot: np.ndarray,
    robot_module: RobotModule,
    abs_save_path: str,
    alternative_pose_robot: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Run an executable plan, trying the opposite symmetry branch first."""
    result = _run_h2r_pipeline(
        args, mesh_name, pose_robot, robot_module, abs_save_path,
    )
    primary_failed = result is None or bool(result.get("planning_failed"))
    if primary_failed and alternative_pose_robot is not None:
        failure = "unknown" if result is None else str(result["planning_failed"])
        print(f"\n[pipe] Primary symmetry branch failed ({failure}).")
        print("[pipe] Retrying collision-aware planning on the opposite equivalent branch.")
        alternative_result = _run_h2r_pipeline(
            args,
            mesh_name,
            alternative_pose_robot,
            robot_module,
            abs_save_path,
        )
        if alternative_result is not None and not alternative_result.get("planning_failed"):
            alternative_result["object_pose_robot"] = np.asarray(
                alternative_pose_robot, dtype=np.float64
            )
            alternative_result["symmetry_branch"] = "alternate"
            print("[pipe] Opposite symmetry branch passed executable planning.")
            return alternative_result
        result = alternative_result if alternative_result is not None else result
    if result is not None and result.get("planning_failed"):
        failure = str(result["planning_failed"])
        print(f"\n[pipe] WARNING: executable planning failed ({failure}).")
        print("[pipe] Falling back to the FoundationPose object + final grasp scene.")
        return _build_foundationpose_grasp_scene(
            args, mesh_name, pose_robot, failure_phase=failure
        )
    if result is not None or not args.collision_ignored_fallback:
        return result

    print("\n[pipe] WARNING: collision-aware planning failed.")
    print("[pipe] Retrying the FULL trajectory with world + self collisions ignored.")
    print("[pipe] This fallback is for Viser diagnosis only and REAL execution is disabled.\n")
    result = _run_h2r_pipeline(
        args,
        mesh_name,
        pose_robot,
        robot_module,
        abs_save_path,
        ignore_collisions=True,
    )
    if result is not None and result.get("planning_failed"):
        print("[pipe] collision-ignored planning also failed; showing final grasp scene")
        return _build_foundationpose_grasp_scene(
            args,
            mesh_name,
            pose_robot,
            failure_phase=str(result["planning_failed"]),
        )
    if result is None:
        print("[pipe] collision-ignored fallback also failed; no trajectory to visualize")
    return result


def _real_execution_allowed(pipeline_result: Optional[Dict[str, Any]]) -> bool:
    """Collision-ignored diagnostic trajectories must never reach hardware."""
    return bool(
        pipeline_result is not None
        and not pipeline_result.get("diagnostic_only", False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "--mesh_name", dest="mesh_name", required=True)
    parser.add_argument("--save_path", default="capture/test_allegro",
                        help="relative path under shared_data")
    parser.add_argument("--rpc_addr", default="tcp://192.168.0.13:5570")
    parser.add_argument("--rpc_timeout_ms", type=int, default=300000)
    parser.add_argument("--rcc_entry", default="image_main.py")
    parser.add_argument("--mesh_root_dir", default=os.path.join(shared_dir, "mesh_new"))
    parser.add_argument(
        "--pose_mesh_frame",
        default="original",
        choices=("original", "viser", "canonical"),
        help=("object frame reported by the 6D pose RPC; FoundationPose uses "
              "original. 'viser' and 'canonical' are aliases"),
    )
    parser.add_argument("--c2r_path", default=None, help="optional C2R.npy override")
    parser.add_argument("--robot_urdf", default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--viser_addr", default="tcp://localhost:5572",
                        help="ZMQ REQ address of a running viser_server. "
                             "Empty string => fall back to launching a local "
                             "ViserViewer (blocking).")
    parser.add_argument("--viser_max_mesh_faces", type=int, default=1000,
                        help="per-mesh face budget for local ViserViewer; 0 disables simplification")
    parser.add_argument("--debug_image_mode", default="popup",
                        choices=["popup", "none"],
                        help="how to show RPC mask overlay PNGs: popup = OS "
                             "image viewer window(s); none")
    parser.add_argument(
        "--normalize", default="none", choices=pose_normalize.MODES,
        help=("generic pose normalization for objects without registered symmetry; "
              "known mesh_new symmetries are handled "
              "automatically and take precedence"))
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
    parser.add_argument("--particle", type=int, default=None,
                        help="particle index 0..7. Default = use <src>_traj_q "
                             "(already-selected best particle). Override to "
                             "pick a specific particle from <src>_traj_q_all.")
    parser.add_argument(
        "--result_root",
        default=os.path.join(shared_dir, "capture", "h2r", "0518"),
        help="root holding <obj>/<scene>/result.npz (pack_results.py output)")
    parser.add_argument(
        "--retrieval_action",
        "--retrieval-action",
        choices=RETRIEVAL_ACTIONS,
        default=None,
        help=("replay evaluated_actions/<action>/approach_grasp_trajectory_object.npz "
              "instead of result_allegro.npz; this implies --pipeline"),
    )
    parser.add_argument(
        "--retrieval_root",
        "--retrieval-root",
        default=DEFAULT_RETRIEVAL_ROOT,
        help=("root holding <object>/scene_<N>/evaluated_actions; default is "
              "shared_data/grasp_retrieval_meshes_20260915_v1/allegro_old"),
    )
    parser.add_argument(
        "--retrieval_start_arm_deg",
        "--retrieval-start-arm-deg",
        nargs=6,
        type=float,
        default=None,
        help=("optional offline override for the retrieval planning start; default "
              "reads the current six xArm joint angles from feedback"),
    )
    parser.add_argument("--pipeline", action="store_true",
                        help="build the h2r preview/execution trajectory")
    parser.add_argument(
        "--no_collision_ignored_fallback",
        dest="collision_ignored_fallback",
        action="store_false",
        help="disable the automatic Viser-only replan with all collision checks "
             "off after collision-aware planning fails",
    )
    parser.set_defaults(collision_ignored_fallback=True)
    parser.add_argument("--home_deg", nargs=6, type=float, default=HOME_DEG,
                        help="xarm home joint angles in degrees (6 values)")
    parser.add_argument(
        "--home_joint1_deg",
        "--start_joint1_deg",
        dest="home_joint1_deg",
        type=float,
        default=None,
        help=("override only xArm joint1 in --home_deg; for example, "
              "--home_joint1_deg -60 starts and returns with joint1 at -60 deg"),
    )
    parser.add_argument("--curobo_robot_cfg", default=DEFAULT_CUROBO_ROBOT_CFG,
                        help="curobo robot config yaml (xarm_allegro_v5)")
    parser.add_argument("--curobo_device", default="cuda:1")
    parser.add_argument("--curobo_seeds", type=int, default=128)
    parser.add_argument("--ik_num_seeds", type=int, default=32,
                        help="curobo IK num_seeds for the per-waypoint chain. "
                             "Lower => stays close to previous solution.")
    parser.add_argument("--ik_position_threshold", type=float, default=0.01,
                        help="Phase B IK position tolerance in meters; default "
                             "matches Phase A MotionGen (0.01 m)")
    parser.add_argument("--ik_rotation_threshold", type=float, default=0.1,
                        help="Phase B IK rotation tolerance in radians; default "
                             "matches Phase A MotionGen (0.1 rad)")
    parser.add_argument("--ik_null_space_weight", type=float, default=1.0,
                        help="penalty on distance from retract_config (= last "
                             "IK solution). Higher => stronger continuity but "
                             "can fight target tracking; lower if arm starts "
                             "drifting sideways instead of toward the target.")
    parser.add_argument("--viser_ik_stride", type=int, default=10,
                        help="solve every Nth waypoint for the Viser-only IK preview; "
                             "real xArm execution uses the dense Cartesian path")
    parser.add_argument(
        "--path_preview_backend",
        choices=("curobo", "xarm_sdk"),
        default="curobo",
        help=("IK backend used to build the arm path shown in Viser. xarm_sdk "
              "bypasses CuRobo, queries the controller SDK without moving the "
              "robot, and requires --arm_execution_mode cartesian. It checks IK "
              "success, joint limits, continuity, and predicted joint speed, but "
              "does not check collisions."),
    )
    parser.add_argument(
        "--xarm_sdk_ip",
        default=None,
        help="xArm controller IP for --path_preview_backend xarm_sdk; default uses system/current/network.json",
    )
    parser.add_argument("--stage_a_linear_speed_mps", type=float, default=0.12)
    parser.add_argument("--stage_a_angular_speed_rps", type=float, default=0.70)
    parser.add_argument("--stage_a_raise_speed_mps", type=float, default=0.03,
                        help="vertical HOME-to-hover speed; kept conservative because "
                             "Cartesian servo IK can amplify joint4 speed")
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
                        help="bridge frames between Phase A end (arm at pregrasp, "
                             "hand open) and Phase B start (hand at first replay "
                             "finger qpos). Arm is held; fingers ramp.")
    parser.add_argument("--grasp_hold_frames", type=int, default=30,
                        help="dwell frames after grasp replay (arm + hand still)")
    parser.add_argument("--grasp_squeeze", type=float, default=0.0,
                        help="add this many radians to every hand joint except "
                             "joint_13_0 (thumb abduction) AFTER the grasp "
                             "sequence finishes, ramped over --squeeze_steps. "
                             "0 disables.")
    parser.add_argument("--squeeze_steps", type=int, default=20)
    parser.add_argument("--lift_height_m", type=float, default=0.12)
    parser.add_argument("--release_steps", type=int, default=30,
                        help="after lift, fully open the hand over this many "
                             "frames while the arm stays still. 0 disables.")
    parser.add_argument("--post_lift_hold_seconds", type=float, default=5.0,
                        help="hold the lifted grasp before release")
    # --- real-robot execution ---
    parser.add_argument("--rate_hz", type=float, default=30.0)
    parser.add_argument("--replay_rate_hz", type=float, default=15.0)
    parser.add_argument("--home_tol_deg", type=float, default=2.0)
    parser.add_argument("--home_hand_tol_rad", type=float, default=ALLEGRO_OPEN_TOL_RAD,
                        help="base Allegro open-pose feedback tolerance; the "
                             "hysteretic thumb stretch joint uses a separate minimum")
    parser.add_argument(
        "--enable_allegro_start_posture_check",
        action="store_true",
        help=("enable the legacy pre-execution check that waits for measured "
              "Allegro joints to converge to the Phase A hand target; disabled "
              "by default because the physical thumb base may settle above its "
              "URDF lower limit"),
    )
    parser.add_argument("--home_timeout", type=float, default=20.0)
    parser.add_argument("--home_max_joint_speed_deg_s", type=float, default=10.0,
                        help="maximum xArm joint speed while prepositioning to HOME")
    parser.add_argument("--return_home_max_joint_speed_deg_s", type=float, default=30.0,
                        help="maximum xArm joint speed when returning HOME after release")
    parser.add_argument("--home_preposition_hz", type=float, default=30.0,
                        help="joint target stream rate while prepositioning to HOME")
    parser.add_argument("--arm_servo_rate_hz", type=float, default=100.0,
                        help="dense xArm servo command rate during replay")
    parser.add_argument(
        "--arm_execution_mode",
        choices=("joint", "cartesian"),
        default="joint",
        help=("xArm command representation during the grasp sequence. joint keeps "
              "the CuRobo/Viser joint path; cartesian streams the staged link6 pose "
              "path through xArm's Cartesian servo. HOME preposition and final HOME "
              "return remain bounded joint-space motions."),
    )
    parser.add_argument("--max_planned_joint_speed_deg_s", type=float, default=30.0,
                        help="reject previewed IK paths that exceed this joint speed")
    parser.add_argument("--max_cartesian_linear_speed_mps", type=float, default=0.10)
    parser.add_argument("--max_cartesian_angular_speed_rps", type=float, default=0.50)
    parser.add_argument(
        "--enable_live_joint_watchdog",
        action="store_true",
        help=("enable the legacy software watchdog for measured xArm joint speed "
              "and joint-limit margins; disabled by default"),
    )
    parser.add_argument("--max_measured_joint_speed_deg_s", type=float, default=60.0)
    parser.add_argument("--live_joint_limit_margin_rad", type=float, default=0.03)
    parser.add_argument("--home_hand_max_speed_rad_s", type=float, default=0.5,
                        help="maximum Allegro joint speed while opening before execution")
    parser.add_argument("--hand_side", default="right", choices=["right", "left"])
    parser.add_argument("--allegro_namespace", default="",
                        help="ROS2 namespace for Allegro topics; empty selects /allegroHand_0/...; "
                             "use right for /right/allegroHand_0/...")
    parser.add_argument("--record_video", action="store_true", default=True)
    parser.add_argument("--no_record_video", dest="record_video", action="store_false")
    parser.add_argument("--realsense", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help="skip the post-viser execute confirmation prompt")
    args = parser.parse_args()
    if args.retrieval_action is not None and not args.pipeline:
        args.pipeline = True
        print("[pipe] --retrieval_action implies --pipeline")
    if args.home_joint1_deg is not None:
        args.home_deg = list(args.home_deg)
        args.home_deg[0] = float(args.home_joint1_deg)
        print(f"[pipe] HOME joint1 override: {args.home_deg[0]:.1f} deg")
    validate_motion_args(args)
    if args.path_preview_backend == "xarm_sdk":
        if args.arm_execution_mode != "cartesian":
            parser.error("--path_preview_backend xarm_sdk requires --arm_execution_mode cartesian")
        if args.xarm_sdk_ip is None:
            from paradex.utils.system import network_info

            args.xarm_sdk_ip = network_info["xarm"]["param"]["ip"]
    args.result_root = os.path.expanduser(args.result_root)
    args.retrieval_root = os.path.expanduser(args.retrieval_root)
    if args.pipeline and args.scene is None:
        if args.retrieval_action is not None:
            args.scene = _resolve_retrieval_scene(
                args.retrieval_root, args.mesh_name, args.scene
            )
        else:
            args.scene = resolve_scene_arg(
                args.result_root, args.mesh_name, "result_allegro.npz", args.scene
            )
        print(f"[pipe] auto-selected scene={args.scene}")

    mesh_path = _resolve_mesh_path(
        args.mesh_name, os.path.expanduser(args.mesh_root_dir), args.pose_mesh_frame
    )
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"mesh not found: {mesh_path}")

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
    robot_urdf_path = os.path.expanduser(args.robot_urdf)
    if not os.path.exists(robot_urdf_path):
        raise FileNotFoundError(f"robot urdf not found: {robot_urdf_path}")
    robot_module = RobotModule(robot_urdf_path)
    if args.retrieval_action is not None:
        if args.retrieval_start_arm_deg is None:
            print("[pipe] reading current xArm state for retrieval approach start ...")
            args.retrieval_start_arm_qpos = _read_current_xarm_qpos()
        else:
            args.retrieval_start_arm_qpos = np.deg2rad(
                np.asarray(args.retrieval_start_arm_deg, dtype=np.float64)
            )
        print(
            "[pipe] retrieval planning START deg="
            f"{np.rad2deg(args.retrieval_start_arm_qpos).round(2).tolist()}"
        )
    symmetry_info = None
    normalization_applied = "none"
    symmetry_wrist_object = None
    symmetry_home_link6_robot = None
    alternative_pose_robot = None
    if (
        args.pipeline
        and args.object_symmetry_normalization
        and args.pose_mesh_frame == "original"
        and pose_normalize.has_known_object_symmetry(args.mesh_name)
    ):
        if args.retrieval_action is not None:
            retrieval_wrist_object, _, _ = _load_retrieval_replay(
                args, args.mesh_name, robot_module
            )
            symmetry_wrist_object = np.asarray(
                retrieval_wrist_object[0], dtype=np.float64
            )
        else:
            symmetry_result_path = os.path.join(
                args.result_root, args.mesh_name, str(args.scene), "result_allegro.npz"
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
        home_qpos = np.zeros(robot_module.get_num_joints(), dtype=np.float64)
        home_names = list(robot_module.get_joint_names())
        for joint_i, value in enumerate(np.deg2rad(args.home_deg), start=1):
            joint_name = f"joint{joint_i}"
            if joint_name in home_names:
                home_qpos[home_names.index(joint_name)] = float(value)
        for joint_name, value in zip(ALLEGRO_HAND_JOINTS, OPEN_HAND_QPOS):
            if joint_name in home_names:
                home_qpos[home_names.index(joint_name)] = float(value)
        # Packed wrist[6] poses are consumed as the CuRobo end-effector target,
        # whose configured frame is xArm link6. Compare like-for-like here;
        # Allegro's fixed `wrist` frame is rotated 90 degrees from link6.
        symmetry_home_link6_robot = robot_module.forward_kinematics(home_qpos)[
            "link6"
        ]
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
            # Allegro V5 convention: local -Z is palm-to-wrist and should
            # point from the object back toward the xArm base.
            wrist_robotward_axis_local=np.array([0.0, 0.0, -1.0]),
            reference_wrist_pose_robot=symmetry_home_link6_robot,
        )
        alternative_delta = symmetry_info.get(
            "alternate_wrist_symmetry_delta_rad"
        ) if symmetry_info is not None else None
        if alternative_delta is not None:
            from scipy.spatial.transform import Rotation

            symmetry_axis = np.asarray(symmetry_info["axis_object"], dtype=np.float64)
            symmetry_axis /= np.linalg.norm(symmetry_axis)
            symmetry_transform = np.eye(4, dtype=np.float64)
            symmetry_transform[:3, :3] = Rotation.from_rotvec(
                symmetry_axis * float(alternative_delta)
            ).as_matrix()
            symmetry_center = np.asarray(
                symmetry_info.get("symmetry_center_object", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            symmetry_transform[:3, 3] = (
                symmetry_center
                - symmetry_transform[:3, :3] @ symmetry_center
            )
            alternative_pose_robot = pose_robot @ symmetry_transform
    if symmetry_info is not None:
        normalization_applied = symmetry_info["normalize_mode"]
        if args.normalize != "none":
            print(
                f"[normalize] ignoring generic mode={args.normalize} for "
                f"{args.mesh_name}; object-specific symmetry axis takes precedence"
            )
        if symmetry_info["symmetry_type"] == "continuous_axis":
            print(
                f"[normalize] {args.mesh_name} continuous symmetry: "
                f"axis={np.round(symmetry_info['axis_object'], 6).tolist()}, "
                f"spin={np.rad2deg(symmetry_info['angle_rad']):+.1f} deg, "
                f"flipped={'yes' if symmetry_info['axis_flipped'] else 'no'}"
            )
        else:
            print(
                f"[normalize] {args.mesh_name} {symmetry_info['symmetry_type']} symmetry: "
                f"variant={symmetry_info['variant_index']}/"
                f"{symmetry_info.get('variant_count', 4) - 1}"
            )
        if symmetry_wrist_object is not None:
            home_distance = symmetry_info.get("wrist_home_orientation_distance_rad")
            preferred_facing = symmetry_info.get("preferred_wrist_facing_angle_rad")
            selection_detail = ""
            if home_distance is not None and preferred_facing is not None:
                selection_detail = (
                    f", HOME wrist rotation: {np.rad2deg(home_distance):.1f} deg, "
                    f"preferred cone: {np.rad2deg(preferred_facing):.1f} deg"
                )
            print(
                "[normalize] pregrasp wrist relative height: "
                f"{symmetry_info['wrist_relative_height_before_m']:+.3f} -> "
                f"{symmetry_info['wrist_relative_height_after_m']:+.3f} m "
                f"(robot-facing angle: "
                f"{np.rad2deg(symmetry_info['wrist_facing_angle_rad']):.1f} deg, "
                f"constraints={'ok' if symmetry_info['wrist_constraints_satisfied'] else 'miss'}"
                f"{selection_detail})"
            )
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

    if args.no_vis:
        return

    pipeline_result = None
    if args.pipeline:
        pipeline_result = _run_pipeline_with_collision_fallback(
            args,
            args.mesh_name,
            pose_robot,
            robot_module,
            abs_save_path,
            alternative_pose_robot=alternative_pose_robot,
        )
        if pipeline_result is not None and "object_pose_robot" in pipeline_result:
            pose_robot = _to_4x4(pipeline_result["object_pose_robot"])
            pose_world = _to_4x4(c2r @ pose_robot)
            robot_out.update({
                "pose_robot": pose_robot.tolist(),
                "R_robot": pose_robot[:3, :3].tolist(),
                "t_robot": pose_robot[:3, 3].tolist(),
                "pose_world": pose_world.tolist(),
            })
            if symmetry_info is not None:
                symmetry_info["planner_selected_branch"] = "alternate"
                symmetry_info["angle_rad"] = symmetry_info.get(
                    "alternate_angle_rad", symmetry_info["angle_rad"]
                )
                robot_out["object_symmetry"] = symmetry_info
            with open(robot_pose_json_path, "w", encoding="utf-8") as f:
                json.dump(robot_out, f, ensure_ascii=False, indent=2)
            print(
                "[normalize] saved planner-selected opposite symmetry branch: "
                f"{robot_pose_json_path}"
            )
    symmetry_constraints_ok = bool(
        symmetry_info is None
        or symmetry_wrist_object is None
        or symmetry_info.get("wrist_constraints_satisfied", False)
    )
    if pipeline_result is not None and not symmetry_constraints_ok:
        pipeline_result["diagnostic_only"] = True
        pipeline_result["diagnostic_reason"] = (
            "no symmetry-equivalent wrist pose satisfied both the upper-half "
            f"condition and the {args.symmetry_max_wrist_facing_deg:.1f}-degree "
            "robot-facing orientation limit"
        )
        print(f"[pipe] WARNING: {pipeline_result['diagnostic_reason']}")
        print("[pipe] Viser preview is retained; REAL execution is disabled.")

    mesh = _load_mesh(mesh_path)
    grasp_scene_only = bool(
        pipeline_result is not None
        and pipeline_result.get("diagnostic_kind") == "foundationpose_grasp_scene"
    )
    if pipeline_result is not None and "full_traj" in pipeline_result:
        rest_qpos = pipeline_result["full_traj"][0].astype(np.float32)
    else:
        rest_qpos = np.zeros(robot_module.get_num_joints(), dtype=np.float32)
        if grasp_scene_only:
            full_joint_names = list(robot_module.get_joint_names())
            rest_by_name = {name: i for i, name in enumerate(full_joint_names)}
            for joint_i, value in enumerate(np.deg2rad(args.home_deg), start=1):
                joint_name = f"joint{joint_i}"
                if joint_name in rest_by_name:
                    rest_qpos[rest_by_name[joint_name]] = float(value)
            for joint_name, value in zip(ALLEGRO_HAND_JOINTS, OPEN_HAND_QPOS):
                if joint_name in rest_by_name:
                    rest_qpos[rest_by_name[joint_name]] = float(value)

    debug_dir = os.path.join(abs_save_path, "debug")
    if args.debug_image_mode == "popup":
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
                f"--robot_urdf {robot_urdf_path}` or rerun with --viser_addr ''.") from e
        vc.clear_objects()
        vc.set_object(args.mesh_name, mesh, pose_robot, opacity=1.0)
        if grasp_scene_only:
            vc.show_only_robot("robot")
            vc.set_qpos(rest_qpos, name="robot")
            response = vc.set_static_robot(
                "foundationpose_grasp_hand",
                pipeline_result["hand_urdf"],
                pipeline_result["hand_pose"],
                pipeline_result["hand_qpos"],
                show_only=False,
            )
            if not response.get("ok", False):
                raise RuntimeError(f"viser_server could not show grasp hand: {response}")
            print("[viser] pushed HOME xArm reference + FoundationPose final grasp")
        else:
            vc.show_only_robot("robot")
            vc.set_qpos(rest_qpos, name="robot")
        if pipeline_result is not None and "full_traj" in pipeline_result:
            diagnostic_only = bool(pipeline_result.get("diagnostic_only", False))
            traj_name = ("COLLISION_IGNORED_DIAGNOSTIC_ONLY"
                         if diagnostic_only else "h2r_pipeline")
            vc.set_traj(pipeline_result["full_traj"], name=traj_name, robot_name="robot")
            print(f"[pipe] pushed {pipeline_result['full_traj'].shape[0]} frames to "
                  f"viser_server; ranges {pipeline_result['range_str']}.")
            if not pipeline_result.get("viser_ik_complete", True):
                print("[viser] WARNING: xArm mesh IK is incomplete for "
                      f"{pipeline_result.get('viser_ik_failures', [])}; the Cartesian "
                      "link6 path remains executable and may not match the displayed joints.")
            if diagnostic_only:
                reason = pipeline_result.get(
                    "diagnostic_reason", "displayed trajectory ignores ALL collisions"
                )
                print(f"[viser] WARNING: {reason}; visualization only, execution disabled")
        elif pipeline_result is None:
            print(f"[viser] pushed object + start qpos to viser_server at {args.viser_addr}")
        vc.close()
    else:
        vis = ViserViewer()
        vis.add_floor(height=0.0)
        max_mesh_faces = args.viser_max_mesh_faces if args.viser_max_mesh_faces > 0 else None
        if grasp_scene_only:
            vis.add_robot("robot", robot_urdf_path, max_mesh_faces=max_mesh_faces)
            vis.robot_dict["robot"].update_cfg(rest_qpos)
            vis.add_robot(
                "foundationpose_grasp_hand",
                pipeline_result["hand_urdf"],
                pose=pipeline_result["hand_pose"],
                max_mesh_faces=max_mesh_faces,
            )
            vis.robot_dict["foundationpose_grasp_hand"].update_cfg(
                pipeline_result["hand_qpos"]
            )
        else:
            vis.add_robot("robot", robot_urdf_path, max_mesh_faces=max_mesh_faces)
            vis.robot_dict["robot"].update_cfg(rest_qpos)
        vis.add_object(args.mesh_name, mesh, pose_robot, opacity=1.0,
                       max_mesh_faces=max_mesh_faces)
        if grasp_scene_only:
            vis.server.gui.add_markdown(
                "## ⚠️ EXECUTABLE PLANNING FAILED\n"
                "FoundationPose object + packed final grasp. The HOME xArm is "
                "shown only as a robot-coordinate reference. "
                "No arm trajectory exists; real-robot execution is disabled."
            )
            print("[viser] showing HOME xArm reference + FoundationPose final grasp")
        elif pipeline_result is not None:
            diagnostic_only = bool(pipeline_result.get("diagnostic_only", False))
            traj_name = ("COLLISION_IGNORED_DIAGNOSTIC_ONLY"
                         if diagnostic_only else "h2r_pipeline")
            if diagnostic_only:
                reason = pipeline_result.get(
                    "diagnostic_reason", "Trajectory ignores all collisions."
                )
                vis.server.gui.add_markdown(
                    "## ⚠️ DIAGNOSTIC ONLY\n"
                    f"{reason} Diagnostic playback only; real execution is disabled."
                )
            elif not pipeline_result.get("viser_ik_complete", True):
                vis.server.gui.add_markdown(
                    "## CARTESIAN EXECUTION / INCOMPLETE VISER IK\n"
                    "The displayed xArm joints contain frozen best-effort IK sections. "
                    "Real execution streams the link6 pose path, but full reachability "
                    "and collision clearance were not proven."
                )
            elif not pipeline_result.get("collision_verified", True):
                vis.server.gui.add_markdown(
                    "## xArm SDK IK PREFLIGHT\n"
                    "The displayed arm uses xArm SDK inverse kinematics and passed "
                    "IK/joint-limit checks. Collision clearance is not checked by "
                    "the xArm SDK."
                )
            vis.add_traj(traj_name, robot_traj={"robot": pipeline_result["full_traj"]})
            print(f"[pipe] added {pipeline_result['full_traj'].shape[0]} frames to "
                  f"viewer; Playback ranges {pipeline_result['range_str']}.")
        if _real_execution_allowed(pipeline_result):
            print("Viser running. Press Ctrl+C to continue to the REAL-execution prompt.")
        else:
            print("Viser running in diagnostic-only mode. Press Ctrl+C to exit; "
                  "REAL execution is disabled.")
        try:
            while True:
                vis.update()
        except KeyboardInterrupt:
            pass

    if pipeline_result is not None and not _real_execution_allowed(pipeline_result):
        print("[exec] DISABLED: diagnostic fallback output is Viser-only and cannot "
              "be sent to the real robot.")
    elif _real_execution_allowed(pipeline_result):
        if args.yes:
            ans = "y"
        else:
            ans = ""
            while ans not in ("y", "n"):
                try:
                    unverified = []
                    if not pipeline_result.get("viser_ik_complete", True):
                        unverified.append("IK")
                    if not pipeline_result.get("collision_verified", True):
                        unverified.append("COLLISION")
                    verification_note = (
                        f" [{' / '.join(unverified)} UNVERIFIED]"
                        if unverified else ""
                    )
                    ans = input(
                        f"\n[exec] Execute {pipeline_result['full_traj'].shape[0]}-frame "
                        f"trajectory on REAL robot in {args.arm_execution_mode} mode "
                        f"@ {args.rate_hz} Hz "
                        f"(record_video={args.record_video}){verification_note}? [y/n] "
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
