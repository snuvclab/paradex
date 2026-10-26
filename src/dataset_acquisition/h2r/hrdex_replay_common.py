"""Shared frame and staged-motion helpers for HRDex real-robot replay.

The helpers in this module deliberately stay independent of CuRobo and ROS so
that frame and trajectory contracts can be unit-tested on a non-robot machine.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resolve_scene_arg(
    result_root: str,
    mesh_name: str,
    result_filename: str,
    scene: Optional[int],
) -> int:
    """Resolve an omitted scene from the unique packed-result directory."""

    if scene is not None:
        return int(scene)

    object_dir = Path(result_root).expanduser() / mesh_name
    if not object_dir.is_dir():
        raise FileNotFoundError(
            f"cannot auto-select --scene: object result directory not found: {object_dir}"
        )

    candidates: list[int] = []
    for child in object_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            scene_id = int(child.name)
        except ValueError:
            continue
        if (child / result_filename).is_file():
            candidates.append(scene_id)
    candidates.sort()

    if not candidates:
        raise FileNotFoundError(
            f"cannot auto-select --scene: no <scene>/{result_filename} under {object_dir}"
        )
    if len(candidates) != 1:
        raise ValueError(
            "cannot auto-select --scene because multiple packed scenes exist for "
            f"{mesh_name}: {candidates}; pass --scene explicitly"
        )
    return candidates[0]


def curobo_active_joint_names(robot_cfg: dict[str, Any]) -> list[str]:
    """Return CuRobo optimization joints after ``lock_joints`` is applied."""

    kinematics = robot_cfg.get("robot_cfg", {}).get("kinematics", {})
    cspace_names = [str(name) for name in kinematics.get("cspace", {}).get("joint_names", [])]
    locked_names = {str(name) for name in (kinematics.get("lock_joints") or {})}
    active_names = [name for name in cspace_names if name not in locked_names]
    if not active_names:
        raise ValueError("CuRobo config has no active cspace joints")
    return active_names


def select_curobo_active_qpos(
    qpos: np.ndarray,
    robot_cfg: dict[str, Any],
    active_joint_names: list[str],
) -> np.ndarray:
    """Project a full cspace vector onto CuRobo's active joint order."""

    value = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if value.shape[0] == len(active_joint_names):
        return value.copy()
    cspace_names = [
        str(name)
        for name in robot_cfg.get("robot_cfg", {})
        .get("kinematics", {})
        .get("cspace", {})
        .get("joint_names", [])
    ]
    if value.shape[0] != len(cspace_names):
        raise ValueError(
            "cannot align CuRobo seed: "
            f"qpos={value.shape[0]}, cspace={len(cspace_names)}, active={len(active_joint_names)}"
        )
    by_name = dict(zip(cspace_names, value.tolist()))
    missing = [name for name in active_joint_names if name not in by_name]
    if missing:
        raise ValueError(f"CuRobo active joints missing from cspace: {missing}")
    return np.asarray([by_name[name] for name in active_joint_names], dtype=np.float32)


def sparse_waypoint_indices(count: int, stride: int) -> np.ndarray:
    """Return preview sample indices, always including both trajectory ends."""

    count = int(count)
    stride = int(stride)
    if count < 0:
        raise ValueError("waypoint count must be non-negative")
    if stride <= 0:
        raise ValueError("waypoint stride must be positive")
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    indices = np.arange(0, count, stride, dtype=np.int64)
    if indices[-1] != count - 1:
        indices = np.append(indices, count - 1)
    return indices


def interpolate_joint_anchors(
    anchor_indices: np.ndarray,
    anchor_qpos: np.ndarray,
    count: int,
) -> np.ndarray:
    """Interpolate sparse IK anchors into a dense Viser joint trajectory."""

    indices = np.asarray(anchor_indices, dtype=np.int64).reshape(-1)
    anchors = np.asarray(anchor_qpos, dtype=np.float32)
    count = int(count)
    if anchors.ndim != 2 or len(indices) != len(anchors):
        raise ValueError("anchor indices and qpos must have matching lengths")
    if count == 0:
        return np.zeros((0, anchors.shape[1]), dtype=np.float32)
    if len(indices) == 0 or indices[0] != 0 or indices[-1] != count - 1:
        raise ValueError("joint anchors must include first and last waypoints")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("joint anchor indices must be strictly increasing")
    dense_index = np.arange(count, dtype=np.float64)
    dense = np.empty((count, anchors.shape[1]), dtype=np.float32)
    for joint in range(anchors.shape[1]):
        dense[:, joint] = np.interp(dense_index, indices, anchors[:, joint])
    return dense


def trajectory_phase(index: int, pipeline_result: dict[str, Any]) -> str:
    """Return the named trajectory phase containing ``index``."""

    index = int(index)
    n_approach = int(pipeline_result.get("n_approach", 0))
    if index < n_approach:
        ends = pipeline_result.get("stage_a_ends") or {}
        if index < int(ends.get("raise", 0)):
            return "approach_raise"
        if index < int(ends.get("planar_rotate", 0)):
            return "approach_planar_rotate"
        return "approach_descend"

    cursor = n_approach
    for name, key in (
        ("hand_interp", "n_hand_interp"),
        ("replay", "n_replay"),
        ("grasp_hold", "n_grasp_hold"),
        ("lift", "n_lift"),
        ("release", "n_release"),
    ):
        cursor += int(pipeline_result.get(key, 0))
        if index < cursor:
            return name
    return "out_of_range"


def end_effector_poses_to_command_link(
    robot_module: Any,
    end_effector_poses: np.ndarray,
    reference_qpos: np.ndarray,
    *,
    end_effector_link: str,
    command_link: str = "link6",
) -> np.ndarray:
    """Convert desired end-effector poses to xArm command-link poses.

    The end effector is rigidly attached to ``command_link``. Its fixed mount
    transform is measured once from URDF FK, so no IK is required.
    """

    targets = np.asarray(end_effector_poses, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[1:] != (4, 4):
        raise ValueError("end-effector trajectory must have shape (N,4,4)")
    reference_fk = robot_module.forward_kinematics(
        np.asarray(reference_qpos, dtype=np.float64)
    )
    robot_from_command = np.asarray(reference_fk[command_link], dtype=np.float64)
    robot_from_eef = np.asarray(reference_fk[end_effector_link], dtype=np.float64)
    command_from_eef = np.linalg.inv(robot_from_command) @ robot_from_eef
    return targets @ np.linalg.inv(command_from_eef)


def canonical_frame_id(frame: str) -> str:
    """Normalize historical object-frame labels.

    ``canonical`` and ``viser`` name the same frame: the raw mesh after
    ``viser_align`` has been applied.  Keeping this alias here prevents the
    transform from accidentally being applied twice.
    """

    value = str(frame).strip().lower()
    if value == "viser":
        value = "canonical"
    if value not in {"original", "canonical"}:
        raise ValueError(f"unsupported object frame: {frame!r}")
    return value


def pack_frame_to_pose_frame(pack: Any, pose_mesh_frame: str) -> np.ndarray:
    """Return the transform that re-expresses packed wrists in the pose frame."""

    pack_frame = canonical_frame_id(
        str(pack["wrist_frame_id"])
        if "wrist_frame_id" in pack.files
        else "canonical"
    )
    pose_frame = canonical_frame_id(pose_mesh_frame)
    if pack_frame == pose_frame:
        return np.eye(4, dtype=np.float64)
    if "viser_align" not in pack.files:
        raise ValueError(
            "pack lacks viser_align; cannot convert wrist frame "
            f"{pack_frame} -> {pose_frame}"
        )
    align = np.asarray(pack["viser_align"], dtype=np.float64)
    if align.shape != (4, 4) or not np.all(np.isfinite(align)):
        raise ValueError("pack viser_align must be a finite 4x4 transform")
    # align = T_canonical_from_original.
    return np.linalg.inv(align) if pose_frame == "original" else align


def resolve_pose_mesh(mesh_root_dir: str, mesh_name: str, pose_mesh_frame: str) -> str:
    """Resolve the mesh whose coordinates match the Object6D pose frame."""

    root = Path(mesh_root_dir).expanduser()
    if root.suffix.lower() == ".obj":
        return str(root)
    suffix = "_viser" if canonical_frame_id(pose_mesh_frame) == "canonical" else ""
    candidates = (
        root / mesh_name / f"{mesh_name}{suffix}.obj",
        root / f"{mesh_name}{suffix}.obj",
    )
    return str(next((path for path in candidates if path.exists()), candidates[0]))


def _as_pose(pose: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be a finite 4x4 transform")
    return value


def pose_segment(
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    linear_speed_mps: float,
    angular_speed_rps: float,
    rate_hz: float,
    min_seconds: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Interpolate one bounded-speed Cartesian segment, including both ends."""

    start = _as_pose(start_pose, "segment start")
    target = _as_pose(target_pose, "segment target")
    if linear_speed_mps <= 0 or angular_speed_rps <= 0 or rate_hz <= 0:
        raise ValueError("Cartesian speeds and rate must be positive")
    translation = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    rotation = float(
        Rotation.from_matrix(start[:3, :3].T @ target[:3, :3]).magnitude()
    )
    seconds = max(
        float(min_seconds),
        1.5 * translation / float(linear_speed_mps),
        1.5 * rotation / float(angular_speed_rps),
        1.0 / float(rate_hz),
    )
    frame_count = max(2, int(np.ceil(seconds * rate_hz)) + 1)
    alpha = np.linspace(0.0, 1.0, frame_count, dtype=np.float64)
    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(np.stack((start[:3, :3], target[:3, :3]))),
    )(smooth).as_matrix()
    poses = np.tile(np.eye(4, dtype=np.float64), (frame_count, 1, 1))
    poses[:, :3, :3] = rotations
    poses[:, :3, 3] = (
        (1.0 - smooth[:, None]) * start[:3, 3]
        + smooth[:, None] * target[:3, 3]
    )
    poses[0] = start
    poses[-1] = target
    return poses, seconds


def staged_home_to_pregrasp(
    home_pose: np.ndarray,
    pregrasp_pose: np.ndarray,
    *,
    rate_hz: float,
    planar_speed_mps: float,
    angular_speed_rps: float,
    descent_speed_mps: float,
    raise_speed_mps: Optional[float] = None,
    min_hover_clearance_m: float,
    table_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> tuple[np.ndarray, dict[str, int]]:
    """Move at safe height while rotating, then descend along the table normal.

    The first segment raises only when HOME is lower than the requested hover
    clearance.  The second segment performs the floor-parallel translation and
    completes wrist rotation.  The final segment changes only table-normal
    height, with target orientation held fixed.
    """

    home = _as_pose(home_pose, "HOME pose")
    target = _as_pose(pregrasp_pose, "pregrasp pose")
    raise_speed = (
        float(descent_speed_mps)
        if raise_speed_mps is None
        else float(raise_speed_mps)
    )
    if raise_speed <= 0.0 or not np.isfinite(raise_speed):
        raise ValueError("raise speed must be finite and positive")
    normal = np.asarray(table_normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        raise ValueError("table normal must be non-zero")
    normal /= norm
    home_h = float(normal @ home[:3, 3])
    target_h = float(normal @ target[:3, 3])
    hover_h = max(home_h, target_h + float(min_hover_clearance_m))

    raised = home.copy()
    raised[:3, 3] += (hover_h - home_h) * normal
    hover = target.copy()
    hover[:3, 3] += (hover_h - target_h) * normal

    pieces: list[np.ndarray] = []
    phase_ends: dict[str, int] = {}
    cursor = 0
    if hover_h - home_h > 1e-6:
        segment, _ = pose_segment(
            home,
            raised,
            linear_speed_mps=raise_speed,
            angular_speed_rps=angular_speed_rps,
            rate_hz=rate_hz,
        )
        pieces.append(segment)
        cursor += len(segment)
        phase_ends["raise"] = cursor
    else:
        pieces.append(home[None])
        cursor = 1
        phase_ends["raise"] = cursor

    segment, _ = pose_segment(
        raised,
        hover,
        linear_speed_mps=planar_speed_mps,
        angular_speed_rps=angular_speed_rps,
        rate_hz=rate_hz,
    )
    pieces.append(segment[1:])
    cursor += len(segment) - 1
    phase_ends["planar_rotate"] = cursor

    segment, _ = pose_segment(
        hover,
        target,
        linear_speed_mps=descent_speed_mps,
        angular_speed_rps=angular_speed_rps,
        rate_hz=rate_hz,
    )
    pieces.append(segment[1:])
    cursor += len(segment) - 1
    phase_ends["descend"] = cursor
    return np.concatenate(pieces, axis=0), phase_ends


def arm_link_pose_trajectory(
    robot_module: Any,
    full_trajectory: np.ndarray,
    *,
    link_name: str = "link6",
) -> np.ndarray:
    """Convert a preview joint trajectory into Cartesian commands for xArm."""

    trajectory = np.asarray(full_trajectory, dtype=np.float64)
    poses = np.empty((len(trajectory), 4, 4), dtype=np.float64)
    for index, qpos in enumerate(trajectory):
        poses[index] = robot_module.forward_kinematics(qpos)[link_name]
    return poses


def unwrap_xarm_joints(
    qpos: np.ndarray,
    joint_names: list[str],
    target_names: tuple[str, ...] = ("joint1", "joint4", "joint6"),
) -> np.ndarray:
    """Choose the nearest 2-pi-equivalent value for continuous xArm joints."""

    out = np.asarray(qpos, dtype=np.float64).copy()
    for name in target_names:
        if name not in joint_names:
            continue
        joint = joint_names.index(name)
        for index in range(1, len(out)):
            out[index, joint] -= 2.0 * np.pi * round(
                (out[index, joint] - out[index - 1, joint]) / (2.0 * np.pi)
            )
    return out.astype(np.asarray(qpos).dtype, copy=False)


def assert_bounded_pose_steps(
    poses: np.ndarray,
    *,
    rate_hz: float,
    max_linear_speed_mps: float,
    max_angular_speed_rps: float,
    slack: float = 1.05,
    step_periods: Optional[np.ndarray] = None,
) -> None:
    """Reject a Cartesian stream whose adjacent commands exceed its limits."""

    value = np.asarray(poses, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4):
        raise ValueError("pose trajectory must have shape (N,4,4)")
    if len(value) < 2:
        return
    if step_periods is None:
        dt = np.full(len(value) - 1, 1.0 / float(rate_hz), dtype=np.float64)
    else:
        periods = np.asarray(step_periods, dtype=np.float64).reshape(-1)
        if len(periods) != len(value) or np.any(periods <= 0):
            raise ValueError("pose trajectory periods must be positive and match poses")
        dt = periods[:-1]
    linear = np.linalg.norm(np.diff(value[:, :3, 3], axis=0), axis=1) / dt
    angular = np.array(
        [
            Rotation.from_matrix(value[i, :3, :3].T @ value[i + 1, :3, :3]).magnitude()
            / dt[i]
            for i in range(len(value) - 1)
        ]
    )
    if float(np.max(linear)) > max_linear_speed_mps * slack:
        raise ValueError(
            f"Cartesian path exceeds linear speed limit: {linear.max():.3f} m/s"
        )
    if float(np.max(angular)) > max_angular_speed_rps * slack:
        raise ValueError(
            f"Cartesian path exceeds angular speed limit: {angular.max():.3f} rad/s"
        )


def assert_bounded_joint_steps(
    qpos: np.ndarray,
    periods: np.ndarray,
    *,
    max_joint_speed_deg_s: float,
) -> None:
    """Reject branch flips or trajectories that exceed the execution limit."""

    values = np.asarray(qpos, dtype=np.float64)
    dt = np.asarray(periods, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or len(values) != len(dt):
        raise ValueError("joint trajectory and periods must have the same length")
    if len(values) < 2:
        return
    speed = np.abs(np.diff(values, axis=0)) / dt[:-1, None]
    worst = float(np.max(np.rad2deg(speed)))
    if worst > float(max_joint_speed_deg_s):
        index, joint = np.unravel_index(np.argmax(speed), speed.shape)
        raise ValueError(
            "planned arm motion exceeds joint-speed limit: "
            f"{worst:.1f} deg/s at transition {index}->{index + 1}, joint {joint + 1}"
        )


def retime_step_periods(
    qpos: Optional[np.ndarray],
    poses: np.ndarray,
    periods: np.ndarray,
    *,
    max_joint_speed_deg_s: Optional[float],
    max_linear_speed_mps: float,
    max_angular_speed_rps: float,
) -> tuple[np.ndarray, int, float]:
    """Slow only transitions that exceed joint or Cartesian speed limits."""

    cartesian = np.asarray(poses, dtype=np.float64)
    original = np.asarray(periods, dtype=np.float64).reshape(-1)
    if cartesian.ndim != 3 or cartesian.shape[1:] != (4, 4):
        raise ValueError("retiming expects poses with shape (N,4,4)")
    if len(original) != len(cartesian) or np.any(original <= 0):
        raise ValueError("retiming periods must be positive and match trajectory length")
    cartesian_limits = (float(max_linear_speed_mps), float(max_angular_speed_rps))
    if not all(np.isfinite(limit) and limit > 0 for limit in cartesian_limits):
        raise ValueError("retiming speed limits must be finite and positive")
    joints = None if qpos is None else np.asarray(qpos, dtype=np.float64)
    if joints is not None:
        if joints.ndim != 2 or len(joints) != len(cartesian):
            raise ValueError("retiming qpos must have shape (N,D) and match poses")
        if max_joint_speed_deg_s is None:
            raise ValueError("joint speed limit is required when qpos is provided")
        joint_limit = float(max_joint_speed_deg_s)
        if not np.isfinite(joint_limit) or joint_limit <= 0:
            raise ValueError("retiming joint speed limit must be finite and positive")
    if len(cartesian) < 2:
        return original.copy(), 0, 1.0

    joint_seconds = np.zeros(len(cartesian) - 1, dtype=np.float64)
    if joints is not None:
        joint_delta = np.max(np.abs(np.diff(joints, axis=0)), axis=1)
        joint_seconds = joint_delta / np.deg2rad(joint_limit)
    linear_seconds = (
        np.linalg.norm(np.diff(cartesian[:, :3, 3], axis=0), axis=1)
        / cartesian_limits[0]
    )
    angular_seconds = np.asarray(
        [
            Rotation.from_matrix(
                cartesian[i, :3, :3].T @ cartesian[i + 1, :3, :3]
            ).magnitude()
            / cartesian_limits[1]
            for i in range(len(cartesian) - 1)
        ],
        dtype=np.float64,
    )
    required = np.maximum.reduce((joint_seconds, linear_seconds, angular_seconds))
    retimed = original.copy()
    before = retimed[:-1].copy()
    retimed[:-1] = np.maximum(before, required * 1.001)
    changed = int(np.count_nonzero(retimed[:-1] > before + 1e-9))
    max_stretch = float(np.max(retimed[:-1] / before))
    return retimed, changed, max_stretch


def densify_timed_commands(
    commands: np.ndarray,
    periods: np.ndarray,
    *,
    command_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly subdivide timed servo targets at a fixed maximum interval.

    ``periods[i]`` is the time from source target ``i`` to ``i + 1``. Merely
    increasing that period is not enough for a position-servo API: the next
    target still arrives as one large step and the hardware may traverse it at
    its internal maximum speed. This helper emits intermediate targets so the
    retimed joint velocity is represented by the actual command stream.

    The returned source-index array maps every dense command to the original
    trajectory frame for phase-aware logging and watchdog errors.
    """

    values = np.asarray(commands, dtype=np.float64)
    dt = np.asarray(periods, dtype=np.float64).reshape(-1)
    rate = float(command_rate_hz)
    if values.ndim != 2 or len(values) != len(dt) or len(values) == 0:
        raise ValueError("commands must be a non-empty (N,D) array matching periods")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(dt))
        or np.any(dt <= 0)
    ):
        raise ValueError("commands and periods must be finite with positive periods")
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("command_rate_hz must be finite and positive")

    dense_values = []
    dense_periods = []
    source_indices = []
    max_period = 1.0 / rate
    for index in range(len(values) - 1):
        subdivisions = max(1, int(np.ceil(dt[index] / max_period)))
        step_period = float(dt[index]) / subdivisions
        for step in range(subdivisions):
            alpha = float(step) / subdivisions
            dense_values.append(
                (1.0 - alpha) * values[index] + alpha * values[index + 1]
            )
            dense_periods.append(step_period)
            source_indices.append(index)
    dense_values.append(values[-1].copy())
    dense_periods.append(float(dt[-1]))
    source_indices.append(len(values) - 1)
    return (
        np.asarray(dense_values, dtype=np.float64),
        np.asarray(dense_periods, dtype=np.float64),
        np.asarray(source_indices, dtype=np.int64),
    )


def assert_arm_joint_limits(
    robot_module: Any,
    full_trajectory: np.ndarray,
    *,
    margin_rad: float = 0.02,
) -> None:
    """Reject any xArm preview sample at or beyond a configured joint limit."""

    trajectory = np.asarray(full_trajectory, dtype=np.float64)
    names = list(robot_module.get_joint_names())
    limits = robot_module.get_joint_limits()
    for name in (f"joint{i}" for i in range(1, 7)):
        if name not in names or name not in limits:
            raise ValueError(f"missing xArm joint limit for {name}")
        index = names.index(name)
        lower, upper = (float(value) for value in limits[name])
        low = float(np.min(trajectory[:, index]))
        high = float(np.max(trajectory[:, index]))
        if low < lower + margin_rad or high > upper - margin_rad:
            raise ValueError(
                f"{name} violates {margin_rad:.3f} rad joint-limit margin: "
                f"path=[{low:.3f}, {high:.3f}], limits=[{lower:.3f}, {upper:.3f}]"
            )


def move_xarm_home_safely(
    xarm: Any,
    target_rad: np.ndarray,
    *,
    max_joint_speed_deg_s: float,
    stream_hz: float,
) -> None:
    """Move from measured xArm joints to HOME without a one-shot command."""

    current = np.asarray(xarm.get_data().get("qpos"), dtype=np.float64)
    target = np.asarray(target_rad, dtype=np.float64)
    if current.shape != (6,) or not np.all(np.isfinite(current)):
        raise RuntimeError("xArm joint feedback unavailable; refusing HOME preposition")
    speed_rad_s = np.deg2rad(float(max_joint_speed_deg_s))
    if speed_rad_s <= 0 or stream_hz <= 0:
        raise ValueError("HOME joint speed and stream rate must be positive")
    max_delta = float(np.max(np.abs(target - current)))
    if max_delta <= 1e-6:
        return
    duration = max_delta / speed_rad_s
    frame_count = max(2, int(np.ceil(duration * stream_hz)) + 1)
    positions = np.linspace(current, target, frame_count, dtype=np.float64)
    times = np.linspace(0.0, duration, frame_count, dtype=np.float64)
    print(
        f"[exec] bounded HOME preposition: {duration:.2f}s, {frame_count} frames, "
        f"<= {max_joint_speed_deg_s:.1f} deg/s"
    )
    xarm.move_joint_trajectory(positions, times)


class XArmJointWatchdog:
    """Fail closed on missing, over-speed, or near-limit live xArm feedback."""

    def __init__(
        self,
        robot_module: Any,
        *,
        max_speed_deg_s: float,
        limit_margin_rad: float,
        min_sample_interval_s: float = 1.0 / 60.0,
        speed_tolerance_ratio: float = 0.10,
        consecutive_overspeed_samples: int = 2,
    ) -> None:
        self.max_speed_rad_s = np.deg2rad(float(max_speed_deg_s))
        self.limit_margin_rad = float(limit_margin_rad)
        self.min_sample_interval_s = float(min_sample_interval_s)
        self.speed_tolerance_ratio = float(speed_tolerance_ratio)
        self.consecutive_overspeed_samples = int(consecutive_overspeed_samples)
        if not np.isfinite(self.min_sample_interval_s) or self.min_sample_interval_s < 0.0:
            raise ValueError("watchdog sample interval must be finite and non-negative")
        if not np.isfinite(self.speed_tolerance_ratio) or self.speed_tolerance_ratio < 0.0:
            raise ValueError("watchdog speed tolerance must be finite and non-negative")
        if self.consecutive_overspeed_samples < 1:
            raise ValueError("watchdog consecutive overspeed samples must be positive")
        names = list(robot_module.get_joint_names())
        limits = robot_module.get_joint_limits()
        self.limits = []
        for name in (f"joint{i}" for i in range(1, 7)):
            if name not in names or name not in limits:
                raise ValueError(f"missing xArm joint limit for {name}")
            self.limits.append(tuple(float(value) for value in limits[name]))
        self.previous: Optional[np.ndarray] = None
        self.previous_time: Optional[float] = None
        self._overspeed_counts = np.zeros(6, dtype=np.int64)

    def check(self, xarm: Any, *, context: Optional[str] = None) -> None:
        data = xarm.get_data()
        qpos = np.asarray(data.get("qpos"), dtype=np.float64)
        context_suffix = f" ({context})" if context else ""
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise RuntimeError(f"xArm joint feedback became unavailable{context_suffix}")
        for index, (value, bounds) in enumerate(zip(qpos, self.limits), start=1):
            lower, upper = bounds
            if value < lower + self.limit_margin_rad or value > upper - self.limit_margin_rad:
                raise RuntimeError(
                    f"xArm joint{index} entered the limit margin: {value:.3f} rad "
                    f"not in [{lower + self.limit_margin_rad:.3f}, "
                    f"{upper - self.limit_margin_rad:.3f}]{context_suffix}"
                )
        state_time = data.get("state_monotonic_time")
        now = (
            float(state_time)
            if state_time is not None and np.isfinite(float(state_time))
            else time.monotonic()
        )
        if self.previous is not None and self.previous_time is not None:
            dt = now - self.previous_time
            # No new ROS joint-state sample: do not move the timing baseline.
            if dt <= 0.0:
                return
            # Calls immediately adjacent to initialization or command dispatch
            # can be only a few milliseconds apart. Dividing encoder
            # quantization by that tiny dt creates false 60+ deg/s spikes.
            # Keep the previous baseline until one meaningful sample period has
            # elapsed; the next check still measures the aggregate motion.
            if dt < self.min_sample_interval_s:
                return
            speed = np.abs(qpos - self.previous) / dt
            trip_speed = self.max_speed_rad_s * (1.0 + self.speed_tolerance_ratio)
            over = speed > trip_speed
            self._overspeed_counts[over] += 1
            self._overspeed_counts[~over] = 0
            if int(np.max(self._overspeed_counts)) >= self.consecutive_overspeed_samples:
                joint_index = int(np.argmax(self._overspeed_counts))
                raise RuntimeError(
                    f"xArm joint{joint_index + 1} feedback sustained watchdog overspeed: "
                    f"{np.rad2deg(speed[joint_index]):.1f} deg/s "
                    f"(trip>{np.rad2deg(trip_speed):.1f} deg/s for "
                    f"{self.consecutive_overspeed_samples} samples){context_suffix}"
                )
        self.previous = qpos.copy()
        self.previous_time = now


def validate_motion_args(args: Any) -> None:
    """Validate the shared HRDex motion-safety command-line contract."""

    positive = (
        "rate_hz",
        "replay_rate_hz",
        "stage_a_linear_speed_mps",
        "stage_a_angular_speed_rps",
        "stage_a_raise_speed_mps",
        "stage_a_descent_speed_mps",
        "home_max_joint_speed_deg_s",
        "return_home_max_joint_speed_deg_s",
        "home_preposition_hz",
        "arm_servo_rate_hz",
        "max_planned_joint_speed_deg_s",
        "max_measured_joint_speed_deg_s",
        "max_cartesian_linear_speed_mps",
        "max_cartesian_angular_speed_rps",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} must be positive")
    for name in (
        "stage_a_hover_clearance_m",
        "live_joint_limit_margin_rad",
        "post_lift_hold_seconds",
        "release_steps",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    try:
        table_normal = np.asarray(
            [float(value) for value in str(args.table_normal_xyz).split(",")],
            dtype=np.float64,
        )
    except ValueError as exc:
        raise ValueError("--table_normal_xyz must contain three comma-separated numbers") from exc
    if (
        table_normal.shape != (3,)
        or not np.all(np.isfinite(table_normal))
        or float(np.linalg.norm(table_normal)) < 1e-8
    ):
        raise ValueError("--table_normal_xyz must be a finite, non-zero 3-vector")
