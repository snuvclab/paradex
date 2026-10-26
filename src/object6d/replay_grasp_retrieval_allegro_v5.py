#!/usr/bin/env python3
"""Replay an object-relative Allegro V5 grasp with a cuRobo approach.

The retrieval files use ``T_object_from_wrist``. Their companion metadata, or
the validated ``allegro_old_dexonomy_trajectory`` dataset contract, defines the
stored wrist frame as ``palm_link``. A live first-frame Object6D estimate is
therefore used to reconstruct the requested robot-frame palm trajectory as::

    T_robot_from_palm[t] = T_robot_from_object @ T_object_from_palm[t]

Only the approach to frame zero is required to be collision-free against the
object and the configured five-sided rectangular frame.  The object is not
treated as a forbidden obstacle during grasp replay because contact is the
purpose of that phase.  Real motion is disabled unless ``--execute`` is given,
cuRobo succeeded, and the operator confirms execution with ``PLAY`` or ``y``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paradex.utils.path import shared_dir


DEFAULT_DATASET_ROOT = (
    Path(shared_dir)
    / "grasp_retrieval_meshes_20260915_v1"
    / "allegro_old_dexonomy_trajectory"
)
DEFAULT_CUROBO_CFG = (
    PROJECT_ROOT
    / "rsc/curobo/content/configs/robot/xarm_allegro_v5.yml"
)
DEFAULT_ROBOT_URDF = PROJECT_ROOT / "rsc/robot/xarm_allegro_v5.urdf"
DEFAULT_HAND_URDF = (
    PROJECT_ROOT / "rsc/robot/allegro_v5/allegro_right_A.urdf"
)
DEFAULT_COLLISION_SPHERES = (
    PROJECT_ROOT / "rsc/curobo/content/configs/robot/spheres/xarm_allegro_v5.yml"
)
DEFAULT_OBJECT6D_RPC_ADDR = "tcp://192.168.0.13:5570"
DEFAULT_VISER_ADDR = "tcp://localhost:5572"
ACTION_NAMES = ("ours", "retarget", "retarget_opt")
CONTINUOUS_Y_SYMMETRY_OBJECTS = frozenset(
    {"yellow_wash_tub", "grey_pot", "smallbowl"}
)
POINT_SYMMETRY_OBJECTS = frozenset({"pingpong"})
DISCRETE_VERTICAL_SYMMETRY_OBJECTS = frozenset(
    {
        "fruit_cutter_green",
        "pastel_light_green_bowl",
        "lilac_food_container",
        "spicemill",
    }
)
HALF_TURN_VERTICAL_SYMMETRY_OBJECTS = frozenset({"wood_dish_rack"})
ALLEGRO_JOINT_NAMES = tuple(f"joint_{index}_0" for index in range(16))
ALLEGRO_DRIVER_ABS_LIMIT_RAD = 2.1
# The xArm base is 15 cm behind the open (-x) edge. The enclosure spans
# x=0.15..0.92 m, with y as width and z as height: 77 cm deep, 150 cm wide,
# and 65 cm physically tall. The measured floor is z=0.016 m; use z=0.020 m
# as the conservative collision surface while retaining the physical ceiling
# at z=0.666 m.
FRAME_COLLISION_FLOOR_Z_M = 0.020
FRAME_PHYSICAL_CEILING_Z_M = 0.666
FRAME_INNER_SIZE_XYZ_M = np.array(
    [0.77, 1.50, FRAME_PHYSICAL_CEILING_Z_M - FRAME_COLLISION_FLOOR_Z_M],
    dtype=np.float64,
)
FRAME_CENTER_XYZ_M = np.array(
    [0.535, 0.0, (FRAME_COLLISION_FLOOR_Z_M + FRAME_PHYSICAL_CEILING_Z_M) / 2.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RetrievalTrajectory:
    path: Path
    times_s: np.ndarray
    object_from_palm: np.ndarray
    hand_qpos: np.ndarray


def _finite_transform(value: Any, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape == (3, 4):
        transform = np.vstack((transform, [0.0, 0.0, 0.0, 1.0]))
    if (
        transform.shape != (4, 4)
        or not np.all(np.isfinite(transform))
        or not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7)
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
        )
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5)
    ):
        raise ValueError(f"{label} must be a finite rigid homogeneous transform")
    return transform


def _joint_vector(text: str, label: str, count: int) -> np.ndarray:
    value = np.asarray([float(item) for item in text.split(",")], dtype=np.float64)
    if value.shape != (count,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must contain {count} finite comma-separated values")
    return value


def resolve_scene_root(dataset_root: Path, object_name: str, scene: int | None) -> Path:
    object_root = Path(dataset_root).expanduser() / object_name
    if scene is not None:
        result = object_root / f"scene_{scene}"
        if not result.is_dir():
            raise FileNotFoundError(f"scene directory not found: {result}")
        return result
    candidates = sorted(path for path in object_root.glob("scene_*") if path.is_dir())
    if len(candidates) != 1:
        raise ValueError(
            f"expected one scene under {object_root}, found {len(candidates)}; pass --scene"
        )
    return candidates[0]


def load_retrieval_trajectory(scene_root: Path, action: str) -> RetrievalTrajectory:
    if action not in ACTION_NAMES:
        raise ValueError(f"unsupported action {action!r}")
    path = scene_root / "evaluated_actions" / action / "approach_grasp_trajectory_object.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    is_dexonomy_trajectory = "allegro_old_dexonomy_trajectory" in path.parts
    with np.load(path, allow_pickle=False) as values:
        required = {"time_s", "T_object_from_wrist", "qpos", "joint_names"}
        missing = required.difference(values.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        times = np.asarray(values["time_s"], dtype=np.float64).reshape(-1)
        transforms = np.asarray(values["T_object_from_wrist"], dtype=np.float64)
        hand = np.asarray(values["qpos"], dtype=np.float64)
        names = [str(name) for name in values["joint_names"].reshape(-1)]
        redundant_positions = (
            np.asarray(values["wrist_pos_object"], dtype=np.float64)
            if "wrist_pos_object" in values.files else None
        )
        redundant_rotations = (
            np.asarray(values["wrist_rotmat_object"], dtype=np.float64)
            if "wrist_rotmat_object" in values.files else None
        )

    count = len(times)
    if count < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path}: time_s must be finite and strictly increasing")
    if transforms.shape != (count, 4, 4):
        raise ValueError(f"{path}: T_object_from_wrist must have shape ({count},4,4)")
    for index, transform in enumerate(transforms):
        _finite_transform(transform, f"{path}:T_object_from_wrist[{index}]")
    if hand.shape != (count, 16) or not np.all(np.isfinite(hand)):
        raise ValueError(f"{path}: qpos must be a finite ({count},16) array")
    over_limit = np.abs(hand) > ALLEGRO_DRIVER_ABS_LIMIT_RAD + 1e-9
    if np.any(over_limit):
        if not is_dexonomy_trajectory:
            raise ValueError(
                f"{path}: qpos exceeds the Allegro V5 driver command envelope "
                f"(+/-{ALLEGRO_DRIVER_ABS_LIMIT_RAD} rad)"
            )
        print(
            f"[load] WARNING: clipping {int(np.count_nonzero(over_limit))} Dexonomy "
            f"hand value(s) from max |q|={float(np.max(np.abs(hand))):.3f}rad to "
            f"+/-{ALLEGRO_DRIVER_ABS_LIMIT_RAD:.1f}rad"
        )
        hand = np.clip(
            hand, -ALLEGRO_DRIVER_ABS_LIMIT_RAD, ALLEGRO_DRIVER_ABS_LIMIT_RAD
        )
    if len(names) != 16 or set(names) != set(ALLEGRO_JOINT_NAMES):
        raise ValueError(f"{path}: joint_names must contain each Allegro V5 joint once")
    order = [names.index(name) for name in ALLEGRO_JOINT_NAMES]

    metadata_path = path.with_name("grasp_action_object.json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("wrist_frame") != "palm_link":
            raise ValueError(f"{metadata_path}: expected wrist_frame='palm_link'")
        if metadata.get("coordinate_frame") != "object mesh native frame":
            raise ValueError(f"{metadata_path}: unexpected object coordinate frame")
    else:
        if not is_dexonomy_trajectory:
            raise FileNotFoundError(
                f"metadata required to prove the transform convention: {metadata_path}"
            )
        if (
            redundant_positions is None
            or redundant_positions.shape != (count, 3)
            or not np.allclose(redundant_positions, transforms[:, :3, 3], atol=1e-7)
        ):
            raise ValueError(
                f"{path}: dexonomy wrist_pos_object does not match T_object_from_wrist"
            )
        if (
            redundant_rotations is None
            or redundant_rotations.shape != (count, 3, 3)
            or not np.allclose(redundant_rotations, transforms[:, :3, :3], atol=1e-7)
        ):
            raise ValueError(
                f"{path}: dexonomy wrist_rotmat_object does not match T_object_from_wrist"
            )

    return RetrievalTrajectory(
        path=path,
        times_s=times - times[0],
        object_from_palm=transforms.copy(),
        hand_qpos=hand[:, order].copy(),
    )


def reconstruct_palm_trajectory(
    robot_from_object: np.ndarray, object_from_palm: np.ndarray
) -> np.ndarray:
    object_pose = _finite_transform(robot_from_object, "robot_from_object")
    relative = np.asarray(object_from_palm, dtype=np.float64)
    if relative.ndim != 3 or relative.shape[1:] != (4, 4):
        raise ValueError("object_from_palm must have shape (N,4,4)")
    return object_pose[None] @ relative


def _rotation_about_object_center(
    axis: str, angle_rad: float, center_object: Sequence[float]
) -> np.ndarray:
    center = np.asarray(center_object, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("object symmetry center must contain three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler(axis, angle_rad).as_matrix()
    transform[:3, 3] = center - transform[:3, :3] @ center
    return transform


def align_axisymmetric_object_pose_to_palm_direction(
    robot_from_object: np.ndarray,
    object_from_palm: np.ndarray,
    robot_base_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    hand_sphere_centers_in_palm: np.ndarray | None = None,
    hand_sphere_radii: np.ndarray | None = None,
    minimum_floor_z: float | None = None,
    angle_step_deg: float = 1.0,
    candidate_angles_rad: Sequence[float] | None = None,
    symmetry_center_object: Sequence[float] = (0.0, 0.0, 0.0),
    symmetry_axis: str = "x",
) -> tuple[np.ndarray, float, float | None]:
    """Resolve continuous object-axis symmetry using the first palm direction.

    ``object_from_palm`` is never modified. Instead, an equivalent detected
    object pose is selected so the palm's +Z axis points as closely as possible
    from the xArm base toward the object.  In the Allegro V5 convention used by
    this repository, local -Z is the palm-to-wrist (robotward) direction.
    The returned angle is in radians.
    """

    object_pose = _finite_transform(robot_from_object, "robot_from_object")
    relative = np.asarray(object_from_palm, dtype=np.float64)
    if relative.ndim != 3 or relative.shape[1:] != (4, 4) or len(relative) == 0:
        raise ValueError("object_from_palm must have shape (N,4,4), N>=1")
    base = np.asarray(robot_base_xyz, dtype=np.float64)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError("robot_base_xyz must contain three finite values")

    center = np.asarray(symmetry_center_object, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("symmetry_center_object must contain three finite values")
    if symmetry_axis not in {"x", "y", "z"}:
        raise ValueError("symmetry_axis must be one of: x, y, z")
    object_center_robot = (object_pose @ np.asarray([*center, 1.0]))[:3]
    xarm_to_object = object_center_robot - base
    target_norm = np.linalg.norm(xarm_to_object)
    if target_norm < 1e-9:
        raise ValueError("object center coincides with the xArm base")
    target_direction = xarm_to_object / target_norm
    if not np.isfinite(angle_step_deg) or angle_step_deg <= 0.0:
        raise ValueError("angle step must be finite and positive")

    centers = None
    radii = None
    if minimum_floor_z is not None:
        centers = np.asarray(hand_sphere_centers_in_palm, dtype=np.float64)
        radii = np.asarray(hand_sphere_radii, dtype=np.float64)
        if (
            centers.ndim != 3
            or centers.shape[0] != len(relative)
            or centers.shape[2] != 3
            or radii.shape != (centers.shape[1],)
            or not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError("hand collision spheres do not match the palm trajectory")
        if not np.isfinite(minimum_floor_z):
            raise ValueError("floor height must be finite")

    if candidate_angles_rad is None:
        steps = max(1, int(math.ceil(360.0 / float(angle_step_deg))))
        candidate_angles = 2.0 * math.pi * np.arange(steps, dtype=np.float64) / steps
    else:
        candidate_angles = np.asarray(candidate_angles_rad, dtype=np.float64).reshape(-1)
        if len(candidate_angles) == 0 or not np.all(np.isfinite(candidate_angles)):
            raise ValueError("candidate symmetry angles must be non-empty and finite")

    selected_angle = None
    selected_minimum_z = None
    selected_score = -math.inf
    best_minimum_z = -math.inf
    for candidate in candidate_angles:
        object_axis_rotation = _rotation_about_object_center(
            symmetry_axis, candidate, center
        )
        candidate_object_pose = object_pose @ object_axis_rotation
        robot_from_palm = candidate_object_pose[None] @ relative
        candidate_minimum_z = None
        if centers is not None and radii is not None:
            bottoms = []
            for pose, frame_centers in zip(robot_from_palm, centers):
                world_centers = (
                    pose[:3, :3] @ frame_centers.T
                ).T + pose[:3, 3]
                bottoms.append(np.min(world_centers[:, 2] - radii))
            candidate_minimum_z = float(np.min(bottoms))
            best_minimum_z = max(best_minimum_z, candidate_minimum_z)
            if candidate_minimum_z < float(minimum_floor_z) - 1e-9:
                continue
        palm_direction = robot_from_palm[0, :3, 2]
        score = float(np.dot(palm_direction, target_direction))
        if score > selected_score:
            selected_angle = candidate
            selected_minimum_z = candidate_minimum_z
            selected_score = score

    if selected_angle is None:
        raise ValueError(
            f"no object-{symmetry_axis} rotation keeps the Allegro collision spheres above "
            f"z={float(minimum_floor_z):.3f}m; best minimum z={best_minimum_z:.3f}m"
        )
    angle = math.atan2(math.sin(selected_angle), math.cos(selected_angle))
    object_axis_rotation = _rotation_about_object_center(symmetry_axis, angle, center)
    return object_pose @ object_axis_rotation, angle, selected_minimum_z


def select_equivalent_object_y_pose(
    robot_from_object: np.ndarray,
    object_from_palm: np.ndarray,
    robot_base_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    hand_sphere_centers_in_palm: np.ndarray | None = None,
    hand_sphere_radii: np.ndarray | None = None,
    minimum_floor_z: float | None = None,
    symmetry_center_object: Sequence[float] = (0.0, 0.0, 0.0),
    symmetry_axis: str = "y",
    position_first: bool = False,
    prefer_positive_object_axis: str | None = None,
    prefer_positive_robot_axis: str | None = None,
    finger_extension_axis_in_palm: Sequence[float] | None = None,
    prefer_positive_finger_extension_robot_axis: str | None = None,
    candidate_angles_rad: Sequence[float] | None = None,
) -> tuple[np.ndarray, float, float | None]:
    """Choose an object-local axis symmetry pose at 0/90/180/270 degrees."""

    object_pose = _finite_transform(robot_from_object, "robot_from_object")
    relative = np.asarray(object_from_palm, dtype=np.float64)
    if relative.ndim != 3 or relative.shape[1:] != (4, 4) or len(relative) == 0:
        raise ValueError("object_from_palm must have shape (N,4,4), N>=1")
    base = np.asarray(robot_base_xyz, dtype=np.float64)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError("robot_base_xyz must contain three finite values")
    center = np.asarray(symmetry_center_object, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("symmetry_center_object must contain three finite values")
    if symmetry_axis not in {"x", "y", "z"}:
        raise ValueError("symmetry_axis must be one of: x, y, z")
    if prefer_positive_object_axis not in {None, "x", "y", "z"}:
        raise ValueError("prefer_positive_object_axis must be x, y, z, or None")
    if prefer_positive_robot_axis not in {None, "x", "y", "z"}:
        raise ValueError("prefer_positive_robot_axis must be x, y, z, or None")
    if prefer_positive_finger_extension_robot_axis not in {None, "x", "y", "z"}:
        raise ValueError(
            "prefer_positive_finger_extension_robot_axis must be x, y, z, or None"
        )
    finger_axis = None
    if finger_extension_axis_in_palm is not None:
        finger_axis = np.asarray(finger_extension_axis_in_palm, dtype=np.float64)
        if finger_axis.shape != (3,) or not np.all(np.isfinite(finger_axis)):
            raise ValueError("finger_extension_axis_in_palm must contain three finite values")
        norm = np.linalg.norm(finger_axis)
        if norm < 1e-9:
            raise ValueError("finger_extension_axis_in_palm must be non-zero")
        finger_axis /= norm
    elif prefer_positive_finger_extension_robot_axis is not None:
        raise ValueError(
            "finger_extension_axis_in_palm is required when preferring its robot axis"
        )
    object_center_robot = (object_pose @ np.asarray([*center, 1.0]))[:3]
    target = (base - object_center_robot) if position_first else (object_center_robot - base)
    target_norm = np.linalg.norm(target)
    if target_norm < 1e-9:
        raise ValueError("object center coincides with the xArm base")
    target /= target_norm

    centers = np.asarray(hand_sphere_centers_in_palm, dtype=np.float64)
    radii = np.asarray(hand_sphere_radii, dtype=np.float64)
    if (
        minimum_floor_z is None
        or centers.ndim != 3
        or centers.shape[0] != len(relative)
        or centers.shape[2] != 3
        or radii.shape != (centers.shape[1],)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError("valid hand collision spheres and floor height are required")

    selected_pose = None
    selected_angle = 0.0
    selected_minimum_z = None
    selected_score = -math.inf
    selected_position_axis_score = -math.inf
    selected_robot_axis_score = -math.inf
    selected_finger_axis_score = -math.inf
    best_minimum_z = -math.inf
    candidate_angles = (
        (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
        if candidate_angles_rad is None
        else tuple(float(value) for value in candidate_angles_rad)
    )
    if not candidate_angles:
        raise ValueError("candidate_angles_rad must be non-empty")
    for angle in candidate_angles:
        object_axis_rotation = _rotation_about_object_center(symmetry_axis, angle, center)
        candidate_object_pose = object_pose @ object_axis_rotation
        robot_from_palm = candidate_object_pose[None] @ relative
        bottoms = []
        for pose, frame_centers in zip(robot_from_palm, centers):
            world_centers = (pose[:3, :3] @ frame_centers.T).T + pose[:3, 3]
            bottoms.append(np.min(world_centers[:, 2] - radii))
        candidate_minimum_z = float(np.min(bottoms))
        best_minimum_z = max(best_minimum_z, candidate_minimum_z)
        if candidate_minimum_z < float(minimum_floor_z) - 1e-9:
            continue
        if position_first:
            palm_offset = robot_from_palm[0, :3, 3] - object_center_robot
            offset_norm = np.linalg.norm(palm_offset)
            score = (
                float(np.dot(palm_offset / offset_norm, target))
                if offset_norm > 1e-9
                else -math.inf
            )
        else:
            # Allegro V5 local -Z points palm-to-wrist/robotward, so +Z is the
            # absolute palm direction that should face from the xArm to object.
            score = float(np.dot(robot_from_palm[0, :3, 2], target))
        position_axis_score = -math.inf
        if prefer_positive_object_axis is not None:
            axis_index = {"x": 0, "y": 1, "z": 2}[prefer_positive_object_axis]
            # Apply the candidate's object-local half-turn to the recorded
            # wrist offset, then prefer positive object-local X/Y/Z.
            # The symmetry rotation is about the mesh centroid, not the
            # object-frame origin.  Include the centroid translation when
            # evaluating the candidate wrist position.
            relative_wrist = (
                object_axis_rotation[:3, :3]
                @ (relative[0, :3, 3] - center)
                + center
            )
            position_axis_score = float(relative_wrist[axis_index])
        robot_axis_score = -math.inf
        if prefer_positive_robot_axis is not None:
            axis_index = {"x": 0, "y": 1, "z": 2}[prefer_positive_robot_axis]
            # This is deliberately the absolute robot/world wrist position,
            # not an object-local coordinate. For vertical-symmetry objects
            # the xArm-facing convention is to favor the +X wrist candidate.
            robot_axis_score = float(robot_from_palm[0, :3, 3][axis_index])

        finger_axis_score = -math.inf
        if prefer_positive_finger_extension_robot_axis is not None:
            axis_index = {
                "x": 0,
                "y": 1,
                "z": 2,
            }[prefer_positive_finger_extension_robot_axis]
            # This is the orientation of the open index/middle/ring fingers,
            # not the palm normal or wrist position.
            finger_axis_score = float((robot_from_palm[0, :3, :3] @ finger_axis)[axis_index])

        if prefer_positive_finger_extension_robot_axis is not None:
            better_candidate = finger_axis_score > selected_finger_axis_score + 1e-9 or (
                abs(finger_axis_score - selected_finger_axis_score) <= 1e-9
                and score > selected_score
            )
        elif prefer_positive_robot_axis is not None:
            better_candidate = robot_axis_score > selected_robot_axis_score + 1e-9 or (
                abs(robot_axis_score - selected_robot_axis_score) <= 1e-9
                and score > selected_score
            )
        elif prefer_positive_object_axis is not None:
            better_candidate = position_axis_score > selected_position_axis_score + 1e-9 or (
                abs(position_axis_score - selected_position_axis_score) <= 1e-9
                and score > selected_score
            )
        else:
            better_candidate = score > selected_score
        if better_candidate:
            selected_pose = candidate_object_pose
            selected_angle = angle
            selected_minimum_z = candidate_minimum_z
            selected_score = score
            selected_position_axis_score = position_axis_score
            selected_robot_axis_score = robot_axis_score
            selected_finger_axis_score = finger_axis_score
    if selected_pose is None:
        raise ValueError(
            f"neither object-{symmetry_axis.upper()} rotation keeps the Allegro collision spheres "
            f"above z={float(minimum_floor_z):.3f}m; best minimum z={best_minimum_z:.3f}m"
        )
    return selected_pose, selected_angle, selected_minimum_z


def select_point_symmetric_object_pose(
    robot_from_object: np.ndarray,
    object_from_palm: np.ndarray,
    robot_base_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    hand_sphere_centers_in_palm: np.ndarray | None = None,
    hand_sphere_radii: np.ndarray | None = None,
    minimum_floor_z: float | None = None,
    symmetry_center_object: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, str, float | None]:
    """Choose the best equivalent half-turn for a point-symmetric object.

    For a centrally/point-symmetric pingpong mesh, the physically realizable
    equivalents are the identity and 180-degree rotations through its mesh
    centroid (the three principal object axes).  Selection uses palm position
    relative to the mesh centroid (not palm rotation), with floor clearance and
    a preference against underside grasps.
    """
    relative = np.asarray(object_from_palm, dtype=np.float64)
    if relative.ndim != 3 or relative.shape[1:] != (4, 4) or len(relative) == 0:
        raise ValueError("object_from_palm must have shape (N,4,4), N>=1")
    base = np.asarray(robot_base_xyz, dtype=np.float64)
    center = np.asarray(symmetry_center_object, dtype=np.float64)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError("robot_base_xyz must contain three finite values")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("symmetry_center_object must contain three finite values")
    object_pose = _finite_transform(robot_from_object, "robot_from_object")
    object_center_robot = (object_pose @ np.asarray([*center, 1.0]))[:3]
    target = base - object_center_robot
    norm = np.linalg.norm(target)
    if norm < 1e-9:
        raise ValueError("object center coincides with the xArm base")
    target /= norm
    # Point symmetry is resolved from palm position first: choose the highest
    # wrist, then use the palm +X (Viser red axis) direction as a deterministic
    # tie-break, followed by distance to the xArm-to-object line.
    centers = np.asarray(hand_sphere_centers_in_palm, dtype=np.float64)
    radii = np.asarray(hand_sphere_radii, dtype=np.float64)
    if (
        minimum_floor_z is None
        or centers.ndim != 3
        or centers.shape[0] != len(relative)
        or centers.shape[2] != 3
        or radii.shape != (centers.shape[1],)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(radii))
        or np.any(radii <= 0.0)
    ):
        raise ValueError("valid hand collision spheres and floor height are required")
    selected = None
    selected_label = "identity"
    selected_min_z = None
    selected_line_error = math.inf
    selected_palm_z = -math.inf
    selected_red_alignment = -math.inf
    best_min_z = -math.inf
    selected_above_center = False
    for axis, angle in (("identity", 0.0), ("x", math.pi), ("y", math.pi), ("z", math.pi)):
        symmetry = np.eye(4) if axis == "identity" else _rotation_about_object_center(axis, angle, center)
        candidate_pose = object_pose @ symmetry
        palms = candidate_pose[None] @ relative
        bottoms = []
        for pose, frame_centers in zip(palms, centers):
            world_centers = (pose[:3, :3] @ frame_centers.T).T + pose[:3, 3]
            bottoms.append(np.min(world_centers[:, 2] - radii))
        minimum_z = float(np.min(bottoms))
        best_min_z = max(best_min_z, minimum_z)
        if minimum_z < float(minimum_floor_z) - 1e-9:
            continue
        palm_offset = palms[0, :3, 3] - object_center_robot
        offset_norm = np.linalg.norm(palm_offset)
        if offset_norm < 1e-9:
            line_error = math.inf
        else:
            along_line = float(np.dot(palm_offset, target))
            line_error = float(
                np.linalg.norm(palm_offset - along_line * target)
            )
        above_center = bool(palms[0, 2, 3] >= object_center_robot[2] - 1e-6)
        palm_z = float(palms[0, 2, 3])
        red_alignment = float(np.dot(palms[0, :3, 0], -target))
        better = False
        if above_center and not selected_above_center:
            better = True
        elif above_center == selected_above_center:
            # Height is primary; 1 mm avoids selecting a barely higher wrist.
            if palm_z > selected_palm_z + 1e-3:
                better = True
            elif abs(palm_z - selected_palm_z) <= 1e-3:
                if red_alignment > selected_red_alignment + 1e-6:
                    better = True
                elif (
                    abs(red_alignment - selected_red_alignment) <= 1e-6
                    and line_error < selected_line_error
                ):
                    better = True
        if better:
            selected = candidate_pose
            selected_label = axis
            selected_min_z = minimum_z
            selected_line_error = line_error
            selected_palm_z = palm_z
            selected_red_alignment = red_alignment
            selected_above_center = above_center
    if selected is None:
        raise ValueError(
            "no point-symmetric half-turn keeps the Allegro collision spheres above "
            f"z={float(minimum_floor_z):.3f}m; best minimum z={best_min_z:.3f}m"
        )
    return selected, selected_label, selected_min_z


def hand_collision_spheres_in_palm(
    hand_urdf: Path,
    hand_qpos: np.ndarray,
    sphere_config: Path = DEFAULT_COLLISION_SPHERES,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate configured Allegro collision-sphere centers in the palm frame."""

    from paradex.utils.file_io import load_yaml
    from paradex.visualization.robot import RobotModule

    qpos = np.asarray(hand_qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 16 or not np.all(np.isfinite(qpos)):
        raise ValueError("hand_qpos must have shape (N,16)")
    specs = load_yaml(str(Path(sphere_config).expanduser().resolve()))[
        "collision_spheres"
    ]
    robot = RobotModule(str(Path(hand_urdf).expanduser().resolve()))
    frame_centers = []
    radii = []
    for frame_index, frame_qpos in enumerate(qpos):
        link_poses = robot.forward_kinematics(frame_qpos)
        centers = []
        frame_radii = []
        for link_name, spheres in specs.items():
            if link_name not in link_poses:
                continue
            palm_from_link = np.asarray(link_poses[link_name], dtype=np.float64)
            for sphere in spheres:
                center_link = np.asarray([*sphere["center"], 1.0], dtype=np.float64)
                centers.append((palm_from_link @ center_link)[:3])
                frame_radii.append(float(sphere["radius"]))
        if not centers:
            raise ValueError("collision sphere config contains no Allegro hand links")
        if frame_index == 0:
            radii = frame_radii
        elif not np.allclose(frame_radii, radii):
            raise RuntimeError("collision sphere radii changed between hand frames")
        frame_centers.append(centers)
    return np.asarray(frame_centers, dtype=np.float64), np.asarray(radii, dtype=np.float64)


def non_thumb_finger_extension_axis_in_palm(hand_urdf: Path) -> np.ndarray:
    """Return the open index/middle/ring fingertip direction in palm coordinates."""

    from paradex.visualization.robot import RobotModule

    robot = RobotModule(str(Path(hand_urdf).expanduser().resolve()))
    link_poses = robot.forward_kinematics(_allegro_open_hand_qpos())
    tip_links = ("link_3_0_tip", "link_7_0_tip", "link_11_0_tip")
    try:
        mean_tip_position = np.mean(
            [np.asarray(link_poses[name], dtype=np.float64)[:3, 3] for name in tip_links],
            axis=0,
        )
    except KeyError as exc:
        raise ValueError(f"hand URDF has no required non-thumb fingertip link: {exc}") from exc
    magnitude = np.linalg.norm(mean_tip_position)
    if not np.isfinite(magnitude) or magnitude < 1e-9:
        raise ValueError("non-thumb fingertip direction is invalid")
    return mean_tip_position / magnitude


def robot_trajectory_floor_clearance(
    robot_urdf: Path,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray,
    sphere_config: Path = DEFAULT_COLLISION_SPHERES,
) -> tuple[float, int, str]:
    """Return the lowest full-robot collision sphere over a replay trajectory."""

    from paradex.utils.file_io import load_yaml
    from paradex.visualization.robot import RobotModule

    arm = np.asarray(arm_qpos, dtype=np.float64)
    hand = np.asarray(hand_qpos, dtype=np.float64)
    if arm.ndim != 2 or arm.shape[1] != 6 or hand.shape != (len(arm), 16):
        raise ValueError("replay floor check requires arm (N,6) and hand (N,16)")
    if len(arm) == 0 or not np.all(np.isfinite(arm)) or not np.all(np.isfinite(hand)):
        raise ValueError("replay floor check requires a non-empty finite trajectory")

    specs = load_yaml(str(Path(sphere_config).expanduser().resolve()))[
        "collision_spheres"
    ]
    robot = RobotModule(str(Path(robot_urdf).expanduser().resolve()))
    joint_names = robot.get_joint_names()
    joint_indices = {name: index for index, name in enumerate(joint_names)}
    required_names = [f"joint{index}" for index in range(1, 7)] + list(
        ALLEGRO_JOINT_NAMES
    )
    missing = sorted(set(required_names).difference(joint_indices))
    if missing:
        raise ValueError(f"combined robot URDF is missing joints: {missing}")

    minimum_z = math.inf
    minimum_frame = -1
    minimum_link = ""
    full_qpos = np.zeros(robot.get_num_joints(), dtype=np.float64)
    for frame_index, (arm_frame, hand_frame) in enumerate(zip(arm, hand)):
        for joint_index, value in enumerate(arm_frame, start=1):
            full_qpos[joint_indices[f"joint{joint_index}"]] = value
        for name, value in zip(ALLEGRO_JOINT_NAMES, hand_frame):
            full_qpos[joint_indices[name]] = value
        link_poses = robot.forward_kinematics(full_qpos)
        for link_name, spheres in specs.items():
            if link_name not in link_poses:
                continue
            robot_from_link = np.asarray(link_poses[link_name], dtype=np.float64)
            for sphere in spheres:
                center_link = np.asarray([*sphere["center"], 1.0], dtype=np.float64)
                center_robot = robot_from_link @ center_link
                bottom_z = float(center_robot[2] - float(sphere["radius"]))
                if bottom_z < minimum_z:
                    minimum_z = bottom_z
                    minimum_frame = frame_index
                    minimum_link = link_name
    if minimum_frame < 0:
        raise ValueError("collision sphere config has no links in the combined robot URDF")
    return minimum_z, minimum_frame, minimum_link


def resolve_object_mesh(scene_root: Path, object_name: str) -> Path:
    scene_root = Path(scene_root)
    asset_scene_roots = [scene_root]
    dataset_root = scene_root.parent.parent
    if dataset_root.name == "allegro_old_dexonomy_trajectory":
        asset_scene_roots.extend(
            dataset_root.parent / source / scene_root.parent.name / scene_root.name
            for source in ("allegro_old", "allegro")
        )
    candidates = tuple(
        root / "query" / "object_asset" / f"{object_name}{suffix}"
        for root in asset_scene_roots
        for suffix in (".obj", ".ply")
    )
    mesh = next((path for path in candidates if path.is_file()), None)
    if mesh is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"native-frame object mesh not found; searched: {searched}")
    return mesh.resolve()


def mesh_vertex_centroid(mesh_path: Path) -> np.ndarray:
    """Return the arithmetic mean of all vertices in the native mesh frame."""
    import trimesh

    loaded = trimesh.load_mesh(str(Path(mesh_path).expanduser().resolve()), process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"object mesh has no valid vertices: {mesh_path}")
    return np.mean(vertices, axis=0)


def build_frame_cuboids(
    center_xyz: Sequence[float], inner_size_xyz: Sequence[float], thickness_m: float
) -> dict[str, dict[str, list[float]]]:
    """Build floor, ceiling, side walls, and back wall; the -x side is open."""

    center = np.asarray(center_xyz, dtype=np.float64)
    inner = np.asarray(inner_size_xyz, dtype=np.float64)
    thickness = float(thickness_m)
    if (
        center.shape != (3,)
        or inner.shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(inner))
        or np.any(inner <= 0.0)
        or not np.isfinite(thickness)
        or thickness <= 0.0
    ):
        raise ValueError("frame center/size/thickness must be finite and positive")

    def cuboid(dims: Sequence[float], position: Sequence[float]) -> dict[str, list[float]]:
        return {
            "dims": [float(value) for value in dims],
            "pose": [*[float(value) for value in position], 1.0, 0.0, 0.0, 0.0],
        }

    x, y, z = center
    sx, sy, sz = inner
    t = thickness
    return {
        "frame_floor": cuboid([sx, sy + 2.0 * t, t], [x, y, z - sz / 2.0 - t / 2.0]),
        "frame_ceiling": cuboid([sx, sy + 2.0 * t, t], [x, y, z + sz / 2.0 + t / 2.0]),
        "frame_left": cuboid([sx, t, sz], [x, y + sy / 2.0 + t / 2.0, z]),
        "frame_right": cuboid([sx, t, sz], [x, y - sy / 2.0 - t / 2.0, z]),
        "frame_back": cuboid(
            [t, sy + 2.0 * t, sz + 2.0 * t],
            [x + sx / 2.0 + t / 2.0, y, z],
        ),
    }


def build_collision_world(
    object_mesh: Path,
    robot_from_object: np.ndarray,
    frame_center_xyz: Sequence[float],
    frame_inner_size_xyz: Sequence[float],
    frame_thickness_m: float,
) -> dict[str, Any]:
    from src.object6d.capture_rcc import _rotmat_to_wxyz

    pose = _finite_transform(robot_from_object, "robot_from_object")
    quaternion = _rotmat_to_wxyz(pose[:3, :3])
    return {
        "cuboid": build_frame_cuboids(
            frame_center_xyz, frame_inner_size_xyz, frame_thickness_m
        ),
        "mesh": {
            "object": {
                "file_path": str(Path(object_mesh).resolve()),
                "pose": [
                    *pose[:3, 3].astype(float).tolist(),
                    *np.asarray(quaternion, dtype=float).tolist(),
                ],
            }
        },
    }


def configure_curobo_robot(
    config_path: Path, first_hand_qpos: np.ndarray, collision_buffer_m: float
) -> dict[str, Any]:
    from paradex.utils.file_io import load_yaml
    from src.object6d.capture_rcc import (
        _normalize_curobo_robot_cfg_paths,
        _normalize_curobo_robot_cfg_paths_with_cfg_file,
    )

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    config = _normalize_curobo_robot_cfg_paths_with_cfg_file(load_yaml(str(path)), str(path))
    config = _normalize_curobo_robot_cfg_paths(config)
    config = copy.deepcopy(config)
    kinematics = config["robot_cfg"]["kinematics"]
    kinematics["ee_link"] = "palm_link"
    kinematics["collision_sphere_buffer"] = float(collision_buffer_m)
    locked = kinematics.setdefault("lock_joints", {})
    for name, value in zip(ALLEGRO_JOINT_NAMES, first_hand_qpos):
        locked[name] = float(value)
    return config


def _load_pose(path: Path) -> np.ndarray:
    path = Path(path).expanduser()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("pose_robot", "robot_from_object", "pose"):
            if key in payload:
                return _finite_transform(payload[key], f"{path}:{key}")
        raise ValueError(f"no robot-frame object pose found in {path}")
    if path.suffix.lower() == ".npy":
        return _finite_transform(np.load(path), str(path))
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as values:
            key = "pose_robot" if "pose_robot" in values else values.files[0]
            return _finite_transform(values[key], f"{path}:{key}")
    return _finite_transform(np.loadtxt(path), str(path))


def capture_object_pose(args: argparse.Namespace, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if args.current_object_pose is not None:
        pose = _load_pose(args.current_object_pose)
        return pose, np.eye(4, dtype=np.float64)

    from src.object6d import capture_sil

    capture_dir = output_dir / "initial_object_capture"
    relative = capture_dir.resolve().relative_to(Path(shared_dir).resolve())
    remote_path = (Path("shared_data") / relative / "raw").as_posix()
    capture_sil.capture_once(capture_dir, remote_path, args.rcc_entry)
    response = capture_sil.send_rpc_once(
        args.rpc_addr,
        {
            "command": "infer",
            "image_path": capture_sil.to_shared_data_path(capture_dir),
            "mesh_name": args.object,
        },
        args.rpc_timeout_ms,
    )
    payload = capture_sil.extract_pose(response)
    c2r = _finite_transform(np.load(capture_dir / "C2R.npy"), "C2R")
    world_pose = _finite_transform(payload["pose_world"], "Object6D pose_world")
    robot_pose = np.linalg.inv(c2r) @ world_pose
    payload["C2R"] = c2r.tolist()
    payload["pose_robot"] = robot_pose.tolist()
    (capture_dir / "object_6d.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return robot_pose, c2r


def _read_live_arm_qpos() -> np.ndarray:
    from paradex.io.robot_controller import get_arm

    arm = get_arm("xarm", servo_api="cartesian_aa")
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            qpos = np.asarray(arm.get_data()["qpos"], dtype=np.float64)
            if qpos.shape == (6,) and np.all(np.isfinite(qpos)):
                return qpos
            time.sleep(0.05)
        raise RuntimeError("timed out waiting for a finite xArm joint state")
    finally:
        arm.end(set_break=False)


def plan_approach(
    robot_config: Mapping[str, Any],
    collision_world: Mapping[str, Any],
    start_arm_qpos: np.ndarray,
    target_robot_from_palm: np.ndarray,
    *,
    robot_urdf: Path,
    first_hand_qpos: np.ndarray,
    device: str,
    seeds: int,
    hover_clearance_m: float,
    min_stage_translation_m: float,
    min_stage_rotation_rad: float,
    allow_terminal_object_contact: bool = False,
) -> dict[str, Any]:
    """Plan the reference pipeline's raise/translate/descend approach stages.

    The requested object and enclosure collisions are enforced through the
    entire approach, including the frame-0 target.  The
    reference pipeline's HOME pose, symmetry, wrist-facing, workspace-half,
    hand-preshape, and self-collision constraints are intentionally omitted.
    """

    from src.object6d.capture_rcc import _run_curobo_plan
    from src.dataset_acquisition.h2r.hrdex_replay_common import (
        staged_home_to_pregrasp,
    )

    start = np.asarray(start_arm_qpos, dtype=np.float64).reshape(-1)
    hand = np.asarray(first_hand_qpos, dtype=np.float64).reshape(-1)
    if start.shape != (6,) or hand.shape != (16,):
        raise ValueError("staged approach requires 6 arm and 16 hand joints")

    current_robot_from_palm = _palm_pose_from_qpos(robot_urdf, start, hand)
    staged_poses, phase_ends = staged_home_to_pregrasp(
        current_robot_from_palm,
        target_robot_from_palm,
        rate_hz=30.0,
        planar_speed_mps=0.12,
        angular_speed_rps=0.70,
        raise_speed_mps=0.03,
        descent_speed_mps=0.06,
        min_hover_clearance_m=float(hover_clearance_m),
        table_normal=np.array([0.0, 0.0, 1.0]),
    )
    raw_stage_targets: list[tuple[str, np.ndarray]] = []
    for phase, end in phase_ends.items():
        target = np.asarray(staged_poses[int(end) - 1], dtype=np.float64)
        if not raw_stage_targets or not np.allclose(
            raw_stage_targets[-1][1], target, atol=1e-7
        ):
            raw_stage_targets.append((phase, target))

    stage_targets = _coalesce_approach_stages(
        current_robot_from_palm,
        raw_stage_targets,
        min_translation_m=float(min_stage_translation_m),
        min_rotation_rad=float(min_stage_rotation_rad),
    )

    terminal_world = copy.deepcopy(dict(collision_world))
    if allow_terminal_object_contact:
        terminal_world.get("mesh", {}).pop("object", None)

    def run(
        start_qpos: np.ndarray,
        target_pose: np.ndarray,
        world: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Each call constructs a MotionGen instance. Release the previous
        # instance's cached CUDA allocations before advancing to the next
        # approach stage.
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        return _run_curobo_plan(
            robot_cfg=dict(robot_config),
            world_cfg_dict=dict(world),
            start_qpos=np.asarray(start_qpos, dtype=np.float32),
            target_eef_pose_robot=np.asarray(target_pose, dtype=np.float32),
            device=device,
            require_dof=6,
            seeds=seeds,
            self_collision=False,
        )

    def validated_trajectory(
        result: Mapping[str, Any], expected_start: np.ndarray
    ) -> np.ndarray | None:
        if not result.get("success"):
            return None
        trajectory = np.asarray(result.get("trajectory"), dtype=np.float64)
        names = [str(name) for name in result.get("joint_names", [])]
        expected = [f"joint{index}" for index in range(1, 7)]
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 6
            or len(trajectory) < 2
            or not np.all(np.isfinite(trajectory))
        ):
            return None
        if names and names != expected:
            try:
                trajectory = trajectory[:, [names.index(name) for name in expected]]
            except ValueError:
                return None
        if np.max(np.abs(trajectory[0] - expected_start)) > 0.05:
            return None
        return trajectory

    current_qpos = start.copy()
    trajectory_parts: list[np.ndarray] = []
    segment_metadata: list[dict[str, Any]] = []
    staged_failure: dict[str, Any] | None = None
    for stage_index, (phase, target) in enumerate(stage_targets):
        is_replay_entry = stage_index == len(stage_targets) - 1
        stage_world = terminal_world if is_replay_entry else collision_world
        suffix = (
            " (object contact allowed)"
            if is_replay_entry and allow_terminal_object_contact
            else ""
        )
        print(f"[plan] staged approach: {phase}{suffix}")
        segment = run(current_qpos, target, stage_world)
        segment_trajectory = validated_trajectory(segment, current_qpos)
        segment_metadata.append(
            {
                "phase": phase,
                "success": segment_trajectory is not None,
                "status": segment.get("status"),
                "attempts": segment.get("attempts"),
                "timing_s": segment.get("timing_s"),
            }
        )
        if segment_trajectory is None:
            staged_failure = dict(segment)
            staged_failure["failed_phase"] = phase
            break
        trajectory_parts.append(
            segment_trajectory if not trajectory_parts else segment_trajectory[1:]
        )
        current_qpos = segment_trajectory[-1]

    if staged_failure is None and trajectory_parts:
        return {
            "success": True,
            "trajectory": np.concatenate(trajectory_parts, axis=0),
            "joint_names": [f"joint{index}" for index in range(1, 7)],
            "planning_mode": "staged_raise_planar_rotate_descend",
            "self_collision": False,
            "terminal_object_collision_ignored": allow_terminal_object_contact,
            "segments": segment_metadata,
        }

    terminal_policy = (
        "object contact allowed"
        if allow_terminal_object_contact
        else "object collision active"
    )
    print(
        "[plan] staged approach failed; retrying direct replay entry with "
        f"{terminal_policy}"
    )
    direct = run(start, target_robot_from_palm, terminal_world)
    direct_trajectory = validated_trajectory(direct, start)
    if direct_trajectory is not None:
        return {
            **direct,
            "success": True,
            "trajectory": direct_trajectory,
            "planning_mode": "direct_fallback",
            "self_collision": False,
            "terminal_object_collision_ignored": allow_terminal_object_contact,
            "staged_segments": segment_metadata,
        }
    return {
        **direct,
        "success": False,
        "planning_mode": "staged_then_direct_failed",
        "self_collision": False,
        "terminal_object_collision_ignored": allow_terminal_object_contact,
        "staged_segments": segment_metadata,
        "staged_failure": staged_failure,
        "partial_approach_qpos": (
            np.concatenate(trajectory_parts, axis=0)
            if trajectory_parts else np.empty((0, 6))
        ),
    }


def _coalesce_approach_stages(
    start_pose: np.ndarray,
    stage_targets: Sequence[tuple[str, np.ndarray]],
    *,
    min_translation_m: float,
    min_rotation_rad: float,
) -> list[tuple[str, np.ndarray]]:
    """Merge nearby intermediate targets so IK is not sampled at micro-stages."""

    if min_translation_m < 0.0 or min_rotation_rad < 0.0:
        raise ValueError("minimum approach stage distances must be non-negative")
    if not stage_targets:
        return []

    kept: list[tuple[str, np.ndarray]] = []
    previous = np.asarray(start_pose, dtype=np.float64)
    for index, (phase, target_value) in enumerate(stage_targets):
        target = np.asarray(target_value, dtype=np.float64)
        translation = float(np.linalg.norm(target[:3, 3] - previous[:3, 3]))
        rotation = float(
            Rotation.from_matrix(previous[:3, :3].T @ target[:3, :3]).magnitude()
        )
        is_final = index == len(stage_targets) - 1
        too_close = (
            translation < float(min_translation_m)
            and rotation < float(min_rotation_rad)
        )
        if too_close and not is_final:
            continue
        if too_close and is_final and kept:
            prior_phase, _ = kept[-1]
            kept[-1] = (f"{prior_phase}+{phase}", target)
        else:
            kept.append((phase, target))
        previous = kept[-1][1]
    return kept


def solve_replay_arm_ik(
    robot_config: Mapping[str, Any],
    palm_poses: np.ndarray,
    start_arm_qpos: np.ndarray,
    *,
    device: str,
    seeds: int,
) -> np.ndarray:
    """Solve a continuous joint path used to enforce replay joint speed."""
    import gc

    local_curobo_src = str(PROJECT_ROOT / "curobo" / "src")
    if local_curobo_src not in sys.path:
        sys.path.insert(0, local_curobo_src)
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from src.object6d.capture_rcc import _rotmat_to_wxyz

    poses = np.asarray(palm_poses, dtype=np.float64)
    start = np.asarray(start_arm_qpos, dtype=np.float32).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 1:
        raise ValueError("replay palm poses must have shape (N,4,4)")
    if start.shape != (6,):
        raise ValueError("replay IK start must contain six xArm joints")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dev = torch.device(device if "cuda" in str(device) and torch.cuda.is_available() else "cpu")
    tensor_args = TensorDeviceType(device=dev)
    cfg = copy.deepcopy(dict(robot_config))
    cspace = cfg["robot_cfg"]["kinematics"]["cspace"]
    cspace["null_space_weight"] = [5.0] * len(cspace["joint_names"])
    ik_cfg = IKSolverConfig.load_from_robot_config(
        cfg,
        None,
        tensor_args=tensor_args,
        num_seeds=int(seeds),
        position_threshold=0.01,
        rotation_threshold=0.1,
        self_collision_check=False,
        self_collision_opt=False,
        use_cuda_graph=False,
    )
    ik = IKSolver(ik_cfg)
    joint_names = list(ik.joint_names)
    expected = [f"joint{index}" for index in range(1, 7)]
    if joint_names != expected:
        raise RuntimeError(f"unexpected replay IK joint order: {joint_names}")

    qpos = np.empty((len(poses), 6), dtype=np.float64)
    qpos[0] = start
    current = start.copy()
    for index in range(1, len(poses)):
        pose = poses[index]
        goal = Pose(
            position=tensor_args.to_device(pose[None, :3, 3].astype(np.float32)),
            quaternion=tensor_args.to_device(
                _rotmat_to_wxyz(pose[:3, :3])[None].astype(np.float32)
            ),
        )
        seed = tensor_args.to_device(current.reshape(1, 1, 6))
        retract = tensor_args.to_device(current.reshape(1, 6))
        result = ik.solve_single(
            goal, retract_config=retract, seed_config=seed, num_seeds=1
        )
        success = bool(result.success.reshape(-1)[0].item())
        if not success and seeds > 1:
            result = ik.solve_single(
                goal,
                retract_config=retract,
                seed_config=seed,
                num_seeds=int(seeds),
            )
            success = bool(result.success.reshape(-1)[0].item())
        if not success:
            raise RuntimeError(f"continuous replay IK failed at frame {index}/{len(poses) - 1}")
        current = result.solution.detach().cpu().numpy().reshape(-1, 6)[0].astype(np.float32)
        qpos[index] = current
    return qpos


def append_post_grasp_lift(
    trajectory: RetrievalTrajectory,
    palm_poses: np.ndarray,
    distance_m: float,
    duration_s: float,
    post_lift_hold_s: float,
    release_duration_s: float,
) -> tuple[RetrievalTrajectory, np.ndarray]:
    """Lift holding the grasp, hold until the release time, then open."""
    poses = np.asarray(palm_poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 1:
        raise ValueError("post-grasp lift requires palm poses with shape (N,4,4)")
    if (
        distance_m <= 0.0
        or duration_s <= 0.0
        or post_lift_hold_s <= 0.0
        or release_duration_s <= 0.0
    ):
        raise ValueError("post-grasp lift distance and durations must be positive")

    lifted_palm = poses[-1].copy()
    lifted_palm[2, 3] += float(distance_m)
    robot_from_object = poses[-1] @ np.linalg.inv(trajectory.object_from_palm[-1])
    lifted_object_from_palm = np.linalg.inv(robot_from_object) @ lifted_palm
    extended = RetrievalTrajectory(
        path=trajectory.path,
        times_s=np.concatenate(
            (
                trajectory.times_s,
                [
                    trajectory.times_s[-1] + float(duration_s),
                    trajectory.times_s[-1]
                    + float(duration_s)
                    + float(post_lift_hold_s),
                    trajectory.times_s[-1]
                    + float(duration_s)
                    + float(post_lift_hold_s)
                    + float(release_duration_s),
                ],
            )
        ),
        object_from_palm=np.concatenate(
            (
                trajectory.object_from_palm,
                lifted_object_from_palm[None],
                lifted_object_from_palm[None],
                lifted_object_from_palm[None],
            ),
            axis=0,
        ),
        hand_qpos=np.concatenate(
            (
                trajectory.hand_qpos,
                trajectory.hand_qpos[-1][None],
                trajectory.hand_qpos[-1][None],
                _allegro_open_hand_qpos()[None],
            ),
            axis=0,
        ),
    )
    return extended, np.concatenate(
        (poses, lifted_palm[None], lifted_palm[None], lifted_palm[None]), axis=0
    )


def _allegro_open_hand_qpos() -> np.ndarray:
    """Physical right-A open-hand convention used by the capture pipeline."""
    open_hand = np.zeros(16, dtype=np.float64)
    open_hand[12] = 1.6
    return open_hand


def _palm_pose_from_qpos(
    robot_urdf: Path, arm_qpos: np.ndarray, hand_qpos: np.ndarray
) -> np.ndarray:
    from paradex.visualization.robot import RobotModule

    robot = RobotModule(str(Path(robot_urdf).expanduser().resolve()))
    full_qpos = _full_robot_qpos(robot_urdf, arm_qpos, hand_qpos)
    return np.asarray(robot.forward_kinematics(full_qpos)["palm_link"], dtype=np.float64)


def _approach_times(trajectory: np.ndarray, max_speed_deg_s: float) -> np.ndarray:
    qpos = np.asarray(trajectory, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 6 or len(qpos) < 2:
        raise ValueError("approach trajectory must have shape (N,6), N>=2")
    limit = math.radians(float(max_speed_deg_s))
    periods = np.maximum(0.01, np.max(np.abs(np.diff(qpos, axis=0)), axis=1) / limit)
    return np.concatenate(([0.0], np.cumsum(periods)))


def palm_poses_to_link6(robot_urdf: Path, palm_poses: np.ndarray) -> np.ndarray:
    from paradex.visualization.robot import RobotModule

    robot = RobotModule(str(Path(robot_urdf).expanduser().resolve()))
    qpos = np.zeros(robot.get_num_joints(), dtype=np.float64)
    fk = robot.forward_kinematics(qpos)
    link6_from_palm = np.linalg.inv(fk["link6"]) @ fk["palm_link"]
    return np.asarray(palm_poses, dtype=np.float64) @ np.linalg.inv(link6_from_palm)


def retime_replay(
    times_s: np.ndarray,
    link6_poses: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    speed_scale: float,
    max_linear_speed_mps: float,
    max_angular_speed_rps: float,
    max_hand_speed_rps: float,
    arm_qpos: np.ndarray | None = None,
    max_arm_joint_speed_deg_s: float | None = None,
) -> np.ndarray:
    times = np.asarray(times_s, dtype=np.float64)
    source_periods = np.diff(times) / float(speed_scale)
    linear = np.linalg.norm(np.diff(link6_poses[:, :3, 3], axis=0), axis=1)
    angular = np.asarray(
        [
            Rotation.from_matrix(link6_poses[index, :3, :3].T @ link6_poses[index + 1, :3, :3]).magnitude()
            for index in range(len(link6_poses) - 1)
        ]
    )
    hand = np.max(np.abs(np.diff(hand_qpos, axis=0)), axis=1)
    arm_seconds = np.zeros_like(source_periods)
    if arm_qpos is not None:
        arm = np.asarray(arm_qpos, dtype=np.float64)
        if arm.shape != (len(times), 6):
            raise ValueError("replay arm qpos must have shape (N,6)")
        if max_arm_joint_speed_deg_s is None or max_arm_joint_speed_deg_s <= 0:
            raise ValueError("positive xArm joint speed limit required for replay qpos")
        arm_seconds = np.max(np.abs(np.diff(arm, axis=0)), axis=1) / math.radians(
            float(max_arm_joint_speed_deg_s)
        )
    periods = np.maximum.reduce(
        (
            source_periods,
            linear / float(max_linear_speed_mps),
            angular / float(max_angular_speed_rps),
            hand / float(max_hand_speed_rps),
            arm_seconds,
        )
    )
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("replay retiming produced invalid periods")
    return np.concatenate(([0.0], np.cumsum(periods)))


def densify_joint_replay(
    arm_qpos: np.ndarray, hand_qpos: np.ndarray, times_s: np.ndarray, rate_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arm = np.asarray(arm_qpos, dtype=np.float64)
    hand = np.asarray(hand_qpos, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    dense_times = np.arange(0.0, times[-1], 1.0 / float(rate_hz))
    dense_times = np.append(dense_times, times[-1])
    dense_arm = np.column_stack([np.interp(dense_times, times, arm[:, j]) for j in range(6)])
    dense_hand = np.column_stack(
        [np.interp(dense_times, times, hand[:, j]) for j in range(hand.shape[1])]
    )
    return dense_arm, dense_hand, dense_times


def densify_replay(
    poses: np.ndarray, hand: np.ndarray, times_s: np.ndarray, rate_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_poses: list[np.ndarray] = []
    dense_hand: list[np.ndarray] = []
    dense_times: list[float] = []
    for index in range(len(times_s) - 1):
        duration = float(times_s[index + 1] - times_s[index])
        steps = max(1, int(math.ceil(duration * float(rate_hz))))
        alphas = np.arange(steps, dtype=np.float64) / steps
        rotations = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(np.stack((poses[index, :3, :3], poses[index + 1, :3, :3]))),
        )(alphas).as_matrix()
        for step, alpha in enumerate(alphas):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotations[step]
            pose[:3, 3] = (1.0 - alpha) * poses[index, :3, 3] + alpha * poses[index + 1, :3, 3]
            dense_poses.append(pose)
            dense_hand.append((1.0 - alpha) * hand[index] + alpha * hand[index + 1])
            dense_times.append(float(times_s[index] + alpha * duration))
    dense_poses.append(np.asarray(poses[-1], dtype=np.float64).copy())
    dense_hand.append(np.asarray(hand[-1], dtype=np.float64).copy())
    dense_times.append(float(times_s[-1]))
    return np.asarray(dense_poses), np.asarray(dense_hand), np.asarray(dense_times)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def execution_confirmation_accepted(response: str) -> bool:
    """Accept the explicit PLAY command or a short yes alias."""

    return str(response).strip().lower() in {"play", "y"}


def save_plan(
    output_dir: Path,
    trajectory: RetrievalTrajectory,
    robot_from_object: np.ndarray,
    robot_from_palm: np.ndarray,
    robot_from_link6: np.ndarray,
    replay_times: np.ndarray,
    replay_arm_qpos: np.ndarray,
    approach_times: np.ndarray,
    start_arm_qpos: np.ndarray,
    collision_world: Mapping[str, Any],
    planning: Mapping[str, Any],
) -> Path:
    action_dir = output_dir / trajectory.path.parent.name
    action_dir.mkdir(parents=True, exist_ok=True)
    approach = np.asarray(planning.get("trajectory", np.zeros((0, 6))), dtype=np.float64)
    plan_path = action_dir / "replay_plan.npz"
    np.savez_compressed(
        plan_path,
        source_npz=str(trajectory.path),
        robot_from_object=robot_from_object,
        robot_from_palm=robot_from_palm,
        robot_from_link6=robot_from_link6,
        hand_qpos=trajectory.hand_qpos,
        replay_time_s=replay_times,
        replay_arm_qpos=replay_arm_qpos,
        start_arm_qpos=start_arm_qpos,
        approach_arm_qpos=approach,
        approach_time_s=np.asarray(approach_times, dtype=np.float64),
        planning_success=bool(planning.get("success", False)),
    )
    (action_dir / "plan_metadata.json").write_text(
        json.dumps(
            {
                "source_npz": str(trajectory.path),
                "transform_convention": "robot_T_palm = robot_T_object @ object_T_palm",
                "target_link": "palm_link",
                "approach_collision_policy": (
                    "object + five-sided frame before replay entry; object contact "
                    "allowed at terminal replay entry; self collision omitted"
                    if "object" in collision_world.get("mesh", {})
                    else "five-sided frame only; object and self collision omitted (diagnostic)"
                ),
                "replay_collision_policy": (
                    "object contact intentional; every replay frame checked against "
                    "the floor with full-robot collision spheres"
                ),
                "collision_world": collision_world,
                "planning": planning,
            },
            ensure_ascii=False,
            indent=2,
            default=_jsonable,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_path


def _full_robot_qpos(
    robot_urdf: Path, arm_qpos: np.ndarray, hand_qpos: np.ndarray
) -> np.ndarray:
    from paradex.visualization.robot import RobotModule

    robot = RobotModule(str(Path(robot_urdf).expanduser().resolve()))
    values = np.zeros(robot.get_num_joints(), dtype=np.float64)
    indices = {name: index for index, name in enumerate(robot.get_joint_names())}
    commands = {
        **{f"joint{index + 1}": float(value) for index, value in enumerate(arm_qpos)},
        **{name: float(value) for name, value in zip(ALLEGRO_JOINT_NAMES, hand_qpos)},
    }
    missing = sorted(set(commands).difference(indices))
    if missing:
        raise ValueError(f"combined robot URDF is missing joints: {missing}")
    for name, value in commands.items():
        values[indices[name]] = value
    return values


def _combined_robot_trajectory(
    robot_urdf: Path, arm_qpos: np.ndarray, hand_qpos: np.ndarray
) -> np.ndarray:
    from paradex.visualization.robot import RobotModule

    arm = np.asarray(arm_qpos, dtype=np.float64)
    hand = np.asarray(hand_qpos, dtype=np.float64)
    if arm.ndim != 2 or arm.shape[1] != 6 or hand.shape != (len(arm), 16):
        raise ValueError("combined robot trajectory requires arm (N,6) and hand (N,16)")
    robot = RobotModule(str(robot_urdf))
    names = robot.get_joint_names()
    full = np.tile(_full_robot_qpos(robot_urdf, arm[0], hand[0]), (len(arm), 1))
    for index in range(6):
        full[:, names.index(f"joint{index + 1}")] = arm[:, index]
    for index, name in enumerate(ALLEGRO_JOINT_NAMES):
        full[:, names.index(name)] = hand[:, index]
    return full


def publish_viser_preview(
    *,
    address: str,
    timeout_ms: int,
    object_name: str,
    object_mesh: Path,
    robot_from_object: np.ndarray,
    robot_urdf: Path,
    hand_urdf: Path,
    start_arm_qpos: np.ndarray,
    robot_from_palm: np.ndarray,
    hand_qpos: np.ndarray,
    action: str,
    planning_success: bool,
    approach_qpos: np.ndarray | None = None,
    replay_arm_qpos: np.ndarray | None = None,
    failed_approach_qpos: np.ndarray | None = None,
    frame_cuboids: dict | None = None,
    viser_fps: float = 120.0,
) -> bool:
    """Publish the measured object and floating-hand replay, even if planning failed."""

    import trimesh

    from paradex.visualization.visualizer.viser_client import ViserClient

    if frame_cuboids is None:
        frame_cuboids = build_frame_cuboids(
            FRAME_CENTER_XYZ_M, FRAME_INNER_SIZE_XYZ_M, 0.03
        )
    frame_objects = []
    for name, spec in frame_cuboids.items():
        pose = np.eye(4)
        pose[:3, 3] = spec["pose"][:3]
        frame_objects.append((name, trimesh.creation.box(extents=spec["dims"]), pose))
    loaded = trimesh.load_mesh(str(object_mesh), process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"object mesh has no valid vertices: {object_mesh}")
    object_vertex_centroid = np.mean(vertices, axis=0)
    object_axes_pose = np.asarray(robot_from_object, dtype=np.float64).copy()
    object_axes_pose[:3, 3] = (
        object_axes_pose
        @ np.asarray([*object_vertex_centroid, 1.0], dtype=np.float64)
    )[:3]
    palm_axes_mesh = trimesh.creation.axis(
        origin_size=0.006,
        axis_length=0.08,
        axis_radius=0.002,
    )
    object_axes_mesh = trimesh.creation.axis(
        origin_size=0.006,
        axis_length=0.10,
        axis_radius=0.002,
    )

    client = ViserClient(address, timeout_ms=timeout_ms)
    try:
        response = client.ping()
        if not response.get("ok", False):
            raise RuntimeError(f"Viser ping failed: {response}")
        client.clear_objects()
        for name, frame_mesh, pose in frame_objects:
            response = client.set_object(name, frame_mesh, pose, opacity=0.15)
            if not response.get("ok", False):
                raise RuntimeError(f"Viser frame update failed: {response}")
        response = client.set_object(object_name, mesh, robot_from_object, opacity=1.0)
        if not response.get("ok", False):
            raise RuntimeError(f"Viser object update failed: {response}")
        response = client.set_object(
            "detected_object_axes",
            object_axes_mesh,
            object_axes_pose,
            opacity=1.0,
        )
        if not response.get("ok", False):
            raise RuntimeError(f"Viser object-axis update failed: {response}")
        response = client.set_object(
            "replay_start_palm_axes",
            palm_axes_mesh,
            robot_from_palm[0],
            opacity=1.0,
        )
        if not response.get("ok", False):
            raise RuntimeError(f"Viser palm-axis update failed: {response}")

        replay_arm = np.asarray(replay_arm_qpos, dtype=np.float64)
        approach_arm = np.asarray(approach_qpos, dtype=np.float64)
        if (
            planning_success
            and replay_arm.shape == (len(hand_qpos), 6)
            and approach_arm.ndim == 2
            and approach_arm.shape[1] == 6
            and len(approach_arm) >= 2
        ):
            approach_full = _combined_robot_trajectory(
                robot_urdf,
                approach_arm,
                np.tile(hand_qpos[0], (len(approach_arm), 1)),
            )
            replay_full = _combined_robot_trajectory(robot_urdf, replay_arm, hand_qpos)
            full = np.concatenate((approach_full, replay_full[1:]), axis=0)
            response = client.show_only_robot("robot")
            if not response.get("ok", False):
                raise RuntimeError(f"Viser robot selection failed: {response}")
            response = client.set_traj(
                full, name=f"{action}_APPROACH_AND_REPLAY", robot_name="robot"
            )
            if not response.get("ok", False):
                raise RuntimeError(f"Viser full replay update failed: {response}")
            print(f"[viser] {action}: published {len(full)} approach + replay frames")
            return True

        if failed_approach_qpos is not None:
            arm_path = np.asarray(failed_approach_qpos, dtype=np.float64)
            if arm_path.ndim != 2 or arm_path.shape[1] != 6 or len(arm_path) < 2:
                print("[viser] No successful approach segment; showing object and relative hand replay")
            else:
                from paradex.visualization.robot import RobotModule

                robot = RobotModule(str(robot_urdf))
                names = robot.get_joint_names()
                full = np.tile(
                    _full_robot_qpos(robot_urdf, arm_path[0], hand_qpos[0]),
                    (len(arm_path), 1),
                )
                for index in range(6):
                    full[:, names.index(f"joint{index + 1}")] = arm_path[:, index]
                response = client.show_only_robot("robot")
                if not response.get("ok", False):
                    raise RuntimeError(f"Viser robot selection failed: {response}")
                response = client.set_traj(
                    full, name=f"{action}_PARTIAL_APPROACH_FAILED_PLAN", robot_name="robot"
                )
                if not response.get("ok", False):
                    raise RuntimeError(f"Viser partial approach update failed: {response}")
                print(f"[viser] Failed plan: published {len(full)} successful approach frames; use Playback")
                return True

        combined_qpos = _full_robot_qpos(robot_urdf, start_arm_qpos, hand_qpos[0])
        response = client.set_qpos(combined_qpos, name="robot")
        if not response.get("ok", False):
            raise RuntimeError(
                "Viser server must be started with the xarm_allegro_v5 URDF: "
                f"{response}"
            )
        preview_name = "retrieval_replay_hand"
        response = client.set_static_robot(
            preview_name,
            str(Path(hand_urdf).expanduser().resolve()),
            robot_from_palm[0],
            hand_qpos[0],
            show_only=False,
        )
        if not response.get("ok", False):
            raise RuntimeError(f"Viser hand update failed: {response}")
        response = client.set_traj(
            hand_qpos,
            name=(
                f"{action}_replay"
                if planning_success
                else f"{action}_REPLAY_ONLY_PLANNING_FAILED"
            ),
            robot_name=preview_name,
            robot_pose_traj={preview_name: robot_from_palm},
        )
        if not response.get("ok", False):
            raise RuntimeError(f"Viser trajectory update failed: {response}")
        status = "collision-free approach found" if planning_success else "approach planning failed"
        print(
            f"[viser] {action}: detected object + {len(hand_qpos)}-frame relative "
            f"hand replay pushed to {address} ({status}; replay preview only)"
        )
        return True
    except Exception as exc:
        print(f"[viser] WARNING: preview unavailable at {address}: {exc}")
        client.close()
        try:
            from paradex.visualization.visualizer.viser import ViserViewer
            from paradex.visualization.robot import RobotModule

            vis = ViserViewer(
                    server_host="127.0.0.1", server_port=8080,
                    scene_title=f"{object_name}: replay preview",
                )
            vis.add_robot("robot", str(robot_urdf), max_mesh_faces=1000)
            vis.robot_dict["robot"].update_cfg(
                _full_robot_qpos(robot_urdf, start_arm_qpos, hand_qpos[0])
            )
            vis.add_object(
                object_name, mesh,
                robot_from_object,
            )
            vis.add_object(
                "detected_object_axes",
                object_axes_mesh,
                object_axes_pose,
            )
            vis.add_object(
                "replay_start_palm_axes",
                palm_axes_mesh,
                robot_from_palm[0],
            )
            for name, frame_mesh, pose in frame_objects:
                vis.add_object(name, frame_mesh, pose, opacity=0.15)

            if planning_success:
                arm_path = np.asarray(approach_qpos, dtype=np.float64)
                replay_arm = np.asarray(replay_arm_qpos, dtype=np.float64)
                approach_full = _combined_robot_trajectory(
                    robot_urdf, arm_path,
                    np.tile(hand_qpos[0], (len(arm_path), 1)),
                )
                replay_full = _combined_robot_trajectory(
                    robot_urdf, replay_arm, hand_qpos
                )
                full = np.concatenate((approach_full, replay_full[1:]), axis=0)
                vis.add_traj("successful_approach_and_replay", {"robot": full})
                label = "Planning succeeded. Showing the joint-limited xArm approach and replay."
            else:
                partial = np.asarray(failed_approach_qpos)
                if partial.ndim == 2 and partial.shape[1] == 6 and len(partial) >= 2:
                    names = RobotModule(str(robot_urdf)).get_joint_names()
                    full = np.tile(
                        _full_robot_qpos(robot_urdf, partial[0], hand_qpos[0]),
                        (len(partial), 1),
                    )
                    for index in range(6):
                        full[:, names.index(f"joint{index + 1}")] = partial[:, index]
                    vis.add_traj("successful_approach_prefix", {"robot": full})
                    label = "Successful approach prefix only; subsequent planning failed."
                else:
                    vis.add_robot("replay_hand", str(hand_urdf), max_mesh_faces=1000)
                    vis.add_traj(
                        "relative_hand_replay", {"replay_hand": hand_qpos},
                        robot_pose_traj={"replay_hand": robot_from_palm},
                    )
                    label = "No successful arm segment. Showing requested relative hand replay."
            vis.server.gui.add_markdown(label + " Visualization only.")
            vis.update_scene(0)
            vis.gui_framerate.value = float(viser_fps)
            vis.gui_playing.value = True
            print("[viser] Local viewer opened; use the HTTP address above. Playback starts automatically; Ctrl+C closes preview.")
            try:
                vis.start_viewer()
            finally:
                import atexit

                atexit.unregister(vis.server.stop)
                websock_server = getattr(vis.server, "_websock_server", None)
                if websock_server is not None:
                    # viser registers the underlying WebsockServer.stop(), not
                    # ViserServer.stop(), with atexit.
                    atexit.unregister(websock_server.stop)
                try:
                    vis.server.stop()
                except RuntimeError as exc:
                    # viser 0.2.x iterates its live client dictionary while
                    # websocket disconnect callbacks can mutate it.  The stop
                    # signal has already been delivered at this point, so this
                    # specific shutdown-only race must not turn a successful
                    # preview into an execution failure.
                    if "dictionary changed size during iteration" not in str(exc):
                        raise
                    server_thread = getattr(websock_server, "_server_thread", None)
                    if server_thread is not None:
                        server_thread.join(timeout=1.0)
            return True
        except Exception as local_exc:
            print(f"[viser] Local viewer failed: {local_exc}")
        print(
            "[viser] start it with: python -m "
            "paradex.visualization.visualizer.viser_server "
            "--robot_urdf rsc/robot/xarm_allegro_v5.urdf"
        )
        return False
    finally:
        client.close()


def execute_plan(
    approach_qpos: np.ndarray,
    approach_times_s: np.ndarray,
    replay_arm_qpos: np.ndarray,
    hand_qpos: np.ndarray,
    replay_times_s: np.ndarray,
    *,
    max_hand_speed_rps: float,
    command_rate_hz: float,
    allegro_namespace: str,
    capture_raw_dir: Path | None = None,
) -> None:
    from paradex.io.robot_controller import get_arm, get_hand

    arm = None
    hand = None
    arm_recording = False
    hand_recording = False
    try:
        arm = get_arm(
            "xarm",
            servo_api="angle_j",
            fps=float(command_rate_hz),
        )
        hand = get_hand(
            "allegro_v5",
            hand_side="right",
            namespace=allegro_namespace,
        )
        if not hand.connection_event.wait(timeout=5.0):
            raise RuntimeError(
                "timed out waiting for Allegro V5 joint state on "
                f"{hand.state_topic}"
            )
        if capture_raw_dir is not None:
            raw_dir = Path(capture_raw_dir)
            arm.start(str(raw_dir / "arm"))
            arm_recording = True
            hand.start(str(raw_dir / "hand"))
            hand_recording = True
        measured = np.asarray(hand.get_data()["qpos"], dtype=np.float64)
        hand.set_command_slew_rate(max_hand_speed_rps, initial_action=measured)
        hand.move(hand_qpos[0])
        settle = float(np.max(np.abs(hand_qpos[0] - measured)) / max_hand_speed_rps + 0.5)
        if not hand.wait_for_published_action(hand_qpos[0], timeout_seconds=settle):
            raise RuntimeError("Allegro pre-position command did not finish")
        if arm.is_error() or hand.is_error():
            raise RuntimeError("controller error before approach")

        dense_approach, _, dense_approach_times = densify_joint_replay(
            approach_qpos,
            np.tile(hand_qpos[0], (len(approach_qpos), 1)),
            approach_times_s,
            command_rate_hz,
        )
        print(
            f"[execute] approach resampled {len(approach_qpos)} -> "
            f"{len(dense_approach)} points at {command_rate_hz:.1f}Hz"
        )
        arm.move_joint_trajectory(dense_approach, dense_approach_times)
        dense_arm, dense_hand, dense_times = densify_joint_replay(
            replay_arm_qpos, hand_qpos, replay_times_s, command_rate_hz
        )
        started = time.perf_counter()
        for index, (arm_command, hand_command, target_time) in enumerate(
            zip(dense_arm, dense_hand, dense_times)
        ):
            arm.move(arm_command, is_servo=True)
            hand.move(hand_command)
            if arm.is_error() or hand.is_error():
                raise RuntimeError(f"controller error at replay frame {index}")
            delay = started + float(target_time) - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)
    finally:
        if hand is not None:
            try:
                try:
                    if hand_recording:
                        hand.stop()
                finally:
                    hand.set_command_slew_rate(None)
                    hand.end()
            except Exception as exc:
                print(f"[capture] WARNING: Allegro cleanup failed: {exc}")
        if arm is not None:
            try:
                try:
                    if arm_recording:
                        arm.stop()
                finally:
                    arm.end(set_break=False)
            except Exception as exc:
                print(f"[capture] WARNING: xArm cleanup failed: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, help="retrieval/Object6D mesh name")
    parser.add_argument(
        "--align-axisymmetric-x",
        action="store_true",
        help=(
            "resolve object-x pose symmetry from absolute palm direction and floor "
            "clearance (continuous object-X for brown_ramen, continuous object-Y "
            "for yellow_wash_tub, grey_pot, and smallbowl, and object-Y "
            "rotation 0/90/180/270 degrees for coffee_tin (object-Y) and vertical-symmetry "
            "objects (object-Z); both about the mesh centroid)"
        ),
    )
    parser.add_argument(
        "--align-point-symmetric",
        action="store_true",
        help="for pingpong, choose identity or a 180-degree mesh-centroid half-turn",
    )
    parser.add_argument("--scene", type=int, default=None, help="scene number; inferred when unique")
    parser.add_argument("--action", choices=(*ACTION_NAMES, "all"), default="ours")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--current-object-pose", type=Path, default=None, help="offline robot-frame pose; skips capture/RPC")
    parser.add_argument("--start-arm-qpos", default=None, help="six comma-separated radians; otherwise read live xArm")
    parser.add_argument("--frame-thickness-m", type=float, default=0.03)
    parser.add_argument("--curobo-config", type=Path, default=DEFAULT_CUROBO_CFG)
    parser.add_argument("--curobo-device", default="cuda:0")
    parser.add_argument("--curobo-seeds", type=int, default=256)
    parser.add_argument("--collision-buffer-m", type=float, default=0.006)
    parser.add_argument(
        "--ignore-object-collision",
        action="store_true",
        help="diagnostic only: show the object but omit it from cuRobo collision checks",
    )
    parser.add_argument("--approach-hover-clearance-m", type=float, default=0.12)
    parser.add_argument(
        "--open-hand-entry", action="store_true",
        help="deprecated compatibility flag; direct frame-0 approach is always used",
    )
    parser.add_argument("--entry-distance-m", type=float, default=0.05)
    parser.add_argument("--entry-duration-s", type=float, default=1.0)
    parser.add_argument("--post-grasp-lift-m", type=float, default=0.10)
    parser.add_argument("--post-grasp-lift-duration-s", type=float, default=2.0)
    parser.add_argument(
        "--post-lift-hold-s", "--post-grasp-hold-s",
        dest="post_lift_hold_s", type=float, default=5.0,
        help="seconds to hold the grasp after the +z lift has fully completed",
    )
    parser.add_argument("--post-lift-open-duration-s", type=float, default=2.0)
    parser.add_argument("--min-approach-stage-translation-m", type=float, default=0.05)
    parser.add_argument("--min-approach-stage-rotation-deg", type=float, default=10.0)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--hand-urdf", type=Path, default=DEFAULT_HAND_URDF)
    parser.add_argument("--rpc-addr", default=DEFAULT_OBJECT6D_RPC_ADDR)
    parser.add_argument("--rpc-timeout-ms", type=int, default=300_000)
    parser.add_argument("--rcc-entry", default="image_main.py")
    parser.add_argument("--viser-addr", default=DEFAULT_VISER_ADDR)
    parser.add_argument("--viser-timeout-ms", type=int, default=5_000)
    parser.add_argument("--viser-fps", type=float, default=120.0)
    parser.add_argument("--no-viser", action="store_true")
    parser.add_argument(
        "--viser-on-failure", action="store_true",
        help="on planning failure, show the successful approach prefix on the xArm in Viser",
    )
    parser.add_argument("--save-path", default="object_6d/grasp_retrieval_replay")
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--max-arm-joint-speed-deg-s", type=float, default=15.0)
    parser.add_argument("--max-linear-speed-mps", type=float, default=0.075)
    parser.add_argument("--max-angular-speed-rps", type=float, default=0.4)
    parser.add_argument("--max-hand-speed-rps", type=float, default=1.0)
    parser.add_argument("--command-rate-hz", type=float, default=100.0)
    capture_group = parser.add_mutually_exclusive_group()
    capture_group.add_argument(
        "--record-video", "--record_video",
        dest="record_video", action="store_true",
        help="capture remote cameras plus xArm/Allegro logs for the full execution",
    )
    capture_group.add_argument(
        "--no-record-video", "--no_record_video",
        dest="record_video", action="store_false",
        help="disable full execution capture",
    )
    parser.set_defaults(record_video=True)
    parser.add_argument("--capture-pre-roll-s", type=float, default=1.0)
    parser.add_argument("--capture-post-roll-s", type=float, default=1.0)
    parser.add_argument(
        "--allegro-namespace", "--allegro_namespace", default="",
        help="Allegro ROS namespace; empty matches /allegroHand_0/*",
    )
    parser.add_argument("--execute", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--entry-distance-m": args.entry_distance_m,
        "--entry-duration-s": args.entry_duration_s,
        "--post-grasp-lift-m": args.post_grasp_lift_m,
        "--post-grasp-lift-duration-s": args.post_grasp_lift_duration_s,
        "--post-lift-hold-s": args.post_lift_hold_s,
        "--post-lift-open-duration-s": args.post_lift_open_duration_s,
        "--frame-thickness-m": args.frame_thickness_m,
        "--curobo-seeds": args.curobo_seeds,
        "--speed-scale": args.speed_scale,
        "--max-arm-joint-speed-deg-s": args.max_arm_joint_speed_deg_s,
        "--max-linear-speed-mps": args.max_linear_speed_mps,
        "--max-angular-speed-rps": args.max_angular_speed_rps,
        "--max-hand-speed-rps": args.max_hand_speed_rps,
        "--command-rate-hz": args.command_rate_hz,
        "--capture-pre-roll-s": args.capture_pre_roll_s,
        "--capture-post-roll-s": args.capture_post_roll_s,
        "--approach-hover-clearance-m": args.approach_hover_clearance_m,
        "--viser-fps": args.viser_fps,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0.0]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")
    if args.collision_buffer_m < 0.0:
        raise ValueError("--collision-buffer-m must be non-negative")
    if args.min_approach_stage_translation_m < 0.0:
        raise ValueError("--min-approach-stage-translation-m must be non-negative")
    if args.min_approach_stage_rotation_deg < 0.0:
        raise ValueError("--min-approach-stage-rotation-deg must be non-negative")
    if args.execute and args.action == "all":
        raise ValueError("--action all is plan-only; select one action for execution")
    if args.execute and args.ignore_object_collision:
        raise ValueError("--ignore-object-collision is visualization/diagnostic only; execution refused")
    if args.align_axisymmetric_x and args.object not in {
        "brown_ramen", "coffee_tin", "yellow_wash_tub", "grey_pot", "smallbowl",
        *DISCRETE_VERTICAL_SYMMETRY_OBJECTS,
        *HALF_TURN_VERTICAL_SYMMETRY_OBJECTS,
    }:
        raise ValueError(
            "--align-axisymmetric-x is currently supported only for "
            "brown_ramen, coffee_tin, yellow_wash_tub, grey_pot, smallbowl, "
            "fruit_cutter_green, pastel_light_green_bowl, lilac_food_container, "
            "and spicemill"
        )
    if args.align_point_symmetric and args.object not in POINT_SYMMETRY_OBJECTS:
        raise ValueError(
            "--align-point-symmetric is currently supported only for pingpong"
        )
    if args.align_axisymmetric_x and args.align_point_symmetric:
        raise ValueError("choose only one object symmetry alignment option")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    scene_root = resolve_scene_root(args.dataset_root, args.object, args.scene)
    object_mesh = resolve_object_mesh(scene_root, args.object)
    symmetry_center_object = mesh_vertex_centroid(object_mesh)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir
        or Path(shared_dir) / args.save_path / args.object / timestamp
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.current_object_pose is None:
        try:
            output_dir.relative_to(Path(shared_dir).resolve())
        except ValueError as exc:
            raise ValueError("live capture output must be under shared_data") from exc

    detected_robot_from_object, c2r = capture_object_pose(args, output_dir)
    start_arm_qpos = (
        _joint_vector(args.start_arm_qpos, "--start-arm-qpos", 6)
        if args.start_arm_qpos is not None
        else _read_live_arm_qpos()
    )
    actions = ACTION_NAMES if args.action == "all" else (args.action,)
    successful: dict[
        str,
        tuple[RetrievalTrajectory, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]],
    ] = {}

    for action in actions:
        trajectory = load_retrieval_trajectory(scene_root, action)
        replay_robot_from_object = detected_robot_from_object.copy()
        symmetry_alignment_rad = 0.0
        # Registered vertical (object-Z) symmetries are always enabled; other
        # symmetry families remain explicit opt-in flags.
        symmetry_alignment_enabled = (
            args.align_axisymmetric_x
            or args.align_point_symmetric
            or args.object in DISCRETE_VERTICAL_SYMMETRY_OBJECTS
            or args.object in HALF_TURN_VERTICAL_SYMMETRY_OBJECTS
        )
        if symmetry_alignment_enabled:
            hand_sphere_centers, hand_sphere_radii = hand_collision_spheres_in_palm(
                args.hand_urdf, trajectory.hand_qpos
            )
            finger_extension_axis_in_palm = non_thumb_finger_extension_axis_in_palm(
                args.hand_urdf
            )
            floor_limit_z = FRAME_COLLISION_FLOOR_Z_M
            if args.align_point_symmetric:
                replay_robot_from_object, point_symmetry_label, minimum_hand_z = (
                    select_point_symmetric_object_pose(
                        detected_robot_from_object,
                        trajectory.object_from_palm,
                        hand_sphere_centers_in_palm=hand_sphere_centers,
                        hand_sphere_radii=hand_sphere_radii,
                        minimum_floor_z=floor_limit_z,
                        symmetry_center_object=symmetry_center_object,
                    )
                )
                symmetry_alignment_rad = 0.0 if point_symmetry_label == "identity" else math.pi
                print(
                    "[plan] pingpong point symmetry: selected "
                    f"{point_symmetry_label} half-turn about mesh centroid; "
                    f"minimum hand sphere z={minimum_hand_z:.3f}m "
                    f"(required >= {floor_limit_z:.3f}m)"
                )
            elif (
                args.object == "coffee_tin"
                or args.object in DISCRETE_VERTICAL_SYMMETRY_OBJECTS
                or args.object in HALF_TURN_VERTICAL_SYMMETRY_OBJECTS
            ):
                discrete_axis = "y" if args.object == "coffee_tin" else "z"
                replay_robot_from_object, symmetry_alignment_rad, minimum_hand_z = (
                    select_equivalent_object_y_pose(
                        detected_robot_from_object,
                        trajectory.object_from_palm,
                        hand_sphere_centers_in_palm=hand_sphere_centers,
                        hand_sphere_radii=hand_sphere_radii,
                        minimum_floor_z=floor_limit_z,
                        symmetry_center_object=symmetry_center_object,
                        symmetry_axis=discrete_axis,
                        position_first=(args.object in DISCRETE_VERTICAL_SYMMETRY_OBJECTS),
                        finger_extension_axis_in_palm=finger_extension_axis_in_palm,
                        prefer_positive_finger_extension_robot_axis=(
                            "x"
                            if (
                                args.object in DISCRETE_VERTICAL_SYMMETRY_OBJECTS
                                or args.object in HALF_TURN_VERTICAL_SYMMETRY_OBJECTS
                            )
                            else None
                        ),
                        candidate_angles_rad=(
                            (0.0, math.pi)
                            if args.object in HALF_TURN_VERTICAL_SYMMETRY_OBJECTS
                            else None
                        ),
                    )
                )
                print(
                    f"[plan] {args.object} object-local {discrete_axis.upper()} rotation "
                    "about mesh centroid: "
                    f"selected {math.degrees(symmetry_alignment_rad):.0f}deg; "
                    f"minimum hand sphere "
                    f"z={minimum_hand_z:.3f}m (required >= {floor_limit_z:.3f}m)"
                )
            else:
                symmetry_axis = (
                    "y" if args.object in CONTINUOUS_Y_SYMMETRY_OBJECTS else "x"
                )
                replay_robot_from_object, symmetry_alignment_rad, minimum_hand_z = (
                    align_axisymmetric_object_pose_to_palm_direction(
                        detected_robot_from_object,
                        trajectory.object_from_palm,
                        hand_sphere_centers_in_palm=hand_sphere_centers,
                        hand_sphere_radii=hand_sphere_radii,
                        minimum_floor_z=floor_limit_z,
                        symmetry_center_object=symmetry_center_object,
                        symmetry_axis=symmetry_axis,
                    )
                )
                print(
                    f"[plan] {args.object} {symmetry_axis}-symmetry alignment: "
                    "selected equivalent "
                    f"object pose {math.degrees(symmetry_alignment_rad):+.1f}deg; "
                    f"minimum hand sphere z={minimum_hand_z:.3f}m "
                    f"(required >= {floor_limit_z:.3f}m)"
                )
        palm_poses = reconstruct_palm_trajectory(
            replay_robot_from_object, trajectory.object_from_palm
        )
        if args.open_hand_entry:
            print(
                "[plan] --open-hand-entry is deprecated and ignored: "
                "approaching replay palm frame 0 directly with frame-0 hand qpos"
            )
        trajectory, palm_poses = append_post_grasp_lift(
            trajectory,
            palm_poses,
            args.post_grasp_lift_m,
            args.post_grasp_lift_duration_s,
            args.post_lift_hold_s,
            args.post_lift_open_duration_s,
        )
        print(
            f"[plan] post-grasp lift: +z {args.post_grasp_lift_m * 100:.1f}cm "
            f"over at least {args.post_grasp_lift_duration_s:.1f}s, then hold grasp "
            f"for {args.post_lift_hold_s:.1f}s, then open over at least "
            f"{args.post_lift_open_duration_s:.1f}s"
        )
        link6_poses = palm_poses_to_link6(args.robot_urdf, palm_poses)
        replay_times = retime_replay(
            trajectory.times_s,
            link6_poses,
            trajectory.hand_qpos,
            speed_scale=args.speed_scale,
            max_linear_speed_mps=args.max_linear_speed_mps,
            max_angular_speed_rps=args.max_angular_speed_rps,
            max_hand_speed_rps=args.max_hand_speed_rps,
        )
        world = build_collision_world(
            object_mesh,
            replay_robot_from_object,
            FRAME_CENTER_XYZ_M,
            FRAME_INNER_SIZE_XYZ_M,
            args.frame_thickness_m,
        )
        if args.ignore_object_collision:
            world["mesh"].pop("object", None)
            print(
                "[plan] DIAGNOSTIC: object omitted from cuRobo collisions; "
                "five-sided frame remains active"
            )
        robot_config = configure_curobo_robot(
            args.curobo_config, trajectory.hand_qpos[0], args.collision_buffer_m
        )
        print(f"[plan] {action}: cuRobo current qpos -> reconstructed palm frame 0")
        planning = plan_approach(
            robot_config,
            world,
            start_arm_qpos,
            palm_poses[0],
            robot_urdf=args.robot_urdf,
            first_hand_qpos=trajectory.hand_qpos[0],
            device=args.curobo_device,
            seeds=args.curobo_seeds,
            hover_clearance_m=args.approach_hover_clearance_m,
            min_stage_translation_m=args.min_approach_stage_translation_m,
            min_stage_rotation_rad=math.radians(args.min_approach_stage_rotation_deg),
        )
        planning["object_x_symmetry_alignment_rad"] = symmetry_alignment_rad
        planning["object_symmetry_alignment_rad"] = symmetry_alignment_rad
        planning["symmetry_alignment_mode"] = (
            "point_symmetric_identity_or_180_half_turn_about_mesh_centroid"
            if args.align_point_symmetric
            else "equivalent_object_y_rotation_0_90_180_270_about_mesh_centroid"
            if args.align_axisymmetric_x and args.object == "coffee_tin"
            else "equivalent_object_z_rotation_0_90_180_270_about_mesh_centroid"
            if (
                args.object in DISCRETE_VERTICAL_SYMMETRY_OBJECTS
                or args.object in HALF_TURN_VERTICAL_SYMMETRY_OBJECTS
            )
            else "equivalent_object_y_rotation_about_mesh_centroid"
            if args.align_axisymmetric_x
            and args.object in CONTINUOUS_Y_SYMMETRY_OBJECTS
            else "equivalent_object_x_rotation"
            if args.align_axisymmetric_x
            else "disabled"
        )
        planning["object_relative_wrist_position_modified"] = False
        planning["symmetry_center_object"] = symmetry_center_object
        approach = np.asarray(
            planning.get("trajectory", np.zeros((0, 6))), dtype=np.float64
        )
        replay_arm_qpos = np.empty((0, 6), dtype=np.float64)
        if planning.get("success"):
            try:
                print(f"[plan] {action}: solving continuous replay IK for xArm speed limiting")
                replay_arm_qpos = solve_replay_arm_ik(
                    robot_config,
                    palm_poses,
                    approach[-1],
                    device=args.curobo_device,
                    seeds=args.curobo_seeds,
                )
                replay_times = retime_replay(
                    trajectory.times_s,
                    link6_poses,
                    trajectory.hand_qpos,
                    speed_scale=args.speed_scale,
                    max_linear_speed_mps=args.max_linear_speed_mps,
                    max_angular_speed_rps=args.max_angular_speed_rps,
                    max_hand_speed_rps=args.max_hand_speed_rps,
                    arm_qpos=replay_arm_qpos,
                    max_arm_joint_speed_deg_s=args.max_arm_joint_speed_deg_s,
                )
                floor_check_arm, floor_check_hand, _ = densify_joint_replay(
                    replay_arm_qpos,
                    trajectory.hand_qpos,
                    replay_times,
                    args.command_rate_hz,
                )
                replay_minimum_z, collision_frame, collision_link = (
                    robot_trajectory_floor_clearance(
                        args.robot_urdf, floor_check_arm, floor_check_hand
                    )
                )
                replay_floor_limit_z = FRAME_COLLISION_FLOOR_Z_M
                replay_floor_safe = replay_minimum_z >= replay_floor_limit_z - 1e-9
                planning["replay_floor_check"] = {
                    "success": replay_floor_safe,
                    "minimum_sphere_bottom_z_m": replay_minimum_z,
                    "required_minimum_z_m": replay_floor_limit_z,
                    "minimum_frame": collision_frame,
                    "minimum_link": collision_link,
                    "frames_checked": len(floor_check_arm),
                    "check_rate_hz": args.command_rate_hz,
                }
                if not replay_floor_safe:
                    planning["success"] = False
                    planning["status"] = "REPLAY_FLOOR_COLLISION"
                    planning["partial_approach_qpos"] = approach
                    print(
                        f"[plan] {action}: replay floor collision at frame "
                        f"{collision_frame}, link={collision_link}, sphere bottom "
                        f"z={replay_minimum_z:.3f}m < required "
                        f"{replay_floor_limit_z:.3f}m; execution refused"
                    )
                else:
                    print(
                        f"[plan] {action}: replay floor check passed for "
                        f"{len(floor_check_arm)} command-rate frames; minimum sphere bottom "
                        f"z={replay_minimum_z:.3f}m"
                    )
                    planning["replay_arm_joint_speed_limit_deg_s"] = (
                        args.max_arm_joint_speed_deg_s
                    )
                    planning["replay_execution_mode"] = "joint_servo"
                    planning["execution_command_rate_hz"] = args.command_rate_hz
                    planning["approach_execution_resampling"] = (
                        "linear joint interpolation on retimed path"
                    )
            except Exception as exc:
                planning["success"] = False
                planning["status"] = "REPLAY_IK_FAIL"
                planning["replay_ik_error"] = str(exc)
                planning["partial_approach_qpos"] = approach
                print(f"[plan] {action}: replay IK failed; execution refused: {exc}")
        planning["entry_policy"] = "direct to replay palm frame 0 with frame-0 hand qpos"
        planning["post_grasp_lift"] = {
            "axis_robot": [0.0, 0.0, 1.0],
            "distance_m": args.post_grasp_lift_m,
            "requested_duration_s": args.post_grasp_lift_duration_s,
            "hold_after_lift_s": args.post_lift_hold_s,
            "release_duration_s": args.post_lift_open_duration_s,
            "hand_policy": "hold final grasp during lift, then open at lifted pose",
            "collision_policy": (
                "continuous replay IK plus full-robot floor collision-sphere check; "
                "object contact remains intentional"
            ),
        }
        approach_times = (
            _approach_times(approach, args.max_arm_joint_speed_deg_s)
            if len(approach) >= 2
            else np.zeros(0, dtype=np.float64)
        )
        plan_path = save_plan(
            output_dir,
            trajectory,
            replay_robot_from_object,
            palm_poses,
            link6_poses,
            replay_times,
            replay_arm_qpos,
            approach_times,
            start_arm_qpos,
            world,
            planning,
        )
        print(f"[plan] {action}: success={bool(planning.get('success'))}; saved {plan_path}")
        if not args.no_viser:
            preview_approach = approach
            preview_replay_arm = replay_arm_qpos
            preview_hand = trajectory.hand_qpos
            if planning.get("success"):
                preview_approach, _, _ = densify_joint_replay(
                    approach,
                    np.tile(trajectory.hand_qpos[0], (len(approach), 1)),
                    approach_times,
                    30.0,
                )
                preview_replay_arm, preview_hand, _ = densify_joint_replay(
                    replay_arm_qpos, trajectory.hand_qpos, replay_times, 30.0
                )
            publish_viser_preview(
                address=args.viser_addr,
                timeout_ms=args.viser_timeout_ms,
                object_name=args.object,
                object_mesh=object_mesh,
                robot_from_object=replay_robot_from_object,
                robot_urdf=args.robot_urdf,
                hand_urdf=args.hand_urdf,
                start_arm_qpos=start_arm_qpos,
                robot_from_palm=palm_poses,
                hand_qpos=preview_hand,
                action=action,
                planning_success=bool(planning.get("success")),
                approach_qpos=preview_approach,
                replay_arm_qpos=preview_replay_arm,
                frame_cuboids=world["cuboid"],
                viser_fps=args.viser_fps,
                failed_approach_qpos=(
                    np.asarray(planning.get("partial_approach_qpos", np.empty((0, 6))))
                    if args.viser_on_failure and not planning.get("success") else None
                ),
            )
        if planning.get("success"):
            successful[action] = (
                trajectory, palm_poses, link6_poses, replay_times, replay_arm_qpos, planning
            )

    np.savez_compressed(
        output_dir / "object_frame.npz",
        robot_from_object=detected_robot_from_object,
        C2R=c2r,
        object_mesh=str(object_mesh),
    )
    if not args.execute:
        print(f"[plan] complete; no robot commands were sent. output={output_dir}")
        return
    if args.action not in successful:
        raise RuntimeError("cuRobo did not produce a valid collision-free approach; execution refused")
    trajectory, _, _, replay_times, replay_arm_qpos, planning = successful[args.action]
    if not execution_confirmation_accepted(
        input("Type PLAY (or y) to execute the collision-checked approach and grasp replay: ")
    ):
        print("[execute] cancelled; no robot commands were sent")
        return
    approach = np.asarray(planning["trajectory"], dtype=np.float64)
    capture_session = None
    capture_started = False
    capture_dir = output_dir / args.action / "execution_capture"
    try:
        if args.record_video:
            from paradex.dataset_acqusition.capture import CaptureSession

            try:
                capture_rel = capture_dir.relative_to(Path(shared_dir).resolve())
            except ValueError as exc:
                raise ValueError(
                    "execution capture output must be located under shared_data"
                ) from exc
            capture_session = CaptureSession(
                camera=True,
                realsense=False,
                timestamp=True,
            )
            capture_session.start(capture_rel.as_posix())
            capture_started = True
            np.save(capture_dir / "C2R.npy", c2r)
            print(f"[capture] full execution recording -> {capture_dir}")
            time.sleep(args.capture_pre_roll_s)

        execute_plan(
            approach,
            _approach_times(approach, args.max_arm_joint_speed_deg_s),
            replay_arm_qpos,
            trajectory.hand_qpos,
            replay_times,
            max_hand_speed_rps=args.max_hand_speed_rps,
            command_rate_hz=args.command_rate_hz,
            allegro_namespace=args.allegro_namespace,
            capture_raw_dir=(capture_dir / "raw" if capture_started else None),
        )
        if capture_started:
            time.sleep(args.capture_post_roll_s)
    finally:
        if capture_session is not None:
            try:
                if capture_started:
                    capture_session.stop()
            finally:
                capture_session.end()
            if capture_started:
                print(f"[capture] saved full execution recording to {capture_dir}")
    print("[execute] replay complete")


if __name__ == "__main__":
    main()
