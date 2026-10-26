"""Per-object 6D pose normalization — the same technique the H2R grasp
pipeline (hrdex_afford / build_scene_assets) applies to capture data.

A normalization mode rolls / canonicalizes the object pose so the optimizer
and the paired dataset see a consistent object orientation:
  none      : no change
  x_roll    : roll about the aligned object's local +X so +Y points world +X
              (cylindrical objects)
  axis_sign : 24-fold cuboid-symmetry canonicalization (box-like objects)
  sphere    : replace the aligned rotation with identity (spherical objects)

All functions are numpy-only. `pose_C` is the object pose as returned by the
6D estimator; `c2r` is the 4x4 camera->robot matrix (C2R.npy); `T_align` is
`<mesh>_viser_align.npy` (identity if absent).
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

MODES = ("none", "x_roll", "axis_sign", "sphere")
PREFERRED_WRIST_FACING_ANGLE_RAD = np.deg2rad(58.5)
# The potato-masher half-turn annotation and captured pregrasp can place the
# wrist several millimetres below the annotated symmetry center even though the
# grasp is effectively level. Captured values are logged to 1 mm precision, so
# use 6 mm to safely include an observed ``-0.005 m`` display value. Keep every
# other object's upper-half test strict.
WRIST_HEIGHT_TOLERANCE_M = {"potato_masher": 0.006}


def normalize_known_object_symmetry(
    mesh_name: str,
    pose_robot: np.ndarray,
    wrist_pose_object: Optional[np.ndarray] = None,
    table_normal_robot: np.ndarray = np.array([0.0, 0.0, 1.0]),
    max_wrist_facing_angle_rad: float = np.deg2rad(90.0),
    wrist_robotward_axis_local: Optional[np.ndarray] = None,
    reference_wrist_pose_robot: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Choose a deterministic representative of a registered mesh symmetry.

    Symmetry metadata is defined in the original ``mesh_new`` object frame.
    Packed H2R grasps do not carry a source object pose in robot coordinates,
    so robot-frame identity rotation is used as the deterministic reference.
    """
    from paradex.retargetor.experiment.replay_pose_retrieval import (
        align_pose_by_continuous_symmetry,
    )
    from scipy.spatial.transform import Rotation
    from src.object6d.object_symmetry import (
        AXIS_FLIP_SYMMETRIC_OBJECTS,
        CONTINUOUS_SYMMETRY_AXIS_OBJECT,
        CONTINUOUS_SYMMETRY_CENTER_OBJECT,
        CUBOID_SYMMETRY_CENTER_OBJECT,
        DISCRETE_HALF_TURN_SYMMETRY_OBJECT,
    )

    object_name = str(mesh_name).strip().lower()
    axis = CONTINUOUS_SYMMETRY_AXIS_OBJECT.get(object_name)
    continuous_center = CONTINUOUS_SYMMETRY_CENTER_OBJECT.get(object_name)
    current = np.asarray(pose_robot, dtype=np.float64)
    cuboid_center = CUBOID_SYMMETRY_CENTER_OBJECT.get(object_name)
    cuboid_symmetry = cuboid_center is not None
    half_turn = DISCRETE_HALF_TURN_SYMMETRY_OBJECT.get(object_name)
    half_turn_symmetry = half_turn is not None
    if axis is None and not cuboid_symmetry and not half_turn_symmetry:
        return current.copy(), None
    max_facing_limit = float(max_wrist_facing_angle_rad)
    if (
        not np.isfinite(max_facing_limit)
        or max_facing_limit < 0.0
        or max_facing_limit > np.pi
    ):
        raise ValueError("max_wrist_facing_angle_rad must be in [0, pi]")

    if cuboid_symmetry or half_turn_symmetry:
        if cuboid_symmetry:
            discrete_rotations = _CUBOID_ROTS
            discrete_center = np.asarray(cuboid_center, dtype=np.float64)
            symmetry_type = "cuboid_axis_sign"
            normalize_mode = "cuboid_object_symmetry"
        else:
            half_turn_axis = np.asarray(half_turn["axis"], dtype=np.float64)
            half_turn_axis /= np.linalg.norm(half_turn_axis)
            discrete_rotations = (
                np.eye(3, dtype=np.float64),
                Rotation.from_rotvec(half_turn_axis * np.pi).as_matrix(),
            )
            discrete_center = np.asarray(half_turn["center"], dtype=np.float64)
            symmetry_type = "discrete_half_turn"
            normalize_mode = "half_turn_object_symmetry"
        candidates = []
        orientation_costs = []
        for rotation in discrete_rotations:
            symmetry_transform = np.eye(4, dtype=np.float64)
            symmetry_transform[:3, :3] = rotation
            symmetry_transform[:3, 3] = (
                discrete_center - rotation @ discrete_center
            )
            candidate = current @ symmetry_transform
            candidates.append(candidate)
            orientation_costs.append(3.0 - float(np.trace(candidate[:3, :3])))
        canonical_index = int(np.argmin(orientation_costs))
        selected_index = canonical_index
        metadata = {
            "object": object_name,
            "symmetry_type": symmetry_type,
            "normalize_mode": normalize_mode,
            "symmetry_center_object": discrete_center.tolist(),
            "variant_index": canonical_index,
            "variant_count": len(discrete_rotations),
            "reference": "robot_identity_rotation",
        }
        if half_turn_symmetry:
            metadata["axis_object"] = half_turn_axis.tolist()
            metadata["angles_rad"] = [0.0, float(np.pi)]
        if wrist_pose_object is not None:
            wrist_object = np.asarray(wrist_pose_object, dtype=np.float64)
            normal = np.asarray(table_normal_robot, dtype=np.float64).reshape(-1)
            if wrist_object.shape != (4, 4) or not np.all(np.isfinite(wrist_object)):
                raise ValueError("wrist_pose_object must be a finite 4x4 transform")
            if normal.shape != (3,) or not np.all(np.isfinite(normal)):
                raise ValueError("table_normal_robot must be a finite 3-vector")
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm <= 1e-12:
                raise ValueError("table_normal_robot must be non-zero")
            normal /= normal_norm
            robotward_axis_local = None
            if wrist_robotward_axis_local is not None:
                robotward_axis_local = np.asarray(
                    wrist_robotward_axis_local, dtype=np.float64
                ).reshape(-1)
                if (
                    robotward_axis_local.shape != (3,)
                    or not np.all(np.isfinite(robotward_axis_local))
                    or np.linalg.norm(robotward_axis_local) <= 1e-12
                ):
                    raise ValueError(
                        "wrist_robotward_axis_local must be a finite non-zero 3-vector"
                    )
                robotward_axis_local /= np.linalg.norm(robotward_axis_local)
            object_center_h = np.concatenate([discrete_center, [1.0]])
            object_position = (current @ object_center_h)[:3]
            wrist_positions = np.asarray(
                [(candidate @ wrist_object)[:3, 3] for candidate in candidates]
            )
            heights = (wrist_positions - object_position) @ normal
            planar_positions = wrist_positions - np.outer(wrist_positions @ normal, normal)
            planar_distances = np.linalg.norm(planar_positions, axis=1)
            object_planar = object_position - normal * float(normal @ object_position)
            object_to_base = -object_planar
            base_direction_norm = float(np.linalg.norm(object_to_base))
            if robotward_axis_local is None:
                wrist_directions = wrist_positions - object_position
                facing_source = "object_to_wrist_position"
            else:
                wrist_directions = np.asarray([
                    (candidate @ wrist_object)[:3, :3] @ robotward_axis_local
                    for candidate in candidates
                ])
                facing_source = "wrist_orientation_local_axis"
            wrist_directions -= np.outer(wrist_directions @ normal, normal)
            wrist_direction_norms = np.linalg.norm(wrist_directions, axis=1)
            facing_angles = np.full(len(candidates), np.inf, dtype=np.float64)
            home_orientation_costs = np.zeros(len(candidates), dtype=np.float64)
            valid = wrist_direction_norms > 1e-9
            if base_direction_norm > 1e-9:
                cosines = (
                    wrist_directions[valid] @ object_to_base
                    / (wrist_direction_norms[valid] * base_direction_norm)
                )
                facing_angles[valid] = np.arccos(np.clip(cosines, -1.0, 1.0))
            reference_wrist = None
            if reference_wrist_pose_robot is not None:
                reference_wrist = np.asarray(
                    reference_wrist_pose_robot, dtype=np.float64
                )
                if (
                    reference_wrist.shape != (4, 4)
                    or not np.all(np.isfinite(reference_wrist))
                ):
                    raise ValueError(
                        "reference_wrist_pose_robot must be a finite 4x4 transform"
                    )
                for index, candidate in enumerate(candidates):
                    wrist_rotation = (candidate @ wrist_object)[:3, :3]
                    home_orientation_costs[index] = Rotation.from_matrix(
                        reference_wrist[:3, :3].T @ wrist_rotation
                    ).magnitude()
            max_facing = max_facing_limit
            height_tolerance = float(WRIST_HEIGHT_TOLERANCE_M.get(object_name, 1e-6))
            feasible = (heights >= -height_tolerance) & (facing_angles <= max_facing)
            if np.any(feasible):
                pool = np.flatnonzero(feasible)
                local_order = np.lexsort((
                    np.asarray(orientation_costs)[pool],
                    -heights[pool],
                    facing_angles[pool],
                    home_orientation_costs[pool],
                ))
                selected_index = int(pool[int(local_order[0])])
            else:
                upper = np.flatnonzero(heights >= -height_tolerance)
                pool = upper if len(upper) else np.arange(len(candidates))
                local_order = np.lexsort((
                    np.asarray(orientation_costs)[pool],
                    -heights[pool],
                    facing_angles[pool],
                    home_orientation_costs[pool],
                ))
                selected_index = int(pool[int(local_order[0])])
            metadata.update({
                "variant_index": selected_index,
                "canonical_variant_index": canonical_index,
                "wrist_relative_height_before_m": float(heights[canonical_index]),
                "wrist_relative_height_after_m": float(heights[selected_index]),
                "wrist_planar_distance_to_robot_base_m": float(
                    planar_distances[selected_index]
                ),
                "wrist_facing_angle_rad": float(facing_angles[selected_index]),
                "max_wrist_facing_angle_rad": max_facing,
                "wrist_height_tolerance_m": height_tolerance,
                "wrist_constraints_satisfied": bool(feasible[selected_index]),
                "wrist_facing_source": facing_source,
                "wrist_home_orientation_distance_rad": float(
                    home_orientation_costs[selected_index]
                ),
                "wrist_selection_priority": (
                    "home_orientation_then_facing_angle_then_height"
                    if reference_wrist is not None
                    else "facing_angle_then_height"
                ),
                "wrist_robotward_axis_local": (
                    robotward_axis_local.tolist()
                    if robotward_axis_local is not None else None
                ),
                "reference": "pregrasp_wrist_robot_facing",
            })
        if half_turn_symmetry:
            # Applying the same centered half-turn once more reaches the only
            # other equivalent pose, regardless of which variant was selected.
            metadata["alternate_wrist_symmetry_delta_rad"] = float(np.pi)
            metadata["alternate_angle_rad"] = float(
                np.pi if selected_index == 0 else 0.0
            )
        return candidates[selected_index], metadata

    reference = np.eye(4, dtype=np.float64)
    reference[:3, 3] = current[:3, 3]
    aligned, angle, flipped = align_pose_by_continuous_symmetry(
        reference,
        current,
        continuous_symmetry_axis_object=axis,
        continuous_symmetry_allow_axis_flip=(
            object_name in AXIS_FLIP_SYMMETRIC_OBJECTS
        ),
    )
    if continuous_center is not None:
        if flipped:
            raise ValueError(
                f"centered continuous symmetry with axis flip is unsupported: {object_name}"
            )
        canonical_spin = np.eye(4, dtype=np.float64)
        canonical_spin[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        canonical_spin[:3, 3] = (
            continuous_center - canonical_spin[:3, :3] @ continuous_center
        )
        aligned = current @ canonical_spin
    metadata = {
        "object": object_name,
        "symmetry_type": "continuous_axis",
        "normalize_mode": "continuous_object_symmetry",
        "axis_object": np.asarray(axis, dtype=np.float64).tolist(),
        "symmetry_center_object": (
            np.asarray(continuous_center, dtype=np.float64).tolist()
            if continuous_center is not None else [0.0, 0.0, 0.0]
        ),
        "angle_rad": float(angle),
        "axis_flipped": bool(flipped),
        "reference": "robot_identity_rotation",
    }
    if wrist_pose_object is None:
        return aligned, metadata

    wrist_object = np.asarray(wrist_pose_object, dtype=np.float64)
    if wrist_object.shape != (4, 4) or not np.all(np.isfinite(wrist_object)):
        raise ValueError("wrist_pose_object must be a finite 4x4 transform")
    normal = np.asarray(table_normal_robot, dtype=np.float64).reshape(-1)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("table_normal_robot must be a finite 3-vector")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        raise ValueError("table_normal_robot must be non-zero")
    normal /= normal_norm
    robotward_axis_local = None
    if wrist_robotward_axis_local is not None:
        robotward_axis_local = np.asarray(
            wrist_robotward_axis_local, dtype=np.float64
        ).reshape(-1)
        if (
            robotward_axis_local.shape != (3,)
            or not np.all(np.isfinite(robotward_axis_local))
            or np.linalg.norm(robotward_axis_local) <= 1e-12
        ):
            raise ValueError(
                "wrist_robotward_axis_local must be a finite non-zero 3-vector"
            )
        robotward_axis_local /= np.linalg.norm(robotward_axis_local)
    reference_wrist = None
    if reference_wrist_pose_robot is not None:
        reference_wrist = np.asarray(reference_wrist_pose_robot, dtype=np.float64)
        if reference_wrist.shape != (4, 4) or not np.all(np.isfinite(reference_wrist)):
            raise ValueError("reference_wrist_pose_robot must be a finite 4x4 transform")

    # A continuous symmetry gives a one-DOF orbit of physically equivalent
    # grasp placements. First enforce upper-half and facing-cone feasibility.
    # When the caller provides the HOME wrist pose, first stay inside a 58.5-deg
    # preferred cone, then minimize rotation away from HOME. The caller's
    # max_facing remains the hard constraint. This avoids both extremes seen
    # on the real setup: forcing zero-degree facing wound joint6 to its limit,
    # while selecting directly on the 90-degree boundary made the staged
    # approach lose IK near the pregrasp. The small margin below 60 degrees
    # also keeps the selected candidate away from the observed joint6 limit.
    axis_unit = np.asarray(axis, dtype=np.float64)
    axis_unit /= np.linalg.norm(axis_unit)
    offsets = np.linspace(-np.pi, np.pi, 1440, endpoint=False, dtype=np.float64)
    heights = np.empty(len(offsets), dtype=np.float64)
    planar_distances = np.empty(len(offsets), dtype=np.float64)
    facing_angles = np.full(len(offsets), np.inf, dtype=np.float64)
    home_orientation_costs = np.zeros(len(offsets), dtype=np.float64)
    candidates = []
    symmetry_center = (
        np.asarray(continuous_center, dtype=np.float64)
        if continuous_center is not None else np.zeros(3, dtype=np.float64)
    )
    object_position = (aligned @ np.concatenate([symmetry_center, [1.0]]))[:3]
    for index, offset in enumerate(offsets):
        spin = np.eye(4, dtype=np.float64)
        spin[:3, :3] = Rotation.from_rotvec(axis_unit * offset).as_matrix()
        spin[:3, 3] = symmetry_center - spin[:3, :3] @ symmetry_center
        candidate = aligned @ spin
        candidates.append(candidate)
        wrist_position = (candidate @ wrist_object)[:3, 3]
        heights[index] = float(normal @ (wrist_position - object_position))
        planar_position = wrist_position - normal * float(normal @ wrist_position)
        planar_distances[index] = float(np.linalg.norm(planar_position))
        object_planar = object_position - normal * float(normal @ object_position)
        object_to_base = -object_planar
        if robotward_axis_local is None:
            wrist_direction = wrist_position - object_position
            facing_source = "object_to_wrist_position"
        else:
            wrist_direction = (
                (candidate @ wrist_object)[:3, :3] @ robotward_axis_local
            )
            facing_source = "wrist_orientation_local_axis"
        wrist_direction -= normal * float(normal @ wrist_direction)
        denominator = float(np.linalg.norm(object_to_base) * np.linalg.norm(wrist_direction))
        if denominator > 1e-9:
            cosine = float(np.dot(object_to_base, wrist_direction) / denominator)
            facing_angles[index] = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        if reference_wrist is not None:
            wrist_rotation = (candidate @ wrist_object)[:3, :3]
            home_orientation_costs[index] = Rotation.from_matrix(
                reference_wrist[:3, :3].T @ wrist_rotation
            ).magnitude()
    max_facing = max_facing_limit
    feasible = (heights >= -1e-6) & (facing_angles <= max_facing)
    preferred_pool = np.array([], dtype=np.int64)
    if np.any(feasible):
        preferred_facing = min(PREFERRED_WRIST_FACING_ANGLE_RAD, max_facing)
        preferred_feasible = feasible & (facing_angles <= preferred_facing)
        preferred_pool = np.flatnonzero(
            preferred_feasible if np.any(preferred_feasible) else feasible
        )
        pool = preferred_pool
        local_order = np.lexsort((
            np.abs(offsets[pool]),
            -heights[pool],
            facing_angles[pool],
            home_orientation_costs[pool],
        ))
        selected_index = int(pool[int(local_order[0])])
    else:
        upper = np.flatnonzero(heights >= -1e-6)
        pool = upper if len(upper) else np.arange(len(offsets))
        local_order = np.lexsort((
            np.abs(offsets[pool]),
            -heights[pool],
            facing_angles[pool],
            home_orientation_costs[pool],
        ))
        selected_index = int(pool[int(local_order[0])])
    selected = candidates[selected_index]
    selected_offset = float(offsets[selected_index])
    wrist_before = (aligned @ wrist_object)[:3, 3]
    wrist_after = (selected @ wrist_object)[:3, 3]
    total_angle = float(np.arctan2(
        np.sin(float(angle) + selected_offset),
        np.cos(float(angle) + selected_offset),
    ))
    alternate_delta = None
    alternate_angle = None
    if reference_wrist is not None and len(preferred_pool) > 1:
        deltas = np.arctan2(
            np.sin(offsets[preferred_pool] - selected_offset),
            np.cos(offsets[preferred_pool] - selected_offset),
        )
        alternate_local_index = int(np.argmax(np.abs(deltas)))
        candidate_delta = float(deltas[alternate_local_index])
        if abs(candidate_delta) > np.deg2rad(5.0):
            alternate_delta = candidate_delta
            alternate_angle = float(np.arctan2(
                np.sin(total_angle + candidate_delta),
                np.cos(total_angle + candidate_delta),
            ))
    metadata.update({
        "angle_rad": total_angle,
        "canonical_angle_rad": float(angle),
        "wrist_symmetry_offset_rad": selected_offset,
        "alternate_wrist_symmetry_delta_rad": alternate_delta,
        "alternate_angle_rad": alternate_angle,
        "wrist_relative_height_before_m": float(normal @ (wrist_before - object_position)),
        "wrist_relative_height_after_m": float(normal @ (wrist_after - object_position)),
        "wrist_planar_distance_to_robot_base_m": float(
            planar_distances[selected_index]
        ),
        "wrist_facing_angle_rad": float(facing_angles[selected_index]),
        "max_wrist_facing_angle_rad": max_facing,
        "preferred_wrist_facing_angle_rad": float(
            min(PREFERRED_WRIST_FACING_ANGLE_RAD, max_facing)
        ),
        "wrist_constraints_satisfied": bool(feasible[selected_index]),
        "wrist_home_orientation_distance_rad": float(
            home_orientation_costs[selected_index]
        ),
        "wrist_selection_priority": (
            "preferred_facing_cone_then_home_orientation_then_facing_angle"
            if reference_wrist is not None
            else "facing_angle_then_height"
        ),
        "wrist_facing_source": facing_source,
        "wrist_robotward_axis_local": (
            robotward_axis_local.tolist()
            if robotward_axis_local is not None else None
        ),
        "reference": "pregrasp_wrist_robot_facing",
    })
    return selected, metadata


def has_known_object_symmetry(mesh_name: str) -> bool:
    """Return whether ``mesh_name`` has a registered rotational symmetry."""
    from src.object6d.object_symmetry import (
        CONTINUOUS_SYMMETRY_AXIS_OBJECT,
        CUBOID_SYMMETRY_CENTER_OBJECT,
        DISCRETE_HALF_TURN_SYMMETRY_OBJECT,
    )

    object_name = str(mesh_name).strip().lower()
    return (
        object_name in CONTINUOUS_SYMMETRY_AXIS_OBJECT
        or object_name in CUBOID_SYMMETRY_CENTER_OBJECT
        or object_name in DISCRETE_HALF_TURN_SYMMETRY_OBJECT
    )


def load_packed_pregrasp_wrist(
    result_path: str,
    source: str,
    particle: Optional[int],
    pack_to_pose: np.ndarray,
) -> np.ndarray:
    """Load the first packed wrist transform in the current object frame."""
    from scipy.spatial.transform import Rotation

    packed = np.load(result_path, allow_pickle=True)
    try:
        best_key = f"{source}_traj_q"
        all_key = f"{source}_traj_q_all"
        if particle is None:
            if best_key not in packed.files:
                raise KeyError(f"{best_key} missing in {result_path}")
            trajectory = np.asarray(packed[best_key], dtype=np.float64)
        else:
            if all_key not in packed.files:
                raise KeyError(f"{all_key} missing in {result_path}")
            trajectory = np.asarray(
                packed[all_key][int(particle)], dtype=np.float64
            )
    finally:
        packed.close()
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 6:
        raise ValueError(
            f"packed trajectory must have shape (W, >=6), got {trajectory.shape}"
        )
    first = trajectory[0]
    wrist = np.eye(4, dtype=np.float64)
    wrist[:3, :3] = Rotation.from_euler("XYZ", first[3:6]).as_matrix()
    wrist[:3, 3] = first[:3]
    bridge = np.asarray(pack_to_pose, dtype=np.float64)
    if bridge.shape != (4, 4):
        raise ValueError(f"pack_to_pose must be 4x4, got {bridge.shape}")
    return bridge @ wrist


def _canonicalize_axis_sign(vx: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Pick a consistent sign for the X axis: prefer world +Z, then +X, +Y."""
    for c in (2, 0, 1):
        if vx[c] > eps:
            return vx
        if vx[c] < -eps:
            return -vx
    return vx


def normalize_x_axis_roll(pose_R_aligned: np.ndarray,
                          target_world_axis: np.ndarray = np.array([1., 0., 0.]),
                          canonicalize_flip: bool = True) -> np.ndarray:
    """Roll about the aligned object's local +X so the aligned +Y points along
    `target_world_axis`. Translation unchanged."""
    out = pose_R_aligned.copy()
    R = out[:3, :3]
    vx = R[:, 0]
    nx = np.linalg.norm(vx)
    if nx < 1e-9:
        return out
    vx = vx / nx
    if canonicalize_flip:
        vx = _canonicalize_axis_sign(vx)
    proj = target_world_axis - np.dot(target_world_axis, vx) * vx
    pn = np.linalg.norm(proj)
    if pn < 1e-6:
        return out                       # X colinear with target: roll ambiguous
    vy = proj / pn
    vz = np.cross(vx, vy)
    out[:3, :3] = np.column_stack([vx, vy, vz])
    return out


def _axis_flip_group() -> np.ndarray:
    """Klein four-group: identity + 180° rotations about each principal axis."""
    diags = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                     dtype=float)
    return np.stack([np.diag(d) for d in diags], axis=0)


_CUBOID_ROTS = _axis_flip_group()


def normalize_cuboid_symmetry(pose_R_aligned: np.ndarray,
                              target_R: np.ndarray = np.eye(3)) -> np.ndarray:
    """Pick the cuboid-symmetry variant whose rotation is closest to target_R."""
    out = pose_R_aligned.copy()
    R_cur = out[:3, :3]
    Rvs = np.einsum("ij,kjl->kil", R_cur, _CUBOID_ROTS)
    scores = np.einsum("kii->k", Rvs @ target_R.T)
    out[:3, :3] = R_cur @ _CUBOID_ROTS[int(np.argmax(scores))]
    return out


def normalize_pose_x_in_world(pose_C, c2r, T_align):
    r2c = np.linalg.inv(c2r)
    pose_R_aligned = r2c @ pose_C @ np.linalg.inv(T_align)
    return c2r @ normalize_x_axis_roll(pose_R_aligned) @ T_align


def normalize_pose_cuboid_in_world(pose_C, c2r, T_align):
    r2c = np.linalg.inv(c2r)
    pose_R_aligned = r2c @ pose_C @ np.linalg.inv(T_align)
    return c2r @ normalize_cuboid_symmetry(pose_R_aligned) @ T_align


def normalize_pose_full_in_world(pose_C, c2r, T_align):
    r2c = np.linalg.inv(c2r)
    pose_R_aligned = r2c @ pose_C @ np.linalg.inv(T_align)
    pose_R_aligned[:3, :3] = np.eye(3)
    return c2r @ pose_R_aligned @ T_align


def normalize_object_pose(mode: str, pose_C: np.ndarray, c2r: np.ndarray,
                          T_align: np.ndarray) -> np.ndarray:
    """Dispatch on `mode`; returns the normalized 4x4 object pose (camera/world
    frame, same frame as the input pose_C)."""
    pose_C = np.asarray(pose_C, dtype=np.float64)
    c2r = np.asarray(c2r, dtype=np.float64)
    T_align = np.asarray(T_align, dtype=np.float64)
    if mode == "none":
        return pose_C.copy()
    if mode == "x_roll":
        return normalize_pose_x_in_world(pose_C, c2r, T_align)
    if mode == "axis_sign":
        return normalize_pose_cuboid_in_world(pose_C, c2r, T_align)
    if mode == "sphere":
        return normalize_pose_full_in_world(pose_C, c2r, T_align)
    raise ValueError(f"unknown normalize mode: {mode} (choices: {MODES})")
