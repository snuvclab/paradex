#!/usr/bin/env python3
"""Render a recorded robot hand trajectory interacting with a simulated object."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

from physics_eval import load_prediction_rows, set_free_pose, set_joint_targets
from recover_v0 import (
    ASSETS,
    ROBOTS,
    compute_link_transforms,
    inspire_action_to_qpos_dof6,
    inspire_f1_action_to_qpos_dof6,
    numeric,
    parse_urdf,
)
from render_zero_force_videos import _compact_mesh, geom_local_mesh, transcode_h264
from vendor_model_builder import build_model, build_model_from_template


BASE = Path(__file__).resolve().parent
CAPTURE = Path("/home/temp_id/shared_data/capture/eccv2026/v0")
OBJECT_MESH_ROOT = Path("/home/temp_id/shared_data/mesh_new")
HAND_COLLISION_ROOT = Path("/home/temp_id/human_robot_retarget/assets")
HAND_COLLISION_ASSETS = {
    "allegro_v5": (
        HAND_COLLISION_ROOT / "allegro_v5_physics/allegro_hand_right.urdf",
        HAND_COLLISION_ROOT / "allegro_v5_physics/allegro_hand_right.srdf",
    ),
    "inspire_dftp": (
        HAND_COLLISION_ROOT / "inspire_dftp_refined/inspire_hand_right.urdf",
        None,
    ),
    "inspire_f1": (
        HAND_COLLISION_ROOT / "inspire_f1_physics/inspire_hand_right.urdf",
        HAND_COLLISION_ROOT / "inspire_f1_physics/inspire_hand_right.srdf",
    ),
}
HAND_COLLISION_TEMPLATES = {
    collection: BASE / "collision_assets" / collection
    for collection in ("allegro_v5", "inspire_dftp", "inspire_f1")
}
# These non-adjacent pairs touch during verified, recorded grasp trajectories.
# Excluding them prevents collision response from opposing valid closure commands.
NORMAL_TRAJECTORY_SELF_COLLISION_EXCLUDES = {
    "allegro_v5": [
        ("link_15_0", "link_3_0_tip"),
        ("link_15_0_tip", "link_3_0"),
        ("link_15_0_tip", "link_3_0_tip"),
        ("link_15_0_tip", "link_7_0_tip"),
    ],
    "inspire_dftp": [
        ("right_middle_2", "right_thumb_3"),
        ("right_middle_2", "right_thumb_4"),
        ("right_index_2", "right_thumb_4"),
    ],
    "inspire_f1": [
        ("middle_tip", "right_thumb_4"),
        ("middle_tip", "thumb_tip"),
        ("right_middle_2", "right_thumb_4"),
        ("right_middle_2", "thumb_tip"),
        ("right_ring_2", "right_thumb_4"),
    ],
}
WIDTH = 960
HEIGHT = 720
HEADER_H = 108
FOOTER_H = 38
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
SMALL_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
FLOOR_RAISE_M = 0.016


@dataclass(frozen=True)
class HandMesh:
    geom_id: int
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class TrajectorySample:
    source_frame: int
    time_s: float
    hand_qpos: np.ndarray
    robot_from_palm: np.ndarray


@dataclass(frozen=True)
class PhysicsFrame:
    geom_poses: dict[int, tuple[np.ndarray, np.ndarray]]
    object_pose: np.ndarray
    contact_count: int
    self_contact_count: int
    object_contact_count: int
    min_self_contact_distance_m: float
    self_contact_pair_depth_m: dict[str, float]


def nearest_indices(reference_times: np.ndarray, query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_times = np.asarray(reference_times, dtype=float).reshape(-1)
    query_times = np.asarray(query_times, dtype=float).reshape(-1)
    if not len(reference_times) or np.any(np.diff(reference_times) < 0):
        raise ValueError("state timestamps must be nonempty and sorted")
    right = np.searchsorted(reference_times, query_times, side="left")
    right = np.clip(right, 0, len(reference_times) - 1)
    left = np.clip(right - 1, 0, len(reference_times) - 1)
    choose_left = np.abs(reference_times[left] - query_times) <= np.abs(reference_times[right] - query_times)
    indices = np.where(choose_left, left, right)
    return indices, reference_times[indices] - query_times


def anchor_object_frame(target_pose: np.ndarray, target_object_from_palm: np.ndarray) -> np.ndarray:
    """Choose canonical_from_object so the recorded trajectory lands on the paired label."""
    return np.asarray(target_pose, dtype=float) @ np.linalg.inv(np.asarray(target_object_from_palm, dtype=float))


def robot_from_world_pose(world_from_robot: np.ndarray, world_from_pose: np.ndarray) -> np.ndarray:
    """Express a v2 world-frame pose in the robot base frame using C2R."""
    return np.linalg.inv(np.asarray(world_from_robot, dtype=float)) @ np.asarray(world_from_pose, dtype=float)


def scene_for_row(row: dict[str, Any]) -> Path:
    if row.get("robot_scene"):
        return Path(row["robot_scene"])
    return CAPTURE / row["collection"] / row["object_id"] / str(row["robot_episode"])


def hand_paths(collection: str) -> tuple[str, str]:
    if collection == "inspire_f1":
        return "raw/hand/right_joint_states.npy", "raw/hand/right_joint_states_time.npy"
    return "raw/hand/position.npy", "raw/hand/time.npy"


def convert_hand_qpos(collection: str, raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float).reshape(1, -1)
    if collection == "inspire_dftp":
        return np.asarray(inspire_action_to_qpos_dof6(raw), dtype=float).reshape(-1)
    if collection == "inspire_f1":
        return np.asarray(inspire_f1_action_to_qpos_dof6(raw), dtype=float).reshape(-1)
    return raw.reshape(-1)


def load_object_poses(scene: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    path = scene / "object_6d_pose_v2.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing v2 object pose archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            if not key.startswith("frame_"):
                continue
            pose = np.asarray(archive[key], dtype=float)
            if pose.shape == (4, 4) and np.isfinite(pose).all():
                poses[int(key[6:])] = pose
    if not poses:
        raise ValueError(f"no object 6D poses in {scene}")
    return poses


@lru_cache(maxsize=None)
def robot_model(collection: str):
    urdf, palm = ROBOTS[collection]
    return parse_urdf(ASSETS / urdf), palm


def raw_object_from_palm(
    row: dict[str, Any],
    camera_from_robot: np.ndarray,
    camera_from_object: np.ndarray,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray,
) -> np.ndarray:
    model, palm = robot_model(row["collection"])
    robot_from_palm = compute_link_transforms(model, np.r_[arm_qpos, hand_qpos])[palm]
    return np.linalg.inv(camera_from_object) @ camera_from_robot @ robot_from_palm


def interpolate_transform(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    transform = np.eye(4)
    transform[:3, 3] = (1.0 - fraction) * first[:3, 3] + fraction * second[:3, 3]
    transform[:3, :3] = Slerp(
        [0.0, 1.0], Rotation.from_matrix(np.stack([first[:3, :3], second[:3, :3]]))
    )([fraction]).as_matrix()[0]
    return transform


def sample_source_frames(
    video_times: np.ndarray,
    valid_frames: list[int],
    fps: float,
    paired_frame: int,
    pre_seconds: float | None,
    post_seconds: float | None,
) -> list[int]:
    valid = np.asarray(sorted(valid_frames), dtype=int)
    if (pre_seconds is None) != (post_seconds is None):
        raise ValueError("pre_seconds and post_seconds must be supplied together")
    if pre_seconds is not None and post_seconds is not None:
        paired_time = float(video_times[paired_frame])
        within_window = (
            (video_times[valid] >= paired_time - pre_seconds)
            & (video_times[valid] <= paired_time + post_seconds)
        )
        valid = valid[within_window]
    if not len(valid):
        raise ValueError("no object poses in requested paired-frame time window")
    times = video_times[valid]
    desired = np.arange(times[0], times[-1] + 0.5 / fps, 1.0 / fps)
    indices, _ = nearest_indices(times, desired)
    selected = list(dict.fromkeys(valid[indices].tolist()))
    for required in (int(valid[0]), paired_frame, int(valid[-1])):
        if required in valid and required not in selected:
            selected.append(required)
    selected.sort()
    return selected


def load_trajectory(
    row: dict[str, Any], fps: float, pre_seconds: float | None = None, post_seconds: float | None = None
) -> tuple[list[TrajectorySample], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    scene = scene_for_row(row)
    video_times = numeric(scene / "raw/timestamps/timestamp.npy").reshape(-1)
    arm = numeric(scene / "raw/arm/position.npy")
    arm_times = numeric(scene / "raw/arm/time.npy").reshape(-1)
    hand_path, hand_time_path = hand_paths(row["collection"])
    hand = numeric(scene / hand_path)
    hand_times = numeric(scene / hand_time_path).reshape(-1)
    object_poses = load_object_poses(scene)
    paired_frame = int(row["robot_frame"])
    valid_frames = [i for i in object_poses if 0 <= i < len(video_times)]
    if paired_frame not in valid_frames:
        raise ValueError(f"paired frame {paired_frame} has no object pose")
    frames = sample_source_frames(video_times, valid_frames, fps, paired_frame, pre_seconds, post_seconds)
    query_times = video_times[frames]
    arm_indices, arm_delta = nearest_indices(arm_times, query_times)
    hand_indices, hand_delta = nearest_indices(hand_times, query_times)
    valid = (np.abs(arm_delta) <= 0.1) & (np.abs(hand_delta) <= 0.1)
    frames = [frame for frame, keep in zip(frames, valid) if keep]
    arm_indices = arm_indices[valid]
    hand_indices = hand_indices[valid]
    if paired_frame not in frames:
        raise ValueError("paired frame lacks arm/hand state within 100 ms")

    world_from_robot = np.asarray(np.load(scene / "C2R.npy"), dtype=float)
    if world_from_robot.shape != (4, 4) or not np.isfinite(world_from_robot).all():
        raise ValueError(f"invalid C2R.npy in {scene}")
    model, palm = robot_model(row["collection"])
    raw_samples = []
    for frame, arm_index, hand_index in zip(frames, arm_indices, hand_indices):
        hand_qpos = convert_hand_qpos(row["collection"], hand[hand_index])
        robot_from_palm = compute_link_transforms(
            model, np.r_[np.asarray(arm[arm_index], dtype=float).reshape(6), hand_qpos]
        )[palm]
        raw_samples.append((frame, hand_qpos, robot_from_palm))

    first_v2_frame = min(valid_frames)
    initial_object_pose = robot_from_world_pose(world_from_robot, object_poses[first_v2_frame])
    gravity = np.array([0.0, 0.0, -9.81])
    t0 = float(video_times[frames[0]])
    samples = [
        TrajectorySample(
            source_frame=frame,
            time_s=float(video_times[frame] - t0),
            hand_qpos=hand_qpos,
            robot_from_palm=robot_from_palm,
        )
        for frame, hand_qpos, robot_from_palm in raw_samples
    ]
    metadata = {
        "source_scene": str(scene),
        "source_frames": [int(samples[0].source_frame), int(samples[-1].source_frame)],
        "paired_frame": paired_frame,
        "duration_s": float(samples[-1].time_s),
        "frames": len(samples),
        "sampling_fps": fps,
        "requested_window_s": (
            {"before_paired": pre_seconds, "after_paired": post_seconds}
            if pre_seconds is not None else None
        ),
        "state_alignment": "nearest raw arm and hand state at each video timestamp; maximum 100 ms",
        "coordinate_alignment": "robot_from_object_0 = inv(C2R) @ world_from_object_v2[first]; robot_from_palm = raw arm+hand FK",
        "C2R_path": str(scene / "C2R.npy"),
        "object_pose_path": str(scene / "object_6d_pose_v2.npz"),
        "object_initial_frame": int(first_v2_frame),
        "object_initialization": "object_6d_pose_v2 first pose only; thereafter free MuJoCo dynamics",
        "gravity_m_s2": gravity.tolist(),
    }
    return samples, metadata, np.eye(4), initial_object_pose, gravity


def initialize_mimics(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for _ in range(3):
        for equality in range(model.neq):
            if model.eq_type[equality] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            destination, source = model.eq_obj1id[equality], model.eq_obj2id[equality]
            if source >= 0:
                value = data.qpos[model.jnt_qposadr[source]]
                data.qpos[model.jnt_qposadr[destination]] = np.polynomial.polynomial.polyval(
                    value, model.eq_data[equality, :5]
                )


def prepare_hand_model(
    row: dict[str, Any], model_dir: Path, object_mesh_path: Path
) -> tuple[mujoco.MjModel, dict[str, Any], list[HandMesh]]:
    template_dir = HAND_COLLISION_TEMPLATES[row["collection"]]
    if template_dir.is_dir():
        xml_path, meta = build_model_from_template(
            template_dir,
            object_mesh_path,
            model_dir,
            collision_excludes=NORMAL_TRAJECTORY_SELF_COLLISION_EXCLUDES[row["collection"]],
        )
    else:
        hand_urdf, hand_srdf = HAND_COLLISION_ASSETS[row["collection"]]
        if not hand_urdf.is_file() or (hand_srdf is not None and not hand_srdf.is_file()):
            raise FileNotFoundError(f"missing generated collision asset for {row['collection']}")
        xml_path, meta = build_model(
            hand_urdf,
            object_mesh_path,
            model_dir,
            srdf_path=hand_srdf,
            balance_inertia=True,
        )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    object_body = model.body(meta["object_body"]).id
    meshes = []
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 2 or int(model.geom_bodyid[geom_id]) in (0, object_body):
            continue
        vertices, faces = geom_local_mesh(model, geom_id)
        meshes.append(HandMesh(geom_id, vertices, faces))
    meta["hand_collision_geom_count"] = sum(
        int(model.geom_group[geom_id]) == 3
        and int(model.geom_bodyid[geom_id]) not in (0, object_body)
        for geom_id in range(model.ngeom)
    )
    return model, meta, meshes


def simulate_trajectory(
    model: mujoco.MjModel,
    meta: dict[str, Any],
    row: dict[str, Any],
    samples: list[TrajectorySample],
    geom_ids: list[int],
    initial_object_pose: np.ndarray,
    gravity: np.ndarray,
    canonical_object_vertices: np.ndarray,
    timestep: float = 0.002,
) -> tuple[list[PhysicsFrame], list[int]]:
    model.opt.gravity[:] = gravity
    model.opt.timestep = timestep
    # Recorded grasps and convex collision hulls can start with a small overlap.
    # A softer contact time constant resolves it without an impulse explosion.
    model.geom_solref[:, 0] = np.maximum(model.geom_solref[:, 0], 0.02)
    data = mujoco.MjData(model)
    object_body = model.body(meta["object_body"]).id
    hand_body_ids = {
        int(model.geom_bodyid[geom_id])
        for geom_id in range(model.ngeom)
        if int(model.geom_group[geom_id]) == 3
        and int(model.geom_bodyid[geom_id]) not in (0, object_body)
    }
    mocap_id = model.body_mocapid[model.body(meta["mocap_body"]).id]
    set_free_pose(model, data, meta["object_free_joint"], initial_object_pose)
    set_free_pose(model, data, meta["hand_free_joint"], samples[0].robot_from_palm)
    data.mocap_pos[mocap_id] = samples[0].robot_from_palm[:3, 3]
    data.mocap_quat[mocap_id] = Rotation.from_matrix(samples[0].robot_from_palm[:3, :3]).as_quat()[[3, 0, 1, 2]]
    set_joint_targets(model, data, meta, list(row["joint_names"]), samples[0].hand_qpos.tolist())
    initialize_mimics(model, data)

    # Place the support box beneath the recorded initial object, perpendicular to gravity.
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    up = -gravity / np.linalg.norm(gravity)
    world_vertices = canonical_object_vertices @ initial_object_pose[:3, :3].T + initial_object_pose[:3, 3]
    bottom_height = float(np.min(world_vertices @ up))
    table_rotation = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])[0].as_matrix().T
    tangent_center = world_vertices.mean(axis=0) - up * float(world_vertices.mean(axis=0) @ up)
    model.geom_quat[table] = Rotation.from_matrix(table_rotation).as_quat()[[3, 0, 1, 2]]
    model.geom_pos[table] = tangent_center + up * (
        bottom_height - float(model.geom_size[table, 2]) - 0.001 + FLOOR_RAISE_M
    )
    # The compiler marks an identity world geom as sharing the world frame. Clear
    # that optimization because this support orientation is selected at runtime.
    model.geom_sameframe[table] = 0
    mujoco.mj_forward(model, data)

    def capture() -> PhysicsFrame:
        object_pose = np.eye(4)
        object_pose[:3, :3] = np.asarray(data.xmat[object_body]).reshape(3, 3)
        object_pose[:3, 3] = np.asarray(data.xpos[object_body])
        self_distances = []
        self_pair_depths: dict[str, float] = {}
        object_contacts = 0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first_body = int(model.geom_bodyid[contact.geom1])
            second_body = int(model.geom_bodyid[contact.geom2])
            if first_body in hand_body_ids and second_body in hand_body_ids:
                self_distances.append(float(contact.dist))
                names = sorted((
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, first_body),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, second_body),
                ))
                pair = " | ".join(name.removeprefix("urdf_") for name in names)
                self_pair_depths[pair] = max(self_pair_depths.get(pair, 0.0), max(0.0, -float(contact.dist)))
            elif (
                (first_body in hand_body_ids and second_body == object_body)
                or (second_body in hand_body_ids and first_body == object_body)
            ):
                object_contacts += 1
        return PhysicsFrame(
            geom_poses={
                geom_id: (
                    np.asarray(data.geom_xpos[geom_id]).copy(),
                    np.asarray(data.geom_xmat[geom_id]).reshape(3, 3).copy(),
                ) for geom_id in geom_ids
            },
            object_pose=object_pose,
            contact_count=int(data.ncon),
            self_contact_count=len(self_distances),
            object_contact_count=object_contacts,
            min_self_contact_distance_m=min(self_distances, default=0.0),
            self_contact_pair_depth_m=self_pair_depths,
        )

    frames = [capture()]
    for first, second in zip(samples[:-1], samples[1:]):
        duration = max(second.time_s - first.time_s, model.opt.timestep)
        steps = max(1, int(np.ceil(duration / model.opt.timestep)))
        for step in range(1, steps + 1):
            fraction = step / steps
            hand_pose = interpolate_transform(first.robot_from_palm, second.robot_from_palm, fraction)
            hand_qpos = (1.0 - fraction) * first.hand_qpos + fraction * second.hand_qpos
            data.mocap_pos[mocap_id] = hand_pose[:3, 3]
            data.mocap_quat[mocap_id] = Rotation.from_matrix(hand_pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            set_joint_targets(model, data, meta, list(row["joint_names"]), hand_qpos.tolist())
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(
                    f"MuJoCo produced non-finite state at source frames "
                    f"{first.source_frame}-{second.source_frame}"
                )
        frames.append(capture())
    return frames, [int(warning.number) for warning in data.warning]


def camera_basis(gravity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_up = -np.asarray(gravity, dtype=float) / np.linalg.norm(gravity)
    reference = np.array([0.0, 0.0, 1.0])
    tangent = reference - world_up * float(reference @ world_up)
    if np.linalg.norm(tangent) < 0.2:
        reference = np.array([0.0, 1.0, 0.0])
        tangent = reference - world_up * float(reference @ world_up)
    tangent /= np.linalg.norm(tangent)
    forward = tangent - 0.55 * world_up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def view_parameters(
    object_vertices: np.ndarray,
    samples: list[TrajectorySample],
    physics_frames: list[PhysicsFrame],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float, float]:
    right, up, _ = basis
    points = [object_vertices]
    points.append(np.stack([sample.robot_from_palm[:3, 3] for sample in samples]))
    points.append(np.concatenate([[pose[0] for pose in frame.geom_poses.values()] for frame in physics_frames], axis=0))
    points.append(np.stack([frame.object_pose[:3, 3] for frame in physics_frames]))
    points = np.concatenate(points, axis=0)
    horizontal = points @ right
    vertical = points @ up
    margin = 0.035
    span_x = max(float(np.ptp(horizontal)) + 2 * margin, 0.1)
    span_y = max(float(np.ptp(vertical)) + 2 * margin, 0.1)
    scale = min((WIDTH - 40) / span_x, (HEIGHT - HEADER_H - FOOTER_H - 25) / span_y)
    return float((horizontal.min() + horizontal.max()) / 2), float((vertical.min() + vertical.max()) / 2), float(scale)


def project(
    points: np.ndarray,
    u0: float,
    v0: float,
    scale: float,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    right, up, forward = basis
    camera = np.column_stack([points @ right - u0, points @ up - v0, points @ forward])
    pixels = np.column_stack([
        WIDTH * 0.5 + camera[:, 0] * scale,
        HEADER_H + (HEIGHT - HEADER_H - FOOTER_H) * 0.5 - camera[:, 1] * scale,
    ]).astype(np.int32)
    return pixels, camera[:, 2]


def table_top_vertices(model: mujoco.MjModel) -> np.ndarray:
    """Return the four world-space corners of the physical table's top face."""
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    half_x, half_y, half_z = np.asarray(model.geom_size[table], dtype=float)
    local = np.array([
        [-half_x, -half_y, half_z],
        [half_x, -half_y, half_z],
        [half_x, half_y, half_z],
        [-half_x, half_y, half_z],
    ])
    rotation = Rotation.from_quat(np.asarray(model.geom_quat[table])[[1, 2, 3, 0]]).as_matrix()
    return local @ rotation.T + np.asarray(model.geom_pos[table])


def render_frame(
    row: dict[str, Any],
    sample: TrajectorySample,
    sample_index: int,
    samples: list[TrajectorySample],
    hand_meshes: list[HandMesh],
    physics_frame: PhysicsFrame,
    canonical_object_vertices: np.ndarray,
    object_faces: np.ndarray,
    floor_vertices: np.ndarray,
    view: tuple[float, float, float],
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],
    object_motion: str = "dynamic",
) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), (249, 248, 246), dtype=np.uint8)
    u0, v0, scale = view
    floor_pixels, _ = project(floor_vertices, u0, v0, scale, basis)
    cv2.fillConvexPoly(image, floor_pixels, (205, 210, 215), lineType=cv2.LINE_AA)
    cv2.polylines(image, [floor_pixels], True, (145, 150, 155), 2, cv2.LINE_AA)
    for fraction in np.linspace(0.1, 0.9, 9):
        first = np.rint((1.0 - fraction) * floor_pixels[0] + fraction * floor_pixels[1]).astype(np.int32)
        second = np.rint((1.0 - fraction) * floor_pixels[3] + fraction * floor_pixels[2]).astype(np.int32)
        cv2.line(image, tuple(first), tuple(second), (185, 190, 195), 1, cv2.LINE_AA)
        first = np.rint((1.0 - fraction) * floor_pixels[0] + fraction * floor_pixels[3]).astype(np.int32)
        second = np.rint((1.0 - fraction) * floor_pixels[1] + fraction * floor_pixels[2]).astype(np.int32)
        cv2.line(image, tuple(first), tuple(second), (185, 190, 195), 1, cv2.LINE_AA)
    triangles = []
    light = np.array([-0.3, -0.4, 0.85]); light /= np.linalg.norm(light)

    def add_mesh(vertices: np.ndarray, faces: np.ndarray, color: tuple[int, int, int]) -> None:
        pixels, depth = project(vertices, u0, v0, scale, basis)
        for face in faces:
            vertices3 = vertices[face]
            normal = np.cross(vertices3[1] - vertices3[0], vertices3[2] - vertices3[0])
            length = np.linalg.norm(normal)
            if length < 1e-10:
                continue
            normal /= length
            shade = 0.48 + 0.52 * abs(float(normal @ light))
            shaded = tuple(int(np.clip(channel * shade, 0, 255)) for channel in color)
            triangles.append((float(depth[face].mean()), pixels[face], shaded))

    object_vertices = (
        canonical_object_vertices @ physics_frame.object_pose[:3, :3].T
        + physics_frame.object_pose[:3, 3]
    )
    add_mesh(object_vertices, object_faces, (40, 110, 230))
    for mesh in hand_meshes:
        position, rotation = physics_frame.geom_poses[mesh.geom_id]
        add_mesh(mesh.vertices @ rotation.T + position, mesh.faces, (90, 170, 55))
    for _, polygon, color in sorted(triangles, key=lambda item: item[0], reverse=True):
        cv2.fillConvexPoly(image, polygon, color, lineType=cv2.LINE_AA)

    trace = np.stack([item.robot_from_palm[:3, 3] for item in samples[: sample_index + 1]])
    trace_pixels, _ = project(trace, u0, v0, scale, basis)
    if len(trace_pixels) > 1:
        cv2.polylines(image, [trace_pixels], False, (70, 120, 70), 2, cv2.LINE_AA)
    cv2.circle(image, tuple(trace_pixels[-1]), 5, (40, 100, 40), -1, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.rectangle((0, 0, WIDTH, HEADER_H), fill=(15, 20, 28))
    replay = object_motion == "recorded"
    title_motion = "kinematic visualization" if replay else "dynamic object"
    draw.text((18, 12), f"recorded robot trajectory + {title_motion} — {row['collection']} / {row['object_id']}", fill="white", font=FONT)
    marker = "  [PAIRED TARGET]" if sample.source_frame == int(row["robot_frame"]) else ""
    draw.text((18, 43), f"t={sample.time_s:.2f}s  source frame={sample.source_frame}{marker}", fill=(225, 230, 238), font=SMALL_FONT)
    draw.text(
        (18, 67),
        ("recorded wrist/joints/object pose" if replay else
         f"recorded wrist/joints; dynamic object; hand-object={physics_frame.object_contact_count}; self={physics_frame.self_contact_count}"),
        fill=(225, 230, 238),
        font=SMALL_FONT,
    )
    draw.line((0, HEIGHT - FOOTER_H, WIDTH, HEIGHT - FOOTER_H), fill=(215, 219, 225))
    object_label = "recorded object" if replay else "free dynamic object"
    draw.text((18, HEIGHT - 28), f"orange={object_label}  green=recorded robot hand  gray=physical floor  line=wrist path", fill=(40, 45, 52), font=SMALL_FONT)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def render_row(
    row: dict[str, Any], output_dir: Path, fps: float,
    pre_seconds: float | None = None, post_seconds: float | None = None,
) -> dict[str, Any]:
    samples, trajectory_meta, canonical_from_object, initial_object_pose, gravity = load_trajectory(
        row, fps, pre_seconds, post_seconds
    )
    model_dir = output_dir / "models" / str(row["pair_id"]).replace("/", "__")
    object_source = OBJECT_MESH_ROOT / row["object_id"] / f"{row['object_id']}.obj"
    if not object_source.is_file():
        raise FileNotFoundError(f"missing object mesh: {object_source}")
    model, model_meta, hand_meshes = prepare_hand_model(row, model_dir, object_source)
    object_mesh = trimesh.load(object_source, force="mesh", process=False)
    canonical_object_vertices = np.asarray(object_mesh.vertices) @ canonical_from_object[:3, :3].T + canonical_from_object[:3, 3]
    canonical_object_vertices, object_faces = _compact_mesh(canonical_object_vertices, np.asarray(object_mesh.faces), max_faces=900)
    physics_frames = None
    simulation_timestep = None
    simulation_warning_counts = None
    last_error = None
    for candidate_timestep in (0.002, 0.001, 0.0005):
        try:
            candidate_frames, candidate_warnings = simulate_trajectory(
                model,
                model_meta,
                row,
                samples,
                [mesh.geom_id for mesh in hand_meshes],
                initial_object_pose,
                gravity,
                canonical_object_vertices,
                timestep=candidate_timestep,
            )
            physics_frames = candidate_frames
            simulation_timestep = candidate_timestep
            simulation_warning_counts = candidate_warnings
            if not any(candidate_warnings):
                break
        except RuntimeError as exc:
            last_error = exc
    if physics_frames is None or simulation_timestep is None or simulation_warning_counts is None:
        raise RuntimeError(f"MuJoCo failed at every timestep: {last_error}")
    basis = camera_basis(gravity)
    view = view_parameters(canonical_object_vertices, samples, physics_frames, basis)
    floor_vertices = table_top_vertices(model)

    stem = f"{row['collection']}_{str(row['pair_id']).replace('/', '__')}_target_trajectory"
    output_path = output_dir / f"{stem}.mp4"
    raw_path = output_dir / f"{stem}_mp4v_tmp.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {raw_path}")
    preview = None
    last_frame = None
    for index, (sample, physics_frame) in enumerate(zip(samples, physics_frames)):
        frame = render_frame(
            row, sample, index, samples, hand_meshes, physics_frame,
            canonical_object_vertices, object_faces, floor_vertices, view, basis,
        )
        writer.write(frame)
        last_frame = frame
        if sample.source_frame == int(row["robot_frame"]):
            preview = frame.copy()
    if preview is None:
        preview = frame.copy()
    for _ in range(round(fps)):
        writer.write(last_frame)
    writer.release()
    encoding = transcode_h264(raw_path, output_path)
    preview_path = output_path.with_suffix(".png")
    cv2.imwrite(str(preview_path), preview)
    self_contact_pair_depth_m: dict[str, float] = {}
    for frame in physics_frames:
        for pair, depth in frame.self_contact_pair_depth_m.items():
            self_contact_pair_depth_m[pair] = max(self_contact_pair_depth_m.get(pair, 0.0), depth)
    return {
        "collection": row["collection"],
        "object_id": row["object_id"],
        "evaluation_object_id": row["evaluation_object_id"],
        "pair_id": row["pair_id"],
        "video": str(output_path),
        "preview": str(preview_path),
        "object_mesh": str(object_source),
        "encoding": encoding,
        "trajectory": trajectory_meta,
        "physics": {
            "engine": "MuJoCo",
            "object_motion": "free joint; initialized once from object_6d_pose_v2 first pose",
            "hand_motion": "mocap wrist target and joint controls interpolated at the simulation timestep",
            "simulation_timestep_s": simulation_timestep,
            "warning_counts": simulation_warning_counts,
            "numerically_clean": not any(simulation_warning_counts),
            "support": "physical box floor perpendicular to robot-frame gravity; raised 16 mm",
            "floor_raise_m": FLOOR_RAISE_M,
            "hand_collision_urdf": model_meta.get("urdf_path"),
            "hand_collision_srdf": model_meta.get("srdf_path"),
            "hand_collision_template": model_meta.get("collision_template"),
            "hand_collision_geom_count": model_meta["hand_collision_geom_count"],
            "self_collision": model_meta["self_collision"],
            "self_collision_excludes": model_meta["collision_excludes"],
            "max_self_contact_count": max(frame.self_contact_count for frame in physics_frames),
            "max_object_contact_count": max(frame.object_contact_count for frame in physics_frames),
            "max_self_penetration_m": max(
                0.0, -min(frame.min_self_contact_distance_m for frame in physics_frames)
            ),
            "self_contact_pair_depth_m": dict(
                sorted(self_contact_pair_depth_m.items(), key=lambda item: -item[1])
            ),
            "final_object_translation_m": float(np.linalg.norm(physics_frames[-1].object_pose[:3, 3] - physics_frames[0].object_pose[:3, 3])),
            "final_object_rotation_deg": float(np.degrees(Rotation.from_matrix(physics_frames[0].object_pose[:3, :3].T @ physics_frames[-1].object_pose[:3, :3]).magnitude())),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=BASE / "runs/predictions.json")
    parser.add_argument("--out-dir", type=Path, default=BASE / "videos_target_trajectory")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--collections", nargs="*")
    parser.add_argument("--objects", nargs="*")
    parser.add_argument("--pair-id")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--workers", type=int, default=1, help="parallel video render processes")
    parser.add_argument("--pre-seconds", type=float, help="optional motion window before paired frame; default is full episode")
    parser.add_argument("--post-seconds", type=float, help="optional motion window after paired frame; default is full episode")
    args = parser.parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if (args.pre_seconds is None) != (args.post_seconds is None):
        raise SystemExit("--pre-seconds and --post-seconds must be supplied together")
    if args.pre_seconds is not None and (args.pre_seconds < 0 or args.post_seconds < 0):
        raise SystemExit("trajectory window durations must be nonnegative")
    rows, excluded = load_prediction_rows([args.predictions], per_collection=None, one_per_object=True)
    if args.collections:
        rows = [row for row in rows if row["collection"] in args.collections]
    if args.objects:
        rows = [row for row in rows if row["evaluation_object_id"] in args.objects or row["object_id"] in args.objects]
    if args.pair_id:
        rows = [row for row in rows if row["pair_id"] == args.pair_id]
    if args.max_videos is not None:
        rows = rows[: args.max_videos]
    if not rows:
        raise SystemExit("no matching gravity-valid prediction rows")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    videos = []
    failures = []
    if args.workers == 1:
        for index, row in enumerate(rows, 1):
            print(f"[{index}/{len(rows)}] {row['pair_id']}", flush=True)
            try:
                videos.append(render_row(row, args.out_dir, args.fps, args.pre_seconds, args.post_seconds))
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                failures.append({"pair_id": row["pair_id"], "reason": f"{type(exc).__name__}: {exc}"})
                print(f"  skipped: {exc}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {
                executor.submit(render_row, row, args.out_dir, args.fps, args.pre_seconds, args.post_seconds): row
                for row in rows
            }
            for index, future in enumerate(as_completed(pending), 1):
                row = pending[future]
                try:
                    videos.append(future.result())
                    print(f"[{index}/{len(rows)}] rendered {row['pair_id']}", flush=True)
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    failures.append({"pair_id": row["pair_id"], "reason": f"{type(exc).__name__}: {exc}"})
                    print(f"[{index}/{len(rows)}] skipped {row['pair_id']}: {exc}", flush=True)
    videos.sort(key=lambda item: (item["collection"], item["evaluation_object_id"], item["pair_id"]))
    manifest = {
        "renderer": "recorded v0 robot-frame wrist/hand trajectory with generated hand collision meshes, self-collision, and a free MuJoCo object",
        "coordinate_frame": "robot base frame; object starts at inv(C2R) @ object_6d_pose_v2[first]",
        "selection": "one gravity-valid test pair per embodiment and evaluation object",
        "videos": videos,
        "failures": failures,
        "selection_exclusions": excluded,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "videos": len(videos), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
