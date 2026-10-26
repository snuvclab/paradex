"""Retarget a human MANO episode to xArm + Inspire DFTP commands.

The source motion is replayed from its first frame through its last frame.  A
single rigid transform aligns the source episode's initial object pose with the
object pose measured on the robot server; no task-specific placement motion is
added.

Planning is the default.  Hardware commands are sent only with both
``--execute`` and ``--yes``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from paradex.retargetor.experiment.replay_pose_retrieval import (
    align_pose_by_continuous_symmetry,
)
from src.object6d.object_symmetry import (
    AXIS_FLIP_SYMMETRIC_OBJECTS,
    CONTINUOUS_SYMMETRY_AXIS_OBJECT,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_ROOT = Path("/home/temp_id/shared_data/capture/eccv2026/v0/human")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "human_dftp_replay"
DEFAULT_RPC_ADDR = "tcp://192.168.0.13:5570"
DEFAULT_CAPTURE_SAVE_PATH = Path("object_6d/human_dftp")
DEFAULT_MESH_ROOT = Path.home() / "shared_data" / "mesh_blender"
XARM_INSPIRE_URDF = REPO_ROOT / "rsc/robot/xarm_inspire_DFTP.urdf"
INSPIRE_FLOAT_URDF = REPO_ROOT / "rsc/robot/inspire_float.urdf"

# Maps the dex-retargeting Inspire model to inspire_action_to_qpos() order.
INSPIRE_COMMAND_JOINTS = (
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)
# DexPilot's Inspire URDF uses a fixed base transform whose +Z points from the
# wrist toward the fingers. Keep this mapping for finger-vector retargeting.
OPERATOR2MANO_RIGHT = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
# The physical DFTP/floating-hand URDF instead extends the fingers along local
# -Z, with local +X toward the index finger. Its wrist target therefore needs a
# separate anatomical frame; using the DexPilot base frame here reverses the
# rendered and commanded hand by 180 degrees.
OPERATOR2INSPIRE_DFTP_RIGHT = np.array(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
MANO_BONES = np.asarray(
    [(0, 1), (1, 2), (2, 3), (3, 4)]
    + [(0, 5), (5, 6), (6, 7), (7, 8)]
    + [(0, 9), (9, 10), (10, 11), (11, 12)]
    + [(0, 13), (13, 14), (14, 15), (15, 16)]
    + [(0, 17), (17, 18), (18, 19), (19, 20)],
    dtype=np.int64,
)
MANO_FINGERTIP_JOINTS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
INSPIRE_DFTP_RETARGET_JOINTS = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
)
INSPIRE_DFTP_TIP_SPECS = (
    ("right_thumb_4", "right_thumb_anyteleop_tip", (0.0334, 0.0206, -0.0074)),
    ("right_index_2", "right_index_anyteleop_tip", (0.0, 0.0536, -0.0061)),
    ("right_middle_2", "right_middle_anyteleop_tip", (0.0, 0.0573, -0.0061)),
    ("right_ring_2", "right_ring_anyteleop_tip", (0.0, 0.0537, -0.0061)),
    ("right_little_2", "right_little_anyteleop_tip", (0.0439, 0.0, -0.0061)),
)
INSPIRE_RETARGET_TIP_LINKS = tuple(spec[1] for spec in INSPIRE_DFTP_TIP_SPECS)


@dataclass(frozen=True)
class HumanEpisode:
    object_name: str
    episode_id: str
    root: Path
    joints_world: np.ndarray
    object_poses_world: np.ndarray
    camera_to_robot: np.ndarray


@dataclass(frozen=True)
class ReplayPlan:
    object_name: str
    episode_id: str
    episode_root: Path
    times_s: np.ndarray
    link6_poses_robot: np.ndarray
    hand_qpos: np.ndarray
    hand_raw: np.ndarray
    source_object_robot: np.ndarray
    current_object_robot: np.ndarray
    aligned_current_object_robot: np.ndarray
    alignment_robot: np.ndarray
    symmetry_angle_rad: float
    symmetry_flipped: bool
    reference_mano_joints_robot: np.ndarray
    fingertip_rmse_m: np.ndarray
    safety_errors: tuple[str, ...]


@dataclass(frozen=True)
class HumanPoseMatch:
    episode_root: Path
    source_object_robot: np.ndarray
    aligned_current_object_robot: np.ndarray
    symmetry_angle_rad: float
    symmetry_flipped: bool
    translation_error_m: float
    rotation_error_rad: float
    score: float


def _as_transform(value: Any, *, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (3, 4):
        matrix = np.vstack((matrix, [0.0, 0.0, 0.0, 1.0]))
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 4x4 (or 3x4) matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{label} has an invalid homogeneous last row")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-3):
        raise ValueError(f"{label} rotation is not orthonormal")
    if np.linalg.det(matrix[:3, :3]) < 0.99:
        raise ValueError(f"{label} rotation must be right-handed")
    return matrix


def _numeric_suffix(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"file has no numeric frame id: {path}")
    return int(digits)


def discover_objects(human_root: Path) -> list[str]:
    root = Path(human_root)
    if not root.is_dir():
        raise FileNotFoundError(f"human dataset root does not exist: {root}")
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and any(child.is_dir() for child in path.iterdir())
    )


def discover_episodes(human_root: Path, object_name: str) -> list[Path]:
    object_root = Path(human_root) / object_name
    if not object_root.is_dir():
        available = discover_objects(human_root)
        preview = ", ".join(available[:12])
        raise FileNotFoundError(
            f"object {object_name!r} is not in {human_root}; examples: {preview}"
        )
    episodes = [
        path
        for path in object_root.iterdir()
        if path.is_dir()
        and (path / "hand/mano_params").is_dir()
        and (path / "C2R.npy").is_file()
        and any(
            (path / name).is_file()
            for name in ("object_6d_pose_v2.npz", "object_6d_pose_v1.npz")
        )
    ]
    return sorted(
        episodes,
        key=lambda path: (
            not path.name.isdigit(),
            int(path.name) if path.name.isdigit() else path.name,
        ),
    )


def _load_object_poses(episode_root: Path) -> np.ndarray:
    archive_path = next(
        (
            episode_root / name
            for name in ("object_6d_pose_v2.npz", "object_6d_pose_v1.npz")
            if (episode_root / name).is_file()
        ),
        None,
    )
    if archive_path is None:
        raise FileNotFoundError(f"object pose archive missing in {episode_root}")
    with np.load(archive_path) as archive:
        keys = sorted(archive.files, key=lambda key: int(key.rsplit("_", 1)[-1]))
        if not keys:
            raise ValueError(f"empty object pose archive: {archive_path}")
        return np.stack(
            [_as_transform(archive[key], label=f"{archive_path}:{key}") for key in keys]
        )


def load_human_episode(
    episode_root: Path, object_name: str | None = None
) -> HumanEpisode:
    root = Path(episode_root)
    param_files = sorted(
        (root / "hand/mano_params").glob("*.json"), key=_numeric_suffix
    )
    if not param_files:
        raise FileNotFoundError(f"MANO parameter JSON files missing in {root}")
    frame_ids = [_numeric_suffix(path) for path in param_files]
    if frame_ids != list(range(frame_ids[0], frame_ids[0] + len(frame_ids))):
        raise ValueError(f"MANO frame ids are not contiguous in {root}")
    joints: list[np.ndarray] = []
    for path in param_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            frame_joints = np.asarray(payload["joints"], dtype=np.float64)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid MANO parameter file: {path}") from exc
        if frame_joints.shape != (21, 3) or not np.all(np.isfinite(frame_joints)):
            raise ValueError(f"MANO joints must be finite [21,3]: {path}")
        joints.append(frame_joints)
    object_poses = _load_object_poses(root)
    if len(object_poses) != len(joints):
        raise ValueError(
            f"MANO/object frame count mismatch in {root}: "
            f"{len(joints)} != {len(object_poses)}"
        )
    c2r = _as_transform(np.load(root / "C2R.npy"), label=f"{root}/C2R.npy")
    return HumanEpisode(
        object_name=object_name or root.parent.name,
        episode_id=root.name,
        root=root,
        joints_world=np.stack(joints),
        object_poses_world=object_poses,
        camera_to_robot=c2r,
    )


def load_server_object_pose(
    path: Path,
    *,
    pose_frame: str = "auto",
    current_c2r: Path | None = None,
) -> tuple[str | None, np.ndarray]:
    """Load a server-produced pose and return it in the robot base frame."""
    path = Path(path)
    inferred_frame: str | None = None
    object_name: str | None = None
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("object_6d"), dict):
            payload = payload["object_6d"]
        object_name = (
            payload.get("object_name")
            or payload.get("detected_object_name")
            or payload.get("object")
        )
        for key, frame in (
            ("pose_robot", "robot"),
            ("object_pose_robot", "robot"),
            ("pose_world", "world"),
            ("object_pose_world", "world"),
            ("pose", None),
            ("matrix", None),
        ):
            if key in payload:
                value = payload[key]
                inferred_frame = frame or payload.get("frame")
                break
        else:
            raise ValueError(f"no object pose matrix found in {path}")
    elif path.suffix.lower() == ".npy":
        value = np.load(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            keys = archive.files
            key = next(
                (
                    name
                    for name in ("pose_robot", "pose_world", "pose", "matrix")
                    if name in keys
                ),
                None,
            )
            if key is None:
                raise ValueError(f"no object pose matrix found in {path}")
            value = archive[key]
            inferred_frame = (
                "robot"
                if key.endswith("robot")
                else "world" if key.endswith("world") else None
            )
    else:
        value = np.loadtxt(path)
    pose = _as_transform(value, label=str(path))
    frame = inferred_frame if pose_frame == "auto" else pose_frame
    frame = frame or "robot"
    if frame not in {"robot", "world"}:
        raise ValueError(f"pose frame must be robot or world, got {frame!r}")
    if frame == "world":
        if current_c2r is None:
            raise ValueError("--current-c2r is required for a world-frame server pose")
        c2r = _as_transform(np.load(current_c2r), label=str(current_c2r))
        pose = np.linalg.inv(c2r) @ pose
    return object_name, pose


def capture_root_to_server_path(capture_root: Path) -> str:
    """Convert a local ~/shared_data path to the server's shared_data path."""
    root = Path.home() / "shared_data"
    path = Path(capture_root).expanduser().resolve()
    try:
        relative = path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"RPC capture root must be under {root}: {path}") from exc
    return (Path("shared_data") / relative).as_posix()


def build_object6d_rpc_request(
    capture_root: Path,
    object_candidates: Sequence[str],
) -> dict[str, Any]:
    candidates = list(dict.fromkeys(str(name) for name in object_candidates))
    if not candidates:
        raise ValueError("Object6D RPC requires at least one object candidate")
    request: dict[str, Any] = {
        "command": "infer",
        "image_path": capture_root_to_server_path(capture_root),
        "mesh_names": candidates,
        "object_candidates": candidates,
        "detect_object_name": len(candidates) > 1,
        "generate_grasp_trajectory": False,
        "save_projection_grids": True,
        "wooden_object_triangulation": False,
    }
    if len(candidates) == 1:
        request["mesh_name"] = candidates[0]
    return request


def send_object6d_rpc(
    address: str, request: Mapping[str, Any], timeout_ms: int
) -> dict[str, Any]:
    """Send one Object6D ZMQ request and return its JSON response."""
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "pyzmq is required for --capture/--rpc-capture-root"
        ) from exc
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(address)
        started = time.perf_counter()
        socket.send_json(dict(request))
        response = socket.recv_json()
        elapsed = time.perf_counter() - started
    except zmq.Again as exc:
        raise TimeoutError(
            f"Object6D RPC timed out after {timeout_ms} ms: {address}"
        ) from exc
    finally:
        socket.close()
        context.term()
    if not isinstance(response, dict):
        raise RuntimeError(f"Object6D RPC returned non-object JSON: {type(response)}")
    if response.get("status") == "error":
        raise RuntimeError(response.get("error", "Object6D RPC failed"))
    print(f"[rpc] Object6D response received from {address} in {elapsed:.3f} s")
    return response


def extract_rpc_object_pose(
    response: Mapping[str, Any], current_c2r: np.ndarray | None
) -> tuple[str | None, np.ndarray]:
    payload = response.get("object_6d", response)
    if not isinstance(payload, Mapping):
        raise ValueError("Object6D response field object_6d must be a JSON object")
    object_name = (
        payload.get("object_name")
        or payload.get("detected_object_name")
        or response.get("object_name")
        or response.get("detected_object_name")
    )
    robot_value = payload.get("pose_robot")
    if robot_value is None:
        robot_value = payload.get("object_pose_robot")
    if robot_value is not None:
        return str(object_name) if object_name else None, _as_transform(
            robot_value, label="Object6D pose_robot"
        )
    world_value = payload.get("pose_world")
    if world_value is None:
        world_value = payload.get("object_pose_world")
    if world_value is None:
        raise ValueError("Object6D response has neither pose_robot nor pose_world")
    if current_c2r is None:
        raise ValueError("C2R is required when the Object6D server returns pose_world")
    c2r = _as_transform(current_c2r, label="current C2R")
    pose_robot = np.linalg.inv(c2r) @ _as_transform(
        world_value, label="Object6D pose_world"
    )
    return str(object_name) if object_name else None, pose_robot


def capture_multiview(capture_root: Path, rcc_entry: str) -> None:
    """Use the established capture_sil camera path for one fresh observation."""
    from src.object6d.capture_sil import capture_once

    relative = (
        Path(capture_root)
        .resolve()
        .relative_to((Path.home() / "shared_data").resolve())
    )
    remote_raw_path = (Path("shared_data") / relative / "raw").as_posix()
    capture_once(Path(capture_root), remote_raw_path, rcc_entry)


def request_current_object_pose(
    *,
    capture_root: Path,
    human_root: Path,
    object_name: str | None,
    rpc_addr: str,
    rpc_timeout_ms: int,
    current_c2r_path: Path | None = None,
) -> tuple[str | None, np.ndarray, dict[str, Any]]:
    candidates = [object_name] if object_name else discover_objects(human_root)
    request = build_object6d_rpc_request(capture_root, candidates)
    print(
        f"[rpc] requesting Object6D from {rpc_addr}; "
        f"image_path={request['image_path']}, candidates={len(candidates)}"
    )
    response = send_object6d_rpc(rpc_addr, request, rpc_timeout_ms)
    c2r_path = current_c2r_path or (Path(capture_root) / "C2R.npy")
    current_c2r = np.load(c2r_path) if c2r_path.is_file() else None
    detected_name, pose_robot = extract_rpc_object_pose(response, current_c2r)
    return detected_name, pose_robot, response


def rank_episodes(
    episode_roots: Sequence[Path],
    current_object_robot: np.ndarray,
    *,
    object_name: str,
    translation_scale_m: float = 0.05,
    rotation_scale_rad: float = 0.25,
) -> list[HumanPoseMatch]:
    """Rank demonstrations using the same symmetry quotient as sorting replay."""
    if translation_scale_m <= 0 or rotation_scale_rad <= 0:
        raise ValueError("retrieval translation/rotation scales must be positive")
    current = _as_transform(current_object_robot, label="current object pose")
    symmetry_axis = CONTINUOUS_SYMMETRY_AXIS_OBJECT.get(object_name)
    ranked: list[HumanPoseMatch] = []
    for root in episode_roots:
        source_world = _load_object_poses(root)[0]
        c2r = _as_transform(np.load(root / "C2R.npy"), label=f"{root}/C2R.npy")
        source_robot = np.linalg.inv(c2r) @ source_world
        aligned_current, symmetry_angle, symmetry_flipped = (
            align_pose_by_continuous_symmetry(
                source_robot,
                current,
                continuous_symmetry_axis_object=symmetry_axis,
                continuous_symmetry_allow_axis_flip=(
                    object_name in AXIS_FLIP_SYMMETRIC_OBJECTS
                ),
            )
        )
        translation = float(np.linalg.norm(current[:3, 3] - source_robot[:3, 3]))
        rotation = float(
            R.from_matrix(aligned_current[:3, :3] @ source_robot[:3, :3].T).magnitude()
        )
        ranked.append(
            HumanPoseMatch(
                episode_root=root,
                source_object_robot=source_robot,
                aligned_current_object_robot=aligned_current,
                symmetry_angle_rad=symmetry_angle,
                symmetry_flipped=symmetry_flipped,
                translation_error_m=translation,
                rotation_error_rad=rotation,
                score=float(
                    np.hypot(
                        translation / translation_scale_m,
                        rotation / rotation_scale_rad,
                    )
                ),
            )
        )
    return sorted(
        ranked,
        key=lambda match: (
            match.score,
            not match.episode_root.name.isdigit(),
            (
                int(match.episode_root.name)
                if match.episode_root.name.isdigit()
                else match.episode_root.name
            ),
        ),
    )


def estimate_mano_wrist_frame(keypoints: np.ndarray) -> np.ndarray:
    points = np.asarray(keypoints, dtype=np.float64)[[0, 5, 9]]
    centered = points - points.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered)
    if singular[1] < 1e-7:
        raise ValueError("degenerate MANO wrist landmarks")
    normal = vh[2]
    x_axis = points[0] - points[2]
    x_axis -= np.dot(x_axis, normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, normal)
    if np.dot(z_axis, points[1] - points[2]) < 0:
        normal *= -1
        z_axis *= -1
    return R.from_matrix(np.stack((x_axis, normal, z_axis), axis=1)).as_matrix()


def load_link6_from_wrist(robot_urdf: Path = XARM_INSPIRE_URDF) -> np.ndarray:
    root = ET.parse(Path(robot_urdf)).getroot()
    joint = root.find(".//joint[@name='arm_to_hand']")
    if joint is None or joint.find("origin") is None:
        raise ValueError(f"arm_to_hand origin missing from {robot_urdf}")
    origin = joint.find("origin")
    assert origin is not None
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"invalid arm_to_hand origin in {robot_urdf}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def place_wrist_from_fingertips(
    human_tips: np.ndarray,
    robot_tips_wrist: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Register the robot fingertip centroid to the MANO fingertips."""
    human = np.asarray(human_tips, dtype=np.float64)
    robot = np.asarray(robot_tips_wrist, dtype=np.float64)
    wrist_rotation = np.asarray(rotation, dtype=np.float64)
    if human.shape != robot.shape or human.ndim != 2 or human.shape[1] != 3:
        raise ValueError("human and robot fingertips must have equal [N,3] shape")
    if wrist_rotation.shape != (3, 3):
        raise ValueError("wrist rotation must have shape [3,3]")
    if not all(np.all(np.isfinite(value)) for value in (human, robot, wrist_rotation)):
        raise ValueError("fingertip registration inputs must be finite")
    translation = np.mean(human - robot @ wrist_rotation.T, axis=0)
    fitted = robot @ wrist_rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum((human - fitted) ** 2, axis=1))))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = wrist_rotation
    pose[:3, 3] = translation
    return pose, rmse


def align_wrist_trajectory(
    episode: HumanEpisode,
    current_object_robot: np.ndarray,
    wrist_poses_world: np.ndarray,
    *,
    wrist_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_from_wrist = np.asarray(wrist_poses_world, dtype=np.float64)
    if world_from_wrist.shape != (len(episode.joints_world), 4, 4):
        raise ValueError(
            "registered wrist poses must have shape "
            f"[{len(episode.joints_world)},4,4], got {world_from_wrist.shape}"
        )
    robot_from_world = np.linalg.inv(episode.camera_to_robot)
    robot_from_source_object = robot_from_world @ episode.object_poses_world[0]
    alignment = _as_transform(
        current_object_robot, label="current object pose"
    ) @ np.linalg.inv(robot_from_source_object)
    robot_from_wrist = alignment[None] @ robot_from_world[None] @ world_from_wrist
    offset = np.asarray(wrist_offset_m, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("wrist offset must contain three finite values")
    robot_from_wrist[:, :3, 3] += np.einsum(
        "tij,j->ti", robot_from_wrist[:, :3, :3], offset
    )
    robot_from_link6 = robot_from_wrist @ np.linalg.inv(load_link6_from_wrist())[None]
    return robot_from_link6, robot_from_source_object, alignment


def derive_inspire_dftp_retarget_urdf(source: Path, output: Path) -> Path:
    """Add AnyTeleop fingertip frames to the physical Inspire DFTP URDF."""
    source = Path(source).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        filename = Path(mesh.attrib["filename"])
        if not filename.is_absolute():
            mesh.set("filename", str((source.parent / filename).resolve()))
    for parent, child, xyz in INSPIRE_DFTP_TIP_SPECS:
        if root.find(f"link[@name='{child}']") is not None:
            raise ValueError(f"virtual fingertip link already exists: {child}")
        ET.SubElement(root, "link", name=child)
        joint = ET.SubElement(root, "joint", name=f"{child}_fixed", type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(
            joint,
            "origin",
            xyz=" ".join(str(value) for value in xyz),
            rpy="0 0 0",
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def retarget_inspire_dexpilot(
    joints_world: np.ndarray,
    *,
    scaling_factor: float = 1.15,
    low_pass_alpha: float = 0.2,
    urdf_path: Path = XARM_INSPIRE_URDF,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return qpos plus fingertip-registered wrist poses and residuals."""
    try:
        from dex_retargeting.retargeting_config import RetargetingConfig
    except ImportError as exc:
        raise RuntimeError(
            "dex-retargeting is unavailable; run this script in the repository's robot environment"
        ) from exc
    temporary_urdf = tempfile.TemporaryDirectory(prefix="paradex_dftp_retarget_")
    derived_urdf = derive_inspire_dftp_retarget_urdf(
        urdf_path, Path(temporary_urdf.name) / "inspire_dftp_anyteleop.urdf"
    )
    config = RetargetingConfig.from_dict(
        {
            "type": "DexPilot",
            "urdf_path": str(derived_urdf),
            "target_joint_names": list(INSPIRE_DFTP_RETARGET_JOINTS),
            "wrist_link_name": "wrist",
            "finger_tip_link_names": list(INSPIRE_RETARGET_TIP_LINKS),
            "scaling_factor": float(scaling_factor),
            "low_pass_alpha": float(low_pass_alpha),
        }
    )
    retargeter = config.build()
    dof_names = list(retargeter.optimizer.robot.dof_joint_names)
    try:
        command_indices = [
            dof_names.index(name) for name in INSPIRE_DFTP_RETARGET_JOINTS
        ]
    except ValueError as exc:
        raise RuntimeError(
            f"Inspire retargeting URDF has unexpected joints: {dof_names}"
        ) from exc
    human_indices = retargeter.optimizer.target_link_human_indices
    n_fixed = len(retargeter.optimizer.idx_pin2fixed)
    fixed_qpos = np.zeros(n_fixed, dtype=np.float32)
    output = np.empty(
        (len(joints_world), len(INSPIRE_COMMAND_JOINTS)), dtype=np.float64
    )
    wrist_poses_world = np.repeat(
        np.eye(4, dtype=np.float64)[None], len(joints_world), axis=0
    )
    fingertip_rmse_m = np.empty(len(joints_world), dtype=np.float64)
    robot = retargeter.optimizer.robot
    wrist_link_index = robot.get_link_index("wrist")
    tip_link_indices = [
        robot.get_link_index(link_name) for link_name in INSPIRE_RETARGET_TIP_LINKS
    ]
    robot_qpos = np.zeros(robot.dof, dtype=np.float64)
    robot.compute_forward_kinematics(robot_qpos)
    base_rotation_wrist = robot.get_link_pose(wrist_link_index)[:3, :3].copy()
    for frame, keypoints in enumerate(np.asarray(joints_world, dtype=np.float64)):
        local = keypoints - keypoints[0]
        rotation = estimate_mano_wrist_frame(local) @ OPERATOR2INSPIRE_DFTP_RIGHT
        local = local @ rotation
        reference = (
            local[human_indices[1]] - local[human_indices[0]]
        ) @ base_rotation_wrist.T
        qpos = np.asarray(retargeter.retarget(reference, fixed_qpos=fixed_qpos))
        output[frame] = qpos[command_indices]
        robot_qpos.fill(0.0)
        robot_qpos[command_indices] = output[frame]
        adaptor = retargeter.optimizer.adaptor
        fk_qpos = (
            adaptor.forward_qpos(robot_qpos) if adaptor is not None else robot_qpos
        )
        robot.compute_forward_kinematics(fk_qpos)
        wrist_from_base = np.linalg.inv(robot.get_link_pose(wrist_link_index))
        tips_wrist = np.asarray(
            [
                (wrist_from_base @ robot.get_link_pose(link_index))[:3, 3]
                for link_index in tip_link_indices
            ]
        )
        human_tips = keypoints[MANO_FINGERTIP_JOINTS]
        wrist_pose, fingertip_rmse_m[frame] = place_wrist_from_fingertips(
            human_tips,
            tips_wrist,
            rotation,
        )
        wrist_poses_world[frame] = wrist_pose
    if not np.all(np.isfinite(output)):
        raise RuntimeError("dex-retargeting produced non-finite Inspire commands")
    temporary_urdf.cleanup()
    return output, wrist_poses_world, fingertip_rmse_m


def hand_qpos_to_dftp_raw(hand_qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(hand_qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 6 or not np.all(np.isfinite(qpos)):
        raise ValueError(f"Inspire qpos must be finite [T,6], got {qpos.shape}")
    raw_axis = np.arange(1001, dtype=np.float64)
    calibration = dftp_raw_to_hand_qpos(np.repeat(raw_axis[:, None], 6, axis=1))
    raw = np.empty_like(qpos)
    for qpos_index, raw_index in enumerate((5, 4, 3, 2, 1, 0)):
        curve = calibration[:, qpos_index]
        order = np.argsort(curve)
        raw[:, raw_index] = np.interp(
            qpos[:, qpos_index], curve[order], raw_axis[order]
        )
    return np.clip(raw, 0.0, 1000.0)


def dftp_raw_to_hand_qpos(raw_commands: np.ndarray) -> np.ndarray:
    """Match paradex.robot.inspire.inspire_action_to_qpos without robot imports."""
    raw = np.asarray(raw_commands, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.all(np.isfinite(raw)):
        raise ValueError(f"DFTP raw commands must be finite [T,6], got {raw.shape}")
    qpos = np.zeros_like(raw)
    qpos[:, 0] = 1.40 * (1.0 - raw[:, 5] / 1000.0)
    qpos[:, 1] = 0.60 * (1.0 - raw[:, 4] / 1000.0)

    def long_finger(values: np.ndarray) -> np.ndarray:
        degrees = -4e-8 * values**3 + 3e-5 * values**2 - 0.0704 * values + 83.572
        return np.deg2rad(degrees)

    qpos[:, 2] = long_finger(raw[:, 3])
    qpos[:, 3] = long_finger(raw[:, 2])
    qpos[:, 4] = long_finger(raw[:, 1])
    qpos[:, 5] = long_finger(raw[:, 0])
    return qpos


def resample_commands(
    poses: np.ndarray,
    qpos: np.ndarray,
    *,
    source_hz: float,
    control_hz: float,
    speed_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_hz <= 0 or control_hz <= 0 or speed_scale <= 0:
        raise ValueError("source_hz, control_hz, and speed_scale must be positive")
    if len(poses) != len(qpos) or len(poses) == 0:
        raise ValueError("pose/qpos trajectories must be non-empty and equal length")
    source_times = np.arange(len(poses), dtype=np.float64) / source_hz / speed_scale
    if len(poses) == 1:
        return poses.copy(), qpos.copy(), np.array([0.0])
    duration = float(source_times[-1])
    target_times = np.arange(0.0, duration + 0.25 / control_hz, 1.0 / control_hz)
    if target_times[-1] < duration - 1e-9:
        target_times = np.append(target_times, duration)
    target_times = np.unique(np.minimum(target_times, duration))
    positions = np.column_stack(
        [np.interp(target_times, source_times, poses[:, axis, 3]) for axis in range(3)]
    )
    rotations = Slerp(source_times, R.from_matrix(poses[:, :3, :3]))(
        target_times
    ).as_matrix()
    resampled_poses = np.repeat(np.eye(4)[None], len(target_times), axis=0)
    resampled_poses[:, :3, :3] = rotations
    resampled_poses[:, :3, 3] = positions
    resampled_qpos = np.column_stack(
        [
            np.interp(target_times, source_times, qpos[:, joint])
            for joint in range(qpos.shape[1])
        ]
    )
    return resampled_poses, resampled_qpos, target_times


def transform_and_resample_mano_joints(
    episode: HumanEpisode,
    alignment_robot: np.ndarray,
    target_times: np.ndarray,
    *,
    source_hz: float,
    speed_scale: float,
) -> np.ndarray:
    """Put source MANO joints in the aligned robot frame at plan timestamps."""
    robot_from_world = np.linalg.inv(episode.camera_to_robot)
    source_robot = (
        np.einsum("ij,tkj->tki", robot_from_world[:3, :3], episode.joints_world)
        + robot_from_world[:3, 3]
    )
    alignment = _as_transform(alignment_robot, label="object alignment")
    aligned = (
        np.einsum("ij,tkj->tki", alignment[:3, :3], source_robot) + alignment[:3, 3]
    )
    source_times = np.arange(len(aligned), dtype=np.float64) / source_hz / speed_scale
    if len(aligned) == 1:
        return np.repeat(aligned, len(target_times), axis=0)
    flattened = aligned.reshape(len(aligned), -1)
    return np.column_stack(
        [
            np.interp(target_times, source_times, flattened[:, coordinate])
            for coordinate in range(flattened.shape[1])
        ]
    ).reshape(len(target_times), 21, 3)


def validate_plan(
    poses: np.ndarray,
    raw: np.ndarray,
    *,
    min_z_m: float,
    max_z_m: float,
    max_radius_m: float,
    max_linear_step_m: float,
    max_angular_step_rad: float,
    max_hand_raw_step: float,
) -> tuple[str, ...]:
    errors: list[str] = []
    xyz = np.asarray(poses)[:, :3, 3]
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    if np.min(xyz[:, 2]) < min_z_m:
        errors.append(f"link6 z below {min_z_m:.3f} m (min={np.min(xyz[:, 2]):.3f})")
    if np.max(xyz[:, 2]) > max_z_m:
        errors.append(f"link6 z above {max_z_m:.3f} m (max={np.max(xyz[:, 2]):.3f})")
    if np.max(radius) > max_radius_m:
        errors.append(
            f"link6 xy radius exceeds {max_radius_m:.3f} m (max={np.max(radius):.3f})"
        )
    if len(poses) > 1:
        linear = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        relative = np.einsum("tji,tjk->tik", poses[:-1, :3, :3], poses[1:, :3, :3])
        angular = R.from_matrix(relative).magnitude()
        hand_step = np.max(np.abs(np.diff(raw, axis=0)), axis=1)
        if np.max(linear) > max_linear_step_m:
            errors.append(
                f"arm translation step exceeds {max_linear_step_m:.3f} m (max={np.max(linear):.3f})"
            )
        if np.max(angular) > max_angular_step_rad:
            errors.append(
                f"arm rotation step exceeds {max_angular_step_rad:.3f} rad (max={np.max(angular):.3f})"
            )
        if np.max(hand_step) > max_hand_raw_step:
            errors.append(
                f"hand raw step exceeds {max_hand_raw_step:.1f} (max={np.max(hand_step):.1f})"
            )
    return tuple(errors)


def build_replay_plan(
    episode: HumanEpisode,
    current_object_robot: np.ndarray,
    *,
    aligned_current_object_robot: np.ndarray | None = None,
    symmetry_angle_rad: float = 0.0,
    symmetry_flipped: bool = False,
    source_hz: float = 30.0,
    control_hz: float = 30.0,
    speed_scale: float = 1.0,
    wrist_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
    scaling_factor: float = 1.15,
    low_pass_alpha: float = 0.2,
    safety: Mapping[str, float] | None = None,
) -> ReplayPlan:
    aligned_current = (
        current_object_robot
        if aligned_current_object_robot is None
        else aligned_current_object_robot
    )
    qpos, wrist_poses_world, fingertip_rmse_m = retarget_inspire_dexpilot(
        episode.joints_world,
        scaling_factor=scaling_factor,
        low_pass_alpha=low_pass_alpha,
    )
    poses, source_object_robot, alignment = align_wrist_trajectory(
        episode,
        aligned_current,
        wrist_poses_world,
        wrist_offset_m=wrist_offset_m,
    )
    poses, qpos, times = resample_commands(
        poses,
        qpos,
        source_hz=source_hz,
        control_hz=control_hz,
        speed_scale=speed_scale,
    )
    reference_mano_joints_robot = transform_and_resample_mano_joints(
        episode,
        alignment,
        times,
        source_hz=source_hz,
        speed_scale=speed_scale,
    )
    raw = hand_qpos_to_dftp_raw(qpos)
    limits = {
        "min_z_m": 0.03,
        "max_z_m": 1.00,
        "max_radius_m": 0.90,
        "max_linear_step_m": 0.04,
        "max_angular_step_rad": 0.45,
        "max_hand_raw_step": 180.0,
    }
    if safety is not None:
        limits.update({key: float(value) for key, value in safety.items()})
    errors = validate_plan(poses, raw, **limits)
    return ReplayPlan(
        object_name=episode.object_name,
        episode_id=episode.episode_id,
        episode_root=episode.root,
        times_s=times,
        link6_poses_robot=poses,
        hand_qpos=qpos,
        hand_raw=raw,
        source_object_robot=source_object_robot,
        current_object_robot=_as_transform(
            current_object_robot, label="current object pose"
        ),
        aligned_current_object_robot=_as_transform(
            aligned_current, label="symmetry-aligned current object pose"
        ),
        alignment_robot=alignment,
        symmetry_angle_rad=float(symmetry_angle_rad),
        symmetry_flipped=bool(symmetry_flipped),
        reference_mano_joints_robot=reference_mano_joints_robot,
        fingertip_rmse_m=fingertip_rmse_m,
        safety_errors=errors,
    )


def save_plan(plan: ReplayPlan, output_root: Path) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{plan.object_name}_human_{plan.episode_id}_inspire_dftp.npz"
    metadata = {
        "object_name": plan.object_name,
        "episode_id": plan.episode_id,
        "episode_root": str(plan.episode_root),
        "retargeting": "dex-retargeting DexPilot / Inspire right",
        "arm_command_frame": "robot_from_link6",
        "hand_qpos_joint_names": list(INSPIRE_COMMAND_JOINTS),
        "hand_raw_order": [
            "little",
            "ring",
            "middle",
            "index",
            "thumb_pitch",
            "thumb_yaw",
        ],
        "full_source_episode": True,
        "symmetry_angle_rad": plan.symmetry_angle_rad,
        "symmetry_flipped": plan.symmetry_flipped,
        "fingertip_registration": "equal-weight unscaled fingertip centroid",
        "fingertip_rmse_mean_m": float(np.mean(plan.fingertip_rmse_m)),
        "fingertip_rmse_max_m": float(np.max(plan.fingertip_rmse_m)),
        "safety_errors": list(plan.safety_errors),
    }
    np.savez_compressed(
        path,
        times_s=plan.times_s,
        link6_poses_robot=plan.link6_poses_robot,
        hand_qpos=plan.hand_qpos,
        hand_raw=plan.hand_raw,
        source_object_robot=plan.source_object_robot,
        current_object_robot=plan.current_object_robot,
        aligned_current_object_robot=plan.aligned_current_object_robot,
        alignment_robot=plan.alignment_robot,
        reference_mano_joints_robot=plan.reference_mano_joints_robot,
        fingertip_rmse_m=plan.fingertip_rmse_m,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    return path


def save_capture_artifacts(
    capture_root: Path,
    response: Mapping[str, Any],
    plan: ReplayPlan,
    *,
    rpc_addr: str,
) -> tuple[Path, Path, Path]:
    """Persist server and replay artifacts beside the captured images."""
    capture_root = Path(capture_root)
    capture_root.mkdir(parents=True, exist_ok=True)
    raw_response_path = capture_root / "object_6d_server_response.json"
    raw_response_path.write_text(
        json.dumps(dict(response), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = response.get("object_6d", response)
    pose_world = payload.get("pose_world") if isinstance(payload, Mapping) else None
    normalized = {
        "object_name": plan.object_name,
        "pose_world": pose_world,
        "pose_robot": plan.current_object_robot.astype(float).tolist(),
        "C2R_path": str(capture_root / "C2R.npy"),
        "rpc_addr": rpc_addr,
        "selected_human_episode": plan.episode_id,
        "selected_human_episode_root": str(plan.episode_root),
        "aligned_current_object_pose_robot": (
            plan.aligned_current_object_robot.astype(float).tolist()
        ),
        "symmetry_angle_rad": plan.symmetry_angle_rad,
        "symmetry_flipped": plan.symmetry_flipped,
        "fingertip_rmse_mean_m": float(np.mean(plan.fingertip_rmse_m)),
        "fingertip_rmse_max_m": float(np.max(plan.fingertip_rmse_m)),
        "replay_plan_frame_count": len(plan.times_s),
        "replay_plan_duration_s": float(plan.times_s[-1]),
        "safety_errors": list(plan.safety_errors),
    }
    normalized_path = capture_root / "object_6d.json"
    normalized_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plan_path = save_plan(plan, capture_root / "replay")
    return raw_response_path, normalized_path, plan_path


def visualize_replay_plan(
    plan: ReplayPlan,
    *,
    mesh_root: Path,
    preview_fps: float,
    object_max_faces: int,
    robot_link_max_faces: int,
) -> None:
    """Preview simplified object/hand meshes without camera imagery."""
    from threading import Event

    from paradex.visualization.visualizer.viser import ViserViewer

    viewer = ViserViewer(
        up_direction=np.array([0.0, 0.0, 1.0]),
        scene_title=(
            f"Human MANO -> Inspire DFTP: {plan.object_name}/{plan.episode_id}"
        ),
    )
    viewer.add_floor(height=0.0)
    viewer.add_frame("robot_origin", np.eye(4, dtype=np.float32))
    try:
        from src.object6d.capture_sil import load_mesh
        from paradex.visualization.robot import simplify_mesh

        mesh = load_mesh(Path(mesh_root).expanduser(), plan.object_name)
        source_faces = len(mesh.faces)
        mesh = simplify_mesh(mesh, object_max_faces)
        print(f"[preview] object mesh: {source_faces} -> {len(mesh.faces)} faces")
        viewer.add_object(
            plan.object_name,
            mesh,
            plan.current_object_robot,
            opacity=1.0,
        )
    except Exception as exc:
        print(f"[preview] object mesh visualization skipped: {exc}", file=sys.stderr)
        viewer.add_frame("current_object_pose", plan.current_object_robot)

    wrist_poses = plan.link6_poses_robot @ load_link6_from_wrist()[None]
    wrist_euler_xyz = R.from_matrix(wrist_poses[:, :3, :3]).as_euler("XYZ")
    floating_hand_qpos = np.concatenate(
        (wrist_poses[:, :3, 3], wrist_euler_xyz, plan.hand_qpos), axis=1
    )
    viewer.add_robot(
        name="inspire_dftp",
        urdf_path=str(INSPIRE_FLOAT_URDF),
        include_arm_meshes=True,
        max_mesh_faces=robot_link_max_faces,
    )
    expected_joint_names = [
        "x_joint",
        "y_joint",
        "z_joint",
        "x_rotation_joint",
        "y_rotation_joint",
        "z_rotation_joint",
        "right_thumb_1_joint",
        "right_thumb_2_joint",
        "right_index_1_joint",
        "right_middle_1_joint",
        "right_ring_1_joint",
        "right_little_1_joint",
    ]
    actual_joint_names = viewer.robot_dict["inspire_dftp"].urdf.get_joint_names()
    if actual_joint_names != expected_joint_names:
        raise RuntimeError(
            "floating Inspire URDF joint order mismatch: "
            f"expected={expected_joint_names}, actual={actual_joint_names}"
        )
    viewer.server.scene.add_spline_catmull_rom(
        name="/human_dftp/wrist_path",
        positions=wrist_poses[:, :3, 3],
        color=(60, 220, 255),
        line_width=3.0,
    )
    mano_joints = plan.reference_mano_joints_robot
    mano_colors = np.tile(
        np.array([[255, 80, 200]], dtype=np.uint8), (mano_joints.shape[1], 1)
    )
    mano_bone_colors = np.tile(
        np.array([[[255, 80, 200], [255, 80, 200]]], dtype=np.uint8),
        (len(MANO_BONES), 1, 1),
    )
    mano_points = viewer.server.scene.add_point_cloud(
        name="/human_dftp/reference_mano/joints",
        points=mano_joints[0],
        colors=mano_colors,
        point_size=0.008,
        point_shape="circle",
    )
    mano_bones = viewer.server.scene.add_line_segments(
        name="/human_dftp/reference_mano/bones",
        points=mano_joints[0][MANO_BONES],
        colors=mano_bone_colors,
        line_width=4.0,
    )
    viewer.server.scene.add_spline_catmull_rom(
        name="/human_dftp/reference_mano/wrist_path",
        positions=mano_joints[:, 0],
        color=(255, 80, 200),
        line_width=2.0,
    )
    viewer.add_traj(
        "retargeted_human_episode",
        robot_traj={"inspire_dftp": floating_hand_qpos},
    )

    original_update_scene = viewer.update_scene

    def update_scene_with_mano(timestep: int) -> None:
        original_update_scene(timestep)
        frame = max(0, min(len(mano_joints) - 1, int(timestep)))
        with viewer.server.atomic():
            mano_points.points = mano_joints[frame]
            mano_bones.points = mano_joints[frame][MANO_BONES]

    viewer.update_scene = update_scene_with_mano
    viewer.gui_framerate.value = float(preview_fps)
    viewer.gui_playing.value = False
    viewer.gui_timestep.value = 0
    viewer.update_scene(0)

    finish_event = Event()
    finish_button = viewer.server.gui.add_button("Finish preview")

    @finish_button.on_click
    def _(_) -> None:
        finish_event.set()

    print(
        "[preview] Viser is paused at frame 0. Use Resume or the timestep "
        "slider to inspect the full retargeted episode, then click Finish preview."
    )
    try:
        while not finish_event.is_set():
            viewer.update()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        stop = getattr(viewer.server, "stop", None)
        if callable(stop):
            stop()


def _interpolate_pose(start: np.ndarray, target: np.ndarray, steps: int) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, steps + 1)[1:]
    result = np.repeat(np.eye(4)[None], steps, axis=0)
    result[:, :3, 3] = start[:3, 3] + fractions[:, None] * (
        target[:3, 3] - start[:3, 3]
    )
    result[:, :3, :3] = Slerp(
        [0.0, 1.0], R.from_matrix([start[:3, :3], target[:3, :3]])
    )(fractions).as_matrix()
    return result


def execute_plan(plan: ReplayPlan, args: argparse.Namespace) -> None:
    if plan.safety_errors:
        raise RuntimeError(
            "execution blocked by trajectory safety checks: "
            + "; ".join(plan.safety_errors)
        )
    if not args.yes:
        raise RuntimeError("hardware execution requires both --execute and --yes")
    from paradex.io.robot_controller import get_arm
    from src.object6d.capture_sil_sorting import RightInspireDFTP

    arm = None
    hand = None
    try:
        hand = RightInspireDFTP(
            args.inspire_right_ip,
            args.inspire_port,
            args.inspire_timeout,
            args.inspire_interface,
        )
        arm = get_arm("xarm", servo_api="cartesian_aa")
        current_pose = _as_transform(
            arm.get_data()["position"], label="current xArm pose"
        )
        current_raw = np.asarray(hand.get_data()["joint_value"], dtype=np.float64)
        target_pose = plan.link6_poses_robot[0]
        linear_distance = np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3])
        angular_distance = R.from_matrix(
            current_pose[:3, :3].T @ target_pose[:3, :3]
        ).magnitude()
        duration = max(
            args.approach_min_seconds,
            linear_distance / args.approach_linear_speed_mps,
            angular_distance / args.approach_angular_speed_rps,
        )
        steps = max(
            1,
            int(np.ceil(duration * args.approach_rate_hz)),
            int(
                np.ceil(
                    np.max(np.abs(plan.hand_raw[0] - current_raw))
                    / args.max_hand_raw_step
                )
            ),
        )
        duration = max(duration, steps / args.approach_rate_hz)
        approach_poses = _interpolate_pose(current_pose, target_pose, steps)
        approach_raw = np.linspace(current_raw, plan.hand_raw[0], steps + 1)[1:]

        def stream(
            poses: np.ndarray, raw: np.ndarray, times_s: np.ndarray, stage: str
        ) -> None:
            started = time.perf_counter()
            for frame, (pose, command, target_time) in enumerate(
                zip(poses, raw, times_s)
            ):
                arm.move(pose)
                hand.move(command)
                if arm.is_error():
                    raise RuntimeError(f"xArm error during {stage}, frame {frame}")
                time.sleep(max(0.0, started + float(target_time) - time.perf_counter()))

        approach_times = np.linspace(duration / steps, duration, steps)
        print(f"[execute] approach: {steps} frames, {duration:.2f} s")
        stream(approach_poses, approach_raw, approach_times, "approach")
        print(
            f"[execute] original human episode {plan.object_name}/{plan.episode_id}: "
            f"{len(plan.times_s)} frames, {plan.times_s[-1]:.2f} s"
        )
        stream(plan.link6_poses_robot, plan.hand_raw, plan.times_s, "episode replay")
        print("[execute] replay complete; final arm and hand commands are held")
    finally:
        if arm is not None:
            arm.end()
        if hand is not None:
            hand.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-root", type=Path, default=DEFAULT_HUMAN_ROOT)
    parser.add_argument("--object", dest="object_name")
    parser.add_argument(
        "--episode", default="auto", help="episode directory name or 'auto'"
    )
    parser.add_argument(
        "--object-pose", type=Path, help="server-provided current object 6D pose"
    )
    parser.add_argument(
        "--rpc-capture-root",
        type=Path,
        help="existing ~/shared_data capture directory to submit to Object6D",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="capture fresh multiview images, then request Object6D from capture13",
    )
    parser.add_argument(
        "--capture-save-path",
        type=Path,
        default=DEFAULT_CAPTURE_SAVE_PATH,
        help="fresh capture path relative to ~/shared_data",
    )
    parser.add_argument("--rcc-entry", default="image_main.py")
    parser.add_argument("--rpc-addr", "--rpc_addr", default=DEFAULT_RPC_ADDR)
    parser.add_argument(
        "--rpc-timeout-ms", "--rpc_timeout_ms", type=int, default=300_000
    )
    parser.add_argument(
        "--pose-frame", choices=("auto", "robot", "world"), default="auto"
    )
    parser.add_argument("--current-c2r", type=Path)
    parser.add_argument("--list-objects", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--preview-object-max-faces", type=int, default=3_000)
    parser.add_argument("--preview-robot-link-max-faces", type=int, default=1_500)
    parser.add_argument(
        "--no-preview",
        "--no_vis",
        action="store_true",
        help="skip the Viser object and retargeted-hand trajectory preview",
    )
    parser.add_argument(
        "--preview-fps",
        type=float,
        help="Viser playback rate; defaults to --control-hz",
    )
    parser.add_argument("--source-hz", type=float, default=30.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--retrieval-translation-scale-m", type=float, default=0.05)
    parser.add_argument("--retrieval-rotation-scale-rad", type=float, default=0.25)
    parser.add_argument(
        "--wrist-offset-m", nargs=3, type=float, default=(0.0, 0.0, 0.0)
    )
    parser.add_argument("--scaling-factor", type=float, default=1.15)
    parser.add_argument("--low-pass-alpha", type=float, default=0.2)
    parser.add_argument("--min-z-m", type=float, default=0.03)
    parser.add_argument("--max-z-m", type=float, default=1.00)
    parser.add_argument("--max-radius-m", type=float, default=0.90)
    parser.add_argument("--max-linear-step-m", type=float, default=0.04)
    parser.add_argument("--max-angular-step-rad", type=float, default=0.45)
    parser.add_argument("--max-hand-raw-step", type=float, default=180.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--inspire-right-ip", default="192.168.11.211")
    parser.add_argument("--inspire-port", type=int, default=6000)
    parser.add_argument("--inspire-timeout", type=float, default=1.0)
    parser.add_argument("--inspire-interface", default="enp8s0f2")
    parser.add_argument("--approach-rate-hz", type=float, default=30.0)
    parser.add_argument("--approach-min-seconds", type=float, default=2.0)
    parser.add_argument("--approach-linear-speed-mps", type=float, default=0.08)
    parser.add_argument("--approach-angular-speed-rps", type=float, default=0.5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Keep logs immediate when stdout/stderr are pipes. `conda run` additionally
    # needs `--no-capture-output`; its parent-side capture cannot be disabled by
    # this child process.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)
    args = build_arg_parser().parse_args(argv)
    if args.list_objects:
        for object_name in discover_objects(args.human_root):
            print(object_name)
        return 0
    input_count = sum(
        (args.object_pose is not None, args.rpc_capture_root is not None, args.capture)
    )
    if input_count != 1:
        raise SystemExit(
            "choose exactly one current-pose source: --object-pose, "
            "--rpc-capture-root, or --capture"
        )
    if args.preview_object_max_faces <= 0 or args.preview_robot_link_max_faces <= 0:
        raise SystemExit("preview mesh face limits must be positive")
    capture_root: Path | None = None
    rpc_response: dict[str, Any] | None = None
    if args.object_pose is not None:
        pose_object_name, current_object_robot = load_server_object_pose(
            args.object_pose,
            pose_frame=args.pose_frame,
            current_c2r=args.current_c2r,
        )
    else:
        if args.capture:
            if (
                args.capture_save_path.is_absolute()
                or ".." in args.capture_save_path.parts
            ):
                raise SystemExit(
                    "--capture-save-path must be relative to ~/shared_data"
                )
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            capture_root = (
                Path.home() / "shared_data" / args.capture_save_path / timestamp
            )
            capture_multiview(capture_root, args.rcc_entry)
        else:
            capture_root = args.rpc_capture_root.expanduser()
            if not capture_root.is_absolute():
                capture_root = Path.home() / "shared_data" / capture_root
            if not capture_root.is_dir():
                raise SystemExit(f"RPC capture root does not exist: {capture_root}")
        pose_object_name, current_object_robot, rpc_response = (
            request_current_object_pose(
                capture_root=capture_root,
                human_root=args.human_root,
                object_name=args.object_name,
                rpc_addr=args.rpc_addr,
                rpc_timeout_ms=args.rpc_timeout_ms,
                current_c2r_path=args.current_c2r,
            )
        )
        response_path = args.output_root / "latest_object6d_response.json"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(rpc_response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[rpc] saved raw server response: {response_path}")
        capture_response_path = capture_root / "object_6d_server_response.json"
        capture_response_path.write_text(
            json.dumps(rpc_response, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[rpc] saved raw response beside capture: {capture_response_path}")
    object_name = args.object_name or pose_object_name
    if not object_name:
        raise SystemExit("--object is required when the pose file has no object_name")
    if args.object_name and pose_object_name and args.object_name != pose_object_name:
        raise SystemExit(
            f"object mismatch: --object={args.object_name!r}, pose file={pose_object_name!r}"
        )
    episode_roots = discover_episodes(args.human_root, object_name)
    if not episode_roots:
        raise SystemExit(f"no valid human episodes found for {object_name!r}")
    ranked = rank_episodes(
        episode_roots,
        current_object_robot,
        object_name=object_name,
        translation_scale_m=args.retrieval_translation_scale_m,
        rotation_scale_rad=args.retrieval_rotation_scale_rad,
    )
    if args.episode == "auto":
        selected = ranked[0]
    else:
        episode_root = args.human_root / object_name / args.episode
        if episode_root not in episode_roots:
            raise SystemExit(
                f"episode {object_name}/{args.episode} is missing or incomplete"
            )
        selected = next(match for match in ranked if match.episode_root == episode_root)
    episode_root = selected.episode_root
    symmetry_axis = CONTINUOUS_SYMMETRY_AXIS_OBJECT.get(object_name)
    print(f"[select] episode candidates by frame-0 {object_name} pose:")
    for match in ranked[:5]:
        symmetry_text = (
            f", symmetry={np.rad2deg(match.symmetry_angle_rad):+.1f} deg"
            f", flipped={'yes' if match.symmetry_flipped else 'no'}"
            if symmetry_axis is not None
            else ""
        )
        marker = " *" if match.episode_root == episode_root else ""
        print(
            f"  {match.episode_root.name}: "
            f"translation={match.translation_error_m:.4f} m, "
            f"rotation={np.rad2deg(match.rotation_error_rad):.1f} deg"
            f"{symmetry_text}, score={match.score:.3f}{marker}"
        )
    episode = load_human_episode(episode_root, object_name)
    plan = build_replay_plan(
        episode,
        current_object_robot,
        aligned_current_object_robot=selected.aligned_current_object_robot,
        symmetry_angle_rad=selected.symmetry_angle_rad,
        symmetry_flipped=selected.symmetry_flipped,
        source_hz=args.source_hz,
        control_hz=args.control_hz,
        speed_scale=args.speed_scale,
        wrist_offset_m=args.wrist_offset_m,
        scaling_factor=args.scaling_factor,
        low_pass_alpha=args.low_pass_alpha,
        safety={
            "min_z_m": args.min_z_m,
            "max_z_m": args.max_z_m,
            "max_radius_m": args.max_radius_m,
            "max_linear_step_m": args.max_linear_step_m,
            "max_angular_step_rad": args.max_angular_step_rad,
            "max_hand_raw_step": args.max_hand_raw_step,
        },
    )
    path = save_plan(plan, args.output_root)
    print(
        f"[plan] {object_name}/{episode.episode_id}: {len(episode.joints_world)} source frames -> "
        f"{len(plan.times_s)} commands, duration={plan.times_s[-1]:.2f}s"
    )
    print(
        "[plan] fingertip registration RMSE: "
        f"mean={np.mean(plan.fingertip_rmse_m) * 1000.0:.1f} mm, "
        f"max={np.max(plan.fingertip_rmse_m) * 1000.0:.1f} mm"
    )
    print(f"[plan] saved {path}")
    if capture_root is not None and rpc_response is not None:
        raw_path, pose_path, capture_plan_path = save_capture_artifacts(
            capture_root,
            rpc_response,
            plan,
            rpc_addr=args.rpc_addr,
        )
        print(
            "[artifact] saved beside capture: "
            f"{raw_path.name}, {pose_path.name}, replay/{capture_plan_path.name}"
        )
    if plan.safety_errors:
        print(
            "[safety] preview only: " + "; ".join(plan.safety_errors), file=sys.stderr
        )
    if not args.no_preview:
        preview_fps = args.preview_fps or args.control_hz
        if preview_fps <= 0:
            raise ValueError("--preview-fps must be positive")
        visualize_replay_plan(
            plan,
            mesh_root=args.mesh_root,
            preview_fps=preview_fps,
            object_max_faces=args.preview_object_max_faces,
            robot_link_max_faces=args.preview_robot_link_max_faces,
        )
    if args.execute:
        execute_plan(plan, args)
    else:
        print(
            "[plan] no hardware commands sent; use --execute --yes after inspecting the plan"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
