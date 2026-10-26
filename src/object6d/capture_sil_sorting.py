#!/usr/bin/env python3
"""Capture and sort one detected object to the left board or right pink box.

The RPC request asks the server to choose one of banana, apple, pringles, and
spam_can and to triangulate both placement targets.  The accepted normalized
response shape is::

    {
      "object_6d": {
        "object_name": "apple",
        "pose_world": [[...], ...],
        "targets": {
          "cutting_board": {"status": "success", "point_world": [x, y, z]},
          "pink_box": {"status": "success", "point_world": [x, y, z]}
        }
      }
    }

Several field aliases are accepted by :func:`extract_sorting_detection` so a
server can migrate from the older ``wooden_object`` response incrementally.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paradex.retargetor.experiment.replay_pose_retrieval import (
    load_candidate_episodes,
    rank_by_object_pose,
    replay_closest_episode_into_box_pink,
)
from paradex.planning.nfg_return import (
    NFGReturnPlanningError,
    SphereObstacle,
    plan_collision_free_return,
)
from paradex.utils.path import shared_dir
from src.object6d import capture_sil as base
from src.object6d.object_symmetry import (
    AXIS_FLIP_SYMMETRIC_OBJECTS,
    CONTINUOUS_SYMMETRY_AXIS_OBJECT,
)


SUPPORTED_OBJECTS = ("banana", "apple", "pringles", "spam_can")
SORTING_MESH_ROOT = Path(shared_dir) / "mesh_new"
DEMOGRASP_HAND_URDF = PROJECT_ROOT / "rsc/robot/inspire_float.urdf"
XARM_INSPIRE_URDF = PROJECT_ROOT / "rsc/robot/xarm_inspire_DFTP.urdf"
INSPIRE_EPISODE_ROOT = (
    Path(shared_dir) / "capture/eccv2026/v0/inspire_dftp"
)
DEFAULT_REPLAY_START_ANNOTATION_NAME = "replay_start_annotations.json"
DEMOGRASP_HAND_JOINTS = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
)
DFTP_RAW_JOINTS = (
    "right_little_1_joint",
    "right_ring_1_joint",
    "right_middle_1_joint",
    "right_index_1_joint",
    "right_thumb_2_joint",
    "right_thumb_1_joint",
)
DFTP_ANGLE_SET = 1486
DFTP_ANGLE_ACT = 1546
DFTP_JOINT_COUNT = 6
IP_UNICAST_IF = 50
DFTP_OPEN_RAW = np.array([998, 1000, 998, 1000, 1000, 990], dtype=np.float64)
READY_ARM_QPOS_DEG = np.array(
    [-56.2, 13.8, -80.3, 75.4, 56.0, 173.3], dtype=np.float64
)
READY_ARM_QPOS_RAD = np.deg2rad(READY_ARM_QPOS_DEG)
TARGET_BY_OBJECT = {
    "banana": "cutting_board",
    "apple": "cutting_board",
    "pringles": "pink_box",
    "spam_can": "pink_box",
}
EXCLUDED_INSPIRE_EPISODES = {
    "pringles": frozenset({7}),
}
TARGET_ALIASES = {
    "cutting_board": (
        "cutting_board",
        "cutting-board",
        "cutting board",
        "cutting_board_triangulation",
        "wooden_object",
        "dish",
        "left_target",
    ),
    "pink_box": (
        "pink_box",
        "pink-box",
        "pink box",
        "pink_box_triangulation",
        "box_pink",
        "right_target",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = base.build_arg_parser()
    parser.description = (
        "Detect one of four objects and place it on the class-dependent board or box."
    )
    hidden_base_options = {
        "mesh_name",
        "mesh_root_dir",
        "replay",
        "retrieval_only",
        "naive_replay",
        "put_into_box_pink",
        "episode_root",
        "candidate_episodes",
        "dish_clearance_m",
    }
    for action in parser._actions:
        if action.dest in hidden_base_options:
            action.help = argparse.SUPPRESS
    parser.add_argument(
        "--cutting-board-z-m",
        type=float,
        default=0.20,
        help="absolute robot-frame z for the thumb-index midpoint over the board",
    )
    parser.add_argument(
        "--inspire-episode-replay",
        "--episode-replay",
        action="store_true",
        help=(
            "retrieve an object-relative trajectory from the Inspire DFTP "
            "capture dataset instead of requesting a DemoGrasp trajectory"
        ),
    )
    parser.add_argument(
        "--inspire-episode-root",
        type=Path,
        default=INSPIRE_EPISODE_ROOT,
        help=(
            "dataset directory containing <object>/<episode> for "
            "--inspire-episode-replay"
        ),
    )
    parser.add_argument(
        "--replay-start-annotations",
        type=Path,
        help=(
            "manual source-frame annotation JSON; defaults to "
            "<inspire-episode-root>/replay_start_annotations.json"
        ),
    )
    parser.add_argument(
        "--hand-control-hz",
        "--demograsp-replay-hz",
        dest="hand_control_hz",
        type=float,
        default=50.0,
        help="post-interpolated DemoGrasp xArm+hand replay rate (default: 50 Hz)",
    )
    parser.add_argument(
        "--demograsp-duration-seconds",
        type=float,
        default=1.5,
        help="target duration after DemoGrasp interpolation (default: 1.5 s)",
    )
    parser.add_argument(
        "--demograsp-z-offset-m",
        type=float,
        default=0.002,
        help="z offset added to every DemoGrasp wrist frame (default: 0.002 m)",
    )
    parser.add_argument(
        "--motion-control-hz",
        type=float,
        default=50.0,
        help="approach, transfer, and hand-restore rate (default: 50 Hz)",
    )
    parser.add_argument(
        "--hand-bridge-seconds",
        type=float,
        default=1.0,
        help="smooth transition time from the current hand state to frame 0",
    )
    parser.add_argument(
        "--hand-max-raw-step",
        type=float,
        default=50.0,
        help="maximum DFTP raw motor change per control command",
    )
    parser.add_argument(
        "--inspire-right-ip",
        default="192.168.11.211",
        help="right Inspire RH56DFTP Modbus TCP address (default: 192.168.11.211)",
    )
    parser.add_argument(
        "--inspire-interface",
        default="enp8s0f2",
        help="NIC connected to the right Inspire hand (default: enp8s0f2)",
    )
    parser.add_argument("--inspire-port", type=int, default=6000)
    parser.add_argument("--inspire-timeout", type=float, default=2.0)
    parser.add_argument(
        "--arm-xy-seconds",
        type=float,
        default=2.0,
        help="duration of the XY-only xArm approach to DemoGrasp frame 0",
    )
    parser.add_argument(
        "--arm-frame0-seconds",
        type=float,
        default=1.0,
        help="duration of the synchronized pose/hand approach to frame 0",
    )
    parser.add_argument(
        "--post-grasp-hold-seconds",
        type=float,
        default=0.0,
        help="optional hold at the pink-box target before restoring the hand",
    )
    parser.add_argument(
        "--post-grasp-z-m",
        type=float,
        default=0.40,
        help="absolute wrist z used after grasp (default: 0.40 m)",
    )
    parser.add_argument(
        "--post-grasp-lift-seconds",
        type=float,
        default=0.5,
        help="minimum duration to move the grasped object to absolute z",
    )
    parser.add_argument(
        "--post-grasp-transfer-seconds",
        type=float,
        default=2.0,
        help="minimum duration to move in XY to the triangulated pink box",
    )
    parser.add_argument(
        "--cutting-board-release-z-m",
        type=float,
        default=0.30,
        help="absolute wrist z before releasing on the cutting board (default: 0.30 m)",
    )
    parser.add_argument(
        "--cutting-board-descent-seconds",
        type=float,
        default=2.0,
        help="minimum apple/banana descent duration after XY transfer",
    )
    parser.add_argument(
        "--post-placement-retreat-z-m",
        type=float,
        default=0.40,
        help="absolute robot-frame z before returning to ready joints (default: 0.40 m)",
    )
    parser.add_argument(
        "--post-placement-retreat-seconds",
        type=float,
        default=2.0,
        help="minimum duration of the post-placement vertical retreat",
    )
    parser.add_argument(
        "--cutting-board-x-offset-m",
        type=float,
        default=-0.10,
        help="placement x offset from cutting-board triangulation (default: -0.10 m)",
    )
    parser.add_argument(
        "--cutting-board-y-offset-m",
        type=float,
        default=-0.08,
        help="placement y offset from cutting-board triangulation (default: -0.08 m)",
    )
    parser.add_argument(
        "--pink-box-x-offset-m",
        type=float,
        default=-0.10,
        help="placement x offset from pink-box triangulation (default: -0.10 m)",
    )
    parser.add_argument(
        "--pink-box-y-offset-m",
        type=float,
        default=0.10,
        help="placement y offset from pink-box triangulation (default: 0.10 m)",
    )
    parser.add_argument(
        "--nfg-return",
        action="store_true",
        help=(
            "use natural-functional-gradient collision planning when returning "
            "to the session-initial xArm pose after placement"
        ),
    )
    parser.add_argument(
        "--nfg-repo",
        type=Path,
        default=Path.home() / "natural-functional-gradient",
        help="natural-functional-gradient checkout used for collision-free returns",
    )
    parser.add_argument(
        "--nfg-backend",
        choices=("auto", "cpu", "warp"),
        default="auto",
        help="NFG collision-scoring backend (final path is always CPU-verified)",
    )
    parser.add_argument("--nfg-num-samples", type=int, default=100)
    parser.add_argument("--nfg-max-iterations", type=int, default=30)
    parser.add_argument("--nfg-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--nfg-obstacle-margin-m",
        type=float,
        default=0.03,
        help="extra collision radius around each previously placed object",
    )
    parser.add_argument("--arm-min-z-m", type=float, default=0.05)
    parser.add_argument("--arm-max-z-m", type=float, default=0.80)
    parser.add_argument("--arm-min-radius-m", type=float, default=0.15)
    parser.add_argument("--arm-max-radius-m", type=float, default=0.85)
    parser.add_argument(
        "--arm-max-step-m",
        type=float,
        default=0.02,
        help="maximum Cartesian translation per control command",
    )
    parser.add_argument(
        "--arm-max-angle-step-deg",
        type=float,
        default=10.0,
        help="maximum Cartesian rotation per control command",
    )
    args = parser.parse_args(argv)
    if args.mesh_name is not None:
        parser.error("--name is not used: the sorting server selects the object")
    if args.replay or args.retrieval_only or args.naive_replay or args.put_into_box_pink:
        parser.error(
            "replay mode flags are implicit in capture_sil_sorting.py and must be omitted"
        )
    if args.episode_root is not None or args.candidate_episodes is not None:
        parser.error(
            "--episode-root/--candidate-episodes cannot describe all four sorting objects"
        )
    if args.cutting_board_z_m < 0:
        parser.error("--cutting-board-z-m must be non-negative")
    if args.hand_control_hz <= 0:
        parser.error("--hand-control-hz must be positive")
    if args.hand_control_hz > 100:
        parser.error("--hand-control-hz must not exceed 100 Hz")
    if args.demograsp_duration_seconds <= 0:
        parser.error("--demograsp-duration-seconds must be positive")
    if not np.isfinite(args.demograsp_z_offset_m):
        parser.error("--demograsp-z-offset-m must be finite")
    if args.motion_control_hz <= 0:
        parser.error("--motion-control-hz must be positive")
    if args.motion_control_hz > 100:
        parser.error("--motion-control-hz must not exceed 100 Hz")
    if args.hand_bridge_seconds < 0:
        parser.error("--hand-bridge-seconds must be non-negative")
    if args.hand_max_raw_step <= 0:
        parser.error("--hand-max-raw-step must be positive")
    if not 0 < args.inspire_port <= 65535:
        parser.error("--inspire-port must be between 1 and 65535")
    if not 0 < args.inspire_timeout <= 10:
        parser.error("--inspire-timeout must be between 0 and 10 seconds")
    if args.arm_xy_seconds <= 0 or args.arm_frame0_seconds <= 0:
        parser.error("xArm approach durations must be positive")
    if args.post_grasp_hold_seconds < 0:
        parser.error("--post-grasp-hold-seconds must be non-negative")
    if args.post_grasp_lift_seconds <= 0 or args.post_grasp_transfer_seconds <= 0:
        parser.error("post-grasp motion durations must be positive")
    if args.cutting_board_descent_seconds <= 0:
        parser.error("--cutting-board-descent-seconds must be positive")
    if args.post_placement_retreat_seconds <= 0:
        parser.error("--post-placement-retreat-seconds must be positive")
    if not all(
        np.isfinite(value)
        for value in (
            args.cutting_board_x_offset_m,
            args.cutting_board_y_offset_m,
            args.pink_box_x_offset_m,
            args.pink_box_y_offset_m,
        )
    ):
        parser.error("placement offsets must be finite")
    if args.nfg_return:
        if args.nfg_num_samples < 1 or args.nfg_max_iterations < 1:
            parser.error("NFG sample and iteration counts must be positive")
        if args.nfg_timeout_seconds <= 0 or args.nfg_obstacle_margin_m < 0:
            parser.error("NFG timeout must be positive and obstacle margin non-negative")
    if not args.arm_min_z_m <= args.post_grasp_z_m <= args.arm_max_z_m:
        parser.error("--post-grasp-z-m must be inside the configured xArm z bounds")
    if not args.arm_min_z_m <= args.cutting_board_release_z_m <= args.arm_max_z_m:
        parser.error(
            "--cutting-board-release-z-m must be inside the configured xArm z bounds"
        )
    if not args.arm_min_z_m <= args.post_placement_retreat_z_m <= args.arm_max_z_m:
        parser.error(
            "--post-placement-retreat-z-m must be inside the configured xArm z bounds"
        )
    if not 0 <= args.arm_min_z_m < args.arm_max_z_m:
        parser.error("xArm z bounds must satisfy 0 <= min < max")
    if not 0 <= args.arm_min_radius_m < args.arm_max_radius_m:
        parser.error("xArm radius bounds must satisfy 0 <= min < max")
    if args.arm_max_step_m <= 0 or args.arm_max_angle_step_deg <= 0:
        parser.error("xArm step limits must be positive")
    args.inspire_episode_root = args.inspire_episode_root.expanduser()
    args.nfg_repo = args.nfg_repo.expanduser()
    args.replay_start_annotations = (
        args.replay_start_annotations.expanduser()
        if args.replay_start_annotations is not None
        else args.inspire_episode_root / DEFAULT_REPLAY_START_ANNOTATION_NAME
    )
    if args.inspire_episode_replay and not args.inspire_episode_root.is_dir():
        parser.error(
            "--inspire-episode-root is not a directory: "
            f"{args.inspire_episode_root}"
        )
    if args.inspire_episode_replay and not args.replay_start_annotations.is_file():
        parser.error(
            "--replay-start-annotations is not a file: "
            f"{args.replay_start_annotations}"
        )

    # Reuse validation and replay defaults from capture_sil without exposing a
    # misleading destination-specific mode flag to the caller.
    args.mesh_root_dir = SORTING_MESH_ROOT
    args.put_into_box_pink = True
    base.validate_args(args)
    return args


def _normalized_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _first_mapping(*values: Any) -> Mapping[str, Any] | None:
    return next((value for value in values if isinstance(value, Mapping)), None)


def _detected_object_name(response: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    candidates: list[Any] = [
        payload.get("detected_object_name"),
        payload.get("predicted_object_name"),
        payload.get("object_name"),
        payload.get("class_name"),
        payload.get("mesh_name"),
        response.get("detected_object_name"),
        response.get("predicted_object_name"),
        response.get("object_name"),
        response.get("class_name"),
        response.get("mesh_name"),
    ]
    object_field = _first_mapping(
        payload.get("object"),
        payload.get("detected_object"),
        response.get("object"),
        response.get("detected_object"),
    )
    if object_field is not None:
        candidates.extend(
            object_field.get(key) for key in ("name", "object_name", "mesh_name")
        )
    for candidate in candidates:
        if candidate is None:
            continue
        name = _normalized_name(candidate)
        if name in SUPPORTED_OBJECTS:
            return name
    raise ValueError(
        "sorting response has no supported detected object name; expected one of "
        f"{SUPPORTED_OBJECTS}"
    )


def _object_record(payload: Mapping[str, Any], object_name: str) -> Mapping[str, Any]:
    objects = payload.get("objects")
    if isinstance(objects, Mapping):
        for key, value in objects.items():
            if _normalized_name(key) == object_name and isinstance(value, Mapping):
                return value
    elif isinstance(objects, list):
        for value in objects:
            if not isinstance(value, Mapping):
                continue
            name = value.get("object_name", value.get("mesh_name", value.get("name")))
            if name is not None and _normalized_name(name) == object_name:
                return value
    return payload


def _point_world(value: Any, *, label: str) -> np.ndarray:
    if isinstance(value, Mapping):
        status = value.get("status")
        if status not in (None, "success", "ok"):
            raise ValueError(f"{label} triangulation failed: status={status!r}")
        value = next(
            (
                value[key]
                for key in ("point_world", "center_world", "position_world", "point")
                if key in value
            ),
            None,
        )
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{label} must contain a finite world-frame xyz point")
    return point


def _target_record(
    response: Mapping[str, Any],
    payload: Mapping[str, Any],
    canonical_name: str,
) -> Any:
    containers = [
        payload.get("targets"),
        payload.get("target_triangulations"),
        payload.get("placement_targets"),
        payload.get("wooden_objects"),
        response.get("targets"),
        response.get("target_triangulations"),
        response.get("placement_targets"),
        response.get("wooden_objects"),
        payload,
        response,
    ]
    aliases = {_normalized_name(alias) for alias in TARGET_ALIASES[canonical_name]}
    for container in containers:
        if isinstance(container, list):
            for value in container:
                if not isinstance(value, Mapping):
                    continue
                name = value.get(
                    "target_name", value.get("name", value.get("label"))
                )
                if name is not None and _normalized_name(name) in aliases:
                    return value
            continue
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            if _normalized_name(key) in aliases:
                return value
    raise ValueError(f"sorting response is missing {canonical_name} triangulation")


def extract_sorting_detection(response: Mapping[str, Any]) -> dict[str, Any]:
    payload = response.get("object_6d", response)
    if not isinstance(payload, Mapping):
        raise ValueError("sorting response object_6d must be a JSON object")
    object_name = _detected_object_name(response, payload)
    record = _object_record(payload, object_name)
    pose_world = record.get("pose_world")
    if pose_world is None:
        raise ValueError(f"sorting response has no pose_world for {object_name}")
    return {
        "object_name": object_name,
        "pose_world": base.to_4x4(pose_world).astype(np.float64),
        "cutting_board_world": _point_world(
            _target_record(response, payload, "cutting_board"),
            label="cutting_board",
        ),
        "pink_box_world": _point_world(
            _target_record(response, payload, "pink_box"),
            label="pink_box",
        ),
    }


def sorting_rpc_request(
    capture_root: Path,
    *,
    generate_grasp_trajectory: bool = True,
) -> dict[str, Any]:
    return {
        "command": "infer",
        "image_path": base.to_shared_data_path(capture_root),
        "mesh_names": list(SUPPORTED_OBJECTS),
        "object_candidates": list(SUPPORTED_OBJECTS),
        "detect_object_name": True,
        "generate_grasp_trajectory": bool(generate_grasp_trajectory),
        "save_projection_grids": True,
        "target_triangulations": {
            "cutting_board": "wooden object",
            "pink_box": "pink box",
        },
        "target_prompts": {
            "cutting_board": "wooden object",
            "pink_box": "pink box",
        },
        # Compatibility hint for servers whose triangulation entry point still
        # uses the older wooden-object flag.
        "wooden_object_triangulation": True,
    }


def placement_target_xy(
    object_name: str,
    cutting_board_robot: np.ndarray,
    pink_box_robot: np.ndarray,
    *,
    cutting_board_x_offset_m: float,
    cutting_board_y_offset_m: float,
    pink_box_x_offset_m: float,
    pink_box_y_offset_m: float,
) -> tuple[str, np.ndarray]:
    """Select class-dependent placement XY and apply target-specific offsets."""
    normalized_name = _normalized_name(object_name)
    target_name = TARGET_BY_OBJECT.get(normalized_name)
    if target_name is None:
        raise ValueError(f"unsupported sorting object: {object_name!r}")
    cutting = np.asarray(cutting_board_robot, dtype=np.float64).reshape(-1)
    pink = np.asarray(pink_box_robot, dtype=np.float64).reshape(-1)
    if cutting.shape != (3,) or pink.shape != (3,):
        raise ValueError("triangulated placement points must each have shape [3]")
    if target_name == "cutting_board":
        xy = cutting[:2] + np.array(
            [cutting_board_x_offset_m, cutting_board_y_offset_m], dtype=np.float64
        )
    else:
        xy = pink[:2] + np.array(
            [pink_box_x_offset_m, pink_box_y_offset_m], dtype=np.float64
        )
    if not np.all(np.isfinite(xy)):
        raise ValueError("resolved placement XY is not finite")
    return target_name, xy


def placed_object_obstacle(
    object_name: str,
    target_robot: np.ndarray,
    *,
    margin_m: float,
    mesh_root: Path = SORTING_MESH_ROOT,
) -> SphereObstacle:
    """Conservatively bound a placed object for NFG collision checking."""

    normalized_name = _normalized_name(object_name)
    diameter_path = (
        Path(mesh_root)
        / normalized_name
        / f"{normalized_name}.obj.diameter.json"
    )
    try:
        diameter_record = json.loads(diameter_path.read_text(encoding="utf-8"))
        physical_radius_m = float(diameter_record["diameter_mm"]) / 2000.0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot build NFG obstacle for {normalized_name}: {diameter_path}"
        ) from exc
    center = np.asarray(target_robot, dtype=np.float64).reshape(-1).copy()
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("placed-object target must be a finite [3] robot-frame point")
    # Target triangulations represent the support location.  Raise the sphere
    # center so its conservative volume sits above that surface.
    center[2] += physical_radius_m
    return SphereObstacle(
        name=normalized_name,
        center_m=center,
        radius_m=physical_radius_m + float(margin_m),
    )


def execute_nfg_return(
    arm: Any,
    return_arm_qpos: np.ndarray,
    obstacles: list[SphereObstacle],
    args: argparse.Namespace,
    capture_root: Path,
) -> None:
    """Plan, persist, then execute a CPU-verified NFG joint return."""

    current_qpos = np.asarray(arm.get_data().get("qpos"), dtype=np.float64)
    try:
        plan = plan_collision_free_return(
            current_qpos,
            return_arm_qpos,
            obstacles,
            repo_root=Path(args.nfg_repo),
            rate_hz=float(args.return_rate_hz),
            speed_rps=float(args.return_joint_speed_rps),
            min_seconds=float(args.return_min_seconds),
            backend=str(args.nfg_backend),
            num_samples=int(args.nfg_num_samples),
            max_iterations=int(args.nfg_max_iterations),
            timeout_seconds=float(args.nfg_timeout_seconds),
        )
    except Exception as exc:
        raise NFGReturnPlanningError(
            "collision-free return was not proven; xArm remains at the placement pose"
        ) from exc
    np.savez_compressed(
        Path(capture_root) / "nfg_collision_free_return.npz",
        arm_qpos=plan.positions,
        timestamps_s=plan.times_s,
        obstacle_name=np.asarray([item.name for item in obstacles]),
        obstacle_center_m=np.asarray([item.center_m for item in obstacles]),
        obstacle_radius_m=np.asarray([item.radius_m for item in obstacles]),
        solve_time_s=plan.solve_time_s,
        iterations=plan.iterations,
        backend=plan.backend,
        min_contact_distance_m=plan.min_contact_distance_m,
    )
    print(
        f"[NFG] collision-free return: {len(plan.positions)} frames, "
        f"{plan.solve_time_s:.2f}s solve, backend={plan.backend}, "
        f"obstacles={len(obstacles)}"
    )
    arm.move_joint_trajectory(plan.positions, plan.times_s)
    if arm.is_error():
        raise NFGReturnPlanningError(
            "xArm controller error during NFG collision-free return"
        )


def execute_slow_ready_return(
    arm: Any,
    args: argparse.Namespace,
    capture_root: Path,
) -> None:
    """Move from the current joints to the fixed sorting ready pose."""
    current_qpos = np.asarray(arm.get_data().get("qpos"), dtype=np.float64)
    positions, times_s, duration_s = base.replay_core._joint_return_trajectory(
        current_qpos,
        READY_ARM_QPOS_RAD,
        speed_rps=float(args.return_joint_speed_rps),
        min_seconds=float(args.return_min_seconds),
        rate_hz=float(args.return_rate_hz),
    )
    np.savez_compressed(
        Path(capture_root) / "ready_joint_return.npz",
        arm_qpos=positions,
        timestamps_s=times_s,
        target_qpos_deg=READY_ARM_QPOS_DEG,
    )
    print(
        "[execute] slowly returning xArm to ready joint pose "
        f"{READY_ARM_QPOS_DEG.tolist()}: {len(positions)} frames over "
        f"{duration_s:.2f}s"
    )
    arm.move_joint_trajectory(positions, times_s)
    if arm.is_error():
        raise RuntimeError("xArm controller error during return to ready joint pose")


def execute_post_placement_retreat(
    arm: Any,
    args: argparse.Namespace,
    capture_root: Path,
) -> None:
    """Retract vertically in robot coordinates before the joint-space return."""
    current_pose = np.asarray(arm.get_data().get("position"), dtype=np.float64)
    if current_pose.shape != (4, 4) or not np.all(np.isfinite(current_pose)):
        raise RuntimeError(f"invalid current xArm pose: {current_pose}")
    target_pose = current_pose.copy()
    target_pose[2, 3] = float(args.post_placement_retreat_z_m)
    validate_arm_workspace(
        target_pose[None],
        min_z_m=float(args.arm_min_z_m),
        max_z_m=float(args.arm_max_z_m),
        min_radius_m=float(args.arm_min_radius_m),
        max_radius_m=float(args.arm_max_radius_m),
    )
    steps = required_pose_steps(
        current_pose,
        target_pose,
        minimum_steps=max(
            1,
            round(
                float(args.post_placement_retreat_seconds)
                * float(args.motion_control_hz)
            ),
        ),
        max_translation_step_m=float(args.arm_max_step_m),
        max_angle_step_deg=float(args.arm_max_angle_step_deg),
    )
    poses = interpolate_pose_segment(current_pose, target_pose, steps)
    times_s = np.arange(1, steps + 1, dtype=np.float64) / float(
        args.motion_control_hz
    )
    np.savez_compressed(
        Path(capture_root) / "post_placement_vertical_retreat.npz",
        arm_pose=poses,
        timestamps_s=times_s,
        target_z_m=float(args.post_placement_retreat_z_m),
    )
    print(
        "[execute] moving xArm vertically to absolute robot-frame "
        f"z={args.post_placement_retreat_z_m:.3f} m: "
        f"{steps} frames over {times_s[-1]:.2f}s"
    )
    started = time.perf_counter()
    for index, (pose, target_time) in enumerate(zip(poses, times_s)):
        arm.move(pose)
        if arm.is_error():
            raise RuntimeError(
                f"xArm controller error during vertical retreat frame {index}"
            )
        time.sleep(max(0.0, started + float(target_time) - time.perf_counter()))


def extract_grasp_trajectory(response: Mapping[str, Any]) -> dict[str, Any]:
    value = response.get("grasp_trajectory")
    if not isinstance(value, Mapping):
        raise ValueError("server response is missing grasp_trajectory")
    if value.get("status") != "success":
        raise ValueError(
            f"DemoGrasp trajectory failed: {value.get('error', value.get('status'))}"
        )
    wrist = np.asarray(value.get("wrist_pose_command_robot"), dtype=np.float64)
    hand = np.asarray(value.get("hand_qpos_command"), dtype=np.float64)
    timestamps = np.asarray(value.get("timestamps_s"), dtype=np.float64)
    if wrist.ndim != 2 or wrist.shape[1] != 7:
        raise ValueError(f"DemoGrasp wrist trajectory must be [T,7], got {wrist.shape}")
    if hand.shape != (len(wrist), 6):
        raise ValueError(f"DemoGrasp hand trajectory must be [T,6], got {hand.shape}")
    if timestamps.shape != (len(wrist),):
        raise ValueError(
            f"DemoGrasp timestamps must be [T], got {timestamps.shape}"
        )
    if not all(np.all(np.isfinite(item)) for item in (wrist, hand, timestamps)):
        raise ValueError("DemoGrasp trajectory contains non-finite values")
    return {
        "wrist": wrist,
        "hand": hand,
        "timestamps": timestamps,
        "active_hand_dof_names": list(value.get("active_hand_dof_names", [])),
        "control_hz": float(value.get("control_hz", 60.0)),
    }


def offset_demograsp_wrist_z(
    trajectory: Mapping[str, Any], z_offset_m: float
) -> dict[str, Any]:
    """Return a DemoGrasp trajectory with a constant robot-frame wrist Z offset."""
    offset = float(z_offset_m)
    if not np.isfinite(offset):
        raise ValueError("DemoGrasp z offset must be finite")
    adjusted = dict(trajectory)
    adjusted_wrist = np.asarray(trajectory["wrist"], dtype=np.float64).copy()
    if adjusted_wrist.ndim != 2 or adjusted_wrist.shape[1] != 7:
        raise ValueError(
            f"DemoGrasp wrist trajectory must be [T,7], got {adjusted_wrist.shape}"
        )
    adjusted_wrist[:, 2] += offset
    adjusted["wrist"] = adjusted_wrist
    return adjusted


def hand_qpos_to_dftp_raw(
    hand_qpos: np.ndarray,
    joint_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Invert the repository's calibrated DFTP raw-to-qpos mapping.

    Input qpos is the DemoGrasp/URDF order. Output is the USB driver's raw
    motor order [little, ring, middle, index, thumb2, thumb1], range 0..1000.
    """
    from paradex.robot.inspire import inspire_action_to_qpos

    names = tuple(str(name) for name in joint_names)
    if names != DEMOGRASP_HAND_JOINTS:
        raise ValueError(
            "DemoGrasp hand joint order mismatch: "
            f"expected={DEMOGRASP_HAND_JOINTS}, actual={names}"
        )
    qpos = np.asarray(hand_qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 6 or not np.all(np.isfinite(qpos)):
        raise ValueError(f"hand qpos must be finite [T,6], got {qpos.shape}")

    raw_axis = np.arange(1001, dtype=np.float64)
    calibration_actions = np.repeat(raw_axis[:, None], 6, axis=1)
    calibrated_qpos = inspire_action_to_qpos(calibration_actions)
    raw = np.empty_like(qpos)
    raw_channel_by_qpos = (5, 4, 3, 2, 1, 0)
    for qpos_id, raw_id in enumerate(raw_channel_by_qpos):
        curve = calibrated_qpos[:, qpos_id]
        order = np.argsort(curve)
        raw[:, raw_id] = np.interp(qpos[:, qpos_id], curve[order], raw_axis[order])
    return np.clip(raw, 0.0, 1000.0)


def resample_hand_qpos(
    hand_qpos: np.ndarray,
    timestamps_s: np.ndarray,
    control_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a timestamped hand trajectory without changing its duration."""
    qpos = np.asarray(hand_qpos, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if qpos.ndim != 2 or qpos.shape != (len(timestamps), 6) or len(qpos) == 0:
        raise ValueError("hand qpos/timestamps must be non-empty [T,6] and [T]")
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(timestamps)):
        raise ValueError("hand qpos/timestamps contain non-finite values")
    timestamps = timestamps - timestamps[0]
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("hand timestamps must be strictly increasing")
    period = 1.0 / float(control_hz)
    duration = float(timestamps[-1])
    target_times = np.arange(0.0, duration + period * 0.25, period)
    if len(target_times) == 0:
        target_times = np.array([0.0])
    if target_times[-1] < duration - 1e-6:
        target_times = np.append(target_times, duration)
    target_times = np.minimum(target_times, duration)
    target_times = np.unique(target_times)
    resampled = np.column_stack(
        [np.interp(target_times, timestamps, qpos[:, joint]) for joint in range(6)]
    )
    return resampled, target_times


def stretch_timestamps_to_duration(
    timestamps_s: np.ndarray,
    target_duration_s: float,
) -> tuple[np.ndarray, float]:
    """Normalize timestamps and stretch them to an exact target duration."""
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if len(timestamps) < 2 or not np.all(np.isfinite(timestamps)):
        raise ValueError("DemoGrasp timestamps must contain at least two finite values")
    normalized = timestamps - timestamps[0]
    if np.any(np.diff(normalized) <= 0) or normalized[-1] <= 0:
        raise ValueError("DemoGrasp timestamps must be strictly increasing")
    if target_duration_s <= 0 or not np.isfinite(target_duration_s):
        raise ValueError("target DemoGrasp duration must be positive and finite")
    scale = float(target_duration_s) / float(normalized[-1])
    return normalized * scale, scale


def limit_raw_command_step(
    raw_commands: np.ndarray,
    max_raw_step: float,
) -> np.ndarray:
    """Insert linear commands so every motor step respects the raw limit."""
    commands = np.asarray(raw_commands, dtype=np.float64)
    if commands.ndim != 2 or commands.shape[1] != 6 or len(commands) == 0:
        raise ValueError(f"raw commands must be non-empty [T,6], got {commands.shape}")
    if not np.all(np.isfinite(commands)) or max_raw_step <= 0:
        raise ValueError("raw commands and max_raw_step must be finite/positive")
    limited = [commands[0].copy()]
    for target in commands[1:]:
        previous = limited[-1]
        substeps = max(
            1,
            int(np.ceil(np.max(np.abs(target - previous)) / float(max_raw_step))),
        )
        for substep in range(1, substeps + 1):
            limited.append(previous + (target - previous) * (substep / substeps))
    return np.asarray(limited, dtype=np.float64)


def verify_inspire_route(ip: str, interface: str) -> str:
    """Require the right-hand address to route through the requested NIC."""
    result = subprocess.run(
        ["ip", "route", "get", ip],
        check=True,
        capture_output=True,
        text=True,
    )
    route = result.stdout.strip()
    if f"dev {interface}" not in route:
        raise RuntimeError(
            f"{ip} routes through a different NIC: {route}; "
            f"expected dev {interface}"
        )
    return route


def _modbus_registers(response: Any, expected_count: int) -> list[int]:
    if response is None or response.isError():
        raise RuntimeError(f"Modbus read failed: {response}")
    registers = getattr(response, "registers", None)
    if registers is None or len(registers) != expected_count:
        raise RuntimeError(f"invalid Modbus response: {registers}")
    return [int(value) for value in registers]


class RightInspireDFTP:
    """Minimal Modbus TCP adapter matching inspire_dftp_smoke_test.py."""

    def __init__(self, ip: str, port: int, timeout: float, interface: str):
        from pymodbus.client.sync import ModbusTcpClient

        verify_inspire_route(ip, interface)
        self.ip = ip
        self.port = int(port)
        self.interface = interface
        self.client = ModbusTcpClient(ip, port=self.port, timeout=timeout)
        interface_index = socket.if_nametoindex(interface)
        bound_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bound_socket.settimeout(timeout)
        try:
            bound_socket.setsockopt(
                socket.IPPROTO_IP,
                IP_UNICAST_IF,
                struct.pack("!I", interface_index),
            )
            bound_socket.connect((ip, self.port))
        except Exception:
            bound_socket.close()
            raise
        self.client.socket = bound_socket

        # Do not write anything until the hand has returned a valid state.
        initial = self.read_angles()
        if any(value < 0 or value > 1000 for value in initial):
            self.close()
            raise RuntimeError(
                f"right Inspire angles outside command scale 0..1000: {initial}"
            )
        print(
            f"[OK] right Inspire DFTP responded on {ip}:{self.port} "
            f"via {interface}; actual_angles={initial}"
        )

    def read_angles(self) -> list[int]:
        angles = _modbus_registers(
            self.client.read_holding_registers(
                DFTP_ANGLE_ACT, DFTP_JOINT_COUNT
            ),
            DFTP_JOINT_COUNT,
        )
        if any(value < 0 or value > 2000 for value in angles):
            raise RuntimeError(f"angle response outside expected 0..2000: {angles}")
        return angles

    def get_data(self) -> dict[str, list[int]]:
        return {"joint_value": self.read_angles()}

    def move(self, command: np.ndarray) -> None:
        values = np.rint(np.asarray(command, dtype=np.float64)).astype(np.int64)
        if values.shape != (DFTP_JOINT_COUNT,) or np.any(values < 0) or np.any(values > 1000):
            raise RuntimeError(f"invalid right Inspire command: {values.tolist()}")
        response = self.client.write_registers(
            DFTP_ANGLE_SET, [int(value) for value in values]
        )
        if response is None or response.isError():
            raise RuntimeError(f"Modbus write failed: {response}")

    def close(self) -> None:
        self.client.close()


def execute_inspire_hand_only(
    hand: Any,
    grasp_trajectory: Mapping[str, Any],
    capture_root: Path,
    *,
    control_hz: float = 10.0,
    bridge_seconds: float = 1.0,
    max_raw_step: float = 50.0,
) -> None:
    """Stream only the DemoGrasp finger commands to the Inspire DFTP hand."""
    qpos_control, command_times = resample_hand_qpos(
        grasp_trajectory["hand"], grasp_trajectory["timestamps"], control_hz
    )
    raw_commands = hand_qpos_to_dftp_raw(
        qpos_control, grasp_trajectory["active_hand_dof_names"]
    )
    original_command_count = len(raw_commands)
    raw_commands = limit_raw_command_step(raw_commands, max_raw_step)
    command_times = np.arange(len(raw_commands), dtype=np.float64) / float(control_hz)
    current_raw = np.asarray(hand.get_data().get("joint_value"), dtype=np.float64)
    if current_raw.shape != (6,) or not np.all(np.isfinite(current_raw)):
        raise RuntimeError(f"Invalid Inspire DFTP current state: {current_raw}")
    current_raw = np.clip(current_raw, 0.0, 1000.0)

    period = 1.0 / float(control_hz)

    def stream(commands: np.ndarray) -> None:
        deadline = time.perf_counter()
        for command in commands:
            hand.move(np.asarray(command, dtype=np.float64))
            deadline += period
            time.sleep(max(0.0, deadline - time.perf_counter()))

    requested_bridge_frames = int(round(float(bridge_seconds) * float(control_hz)))
    required_bridge_frames = int(
        np.ceil(np.max(np.abs(raw_commands[0] - current_raw)) / float(max_raw_step))
    )
    bridge_frames = max(requested_bridge_frames, required_bridge_frames)
    if bridge_frames > 0:
        bridge = np.linspace(
            current_raw, raw_commands[0], bridge_frames + 1, dtype=np.float64
        )[1:]
        print(
            f"[execute] Inspire DFTP bridge: {bridge_frames} frames at "
            f"{control_hz:.0f} Hz"
        )
        stream(bridge)

    max_step = (
        float(np.max(np.abs(np.diff(raw_commands, axis=0))))
        if len(raw_commands) > 1
        else 0.0
    )
    print(
        f"[execute] Inspire DFTP hand-only: {len(raw_commands)} frames at "
        f"{control_hz:.0f} Hz ({original_command_count} source commands), "
        f"duration={command_times[-1]:.3f}s, max_raw_step={max_step:.1f}"
    )
    stream(raw_commands)
    rate_tag = f"{control_hz:g}".replace(".", "p")
    np.save(capture_root / f"demograsp_hand_raw_{rate_tag}hz.npy", raw_commands)
    np.save(
        capture_root / f"demograsp_hand_timestamps_{rate_tag}hz.npy",
        command_times,
    )
    final_feedback = np.asarray(hand.get_data().get("joint_value"), dtype=np.float64)
    if final_feedback.shape != (6,) or not np.all(np.isfinite(final_feedback)):
        raise RuntimeError(f"Invalid final Inspire DFTP feedback: {final_feedback}")
    print(
        "[execute] Inspire DFTP hand trajectory complete; final command is held; "
        f"final_feedback={np.rint(final_feedback).astype(int).tolist()}"
    )


def wrist_rows_to_matrices(wrist: np.ndarray) -> np.ndarray:
    rows = np.asarray(wrist, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 7 or not np.all(np.isfinite(rows)):
        raise ValueError(f"wrist trajectory must be finite [T,7], got {rows.shape}")
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], len(rows), axis=0)
    matrices[:, :3, 3] = rows[:, :3]
    matrices[:, :3, :3] = R.from_quat(rows[:, 3:7]).as_matrix()
    return matrices


def load_link6_from_wrist(robot_urdf: Path = XARM_INSPIRE_URDF) -> np.ndarray:
    """Read T_link6_wrist from the combined xArm+Inspire URDF."""
    root = ET.parse(Path(robot_urdf)).getroot()
    joint = root.find(".//joint[@name='arm_to_hand']")
    if joint is None:
        raise ValueError(f"arm_to_hand joint missing from {robot_urdf}")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError(f"arm_to_hand origin missing from {robot_urdf}")
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    if (
        xyz.shape != (3,)
        or rpy.shape != (3,)
        or not np.all(np.isfinite(xyz))
        or not np.all(np.isfinite(rpy))
    ):
        raise ValueError(f"invalid arm_to_hand origin in {robot_urdf}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = xyz
    transform[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    return transform


def wrist_poses_to_link6(
    wrist_poses: np.ndarray,
    link6_from_wrist: np.ndarray,
) -> np.ndarray:
    """Convert T_robot_wrist targets into xArm T_robot_link6 targets."""
    poses = np.asarray(wrist_poses, dtype=np.float64)
    fixed = np.asarray(link6_from_wrist, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"wrist poses must be [T,4,4], got {poses.shape}")
    if fixed.shape != (4, 4) or not np.all(np.isfinite(fixed)):
        raise ValueError("link6_from_wrist must be a finite 4x4 transform")
    return poses @ np.linalg.inv(fixed)


def interpolate_pose_segment(
    start: np.ndarray,
    target: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Return `steps` SE(3) samples after start, including target."""
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if start.shape != (4, 4) or target.shape != (4, 4) or steps < 1:
        raise ValueError("pose interpolation needs two 4x4 poses and steps >= 1")
    fractions = np.linspace(0.0, 1.0, int(steps) + 1)[1:]
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(fractions), axis=0)
    result[:, :3, 3] = (
        start[:3, 3][None]
        + fractions[:, None] * (target[:3, 3] - start[:3, 3])[None]
    )
    key_rots = R.from_matrix(np.stack((start[:3, :3], target[:3, :3])))
    result[:, :3, :3] = Slerp([0.0, 1.0], key_rots)(fractions).as_matrix()
    return result


def required_pose_steps(
    start: np.ndarray,
    target: np.ndarray,
    *,
    minimum_steps: int,
    max_translation_step_m: float,
    max_angle_step_deg: float,
) -> int:
    distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    angle_deg = float(
        np.rad2deg(
            (R.from_matrix(start[:3, :3]).inv() * R.from_matrix(target[:3, :3])).magnitude()
        )
    )
    return max(
        int(minimum_steps),
        int(np.ceil(distance / max_translation_step_m)),
        int(np.ceil(angle_deg / max_angle_step_deg)),
        1,
    )


def resample_wrist_poses(
    wrist: np.ndarray,
    timestamps_s: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(wrist, dtype=np.float64)
    source_times = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    source_times = source_times - source_times[0]
    positions = np.column_stack(
        [np.interp(target_times, source_times, rows[:, axis]) for axis in range(3)]
    )
    if len(rows) == 1:
        rotations = np.repeat(
            R.from_quat(rows[0, 3:7]).as_matrix()[None], len(target_times), axis=0
        )
    else:
        rotations = Slerp(source_times, R.from_quat(rows[:, 3:7]))(
            target_times
        ).as_matrix()
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(target_times), axis=0)
    poses[:, :3, 3] = positions
    poses[:, :3, :3] = rotations
    return poses


def validate_arm_workspace(
    poses: np.ndarray,
    *,
    min_z_m: float,
    max_z_m: float,
    min_radius_m: float,
    max_radius_m: float,
) -> None:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or not np.all(np.isfinite(values)):
        raise ValueError(f"xArm poses must be finite [T,4,4], got {values.shape}")
    z = values[:, 2, 3]
    radius = np.linalg.norm(values[:, :2, 3], axis=1)
    if np.any(z < min_z_m) or np.any(z > max_z_m):
        raise RuntimeError(
            f"refusing xArm trajectory outside z=[{min_z_m}, {max_z_m}] m: "
            f"observed [{z.min():.3f}, {z.max():.3f}]"
        )
    if np.any(radius < min_radius_m) or np.any(radius > max_radius_m):
        raise RuntimeError(
            f"refusing xArm trajectory outside radial reach "
            f"[{min_radius_m}, {max_radius_m}] m: "
            f"observed [{radius.min():.3f}, {radius.max():.3f}]"
        )


def execute_xarm_and_inspire(
    arm: Any,
    hand: Any,
    grasp_trajectory: Mapping[str, Any],
    placement_xy_robot: np.ndarray,
    placement_target_name: str,
    capture_root: Path,
    *,
    replay_hz: float,
    replay_duration_seconds: float,
    motion_hz: float,
    xy_seconds: float,
    frame0_seconds: float,
    hold_seconds: float,
    post_grasp_z_m: float,
    post_grasp_lift_seconds: float,
    post_grasp_transfer_seconds: float,
    cutting_board_release_z_m: float,
    cutting_board_descent_seconds: float,
    max_raw_step: float,
    max_translation_step_m: float,
    max_angle_step_deg: float,
    min_z_m: float,
    max_z_m: float,
    min_radius_m: float,
    max_radius_m: float,
) -> None:
    """Execute grasp, absolute-z lift, pink-box XY transfer, and hand restore."""
    placement_xy = np.asarray(placement_xy_robot, dtype=np.float64).reshape(-1)
    if placement_xy.shape != (2,) or not np.all(np.isfinite(placement_xy)):
        raise ValueError(f"placement target XY must be finite [2], got {placement_xy}")
    source_timestamps, replay_duration_scale = stretch_timestamps_to_duration(
        grasp_trajectory["timestamps"], replay_duration_seconds
    )
    hand_qpos, target_times = resample_hand_qpos(
        grasp_trajectory["hand"], source_timestamps, replay_hz
    )
    raw = hand_qpos_to_dftp_raw(
        hand_qpos, grasp_trajectory["active_hand_dof_names"]
    )
    wrist_poses = resample_wrist_poses(
        grasp_trajectory["wrist"], source_timestamps, target_times
    )
    print(
        f"[execute] post-interpolated DemoGrasp: "
        f"{len(grasp_trajectory['wrist'])} source frames -> {len(wrist_poses)} "
        f"frames at {replay_hz:g} Hz, duration={target_times[-1]:.3f}s "
        f"({replay_duration_scale:g}x time stretch)"
    )
    link6_from_wrist = load_link6_from_wrist()
    arm_poses = wrist_poses_to_link6(wrist_poses, link6_from_wrist)
    mount_xyz = link6_from_wrist[:3, 3]
    mount_rpy_deg = np.rad2deg(
        R.from_matrix(link6_from_wrist[:3, :3]).as_euler("xyz")
    )
    print(
        "[execute] converting DemoGrasp wrist targets to xArm link6 using "
        f"arm_to_hand xyz={np.round(mount_xyz, 6).tolist()} m, "
        f"rpy={np.round(mount_rpy_deg, 3).tolist()} deg"
    )
    validate_arm_workspace(
        arm_poses,
        min_z_m=min_z_m,
        max_z_m=max_z_m,
        min_radius_m=min_radius_m,
        max_radius_m=max_radius_m,
    )

    current_raw = np.asarray(hand.get_data().get("joint_value"), dtype=np.float64)
    arm_state = arm.get_data()
    current_pose = np.asarray(arm_state.get("position"), dtype=np.float64)
    if current_raw.shape != (6,) or not np.all(np.isfinite(current_raw)):
        raise RuntimeError(f"invalid current Inspire state: {current_raw}")
    if current_pose.shape != (4, 4) or not np.all(np.isfinite(current_pose)):
        raise RuntimeError(f"invalid current xArm pose: {current_pose}")

    def wait_until(deadline: float, rate_hz: float) -> float:
        deadline += 1.0 / float(rate_hz)
        time.sleep(max(0.0, deadline - time.perf_counter()))
        return deadline

    def check_arm(stage: str, index: int) -> None:
        if arm.is_error():
            raise RuntimeError(f"xArm controller error during {stage} frame {index}")

    def stream_hand(commands: np.ndarray, stage: str, rate_hz: float) -> None:
        deadline = time.perf_counter()
        for index, command in enumerate(commands):
            hand.move(command)
            deadline = wait_until(deadline, rate_hz)

    def stream_arm(poses: np.ndarray, stage: str, rate_hz: float) -> None:
        deadline = time.perf_counter()
        for index, pose in enumerate(poses):
            arm.move(pose)
            check_arm(stage, index)
            deadline = wait_until(deadline, rate_hz)

    def stream_both(
        poses: np.ndarray,
        commands: np.ndarray,
        stage: str,
        rate_hz: float,
    ) -> None:
        if len(poses) != len(commands):
            raise ValueError(f"{stage} arm/hand lengths differ")
        deadline = time.perf_counter()
        for index, (pose, command) in enumerate(zip(poses, commands)):
            arm.move(pose)
            hand.move(command)
            check_arm(stage, index)
            deadline = wait_until(deadline, rate_hz)

    def hand_bridge(start: np.ndarray, target: np.ndarray, minimum_steps: int) -> np.ndarray:
        steps = max(
            int(minimum_steps),
            int(np.ceil(np.max(np.abs(target - start)) / max_raw_step)),
            1,
        )
        return np.linspace(start, target, steps + 1, dtype=np.float64)[1:]

    open_bridge = hand_bridge(current_raw, DFTP_OPEN_RAW, round(motion_hz))
    print(f"[execute] opening right hand: {len(open_bridge)} frames at {motion_hz:g} Hz")
    stream_hand(open_bridge, "initial hand open", motion_hz)

    xy_target = current_pose.copy()
    xy_target[0, 3] = arm_poses[0, 0, 3]
    xy_target[1, 3] = arm_poses[0, 1, 3]
    validate_arm_workspace(
        xy_target[None],
        min_z_m=min_z_m,
        max_z_m=max_z_m,
        min_radius_m=min_radius_m,
        max_radius_m=max_radius_m,
    )
    xy_steps = required_pose_steps(
        current_pose,
        xy_target,
        minimum_steps=round(xy_seconds * motion_hz),
        max_translation_step_m=max_translation_step_m,
        max_angle_step_deg=max_angle_step_deg,
    )
    xy_poses = interpolate_pose_segment(current_pose, xy_target, xy_steps)
    print(f"[execute] slow XY-only xArm approach: {xy_steps} frames ({xy_steps/motion_hz:.2f}s) at {motion_hz:g} Hz")
    stream_arm(xy_poses, "XY approach", motion_hz)

    frame0_pose_steps = required_pose_steps(
        xy_target,
        arm_poses[0],
        minimum_steps=round(frame0_seconds * motion_hz),
        max_translation_step_m=max_translation_step_m,
        max_angle_step_deg=max_angle_step_deg,
    )
    frame0_hand_steps = int(
        np.ceil(np.max(np.abs(raw[0] - DFTP_OPEN_RAW)) / max_raw_step)
    )
    frame0_steps = max(frame0_pose_steps, frame0_hand_steps, 1)
    frame0_poses = interpolate_pose_segment(xy_target, arm_poses[0], frame0_steps)
    frame0_hand = np.linspace(
        DFTP_OPEN_RAW, raw[0], frame0_steps + 1, dtype=np.float64
    )[1:]
    print(
        f"[execute] synchronized approach to DemoGrasp frame 0: "
        f"{frame0_steps} frames ({frame0_steps/motion_hz:.2f}s) at {motion_hz:g} Hz"
    )
    stream_both(frame0_poses, frame0_hand, "frame-0 approach", motion_hz)

    synchronized_poses = [arm_poses[0]]
    synchronized_raw = [raw[0]]
    for target_pose, target_raw in zip(arm_poses[1:], raw[1:]):
        steps = required_pose_steps(
            synchronized_poses[-1],
            target_pose,
            minimum_steps=1,
            max_translation_step_m=max_translation_step_m,
            max_angle_step_deg=max_angle_step_deg,
        )
        steps = max(
            steps,
            int(np.ceil(np.max(np.abs(target_raw - synchronized_raw[-1])) / max_raw_step)),
        )
        pose_segment = interpolate_pose_segment(synchronized_poses[-1], target_pose, steps)
        raw_segment = np.linspace(
            synchronized_raw[-1], target_raw, steps + 1, dtype=np.float64
        )[1:]
        synchronized_poses.extend(pose_segment)
        synchronized_raw.extend(raw_segment)
    synchronized_poses_array = np.asarray(synchronized_poses)
    synchronized_raw_array = np.asarray(synchronized_raw)
    print(
        f"[execute] synchronized DemoGrasp replay: {len(synchronized_poses_array)} "
        f"frames at {replay_hz:g} Hz"
    )
    stream_both(
        synchronized_poses_array,
        synchronized_raw_array,
        "DemoGrasp replay",
        replay_hz,
    )

    final_grasp_raw = synchronized_raw_array[-1]
    lift_wrist_target = wrist_poses[-1].copy()
    lift_wrist_target[2, 3] = float(post_grasp_z_m)
    transfer_wrist_target = lift_wrist_target.copy()
    transfer_wrist_target[0, 3] = placement_xy[0]
    transfer_wrist_target[1, 3] = placement_xy[1]
    post_grasp_target_list = [lift_wrist_target, transfer_wrist_target]
    if placement_target_name == "cutting_board":
        release_wrist_target = transfer_wrist_target.copy()
        release_wrist_target[2, 3] = float(cutting_board_release_z_m)
        post_grasp_target_list.append(release_wrist_target)
    placement_release_z_m = (
        float(cutting_board_release_z_m)
        if placement_target_name == "cutting_board"
        else float(post_grasp_z_m)
    )
    post_grasp_wrist_targets = np.stack(post_grasp_target_list)
    post_grasp_link6_targets = wrist_poses_to_link6(
        post_grasp_wrist_targets, link6_from_wrist
    )
    # Inspire episode replay applies the resolved placement coordinates
    # directly to the xArm/link6 trajectory.  Do the same for DemoGrasp so the
    # calibrated board/box XY is identical across both grasp strategies; if XY
    # is assigned in wrist space before the mount transform, link6 ends at an
    # offset position.
    post_grasp_link6_targets[1:, 0, 3] = placement_xy[0]
    post_grasp_link6_targets[1:, 1, 3] = placement_xy[1]
    validate_arm_workspace(
        post_grasp_link6_targets,
        min_z_m=min_z_m,
        max_z_m=max_z_m,
        min_radius_m=min_radius_m,
        max_radius_m=max_radius_m,
    )

    lift_steps = required_pose_steps(
        synchronized_poses_array[-1],
        post_grasp_link6_targets[0],
        minimum_steps=round(post_grasp_lift_seconds * motion_hz),
        max_translation_step_m=max_translation_step_m,
        max_angle_step_deg=max_angle_step_deg,
    )
    lift_link6_poses = interpolate_pose_segment(
        synchronized_poses_array[-1], post_grasp_link6_targets[0], lift_steps
    )
    print(
        f"[execute] post-grasp move to absolute wrist z={post_grasp_z_m:.3f} m: "
        f"{lift_steps} frames ({lift_steps/motion_hz:.2f}s) at {motion_hz:g} Hz"
    )
    stream_both(
        lift_link6_poses,
        np.repeat(final_grasp_raw[None], lift_steps, axis=0),
        "post-grasp absolute-z move",
        motion_hz,
    )

    transfer_steps = required_pose_steps(
        post_grasp_link6_targets[0],
        post_grasp_link6_targets[1],
        minimum_steps=round(post_grasp_transfer_seconds * motion_hz),
        max_translation_step_m=max_translation_step_m,
        max_angle_step_deg=max_angle_step_deg,
    )
    transfer_link6_poses = interpolate_pose_segment(
        post_grasp_link6_targets[0],
        post_grasp_link6_targets[1],
        transfer_steps,
    )
    print(
        f"[execute] post-grasp XY transfer to {placement_target_name} target "
        f"{np.round(placement_xy, 4).tolist()}: {transfer_steps} frames "
        f"({transfer_steps/motion_hz:.2f}s) at {motion_hz:g} Hz, wrist z held at "
        f"{post_grasp_z_m:.3f} m"
    )
    stream_both(
        transfer_link6_poses,
        np.repeat(final_grasp_raw[None], transfer_steps, axis=0),
        f"{placement_target_name} XY transfer",
        motion_hz,
    )

    descent_link6_poses = np.empty((0, 4, 4), dtype=np.float64)
    descent_steps = 0
    if placement_target_name == "cutting_board":
        descent_steps = required_pose_steps(
            post_grasp_link6_targets[1],
            post_grasp_link6_targets[2],
            minimum_steps=round(cutting_board_descent_seconds * motion_hz),
            max_translation_step_m=max_translation_step_m,
            max_angle_step_deg=max_angle_step_deg,
        )
        descent_link6_poses = interpolate_pose_segment(
            post_grasp_link6_targets[1],
            post_grasp_link6_targets[2],
            descent_steps,
        )
        print(
            "[execute] cutting-board descent to absolute wrist "
            f"z={cutting_board_release_z_m:.3f} m: {descent_steps} frames "
            f"({descent_steps/motion_hz:.2f}s) at {motion_hz:g} Hz"
        )
        stream_both(
            descent_link6_poses,
            np.repeat(final_grasp_raw[None], descent_steps, axis=0),
            "cutting-board descent",
            motion_hz,
        )

    if hold_seconds > 0:
        print(
            f"[execute] holding grasp over {placement_target_name} "
            f"for {hold_seconds:g}s"
        )
        time.sleep(hold_seconds)

    restore = hand_bridge(synchronized_raw_array[-1], raw[0], round(motion_hz))
    print(
        f"[execute] restoring hand to DemoGrasp frame 0: "
        f"{len(restore)} frames at {motion_hz:g} Hz"
    )
    stream_hand(restore, "frame-0 hand restore", motion_hz)
    final_feedback = hand.get_data().get("joint_value")
    executed_arm_poses = np.concatenate(
        (
            synchronized_poses_array,
            lift_link6_poses,
            transfer_link6_poses,
            descent_link6_poses,
        ),
        axis=0,
    )
    executed_hand_raw = np.concatenate(
        (
            synchronized_raw_array,
            np.repeat(
                final_grasp_raw[None],
                lift_steps + transfer_steps + descent_steps,
                axis=0,
            ),
        ),
        axis=0,
    )
    replay_tag = f"{replay_hz:g}".replace(".", "p")
    replay_timestamps = np.arange(
        len(synchronized_raw_array), dtype=np.float64
    ) / replay_hz
    post_count = lift_steps + transfer_steps + descent_steps
    post_timestamps = (
        replay_timestamps[-1]
        + 1.0 / motion_hz
        + np.arange(post_count, dtype=np.float64) / motion_hz
    )
    executed_timestamps = np.concatenate((replay_timestamps, post_timestamps))
    np.save(
        capture_root / "demograsp_executed_arm_pose.npy",
        executed_arm_poses,
    )
    np.save(
        capture_root / f"demograsp_resampled_wrist_pose_{replay_tag}hz.npy",
        wrist_poses,
    )
    np.save(
        capture_root / "demograsp_executed_hand_raw.npy",
        executed_hand_raw,
    )
    np.save(
        capture_root / "demograsp_executed_timestamps_s.npy",
        executed_timestamps,
    )
    np.save(
        capture_root / "demograsp_postgrasp_wrist_targets.npy",
        post_grasp_wrist_targets,
    )
    (capture_root / "demograsp_execution_rates.json").write_text(
        json.dumps(
            {
                "demograsp_replay_hz": float(replay_hz),
                "demograsp_duration_seconds": float(replay_duration_seconds),
                "demograsp_duration_scale": float(replay_duration_scale),
                "approach_transfer_restore_hz": float(motion_hz),
                "placement_target_name": str(placement_target_name),
                "placement_target_xy_robot": placement_xy.tolist(),
                "placement_wrist_z_robot_m": placement_release_z_m,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        "[execute] xArm + right Inspire sequence complete; xArm remains over "
        f"the {placement_target_name} target at absolute wrist z="
        f"{placement_release_z_m:.3f} "
        "m and hand is restored to frame 0; "
        f"feedback={final_feedback}"
    )


def confirm_hand_execution() -> bool:
    """Require an explicit Y/N decision after the Viser preview closes."""
    while True:
        try:
            answer = input(
                "[execute] Run the xArm + RIGHT Inspire trajectory? [Y/N] > "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[execute] cancelled; returning to idle.")
            return False
        if answer == "y":
            return True
        if answer == "n":
            print("[execute] cancelled; returning to idle.")
            return False
        print("[execute] Please enter Y or N.")


def confirm_ready_return() -> bool:
    """Ask whether to move the xArm back to the fixed sorting ready pose."""
    while True:
        try:
            answer = input(
                "[execute] Return xArm to the ready joint pose? [Y/N] > "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[execute] return skipped; returning to idle.")
            return False
        if answer == "y":
            return True
        if answer == "n":
            print("[execute] return skipped; returning to idle.")
            return False
        print("[execute] Please enter Y or N.")


def stop_viser_safely(server: Any) -> None:
    """Stop Viser while tolerating its known client-disconnect race."""
    stop = getattr(server, "stop", None)
    if not callable(stop):
        return
    try:
        stop()
    except RuntimeError as exc:
        if "dictionary changed size during iteration" not in str(exc):
            raise
        print(
            "[viser] ignored client-disconnect race during shutdown; "
            "continuing to execution confirmation."
        )


def object_replay_args(args: argparse.Namespace, object_name: str) -> argparse.Namespace:
    configured = copy.copy(args)
    configured.mesh_name = object_name
    configured.object = object_name
    configured.episode_root = None
    configured.candidate_episodes = None
    base.configure_episode_selection(configured)
    return configured


def preload_sorting_episodes(
    args: argparse.Namespace,
) -> dict[str, list[base.replay_core.Episode]]:
    cache: dict[str, list[base.replay_core.Episode]] = {}
    for object_name in SUPPORTED_OBJECTS:
        configured = object_replay_args(args, object_name)
        print(f"[preload] preparing {object_name} episodes.")
        cache[object_name] = base._preload_replay_episodes(configured)
    return cache


def preload_inspire_episodes(
    args: argparse.Namespace,
) -> dict[str, list[base.replay_core.Episode]]:
    """Load valid Inspire DFTP episodes before enabling the idle prompt."""

    cache: dict[str, list[base.replay_core.Episode]] = {}
    for object_name in SUPPORTED_OBJECTS:
        object_root = Path(args.inspire_episode_root) / object_name
        discovered_episode_ids = base._discover_candidate_episode_ids(object_root)
        excluded_episode_ids = EXCLUDED_INSPIRE_EPISODES.get(
            object_name, frozenset()
        )
        episode_ids = tuple(
            episode_id
            for episode_id in discovered_episode_ids
            if episode_id not in excluded_episode_ids
        )
        if excluded_episode_ids:
            excluded_text = ", ".join(
                str(episode_id) for episode_id in sorted(excluded_episode_ids)
            )
            print(
                f"[preload] excluding configured Inspire DFTP {object_name} "
                f"episode(s): {excluded_text}"
            )
        print(
            f"[preload] preparing {len(episode_ids)} Inspire DFTP "
            f"{object_name} episode(s)."
        )
        episodes: list[base.replay_core.Episode] = []
        for episode_id in episode_ids:
            try:
                episodes.extend(
                    load_candidate_episodes(
                        object_root,
                        (episode_id,),
                        robot="inspire_dftp",
                    )
                )
            except (FileNotFoundError, ValueError) as exc:
                print(
                    f"[preload] skipping Inspire DFTP {object_name}/"
                    f"{episode_id}: {exc}"
                )
        if not episodes:
            raise RuntimeError(
                f"no valid Inspire DFTP replay episode for {object_name}"
            )
        cache[object_name] = episodes
    return cache


def resample_pose_matrices(
    poses: np.ndarray,
    timestamps: np.ndarray,
    target_timestamps: np.ndarray,
) -> np.ndarray:
    """Interpolate a sampled SE(3) trajectory at absolute timestamps."""

    values = np.asarray(poses, dtype=np.float64)
    source_times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    target_times = np.asarray(target_timestamps, dtype=np.float64).reshape(-1)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"pose trajectory must be [T,4,4], got {values.shape}")
    if len(values) != len(source_times) or len(values) == 0:
        raise ValueError("pose trajectory and timestamps must be non-empty and aligned")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(source_times)):
        raise ValueError("pose trajectory contains non-finite values")
    unique_times, unique_indices = np.unique(source_times, return_index=True)
    values = values[unique_indices]
    if len(unique_times) == 1:
        return np.repeat(values[:1], len(target_times), axis=0)
    if target_times.min() < unique_times[0] or target_times.max() > unique_times[-1]:
        raise ValueError("target timestamps lie outside the pose trajectory")
    positions = np.column_stack(
        [
            np.interp(target_times, unique_times, values[:, axis, 3])
            for axis in range(3)
        ]
    )
    rotations = Slerp(unique_times, R.from_matrix(values[:, :3, :3]))(
        target_times
    ).as_matrix()
    result = np.repeat(np.eye(4, dtype=np.float64)[None], len(target_times), axis=0)
    result[:, :3, :3] = rotations
    result[:, :3, 3] = positions
    return result


def _densify_pose_and_raw(
    poses: np.ndarray,
    raw_commands: np.ndarray,
    *,
    max_raw_step: float,
    max_translation_step_m: float,
    max_angle_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert synchronized samples until every arm/hand step is safe."""

    pose_values = np.asarray(poses, dtype=np.float64)
    raw_values = np.asarray(raw_commands, dtype=np.float64)
    if len(pose_values) != len(raw_values) or len(pose_values) == 0:
        raise ValueError("arm and Inspire trajectories must be non-empty and aligned")
    dense_poses = [pose_values[0]]
    dense_raw = [raw_values[0]]
    for target_pose, target_raw in zip(pose_values[1:], raw_values[1:]):
        steps = required_pose_steps(
            dense_poses[-1],
            target_pose,
            minimum_steps=1,
            max_translation_step_m=max_translation_step_m,
            max_angle_step_deg=max_angle_step_deg,
        )
        steps = max(
            steps,
            int(np.ceil(np.max(np.abs(target_raw - dense_raw[-1])) / max_raw_step)),
        )
        dense_poses.extend(interpolate_pose_segment(dense_poses[-1], target_pose, steps))
        dense_raw.extend(
            np.linspace(dense_raw[-1], target_raw, steps + 1, dtype=np.float64)[1:]
        )
    return np.asarray(dense_poses), np.asarray(dense_raw)


def automatic_eef_apex(episode_root: Path) -> dict[str, Any]:
    """Apply the repository's standard EEF-height apex detector."""

    from src.object6d.annotate_eef_apex import annotate_episode

    return annotate_episode(
        episode_root,
        smoothing_seconds=0.12,
        minimum_prominence_m=0.01,
        minimum_peak_separation_seconds=0.4,
        search_start_fraction=0.15,
        search_end_fraction=0.90,
    )


def load_replay_start_annotations(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    configured = getattr(args, "replay_start_annotations", None)
    path = (
        Path(configured).expanduser()
        if configured is not None
        else Path(args.inspire_episode_root).expanduser()
        / DEFAULT_REPLAY_START_ANNOTATION_NAME
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"replay-start annotation file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read replay-start annotations {path}: {exc}") from exc
    annotations = payload.get("annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"replay-start annotation file has no annotations object: {path}")
    return path, annotations


def replay_start_frame(
    annotations: Mapping[str, Any],
    object_name: str,
    episode: base.replay_core.Episode,
) -> int | None:
    object_annotations = annotations.get(object_name)
    record = (
        object_annotations.get(episode.root.name)
        if isinstance(object_annotations, Mapping)
        else None
    )
    if not isinstance(record, Mapping) or "arm_frame_index" not in record:
        return None
    frame = int(record["arm_frame_index"])
    if frame < 0 or frame >= len(episode.arm_poses):
        raise ValueError(
            f"replay start frame {frame} is outside {object_name}/"
            f"{episode.root.name} arm trajectory 0..{len(episode.arm_poses) - 1}"
        )
    return frame


def build_inspire_episode_plan(
    args: argparse.Namespace,
    *,
    object_name: str,
    current_object_robot: np.ndarray,
    current_c2r: np.ndarray,
    placement_xy_robot: np.ndarray,
    placement_target_name: str,
    episodes: list[base.replay_core.Episode],
) -> dict[str, Any]:
    """Retrieve, object-align, truncate at apex, transfer, and release."""

    annotation_path, replay_start_annotations = load_replay_start_annotations(args)
    excluded_episode_ids = EXCLUDED_INSPIRE_EPISODES.get(object_name, frozenset())
    candidate_episodes = []
    start_frame_by_episode: dict[str, int] = {}
    for episode in episodes:
        if int(episode.root.name) in excluded_episode_ids:
            continue
        start_frame = replay_start_frame(
            replay_start_annotations,
            object_name,
            episode,
        )
        if start_frame is None:
            print(
                f"[retrieval] excluding {object_name}/{episode.root.name}: "
                "no manual replay-start annotation"
            )
            continue
        candidate_episodes.append(episode)
        start_frame_by_episode[episode.root.name] = start_frame
    if not candidate_episodes:
        raise RuntimeError(
            f"no annotated Inspire DFTP replay episode for {object_name} in "
            f"{annotation_path}"
        )
    symmetry_axis_object = CONTINUOUS_SYMMETRY_AXIS_OBJECT.get(object_name)
    ranked = rank_by_object_pose(
        current_object_robot,
        candidate_episodes,
        translation_scale_m=args.retrieval_translation_scale_m,
        rotation_scale_rad=args.retrieval_rotation_scale_rad,
        continuous_symmetry_axis_object=symmetry_axis_object,
        continuous_symmetry_allow_axis_flip=(
            object_name in AXIS_FLIP_SYMMETRIC_OBJECTS
        ),
    )
    print(f"[retrieval] Inspire DFTP candidates by frame-0 {object_name} pose:")
    for rank, match in enumerate(ranked, start=1):
        symmetry_text = (
            f", symmetry={np.rad2deg(match.symmetry_angle_rad):+.1f} deg"
            f", flipped={'yes' if match.symmetry_flipped else 'no'}"
            if symmetry_axis_object is not None
            else ""
        )
        print(
            f"  {rank}. episode {match.episode.root.name}: "
            f"translation={match.translation_error_m:.4f} m, "
            f"rotation={np.rad2deg(match.rotation_error_rad):.1f} deg"
            f"{symmetry_text}, "
            f"score={match.score:.3f}"
        )
    selected = ranked[0]
    episode = selected.episode

    apex = automatic_eef_apex(episode.root)
    apex_frame = int(apex["arm_frame_index"])
    start_frame = start_frame_by_episode[episode.root.name]
    if start_frame >= apex_frame:
        raise ValueError(
            f"{object_name}/{episode.root.name} replay start frame {start_frame} "
            f"must be before apex frame {apex_frame}"
        )
    source_object_robot = selected.source_object_robot
    aligned_current_object_robot = selected.aligned_current_object_robot
    transformed = base.replay_core.relative_arm_actions(
        source_object_robot,
        aligned_current_object_robot,
        episode.arm_poses[start_frame : apex_frame + 1],
    )
    source_arm_times = episode.arm_times[start_frame : apex_frame + 1]
    source_duration = float(source_arm_times[-1] - source_arm_times[0])
    replay_duration = source_duration / float(args.rate_scale)
    replay_frame_count = max(2, int(np.ceil(replay_duration * args.hand_control_hz)) + 1)
    replay_times = np.linspace(0.0, replay_duration, replay_frame_count)
    source_query_times = source_arm_times[0] + replay_times * float(args.rate_scale)
    replay_poses = resample_pose_matrices(
        transformed,
        source_arm_times,
        source_query_times,
    )
    replay_raw = base.replay_core._zoh_resample(
        episode.hand_times,
        episode.hand_commands,
        source_query_times,
    )
    replay_poses, replay_raw = _densify_pose_and_raw(
        replay_poses,
        replay_raw,
        max_raw_step=float(args.hand_max_raw_step),
        max_translation_step_m=float(args.arm_max_step_m),
        max_angle_step_deg=float(args.arm_max_angle_step_deg),
    )
    replay_times = np.arange(len(replay_poses), dtype=np.float64) / float(
        args.hand_control_hz
    )

    plan_poses = [replay_poses]
    plan_raw = [replay_raw]
    plan_times = [replay_times]
    elapsed = float(replay_times[-1])
    last_pose = replay_poses[-1]
    held_raw = replay_raw[-1]

    def append_pose_stage(target: np.ndarray, minimum_seconds: float) -> None:
        nonlocal elapsed, last_pose
        steps = required_pose_steps(
            last_pose,
            target,
            minimum_steps=max(1, round(minimum_seconds * args.motion_control_hz)),
            max_translation_step_m=float(args.arm_max_step_m),
            max_angle_step_deg=float(args.arm_max_angle_step_deg),
        )
        stage_poses = interpolate_pose_segment(last_pose, target, steps)
        stage_raw = np.repeat(held_raw[None], steps, axis=0)
        stage_times = elapsed + np.arange(1, steps + 1, dtype=np.float64) / float(
            args.motion_control_hz
        )
        plan_poses.append(stage_poses)
        plan_raw.append(stage_raw)
        plan_times.append(stage_times)
        elapsed = float(stage_times[-1])
        last_pose = target

    lift_target = last_pose.copy()
    lift_target[2, 3] = float(args.post_grasp_z_m)
    append_pose_stage(lift_target, float(args.post_grasp_lift_seconds))

    transfer_target = last_pose.copy()
    transfer_target[0, 3] = float(placement_xy_robot[0])
    transfer_target[1, 3] = float(placement_xy_robot[1])
    append_pose_stage(transfer_target, float(args.post_grasp_transfer_seconds))

    if placement_target_name == "cutting_board":
        release_target = last_pose.copy()
        release_target[2, 3] = float(args.cutting_board_release_z_m)
        append_pose_stage(release_target, float(args.cutting_board_descent_seconds))

    release_steps = max(
        1,
        round(float(args.hand_open_seconds) * float(args.motion_control_hz)),
        int(
            np.ceil(
                np.max(np.abs(DFTP_OPEN_RAW - held_raw))
                / float(args.hand_max_raw_step)
            )
        ),
    )
    release_raw = np.linspace(
        held_raw,
        DFTP_OPEN_RAW,
        release_steps + 1,
        dtype=np.float64,
    )[1:]
    release_poses = np.repeat(last_pose[None], release_steps, axis=0)
    release_times = elapsed + np.arange(
        1, release_steps + 1, dtype=np.float64
    ) / float(args.motion_control_hz)
    plan_poses.append(release_poses)
    plan_raw.append(release_raw)
    plan_times.append(release_times)

    arm_poses = np.concatenate(plan_poses)
    hand_raw = np.concatenate(plan_raw)
    times = np.concatenate(plan_times)
    execution_blocker = None
    try:
        validate_arm_workspace(
            arm_poses,
            min_z_m=float(args.arm_min_z_m),
            max_z_m=float(args.arm_max_z_m),
            min_radius_m=float(args.arm_min_radius_m),
            max_radius_m=float(args.arm_max_radius_m),
        )
    except RuntimeError as exc:
        # Keep the unsafe plan available for diagnosis and Viser preview.  The
        # caller checks this blocker again before any controller is contacted.
        execution_blocker = str(exc)
    if np.any(hand_raw < 0) or np.any(hand_raw > 1000):
        hand_blocker = "planned Inspire command lies outside 0..1000"
        execution_blocker = (
            hand_blocker
            if execution_blocker is None
            else f"{execution_blocker}; {hand_blocker}"
        )
    return {
        "episode": episode,
        "selected": selected,
        "ranked": ranked,
        "apex": apex,
        "replay_start_frame": start_frame,
        "replay_start_annotation_path": annotation_path,
        "arm_poses": arm_poses,
        "hand_raw": hand_raw,
        "times": times,
        "replay_frame_count": len(replay_poses),
        "source_object_robot": source_object_robot,
        "current_object_robot": np.asarray(current_object_robot, dtype=np.float64),
        "aligned_current_object_robot": aligned_current_object_robot,
        "symmetry_angle_rad": selected.symmetry_angle_rad,
        "symmetry_flipped": selected.symmetry_flipped,
        "current_c2r": np.asarray(current_c2r, dtype=np.float64),
        "placement_xy_robot": np.asarray(placement_xy_robot, dtype=np.float64),
        "placement_target_name": placement_target_name,
        "execution_blocker": execution_blocker,
    }


def save_inspire_episode_plan(plan: Mapping[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = plan["selected"]
    ranked = plan["ranked"]
    path = output_dir / "inspire_episode_replay_plan.npz"
    np.savez_compressed(
        path,
        arm_action=plan["arm_poses"],
        arm_time=plan["times"],
        hand_action=plan["hand_raw"],
        replay_frame_count=plan["replay_frame_count"],
        selected_episode=selected.episode.root.name,
        apex_arm_frame=int(plan["apex"]["arm_frame_index"]),
        replay_start_arm_frame=int(plan["replay_start_frame"]),
        replay_start_annotation_path=str(plan["replay_start_annotation_path"]),
        apex_manual_review=str(plan["apex"].get("manual_review", "unknown")),
        source_object_robot=plan["source_object_robot"],
        current_object_robot=plan["current_object_robot"],
        aligned_current_object_robot=plan["aligned_current_object_robot"],
        symmetry_angle_rad=plan["symmetry_angle_rad"],
        symmetry_flipped=plan["symmetry_flipped"],
        current_c2r=plan["current_c2r"],
        placement_xy_robot=plan["placement_xy_robot"],
        placement_target_name=plan["placement_target_name"],
        execution_safe=plan["execution_blocker"] is None,
        execution_blocker=(plan["execution_blocker"] or ""),
        retrieval_translation_error_m=selected.translation_error_m,
        retrieval_rotation_error_rad=selected.rotation_error_rad,
        retrieval_score=selected.score,
        candidate_episode=np.asarray([item.episode.root.name for item in ranked]),
        candidate_score=np.asarray([item.score for item in ranked]),
    )
    print(
        f"[retrieval] selected Inspire DFTP {selected.episode.root.name}; "
        f"source arm frames={plan['replay_start_frame']}.."
        f"{plan['apex']['arm_frame_index']} (manual start -> apex), "
        f"({plan['apex'].get('manual_review', 'unknown')})"
    )
    print(f"[plan] saved {path}")
    return path


def inspire_plan_as_preview_trajectory(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Convert link6/raw replay data to the floating-hand preview convention."""

    from paradex.robot.inspire import inspire_action_to_qpos

    link6_poses = np.asarray(plan["arm_poses"], dtype=np.float64)
    wrist_poses = link6_poses @ load_link6_from_wrist()[None]
    wrist_rows = np.column_stack(
        (
            wrist_poses[:, :3, 3],
            R.from_matrix(wrist_poses[:, :3, :3]).as_quat(),
        )
    )
    return {
        "wrist": wrist_rows,
        "hand": inspire_action_to_qpos(np.asarray(plan["hand_raw"])),
        "timestamps": np.asarray(plan["times"], dtype=np.float64),
        "active_hand_dof_names": list(DEMOGRASP_HAND_JOINTS),
        "control_hz": 0.0,
    }


def execute_inspire_episode_plan(
    arm: Any,
    hand: Any,
    plan: Mapping[str, Any],
    capture_root: Path,
    args: argparse.Namespace,
) -> None:
    """Approach frame 0 and stream a planned object-relative Inspire replay."""

    poses = np.asarray(plan["arm_poses"], dtype=np.float64)
    raw = np.asarray(plan["hand_raw"], dtype=np.float64)
    timestamps = np.asarray(plan["times"], dtype=np.float64)
    current_pose = np.asarray(arm.get_data().get("position"), dtype=np.float64)
    current_raw = np.asarray(hand.get_data().get("joint_value"), dtype=np.float64)
    if current_pose.shape != (4, 4) or not np.all(np.isfinite(current_pose)):
        raise RuntimeError(f"invalid current xArm pose: {current_pose}")
    if current_raw.shape != (6,) or not np.all(np.isfinite(current_raw)):
        raise RuntimeError(f"invalid current Inspire state: {current_raw}")

    approach_targets, approach_seconds = base.replay_core._cartesian_approach_trajectory(
        current_pose,
        poses[0],
        linear_speed_mps=float(args.approach_linear_speed_mps),
        angular_speed_rps=float(args.approach_angular_speed_rps),
        min_seconds=float(args.approach_min_seconds),
        rate_hz=float(args.approach_rate_hz),
    )
    approach_steps = max(
        len(approach_targets),
        int(
            np.ceil(
                np.max(np.abs(raw[0] - current_raw))
                / float(args.hand_max_raw_step)
            )
        ),
    )
    approach_seconds = max(
        float(approach_seconds),
        approach_steps / float(args.approach_rate_hz),
    )
    approach_poses = interpolate_pose_segment(current_pose, poses[0], approach_steps)
    approach_raw = np.linspace(
        current_raw, raw[0], approach_steps + 1, dtype=np.float64
    )[1:]

    def stream(
        arm_poses: np.ndarray,
        hand_commands: np.ndarray,
        command_times: np.ndarray,
        stage: str,
    ) -> None:
        started = time.perf_counter()
        for index, (pose, command, target_time) in enumerate(
            zip(arm_poses, hand_commands, command_times)
        ):
            arm.move(pose)
            hand.move(command)
            if arm.is_error():
                raise RuntimeError(f"xArm controller error during {stage} frame {index}")
            time.sleep(max(0.0, started + float(target_time) - time.perf_counter()))

    approach_times = np.linspace(
        approach_seconds / approach_steps,
        approach_seconds,
        approach_steps,
        dtype=np.float64,
    )
    print(
        f"[execute] current state -> Inspire episode frame 0: "
        f"{approach_steps} frames over {approach_seconds:.2f}s"
    )
    stream(approach_poses, approach_raw, approach_times, "episode approach")
    print(
        f"[execute] object-relative Inspire episode {plan['episode'].root.name}: "
        f"{len(poses)} planned frames over {timestamps[-1]:.2f}s"
    )
    stream(poses, raw, timestamps, "Inspire episode replay")
    np.save(capture_root / "inspire_episode_executed_arm_pose.npy", poses)
    np.save(capture_root / "inspire_episode_executed_hand_raw.npy", raw)
    np.save(capture_root / "inspire_episode_executed_timestamps_s.npy", timestamps)
    print(
        f"[execute] Inspire episode replay complete over "
        f"{plan['placement_target_name']}; hand opened at the target."
    )


def run_inspire_episode_replay(
    args: argparse.Namespace,
    *,
    capture_root: Path,
    object_name: str,
    current_object_robot: np.ndarray,
    current_c2r: np.ndarray,
    cutting_board_robot: np.ndarray,
    pink_box_robot: np.ndarray,
    placement_xy_robot: np.ndarray,
    placement_target_name: str,
    episodes: list[base.replay_core.Episode],
    arm: Any | None = None,
    hand: Any | None = None,
    return_arm_qpos: np.ndarray | None = None,
    post_placement_return: Callable[[Any, np.ndarray], None] | None = None,
) -> Path:
    plan = build_inspire_episode_plan(
        args,
        object_name=object_name,
        current_object_robot=current_object_robot,
        current_c2r=current_c2r,
        placement_xy_robot=placement_xy_robot,
        placement_target_name=placement_target_name,
        episodes=episodes,
    )
    plan_path = save_inspire_episode_plan(plan, capture_root / "replay")
    execution_blocker = plan["execution_blocker"]
    if execution_blocker is not None:
        print(
            f"[safety] trajectory is preview-only and will not execute: "
            f"{execution_blocker}",
            file=sys.stderr,
        )

    if not args.no_vis and args.replay_preview:
        preview_args = copy.copy(args)
        preview_args.demograsp_duration_seconds = max(
            float(plan["times"][-1]),
            1.0 / float(args.hand_control_hz),
        )
        visualize_sorting_result(
            preview_args,
            object_name,
            current_object_robot,
            cutting_board_robot,
            pink_box_robot,
            inspire_plan_as_preview_trajectory(plan),
            trajectory_label=(
                f"Inspire episode {plan['episode'].root.name} object-relative replay"
                + (" [UNSAFE - PREVIEW ONLY]" if execution_blocker else "")
            ),
        )

    if execution_blocker is not None:
        if args.execute:
            raise RuntimeError(execution_blocker)
        print("[plan] unsafe trajectory saved for inspection; execution disabled.")
        return plan_path

    if not args.execute:
        print("[plan] complete; no robot or hand motion commands were sent.")
        return plan_path
    if not args.auto_execute and not confirm_hand_execution():
        return plan_path

    owns_hand = hand is None
    owns_arm = arm is None
    try:
        if hand is None:
            print(
                "[execute] connecting RIGHT Inspire DFTP at "
                f"{args.inspire_right_ip}:{args.inspire_port} via "
                f"{args.inspire_interface}."
            )
            hand = RightInspireDFTP(
                args.inspire_right_ip,
                int(args.inspire_port),
                float(args.inspire_timeout),
                args.inspire_interface,
            )
        if arm is None:
            from paradex.io.robot_controller import get_arm

            print("[execute] connecting xArm ROS Cartesian controller.")
            arm = get_arm("xarm", servo_api="cartesian_aa")
        round_initial_qpos = None
        if post_placement_return is not None:
            round_initial_qpos = (
                np.asarray(return_arm_qpos, dtype=np.float64).copy()
                if return_arm_qpos is not None
                else np.asarray(arm.get_data().get("qpos"), dtype=np.float64).copy()
            )
        execute_inspire_episode_plan(arm, hand, plan, capture_root, args)
        if post_placement_return is not None:
            assert round_initial_qpos is not None
            post_placement_return(arm, round_initial_qpos)
    finally:
        if owns_arm and arm is not None:
            arm.end()
        if owns_hand and hand is not None:
            hand.close()
    return plan_path


def visualize_sorting_result(
    args: argparse.Namespace,
    object_name: str,
    object_robot: np.ndarray,
    cutting_board_robot: np.ndarray,
    pink_box_robot: np.ndarray,
    grasp_trajectory: Mapping[str, Any],
    *,
    trajectory_label: str = "DemoGrasp",
) -> None:
    """Show Object6D, targets, and the robot-frame floating Inspire hand."""
    mesh = base.load_mesh(SORTING_MESH_ROOT, object_name)
    viewer = base.ViserViewer(
        up_direction=np.array([0.0, 0.0, 1.0]),
        scene_title=f"Sorting Object6D: {object_name} (robot frame)",
    )
    viewer.add_floor(height=0.0)
    viewer.add_frame("robot_origin", np.eye(4, dtype=np.float32))
    viewer.add_object(object_name, mesh, object_robot, opacity=1.0)

    targets = (
        ("cutting_board", cutting_board_robot, (1.0, 0.15, 0.05)),
        ("pink_box", pink_box_robot, (1.0, 0.1, 0.75)),
    )
    for target_name, point_robot, color in targets:
        point_frame = np.eye(4, dtype=np.float32)
        point_frame[:3, 3] = point_robot
        viewer.add_frame(f"{target_name}_triangulation", point_frame)
        viewer.add_sphere(
            f"{target_name}_triangulation",
            point_robot,
            radius=0.02,
            color=color,
        )
        viewer.server.scene.add_label(
            f"/{target_name}_triangulation/frame/label",
            f"{target_name} triangulation",
        )

    placement_name, placement_xy = placement_target_xy(
        object_name,
        cutting_board_robot,
        pink_box_robot,
        cutting_board_x_offset_m=args.cutting_board_x_offset_m,
        cutting_board_y_offset_m=args.cutting_board_y_offset_m,
        pink_box_x_offset_m=args.pink_box_x_offset_m,
        pink_box_y_offset_m=args.pink_box_y_offset_m,
    )
    placement_point = np.array(
        [
            placement_xy[0],
            placement_xy[1],
            (
                args.cutting_board_release_z_m
                if placement_name == "cutting_board"
                else args.post_grasp_z_m
            ),
        ],
        dtype=np.float64,
    )
    placement_frame = np.eye(4, dtype=np.float32)
    placement_frame[:3, 3] = placement_point
    viewer.add_frame("resolved_placement_target", placement_frame)
    viewer.add_sphere(
        "resolved_placement_target",
        placement_point,
        radius=0.025,
        color=(0.1, 1.0, 0.25),
    )
    viewer.server.scene.add_label(
        "/resolved_placement_target/frame/label",
        f"actual {placement_name} placement target",
    )

    source_wrist = np.asarray(grasp_trajectory["wrist"], dtype=np.float64)
    preview_timestamps, preview_duration_scale = stretch_timestamps_to_duration(
        grasp_trajectory["timestamps"], args.demograsp_duration_seconds
    )
    hand, preview_times = resample_hand_qpos(
        grasp_trajectory["hand"], preview_timestamps, args.hand_control_hz
    )
    wrist_matrices = resample_wrist_poses(
        source_wrist, preview_timestamps, preview_times
    )
    # inspire_float.urdf chains X/Y/Z translation and intrinsic X/Y/Z
    # rotation joints before the six active hand joints.
    wrist_euler_xyz = R.from_matrix(wrist_matrices[:, :3, :3]).as_euler("XYZ")
    floating_hand_qpos = np.concatenate(
        [wrist_matrices[:, :3, 3], wrist_euler_xyz, hand], axis=1
    )
    viewer.add_robot(
        name="sorting_hand",
        urdf_path=str(DEMOGRASP_HAND_URDF),
        # This is already a hand-only floating URDF.  The viewer's arm filter
        # recognizes only ``right_hand_*`` link names and would incorrectly
        # hide its ``right_thumb_*``/``right_index_*`` meshes.
        include_arm_meshes=True,
    )
    expected_joint_names = [
        "x_joint",
        "y_joint",
        "z_joint",
        "x_rotation_joint",
        "y_rotation_joint",
        "z_rotation_joint",
    ] + list(
        grasp_trajectory["active_hand_dof_names"]
    )
    actual_joint_names = viewer.robot_dict["sorting_hand"].urdf.get_joint_names()
    if actual_joint_names != expected_joint_names:
        raise RuntimeError(
            "Floating Inspire URDF joint order mismatch: "
            f"expected={expected_joint_names}, actual={actual_joint_names}"
        )
    viewer.server.scene.add_spline_catmull_rom(
        name="/sorting/wrist_path",
        positions=wrist_matrices[:, :3, 3],
        color=(60, 220, 255),
        line_width=3.0,
    )
    viewer.add_traj(
        "sorting_trajectory",
        robot_traj={"sorting_hand": floating_hand_qpos},
    )
    viewer.gui_framerate.value = float(args.hand_control_hz)
    viewer.update_scene(0)

    print(
        f"Viser is replaying {trajectory_label} ({len(source_wrist)} source frames) as "
        f"{len(wrist_matrices)} interpolated frames at {args.hand_control_hz:g} Hz "
        f"over {args.demograsp_duration_seconds:g}s "
        f"({preview_duration_scale:.2f}x time stretch). "
        "Press Ctrl+C to continue sorting."
    )
    try:
        while True:
            viewer.update()
    except KeyboardInterrupt:
        pass
    finally:
        stop_viser_safely(viewer.server)


def run_sorting_round(
    args: argparse.Namespace,
    *,
    replay_arm: Any | None = None,
    replay_hand: Any | None = None,
    episode_cache: Mapping[str, list[base.replay_core.Episode]],
    return_arm_qpos: np.ndarray | None = None,
    placed_obstacles: list[SphereObstacle] | None = None,
) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    relative_capture = Path(args.save_path) / timestamp
    capture_root = Path(shared_dir) / relative_capture
    remote_path = (Path("shared_data") / relative_capture / "raw").as_posix()
    base.capture_once(capture_root, remote_path, args.rcc_entry)

    request = sorting_rpc_request(
        capture_root,
        generate_grasp_trajectory=not getattr(args, "inspire_episode_replay", False),
    )
    response = base.send_rpc_once(args.rpc_addr, request, args.rpc_timeout_ms)
    detection = extract_sorting_detection(response)
    object_name = detection["object_name"]
    print(f"[detection] server detected object: {object_name}")
    grasp_trajectory = (
        None
        if getattr(args, "inspire_episode_replay", False)
        else extract_grasp_trajectory(response)
    )
    if grasp_trajectory is not None:
        grasp_trajectory = offset_demograsp_wrist_z(
            grasp_trajectory,
            getattr(args, "demograsp_z_offset_m", 0.002),
        )

    c2r = base.to_4x4(np.load(capture_root / "C2R.npy")).astype(np.float64)
    robot_from_world = np.linalg.inv(c2r)
    object_robot = robot_from_world @ detection["pose_world"]

    def world_point_to_robot(point: np.ndarray) -> np.ndarray:
        return (robot_from_world @ np.append(point, 1.0))[:3]

    cutting_board_robot = world_point_to_robot(detection["cutting_board_world"])
    pink_box_robot = world_point_to_robot(detection["pink_box_world"])
    target_name, resolved_placement_xy = placement_target_xy(
        object_name,
        cutting_board_robot,
        pink_box_robot,
        cutting_board_x_offset_m=args.cutting_board_x_offset_m,
        cutting_board_y_offset_m=args.cutting_board_y_offset_m,
        pink_box_x_offset_m=args.pink_box_x_offset_m,
        pink_box_y_offset_m=args.pink_box_y_offset_m,
    )
    target_robot = (
        cutting_board_robot.copy()
        if target_name == "cutting_board"
        else pink_box_robot.copy()
    )
    target_robot[:2] = resolved_placement_xy

    def return_after_placement(arm: Any, goal_qpos: np.ndarray) -> None:
        use_nfg = getattr(args, "nfg_return", False)
        obstacle = None
        obstacles: list[SphereObstacle] = []
        if use_nfg:
            obstacle = placed_object_obstacle(
                object_name,
                target_robot,
                margin_m=float(args.nfg_obstacle_margin_m),
            )
            obstacles = list(placed_obstacles or ()) + [obstacle]
        should_return = confirm_ready_return()
        try:
            if not should_return:
                return
            execute_post_placement_retreat(arm, args, capture_root)
            if use_nfg:
                execute_nfg_return(
                    arm,
                    goal_qpos,
                    obstacles,
                    args,
                    capture_root,
                )
            else:
                execute_slow_ready_return(arm, args, capture_root)
        finally:
            if obstacle is not None and placed_obstacles is not None:
                placed_obstacles.append(obstacle)

    normalized = {
        "server_response": response,
        "object_name": object_name,
        "pose_world": detection["pose_world"].tolist(),
        "pose_robot": object_robot.tolist(),
        "C2R": c2r.tolist(),
        "cutting_board_point_world": detection["cutting_board_world"].tolist(),
        "cutting_board_point_robot": cutting_board_robot.tolist(),
        "pink_box_point_world": detection["pink_box_world"].tolist(),
        "pink_box_point_robot": pink_box_robot.tolist(),
        "selected_target": target_name,
        "selected_target_point_robot": target_robot.tolist(),
    }
    if grasp_trajectory is not None:
        normalized.update(
            {
                "demograsp_wrist_pose_command_robot": grasp_trajectory[
                    "wrist"
                ].tolist(),
                "demograsp_hand_qpos_command": grasp_trajectory["hand"].tolist(),
                "demograsp_timestamps_s": grasp_trajectory["timestamps"].tolist(),
            }
        )
    result_path = capture_root / "sorting_object_6d.json"
    result_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[sorting] detected {object_name}; destination={target_name}, "
        f"target_robot={np.round(target_robot, 4).tolist()}"
    )
    print(f"[sorting] saved normalized response: {result_path}")
    if grasp_trajectory is not None:
        print(
            f"[sorting] DemoGrasp trajectory: {len(grasp_trajectory['wrist'])} "
            f"frames, {grasp_trajectory['control_hz']:.0f} Hz"
        )

    if getattr(args, "inspire_episode_replay", False):
        run_inspire_episode_replay(
            args,
            capture_root=capture_root,
            object_name=object_name,
            current_object_robot=object_robot,
            current_c2r=c2r,
            cutting_board_robot=cutting_board_robot,
            pink_box_robot=pink_box_robot,
            placement_xy_robot=resolved_placement_xy,
            placement_target_name=target_name,
            episodes=episode_cache[object_name],
            arm=replay_arm,
            hand=replay_hand,
            return_arm_qpos=return_arm_qpos,
            post_placement_return=return_after_placement,
        )
        return

    if not args.no_vis:
        if getattr(args, "execute", False):
            print(
                "[execute] After Viser closes and Y is confirmed, xArm and "
                f"RIGHT Inspire replay DemoGrasp at {args.hand_control_hz:g} Hz; "
                f"other motion runs at {args.motion_control_hz:g} Hz."
            )
        visualize_sorting_result(
            args,
            object_name,
            object_robot,
            cutting_board_robot,
            pink_box_robot,
            grasp_trajectory,
        )

    if getattr(args, "execute", False):
        if not confirm_hand_execution():
            return
        hand = replay_hand
        arm = replay_arm
        owns_hand = hand is None
        owns_arm = arm is None
        try:
            if hand is None:
                print(
                    "[execute] connecting RIGHT Inspire DFTP at "
                    f"{args.inspire_right_ip}:{args.inspire_port} via "
                    f"{args.inspire_interface}."
                )
                hand = RightInspireDFTP(
                    args.inspire_right_ip,
                    int(args.inspire_port),
                    float(args.inspire_timeout),
                    args.inspire_interface,
                )
            if arm is None:
                from paradex.io.robot_controller import get_arm

                print("[execute] connecting xArm ROS Cartesian controller.")
                arm = get_arm("xarm", servo_api="cartesian_aa")
            round_initial_qpos = (
                np.asarray(return_arm_qpos, dtype=np.float64).copy()
                if return_arm_qpos is not None
                else READY_ARM_QPOS_RAD.copy()
            )
            execute_xarm_and_inspire(
                arm,
                hand,
                grasp_trajectory,
                resolved_placement_xy,
                target_name,
                capture_root,
                replay_hz=float(args.hand_control_hz),
                replay_duration_seconds=float(args.demograsp_duration_seconds),
                motion_hz=float(args.motion_control_hz),
                xy_seconds=float(args.arm_xy_seconds),
                frame0_seconds=float(args.arm_frame0_seconds),
                hold_seconds=float(args.post_grasp_hold_seconds),
                post_grasp_z_m=float(args.post_grasp_z_m),
                post_grasp_lift_seconds=float(args.post_grasp_lift_seconds),
                post_grasp_transfer_seconds=float(
                    args.post_grasp_transfer_seconds
                ),
                cutting_board_release_z_m=float(args.cutting_board_release_z_m),
                cutting_board_descent_seconds=float(
                    args.cutting_board_descent_seconds
                ),
                max_raw_step=float(args.hand_max_raw_step),
                max_translation_step_m=float(args.arm_max_step_m),
                max_angle_step_deg=float(args.arm_max_angle_step_deg),
                min_z_m=float(args.arm_min_z_m),
                max_z_m=float(args.arm_max_z_m),
                min_radius_m=float(args.arm_min_radius_m),
                max_radius_m=float(args.arm_max_radius_m),
            )
            return_after_placement(arm, round_initial_qpos)
        finally:
            if owns_arm and arm is not None:
                arm.end()
            if owns_hand and hand is not None:
                hand.close()
        return

    replay_args = object_replay_args(args, object_name)
    replay_args.replay_output_dir = capture_root / "replay"
    replay_args.mesh_root_dir = SORTING_MESH_ROOT
    replay_args.placement_target_name = target_name
    replay_args.box_pink_clearance_m = (
        args.cutting_board_z_m
        if target_name == "cutting_board"
        else args.box_pink_clearance_m
    )
    replay_closest_episode_into_box_pink(
        replay_args,
        current_object_robot=object_robot,
        current_c2r=c2r,
        box_point_robot=target_robot,
        arm=replay_arm,
        hand=replay_hand,
        episodes=episode_cache[object_name],
    )


def run_idle_loop(args: argparse.Namespace) -> None:
    if getattr(args, "inspire_episode_replay", False):
        episode_cache = preload_inspire_episodes(args)
    else:
        episode_cache = {} if args.execute else preload_sorting_episodes(args)
    replay_arm = None
    replay_hand = None
    return_arm_qpos = READY_ARM_QPOS_RAD.copy()
    placed_obstacles: list[SphereObstacle] = []
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    sigint_overridden = False
    try:
        if args.execute:
            from paradex.io.robot_controller import get_arm

            print(
                "[idle] connecting persistent xArm and RIGHT Inspire "
                "controllers for this sorting session."
            )
            replay_arm = get_arm("xarm", servo_api="cartesian_aa")
            try:
                replay_hand = RightInspireDFTP(
                    args.inspire_right_ip,
                    int(args.inspire_port),
                    float(args.inspire_timeout),
                    args.inspire_interface,
                )
            except Exception:
                replay_arm.end()
                replay_arm = None
                raise
            # rclpy installs a process-wide SIGINT handler while constructing
            # the xArm controller.  Viser uses Ctrl+C to close its preview, so
            # keep Ctrl+C as a normal KeyboardInterrupt while the persistent
            # controllers remain alive across sorting rounds.
            signal.signal(signal.SIGINT, signal.default_int_handler)
            sigint_overridden = True

        print("[idle] Ready. Enter c to detect and sort, or q to quit.")
        while True:
            try:
                key = input("[idle] c=capture/sort, q=quit > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n[idle] exiting.")
                return
            if key == "q":
                print("[idle] exiting.")
                return
            if key != "c":
                if key:
                    print(f"[idle] unknown command {key!r}; enter c or q.")
                continue
            try:
                run_sorting_round(
                    args,
                    replay_arm=replay_arm,
                    replay_hand=replay_hand,
                    episode_cache=episode_cache,
                    return_arm_qpos=return_arm_qpos,
                    placed_obstacles=placed_obstacles,
                )
            except NFGReturnPlanningError as exc:
                print(f"[safety] {exc}; ending the sorting session.", file=sys.stderr)
                traceback.print_exc()
                return
            except Exception as exc:
                print(f"[round] failed: {exc}", file=sys.stderr)
                traceback.print_exc()
                print("[round] returning to idle after failure.")
            else:
                print("[round] complete; returning to idle.")
    finally:
        if replay_hand is not None:
            replay_hand.close()
        if replay_arm is not None:
            replay_arm.end()
        if sigint_overridden:
            signal.signal(signal.SIGINT, original_sigint_handler)


def main() -> None:
    run_idle_loop(parse_args())


if __name__ == "__main__":
    main()
