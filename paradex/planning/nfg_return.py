"""Collision-checked xArm return planning via natural-functional-gradient."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable
import xml.etree.ElementTree as ET

import numpy as np


class NFGReturnPlanningError(RuntimeError):
    """A collision-free return could not be proven or executed."""


XARM_JOINTS = (
    "robot0:shoulder_pan_joint",
    "robot0:shoulder_lift_joint",
    "robot0:elbow_flex_joint",
    "robot0:forearm_roll_joint",
    "robot0:wrist_flex_joint",
    "robot0:wrist_roll_joint",
)
RIGHT_HAND_JOINTS = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
    "right_index_1_joint",
    "right_index_2_joint",
    "right_middle_1_joint",
    "right_middle_2_joint",
    "right_ring_1_joint",
    "right_ring_2_joint",
    "right_little_1_joint",
    "right_little_2_joint",
)


@dataclass(frozen=True)
class SphereObstacle:
    name: str
    center_m: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=np.float64).reshape(-1)
        radius = float(self.radius_m)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("NFG obstacle center must be a finite [3] vector")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("NFG obstacle radius must be positive and finite")
        object.__setattr__(self, "center_m", center.copy())
        object.__setattr__(self, "radius_m", radius)


@dataclass(frozen=True)
class NFGReturnPlan:
    positions: np.ndarray
    times_s: np.ndarray
    solve_time_s: float
    iterations: int
    backend: str
    min_contact_distance_m: float


def _finite_qpos(value: np.ndarray, label: str) -> np.ndarray:
    qpos = np.asarray(value, dtype=np.float64).reshape(-1)
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"{label} must be a finite xArm joint vector with shape (6,)")
    return qpos.copy()


def _write_scene(
    path: Path,
    *,
    robot_xml: Path,
    obstacles: list[SphereObstacle],
) -> None:
    root = ET.Element("mujoco", {"model": "paradex_nfg_return"})
    ET.SubElement(root, "include", {"file": str(robot_xml.resolve())})
    world = ET.SubElement(root, "worldbody")
    for index, obstacle in enumerate(obstacles):
        body = ET.SubElement(
            world,
            "body",
            {
                "name": f"placed_object_{index}_{obstacle.name}",
                "pos": " ".join(f"{value:.9g}" for value in obstacle.center_m),
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"placed_object_geom_{index}",
                "type": "sphere",
                "size": f"{obstacle.radius_m:.9g}",
                "rgba": "0.9 0.15 0.05 0.6",
                # Obstacles collide with the robot's default group 1 but not
                # with one another, so stacked placements do not make every
                # candidate trajectory invalid through obstacle-only contact.
                "contype": "2",
                "conaffinity": "1",
            },
        )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _load_nfg_api(repo_root: Path) -> tuple[Any, Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    repo_root = repo_root.expanduser().resolve()
    package_root = repo_root / "natural-functional-gradient"
    required = (repo_root / "ri_motion_v5", package_root / "parallel_nfg.py")
    if not all(path.exists() for path in required):
        raise FileNotFoundError(
            f"natural-functional-gradient checkout is incomplete: {repo_root}"
        )
    for path in (repo_root, package_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        from ri_motion_v5.parser import MuJoCoParser
        from parallel_nfg import (
            build_scene_context,
            build_trajectory_context,
            optimize_trajectory,
            validate_trajectory_cpu,
        )
    except ImportError as exc:
        raise RuntimeError(
            "NFG dependencies are unavailable in this Python environment; "
            f"install {repo_root / 'requirements.txt'}"
        ) from exc
    return (
        MuJoCoParser,
        build_scene_context,
        build_trajectory_context,
        optimize_trajectory,
        validate_trajectory_cpu,
    )


def plan_collision_free_return(
    qpos_start: np.ndarray,
    qpos_goal: np.ndarray,
    obstacles: list[SphereObstacle],
    *,
    repo_root: Path,
    rate_hz: float,
    speed_rps: float,
    min_seconds: float,
    backend: str = "auto",
    num_samples: int = 100,
    max_iterations: int = 30,
    timeout_seconds: float = 30.0,
    api_loader: Callable[[Path], tuple[Any, ...]] = _load_nfg_api,
) -> NFGReturnPlan:
    """Plan and CPU-validate a return path; never return an unsafe fallback."""

    start = _finite_qpos(qpos_start, "qpos_start")
    goal = _finite_qpos(qpos_goal, "qpos_goal")
    rate_hz = float(rate_hz)
    speed_rps = float(speed_rps)
    min_seconds = float(min_seconds)
    if rate_hz <= 0.0 or speed_rps <= 0.0 or min_seconds <= 0.0:
        raise ValueError("NFG rate, speed, and minimum duration must be positive")
    duration = max(min_seconds, float(np.max(np.abs(goal - start))) / speed_rps)
    frequency = max(1, int(round(rate_hz)))
    repo_root = Path(repo_root).expanduser().resolve()
    robot_xml = repo_root / "asset/xarm6_inspire/xarm6_inspire_site_added.xml"
    if not robot_xml.is_file():
        raise FileNotFoundError(f"NFG xArm model not found: {robot_xml}")

    (
        parser_type,
        build_scene_context,
        build_trajectory_context,
        optimize_trajectory,
        validate_trajectory_cpu,
    ) = api_loader(repo_root)

    with tempfile.TemporaryDirectory(prefix="paradex-nfg-") as temp_dir:
        scene_path = Path(temp_dir) / "return_scene.xml"
        _write_scene(scene_path, robot_xml=robot_xml, obstacles=list(obstacles))
        env = parser_type(rel_xml_path=str(scene_path), verbose=False)
        active_hand_joints = env.get_active_among_joints(list(RIGHT_HAND_JOINTS))
        hand_qpos = np.zeros(len(active_hand_joints), dtype=np.float64)
        env.forward(q=start, joint_names=list(XARM_JOINTS))
        if active_hand_joints:
            env.forward(q=hand_qpos, joint_names=active_hand_joints)
        scene_context = build_scene_context(
            env=env,
            joints_use=list(XARM_JOINTS),
            joints_ee=active_hand_joints,
            ee_body_name="inspire_rh_palm",
        )
        trajectory_context = build_trajectory_context(
            scene_context=scene_context,
            qpos_start=start,
            qpos_end=goal,
            qpos_ee=hand_qpos,
            frequency=frequency,
            duration=duration,
        )
        result = optimize_trajectory(
            scene_context=scene_context,
            traj_context=trajectory_context,
            backend=backend,
            num_samples=int(num_samples),
            n_max_iter=int(max_iterations),
            timeout=float(timeout_seconds),
            verify_with_cpu=True,
        )
        validation = validate_trajectory_cpu(
            scene_context=scene_context,
            traj_context=trajectory_context,
            traj=result.traj,
            collision_tolerance=0.0,
        )
        if not result.success or not validation.collision_free:
            raise RuntimeError(
                "NFG failed to produce a CPU-verified collision-free return path"
            )
        positions = np.asarray(result.traj, dtype=np.float64)
        times = np.arange(len(positions), dtype=np.float64) / rate_hz
        if len(positions) < 2 or np.any(np.diff(times) <= 0.0):
            raise RuntimeError("NFG returned an invalid timed trajectory")
        return NFGReturnPlan(
            positions=positions,
            times_s=times,
            solve_time_s=float(result.solve_time),
            iterations=int(result.iterations),
            backend=str(result.backend),
            min_contact_distance_m=float(validation.min_contact_dist),
        )
