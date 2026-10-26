#!/usr/bin/env python3
"""Manually annotate the source frame where each Inspire replay should begin."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from threading import Event
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paradex.retargetor.experiment import replay_object_relative as replay_core
from paradex.robot.inspire import inspire_action_to_qpos
from paradex.visualization.robot import simplify_mesh
from paradex.visualization.visualizer.viser import ViserViewer
from src.object6d import capture_sil as capture
from src.object6d import capture_sil_sorting as sorting


DEFAULT_OUTPUT_NAME = "replay_start_annotations.json"


def empty_annotation_document(dataset_root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "description": (
            "Manually selected replay start frames. Frame indices are zero-based "
            "and refer directly to raw/arm/action.npy."
        ),
        "dataset_root": str(dataset_root.resolve()),
        "coordinate_frame": "robot_base",
        "source": "raw/arm/action.npy",
        "annotations": {},
    }


def load_annotation_document(path: Path, dataset_root: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_annotation_document(dataset_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), dict):
        raise ValueError(f"invalid replay-start annotation file: {path}")
    recorded_root = payload.get("dataset_root")
    if recorded_root is not None and Path(recorded_root).resolve() != dataset_root.resolve():
        raise ValueError(
            f"annotation dataset_root is {recorded_root}, expected {dataset_root.resolve()}"
        )
    return payload


def save_annotation_document(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def annotation_record(episode: replay_core.Episode, frame_index: int) -> dict[str, Any]:
    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= len(episode.arm_poses):
        raise ValueError(
            f"replay start frame {frame_index} is outside 0..{len(episode.arm_poses) - 1}"
        )
    arm_times = np.asarray(episode.arm_times, dtype=np.float64).reshape(-1)
    return {
        "arm_frame_index": frame_index,
        "arm_timestamp": float(arm_times[frame_index]),
        "elapsed_seconds": float(arm_times[frame_index] - arm_times[0]),
        "eef_pose_robot": np.asarray(
            episode.arm_poses[frame_index], dtype=np.float64
        ).tolist(),
        "trajectory_frame_count": int(len(episode.arm_poses)),
        "annotation_method": "manual_viser_timestep",
    }


def source_preview_trajectory(
    episode: replay_core.Episode,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact source-frame floating-hand qpos and wrist poses."""

    arm_times = np.asarray(episode.arm_times, dtype=np.float64).reshape(-1)
    hand_raw = replay_core._zoh_resample(
        episode.hand_times,
        episode.hand_commands,
        arm_times,
    )
    hand_qpos = inspire_action_to_qpos(np.asarray(hand_raw, dtype=np.float64))
    wrist_poses = (
        np.asarray(episode.arm_poses, dtype=np.float64)
        @ sorting.load_link6_from_wrist()[None]
    )
    wrist_euler_xyz = R.from_matrix(wrist_poses[:, :3, :3]).as_euler("XYZ")
    floating_hand_qpos = np.concatenate(
        (wrist_poses[:, :3, 3], wrist_euler_xyz, hand_qpos),
        axis=1,
    )
    return floating_hand_qpos, wrist_poses


def annotate_episode_in_viser(
    args: argparse.Namespace,
    object_name: str,
    episode: replay_core.Episode,
    previous_frame: int | None,
) -> tuple[str, int | None]:
    floating_hand_qpos, wrist_poses = source_preview_trajectory(episode)
    source_object_robot = (
        np.linalg.inv(np.asarray(episode.source_c2r, dtype=np.float64))
        @ np.asarray(episode.source_object_world, dtype=np.float64)
    )

    viewer = ViserViewer(
        up_direction=np.array([0.0, 0.0, 1.0]),
        scene_title=f"Replay start annotation: {object_name}/{episode.root.name}",
    )
    viewer.add_floor(height=0.0)
    viewer.add_frame("robot_origin", np.eye(4, dtype=np.float32))
    mesh = simplify_mesh(
        capture.load_mesh(args.mesh_root, object_name),
        args.preview_object_max_faces,
    )
    viewer.add_object(object_name, mesh, source_object_robot, opacity=1.0)
    viewer.add_robot(
        name="source_hand",
        urdf_path=str(sorting.DEMOGRASP_HAND_URDF),
        include_arm_meshes=True,
        max_mesh_faces=args.preview_robot_link_max_faces,
    )
    viewer.server.scene.add_spline_catmull_rom(
        name="/annotation/wrist_path",
        positions=wrist_poses[:, :3, 3],
        color=(60, 220, 255),
        line_width=3.0,
    )
    viewer.add_traj(
        "source_episode",
        robot_traj={"source_hand": floating_hand_qpos},
    )
    viewer.gui_framerate.value = float(args.preview_fps)
    viewer.gui_playing.value = False
    initial_frame = 0 if previous_frame is None else int(previous_frame)
    initial_frame = int(np.clip(initial_frame, 0, len(floating_hand_qpos) - 1))
    viewer.gui_timestep.value = initial_frame
    viewer.update_scene(initial_frame)

    finished = Event()
    result: dict[str, Any] = {"action": None, "frame": None}
    with viewer.server.gui.add_folder("Replay start annotation", expand_by_default=True):
        viewer.server.gui.add_text(
            "Episode",
            initial_value=f"{object_name}/{episode.root.name}",
            disabled=True,
        )
        viewer.server.gui.add_text(
            "Source frame range",
            initial_value=f"0..{len(floating_hand_qpos) - 1}",
            disabled=True,
        )
        save_button = viewer.server.gui.add_button("Save current step")
        skip_button = viewer.server.gui.add_button("Skip episode")
        quit_button = viewer.server.gui.add_button("Save progress and quit")

    @save_button.on_click
    def _(_) -> None:
        result["action"] = "save"
        result["frame"] = int(viewer.gui_timestep.value)
        finished.set()

    @skip_button.on_click
    def _(_) -> None:
        result["action"] = "skip"
        finished.set()

    @quit_button.on_click
    def _(_) -> None:
        result["action"] = "quit"
        finished.set()

    previous_text = "none" if previous_frame is None else str(previous_frame)
    print(
        f"[annotate] {object_name}/{episode.root.name}: source frames "
        f"0..{len(floating_hand_qpos) - 1}, previous={previous_text}. "
        "Scrub Playback/Timestep, then click Save current step."
    )
    try:
        while not finished.is_set():
            viewer.update()
    except KeyboardInterrupt:
        result["action"] = "quit"
    finally:
        replay_core._stop_viser_server(viewer.server)
    return str(result["action"]), result["frame"]


def selected_objects(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.objects or sorting.SUPPORTED_OBJECTS)


def load_episodes_for_annotation(
    args: argparse.Namespace,
    object_name: str,
) -> list[replay_core.Episode]:
    object_root = args.dataset_root / object_name
    episode_ids = capture._discover_candidate_episode_ids(object_root)
    excluded = sorting.EXCLUDED_INSPIRE_EPISODES.get(object_name, frozenset())
    episode_ids = tuple(item for item in episode_ids if item not in excluded)
    if args.episodes is not None:
        requested = set(args.episodes)
        episode_ids = tuple(item for item in episode_ids if item in requested)
    episodes: list[replay_core.Episode] = []
    for episode_id in episode_ids:
        try:
            episodes.extend(
                sorting.load_candidate_episodes(
                    object_root,
                    (episode_id,),
                    robot="inspire_dftp",
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[annotate] skipping {object_name}/{episode_id}: {exc}")
    return episodes


def run(args: argparse.Namespace) -> Path:
    args.dataset_root = args.dataset_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.dataset_root / DEFAULT_OUTPUT_NAME
    )
    payload = load_annotation_document(output, args.dataset_root)
    annotations = payload["annotations"]

    for object_name in selected_objects(args):
        object_annotations = annotations.setdefault(object_name, {})
        for episode in load_episodes_for_annotation(args, object_name):
            previous = object_annotations.get(episode.root.name)
            previous_frame = (
                int(previous["arm_frame_index"])
                if isinstance(previous, dict) and "arm_frame_index" in previous
                else None
            )
            if previous_frame is not None and not args.overwrite:
                print(
                    f"[annotate] keeping {object_name}/{episode.root.name} "
                    f"frame {previous_frame}; use --overwrite to revisit it."
                )
                continue
            action, frame = annotate_episode_in_viser(
                args,
                object_name,
                episode,
                previous_frame,
            )
            if action == "save":
                object_annotations[episode.root.name] = annotation_record(
                    episode, int(frame)
                )
                save_annotation_document(output, payload)
                print(
                    f"[annotate] saved {object_name}/{episode.root.name}: "
                    f"arm frame {frame} -> {output}"
                )
            elif action == "quit":
                save_annotation_document(output, payload)
                print(f"[annotate] progress saved; exiting at {object_name}/{episode.root.name}.")
                return output

    save_annotation_document(output, payload)
    print(f"[annotate] complete: {output}")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=sorting.INSPIRE_EPISODE_ROOT,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--objects",
        nargs="+",
        choices=sorting.SUPPORTED_OBJECTS,
        help="objects to annotate; defaults to all supported objects",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        help="optional numeric episode filter applied to every selected object",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="revisit episodes already present in the output JSON",
    )
    parser.add_argument("--mesh-root", type=Path, default=sorting.SORTING_MESH_ROOT)
    parser.add_argument("--preview-fps", type=float, default=50.0)
    parser.add_argument("--preview-object-max-faces", type=int, default=2_000)
    parser.add_argument("--preview-robot-link-max-faces", type=int, default=750)
    args = parser.parse_args(argv)
    if args.preview_fps <= 0:
        parser.error("--preview-fps must be positive")
    if args.episodes is not None and any(value < 0 for value in args.episodes):
        parser.error("--episodes must contain non-negative IDs")
    if args.preview_object_max_faces <= 0 or args.preview_robot_link_max_faces <= 0:
        parser.error("preview mesh face limits must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
