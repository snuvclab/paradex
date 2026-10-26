#!/usr/bin/env python3
"""Render recorded robot/object replay and pair it with the free-object simulation."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import render_target_trajectory_videos as rv
from physics_eval import load_prediction_rows, set_free_pose, set_joint_targets
from render_zero_force_videos import _compact_mesh, transcode_h264


BASE = Path(__file__).resolve().parent


def _model_and_meshes(model_dir: Path) -> tuple[mujoco.MjModel, dict[str, Any], list[rv.HandMesh]]:
    model = mujoco.MjModel.from_xml_path(str(model_dir / "hand_physics.xml"))
    meta = json.loads((model_dir / "model_metadata.json").read_text())
    object_body = model.body(meta["object_body"]).id
    meshes = []
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 2 or int(model.geom_bodyid[geom_id]) in (0, object_body):
            continue
        vertices, faces = rv.geom_local_mesh(model, geom_id)
        meshes.append(rv.HandMesh(geom_id, vertices, faces))
    return model, meta, meshes


def _configure_floor(
    model: mujoco.MjModel,
    canonical_object_vertices: np.ndarray,
    initial_object_pose: np.ndarray,
    gravity: np.ndarray,
) -> None:
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    up = -gravity / np.linalg.norm(gravity)
    vertices = canonical_object_vertices @ initial_object_pose[:3, :3].T + initial_object_pose[:3, 3]
    bottom = float(np.min(vertices @ up))
    rotation = Rotation.align_vectors([[0.0, 0.0, 1.0]], [up])[0].as_matrix().T
    tangent_center = vertices.mean(axis=0) - up * float(vertices.mean(axis=0) @ up)
    model.geom_quat[table] = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    model.geom_pos[table] = tangent_center + up * (
        bottom - float(model.geom_size[table, 2]) - 0.001 + rv.FLOOR_RAISE_M
    )
    model.geom_sameframe[table] = 0


def _recorded_frames(
    model: mujoco.MjModel,
    meta: dict[str, Any],
    row: dict[str, Any],
    samples: list[rv.TrajectorySample],
    hand_meshes: list[rv.HandMesh],
    object_poses: dict[int, np.ndarray],
    world_from_robot: np.ndarray,
) -> list[rv.PhysicsFrame]:
    data = mujoco.MjData(model)
    object_body = model.body(meta["object_body"]).id
    frames = []
    for sample in samples:
        object_pose = rv.robot_from_world_pose(world_from_robot, object_poses[sample.source_frame])
        set_free_pose(model, data, meta["object_free_joint"], object_pose)
        set_free_pose(model, data, meta["hand_free_joint"], sample.robot_from_palm)
        set_joint_targets(model, data, meta, list(row["joint_names"]), sample.hand_qpos.tolist())
        rv.initialize_mimics(model, data)
        mujoco.mj_forward(model, data)
        frames.append(rv.PhysicsFrame(
            geom_poses={
                mesh.geom_id: (
                    np.asarray(data.geom_xpos[mesh.geom_id]).copy(),
                    np.asarray(data.geom_xmat[mesh.geom_id]).reshape(3, 3).copy(),
                )
                for mesh in hand_meshes
            },
            object_pose=object_pose,
            contact_count=0,
            self_contact_count=0,
            object_contact_count=0,
            min_self_contact_distance_m=0.0,
            self_contact_pair_depth_m={},
        ))
    return frames


def _pair_video(replay: Path, physics: Path, output: Path) -> None:
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(replay), "-i", str(physics),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-preset", "medium", "-crf", "20", str(output),
    ], check=True)


def render_row(row: dict[str, Any], out_dir: Path, model_root: Path, physics_dir: Path, fps: float) -> dict[str, Any]:
    samples, trajectory_meta, canonical_from_object, initial_object_pose, gravity = rv.load_trajectory(row, fps)
    scene = rv.scene_for_row(row)
    object_poses = rv.load_object_poses(scene)
    world_from_robot = np.load(scene / "C2R.npy")
    model_dir = model_root / str(row["pair_id"]).replace("/", "__")
    model, meta, hand_meshes = _model_and_meshes(model_dir)

    object_source = rv.OBJECT_MESH_ROOT / row["object_id"] / f"{row['object_id']}.obj"
    object_mesh = trimesh.load(object_source, force="mesh", process=False)
    canonical_vertices = np.asarray(object_mesh.vertices) @ canonical_from_object[:3, :3].T + canonical_from_object[:3, 3]
    canonical_vertices, object_faces = _compact_mesh(canonical_vertices, np.asarray(object_mesh.faces), max_faces=900)
    _configure_floor(model, canonical_vertices, initial_object_pose, gravity)
    frames = _recorded_frames(model, meta, row, samples, hand_meshes, object_poses, world_from_robot)
    basis = rv.camera_basis(gravity)
    view = rv.view_parameters(canonical_vertices, samples, frames, basis)
    floor_vertices = rv.table_top_vertices(model)

    stem = f"{row['collection']}_{str(row['pair_id']).replace('/', '__')}"
    replay_path = out_dir / f"{stem}_recorded_replay.mp4"
    raw_path = out_dir / f"{stem}_recorded_replay_mp4v_tmp.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (rv.WIDTH, rv.HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {raw_path}")
    preview = None
    last_frame = None
    for index, (sample, frame) in enumerate(zip(samples, frames)):
        image = rv.render_frame(
            row, sample, index, samples, hand_meshes, frame, canonical_vertices,
            object_faces, floor_vertices, view, basis, object_motion="recorded",
        )
        writer.write(image)
        last_frame = image
        if sample.source_frame == int(row["robot_frame"]):
            preview = image.copy()
    for _ in range(round(fps)):
        writer.write(last_frame)
    writer.release()
    encoding = transcode_h264(raw_path, replay_path)
    preview_path = replay_path.with_suffix(".png")
    cv2.imwrite(str(preview_path), preview if preview is not None else last_frame)

    physics_path = physics_dir / f"{stem}_target_trajectory.mp4"
    if not physics_path.is_file():
        raise FileNotFoundError(f"missing physics video: {physics_path}")
    paired_path = out_dir / f"{stem}_recorded_vs_physics.mp4"
    _pair_video(replay_path, physics_path, paired_path)
    return {
        "collection": row["collection"], "object_id": row["object_id"],
        "evaluation_object_id": row["evaluation_object_id"], "pair_id": row["pair_id"],
        "recorded_replay_video": str(replay_path), "physics_video": str(physics_path),
        "paired_video": str(paired_path), "preview": str(preview_path),
        "encoding": encoding, "frames": len(samples) + round(fps),
        "floor_raise_m": rv.FLOOR_RAISE_M,
        "trajectory": trajectory_meta,
        "left_panel": "kinematic visualization only: recorded wrist + recorded joints + recorded object_6d_pose_v2; no simulation step",
        "right_panel": "recorded wrist + recorded joints + free MuJoCo object initialized at first v2 pose",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=BASE / "runs/predictions.json")
    parser.add_argument("--out-dir", type=Path, default=BASE / "videos_recorded_replay_pairs")
    parser.add_argument("--model-root", type=Path, default=BASE / "videos_target_trajectory_collision_final/models")
    parser.add_argument("--physics-dir", type=Path, default=BASE / "videos_target_trajectory_collision_final")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--collections", nargs="*")
    parser.add_argument("--pair-id")
    parser.add_argument("--max-videos", type=int)
    args = parser.parse_args()
    rows, excluded = load_prediction_rows([args.predictions], per_collection=None, one_per_object=True)
    if args.collections:
        rows = [row for row in rows if row["collection"] in args.collections]
    if args.pair_id:
        rows = [row for row in rows if row["pair_id"] == args.pair_id]
    if args.max_videos is not None:
        rows = rows[:args.max_videos]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(render_row, row, args.out_dir, args.model_root, args.physics_dir, args.fps): row for row in rows}
        for index, future in enumerate(as_completed(pending), 1):
            row = pending[future]
            try:
                videos.append(future.result())
                print(f"[{index}/{len(rows)}] rendered {row['pair_id']}", flush=True)
            except Exception as exc:
                failures.append({"pair_id": row["pair_id"], "reason": f"{type(exc).__name__}: {exc}"})
                print(f"[{index}/{len(rows)}] skipped {row['pair_id']}: {exc}", flush=True)
    videos.sort(key=lambda x: (x["collection"], x["evaluation_object_id"], x["pair_id"]))
    manifest = {
        "renderer": "paired kinematic visualization without simulation (left) versus free-object MuJoCo simulation (right)",
        "videos": videos, "failures": failures, "selection_exclusions": excluded,
    }
    path = args.out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(path), "videos": len(videos), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
