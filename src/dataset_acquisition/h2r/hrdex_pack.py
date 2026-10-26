#!/usr/bin/env python3
"""Pack sim-verified HRDex grasp candidates into the real-robot replay schemas.

Reads the hrdex_contactmap benchmark outputs (per object: ``*optimized.npz`` with
``q = [xyz(3), rot6d(6), joints]`` in the object-canonical frame, plus the CPU
PhysX ``*sim*.json`` with per-candidate outcomes), ranks the candidates that
passed both success metrics, and writes per-object/scene result files that the
existing real runners consume unchanged:

  inspire_dftp : result.npz          -> capture_object6d.py (--hand_type inspire_dftp)
                 <src>_pregrasp_traj (K, W, 6+12), <src>_final_q (K, 21)
  allegro_v5   : result_allegro.npz  -> capture_object6d_allegro_v5.py
                 <src>_traj_q_all (K, W, 6+16), <src>_best_particle = 0
  robotiq_2f85 : result_robotiq.npz  -> capture_object6d_robotiq.py
                 <src>_pregrasp_traj (K, W, 6+6 mimic joints), <src>_gripper_cmd (K, W) in 0..1

``<src>`` is ``human`` (Human-Contact condition) or ``robot`` (Transferred).
``scene`` is the human capture episode of the held-out pose, so the same
(object, scene) names the same test pose across embodiments.

The wrist in every schema is the hand URDF root used by the simulator, as
``[x, y, z, euler XYZ]`` (scipy intrinsic 'XYZ', CEDex convention).  Real runners
compose it with the RC object pose (``pose_robot @ wrist``); the canonical frame
equals ``mesh_blender/<obj>/<obj>_viser.obj`` (verified chamfer < 1 mm).
For Robotiq the simulator root is ``robotiq_85_mount_link`` = base_link rotated
by pi about x; the runner applies that fixed offset when it targets base_link.

Pregrasp motion: straight-line wrist retreat of ``--retreat-m`` away from the
object centroid (rotation held), fingers fully open for the first
``--open-fraction`` of the path then linearly closing to the grasp joints.

Run on the machine that holds hrdex_contactmap; the output lands on shared_data.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HRDEX = Path(os.environ.get("HRDEX_CONTACTMAP_ROOT", "/home/capture16/hrdex_contactmap"))
SHARED = Path(os.environ.get("HRDEX_SHARED_DATA", "/home/capture16/shared_data"))
DEFAULT_OUT = SHARED / "capture/h2r/hrdex_sim_verified"
MESH_BLENDER = SHARED / "mesh_blender"

# Benchmarks: per embodiment a list of (condition, case-dir glob, optimized name, sim name).
BENCHMARKS = {
    "legacy40": {
        "description": "40 held-out poses x 25 particles (2026-09-11/12); DFTP/Robotiq sims at "
                       "hold 0.3 s / settle 1.0 s, Allegro at hold 0.5 s / settle 1.0 s",
        "split": None,
        "inspire_dftp": {
            "human": ("runs/f1_recipe_dftp_A_40objects_1000_hold0p7_settle1_v4/[0-9][0-9]_*",
                      "human_a_optimized.npz", "human_a_sim_batch_cpu25_hold0p3_settle1.json"),
            "robot": ("runs/f1_recipe_dftp_transferred_A_40objects_1000_local_refiner_hold0p7_settle1/[0-9][0-9]_*",
                      "transferred_a_optimized.npz", "transferred_a_sim_batch_cpu25_hold0p3_settle1.json"),
        },
        "robotiq_2f85": {
            "human": ("runs/robotiq_v4_then_dftp_cpu_4000_sequence/robotiq_2f85/human/[0-9][0-9]_*",
                      "human_hybrid_optimized.npz", "human_hybrid_cpu25_sim_batch_cpu25_hold0p3_settle1.json"),
            "robot": ("runs/robotiq_v4_then_dftp_cpu_4000_sequence/robotiq_2f85/transferred/[0-9][0-9]_*",
                      "transferred_hybrid_optimized.npz", "transferred_hybrid_cpu25_sim_batch_cpu25_hold0p3_settle1.json"),
        },
        "allegro_v5": {
            "human": ("runs/legacy_f1_allegro_40objects_25particles_hold0p5_settle1_cachev1/allegro/human/[0-9][0-9]_*",
                      "optimized.npz", "simulation.json"),
            "robot": ("runs/legacy_f1_allegro_40objects_25particles_hold0p5_settle1_cachev1/allegro/transferred/[0-9][0-9]_*",
                      "optimized.npz", "simulation.json"),
        },
    },
    "unified34": {
        "description": "34 seen-object/unseen-pose test poses shared by all embodiments, 1000 trials "
                       "per condition, hold 0.5 s / settle 1.0 s, retrained contact maps",
        "split": "splits/unified_seen_object_unseen_pose_34.json",
        "inspire_dftp": {
            "human": ("runs/unified34_benchmark/dftp_human/[0-9][0-9]_*",
                      "human_a_optimized.npz", "human_a_sim_batch_cpu30_hold0p5_settle1.json"),
            "robot": ("runs/unified34_benchmark/dftp_transferred/[0-9][0-9]_*",
                      "transferred_a_optimized.npz", "transferred_a_sim_batch_cpu30_hold0p5_settle1.json"),
        },
        "robotiq_2f85": {
            "human": ("runs/unified34_benchmark/robotiq_human/[0-9][0-9]_*",
                      "human_hybrid_optimized.npz", "human_hybrid_cpu25_sim_batch_cpu30_hold0p5_settle1.json"),
            "robot": ("runs/unified34_benchmark/robotiq_transferred/[0-9][0-9]_*",
                      "transferred_hybrid_optimized.npz", "transferred_hybrid_cpu25_sim_batch_cpu30_hold0p5_settle1.json"),
        },
        "allegro_v5": {
            "human": ("runs/unified34_benchmark/f1_allegro/allegro/human/[0-9][0-9]_*",
                      "optimized.npz", "simulation.json"),
            "robot": ("runs/unified34_benchmark/f1_allegro/allegro/transferred/[0-9][0-9]_*",
                      "optimized.npz", "simulation.json"),
        },
    },
}

EMBODIMENT = {
    "inspire_dftp": {"result": "result.npz", "n_joints": 12, "open": None},
    "allegro_v5": {"result": "result_allegro.npz", "n_joints": 16, "open": None},
    "robotiq_2f85": {"result": "result_robotiq.npz", "n_joints": 6, "open": None},
}
ROBOTIQ_DRIVE = "robotiq_85_left_knuckle_joint"
ROBOTIQ_MAX_RAD = 0.8
SUCCESS_DISPLACEMENT_M = 0.02


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """CEDex convention: the two 3-vectors are the first two columns of R."""
    x = rot6d[:3] / np.linalg.norm(rot6d[:3])
    y = rot6d[3:6] / np.linalg.norm(rot6d[3:6])
    a = x + y
    a /= np.linalg.norm(a)
    b = x - y
    b /= np.linalg.norm(b)
    x = (a + b) / np.linalg.norm(a + b)
    y = (a - b) / np.linalg.norm(a - b)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def wrist_euler(q: np.ndarray) -> np.ndarray:
    """(xyz, rot6d, ...) -> (x, y, z, euler XYZ)."""
    R = rot6d_to_matrix(np.asarray(q[3:9], np.float64))
    return np.r_[q[:3], Rotation.from_matrix(R).as_euler("XYZ")].astype(np.float32)


def force_only_success(row: dict) -> bool:
    end = np.asarray(row["object_state_at_end"][:3], float)
    start = np.asarray(row["object_state_at_force_start"][:3], float)
    return bool(np.linalg.norm(end - start) <= SUCCESS_DISPLACEMENT_M)


def rank_candidates(sim: dict, energy: np.ndarray, top_k: int) -> list[dict]:
    rows = []
    for row in sim["results"]:
        idx = int(row["candidate"])
        cum = bool(row["success"])
        force = force_only_success(row)
        axes = row.get("force_axis_displacements_m", {})
        rows.append({
            "candidate": idx, "cumulative_success": cum, "force_only_success": force,
            "settle_peak_mm": 1e3 * float(row.get("post_close_settle_peak_displacement_m", np.nan)),
            "close_peak_mm": 1e3 * float(row.get("close_peak_displacement_m", np.nan)),
            "worst_axis_mm": 1e3 * float(max(axes.values())) if axes else float("nan"),
            "energy": float(energy[idx]),
        })
    passed = [r for r in rows if r["cumulative_success"] and r["force_only_success"]]
    passed.sort(key=lambda r: (r["settle_peak_mm"], r["worst_axis_mm"], r["energy"]))
    return passed[:top_k], rows


def load_case(case_dir: Path, optimized_name: str, sim_name: str):
    optimized = np.load(case_dir / optimized_name, allow_pickle=True)
    sim = json.loads((case_dir / sim_name).read_text())
    if len(optimized["q"]) != sim["trials"]:
        raise ValueError(f"{case_dir}: {len(optimized['q'])} particles vs {sim['trials']} trials")
    return optimized, sim


def object_centroid(object_id: str) -> np.ndarray:
    cloud = HRDEX / "hrdexdb_confirmed_pair_superset/objects/v0" / f"{object_id}.npz"
    with np.load(cloud) as z:
        return np.asarray(z["points"], np.float64).mean(axis=0)


def pregrasp_trajectory(q: np.ndarray, open_joints: np.ndarray, centroid: np.ndarray,
                        waypoints: int, retreat_m: float, open_fraction: float) -> np.ndarray:
    """(W, 6 + J): straight retreat -> grasp; fingers open then close."""
    grasp = wrist_euler(q)
    joints = np.asarray(q[9:], np.float32)
    direction = grasp[:3] - centroid
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-6 else np.array([0, 0, 1.0])
    start = grasp.copy()
    start[:3] = grasp[:3] + retreat_m * direction
    alphas = np.linspace(0.0, 1.0, waypoints, dtype=np.float32)
    traj = np.zeros((waypoints, 6 + len(joints)), np.float32)
    for i, a in enumerate(alphas):
        traj[i, :6] = (1 - a) * start + a * grasp
        close = 0.0 if a <= open_fraction else (a - open_fraction) / max(1e-6, 1 - open_fraction)
        traj[i, 6:] = (1 - close) * open_joints + close * joints
    return traj


def verify_mesh_frame(object_id: str) -> dict:
    """Canonical sim cloud vs the real pose-estimation mesh (<obj>_viser.obj)."""
    mesh_path = MESH_BLENDER / object_id / f"{object_id}_viser.obj"
    if not mesh_path.exists():
        return {"viser_mesh": str(mesh_path), "exists": False}
    try:
        import trimesh
        from scipy.spatial import cKDTree
        mesh = trimesh.load(mesh_path, process=False, force="mesh")
        with np.load(HRDEX / "hrdexdb_confirmed_pair_superset/objects/v0" / f"{object_id}.npz") as z:
            cloud = np.asarray(z["points"], np.float64)
        sample = mesh.sample(8192)
        chamfer = float(cKDTree(sample).query(cloud)[0].mean())
        return {"viser_mesh": str(mesh_path), "exists": True, "chamfer_m": chamfer,
                "frame_ok": chamfer < 0.003}
    except Exception as exc:  # trimesh missing on a bare machine
        return {"viser_mesh": str(mesh_path), "exists": True, "error": str(exc)}


def pack_object(embodiment: str, object_id: str, scene: str, conditions: dict,
                out_dir: Path, args, split_entry: dict | None) -> dict:
    spec = EMBODIMENT[embodiment]
    out = {}
    manifest = {"embodiment": embodiment, "object": object_id, "scene": scene,
                "benchmark": args.benchmark, "created_utc": datetime.datetime.utcnow().isoformat(),
                "pregrasp": {"waypoints": args.waypoints, "retreat_m": args.retreat_m,
                             "open_fraction": args.open_fraction,
                             "method": "straight retreat from object centroid; rotation held"},
                "selection": {"rule": "cumulative and force-only success, then min settle slip, "
                                      "min worst-axis displacement, min energy",
                              "top_k": args.top_k},
                "conditions": {}, "mesh_frame": verify_mesh_frame(object_id)}
    if split_entry:
        manifest["split_entry"] = split_entry
    centroid = object_centroid(object_id)
    joint_names = None
    for src, (case_dir, optimized_name, sim_name) in conditions.items():
        optimized, sim = load_case(case_dir, optimized_name, sim_name)
        names = [str(x) for x in optimized["joint_names"]]
        if embodiment == "allegro_v5":  # simulator 'joint_3.0' -> Paradex URDF 'joint_3_0'
            names = [n.replace(".0", "_0") for n in names]
        if joint_names is None:
            joint_names = names
        elif names != joint_names:
            raise ValueError(f"joint names differ between conditions: {case_dir}")
        q_all = np.asarray(optimized["q"], np.float32)
        selected, all_rows = rank_candidates(sim, np.asarray(optimized["energy"], float), args.top_k)
        if not selected:
            manifest["conditions"][src] = {"case_dir": str(case_dir), "selected": [],
                                           "note": "no candidate passed both success metrics"}
            continue
        idx = [r["candidate"] for r in selected]
        open_joints = np.zeros(len(names), np.float32)
        trajs = np.stack([pregrasp_trajectory(q_all[i], open_joints, centroid, args.waypoints,
                                              args.retreat_m, args.open_fraction) for i in idx])
        out[f"{src}_final_q"] = q_all[idx]
        out[f"{src}_final_energy"] = np.asarray(optimized["energy"], np.float32)[idx]
        out[f"{src}_candidate_index"] = np.asarray(idx, np.int64)
        out[f"{src}_sim_succ"] = np.ones(len(idx), bool)
        if embodiment == "allegro_v5":
            out[f"{src}_traj_q_all"] = trajs
            out[f"{src}_traj_q"] = trajs[0]
            out[f"{src}_best_particle"] = np.int64(0)
            if "legacy_final_q" in optimized.files:
                out[f"{src}_legacy_final_q"] = np.asarray(optimized["legacy_final_q"], np.float32)[idx]
        else:
            out[f"{src}_pregrasp_traj"] = trajs
        if embodiment == "robotiq_2f85":
            drive = names.index(ROBOTIQ_DRIVE)
            out[f"{src}_gripper_cmd"] = np.clip(trajs[:, :, 6 + drive] / ROBOTIQ_MAX_RAD, 0, 1)
        manifest["conditions"][src] = {
            "case_dir": str(case_dir), "optimized": optimized_name, "sim": sim_name,
            "trials": sim["trials"], "successes": sim["successes"],
            "selected": selected, "all_candidates": all_rows,
            "sim_protocol": {k: sim.get("protocol", {}).get(k) for k in
                             ("hold_object_for_close_seconds", "post_close_settle_seconds",
                              "friction", "gravity_z", "drive_effort_limit_nm")},
        }
    if not out:
        return manifest
    out["revolute_joint_names"] = np.asarray(joint_names)
    out["object_name"] = np.asarray(object_id)
    out["scene"] = np.asarray(scene)
    out["embodiment"] = np.asarray(embodiment)
    out["wrist_frame"] = np.asarray("hand URDF root of the simulator, in the viser-aligned object canonical frame; "
                                    "euler XYZ intrinsic (scipy)")
    if embodiment == "robotiq_2f85":
        out["wrist_root_link"] = np.asarray("robotiq_85_mount_link")
        out["base_link_from_root_rpy"] = np.asarray([np.pi, 0.0, 0.0], np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / spec["result"], **out)
    (out_dir / "candidates.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    return manifest


def discover_cases(embodiment: str, benchmark: dict) -> dict:
    """object_id -> {src: (case_dir, optimized, sim)}"""
    cases = {}
    for src, (pattern, optimized_name, sim_name) in benchmark[embodiment].items():
        for case_dir in sorted(glob.glob(str(HRDEX / pattern))):
            case_dir = Path(case_dir)
            if not (case_dir / optimized_name).exists() or not (case_dir / sim_name).exists():
                continue
            object_id = case_dir.name.split("_", 1)[1]
            cases.setdefault(object_id, {})[src] = (case_dir, optimized_name, sim_name)
    return cases


def scene_lookup(benchmark: dict, embodiment: str, cases: dict) -> tuple[dict, dict]:
    """object -> human episode (scene) from the unified split or the run manifests."""
    scenes, entries = {}, {}
    if benchmark["split"]:
        split = json.loads((HRDEX / benchmark["split"]).read_text())
        for entry in split["entries"]:
            scenes[entry["object_id"]] = str(entry["human_episode"])
            entries[entry["object_id"]] = entry
        return scenes, entries
    superset = {r["pair_id"]: r for r in json.loads(
        (HRDEX / "hrdexdb_confirmed_pair_superset/index.json").read_text())["pairs"]}
    reference = json.loads((HRDEX / "runs/f1_recipe_dftp_transferred_A_40objects_1000_local_refiner_hold0p7_settle1/manifest.json").read_text())
    for entry in reference["entries"]:
        row = superset[entry["pair_id"]]
        scenes[entry["object_id"]] = str(row["human_episode"])
        entries[entry["object_id"]] = {"index": entry["index"], "object_id": entry["object_id"],
                                       "dftp_pair_id": entry["pair_id"],
                                       "human_episode": str(row["human_episode"]),
                                       "human_frame": int(row["human_frame"])}
    return scenes, entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=sorted(BENCHMARKS), required=True)
    ap.add_argument("--embodiments", default="inspire_dftp,allegro_v5,robotiq_2f85")
    ap.add_argument("--objects", nargs="*", help="restrict to these objects")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--waypoints", type=int, default=25)
    ap.add_argument("--retreat-m", type=float, default=0.10)
    ap.add_argument("--open-fraction", type=float, default=0.6)
    ap.add_argument("--copy-sources", action="store_true",
                    help="also copy the optimized/sim source files next to each result")
    args = ap.parse_args()
    benchmark = BENCHMARKS[args.benchmark]
    root = args.out_root / args.benchmark
    root.mkdir(parents=True, exist_ok=True)
    summary = {"benchmark": args.benchmark, "description": benchmark["description"],
               "hrdex_root": str(HRDEX), "created_utc": datetime.datetime.utcnow().isoformat(),
               "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "embodiments": {}}
    for embodiment in args.embodiments.split(","):
        cases = discover_cases(embodiment, benchmark)
        scenes, entries = scene_lookup(benchmark, embodiment, cases)
        packed, skipped = [], []
        for object_id, conditions in sorted(cases.items()):
            if args.objects and object_id not in args.objects:
                continue
            if object_id not in scenes:
                skipped.append({"object": object_id, "reason": "not in test split"})
                continue
            out_dir = root / embodiment / object_id / scenes[object_id]
            manifest = pack_object(embodiment, object_id, scenes[object_id], conditions,
                                   out_dir, args, entries.get(object_id))
            counts = {src: len(c.get("selected", [])) for src, c in manifest["conditions"].items()}
            if args.copy_sources and any(counts.values()):
                for src, (case_dir, optimized_name, sim_name) in conditions.items():
                    for name in (optimized_name, sim_name):
                        shutil.copy2(case_dir / name, out_dir / f"{src}__{name}")
            packed.append({"object": object_id, "scene": scenes[object_id],
                           "selected": counts, "mesh_frame_ok": manifest["mesh_frame"].get("frame_ok")})
            print(f"[{embodiment}] {object_id}/{scenes[object_id]} selected {counts} "
                  f"mesh_frame={manifest['mesh_frame'].get('frame_ok')}", flush=True)
        summary["embodiments"][embodiment] = {"packed": packed, "skipped": skipped,
                                              "result_file": EMBODIMENT[embodiment]["result"]}
    (root / "pack_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"wrote {root / 'pack_summary.json'}")


if __name__ == "__main__":
    main()
