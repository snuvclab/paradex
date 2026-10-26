#!/usr/bin/env python3
"""Derive a CuRobo robot config for a sibling URDF from an existing template.

Two uses:

  * ``--mode filter``: keep the template's links/spheres that exist in the new
    URDF (e.g. xarm_inspire_f1_new.yml -> xarm_inspire_dftp.yml: the DFTP hand
    has the same joint names and a subset of the F1 links).
  * ``--mode spheres-from-mesh``: keep the template's arm entries and generate
    coarse collision spheres for the new hand links from their URDF collision
    meshes (e.g. xarm_inspire_f1_new.yml -> xarm_robotiq_2f85.yml).

The generated spheres are conservative bounding approximations meant for
approach planning; review them in the sphere editor before relying on them
for tight clearances.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARM_LINKS = ["link_base", "link1", "link2", "link3", "link4", "link5", "link6"]


def urdf_links_and_joints(urdf: Path):
    root = ET.parse(urdf).getroot()
    links = [l.get("name") for l in root.iter("link")]
    joints = []
    for j in root.iter("joint"):
        joints.append({"name": j.get("name"), "type": j.get("type"),
                       "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                       "mimic": j.find("mimic").get("joint") if j.find("mimic") is not None else None,
                       "limit": (j.find("limit").attrib if j.find("limit") is not None else None)})
    return links, joints


def link_mesh_spheres(urdf: Path, link: str, n: int = 3) -> list[dict]:
    """Spheres along the principal axis of the link's collision (or visual) mesh."""
    import trimesh
    root = ET.parse(urdf).getroot()
    node = next(l for l in root.iter("link") if l.get("name") == link)
    geoms = node.findall("collision") or node.findall("visual")
    points = []
    for g in geoms:
        mesh = g.find("geometry/mesh")
        origin = g.find("origin")
        if mesh is None:
            continue
        path = mesh.get("filename").replace("package://", "")
        for candidate in (urdf.parent / path, PROJECT_ROOT / "rsc" / path, Path(path)):
            if candidate.exists():
                m = trimesh.load(candidate, process=False, force="mesh")
                break
        else:
            raise FileNotFoundError(f"{link}: mesh {path}")
        scale = mesh.get("scale")
        v = np.asarray(m.vertices, float)
        if scale:
            v = v * np.asarray([float(x) for x in scale.split()])
        if origin is not None:
            xyz = np.asarray([float(x) for x in (origin.get("xyz") or "0 0 0").split()])
            rpy = np.asarray([float(x) for x in (origin.get("rpy") or "0 0 0").split()])
            from scipy.spatial.transform import Rotation
            v = v @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
        points.append(v)
    if not points:
        return []
    v = np.concatenate(points)
    center = v.mean(0)
    axis = np.linalg.svd(v - center, full_matrices=False)[2][0]
    proj = (v - center) @ axis
    lo, hi = proj.min(), proj.max()
    out = []
    for t in np.linspace(lo, hi, n) if hi - lo > 0.01 else [0.0]:
        c = center + axis * t
        sel = np.abs(proj - t) <= max((hi - lo) / (2 * max(n - 1, 1)), 0.01)
        r = float(np.linalg.norm(v[sel] - c, axis=1).max()) if sel.any() else 0.01
        out.append({"center": [round(float(x), 4) for x in c], "radius": round(min(r, 0.06), 4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--ee-link", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spheres-out", type=Path, required=True)
    ap.add_argument("--mode", choices=("filter", "spheres-from-mesh"), default="filter")
    ap.add_argument("--hand-buffer", type=float, default=-0.005)
    ap.add_argument("--asset-root", default=None,
                    help="asset_root_path relative to the repo (default: urdf parent)")
    args = ap.parse_args()
    for name in ("template", "urdf", "out", "spheres_out"):
        setattr(args, name, getattr(args, name).resolve())

    template = yaml.safe_load(args.template.read_text())
    kin = template["robot_cfg"]["kinematics"]
    template_spheres = yaml.safe_load((PROJECT_ROOT / kin["collision_spheres"]).read_text())["collision_spheres"]
    links, joints = urdf_links_and_joints(args.urdf)
    link_set = set(links)
    movable = [j for j in joints if j["type"] != "fixed"]
    independent = [j["name"] for j in movable if j["mimic"] is None]
    hand_links = [l for l in links if l not in ARM_LINKS and l != "world"]

    kin["urdf_path"] = args.urdf.name
    kin["asset_root_path"] = args.asset_root or str(args.urdf.parent.relative_to(PROJECT_ROOT))
    kin["ee_link"] = args.ee_link
    kin["link_names"] = [l for l in kin["link_names"] if l in link_set] + \
        [l for l in hand_links if l not in kin["link_names"]]
    kin["mesh_link_names"] = [l for l in links if l != "world"]

    if args.mode == "filter":
        spheres = {l: s for l, s in template_spheres.items() if l in link_set}
    else:
        spheres = {l: s for l, s in template_spheres.items() if l in ARM_LINKS}
        for l in hand_links:
            s = link_mesh_spheres(args.urdf, l)
            if s:
                spheres[l] = s
    kin["collision_link_names"] = [l for l in kin["link_names"] if l in spheres]
    kin["collision_spheres"] = str(args.spheres_out.relative_to(PROJECT_ROOT))
    kin["self_collision_buffer"] = {l: (0 if l in ARM_LINKS else args.hand_buffer)
                                    for l in kin["collision_link_names"]}
    ignore = {l: [c for c in cs if c in link_set] for l, cs in kin["self_collision_ignore"].items()
              if l in link_set}
    # every hand link ignores every other hand link and link6 (they are rigidly
    # connected or already handled by the joint limits)
    for l in hand_links:
        ignore.setdefault(l, [])
        for other in hand_links + ["link6"]:
            if other != l and other not in ignore[l]:
                ignore[l].append(other)
    kin["self_collision_ignore"] = ignore
    hand_joints = [n for n in independent if not n.startswith("joint")]
    kin["lock_joints"] = {n: 0.0 for n in hand_joints}
    cspace = kin["cspace"]
    cspace["joint_names"] = [f"joint{i}" for i in range(1, 7)] + hand_joints
    n = len(cspace["joint_names"])
    for key in ("retract_config", "null_space_weight", "cspace_distance_weight"):
        if key in cspace:
            values = list(cspace[key])
            cspace[key] = (values + [values[-1]] * n)[:n] if key != "retract_config" \
                else (values[:6] + [0.0] * (n - 6))
    args.spheres_out.parent.mkdir(parents=True, exist_ok=True)
    args.spheres_out.write_text(yaml.safe_dump({"collision_spheres": spheres}, sort_keys=False))
    args.out.write_text(yaml.safe_dump(template, sort_keys=False))
    print(f"wrote {args.out} ({len(kin['link_names'])} links, {len(spheres)} sphere links, "
          f"cspace {cspace['joint_names']})")


if __name__ == "__main__":
    main()
