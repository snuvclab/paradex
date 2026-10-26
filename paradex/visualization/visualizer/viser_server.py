"""Long-running viser server.

Bootstraps a ViserViewer with a single robot URDF, then listens on a ZMQ REP
socket for incoming commands (object swap, trajectory replace, qpos set, etc).
Lets capture scripts attach to an already-running viewer instead of paying the
viser startup cost each run.

Usage:
    python -m paradex.visualization.visualizer.viser_server \
        --robot_urdf rsc/robot/xarm_allegro_v5.urdf \
        --addr tcp://*:5572
"""
from __future__ import annotations

import argparse
import pickle
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh
import zmq

from paradex.visualization.visualizer.viser import ViserViewer


def _make_mesh(vertices, faces, vertex_colors=None) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if vertex_colors is not None:
        mesh.visual.vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)
    return mesh


class ViserServer:
    """Dispatches ZMQ messages to a ViserViewer running in this process."""

    def __init__(self, vis: ViserViewer, max_mesh_faces: Optional[int] = 1000):
        self.vis = vis
        self.max_mesh_faces = max_mesh_faces
        self._lock = threading.Lock()

    # ---- command handlers ------------------------------------------------
    def cmd_ping(self, _msg: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "msg": "pong"}

    def cmd_set_object(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        name = str(msg["name"])
        pose = np.asarray(msg["pose"], dtype=np.float32)
        mesh = _make_mesh(msg["vertices"], msg["faces"], msg.get("vertex_colors"))
        opacity = float(msg.get("opacity", 1.0))
        with self._lock:
            if name in self.vis.obj_dict:
                # Replace by removing then re-adding so geometry/pose both update.
                try:
                    self.vis.obj_dict[name]["frame"].remove()
                except Exception:
                    pass
                try:
                    self.vis.obj_dict[name]["handle"].remove()
                except Exception:
                    pass
                del self.vis.obj_dict[name]
                self.vis.frame_nodes.pop(name, None)
            self.vis.add_object(
                name, mesh, pose, opacity=opacity,
                max_mesh_faces=self.max_mesh_faces,
            )
        return {"ok": True}

    def cmd_set_qpos(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        name = str(msg.get("name", "robot"))
        qpos = np.asarray(msg["qpos"], dtype=np.float32)
        with self._lock:
            if name not in self.vis.robot_dict:
                return {"ok": False, "msg": f"no robot named '{name}'"}
            self.vis.robot_dict[name].update_cfg(qpos)
        return {"ok": True}

    @staticmethod
    def _set_robot_visible(robot, visible: bool) -> None:
        for attr in ("_visual_root_frame", "_collision_root_frame"):
            frame = getattr(robot, attr, None)
            if frame is not None:
                frame.visible = bool(visible)

    def _show_only_robot(self, name: str) -> Dict[str, Any]:
        if name not in self.vis.robot_dict:
            return {"ok": False, "msg": f"no robot named '{name}'"}
        for robot_name, robot in self.vis.robot_dict.items():
            self._set_robot_visible(robot, robot_name == name)
        return {"ok": True}

    def cmd_show_only_robot(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        name = str(msg.get("name", "robot"))
        with self._lock:
            return self._show_only_robot(name)

    def cmd_set_static_robot(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        name = str(msg["name"])
        urdf_path = str(Path(msg["urdf_path"]).expanduser())
        pose = np.asarray(msg["pose"], dtype=np.float32)
        qpos = np.asarray(msg["qpos"], dtype=np.float32)
        if not Path(urdf_path).is_file():
            return {"ok": False, "msg": f"robot URDF not found: {urdf_path}"}
        if pose.shape != (4, 4):
            return {"ok": False, "msg": f"robot pose must be 4x4, got {pose.shape}"}
        with self._lock:
            if name not in self.vis.robot_dict:
                self.vis.add_robot(
                    name,
                    urdf_path,
                    pose=pose,
                    max_mesh_faces=self.max_mesh_faces,
                )
            robot = self.vis.robot_dict[name]
            root = getattr(robot, "_visual_root_frame", None)
            if root is not None:
                from scipy.spatial.transform import Rotation

                root.position = pose[:3, 3]
                root.wxyz = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            robot.update_cfg(qpos)
            self._set_robot_visible(robot, True)
            if bool(msg.get("show_only", False)):
                self._show_only_robot(name)
        return {"ok": True}

    def cmd_set_traj(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Replace any existing trajectory with this one."""
        name = str(msg.get("name", "traj"))
        robot_name = str(msg.get("robot_name", "robot"))
        traj = np.asarray(msg["traj"], dtype=np.float32)
        robot_pose_traj = {
            str(key): np.asarray(value, dtype=np.float32)
            for key, value in msg.get("robot_pose_traj", {}).items()
        }
        with self._lock:
            # Reset trajectory state then add fresh.
            self.vis.traj_list = []
            self.vis.num_frames = 0
            try:
                self.vis.gui_timestep.max = 0
                self.vis.gui_timestep.value = 0
            except Exception:
                pass
            self.vis.add_traj(
                name,
                robot_traj={robot_name: traj},
                robot_pose_traj=robot_pose_traj,
            )
        return {"ok": True, "n_frames": int(traj.shape[0])}

    def cmd_clear_objects(self, _msg: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            for name in list(self.vis.obj_dict.keys()):
                try:
                    self.vis.obj_dict[name]["frame"].remove()
                except Exception:
                    pass
                try:
                    self.vis.obj_dict[name]["handle"].remove()
                except Exception:
                    pass
                del self.vis.obj_dict[name]
                self.vis.frame_nodes.pop(name, None)
        return {"ok": True}

    # ---- ZMQ loop --------------------------------------------------------
    def serve(self, addr: str):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(addr)
        print(f"[viser_server] listening on {addr}")
        handlers = {
            "ping": self.cmd_ping,
            "set_object": self.cmd_set_object,
            "set_qpos": self.cmd_set_qpos,
            "set_static_robot": self.cmd_set_static_robot,
            "show_only_robot": self.cmd_show_only_robot,
            "set_traj": self.cmd_set_traj,
            "clear_objects": self.cmd_clear_objects,
        }
        while True:
            try:
                raw = sock.recv()
                msg = pickle.loads(raw)
                cmd = str(msg.get("cmd", ""))
                handler = handlers.get(cmd)
                if handler is None:
                    resp = {"ok": False, "msg": f"unknown cmd '{cmd}'"}
                else:
                    resp = handler(msg)
            except Exception as e:
                resp = {"ok": False, "msg": f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()}
                print(f"[viser_server] error: {e}\n{traceback.format_exc()}")
            try:
                sock.send(pickle.dumps(resp))
            except Exception as e:
                print(f"[viser_server] send error: {e}")


def _update_loop(vis: ViserViewer, hz: float):
    period = 1.0 / max(1.0, float(hz))
    while True:
        t0 = time.perf_counter()
        try:
            vis.update()
        except Exception as e:
            print(f"[viser_server] update error: {e}")
        sleep = period - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_urdf", required=True)
    ap.add_argument("--robot_name", default="robot")
    ap.add_argument("--addr", default="tcp://*:5572")
    ap.add_argument("--update_hz", type=float, default=30.0)
    ap.add_argument("--max_mesh_faces", type=int, default=1000,
                    help="per-mesh face budget for viewer rendering; 0 disables simplification")
    args = ap.parse_args()

    urdf = str(Path(args.robot_urdf).expanduser())
    if not Path(urdf).exists():
        raise FileNotFoundError(urdf)

    vis = ViserViewer()
    vis.add_floor(height=0.0)
    max_mesh_faces = args.max_mesh_faces if args.max_mesh_faces > 0 else None
    vis.add_robot(args.robot_name, urdf, max_mesh_faces=max_mesh_faces)
    print(f"[viser_server] loaded robot '{args.robot_name}' from {urdf}")

    threading.Thread(target=_update_loop, args=(vis, args.update_hz), daemon=True).start()
    ViserServer(vis, max_mesh_faces=max_mesh_faces).serve(args.addr)


if __name__ == "__main__":
    main()
