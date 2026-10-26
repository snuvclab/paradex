"""MANUS keypoint DexPilot retargeting for the right Allegro V5.

Uses the running TYPE=B V5 driver's kinematics, rather than V4 calibration or
MANUS ergonomics. Commands are radians in driver order: index, middle, ring,
thumb. The pinky is unused. VIVE wrist fusion remains in the input receiver.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from paradex.retargetor.inspire_dexpilot import (
    REPO_ROOT,
    _read_exact,
    estimate_mano_wrist_frame,
    manus_frame_to_mano_keypoints,
)

URDF_PATH = REPO_ROOT / "rsc/robot/allegro_v5/allegro_right_B_kinematics.urdf"
JOINT_NAMES = tuple(f"joint_{i}_0" for i in range(16))
TIP_LINKS = ("link_15_0_tip", "link_3_0_tip", "link_7_0_tip", "link_11_0_tip")
# Wrist estimate: X toward wrist, Z toward index. V5: +Z toward fingertips,
# +Y toward index, +X toward the flexion direction. Same as OPERATOR2MANO_RIGHT.
OPERATOR2ALLEGRO_RIGHT = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]])
KEYPOINT_BYTES = 21 * 3 * 8
COMMAND_BYTES = 16 * 8
PINCH_PROJECT_DISTANCE_M = 0.02
PINCH_ESCAPE_DISTANCE_M = 0.05


def joint_limits(urdf_path=URDF_PATH):
    root = ET.parse(urdf_path).getroot()
    limits = [root.find(f"joint[@name='{name}']/limit") for name in JOINT_NAMES]
    return np.array([[float(tag.get("lower")), float(tag.get("upper"))]
                     for tag in limits])


def validate_keypoints(keypoints):
    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        raise ValueError("Allegro DexPilot requires finite MANO keypoints [21, 3]")
    # Validate before sending data to the worker, including collapsed frames.
    estimate_mano_wrist_frame(points)
    return points


class AllegroV5DexPilotRetargeter:
    """Synchronous optimizer, also usable for offline validation."""

    def __init__(self, is_right=True, scaling_factor=1.15, low_pass_alpha=0.5):
        if not is_right:
            raise ValueError("Allegro V5 DexPilot currently supports the right hand only")
        if not np.isfinite(scaling_factor) or scaling_factor <= 0:
            raise ValueError("scaling_factor must be finite and positive")
        if not np.isfinite(low_pass_alpha) or not 0 < low_pass_alpha <= 1:
            raise ValueError("low_pass_alpha must be in (0, 1]")
        try:
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as exc:
            raise RuntimeError("Allegro DexPilot requires dex-retargeting; use conda env robot") from exc
        self.is_right = True
        self.scaling_factor = float(scaling_factor)
        self.limits = joint_limits()
        self.retargeter = RetargetingConfig.from_dict({
            "type": "DexPilot",
            "urdf_path": str(URDF_PATH),
            "target_joint_names": list(JOINT_NAMES),
            "wrist_link_name": "palm_link",
            "finger_tip_link_names": list(TIP_LINKS),
            "scaling_factor": float(scaling_factor),
            "low_pass_alpha": float(low_pass_alpha),
        }).build()
        # The upstream objective hardcodes 3 cm for non-thumb pairs. Use the
        # Allegro-local variant so every pair enters projection below 2 cm.
        from dex_retargeting.seq_retarget import SeqRetargeting
        from paradex.retargetor.allegro_v5_dexpilot_optimizer import AllegroV5DexPilotOptimizer

        optimizer = AllegroV5DexPilotOptimizer(
            self.retargeter.optimizer.robot,
            list(JOINT_NAMES),
            finger_tip_link_names=list(TIP_LINKS),
            wrist_link_name="palm_link",
            scaling=self.scaling_factor,
            project_dist=PINCH_PROJECT_DISTANCE_M,
            escape_dist=PINCH_ESCAPE_DISTANCE_M,
        )
        self.retargeter = SeqRetargeting(optimizer, lp_filter=self.retargeter.filter)
        names = list(optimizer.robot.dof_joint_names)
        self.command_indices = [names.index(name) for name in JOINT_NAMES]
        self.human_indices = optimizer.target_link_human_indices
        self.fixed_qpos = np.zeros(len(optimizer.idx_pin2fixed), dtype=np.float32)
        # V5 palm_link is near the MCPs, not at the anatomical wrist. Match
        # the human middle MCP to its neutral robot counterpart so wrist-tip
        # vectors do not include an extra human palm length. Inter-tip vectors
        # (including DexPilot pinch constraints) remain unchanged.
        robot = optimizer.robot
        robot.compute_forward_kinematics(np.zeros(robot.dof, dtype=np.float64))
        self.robot_middle_mcp = robot.get_link_pose(
            robot.get_link_index("link_5_0")
        )[:3, 3].copy()

    def reference_vectors(self, keypoints):
        points = validate_keypoints(keypoints)
        local = points - points[0]
        local = local @ estimate_mano_wrist_frame(local) @ OPERATOR2ALLEGRO_RIGHT
        local[0] = local[9] - self.robot_middle_mcp / self.scaling_factor
        return local[self.human_indices[1]] - local[self.human_indices[0]]

    def retarget_keypoints(self, keypoints):
        qpos = np.asarray(self.retargeter.retarget(
            self.reference_vectors(keypoints), fixed_qpos=self.fixed_qpos))
        command = qpos[self.command_indices]
        if command.shape != (16,) or not np.all(np.isfinite(command)):
            raise RuntimeError("Allegro DexPilot returned an invalid joint target")
        return np.clip(command, self.limits[:, 0], self.limits[:, 1])

    def __call__(self, hand_pose_frame, ergonomics=None):
        return self.retarget_keypoints(manus_frame_to_mano_keypoints(hand_pose_frame))


class AsyncAllegroV5DexPilotRetargeter:
    """One solve in flight; no pipe backlog or blocking writes in the ROS loop.

    A worker result older than one second raises instead of silently returning
    a frozen target. No robot publishers are created by this class.
    """

    def __init__(self, is_right=True, scaling_factor=1.15, low_pass_alpha=0.5,
                 startup_timeout=60.0):
        if not is_right:
            raise ValueError("Allegro V5 DexPilot currently supports the right hand only")
        if not np.isfinite(scaling_factor) or scaling_factor <= 0:
            raise ValueError("scaling_factor must be finite and positive")
        if not np.isfinite(low_pass_alpha) or not 0 < low_pass_alpha <= 1:
            raise ValueError("low_pass_alpha must be in (0, 1]")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), env.get("PYTHONPATH"))))
        # A single small solve should not start a large BLAS/OpenMP thread pool.
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest = None
        self._latest_time = None
        self._pending = False
        self._closed = False
        self.process = subprocess.Popen([
            sys.executable, "-m", "paradex.retargetor.allegro_v5_dexpilot",
            str(float(scaling_factor)), str(float(low_pass_alpha)),
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, bufsize=0)
        self._reader = threading.Thread(target=self._read_results, daemon=True,
                                        name="allegro_v5_dexpilot_reader")
        self._reader.start()
        atexit.register(self.close)
        if not self._event.wait(startup_timeout):
            self.close()
            raise RuntimeError("Allegro DexPilot worker startup timed out")
        if self.process.poll() is not None or self._closed:
            self.close()
            raise RuntimeError("Allegro DexPilot worker failed during startup (see stderr)")

    def _read_results(self):
        try:
            while True:
                data = _read_exact(self.process.stdout.fileno(), COMMAND_BYTES)
                if data is None:
                    break
                command = np.frombuffer(data, dtype=np.float64).copy()
                with self._lock:
                    if np.all(np.isfinite(command)):
                        self._latest = command
                        self._latest_time = time.monotonic()
                    self._pending = False
                self._event.set()
        finally:
            self._closed = True
            self._event.set()

    def __call__(self, hand_pose_frame, ergonomics=None):
        if self._closed or self.process.poll() is not None:
            raise RuntimeError("Allegro DexPilot worker has exited")
        points = validate_keypoints(manus_frame_to_mano_keypoints(hand_pose_frame))
        with self._lock:
            if not self._pending:
                self._pending = True
                self._event.clear()
                self.process.stdin.write(np.ascontiguousarray(points).tobytes())
            latest = None if self._latest is None else self._latest.copy()
            latest_time = self._latest_time
        if latest is None or time.monotonic() - latest_time > 1.0:
            if not self._event.wait(1.0):
                raise RuntimeError("Allegro DexPilot produced no fresh command within 1s")
            with self._lock:
                latest = None if self._latest is None else self._latest.copy()
                latest_time = self._latest_time
        if latest is None or time.monotonic() - latest_time > 1.0:
            raise RuntimeError("Allegro DexPilot worker stopped producing fresh commands")
        return latest

    def close(self):
        self._closed = True
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        self._reader.join(timeout=2.0)
        process.stdout.close()
        atexit.unregister(self.close)


def _worker_main(argv):
    out_fd = os.dup(1)
    os.dup2(2, 1)  # Keep library output out of the binary protocol.
    retargeter = AllegroV5DexPilotRetargeter(
        scaling_factor=float(argv[0]), low_pass_alpha=float(argv[1]))
    os.write(out_fd, np.full(16, np.nan, dtype=np.float64).tobytes())
    while True:
        data = _read_exact(sys.stdin.fileno(), KEYPOINT_BYTES)
        if data is None:
            return
        points = np.frombuffer(data, dtype=np.float64).reshape(21, 3)
        command = retargeter.retarget_keypoints(points)
        os.write(out_fd, np.asarray(command, dtype=np.float64).tobytes())


if __name__ == "__main__":
    _worker_main(sys.argv[1:])
