"""AnyTeleop-style (DexPilot) live retargeting for the Inspire RH56DFTP hand.

MANUS keypoints are converted to the 21-point MANO layout, expressed in the
DFTP wrist frame, and fed to dex-retargeting's DexPilot optimizer.  The
resulting joint angles are converted to RH56 ANGLE_SET register values.
Only thumb bend is blended with the MANUS thumb IP angle (default 70%).
Set thumb_bend_weight=0 to retain the original DexPilot output.

Both gloves are retargeted with the right DFTP model; the RH56 register
semantics are identical for both hands.  MANUS world space is mirror-handed
relative to the DFTP model (measured: fingers flex along -(across x forward) on
the right glove), so the RIGHT glove is mirrored into right-hand geometry and
the left glove already has it.

The URDF fingertip specs, wrist-frame estimate, and qpos<->raw calibration are
kept in sync with ``src/object6d/replay_human_dftp.py``.
"""

from __future__ import annotations

import atexit
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
XARM_INSPIRE_DFTP_URDF = REPO_ROOT / "rsc/robot/xarm_inspire_DFTP.urdf"

# The physical DFTP URDF extends the fingers along local -Z with local +X
# toward the index finger.
OPERATOR2INSPIRE_DFTP_RIGHT = np.array(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
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

# MANO 21-keypoint order built from MANUS frame names.  MANUS "proximal" nodes
# sit on the finger knuckles (MANO MCP joints).
MANUS_TO_MANO_KEYPOINTS = (
    "wrist",
    "thumb_metacarpal",
    "thumb_proximal",
    "thumb_distal",
    "thumb_tip",
    *(
        f"{finger}_{joint}"
        for finger in ("index", "middle", "ring", "pinky")
        for joint in ("proximal", "intermediate", "distal", "tip")
    ),
)


def thumb_bend_raw_from_keypoints(keypoints, full_bend_deg=60.0):
    """Map thumb IP bend to raw (1000=open, 0=closed); None if invalid.

    MANO points 2/3/4 are proximal/distal/tip. Bone directions make this
    independent of wrist pose, hand size, and left/right reflection.
    """
    bones = np.diff(np.asarray(keypoints, dtype=np.float64)[2:5], axis=0)
    lengths = np.linalg.norm(bones, axis=1)
    if not np.all(np.isfinite(bones)) or np.any(lengths < 1e-7):
        return None
    cosine = np.dot(bones[0] / lengths[0], bones[1] / lengths[1])
    bend_deg = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return float(1000.0 * (1.0 - np.clip(bend_deg / full_bend_deg, 0.0, 1.0)))


def _validate_thumb_blend(weight, full_bend_deg):
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("thumb_bend_weight must be finite and in [0, 1]")
    if not np.isfinite(full_bend_deg) or not 0.0 < full_bend_deg <= 180.0:
        raise ValueError("thumb_full_bend_deg must be finite and in (0, 180]")


def manus_frame_to_mano_keypoints(hand_pose_frame, is_right=True):
    """Return [21, 3] MANO-ordered positions in right-hand geometry.

    MANUS world space is mirror-handed, so the right glove is reflected and
    the left glove is used as-is.
    """
    missing = [name for name in MANUS_TO_MANO_KEYPOINTS if name not in hand_pose_frame]
    if missing:
        raise ValueError(
            "MANUS frame is missing DexPilot keypoints: " + ", ".join(missing)
        )
    keypoints = np.asarray(
        [
            np.asarray(hand_pose_frame[name], dtype=np.float64)[:3, 3]
            for name in MANUS_TO_MANO_KEYPOINTS
        ]
    )
    if is_right:
        keypoints[:, 0] *= -1.0
    return keypoints


def estimate_mano_wrist_frame(keypoints):
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
    return np.stack((x_axis, normal, z_axis), axis=1)


def derive_inspire_dftp_retarget_urdf(source, output):
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


def dftp_raw_to_hand_qpos(raw_commands):
    """Map RH56 raw registers [T, 6] to qpos (thumb yaw, thumb pitch, index..little)."""
    raw = np.asarray(raw_commands, dtype=np.float64)
    qpos = np.zeros_like(raw)
    qpos[:, 0] = 1.40 * (1.0 - raw[:, 5] / 1000.0)
    qpos[:, 1] = 0.60 * (1.0 - raw[:, 4] / 1000.0)

    def long_finger(values):
        degrees = -4e-8 * values**3 + 3e-5 * values**2 - 0.0704 * values + 83.572
        return np.deg2rad(degrees)

    qpos[:, 2] = long_finger(raw[:, 3])
    qpos[:, 3] = long_finger(raw[:, 2])
    qpos[:, 4] = long_finger(raw[:, 1])
    qpos[:, 5] = long_finger(raw[:, 0])
    return qpos


class _QposToRaw:
    """Invert dftp_raw_to_hand_qpos with a precomputed lookup curve."""

    _QPOS_TO_RAW_INDEX = (5, 4, 3, 2, 1, 0)

    def __init__(self):
        self.raw_axis = np.arange(1001, dtype=np.float64)
        calibration = dftp_raw_to_hand_qpos(
            np.repeat(self.raw_axis[:, None], 6, axis=1)
        )
        self.curves = []
        for qpos_index in range(6):
            curve = calibration[:, qpos_index]
            order = np.argsort(curve)
            self.curves.append((curve[order], self.raw_axis[order]))

    def __call__(self, qpos):
        raw = np.empty(6, dtype=np.float64)
        for qpos_index, raw_index in enumerate(self._QPOS_TO_RAW_INDEX):
            curve, axis = self.curves[qpos_index]
            raw[raw_index] = np.interp(qpos[qpos_index], curve, axis)
        return np.clip(raw, 0.0, 1000.0)


class InspireDftpDexPilotRetargeter:
    """Callable hand retargeter: MANUS frame -> RH56 ANGLE_SET command."""

    def __init__(
        self,
        is_right=True,
        scaling_factor=1.15,
        low_pass_alpha=0.5,
        urdf_path=XARM_INSPIRE_DFTP_URDF,
        thumb_bend_weight=0.7,
        thumb_full_bend_deg=60.0,
    ):
        _validate_thumb_blend(thumb_bend_weight, thumb_full_bend_deg)
        self.thumb_bend_weight = float(thumb_bend_weight)
        self.thumb_full_bend_deg = float(thumb_full_bend_deg)
        self._thumb_bend_raw = None
        self._thumb_filter_alpha = float(low_pass_alpha)
        if not 0.0 <= self._thumb_filter_alpha <= 1.0:
            self._thumb_filter_alpha = 1.0  # Same filter-disable rule as DexPilot.
        try:
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as exc:
            raise RuntimeError(
                "dex-retargeting is unavailable; DexPilot Inspire retargeting "
                "needs the robot conda environment"
            ) from exc

        self.is_right = bool(is_right)
        self._urdf_dir = tempfile.TemporaryDirectory(prefix="paradex_dftp_dexpilot_")
        derived_urdf = derive_inspire_dftp_retarget_urdf(
            urdf_path, Path(self._urdf_dir.name) / "inspire_dftp_anyteleop.urdf"
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
        self.retargeter = config.build()
        optimizer = self.retargeter.optimizer
        dof_names = list(optimizer.robot.dof_joint_names)
        self.command_indices = [
            dof_names.index(name) for name in INSPIRE_DFTP_RETARGET_JOINTS
        ]
        self.human_indices = optimizer.target_link_human_indices
        self.fixed_qpos = np.zeros(len(optimizer.idx_pin2fixed), dtype=np.float32)
        robot = optimizer.robot
        robot.compute_forward_kinematics(np.zeros(robot.dof, dtype=np.float64))
        self.base_rotation_wrist = robot.get_link_pose(
            robot.get_link_index("wrist")
        )[:3, :3].copy()
        self.qpos_to_raw = _QposToRaw()

    def reference_vectors(self, keypoints):
        local = keypoints - keypoints[0]
        rotation = estimate_mano_wrist_frame(local) @ OPERATOR2INSPIRE_DFTP_RIGHT
        local = local @ rotation
        return (
            local[self.human_indices[1]] - local[self.human_indices[0]]
        ) @ self.base_rotation_wrist.T

    def retarget_keypoints(self, keypoints):
        """Retarget right-hand (already mirrored) MANO keypoints [21, 3]."""
        qpos = np.asarray(
            self.retargeter.retarget(
                self.reference_vectors(keypoints), fixed_qpos=self.fixed_qpos
            )
        )
        command = self.qpos_to_raw(qpos[self.command_indices])
        if self.thumb_bend_weight > 0.0:
            glove_bend = thumb_bend_raw_from_keypoints(
                keypoints, self.thumb_full_bend_deg
            )
            if glove_bend is not None:
                if self._thumb_bend_raw is None:
                    self._thumb_bend_raw = glove_bend
                else:
                    self._thumb_bend_raw += self._thumb_filter_alpha * (
                        glove_bend - self._thumb_bend_raw
                    )
                command[4] = (
                    (1.0 - self.thumb_bend_weight) * command[4]
                    + self.thumb_bend_weight * self._thumb_bend_raw
                )
        return command

    def __call__(self, hand_pose_frame, ergonomics=None):
        del ergonomics  # DexPilot uses keypoints only.
        return self.retarget_keypoints(
            manus_frame_to_mano_keypoints(hand_pose_frame, self.is_right)
        )


_KEYPOINT_BYTES = 21 * 3 * 8
_COMMAND_BYTES = 6 * 8


def _read_exact(fd, size):
    data = b""
    while len(data) < size:
        chunk = os.read(fd, size - len(data))
        if not chunk:
            return None
        data += chunk
    return data


class AsyncInspireDftpDexPilotRetargeter:
    """Run DexPilot in a worker process so the teleop loop never waits on it.

    In-process, the torch/nlopt solve competes for the GIL with the ROS
    receiver, robot drivers and Modbus threads, which stretched each solve
    from ~1 ms to tens of milliseconds and throttled the whole teleop loop.
    Each call sends the newest keypoints and returns the newest finished
    command (normally one frame behind).
    """

    def __init__(self, is_right=True, scaling_factor=1.15, low_pass_alpha=0.5,
                 startup_timeout=60.0, wait=True, thumb_bend_weight=0.7,
                 thumb_full_bend_deg=60.0):
        _validate_thumb_blend(thumb_bend_weight, thumb_full_bend_deg)
        self.is_right = bool(is_right)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(REPO_ROOT), env.get("PYTHONPATH")))
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "paradex.retargetor.inspire_dexpilot",
                str(float(scaling_factor)),
                str(float(low_pass_alpha)),
                str(float(thumb_bend_weight)),
                str(float(thumb_full_bend_deg)),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        self._lock = threading.Lock()
        self._latest = None
        self._result_count = 0
        self._result_event = threading.Event()
        self._reader = threading.Thread(
            target=self._read_results,
            name=f"dexpilot_{'right' if self.is_right else 'left'}_reader",
            daemon=True,
        )
        self._reader.start()
        atexit.register(self.close)
        if wait:
            self.wait_ready(startup_timeout)

    def wait_ready(self, timeout=60.0):
        if not self._result_event.wait(timeout):
            self.close()
            raise RuntimeError("Inspire DexPilot worker did not start in time")
        if self.process.poll() is not None:
            raise RuntimeError("Inspire DexPilot worker exited during startup")

    def _read_results(self):
        fd = self.process.stdout.fileno()
        while True:
            data = _read_exact(fd, _COMMAND_BYTES)
            if data is None:
                self._result_event.set()
                return
            command = np.frombuffer(data, dtype=np.float64).copy()
            with self._lock:
                if np.all(np.isfinite(command)):
                    self._latest = command
                self._result_count += 1
            self._result_event.set()

    def __call__(self, hand_pose_frame, ergonomics=None):
        del ergonomics
        if self.process.poll() is not None:
            raise RuntimeError(
                f"Inspire DexPilot worker exited (code {self.process.returncode})"
            )
        keypoints = manus_frame_to_mano_keypoints(hand_pose_frame, self.is_right)
        with self._lock:
            sent_after = self._result_count
            self._result_event.clear()
        self.process.stdin.write(
            np.ascontiguousarray(keypoints, dtype=np.float64).tobytes()
        )
        with self._lock:
            latest = None if self._latest is None else self._latest.copy()
        if latest is None:
            # First command: wait briefly for the worker's first real solve.
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                self._result_event.wait(0.05)
                with self._lock:
                    if self._latest is not None and self._result_count > sent_after:
                        return self._latest.copy()
            raise RuntimeError("Inspire DexPilot worker produced no command")
        return latest

    def close(self):
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=2.0)
        except Exception:
            process.kill()


def _worker_main(argv):
    # Keep the protocol on a private copy of stdout; library prints go to stderr.
    out_fd = os.dup(1)
    os.dup2(2, 1)
    in_fd = sys.stdin.fileno()
    retargeter = InspireDftpDexPilotRetargeter(
        is_right=True,
        scaling_factor=float(argv[0]),
        low_pass_alpha=float(argv[1]),
        thumb_bend_weight=float(argv[2]),
        thumb_full_bend_deg=float(argv[3]),
    )
    os.write(out_fd, np.full(6, np.nan).tobytes())  # ready
    while True:
        data = _read_exact(in_fd, _KEYPOINT_BYTES)
        if data is None:
            return
        # Skip stale requests; only solve the newest keypoints.
        while select.select([in_fd], [], [], 0.0)[0]:
            newer = _read_exact(in_fd, _KEYPOINT_BYTES)
            if newer is None:
                return
            data = newer
        keypoints = np.frombuffer(data, dtype=np.float64).reshape(21, 3)
        try:
            command = retargeter.retarget_keypoints(keypoints)
        except Exception as exc:
            print(f"Inspire DexPilot solve failed: {exc}", file=sys.stderr, flush=True)
            continue
        os.write(out_fd, np.asarray(command, dtype=np.float64).tobytes())


if __name__ == "__main__":
    _worker_main(sys.argv[1:])
