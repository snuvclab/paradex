"""MANUS-only left receiver checks without ROS, camera, or robot dependencies."""

import builtins
import json
import sys
import time
from threading import Event, Lock
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from paradex.io.teleop.vive import receiver as receiver_module


def make_left_receiver():
    receiver = receiver_module.ViveManusROSReceiver.__new__(
        receiver_module.ViveManusROSReceiver
    )
    receiver.hand_side = "left"
    receiver.use_vive = False
    receiver.use_manus = True
    receiver.require_left_control = False
    receiver.lock = Lock()
    receiver.save_event = Event()
    receiver.save_path = None
    receiver.data = None
    receiver.max_age_s = 0.25
    wrist = np.eye(4)
    wrist[:3, 3] = [0.3, -0.2, 0.9]
    finger = wrist.copy()
    finger[:3, 3] += [0.02, 0.04, 0.01]
    receiver._vive_left = None
    receiver._vive_right = None
    receiver._vive_left_time = None
    receiver._vive_right_time = None
    receiver._manus_frames = {
        "Left": {"wrist": wrist, "index_distal": finger}, "Right": None,
    }
    receiver._manus_ergonomics = {"Left": {"IndexMCPStretch": 42.0}, "Right": None}
    receiver._manus_times = {"Left": time.monotonic(), "Right": None}
    receiver._manus_glove_ids = {"Left": 17, "Right": None}
    receiver._vive_frame_id = None
    receiver._vive_left_frame_id = None
    receiver.vive_left_topic = "/vive_trackers/left_hand/pose"
    receiver.vive_right_topic = "/vive_trackers/right_hand/pose"
    receiver.manus_topics = ("/manus_glove_0", "/manus_glove_1")
    return receiver


def test_manus_only_left_needs_no_geometry_messages_or_vive_subscriptions(monkeypatch):
    subscriptions = []

    def subscribe(message_type, topic, callback, qos):
        subscriptions.append(topic)
        return object()

    logger = SimpleNamespace(info=lambda *_: None)
    node = SimpleNamespace(create_subscription=subscribe, get_logger=lambda: logger)
    modules = {
        "rclpy": {"ok": lambda: True, "create_node": lambda *_: node},
        "rclpy.executors": {"SingleThreadedExecutor": lambda: SimpleNamespace(
            add_node=lambda *_: None,
        )},
        "rclpy.qos": {"qos_profile_sensor_data": object()},
        "manus_ros2_msgs.msg": {"ManusGlove": object},
    }
    for name, values in modules.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(receiver_module, "Thread", lambda **_: SimpleNamespace(start=lambda: None))
    original_import = builtins.__import__

    def import_without_geometry(name, *args, **kwargs):
        if name.startswith("geometry_msgs"):
            raise AssertionError("MANUS-only mode must not import VIVE pose messages")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_geometry)

    receiver = receiver_module.ViveManusROSReceiver(
        hand_side="left", use_vive=False, require_left_control=False,
    )

    assert subscriptions == list(receiver.manus_topics)


def test_manus_only_left_needs_no_right_glove_or_tracker():
    receiver = make_left_receiver()
    source = receiver._manus_frames["Left"]

    data = receiver.get_data()

    assert data["Right"] is None
    assert data["ergonomics"] == {"Left": {"IndexMCPStretch": 42.0}}
    for name in ("wrist", "index_distal"):
        np.testing.assert_allclose(data["Left"][name], source[name])
        assert not np.shares_memory(data["Left"][name], source[name])


@pytest.mark.parametrize("missing", [True, False])
def test_manus_only_left_rejects_missing_or_stale_glove(missing):
    receiver = make_left_receiver()
    receiver._manus_times["Left"] = None if missing else time.monotonic() - 1.0

    data = receiver.get_data()

    assert data["Left"] is None
    assert data["Right"] is None


@pytest.mark.parametrize("side", ["left", "right", "bimanual"])
@pytest.mark.parametrize("pause_state", [0, 1])
def test_only_left_mode_ignores_its_own_glove_pause_state(side, pause_state):
    receiver = make_left_receiver()
    receiver.hand_side = side
    receiver.require_left_control = True
    receiver._pause_state = pause_state

    assert receiver.get_state() == (None if side == "left" else pause_state)


def test_manus_only_left_recording_saves_left_glove_frames_and_timestamps(tmp_path):
    receiver = make_left_receiver()
    expected_wrist = receiver._manus_frames["Left"]["wrist"].copy()
    expected_timestamp = receiver._manus_times["Left"]

    receiver.start(tmp_path)
    receiver.get_data()
    receiver.stop()

    assert np.load(tmp_path / "time.npy").shape == (1,)
    left_frames = np.load(tmp_path / "left.npy", allow_pickle=True)
    right_frames = np.load(tmp_path / "right.npy", allow_pickle=True)
    assert len(left_frames) == 1
    assert right_frames.tolist() == [None]
    np.testing.assert_allclose(left_frames[0]["wrist"], expected_wrist)
    np.testing.assert_array_equal(
        np.load(tmp_path / "manus_left_time.npy"), [expected_timestamp],
    )
    for name in ("manus_right_time", "vive_left_time", "vive_right_time"):
        assert not (tmp_path / f"{name}.npy").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["use_vive"] is False
    assert metadata["use_manus"] is True
    assert metadata["manus_topics"] == list(receiver.manus_topics)
    assert "vive_left_topic" not in metadata
    assert "vive_right_topic" not in metadata
