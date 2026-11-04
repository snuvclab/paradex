import json
import sys
from queue import Queue
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock
import time
from threading import Event, Lock

import numpy as np
import pytest

from paradex.io.teleop.vive.receiver import (
    _VIVE_TRACKER_TO_WRIST_ROTATION,
    _canonical_joint_name,
    apply_vive_tracker_mount_rotation,
    manus_ergonomics_to_dict,
    manus_message_to_frame,
    pose_to_matrix,
    reparent_frame,
    ViveManusROSReceiver,
    _pause_state_from_ergonomics,
)
from paradex.retargetor.hand_regargetor import inspire_f1
from paradex.retargetor.state import HandStateExtractor
from paradex.retargetor.unimanual import Retargetor


def make_pose(position=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        position=SimpleNamespace(
            x=position[0],
            y=position[1],
            z=position[2],
        ),
        orientation=SimpleNamespace(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3],
        ),
    )


def make_node(node_id, parent_id, chain, joint, position):
    return SimpleNamespace(
        node_id=node_id,
        parent_node_id=parent_id,
        chain_type=chain,
        joint_type=joint,
        pose=make_pose(position),
    )


def make_manus_message(side="Right", invalid_orientations=False, flex=0.0):
    quaternion = (
        (np.nan, np.nan, np.nan, np.nan)
        if invalid_orientations
        else (0.0, 0.0, 0.0, 1.0)
    )
    nodes = [make_node(0, 0, "Hand", "Invalid", (1.0, 2.0, 3.0))]
    node_id = 1
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        joints = (
            ("MCP", "PIP", "DIP", "TIP")
            if finger == "Thumb"
            else ("MCP", "PIP", "IP", "DIP", "TIP")
        )
        parent_id = 0
        for joint_index, joint in enumerate(joints):
            nodes.append(
                make_node(
                    node_id,
                    parent_id,
                    finger,
                    joint,
                    (1.0 + node_id * 0.01, 2.0 + joint_index * 0.01, 3.0),
                )
            )
            parent_id = node_id
            node_id += 1
    for node in nodes:
        node.pose.orientation = SimpleNamespace(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3],
        )
    ergonomics = []
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        ergonomics.extend(
            [
                SimpleNamespace(type=f"{finger}MCPStretch", value=flex),
                SimpleNamespace(type=f"{finger}PIPStretch", value=flex),
            ]
        )
    return SimpleNamespace(
        side=side,
        glove_id=42,
        raw_nodes=nodes,
        ergonomics=ergonomics,
    )


def test_pose_to_matrix_normalizes_quaternion():
    transform = pose_to_matrix(
        make_pose(position=(1.0, 2.0, 3.0), quaternion=(0.0, 0.0, 0.0, 2.0))
    )
    np.testing.assert_allclose(transform, np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ))


def _axis_rotation(axis, angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)
    if axis == "x":
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
        )
    if axis == "y":
        return np.array(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
        )
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def test_vive_mount_rotation_preserves_existing_right_hand_correction():
    initial = np.eye(4)
    initial[:3, 3] = [0.4, -0.2, 1.1]
    corrected_initial = apply_vive_tracker_mount_rotation(initial)

    np.testing.assert_allclose(corrected_initial[:3, 3], initial[:3, 3])
    np.testing.assert_allclose(
        corrected_initial[:3, :3],
        _VIVE_TRACKER_TO_WRIST_ROTATION,
    )
    for axis in ("x", "y", "z"):
        current = initial.copy()
        current[:3, :3] = _axis_rotation(axis, 0.2)
        corrected_current = apply_vive_tracker_mount_rotation(current)
        np.testing.assert_allclose(
            corrected_current[:3, :3],
            current[:3, :3] @ _VIVE_TRACKER_TO_WRIST_ROTATION,
            atol=1.0e-8,
        )


def test_canonical_joint_mapping_matches_live_manus_names():
    assert _canonical_joint_name("Thumb", "MCP") == "thumb_metacarpal"
    assert _canonical_joint_name("Thumb", "DIP") == "thumb_distal"
    assert _canonical_joint_name("Index", "IP") == "index_intermediate"
    assert _canonical_joint_name("Pinky", "TIP") == "pinky_tip"
    assert _canonical_joint_name("Hand", "Invalid") is None


def test_manus_message_converts_to_xsens_named_contract():
    frame = manus_message_to_frame(make_manus_message())
    assert frame["wrist"].shape == (4, 4)
    for finger in ("index", "middle", "ring", "pinky"):
        for joint in ("metacarpal", "proximal", "intermediate", "distal"):
            assert f"{finger}_{joint}" in frame
    for joint in ("metacarpal", "proximal", "distal"):
        assert f"thumb_{joint}" in frame


def test_manus_nan_orientations_are_reconstructed_from_joint_positions():
    frame = manus_message_to_frame(
        make_manus_message(invalid_orientations=True)
    )

    for transform in frame.values():
        assert np.all(np.isfinite(transform))
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3],
            np.eye(3),
            atol=1.0e-6,
        )


def test_reparent_frame_uses_vive_wrist_and_preserves_manus_relative_pose():
    source = manus_message_to_frame(make_manus_message())
    vive_wrist = np.eye(4)
    vive_wrist[:3, 3] = [10.0, 20.0, 30.0]

    result = reparent_frame(source, vive_wrist)

    np.testing.assert_allclose(result["wrist"], vive_wrist)
    expected_relative = (
        np.linalg.inv(source["wrist"]) @ source["index_distal"]
    )
    actual_relative = np.linalg.inv(result["wrist"]) @ result["index_distal"]
    np.testing.assert_allclose(actual_relative, expected_relative)


def test_frames_feed_existing_right_hand_and_left_state_retargetors():
    right_manus = manus_message_to_frame(make_manus_message("Right"))
    left_manus = manus_message_to_frame(make_manus_message("Left"))
    right_frame = reparent_frame(right_manus, np.eye(4))

    hand_action = inspire_f1(right_frame, is_right=True)
    state = HandStateExtractor().get_state(left_manus)

    assert hand_action.shape == (6,)
    assert state in (0, 1, 2, 3)


def make_receiver_without_ros(sample_age=0.0, hand_side="right"):
    receiver = ViveManusROSReceiver.__new__(ViveManusROSReceiver)
    receiver.hand_side = hand_side
    receiver.vive_right_topic = "/vive_trackers/right_hand/pose"
    receiver.vive_left_topic = "/vive_trackers/left_hand/pose"
    receiver.manus_topics = ("/manus_glove_0", "/manus_glove_1")
    receiver.lock = Lock()
    receiver.save_event = Event()
    receiver.data = None
    receiver.max_age_s = 0.25
    receiver._vive_right = np.eye(4)
    receiver._vive_left = np.eye(4)
    receiver._vive_left[:3, 3] = [-0.4, 0.2, 1.1]
    receiver._manus_frames = {
        "Left": manus_message_to_frame(make_manus_message("Left")),
        "Right": manus_message_to_frame(make_manus_message("Right")),
    }
    receiver._manus_ergonomics = {
        "Left": manus_ergonomics_to_dict(
            make_manus_message("Left").ergonomics
        ),
        "Right": manus_ergonomics_to_dict(
            make_manus_message("Right").ergonomics
        ),
    }
    timestamp = time.monotonic() - sample_age
    receiver._vive_right_time = timestamp
    receiver._vive_left_time = timestamp
    receiver._manus_times = {"Left": timestamp, "Right": timestamp}
    receiver._vive_frame_id = "world"
    receiver._vive_left_frame_id = "world"
    receiver._manus_glove_ids = {"Left": 41, "Right": 42}
    receiver.save_path = None
    return receiver


def test_get_data_fuses_fresh_right_vive_and_both_manus_hands():
    data = make_receiver_without_ros().get_data()

    assert data["Left"] is not None
    assert data["Right"] is not None
    assert data["ergonomics"]["Right"] is not None
    np.testing.assert_allclose(data["Right"]["wrist"], np.eye(4))


def test_get_data_fuses_each_manus_hand_with_its_bimanual_vive_tracker():
    receiver = make_receiver_without_ros(hand_side="bimanual")
    source_left = receiver._manus_frames["Left"]
    source_right = receiver._manus_frames["Right"]

    data = receiver.get_data()

    np.testing.assert_allclose(data["Left"]["wrist"], receiver._vive_left)
    np.testing.assert_allclose(data["Right"]["wrist"], receiver._vive_right)
    for side, source in (("Left", source_left), ("Right", source_right)):
        expected_relative = (
            np.linalg.inv(source["wrist"]) @ source["index_distal"]
        )
        actual_relative = (
            np.linalg.inv(data[side]["wrist"]) @ data[side]["index_distal"]
        )
        np.testing.assert_allclose(actual_relative, expected_relative)


def test_manus_only_left_mode_does_not_require_right_glove_or_vive():
    receiver = make_receiver_without_ros(hand_side="left")
    receiver.use_vive = False
    receiver.require_left_control = False
    receiver._manus_times["Right"] = None
    receiver._manus_frames["Right"] = None

    data = receiver.get_data()

    assert data["Left"] is not None
    assert data["Right"] is None
    assert set(data["ergonomics"]) == {"Left"}


@pytest.mark.parametrize("pause_state", [0, 1, 2, 3])
def test_manus_only_right_ignores_unsolicited_left_control(pause_state):
    receiver = make_receiver_without_ros()
    receiver.use_vive = False
    receiver.require_left_control = False
    receiver._pause_state = pause_state

    assert receiver.get_data()["Right"] is not None
    assert receiver.get_state() is None

    receiver.require_left_control = True
    assert receiver.get_state() == pause_state


def test_get_data_suppresses_commands_when_any_input_is_stale():
    data = make_receiver_without_ros(sample_age=1.0).get_data()

    assert data["Left"] is None
    assert data["Right"] is None


@pytest.mark.parametrize(
    "source",
    ("vive_left", "vive_right", "manus_left", "manus_right"),
)
def test_bimanual_get_data_suppresses_both_sides_when_one_input_is_stale(source):
    receiver = make_receiver_without_ros(hand_side="bimanual")
    stale_time = time.monotonic() - 1.0
    if source == "vive_left":
        receiver._vive_left_time = stale_time
    elif source == "vive_right":
        receiver._vive_right_time = stale_time
    elif source == "manus_left":
        receiver._manus_times["Left"] = stale_time
    else:
        receiver._manus_times["Right"] = stale_time

    data = receiver.get_data()

    assert data["Left"] is None
    assert data["Right"] is None


def test_unimanual_right_does_not_require_a_left_vive_tracker():
    receiver = make_receiver_without_ros(hand_side="right")
    receiver._vive_left = None
    receiver._vive_left_time = None

    data = receiver.get_data()

    assert data["Left"] is not None
    assert data["Right"] is not None


def test_bimanual_recording_saves_both_vive_timestamps_and_metadata(tmp_path):
    receiver = make_receiver_without_ros(hand_side="bimanual")
    save_path = tmp_path / "teleop"

    receiver.start(save_path)
    receiver.get_data()
    receiver.stop()

    assert np.load(save_path / "vive_left_time.npy").shape == (1,)
    assert np.load(save_path / "vive_right_time.npy").shape == (1,)
    with open(save_path / "metadata.json") as stream:
        metadata = json.load(stream)
    assert metadata["mode"] == "vive_manus_bimanual"
    assert metadata["vive_left_topic"] == "/vive_trackers/left_hand/pose"
    assert metadata["vive_right_topic"] == "/vive_trackers/right_hand/pose"
    assert metadata["vive_left_frame_id"] == "world"


def test_left_manus_ergonomics_produces_pause_and_go_states():
    assert _pause_state_from_ergonomics(
        make_manus_message("Left", flex=60.0).ergonomics
    ) == 0
    assert _pause_state_from_ergonomics(
        make_manus_message("Left", flex=10.0).ergonomics
    ) == 1


def test_vive_arm_only_mode_does_not_require_a_robot_hand():
    retargetor = Retargetor(
        arm_name="xarm",
        hand_name=None,
        hand_side="Right",
    )
    retargetor.start(np.eye(4))
    data = {"Right": {"wrist": np.eye(4)}}
    retargetor.get_action(data)
    data["Right"]["wrist"][:3, 3] = [0.1, -0.2, 0.3]

    wrist_action, hand_action = retargetor.get_action(data)

    assert hand_action is None
    np.testing.assert_allclose(wrist_action[:3, 3], [-0.1, 0.2, 0.3])


@pytest.mark.parametrize("side", ["right", "bimanual"])
def test_no_manus_streams_are_needed_for_vive_only_recording(side, tmp_path):
    receiver = make_receiver_without_ros(hand_side=side)
    receiver.use_vive = True
    receiver.use_manus = False
    receiver.require_left_control = False
    receiver._manus_frames = {"Left": None, "Right": None}
    receiver._manus_times = {"Left": None, "Right": None}
    receiver.start(str(tmp_path))
    data = receiver.get_data()
    assert data["Right"] is not None
    assert data["ergonomics"] == {}
    receiver.stop()
    assert not list(tmp_path.glob("manus*"))
    assert len(np.load(tmp_path / "vive_right_time.npy")) == 1


def test_vive_only_constructor_never_imports_or_subscribes_manus(monkeypatch):
    node = Mock()
    rclpy = ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.create_node = lambda _name: node
    executor = ModuleType("rclpy.executors")
    executor.SingleThreadedExecutor = Mock
    qos = ModuleType("rclpy.qos")
    qos.qos_profile_sensor_data = object()
    geometry = ModuleType("geometry_msgs.msg")
    geometry.PoseStamped = object
    for name, module in {"rclpy": rclpy, "rclpy.executors": executor, "rclpy.qos": qos,
                         "geometry_msgs.msg": geometry, "manus_ros2_msgs.msg": None}.items():
        monkeypatch.setitem(sys.modules, name, module)
    receiver = ViveManusROSReceiver(hand_side="bimanual", use_manus=False, require_left_control=False)
    try:
        topics = [call.args[1] for call in node.create_subscription.call_args_list]
        assert topics == ["/vive_trackers/right_hand/pose", "/vive_trackers/left_hand/pose"]
    finally:
        receiver.end()


@pytest.fixture
def fake_ros_executors(monkeypatch):
    nodes, executors = [], []

    class FakeNode:
        def __init__(self, name):
            self.name = name
            self.topics = []
            self.destroyed = False
            nodes.append(self)

        def create_subscription(self, _type, topic, callback, _qos):
            self.topics.append(topic)
            return callback

        def get_logger(self):
            return Mock()

        def destroy_node(self):
            self.destroyed = True

    class FakeExecutor:
        def __init__(self):
            self.jobs = Queue()
            self.nodes = []
            executors.append(self)

        def add_node(self, node):
            self.nodes.append(node)

        def remove_node(self, node):
            self.nodes.remove(node)

        def spin(self):
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                job()

        def shutdown(self):
            self.jobs.put(None)

    rclpy = ModuleType("rclpy")
    rclpy.ok = lambda: True
    rclpy.create_node = FakeNode
    executor = ModuleType("rclpy.executors")
    executor.SingleThreadedExecutor = FakeExecutor
    qos = ModuleType("rclpy.qos")
    qos.qos_profile_sensor_data = object()
    geometry = ModuleType("geometry_msgs.msg")
    geometry.PoseStamped = object
    manus = ModuleType("manus_ros2_msgs.msg")
    manus.ManusGlove = object
    for name, module in {"rclpy": rclpy, "rclpy.executors": executor, "rclpy.qos": qos,
                         "geometry_msgs.msg": geometry, "manus_ros2_msgs.msg": manus}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return nodes, executors


@pytest.mark.parametrize("mode,vive,manus,count", [
    ("split", True, True, 2), ("shared", True, True, 1),
    ("split", True, False, 1), ("split", False, True, 1),
])
def test_input_executor_routing_and_shutdown(fake_ros_executors, mode, vive, manus, count):
    nodes, executors = fake_ros_executors
    receiver = ViveManusROSReceiver(callback_mode=mode, use_vive=vive, use_manus=manus,
                                    hand_side="bimanual", require_left_control=False)
    try:
        assert len(executors) == count
        if count == 2:
            assert nodes[0].topics == ["/vive_trackers/right_hand/pose", "/vive_trackers/left_hand/pose"]
            assert nodes[1].topics == ["/manus_glove_0", "/manus_glove_1"]
            assert receiver.spin_thread is not receiver.manus_spin_thread
    finally:
        receiver.end()
    assert all(node.destroyed for node in nodes)
    assert not receiver.spin_thread.is_alive()
    assert receiver.manus_spin_thread is None or not receiver.manus_spin_thread.is_alive()


@pytest.mark.parametrize("mode", ["split", "shared"])
def test_stalled_manus_conversion_does_not_block_split_vive(fake_ros_executors, monkeypatch, mode):
    from paradex.io.teleop.vive import receiver as module
    entered, release, vive_received = Event(), Event(), Event()
    original = module.manus_message_to_frame

    def stalled_conversion(message):
        entered.set()
        assert release.wait(2)
        return original(message)

    monkeypatch.setattr(module, "manus_message_to_frame", stalled_conversion)
    receiver = ViveManusROSReceiver(callback_mode=mode, require_left_control=False)
    message = SimpleNamespace(pose=make_pose(), header=SimpleNamespace(frame_id="test"))

    def vive_callback():
        receiver._vive_callback(message, "Right")
        vive_received.set()

    try:
        receiver.manus_executor.jobs.put(lambda: receiver._manus_callback(make_manus_message(), "/manus_glove_0"))
        assert entered.wait(1)
        receiver.executor.jobs.put(vive_callback)
        assert vive_received.wait(.15) is (mode == "split")
    finally:
        release.set()
        receiver.end()
    assert vive_received.is_set()


@pytest.mark.parametrize("enabled", ["Left", "Right"])
def test_bimanual_ignores_missing_manus_on_disabled_hand(enabled, tmp_path):
    receiver = make_receiver_without_ros(hand_side="bimanual")
    receiver.manus_sides = (enabled,)
    disabled = "Left" if enabled == "Right" else "Right"
    receiver._manus_frames[disabled] = None
    receiver._manus_times[disabled] = None
    receiver.start(str(tmp_path))
    data = receiver.get_data()
    assert data["Left"] is not None and data["Right"] is not None
    assert set(data["ergonomics"]) == {enabled}
    assert set(data[disabled]) == {"wrist"}
    receiver.stop()
    assert not (tmp_path / f"manus_{disabled.lower()}_time.npy").exists()
    assert len(np.load(tmp_path / f"manus_{enabled.lower()}_time.npy")) == 1


@pytest.mark.parametrize("stale_source", ["vive", "manus"])
def test_independent_control_snapshots_ignore_other_source_freshness(stale_source):
    receiver = make_receiver_without_ros(hand_side="bimanual")
    receiver.use_vive = receiver.use_manus = True
    receiver.manus_sides = ("Left", "Right")
    if stale_source == "vive":
        receiver._vive_left_time = receiver._vive_right_time = None
    else:
        receiver._manus_times = {"Left": None, "Right": None}
    arm_data, hand_data = receiver.get_arm_data(), receiver.get_hand_data()
    for side in ("Left", "Right"):
        assert (arm_data[side] is None) == (stale_source == "vive")
        assert (hand_data[side] is None) == (stale_source == "manus")
    # Synchronized recording retains its previous availability contract.
    assert receiver.get_data()["Right"] is None


def test_hand_snapshot_keeps_manus_wrist_and_active_side_only():
    receiver = make_receiver_without_ros(hand_side="bimanual")
    receiver.use_manus = True
    receiver.manus_sides = ("Left",)
    data = receiver.get_hand_data()
    assert data["Right"] is None
    np.testing.assert_array_equal(data["Left"]["wrist"], receiver._manus_frames["Left"]["wrist"])
    data["Left"]["wrist"][:] = 0
    assert receiver._manus_frames["Left"]["wrist"][3, 3] == 1


def test_independent_hand_frame_preserves_mixed_hand_retargeting():
    receiver = make_receiver_without_ros(hand_side="bimanual")
    receiver.use_vive = receiver.use_manus = True
    receiver.manus_sides = ("Left", "Right")
    receiver._vive_right[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    receiver._vive_right[:3, 3] = [0.5, -0.2, 0.7]
    retargetor = Retargetor(
        arm_name="xarm", hand_side="Bimanual", teleop_name="vive",
        hand_name_left="inspire_dftp", hand_name_right="allegro_v5",
    )
    combined_data = receiver.get_data()
    independent_data = receiver.get_hand_data()
    # The minimal message fixture omits some angles required by robot hands.
    for side in ("Left", "Right"):
        values = {f"{finger}{joint}{kind}": 10.0
                  for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky")
                  for joint in ("MCP", "PIP", "DIP")
                  for kind in ("Stretch", "Spread")}
        values.update({finger + "Spread": 5.0 for finger in ("Index", "Middle", "Ring")})
        combined_data["ergonomics"][side] = values.copy()
        independent_data["ergonomics"][side] = values.copy()
    combined = retargetor.get_hand_action(combined_data)
    independent = retargetor.get_hand_action(independent_data)
    for actual, expected in zip(independent, combined):
        np.testing.assert_allclose(actual, expected, atol=1e-8)
    combined_data.pop("ergonomics")
    independent_data.pop("ergonomics")
    for actual, expected in zip(retargetor.get_hand_action(independent_data),
                                retargetor.get_hand_action(combined_data)):
        np.testing.assert_allclose(actual, expected, atol=1e-8)
