"""Startup ordering without opening robot, camera, or ROS connections."""

from types import SimpleNamespace

import pytest

from paradex.dataset_acqusition import capture as capture_module
from paradex.io.teleop.vive import receiver as receiver_module


@pytest.fixture
def hardware(monkeypatch, tmp_path):
    calls = []

    def component(label):
        def create(*_args, **_kwargs):
            calls.append(label + ".init")
            return SimpleNamespace(**{
                method: lambda *_a, _method=method, **_kw: calls.append(label + "." + _method)
                for method in ("start", "stop", "end", "validate")
            })
        return create

    for name, label in (("remote_camera_controller", "camera"),
                        ("UTGE900", "trigger"), ("TimestampMonitor", "timestamp"),
                        ("Retargetor", "retargetor")):
        monkeypatch.setattr(capture_module, name, component(label))
    monkeypatch.setattr(capture_module, "get_arm",
                        lambda _name, **kw: component(kw["namespace"] + " arm")())
    monkeypatch.setattr(capture_module, "get_hand",
                        lambda **kw: component(kw["hand_side"] + " hand")())
    monkeypatch.setattr(receiver_module, "ViveManusROSReceiver", component("receiver"))
    monkeypatch.setattr(capture_module, "shared_dir", str(tmp_path))
    monkeypatch.setattr(capture_module, "save_current_camparam", lambda *_args: None)
    monkeypatch.setattr(capture_module, "save_current_C2R", lambda *_args: None)
    monkeypatch.setattr(capture_module, "network_info", {
        "signal_generator": {"param": {}}, "timestamp": {"param": {}},
    })
    return calls


def session(**kwargs):
    return capture_module.CaptureSession(
        camera=True, arm="xarm", hand_left="inspire_dftp", hand_right="allegro_v5",
        hand_side="bimanual", teleop="vive", **kwargs,
    )


@pytest.mark.parametrize("hand", [None, "None", "none"])
@pytest.mark.parametrize("side", ["right", "bimanual"])
def test_no_hand_disables_manus_and_left_glove_control(hardware, monkeypatch, hand, side):
    configured = {}
    def receiver(**kwargs):
        configured.update(kwargs)
        return SimpleNamespace(end=lambda: None)
    monkeypatch.setattr(receiver_module, "ViveManusROSReceiver", receiver)
    monkeypatch.setattr(capture_module, "get_arm", lambda *_a, **_kw: SimpleNamespace(end=lambda: None))
    cs = capture_module.CaptureSession(arm="xarm", hand=hand, hand_side=side, teleop="vive", require_left_control=True)
    try:
        assert cs.use_manus is False
        assert configured["use_manus"] is False
        assert configured["require_left_control"] is False
    finally:
        cs.end()


@pytest.mark.parametrize("side", ["left", "right"])
def test_bimanual_only_requires_manus_for_enabled_hand(hardware, monkeypatch, side):
    configured = {}
    monkeypatch.setattr(receiver_module, "ViveManusROSReceiver",
                        lambda **kwargs: (configured.update(kwargs) or SimpleNamespace(end=lambda: None)))
    hands = {"hand_left": "inspire_dftp" if side == "left" else None,
             "hand_right": "allegro_v5" if side == "right" else None}
    cs = capture_module.CaptureSession(teleop="vive", hand_side="bimanual", **hands)
    try:
        assert cs.use_manus is True
        assert configured["manus_sides"] == [side.title()]
    finally:
        cs.end()


def test_no_hand_and_no_vive_rejected_before_any_hardware(hardware):
    with pytest.raises(ValueError, match="No teleop input"):
        capture_module.CaptureSession(camera=True, hand=None, use_vive=False, teleop="vive")
    assert hardware == []


def test_default_still_initializes_cameras_before_receiver(hardware):
    cs = session()
    try:
        assert hardware.index("timestamp.init") < hardware.index("receiver.init")
    finally:
        cs.end()


@pytest.mark.parametrize("timestamp", [True, False])
def test_cameras_initialize_at_startup_and_keep_recording_order(hardware, timestamp):
    cs = session(timestamp=timestamp)
    try:
        expected_init = ["camera.init", "trigger.init"]
        if timestamp:
            expected_init.append("timestamp.init")
        expected_init.extend(["right arm.init", "left arm.init", "left hand.init",
                              "right hand.init", "receiver.init", "retargetor.init"])
        assert hardware == expected_init
        cs.start("first")
        expected = ["camera.start"]
        if timestamp:
            expected.append("timestamp.start")
        expected.extend(["trigger.start", "camera.validate"])
        assert [item for item in hardware if item in expected] == expected
        assert hardware.index("camera.init") < hardware.index("left arm.start")
        assert ("timestamp.init" in hardware) is timestamp
        cs.stop()
        cs.start("second")
        assert hardware.count("camera.init") == 1
        assert hardware.count("timestamp.init") == int(timestamp)
        cs.stop()
    finally:
        cs.end()


def test_exit_without_recording_releases_cameras_initialized_at_startup(hardware):
    cs = session()
    cs.end()
    for label in ("camera", "timestamp", "trigger"):
        assert hardware.count(label + ".init") == 1
        assert hardware.count(label + ".end") == 1
        assert label + ".start" not in hardware


def test_camera_connection_failure_aborts_startup_and_releases_controller(hardware, monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("timestamp unavailable")

    monkeypatch.setattr(capture_module, "TimestampMonitor", fail)
    with pytest.raises(RuntimeError, match="timestamp unavailable"):
        session()
    assert hardware == ["camera.init", "trigger.init", "camera.end"]


def test_startup_profile_identifies_stages_and_failures(hardware, monkeypatch, capsys):
    cs = session(profile_startup=True)
    try:
        output = capsys.readouterr().out
        assert "[startup] right arm: starting" in output
        assert "[startup] right arm: initialized in " in output
        assert "[startup] teleop receiver: initialized in " in output
        assert "[startup] timestamp camera: starting" in output
    finally:
        cs.end()

    def fail(**_kwargs):
        raise RuntimeError("timestamp unavailable")

    monkeypatch.setattr(capture_module, "TimestampMonitor", fail)
    with pytest.raises(RuntimeError):
        session(profile_startup=True)
    assert "[startup] timestamp camera: failed after " in capsys.readouterr().out


def test_camera_disabled_skips_camera_initialization(hardware):
    cs = capture_module.CaptureSession(camera=False)
    cs.start("hand_only")
    cs.stop()
    cs.end()
    assert hardware == []
