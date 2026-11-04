from threading import Event, Thread
import time
from types import SimpleNamespace

import numpy as np
import pytest

from paradex.dataset_acqusition.capture import CaptureSession
from paradex.retargetor.unimanual import Retargetor


class Controller:
    def __init__(self):
        self.moves = []
        self.moved = Event()

    def get_data(self):
        return {"position": np.eye(4)}

    def move(self, target):
        self.moves.append(target)
        self.moved.set()


def make_session():
    session = CaptureSession.__new__(CaptureSession)
    session.teleop_name = "vive"
    session.teleop_control_mode = "split"
    session.hand_side = "Right"
    session.arm = Controller()
    session.hand = Controller()
    session.hand_name = "inspire_dftp"
    session.save_path = None
    session.events = {name: Event() for name in ("exit", "save", "stop")}
    session.vive_arm_smoothing_tau_s = 0
    data = {"Right": {"wrist": np.eye(4)}, "Left": None}
    session.teleop_device = SimpleNamespace(
        get_arm_data=lambda: data,
        get_hand_data=lambda: data,
        get_state=lambda: None,
        get_data=lambda: data,
    )
    session.retargetor = SimpleNamespace(
        start=lambda home: None, start_arm=lambda home: None,
        get_arm_action=lambda data: np.eye(4),
        get_hand_action=lambda data: np.array([1.0]),
    )
    return session


@pytest.fixture(autouse=True)
def quiet_chime(monkeypatch):
    monkeypatch.setattr("chime.warning", lambda **kwargs: None)


def launch(session, **kwargs):
    result = []
    errors = []

    def run():
        try:
            result.append(session.teleop(state_policy="keyboard_control", **kwargs))
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=run, daemon=True)
    thread.start()
    return thread, result, errors


def finish(session, thread, errors):
    session.events["exit"].set()
    thread.join(2)
    assert not thread.is_alive()
    assert errors == []
    assert all(not worker.is_alive() for worker in session._control_threads)


@pytest.mark.parametrize("blocked_lane,stage", [
    ("hand", "input"), ("hand", "retarget"), ("hand", "move"),
    ("arm", "input"), ("arm", "retarget"), ("arm", "move"),
])
def test_blocked_lane_does_not_stop_other_command_loop(blocked_lane, stage):
    session = make_session()
    entered, release = Event(), Event()
    if stage == "input":
        obj, attr = session.teleop_device, "get_" + blocked_lane + "_data"
    elif stage == "retarget":
        obj, attr = session.retargetor, "get_" + blocked_lane + "_action"
    else:
        obj, attr = getattr(session, blocked_lane), "move"
    original = getattr(obj, attr)

    def blocking(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)

    setattr(obj, attr, blocking)
    thread, result, errors = launch(session)
    try:
        assert entered.wait(1)
        other = session.arm if blocked_lane == "hand" else session.hand
        first_count = len(other.moves)
        deadline = time.monotonic() + 1
        while len(other.moves) < first_count + 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(other.moves) >= first_count + 3
    finally:
        release.set()
        finish(session, thread, errors)
    assert result == ["exit"]


@pytest.mark.parametrize("missing", ["arm", "hand"])
def test_missing_input_only_holds_its_own_lane(missing):
    session = make_session()
    setattr(session.teleop_device, "get_" + missing + "_data", lambda: {"Left": None, "Right": None})
    thread, _, errors = launch(session)
    try:
        other = session.arm if missing == "hand" else session.hand
        assert other.moved.wait(1)
        assert getattr(session, missing).moves == []
    finally:
        finish(session, thread, errors)


def test_release_discards_hand_action_computed_before_pedal_transition():
    session = make_session()
    entered, release, paused = Event(), Event(), Event()
    pedal = [0]

    def action(data):
        entered.set()
        assert release.wait(2)
        return np.array([1.0])

    def state():
        if pedal[0] == 1:
            paused.set()
        return pedal[0]

    session.retargetor.get_hand_action = action
    thread, _, errors = launch(session, unimanual_state_provider=state)
    try:
        assert entered.wait(1)
        assert session.arm.moved.wait(1)
        pedal[0] = 1
        assert paused.wait(1)
        time.sleep(0.03)
        count = len(session.arm.moves)
        release.set()
        time.sleep(0.05)
        assert session.hand.moves == []
        assert len(session.arm.moves) == count
    finally:
        release.set()
        finish(session, thread, errors)


def test_worker_error_holds_its_lane_and_reports_failure():
    session = make_session()

    def broken(data):
        raise RuntimeError("test hand failure")

    session.retargetor.get_hand_action = broken
    thread, _, errors = launch(session)
    try:
        assert session.arm.moved.wait(1)
        deadline = time.monotonic() + 1
        while "Hand" not in session._control_failures and time.monotonic() < deadline:
            time.sleep(0.005)
        assert str(session._control_failures["Hand"]) == "test hand failure"
        assert session.hand.moves == []
    finally:
        finish(session, thread, errors)


def test_separate_retarget_methods_preserve_combined_actions_and_arm_only_rebase():
    retargetor = Retargetor(hand_side="Bimanual", arm_name="xarm", teleop_name="vive")
    data = {side: {"wrist": np.eye(4)} for side in ("Left", "Right")}
    home = {side: np.eye(4) for side in ("Left", "Right")}
    retargetor.start(home)
    retargetor.hand_retargetor = {side: lambda frame: np.array([2.0]) for side in home}
    combined = retargetor.get_action(data)
    arms = retargetor.get_arm_action(data)
    hands = retargetor.get_hand_action(data)
    for actual, expected in zip((*arms, *hands), combined):
        np.testing.assert_array_equal(actual, expected)
    class HandRetargeter:
        def reset(self):
            pytest.fail("arm rebase must not reset hand retargeter")
    retargetor.hand_retargetor = {side: HandRetargeter() for side in home}
    retargetor.start_arm(home)


@pytest.mark.parametrize("layout", ["hand_only", "arm_only", "bimanual"])
def test_split_loop_single_device_and_bimanual_lifecycle(layout):
    session = make_session()
    if layout == "hand_only":
        session.arm = None
        session.use_vive = False
    elif layout == "arm_only":
        session.hand = None
    else:
        session.hand_side = "Bimanual"
        session.arm_left, session.arm_right = Controller(), session.arm
        session.hand_left, session.hand_right = Controller(), session.hand
        session.hand_name_left = session.hand_name_right = "inspire_dftp"
        data = {side: {"wrist": np.eye(4)} for side in ("Left", "Right")}
        session.teleop_device.get_arm_data = session.teleop_device.get_hand_data = lambda: data
        session.retargetor.get_arm_action = lambda data: (np.eye(4), np.eye(4))
        session.retargetor.get_hand_action = lambda data: (np.array([1.0]), np.array([2.0]))
    thread, _, errors = launch(session)
    try:
        controllers = ([session.arm_left, session.arm_right, session.hand_left, session.hand_right]
                       if layout == "bimanual" else [controller for controller in (session.arm, session.hand) if controller])
        for controller in controllers:
            assert controller.moved.wait(1)
        assert len(session._control_threads) == (2 if layout == "bimanual" else 1)
    finally:
        finish(session, thread, errors)


def test_session_transition_stops_workers_before_next_teleop():
    session = make_session()
    thread, result, errors = launch(session)
    assert session.hand.moved.wait(1)
    session.events["save"].set()
    thread.join(2)
    assert not thread.is_alive()
    assert not errors
    assert result == ["start"]
    first_workers = session._control_threads
    assert all(not worker.is_alive() for worker in first_workers)
    session.events["save"].clear()
    session.hand.moved.clear()
    thread, _, errors = launch(session)
    try:
        assert session.hand.moved.wait(1)
        assert all(worker not in first_workers for worker in session._control_threads)
    finally:
        finish(session, thread, errors)
