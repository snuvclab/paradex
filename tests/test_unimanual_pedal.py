from pathlib import Path
import runpy
import sys
from threading import Event

import numpy as np
import pytest

from paradex.dataset_acqusition import capture as capture_module


@pytest.fixture
def setup_single(monkeypatch):
    for name in ('warning', 'success'):
        monkeypatch.setattr(capture_module.chime, name, lambda **kwargs: None)

    def make(states, missing_input=False, fail_after_press=False):
        events = {key: Event() for key in ('save', 'stop', 'exit')}
        class Device:
            ticks = 0
            def get_data(self):
                self.ticks += 1
                return {'Left': None, 'Right': None if missing_input else {'wrist': np.eye(4)}}
            def get_state(self):
                pytest.fail('Pedal control must not read the opposite-glove gesture')
        class Arm:
            moves = None
            def __init__(self):
                self.moves = []
            def get_data(self):
                pose = np.eye(4)
                pose[0, 3] = .2 + session.teleop_device.ticks * .01
                return {'position': pose}
            def move(self, pose):
                self.moves.append(pose.copy())
        class Hand:
            def __init__(self):
                self.moves = []
            def get_data(self):
                return {'is_connected': True, 'qpos': np.full(16, .1),
                        'joint_names': capture_module.ALLEGRO_V5_DRIVER_JOINT_NAMES}
            def move(self, action):
                self.moves.append(np.asarray(action).copy())
        class Retargetor:
            def __init__(self):
                self.starts, self.stops, self.actions = [], 0, 0
            def start(self, home):
                self.starts.append(home.copy())
            def stop(self):
                self.stops += 1
            def get_action(self, data):
                self.actions += 1
                return self.starts[-1].copy(), np.full(16, .8)
        session = capture_module.CaptureSession.__new__(capture_module.CaptureSession)
        session.hand_side = 'Right'
        session.teleop_device = Device()
        session.arm, session.hand, session.retargetor = Arm(), Hand(), Retargetor()
        session.hand_name, session.teleop_name, session.save_path = 'allegro_v5', 'vive', None
        def provider():
            index = session.teleop_device.ticks - 1
            if index == len(states) - 1:
                events['exit'].set()
            if fail_after_press and index == 1:
                raise OSError('pedal disconnected')
            return states[index]
        return session, events, provider
    return make


def test_pedal_gates_arm_and_hand_and_rebases_each_press(setup_single):
    session, events, provider = setup_single([1, 0, 1, 0, 1])
    assert session.teleop(session_events=events, state_policy='keyboard_control',
                          unimanual_state_provider=provider) == 'exit'
    assert session.retargetor.actions == 2
    assert session.retargetor.stops == 3
    assert len(session.arm.moves) == 2
    assert len(session.hand.moves) == 5
    for target, expected in zip(session.hand.moves, (.1, .8, .1, .8, .1)):
        np.testing.assert_allclose(target, expected)
    # Initial home plus actual measured arm pose at each rising pedal edge.
    assert len(session.retargetor.starts) == 3
    np.testing.assert_allclose([p[0, 3] for p in session.retargetor.starts], [.2, .22, .24])


def test_pedal_release_holds_current_hand_feedback_even_without_glove_data(setup_single):
    session, events, provider = setup_single([1], missing_input=True)
    session.teleop(session_events=events, state_policy='keyboard_control',
                   unimanual_state_provider=provider)
    assert session.retargetor.actions == 0
    assert not session.arm.moves
    np.testing.assert_allclose(session.hand.moves, np.full((1, 16), .1))


def test_pedal_read_error_pauses_instead_of_falling_back_to_gesture(setup_single):
    session, events, provider = setup_single([0, 1], fail_after_press=True)
    session.teleop(session_events=events, state_policy='keyboard_control',
                   unimanual_state_provider=provider)
    assert session.retargetor.actions == 1
    assert session.retargetor.stops == 1
    assert len(session.arm.moves) == 1
    np.testing.assert_allclose(session.hand.moves[-1], .1)


@pytest.mark.parametrize('bad', [None, 2, -1])
def test_invalid_pedal_state_is_rejected(setup_single, bad):
    session, events, provider = setup_single([bad])
    with pytest.raises(ValueError, match='invalid state'):
        session.teleop(session_events=events, unimanual_state_provider=provider)


@pytest.mark.parametrize('pedal', [True, False])
def test_capture_robot_cli_routes_pedal_and_left_glove_requirement(monkeypatch, pedal):
    import atexit
    from paradex.io import streamdeck_pedal
    from paradex.utils import keyboard_listener, system, file_io
    seen = {}
    class Pedal:
        def __init__(self):
            seen['pedal'] = self
        def get_state(self):
            return 1
        def close(self):
            seen['pedal_closed'] = True
    class Session:
        def __init__(self, **kwargs):
            seen['constructor'] = kwargs
        def teleop(self, **kwargs):
            seen['teleop'] = kwargs
            return 'exit'
        def end(self):
            seen['session_closed'] = True
    monkeypatch.setattr(capture_module, 'CaptureSession', Session)
    monkeypatch.setattr(streamdeck_pedal, 'MiddlePedalState', Pedal)
    monkeypatch.setattr(keyboard_listener, 'listen_keyboard', lambda *a, **k: None)
    monkeypatch.setattr(system, 'get_pc_list', lambda: [])
    monkeypatch.setattr(file_io, 'find_latest_index', lambda *a: 0)
    monkeypatch.setattr(atexit, 'register', lambda f: f)
    args = ['capture_robot', '--name', 'test', '--no-camera', '--arm', 'none']
    if pedal:
        args.append('--pedal')
    monkeypatch.setattr(sys, 'argv', args)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/dataset_acquisition/hri/capture_robot.py'))
    assert seen['constructor']['require_left_control'] is (not pedal)
    assert seen['teleop']['bimanual_state_provider'] is None
    if pedal:
        assert seen['teleop']['unimanual_state_provider']() == 1
        assert seen['pedal_closed']
    else:
        assert seen['teleop']['unimanual_state_provider'] is None
        assert 'pedal' not in seen
    assert seen['session_closed']
