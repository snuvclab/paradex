"""Model and transport checks; no ROS publisher or hardware actuation."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from paradex.retargetor.allegro_v5_dexpilot import (
    AllegroV5DexPilotRetargeter, AsyncAllegroV5DexPilotRetargeter,
    JOINT_NAMES, OPERATOR2ALLEGRO_RIGHT, joint_limits, validate_keypoints,
)
from paradex.retargetor.inspire_dexpilot import MANUS_TO_MANO_KEYPOINTS
from paradex.retargetor.unimanual import Retargetor


def points():
    p = np.zeros((21, 3))
    p[1:5] = [[-.01, -.02, 0], [-.02, -.035, -.005],
              [-.035, -.045, -.01], [-.05, -.05, -.015]]
    for finger, y in enumerate([-.03, -.01, .01, .03]):
        for joint, x in enumerate([-.04, -.075, -.095, -.115]):
            p[5 + 4*finger + joint] = [x, y, 0.]
    return p


def frame(p=None):
    p = points() if p is None else np.array(p, copy=True)
    # Input MANUS right glove is mirror-handed relative to MANO right.
    p[:, 0] *= -1
    transforms = []
    for position in p:
        transform = np.eye(4)
        transform[:3, 3] = position
        transforms.append(transform)
    return dict(zip(MANUS_TO_MANO_KEYPOINTS, transforms))


def vector_adapter():
    r = object.__new__(AllegroV5DexPilotRetargeter)
    r.human_indices = np.array([[0, 0, 0, 0], [4, 8, 12, 16]])
    r.scaling_factor = 1.15
    r.robot_middle_mcp = np.array([0., 0., .015])
    return r


def test_wrist_frame_maps_forward_and_across_to_v5_axes():
    v = vector_adapter().reference_vectors(points())
    assert v[1, 2] > .07  # Fingers extend along V5 +Z from the MCP plane.
    assert v[1, 1] > v[2, 1] > v[3, 1]  # Index -> ring along -Y.
    np.testing.assert_allclose(OPERATOR2ALLEGRO_RIGHT.T @ OPERATOR2ALLEGRO_RIGHT, np.eye(3))
    assert np.linalg.det(OPERATOR2ALLEGRO_RIGHT) == 1


def test_reference_vectors_ignore_global_wrist_rotation_and_translation():
    theta = .7
    rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    p = points()
    adapter = vector_adapter()
    np.testing.assert_allclose(adapter.reference_vectors(p),
        adapter.reference_vectors(p @ rotation.T + [1.2, -.5, .3]), atol=1e-10)


def test_reference_vectors_match_robot_palm_origin_and_preserve_pinch():
    from paradex.retargetor.inspire_dexpilot import estimate_mano_wrist_frame
    r = vector_adapter()
    r.human_indices = np.array([[8, 12, 16, 0, 0, 0, 0], [4, 4, 4, 4, 8, 12, 16]])
    p = points()
    local = (p - p[0]) @ estimate_mano_wrist_frame(p) @ OPERATOR2ALLEGRO_RIGHT
    before = local[r.human_indices[1]] - local[r.human_indices[0]]
    after = r.reference_vectors(p)
    np.testing.assert_allclose(after[:3], before[:3])
    np.testing.assert_allclose(
        after[3:] * r.scaling_factor,
        (local[[4, 8, 12, 16]] - local[9]) * r.scaling_factor + r.robot_middle_mcp,
    )


def test_finger_targets_do_not_depend_on_human_palm_length():
    short = points()
    short[9, 1] = 0.  # Middle MCP lies on the forward palm axis.
    long = short.copy()
    # Change only wrist-to-knuckle distance; finger geometry stays identical.
    long[1:, 0] -= .05
    r = vector_adapter()
    np.testing.assert_allclose(r.reference_vectors(short), r.reference_vectors(long), atol=1e-10)


def test_matching_robot_mcp_with_tip_recovers_robot_mcp_position():
    p = points()
    p[12] = p[9]
    r = vector_adapter()
    # Middle tip at middle MCP: target must be the model's MCP, not the wrist
    # plus the whole human palm (which makes a fist remain partly open).
    target = r.reference_vectors(p)[2] * r.scaling_factor
    np.testing.assert_allclose(target, r.robot_middle_mcp, atol=1e-10)


@pytest.mark.parametrize('bad', [np.zeros((21, 3)), np.full((21, 3), np.nan), np.ones((20, 3))])
def test_reject_invalid_tracking(bad):
    with pytest.raises(ValueError):
        validate_keypoints(bad)


def test_joint_order_and_limits_apply_after_optimizer_reordering():
    r = vector_adapter()
    r.command_indices = np.arange(16)[::-1]
    r.fixed_qpos = np.zeros(0)
    r.limits = joint_limits()
    qpos = np.linspace(-5, 5, 16)
    r.retargeter = SimpleNamespace(retarget=lambda *args, **kwargs: qpos)
    actual = r.retarget_keypoints(points())
    np.testing.assert_allclose(actual, np.clip(qpos[::-1], r.limits[:, 0], r.limits[:, 1]))
    assert JOINT_NAMES == tuple(f'joint_{i}_0' for i in range(16))


def test_reject_nonfinite_optimizer_output():
    r = vector_adapter()
    r.command_indices = np.arange(16)
    r.fixed_qpos = np.zeros(0)
    r.retargeter = SimpleNamespace(retarget=lambda *args, **kwargs: np.full(16, np.nan))
    with pytest.raises(RuntimeError, match='invalid joint target'):
        r.retarget_keypoints(points())


@pytest.mark.parametrize('side', ['Left', 'Bimanual'])
def test_reject_left_v5_model_before_worker_starts(side):
    with pytest.raises(ValueError, match='right hand only'):
        Retargetor(hand_name='allegro_v5', hand_side=side, allegro_retarget='dexpilot')


def test_existing_direct_mode_is_preserved():
    from paradex.retargetor.hand_regargetor import allegro_v5
    r = Retargetor(hand_name='allegro_v5', hand_side='Right')
    assert r.hand_retargetor is allegro_v5


def test_retargetor_forwards_glove_and_composes_vive_arm_pose(monkeypatch):
    import paradex.retargetor.allegro_v5_dexpilot as module
    calls = []
    class Worker:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def __call__(self, hand_frame, ergonomics=None):
            calls.append(hand_frame)
            return np.arange(16)
        def close(self):
            calls.append('closed')
    monkeypatch.setattr(module, 'AsyncAllegroV5DexPilotRetargeter', Worker)
    r = Retargetor(arm_name='xarm', hand_name='allegro_v5', hand_side='Right',
                  teleop_name='vive', hand_scale=1.2, allegro_retarget='dexpilot')
    r.start(np.eye(4))
    glove = frame()
    wrist, command = r.get_action({'Right': glove})
    np.testing.assert_allclose(wrist, np.eye(4))
    glove['wrist'][:3, 3] = [.1, .2, .3]
    wrist2, _ = r.get_action({'Right': glove})
    np.testing.assert_allclose(wrist2[:3, 3], r.device2global[:3, :3] @ [.1, .2, .3])
    np.testing.assert_array_equal(command, np.arange(16))
    assert calls[0] == {'scaling_factor': 1.2}
    r.close()
    assert calls[-1] == 'closed'


def test_real_optimizer_produces_bounded_driver_order_commands():
    pytest.importorskip('dex_retargeting')
    r = AllegroV5DexPilotRetargeter()
    limits = joint_limits()
    # Pinocchio order differs from Allegro driver order (thumb comes second).
    assert r.command_indices != list(range(16))
    for _ in range(5):
        command = r(frame())
        assert command.shape == (16,)
        assert np.all(np.isfinite(command))
        assert np.all(command >= limits[:, 0]) and np.all(command <= limits[:, 1])


def test_real_async_worker_recovers_after_pause_and_cleans_up():
    pytest.importorskip('dex_retargeting')
    r = AsyncAllegroV5DexPilotRetargeter()
    try:
        assert np.all(np.isfinite(r(frame())))
        time.sleep(1.05)
        assert np.all(np.isfinite(r(frame())))
        for _ in range(30):
            assert np.all(np.isfinite(r(frame())))
            time.sleep(.005)
        r.process.terminate()
        r.process.wait(timeout=3)
        with pytest.raises(RuntimeError, match='exited'):
            r(frame())
    finally:
        r.close()
    assert not r._reader.is_alive()


def test_check_waits_for_missing_input_without_enabling_commands(monkeypatch):
    import sys
    import paradex.retargetor.allegro_v5_dexpilot as module
    import paradex.io.teleop.vive.receiver as receiver_module
    from tools import allegro_v5_teleop_check
    calls = []
    class Worker:
        def __init__(self, **kwargs):
            pass
        def __call__(self, hand):
            pytest.fail('Missing MANUS frame must never reach the optimizer')
        def close(self):
            calls.append('worker closed')
    class Receiver:
        def __init__(self, **kwargs):
            pass
        def get_data(self):
            return {'Right': None, 'Left': None}
        def get_diagnostics(self):
            return {}
        def end(self):
            calls.append('receiver closed')
    class Hand:
        def __init__(self, **kwargs):
            assert kwargs['command_enabled'] is False
        def get_data(self):
            return {'is_connected': True}
        def end(self):
            calls.append('hand closed')
    monkeypatch.setattr(module, 'AsyncAllegroV5DexPilotRetargeter', Worker)
    monkeypatch.setattr(receiver_module, 'ViveManusROSReceiver', Receiver)
    monkeypatch.setitem(sys.modules, 'paradex.io.robot_controller.allegro_v5_controller_ros2',
                        SimpleNamespace(AllegroController=Hand))
    monkeypatch.setattr(sys, 'argv', ['check', '--duration', '.001'])
    assert allegro_v5_teleop_check.main() == 2
    assert calls == ['worker closed', 'hand closed', 'receiver closed']


@pytest.fixture(scope='module')
def pinch_retargeter():
    pytest.importorskip('dex_retargeting')
    return AllegroV5DexPilotRetargeter()


@pytest.mark.parametrize('pair', range(6), ids=[
    'thumb-index', 'thumb-middle', 'thumb-ring', 'index-middle', 'index-ring', 'middle-ring'])
@pytest.mark.parametrize('distance,active', [(.0199, True), (.02, False), (.0201, False), (.029, False)])
def test_every_tip_pair_enters_pinch_strictly_below_two_cm(pinch_retargeter, pair, distance, active):
    r = pinch_retargeter
    opt = r.retargeter.optimizer
    reference = np.zeros((10, 3))
    reference[:3, 0] = .015  # Prerequisite thumb contacts for secondary pairs.
    reference[3:6, 0] = .04
    reference[6:, 2] = .1
    reference[pair, 0] = distance
    opt.projected.fill(False)
    opt.get_objective_function(reference, r.fixed_qpos, r.retargeter.last_qpos)
    assert bool(opt.projected[pair]) is active


@pytest.mark.parametrize('pair', range(3), ids=['thumb-index', 'thumb-middle', 'thumb-ring'])
def test_thumb_pinch_retains_existing_five_cm_release_hysteresis(pinch_retargeter, pair):
    r = pinch_retargeter
    opt = r.retargeter.optimizer
    reference = np.zeros((10, 3))
    reference[:, 0] = .07
    opt.projected.fill(False)
    for distance, active in [(.019, True), (.03, True), (.05, True), (.0501, False)]:
        reference[pair, 0] = distance
        opt.get_objective_function(reference, r.fixed_qpos, r.retargeter.last_qpos)
        assert bool(opt.projected[pair]) is active


@pytest.mark.parametrize('distance', [.015, .06])
def test_local_objective_matches_upstream_loss_and_gradient_outside_changed_band(pinch_retargeter, distance):
    from dex_retargeting.optimizer import DexPilotOptimizer
    from paradex.retargetor.allegro_v5_dexpilot import JOINT_NAMES, TIP_LINKS
    r = pinch_retargeter
    local = r.retargeter.optimizer
    upstream = DexPilotOptimizer(local.robot, list(JOINT_NAMES),
        finger_tip_link_names=list(TIP_LINKS), wrist_link_name='palm_link', scaling=r.scaling_factor)
    reference = np.zeros((10, 3))
    reference[:6, 0] = distance
    reference[6:, 2] = .1
    local.projected.fill(False)
    qpos = r.retargeter.last_qpos.copy()
    local_objective = local.get_objective_function(reference, r.fixed_qpos, qpos)
    upstream_objective = upstream.get_objective_function(reference, r.fixed_qpos, qpos)
    local_gradient, upstream_gradient = np.zeros(16), np.zeros(16)
    np.testing.assert_allclose(local_objective(qpos, local_gradient), upstream_objective(qpos, upstream_gradient))
    np.testing.assert_allclose(local_gradient, upstream_gradient)
    np.testing.assert_array_equal(local.projected, upstream.projected)
