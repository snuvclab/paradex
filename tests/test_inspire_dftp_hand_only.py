"""Exercise real hand-only retargeting without robot or ROS connections."""

import numpy as np
import pytest

from paradex.retargetor.unimanual import Retargetor
from paradex.retargetor.hand_regargetor import inspire, inspire_from_manus_ergonomics


def _ergonomics(mcp=0.0, pip=0.0, dip=0.0):
    values = {
        f"{finger}{joint}Stretch": angle
        for finger in ("Pinky", "Ring", "Middle", "Index")
        for joint, angle in (("MCP", mcp), ("PIP", pip), ("DIP", dip))
    }
    values.update(ThumbMCPStretch=20.0, ThumbMCPSpread=20.0)
    return values


def test_left_thumb_curls_even_when_opposed_into_the_palm():
    values = _ergonomics(20.0, 20.0, 20.0)
    values.update(ThumbMCPSpread=35.0, ThumbDIPStretch=0.0)
    commands = []
    for bend in (0.0, 10.0, 30.0, 45.0, 90.0):
        values["ThumbDIPStretch"] = bend
        commands.append(inspire({}, is_right=False, ergonomics=values))
    commands = np.asarray(commands)

    np.testing.assert_allclose(commands[:, 5], 0.0)
    np.testing.assert_allclose(commands[:, 4], [1000.0, 1000.0 * 35 / 45, 1000.0 / 3, 0.0, 0.0])
    assert np.all(np.diff(commands[:4, 4]) < 0)
    assert np.all((commands >= 0.0) & (commands <= 1000.0))


def test_left_thumb_opposition_reaches_full_inward_travel():
    values = _ergonomics()
    values.update(ThumbDIPStretch=15.0)
    commands = []
    for spread in (-10.0, 0.0, 17.5, 35.0, 75.0):
        values["ThumbMCPSpread"] = spread
        commands.append(inspire({}, is_right=False, ergonomics=values))
    commands = np.asarray(commands)

    np.testing.assert_allclose(commands[:, 5], [1000.0, 1000.0, 500.0, 0.0, 0.0])
    np.testing.assert_allclose(commands[:, 4], 1000.0 * 30 / 45)


def test_left_thumb_uses_distal_curl_without_altering_right_or_other_fingers():
    values = _ergonomics(-20.0, 40.0, 20.0)
    values.update(ThumbPIPStretch=80.0, ThumbDIPStretch=15.0)
    right = inspire({}, is_right=True, ergonomics=values)
    left = inspire({}, is_right=False, ergonomics=values)

    np.testing.assert_allclose(left[:4], right[:4])
    assert left[4] == pytest.approx(1000.0 * 30 / 45)
    # The right hand keeps its existing mapping, including ignoring DIP/PIP.
    legacy = {k: v for k, v in values.items() if k not in ("ThumbPIPStretch", "ThumbDIPStretch")}
    np.testing.assert_allclose(right, inspire_from_manus_ergonomics(legacy))
    np.testing.assert_allclose(
        inspire({}, is_right=False, ergonomics=legacy), right,
    )


def test_left_thumb_uses_proximal_curl_when_distal_angle_is_missing():
    values = _ergonomics()
    values.update(ThumbPIPStretch=30.0)
    command = inspire({}, is_right=False, ergonomics=values)
    assert command[4] == pytest.approx(1000.0 / 3)
    values["ThumbDIPStretch"] = 0.0  # A straight distal joint is a valid reading.
    assert inspire({}, is_right=False, ergonomics=values)[4] == pytest.approx(1000.0)


def test_left_only_and_bimanual_use_the_same_tuned_thumb_mapping():
    values = _ergonomics()
    values.update(ThumbMCPSpread=35.0, ThumbDIPStretch=30.0)
    data = {
        "Left": {"wrist": np.eye(4)}, "Right": {"wrist": np.eye(4)},
        "ergonomics": {"Left": values, "Right": values},
    }
    single = Retargetor(hand_name="inspire_dftp", hand_side="Left")
    both = Retargetor(hand_name="inspire_dftp", hand_side="Bimanual")
    single.start(np.eye(4))
    both.start({"Left": np.eye(4), "Right": np.eye(4)})

    np.testing.assert_allclose(single.get_action(data)[1], both.get_action(data)[2])
    np.testing.assert_allclose(
        single.get_action(data)[1], inspire_from_manus_ergonomics(values, is_right=False),
    )


@pytest.mark.parametrize("side", ["Left", "Right", "Bimanual"])
def test_dftp_hand_only_initialization_and_actions_match_inspire(side):
    ergonomics = {
        f"{finger}{joint}Stretch": 20.0
        for finger in ("Pinky", "Ring", "Middle", "Index")
        for joint in ("MCP", "PIP", "DIP")
    }
    ergonomics.update(ThumbMCPStretch=10.0, ThumbMCPSpread=20.0)
    sides = ("Left", "Right") if side == "Bimanual" else (side,)
    data = {name: {"wrist": np.eye(4)} for name in sides}
    data["ergonomics"] = {name: ergonomics for name in sides}
    home = {name: np.eye(4) for name in sides} if side == "Bimanual" else np.eye(4)

    reference = Retargetor(hand_name="inspire", hand_side=side, teleop_name="vive")
    dftp = Retargetor(hand_name="inspire_dftp", hand_side=side, teleop_name="vive")
    reference.start(home)
    dftp.start(home)

    expected = reference.get_action(data)
    actual = dftp.get_action(data)
    for action, expected_action in zip(actual, expected):
        np.testing.assert_allclose(action, expected_action)
        assert np.all(np.isfinite(action))
    assert actual[-1].shape == (6,)
