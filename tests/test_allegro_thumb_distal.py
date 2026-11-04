import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from paradex.retargetor.hand_regargetor import (
    _ALLEGRO_V5_RAW_ERGONOMIC_FIELDS, allegro_v5, inspire,
)
from paradex.retargetor.unimanual import Retargetor
from paradex.retargetor.allegro_v5_legacy_0818 import allegro_v5 as legacy
from paradex.retargetor.experiment.allegro_retargeter_alignment_ui import _make_retargeter


def ergonomics(angle):
    values = dict.fromkeys(_ALLEGRO_V5_RAW_ERGONOMIC_FIELDS, 15.0)
    values["ThumbDIPStretch"] = angle
    return values


@pytest.mark.parametrize("angle", [-30, -2, 0, 15, 30, 60, 90, 120])
def test_linear_maps_dip_one_to_one_with_limits_and_keeps_other_joints(angle):
    inputs = ergonomics(angle)
    old = allegro_v5(None, ergonomics=inputs, thumb_distal_mapping="sine")
    linear = allegro_v5(None, ergonomics=inputs, thumb_distal_mapping="linear")
    assert linear[15] == pytest.approx(np.clip(np.deg2rad(angle), -0.09, 1.8))
    np.testing.assert_array_equal(linear[:15], old[:15])
    assert old[15] == pytest.approx(np.clip(np.sin(np.deg2rad(angle)) * 1.98, -0.09, 1.8))


def test_linear_has_no_early_plateau_or_reversal_through_ninety_degrees():
    result = [allegro_v5(None, ergonomics=ergonomics(angle), thumb_distal_mapping="linear")[15]
              for angle in range(91)]
    np.testing.assert_allclose(np.diff(result), np.deg2rad(1), atol=1e-15)


def test_transform_only_fallback_uses_signed_joint_angle():
    frame = {"wrist": np.eye(4)}
    for finger in ("index", "middle", "ring"):
        for index, joint in enumerate(("metacarpal", "proximal", "intermediate", "distal")):
            pose = np.eye(4)
            pose[:3, 3] = [0, index * 0.02, index * 0.01]
            frame[finger + "_" + joint] = pose
    for joint in ("metacarpal", "proximal", "distal"):
        frame["thumb_" + joint] = np.eye(4)
    frame["thumb_distal"][:3, :3] = Rotation.from_euler("x", 30, degrees=True).as_matrix()
    original = allegro_v5(frame, thumb_distal_mapping="sine")
    linear = allegro_v5(frame, thumb_distal_mapping="linear")
    assert linear[15] == pytest.approx(np.deg2rad(30))
    np.testing.assert_array_equal(linear[:15], original[:15])


@pytest.mark.parametrize("side", ["Right", "Bimanual"])
def test_capture_and_simulator_select_same_linear_mapping(side):
    options = dict(hand_side=side)
    if side == "Bimanual":
        options.update(arm_name="xarm", hand_name_left="inspire_dftp", hand_name_right="allegro_v5")
    else:
        options.update(hand_name="allegro_v5")
    rt = Retargetor(**options)
    mapper = rt.hand_retargetor["Right"] if side == "Bimanual" else rt.hand_retargetor
    if side == "Bimanual":
        assert rt.hand_retargetor["Left"].func is inspire
    preview = _make_retargeter("allegro_v5", "direct")
    data = ergonomics(30)
    np.testing.assert_array_equal(mapper(None, ergonomics=data), preview(None, ergonomics=data))
    assert mapper(None, ergonomics=data)[15] == pytest.approx(np.deg2rad(30))


def test_historical_mapping_is_not_changed_by_direct_curve_option():
    rt = Retargetor(hand_name="allegro_v5", allegro_retarget="legacy-0818",
                    allegro_thumb_distal_mapping="linear")
    assert rt.hand_retargetor is legacy


def test_invalid_curve_is_rejected():
    with pytest.raises(ValueError, match="sine or linear"):
        allegro_v5(None, thumb_distal_mapping="invalid")


def test_omitted_curve_uses_linear_and_sine_remains_selectable():
    data = ergonomics(30)
    assert allegro_v5(None, ergonomics=data)[15] == pytest.approx(np.deg2rad(30))
    rt = Retargetor(hand_name="allegro_v5", allegro_thumb_distal_mapping="sine")
    preview = _make_retargeter("allegro_v5", "direct", "sine")
    expected = np.sin(np.deg2rad(30)) * 1.98
    assert rt.hand_retargetor(None, ergonomics=data)[15] == pytest.approx(expected)
    assert preview(None, ergonomics=data)[15] == pytest.approx(expected)
