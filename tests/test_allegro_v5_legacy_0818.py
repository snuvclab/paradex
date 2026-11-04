"""Frozen August 18 mapping and selection, without hardware access."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from paradex.retargetor import allegro_v5_legacy_0818 as legacy
from paradex.retargetor.hand_regargetor import allegro_v5, inspire
from paradex.retargetor.unimanual import Retargetor


def transform_frame(seed):
    rng = np.random.RandomState(seed)
    frame = {"wrist": np.eye(4)}
    names = [finger + "_" + joint for finger in ("index", "middle", "ring")
             for joint in ("metacarpal", "proximal", "intermediate", "distal")]
    names += ["thumb_" + joint for joint in ("metacarpal", "proximal", "distal")]
    for name in names:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(rng.normal(size=3) * 0.3).as_matrix()
        pose[:3, 3] = rng.normal(size=3) * 0.05
        frame[name] = pose
    return frame


GOLDEN = json.loads((Path(__file__).parent / "fixtures/allegro_v5_0818.json").read_text())


@pytest.mark.parametrize("sample", GOLDEN["ergonomics"])
def test_matches_august_18_ergonomics_outputs(sample):
    actual = legacy.allegro_v5(None, ergonomics=sample["input"])
    np.testing.assert_allclose(actual, sample["output"], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("sample", GOLDEN["transforms"])
def test_matches_august_18_transform_outputs(sample):
    actual = legacy.allegro_v5(transform_frame(sample["seed"]))
    np.testing.assert_allclose(actual, sample["output"], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("side", ["Right", "Left", "Bimanual"])
def test_legacy_selection_changes_only_allegro_v5(side):
    if side == "Bimanual":
        rt = Retargetor(arm_name="xarm", hand_side=side, hand_name_left="inspire",
                        hand_name_right="allegro_v5", allegro_retarget="legacy-0818")
        assert rt.hand_retargetor["Left"].func is inspire
        assert rt.hand_retargetor["Right"] is legacy.allegro_v5
    else:
        rt = Retargetor(hand_name="allegro_v5", hand_side=side, allegro_retarget="legacy-0818")
        assert rt.hand_retargetor is legacy.allegro_v5


def test_legacy_selection_does_not_change_other_hand_models():
    rt = Retargetor(hand_name="inspire", allegro_retarget="legacy-0818")
    assert rt.hand_retargetor.func is inspire


def test_current_default_stays_direct():
    rt = Retargetor(hand_name="allegro_v5")
    assert rt.hand_retargetor is allegro_v5


def test_legacy_thumb_sign_and_unclipped_output_are_preserved():
    sample = GOLDEN["ergonomics"][1]
    old = legacy.allegro_v5(None, ergonomics=sample["input"])
    current = allegro_v5(None, ergonomics=sample["input"])
    assert old[13] < -1.57
    assert current[13] > 0


def test_legacy_requires_complete_ergonomics_like_original():
    with pytest.raises(ValueError, match="missing"):
        legacy.allegro_v5(None, ergonomics={})
