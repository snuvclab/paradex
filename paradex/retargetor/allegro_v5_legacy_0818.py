"""Frozen Allegro V5 mapping from fe18720cf (2026-08-18).

Selected with --allegro-retarget legacy-0818. Constants and mapping functions
are copied from that commit, including its thumb signs/gains and raw output
without the later physical clipping. Current drivers/input handling stay active.
Keep this snapshot independent of current Inspire/Allegro tuning.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

SOURCE_COMMIT = "fe18720cf"

_ALLEGRO_V5_PHYSICAL_UPPER = np.array([0.47, 1.8, 1.709, 1.618] * 3 + [1.78, 1.78, 1.9, 1.8], dtype=np.float64)

_ALLEGRO_V5_RAW_ERGONOMIC_FIELDS = ('ThumbMCPStretch', 'ThumbMCPSpread', 'ThumbPIPStretch', 'ThumbDIPStretch', 'IndexSpread', 'IndexMCPStretch', 'IndexPIPStretch', 'IndexDIPStretch', 'MiddleSpread', 'MiddleMCPStretch', 'MiddlePIPStretch', 'MiddleDIPStretch', 'RingSpread', 'RingMCPStretch', 'RingPIPStretch', 'RingDIPStretch')

_ALLEGRO_V5_THUMB_BASE_FLEX_GAIN = 2.0

_ALLEGRO_V5_THUMB_SECONDARY_FLEX_GAIN = 1.65

_INSPIRE_THUMB_BEND_RANGE_DEG = 53.6 - -13.0

_INSPIRE_THUMB_SECONDARY_GAIN = 0.8

def _open_to_closed_command(flexion_deg, range_deg):
    normalized = np.clip(float(flexion_deg) / float(range_deg), 0.0, 1.0)
    return (1.0 - normalized) * 1000.0

def _inspire_thumb_secondary_fraction(flexion_deg):
    """Return Inspire's secondary-thumb command as a [0, 0.8] fraction."""
    return (1000.0 - _open_to_closed_command(flexion_deg, _INSPIRE_THUMB_BEND_RANGE_DEG)) * (_INSPIRE_THUMB_SECONDARY_GAIN / 1000.0)

def _allegro_v5_raw_from_manus_ergonomics(ergonomics):
    """Convert named MANUS joint angles to the legacy raw-v5 action order.

    VIVE's MANUS stream contains stable per-joint ergonomics in degrees.  The
    former raw extractor inferred these quantities from reconstructed 4x4
    transforms, which is both noisier and sensitive to the VIVE wrist frame.
    This is the direct equivalent of :func:`inspire_from_manus_ergonomics`:
    it uses the MANUS angles whenever the complete named payload is present.

    The non-thumb MCP bend keeps the legacy ``(angle - 0.35) * 1.5``
    calibration, while PIP/DIP retain their direct radians.  The thumb spread
    and distal bends preserve the signs/scales of the old Euler/rotation-matrix
    extraction.  Returning the existing index, middle, ring, thumb order is
    essential because the v5 ROS controller consumes that order directly.
    """
    if not isinstance(ergonomics, dict):
        raise ValueError('MANUS ergonomics must be a name-to-angle mapping')
    missing = [field for field in _ALLEGRO_V5_RAW_ERGONOMIC_FIELDS if field not in ergonomics]
    if missing:
        raise ValueError('MANUS ergonomics is missing Allegro v5 raw inputs: ' + ', '.join(missing))
    try:
        values = {field: float(ergonomics[field]) for field in _ALLEGRO_V5_RAW_ERGONOMIC_FIELDS}
    except (TypeError, ValueError) as error:
        raise ValueError('MANUS ergonomics contains a non-numeric Allegro input') from error
    if not np.all(np.isfinite(tuple(values.values()))):
        raise ValueError('MANUS ergonomics contains a non-finite Allegro input')
    action = np.zeros(16, dtype=np.float64)
    for index, finger in enumerate(('Index', 'Middle', 'Ring')):
        offset = index * 4
        action[offset] = np.deg2rad(values[f'{finger}Spread'])
        action[offset + 1] = (np.deg2rad(values[f'{finger}MCPStretch']) - 0.35) * 1.5
        action[offset + 2] = np.deg2rad(values[f'{finger}PIPStretch'])
        action[offset + 3] = np.deg2rad(values[f'{finger}DIPStretch'])
    action[12] = (_INSPIRE_THUMB_SECONDARY_GAIN - _inspire_thumb_secondary_fraction(values['ThumbMCPStretch'])) * _ALLEGRO_V5_PHYSICAL_UPPER[12] * _ALLEGRO_V5_THUMB_BASE_FLEX_GAIN
    action[12] = min(action[12], _ALLEGRO_V5_PHYSICAL_UPPER[12])
    action[13] = -np.deg2rad(values['ThumbMCPSpread']) * _ALLEGRO_V5_THUMB_SECONDARY_FLEX_GAIN - 1.57
    action[14] = min(np.sin(np.deg2rad(values['ThumbPIPStretch'])) * 1.2 * _ALLEGRO_V5_THUMB_SECONDARY_FLEX_GAIN, _ALLEGRO_V5_PHYSICAL_UPPER[14])
    action[15] = min(np.sin(np.deg2rad(values['ThumbDIPStretch'])) * 1.2 * _ALLEGRO_V5_THUMB_SECONDARY_FLEX_GAIN, _ALLEGRO_V5_PHYSICAL_UPPER[15])
    return action

def _allegro_v5_raw(hand_pose_frame, ergonomics=None):
    """Extract raw Allegro-v5 angles in retargeter action order.

    Prefer named MANUS ergonomics for VIVE teleoperation.  The transform
    implementation remains only as the compatibility fallback for older
    MANUS sources that do not publish ergonomics.
    """
    if ergonomics is not None:
        return _allegro_v5_raw_from_manus_ergonomics(ergonomics)
    allegro_angles = np.zeros(16)
    joint_name_list = ['metacarpal', 'proximal', 'intermediate', 'distal']
    for i, finger_name in enumerate(['index', 'middle', 'ring']):
        metacarpal = finger_name + '_metacarpal'
        distal = finger_name + '_distal'
        tip_position = (np.linalg.inv(hand_pose_frame['wrist']) @ hand_pose_frame[distal])[:3, 3]
        finger_base_position = (np.linalg.inv(hand_pose_frame['wrist']) @ hand_pose_frame[metacarpal])[:3, 3]
        tip_position = tip_position - finger_base_position
        tip_direction = tip_position / np.linalg.norm(tip_position)
        if tip_direction[1] > 0.9:
            allegro_angles[4 * i] = 0
        else:
            allegro_angles[4 * i] = np.arctan2(tip_direction[0], tip_direction[2]) * (0.9 - tip_direction[1])
        for j in range(3):
            parent_name = finger_name + '_' + joint_name_list[j]
            joint_name = finger_name + '_' + joint_name_list[j + 1]
            rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3, :3]) @ hand_pose_frame[joint_name][:3, :3]
            v = rot_mat[1, 1] if rot_mat[2, 1] >= 0 else 1
            v = max(-1, min(1, v))
            allegro_angles[4 * i + j + 1] = np.arccos(v)
        allegro_angles[4 * i + 1] = (allegro_angles[4 * i + 1] - 0.35) * 1.5
    thumb_meta = np.dot(hand_pose_frame['wrist'][:3, :3].T, hand_pose_frame['thumb_metacarpal'][:3, :3])
    thumb_meta_angle = R.from_matrix(thumb_meta).as_euler('xyz')
    allegro_angles[12] = thumb_meta_angle[0]
    allegro_angles[13] = -thumb_meta_angle[2] - 1.57
    for i, (parent_name, joint_name) in enumerate([('thumb_metacarpal', 'thumb_proximal'), ('thumb_proximal', 'thumb_distal')]):
        rot_mat = np.linalg.inv(hand_pose_frame[parent_name][:3, :3]) @ hand_pose_frame[joint_name][:3, :3]
        allegro_angles[14 + i] = rot_mat[2, 1] * 1.2
    return allegro_angles

def allegro_v5(hand_pose_frame, ergonomics=None):
    """Retarget MANUS data through the raw-v5 action contract.

    VIVE teleoperation supplies named ergonomic joint angles, so the raw
    mapper uses those directly.  Older transform-only MANUS inputs still use
    the fallback inside :func:`_allegro_v5_raw`.
    """
    return _allegro_v5_raw(hand_pose_frame, ergonomics=ergonomics)
