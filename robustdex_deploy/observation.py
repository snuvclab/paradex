"""153-D policy observation rebuilt from real-robot quantities, matching warp_env.observe / mj_env.observe exactly:
q(22) | target-q(22) | contact(13) | impulse(13) | 17 frame heights | 6 arm link heights | hand centre (3) |
wrist euler diff from episode start (3) | wrist euler, unwrapped (3) | 17 frame -> nearest object point vectors (51).
Contact / impulse come from the PD-torque proxy (contact_proxy.py) unless caller-supplied values are given."""
import numpy as np

from . import contact_proxy
from .kinematics import raisim_euler


class ObsBuilder:
    def __init__(self, fk, meta):
        self.fk, self.meta = fk, meta
        self.table_z = float(meta['table_height'])
        self.R0 = None
        self.euler_prev = np.zeros(3)
        self.min_dist = None

    def start(self, q):
        """Call once at the start of the grasp episode (reference wrist orientation; euler unwrapping state)."""
        self.R0 = self.fk.fk(q)['Rw']
        self.euler_prev = np.zeros(3)

    def build(self, q, target, points, contact=None, impulse=None):
        q, target = np.asarray(q, float), np.asarray(target, float)
        k = self.fk.fk(q)
        if contact is None or impulse is None:
            contact, impulse = contact_proxy.proxy_np(q[6:], target[6:])
        frames, arm, Rw = k['frames'], k['arm'], k['Rw']
        e_cur = raisim_euler(Rw)
        if np.linalg.norm(self.euler_prev) > 0.01:
            e_cur = np.where(e_cur - self.euler_prev > np.pi, e_cur - 2 * np.pi, e_cur)
            e_cur = np.where(e_cur - self.euler_prev < -np.pi, e_cur + 2 * np.pi, e_cur)
        self.euler_prev = e_cur
        euler_diff = raisim_euler(self.R0.T @ Rw)
        pts = np.asarray(points, float)
        dist = np.linalg.norm(frames[:, None, :] - pts[None], axis=-1)        # 17 x P
        idx = dist.argmin(1)
        af_vec = pts[idx] - frames                                            # world frame (rotation-invariant)
        self.min_dist = dist[np.arange(17), idx]
        obs = np.concatenate([q, target - q, contact, impulse, frames[:, 2] - self.table_z, arm[:, 2] - self.table_z,
                              k['hand_center'], euler_diff, e_cur, af_vec.ravel()]).astype(np.float32)
        assert obs.shape == (153,), obs.shape
        return obs
