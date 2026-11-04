"""Time-based position/SO(3) low-pass smoothing for already-validated poses."""
import numpy as np
from scipy.spatial.transform import Rotation


class PoseSmoother:
    def __init__(self, initial_pose, timestamp, tau_s=0.08):
        self.pose = np.asarray(initial_pose, dtype=float).copy()
        self.timestamp = float(timestamp)
        self.tau = float(tau_s)
        if not np.isfinite(self.tau) or self.tau < 0:
            raise ValueError('Smoothing time constant must be finite and nonnegative')

    def filter(self, target, timestamp):
        dt = min(max(float(timestamp)-self.timestamp, 0.), .05)
        self.timestamp = float(timestamp)
        alpha = 1. if self.tau == 0 else -np.expm1(-dt/self.tau)
        self.pose[:3, 3] += alpha*(target[:3, 3]-self.pose[:3, 3])
        delta = Rotation.from_matrix(self.pose[:3, :3].T @ target[:3, :3]).as_rotvec()
        self.pose[:3, :3] = self.pose[:3, :3] @ Rotation.from_rotvec(alpha*delta).as_matrix()
        return self.pose.copy()
