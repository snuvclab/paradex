"""MLP actor from the exported npz (Linear layers with LeakyReLU(0.01) in between) and the action -> PD-target rule of
training: arm target = current q + a * 0.005 rad, hand target = previous target + a * 0.015 rad, clipped to limits."""
import numpy as np


class Policy:
    def __init__(self, npz_path, meta):
        z = np.load(npz_path)
        idx = sorted({int(k.split('.')[0]) for k in z.files if k.endswith('.weight')})
        self.W = [(z[f'{i}.weight'], z[f'{i}.bias']) for i in idx]
        self.std = np.r_[np.full(6, meta['action_scale']['arm']), np.full(16, meta['action_scale']['hand'])]
        self.lo, self.hi = np.array(meta['joint_lo']), np.array(meta['joint_hi'])
        assert self.W[0][0].shape[1] == 153 and self.W[-1][0].shape[0] == 22

    def act(self, obs):
        x = np.asarray(obs, np.float32)
        for i, (w, b) in enumerate(self.W):
            x = w @ x + b
            if i < len(self.W) - 1:
                x = np.where(x > 0, x, 0.01 * x)
        return x.astype(np.float64)

    def targets(self, action, q, prev_target, scale=1.0, max_arm_delta=None):
        """Next PD target (22,). `scale` < 1 shrinks the policy's steps (first real trials); `max_arm_delta` hard-caps
        the per-step arm joint change (rad)."""
        a = np.asarray(action, float)
        delta = a * self.std * scale
        if max_arm_delta is not None:
            delta[:6] = np.clip(delta[:6], -max_arm_delta, max_arm_delta)
        base = np.r_[np.asarray(q, float)[:6], np.asarray(prev_target, float)[6:]]
        return np.clip(base + delta, self.lo, self.hi)
