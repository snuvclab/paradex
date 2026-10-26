"""Contact observation proxy computable on the real robot (no tactile needed): finger PD torque from the tracking
error of the position-controlled Allegro joints. Shared by the Warp training env (torch) and the deploy obs builder
(numpy) so sim and real compute bit-identical features.

Allegro v5 real2sim actuator: tau = clip(KP * (target - q) - kv * qd, -1, 1); we drop the (small) damping term.
Slot layout (13): 0 palm (no proxy -> 0), 1 + 3f + j <- finger f link j <- joint 4f + 1 + j (joint 4f is abduction).
contact flag: |tau| > TAU_ON ; impulse-like magnitude: |tau| * IMP_SCALE (1 Nm -> 0.1 == IMP_HIGH of the reward).
"""
import numpy as np

KP, TAU_MAX, TAU_ON, IMP_SCALE = 2.0, 1.0, 0.3, 0.1
SLOT_JOINT = np.array([4 * f + 1 + j for f in range(4) for j in range(3)])  # 12 joints feeding slots 1..12


def proxy_np(q_hand, target_hand):
    tau = np.abs(np.clip(KP * (np.asarray(target_hand) - np.asarray(q_hand)), -TAU_MAX, TAU_MAX))[..., SLOT_JOINT]
    z = np.zeros(tau.shape[:-1] + (1,), tau.dtype)
    return np.concatenate([z, (tau > TAU_ON).astype(tau.dtype)], -1), np.concatenate([z, tau * IMP_SCALE], -1)


def proxy_torch(q_hand, target_hand):
    import torch
    tau = (KP * (target_hand - q_hand)).clamp(-TAU_MAX, TAU_MAX).abs()[:, torch.as_tensor(SLOT_JOINT, device=q_hand.device)]
    z = torch.zeros(tau.shape[0], 1, device=tau.device, dtype=tau.dtype)
    return torch.cat([z, (tau > TAU_ON).to(tau.dtype)], 1), torch.cat([z, tau * IMP_SCALE], 1)
