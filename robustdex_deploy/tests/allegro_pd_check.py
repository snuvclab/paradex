"""Hand-only PD identification on the real Allegro v5 (ROS2 `robot` env; the arm is not touched):
step one finger joint's target by +delta while you hold the fingertip still with a rigid object, read the driver's
desired torque (joint_states effort) and the tracking error, and print kp = effort / (target - q).
Expected if the sim's real2sim gains are right: ~2 Nm/rad, saturating at 1 Nm.

  python robustdex_deploy/tests/allegro_pd_check.py --allow-live [--joint 1] [--delta 0.2] [--seconds 3]
Without --allow-live it only prints the current state."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ap = argparse.ArgumentParser()
ap.add_argument('--namespace', default='right')
ap.add_argument('--joint', type=int, default=1, help='policy hand joint index 0-15 (1 = index finger MCP flexion)')
ap.add_argument('--delta', type=float, default=0.2, help='target step (rad)')
ap.add_argument('--seconds', type=float, default=3.0)
ap.add_argument('--allow-live', action='store_true')
a = ap.parse_args()

from paradex.io.robot_controller.allegro_v5_controller_ros2 import AllegroController  # noqa: E402
hand = AllegroController(namespace=a.namespace)
t0 = time.monotonic()
while time.monotonic() - t0 < 5 and not hand.connection_event.is_set():
    time.sleep(0.01)
d = hand.get_data()
q0 = np.asarray(d['qpos'], float)
print('hand q:', np.round(q0, 3).tolist())
print('effort now:', None if d.get('effort') is None else np.round(d['effort'], 3).tolist())
if not a.allow_live:
    hand.end(); sys.exit(0)
input('Block the fingertip of joint %d with something rigid, then press Enter (target +%.2f rad for %.1f s) ' % (a.joint, a.delta, a.seconds))
tgt = q0.copy(); tgt[a.joint] += a.delta
hand.move(tgt)
rows = []
t0 = time.monotonic()
while time.monotonic() - t0 < a.seconds:
    d = hand.get_data()
    q, e = np.asarray(d['qpos'], float), d.get('effort')
    if e is not None:
        rows.append((time.monotonic() - t0, tgt[a.joint] - q[a.joint], float(e[a.joint])))
    time.sleep(0.02)
hand.move(q0)
time.sleep(1.0)
hand.end()
r = np.array(rows)
print('\n t[s]  target-q[rad]  effort[Nm]  kp_est')
for t, err, eff in r[::max(1, len(r) // 12)]:
    print('%5.2f  %13.3f  %10.3f  %6.2f' % (t, err, eff, eff / err if abs(err) > 0.02 else float('nan')))
m = np.abs(r[:, 1]) > 0.02
if m.sum():
    print('\nfitted kp = %.2f Nm/rad (policy proxy assumes 2.0); max |effort| %.2f Nm' % (
        np.sum(r[m, 2] * r[m, 1]) / np.sum(r[m, 1] ** 2), np.abs(r[:, 2]).max()))
