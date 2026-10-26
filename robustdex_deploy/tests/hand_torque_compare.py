"""Compare the hand torque proxy used in the policy input with what the hand actually does, from runner logs:
  real: BHand desired torque (joint_states effort)   vs   proxy 2*(target-q), per joint, per step
  sim : MuJoCo PD actuator force (+-1 Nm)            vs   proxy 2*(target-q)
Usage: hand_torque_compare.py run_real.jsonl [run_sim.jsonl]"""
import json
import sys

import numpy as np


def load(path):
    rows = [json.loads(l) for l in open(path) if '"step"' in l]
    q = np.array([r['q'][6:] for r in rows]); t = np.array([r['target'][6:] for r in rows])
    e = np.array([r.get('hand_effort') or [np.nan] * 16 for r in rows], float)
    return q, t, e


for path in sys.argv[1:]:
    q, t, e = load(path)
    proxy = 2.0 * (t - q)                       # Nm, sign: positive = closing (target ahead of q)
    ok = np.isfinite(e).all(1)
    print('\n==', path, '| %d steps, effort available in %d' % (len(q), ok.sum()))
    if ok.sum() == 0:
        print('   (no hand_effort logged)'); continue
    e, proxy, dq = e[ok], proxy[ok], (t - q)[ok]
    print('   |effort|   mean %.3f  p90 %.3f  max %.3f Nm' % (np.abs(e).mean(), np.percentile(np.abs(e), 90), np.abs(e).max()))
    print('   |proxy|    mean %.3f  p90 %.3f  max %.3f Nm' % (np.abs(proxy).mean(), np.percentile(np.abs(proxy), 90), np.abs(proxy).max()))
    m = np.abs(dq) > 0.02                        # fit effort = kp * (target - q) where the error is meaningful
    if m.sum() > 10:
        kp = np.sum(e[m] * dq[m]) / np.sum(dq[m] ** 2)
        r = np.corrcoef(e[m], dq[m])[0, 1]
        print('   fitted kp (effort vs target-q, |err|>0.02): %.2f Nm/rad  (policy proxy assumes 2.0), corr %.2f' % (kp, r))
    per = [np.sum(e[:, j] * dq[:, j]) / max(np.sum(dq[:, j] ** 2), 1e-9) for j in range(16)]
    print('   per-joint kp:', np.round(per, 2).tolist())
    print('   closing steps with |effort| >= 0.9 Nm (saturated):  %.0f%%' % (100 * (np.abs(e) >= 0.9).mean()))
