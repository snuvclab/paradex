"""Grasp episode runner: pre-grasp (IK) -> 70 policy steps at 5 Hz -> scripted vertical lift -> hold -> (release).

  sim : python -m robustdex_deploy.runner --export EXPORT_DIR --backend sim --scripts-dir SCRIPTS --object 006_mustard_bottle --episodes 20
  real: python -m robustdex_deploy.runner --export EXPORT_DIR --backend real [--allow-live] [--scale 0.5] [--dry-run]

Safety on the real robot: nothing is published without --allow-live; the per-step arm change is hard-capped; the
predicted pose of every hand/arm frame must stay above the table; the wrist must stay inside a workspace box; the
object cloud / joint state must be fresh; Esc (pynput) or Ctrl-C holds the robot immediately."""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np

from .cloud import farthest_point_sample
from .kinematics import FKModel
from .observation import ObsBuilder
from .policy import Policy
from .pregrasp import pregrasp_arm

WORKSPACE = dict(x=(0.15, 0.95), y=(-0.55, 0.55), z=(0.0, 0.70))   # wrist position box, robot base frame (m)
# 17 probe frames -> 13 contact slots: wrist->palm(0); finger f: link1->1+3f, link2->2+3f, link3 & tip->3+3f
FRAME_SLOT = np.array([0] + sum([[1 + 3 * f, 2 + 3 * f, 3 + 3 * f, 3 + 3 * f] for f in range(4)], []))


class Abort(Exception):
    pass


class EscListener:
    """Global Esc key (pynput, needs an X display). pynput's Listener.start() can block forever when the X server is not
    reachable from this terminal, so it is started from a helper thread and simply reported as unavailable on timeout."""

    def __init__(self, timeout=2.0):
        self.hit = False
        self._l = None
        ready = threading.Event()

        def start():
            try:
                from pynput import keyboard
                self._l = keyboard.Listener(on_press=lambda k: setattr(self, 'hit', True) if k == keyboard.Key.esc else None)
                self._l.start()
                ready.set()
            except Exception as e:
                print('[runner] no keyboard listener (%s); use Ctrl-C to abort' % e, flush=True)
        threading.Thread(target=start, daemon=True).start()
        if ready.wait(timeout):
            print('[runner] Esc = hold + abort (keyboard listener active)', flush=True)
        else:
            print('[runner] keyboard listener not ready (no X display?); use Ctrl-C to abort', flush=True)


class Runner:
    def __init__(self, a):
        self.a = a
        self.meta = json.load(open(os.path.join(a.export, 'meta.json')))
        self.fk = FKModel(os.path.join(a.export, 'fk_xarm_allegro.xml'), self.meta)
        self.obs = ObsBuilder(self.fk, self.meta)
        self.policy = Policy(os.path.join(a.export, 'policy.npz'), self.meta)
        self.table_auto = str(a.table_offset).lower() == 'auto'
        self.table_z = float(self.meta['table_height']) + (0.0 if self.table_auto else float(a.table_offset))
        self.obs.table_z = self.table_z
        self.init_finger = np.array(self.meta['init_finger'])
        self.ep = self.meta['episode']
        if a.backend == 'sim':
            from .backends import SimBackend
            self.be = SimBackend(a.scripts_dir, a.object, seed=a.seed, contact=a.contact, obj_cat=a.object_cat,
                                 hand_visual=bool(a.render))
        else:
            from .backends import RealBackend
            self.be = RealBackend(namespace=a.namespace, cloud_endpoint=a.cloud_endpoint, allow_live=a.allow_live,
                                  tactile_contact=a.contact == 'tactile')
        self.esc = EscListener() if a.backend == 'real' else None
        self.renderer = None
        if a.render and a.backend == 'sim':
            from .backends import SimRenderer
            self.renderer = SimRenderer(self.be, a.render)
        os.makedirs(a.log_dir, exist_ok=True)
        self.log = open(os.path.join(a.log_dir, time.strftime('run_%Y%m%d_%H%M%S.jsonl')), 'a')
        self.pub = None
        if a.state_pub and a.backend == 'real':          # viewer overlay (needs pyzmq); sim runs have the mp4 renderer
            try:
                from .cloud import StatePublisher
                self.pub = StatePublisher(a.state_pub)
            except ImportError as e:
                print('[runner] state publisher disabled (%s)' % e, flush=True)

    # ------------------------------------------------------------------ helpers
    def record(self, **kw):
        self.log.write(json.dumps(kw, default=lambda v: np.asarray(v).tolist()) + '\n')
        self.log.flush()

    _last_pts = None
    _last_cloud_id = None

    def points(self):
        raw = self.be.object_points()
        cid = getattr(self.be, 'cloud', None) and self.be.cloud.latest[2]     # receive time of the latest message
        if self._last_pts is None or cid is None or cid != self._last_cloud_id:
            self._last_pts = farthest_point_sample(raw, self.meta['n_object_points'], seed=0)
            self._last_cloud_id = cid
        return self._last_pts

    def check_abort(self):
        if self.esc is not None and self.esc.hit:
            raise Abort('Esc pressed')
        if self.a.backend == 'real':
            if self.be.error():
                raise Abort('controller error')
            if self.be.state_age() > 0.25:
                raise Abort('joint state stale (%.2fs)' % self.be.state_age())

    def check_target(self, target, q, max_delta=None):
        """Predicted pose of the commanded target must keep every frame above the table and the wrist in the box."""
        max_delta = self.a.max_arm_delta if max_delta is None else max_delta
        k = self.fk.fk(target)
        zmin = min(k['frames'][:, 2].min(), k['arm'][:, 2].min())
        if zmin < self.table_z - self.a.table_margin:
            if self.a.dry_run:   # virtual rollout has no contact: report, count, continue
                self.n_table_viol += 1
                if self.n_table_viol <= 3:
                    print('[dry-run] would ABORT (live): frame below table, z=%.3f (table %.3f)' % (zmin, self.table_z), flush=True)
            else:
                raise Abort('frame below table: z=%.3f' % zmin)
        w = k['wrist']
        for i, ax in enumerate('xyz'):
            if not WORKSPACE[ax][0] <= w[i] <= WORKSPACE[ax][1]:
                raise Abort('wrist out of workspace: %s=%.3f' % (ax, w[i]))
        if np.abs(target[:6] - q[:6]).max() > max_delta + 1e-6 and self.a.backend == 'real':
            raise Abort('arm step too large')

    def contact(self, q, target):
        c = self.be.contact()
        if c is not None:
            return c
        if self.a.contact == 'zero':
            return np.zeros(13), np.zeros(13)
        from .contact_proxy import proxy_np
        con, imp = proxy_np(q[6:], target[6:])
        if self.a.contact == 'dist' and self.obs.min_dist is not None:
            d = np.full(13, np.inf)
            np.minimum.at(d, FRAME_SLOT, self.obs.min_dist)
            con = (d < self.a.dist_on).astype(float)
        if self.a.contact == 'tactile':
            lv = self.be.tactile_levels()
            if lv is not None:     # fingertips: slots 3,6,9,12 (index, middle, ring, thumb)
                for f, s in enumerate((3, 6, 9, 12)):
                    if lv[f] > self.a.tactile_on:
                        con[s] = 1.0
                        imp[s] = max(imp[s], 0.05)
        return con, imp

    def publish(self, phase, step, q, target, con=None, **extra):
        if self.pub is None:
            return
        k = self.fk.fk(q)
        kt = self.fk.fk(target)
        md = self.obs.min_dist
        nearest = None if md is None or self._last_pts is None else self._nearest(k['frames'], self._last_pts)
        self.pub.send(phase=phase, step=step, q=q, target=target, virtual=self.virtual_q is not None, frames=k['frames'],
                      target_frames=kt['frames'], hand_center=k['hand_center'], nearest=nearest, contact=con, min_dist=md, **extra)

    @staticmethod
    def _nearest(frames, pts):
        d = np.linalg.norm(frames[:, None, :] - pts[None], axis=-1)
        return pts[d.argmin(1)]

    def confirm(self, msg):
        if self.a.backend != 'real' or self.a.yes:
            return
        print('%s  [Enter to continue / q to abort] ' % msg, end='', flush=True)   # stdout may be a pipe (tee)
        if input().strip().lower() == 'q':
            raise Abort('user')

    def pace(self, t_next, tt=None):
        """Sleep until t_next; returns the (possibly resynchronised) schedule time. A stall longer than one period (e.g. a
        prompt) resets the schedule instead of letting the loop burst to catch up."""
        if not self.be.realtime:
            return t_next
        dt = t_next - time.monotonic()
        if dt > 0:
            time.sleep(dt)
        elif dt < -0.05:
            br = '' if not tt else ' (points %.0f obs %.0f policy %.0f publish %.0f log %.0f ms)' % tuple(
                1e3 * (tt[i + 1] - tt[i]) for i in range(5))
            print('[runner] overrun %.0f ms%s' % (-dt * 1000, br), flush=True)
            if dt < -self.ep['control_dt']:
                return time.monotonic()
        return t_next

    virtual_q = None   # dry-run: commanded targets are integrated kinematically instead of moving the robot
    TABLE_GAP = 0.005  # lowest fused object points vs table top (resting edge + stereo noise), m
    n_table_viol = 0

    def read_q(self):
        return self.virtual_q.copy() if self.virtual_q is not None else self.be.q()

    def send(self, target):
        if self.a.dry_run:
            self.virtual_q = np.asarray(target, float).copy()   # perfect tracking assumption
        else:
            self.be.send_target(target)

    # ------------------------------------------------------------------ episode
    def episode(self, idx):
        a, ep = self.a, self.ep
        info = self.be.reset()
        pts = self.points()
        q = self.be.q()
        raw = np.asarray(self.be.object_points(), float)
        z1, z50, z99 = np.percentile(raw[:, 2], [1, 50, 99])
        # table top from the lowest object points (cloud is in the robot base frame via C2R): the fused model reaches
        # down to the object's resting edge, which sits ~TABLE_GAP above the table
        est = float(z1 - self.TABLE_GAP)
        if self.table_auto and a.backend == 'real':
            self.table_z = est
            self.obs.table_z = est
        print('[ep %d] object cloud: %d pts, z 1%%/50%%/99%% = %.3f / %.3f / %.3f m -> table top estimate %.3f (sim %.3f, '
              'offset %+.3f); using table z = %.3f (%s)' % (idx, len(raw), z1, z50, z99, est, self.meta['table_height'],
                                                             est - self.meta['table_height'], self.table_z,
                                                             'auto from cloud' if self.table_auto and a.backend == 'real' else 'configured'), flush=True)
        if abs(est - self.table_z) > 0.015:
            print('[ep %d] WARNING: cloud-based table estimate and the table z in use differ by %.1f cm' % (idx, 100 * abs(est - self.table_z)), flush=True)
        self.record(ep=idx, event='cloud', n=len(raw), z1=z1, z50=z50, z99=z99, table_est=est, table_z=self.table_z)
        # -- pre-grasp
        if a.backend == 'sim' and a.pregrasp == 'sim':
            q0, pg = q.copy(), dict(source='sim', **(info or {}))
        else:
            q_arm, pg = pregrasp_arm(self.fk, pts, self.init_finger, self.meta, min_frame_z=self.table_z + 0.01)
            if q_arm is None:
                print('[ep %d] no feasible pre-grasp' % idx, pg)
                self.record(ep=idx, event='no_pregrasp', **pg)
                return None
            q0 = np.r_[q_arm, self.init_finger]
            pg['source'] = 'deploy'
        self.check_target(q0, q0)
        self.publish('pregrasp', 0, q, q0, pregrasp=dict(centre=pg.get('centre'), palm_pos=pg.get('palm_pos'), feasible=pg.get('feasible')))
        print('[ep %d] pre-grasp' % idx, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pg.items() if k != 'candidates'})
        self.confirm('move to pre-grasp (arm %s)' % np.round(q0[:6], 3))
        if a.dry_run:
            self.virtual_q = q0.copy()          # virtual rollout starts at the pre-grasp pose
        else:
            self.be.move_to(q0, duration=a.pregrasp_seconds)
        q = self.read_q()
        if a.backend == 'real' and not a.dry_run:
            # arm: stiff servo, must be there. hand: soft PD (kp 2 Nm/rad), give it time to settle and accept a residual
            t0 = time.monotonic()
            while time.monotonic() - t0 < 4.0 and np.abs(q[6:] - q0[6:]).max() > 0.15:
                time.sleep(0.1)
                q = self.read_q()
            err = np.abs(q - q0)
            print('[ep %d] pre-grasp reached: arm max err %.3f rad (joint %d), hand max err %.3f rad (joint %d)' % (
                idx, err[:6].max(), err[:6].argmax() + 1, err[6:].max(), err[6:].argmax()), flush=True)
            if err[:6].max() > 0.1:
                raise Abort('pre-grasp not reached by the arm: joint %d err %.3f rad' % (err[:6].argmax() + 1, err[:6].max()))
            if err[6:].max() > 0.4:
                raise Abort('hand far from the pre-grasp pose: joint %d err %.3f rad (blocked finger?)' % (err[6:].argmax(), err[6:].max()))
        target = np.r_[q[:6], q0[6:]]
        self.obs.start(q)
        self.record(ep=idx, event='start', q=q, target=target, pregrasp=pg, points=len(pts))
        if self.renderer:
            self.renderer.begin(idx, self.be.obj, self.a.contact)
        self.confirm('start policy (%d steps, scale %.2f)' % (a.steps, a.scale))
        # -- grasp (policy)
        t_next = time.monotonic()
        last_a = np.zeros(22)
        tau0 = self.be.arm_torque() if a.backend == 'real' else None     # gravity/bias torque at the pre-grasp pose
        self.z_contact, n_press = None, 0
        for step in range(a.steps):
            tt = [time.perf_counter()]
            self.check_abort()
            q = self.read_q()
            pts = self.points(); tt.append(time.perf_counter())
            self.obs.build(q, target, pts, np.zeros(13), np.zeros(13))   # refresh frame->object distances for 'dist'
            con, imp = self.contact(q, target)
            o = self.obs.build(q, target, pts, con, imp); tt.append(time.perf_counter())
            last_a = self.policy.act(o)
            new_t = self.policy.targets(last_a, q, target, scale=a.scale, max_arm_delta=a.max_arm_delta)
            # virtual arm compliance (real robot): the sim arm is a soft PD (kp 200) that yields when it presses, the
            # real xArm is a stiff position servo. Once the hand touches the object, do not let the policy drive the
            # hand centre deeper than --press-depth below the height of first contact: hold the arm, fingers continue.
            pressed = False
            tau = self.be.arm_torque() if a.backend == 'real' else None
            dtau = None if tau is None or tau0 is None else tau - tau0
            if a.backend == 'real':
                if self.z_contact is None and con.sum() > 0:
                    self.z_contact = float(self.fk.fk(q)['hand_center'][2])
                if self.z_contact is not None and self.fk.fk(new_t)['hand_center'][2] < self.z_contact - a.press_depth:
                    new_t[:6] = target[:6]; pressed = True; n_press += 1
                if dtau is not None and a.tau_soft > 0 and np.abs(dtau).max() > a.tau_soft:
                    # the arm is loading the object/table (joint torque jump): back the wrist off upwards a little
                    kt = self.fk.fk(target)
                    q_up, _ = self.fk.solve_ik(kt['wrist'] + [0, 0, a.backoff], kt['Rw'], target[:6], new_t[6:], iters=30)
                    new_t[:6] = q_up; pressed = True; n_press += 1
                if dtau is not None and a.max_dtau > 0 and np.abs(dtau).max() > a.max_dtau:
                    raise Abort('arm joint torque change %.1f Nm > %.1f (external force)' % (np.abs(dtau).max(), a.max_dtau))
            self.check_target(new_t, q); tt.append(time.perf_counter())
            if self.renderer:
                self.renderer.status('grasp', step + 1, con)
            self.publish('grasp', step + 1, q, new_t, con, action_norm=float(np.abs(last_a).mean())); tt.append(time.perf_counter())
            self.send(new_t)
            he = self.be.hand_effort()
            self.record(ep=idx, step=step, q=q, target=new_t, action=last_a, contact=con, impulse=imp, pressed=pressed,
                        arm_torque=tau, dtau=dtau, hand_effort=he, min_dist=self.obs.min_dist, obs=o if a.log_obs else None,
                        t_ms=dict(points=1e3 * (tt[1] - tt[0]), obs=1e3 * (tt[2] - tt[1]), policy_check=1e3 * (tt[3] - tt[2]),
                                  publish=1e3 * (tt[4] - tt[3]))); tt.append(time.perf_counter())
            if step % 10 == 0 or pressed and n_press <= 3:
                print('[ep %d] step %d  contact %s  min_dist %.3f  |a| %.2f%s%s%s  (compute %.0f ms)' % (
                    idx, step, ''.join(str(int(c)) for c in con), float(self.obs.min_dist.min()), float(np.abs(last_a).mean()),
                    '  dtau_max %.1f Nm' % np.abs(dtau).max() if dtau is not None else '',
                    '  hand|tau| mean %.2f max %.2f Nm' % (np.nanmean(np.abs(he)), np.nanmax(np.abs(he))) if he is not None else '',
                    '  PRESS-LIMITED' if pressed else '', 1e3 * (tt[-1] - tt[0])), flush=True)
            target = new_t
            t_next = self.pace(t_next + ep['control_dt'], tt)
        # -- lift (scripted wrist rise, hand keeps applying the last action, as in training eval)
        if a.no_lift:
            return self.finish(idx, target, None)
        self.confirm('lift %.2f m' % ep['lift_rise'])
        t_next = time.monotonic()                    # the prompt may have taken seconds: restart the 5 Hz schedule
        q = self.read_q()
        k = self.fk.fk(q)
        p0, R0 = k['wrist'].copy(), k['Rw'].copy()
        q_arm = q[:6].copy()
        rise, nr, nh = ep['lift_rise'], ep['lift_rise_steps'], ep['lift_hold_steps']
        for kstep in range(nr + nh):
            self.check_abort()
            dz = rise * min(1.0, (kstep + 1) / nr)
            q_arm, ok = self.fk.solve_ik(p0 + np.array([0, 0, dz]), R0, q_arm, target[6:], iters=60)
            new_t = target.copy()
            new_t[:6] = q_arm
            new_t[6:] = np.clip(target[6:] + last_a[6:] * self.policy.std[6:] * a.scale, self.policy.lo[6:], self.policy.hi[6:])
            self.check_target(new_t, self.read_q(), max_delta=0.08)   # 6 mm/step wrist rise -> small joint steps
            if self.renderer:
                self.renderer.status('lift', kstep + 1, None)
            self.publish('lift', kstep + 1, self.read_q(), new_t, None)
            self.send(new_t)
            target = new_t
            t_next = self.pace(t_next + ep['control_dt'])
        return self.finish(idx, target, True)

    def finish(self, idx, target, lifted):
        a = self.a
        res = dict(ep=idx, event='end', lifted=lifted, table_violations=self.n_table_viol)
        if a.dry_run:
            print('[ep %d] dry-run finished; %d step(s) would have tripped the table guard live' % (idx, self.n_table_viol), flush=True)
        self.n_table_viol = 0
        self.publish('done', 0, self.read_q(), target, None)
        self.virtual_q = None
        if a.backend == 'sim':
            res['object_lift'] = self.be.object_lift()
            res['success'] = bool(res['object_lift'] > 0.1)
            print('[ep %d] object lift %.3f m -> %s' % (idx, res['object_lift'], 'SUCCESS' if res['success'] else 'fail'))
        else:
            print('[ep %d] done; holding.' % idx)
            if a.release:
                self.confirm('release: open the hand')
                if not a.dry_run:
                    q = self.be.q()
                    self.be.move_to(np.r_[q[:6], self.init_finger], duration=3.0)
        self.record(**res)
        return res

    def run(self):
        results = []
        try:
            print('[runner] connecting to the robot / object cloud ...', flush=True)
            self.be.connect()
            print('[runner] connected: q =', np.round(self.be.q(), 3).tolist(), flush=True)
            for i in range(self.a.episodes):
                try:
                    r = self.episode(i)
                except Abort as e:
                    print('[runner] ABORT:', e)
                    self.be.hold()
                    self.record(ep=i, event='abort', reason=str(e))
                    if self.a.backend == 'real':
                        break
                    r = None
                if r is not None:
                    results.append(r)
        except KeyboardInterrupt:
            print('[runner] Ctrl-C: holding the robot and shutting down (further Ctrl-C ignored)', flush=True)
            import signal
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                self.be.hold()
            except Exception as e:
                print('[runner] hold failed:', e)
        finally:
            if self.renderer:
                self.renderer.close()
            if self.pub:
                self.pub.close()
            self.be.close()
            self.log.close()
        if self.a.backend == 'sim' and results:
            s = [r['success'] for r in results if 'success' in r]
            print('SIM RESULT object=%s episodes=%d success=%.3f' % (self.a.object, len(s), np.mean(s)))
        return results


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True, help='directory from export_deploy.py (policy.npz, meta.json, fk xml)')
    ap.add_argument('--backend', choices=['sim', 'real'], default='sim')
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--steps', type=int, default=70)
    ap.add_argument('--scale', type=float, default=1.0, help='scale the policy step sizes (first real trials: 0.5)')
    ap.add_argument('--max-arm-delta', type=float, default=0.03, help='hard cap on per-step arm joint change (rad)')
    ap.add_argument('--contact', choices=['dist', 'proxy', 'tactile', 'sensor', 'zero'], default='dist',
                    help='dist: flag = frame-to-object distance < --dist-on, force = PD torque proxy (real & sim, best in ablation); '
                         'proxy: PD torque proxy for both; tactile: proxy + Allegro fingertip tactile; sensor: sim contact sensors; zero: none')
    ap.add_argument('--dist-on', type=float, default=0.02, help='contact distance threshold for --contact dist (m)')
    ap.add_argument('--render', default=None, help='sim backend: write an mp4 of the episodes to this path')
    ap.add_argument('--tactile-on', type=float, default=50.0)
    ap.add_argument('--pregrasp', choices=['deploy', 'sim'], default='deploy')
    ap.add_argument('--pregrasp-seconds', type=float, default=None)
    ap.add_argument('--table-offset', default='auto',
                    help="real table top z minus the sim value (0.018 m); 'auto' (default, real backend) estimates the table "
                         "top from the lowest object-cloud points (robot base frame via C2R) at each episode start")
    ap.add_argument('--table-margin', type=float, default=0.02, help='frame origins may dip this far below the table top (fingertip body origin sits ~1 cm inside the pad)')
    ap.add_argument('--press-depth', type=float, default=0.008,
                    help='real: max hand-centre descent below the height of first object contact (virtual arm compliance), m')
    ap.add_argument('--tau-soft', type=float, default=12.0,
                    help='real: xArm joint torque change (vs pre-grasp) above which the wrist backs off by --backoff per step (Nm); 0 = off')
    ap.add_argument('--backoff', type=float, default=0.002, help='real: wrist back-off per step while loaded (m)')
    ap.add_argument('--max-dtau', type=float, default=30.0,
                    help='real: abort if an xArm joint torque changes more than this from the pre-grasp value (Nm); 0 = off')
    ap.add_argument('--no-lift', action='store_true')
    ap.add_argument('--release', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='compute everything, send nothing')
    ap.add_argument('--yes', action='store_true', help='skip the confirmation prompts')
    ap.add_argument('--log-dir', default='logs/robustdex_deploy')
    ap.add_argument('--log-obs', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    # sim
    ap.add_argument('--scripts-dir', default=None, help='robustdex_mujoco_v1/scripts (sim backend)')
    ap.add_argument('--object', default='006_mustard_bottle')
    ap.add_argument('--object-cat', default='new_training_set', help='rsc/<cat>/<object> (e.g. hrdex for prep_hrdex_objects.py output)')
    # real
    ap.add_argument('--namespace', default='right')
    ap.add_argument('--cloud-endpoint', default='tcp://127.0.0.1:5571')
    ap.add_argument('--state-pub', default='tcp://127.0.0.1:5572', help='publish policy state for the stereo_track Viser overlay ("" to disable)')
    ap.add_argument('--allow-live', action='store_true', help='actually publish commands to the robot')
    a = ap.parse_args()
    if a.backend == 'sim' and not a.scripts_dir:
        ap.error('--scripts-dir is required for the sim backend')
    return a


def _main():
    import signal

    def on_term(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, on_term)          # SIGTERM -> same hold/cleanup path as Ctrl-C
    code = 0
    try:
        Runner(parse()).run()
    except SystemExit as e:
        code = e.code or 0
    except KeyboardInterrupt:
        print('[runner] interrupted during start-up', flush=True)
        code = 130
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush(); sys.stderr.flush()
    # ROS executor threads can outlive their shutdown; never let them keep a finished runner alive
    os._exit(code if isinstance(code, int) else 1)


if __name__ == '__main__':
    _main()
