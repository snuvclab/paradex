"""Robot backends with one interface: q() -> 22 joint positions, object_points() -> object points in the robot base
frame, contact() -> optional (contact13, impulse13) override, send_target(target22) -> one 0.2 s control step with the
PD target linearly interpolated from the previous one (training's target_interp), move_to(q22, duration) -> slow
joint-space move.

SimBackend : the CPU MuJoCo training environment (TrainEnv) -- used to validate the whole deploy code path.
RealBackend: paradex ROS2 controllers (XArmControllerROS angle_j servo + AllegroController) + stereo_track cloud."""
import threading
import time

import numpy as np

CONTROL_DT = 0.2


class SimBackend:
    def __init__(self, scripts_dir, obj_name, seed=0, contact='proxy', uniform_start=True, obj_cat='new_training_set',
                 hand_visual=False):
        import sys
        sys.path.insert(0, scripts_dir)
        import train_env
        from train_mj import sample_start
        self.kw = dict(hand='allegro', hand_gains='real2sim', hand_visual=hand_visual, init_mode='ik', target_interp=True,
                       hand_target_base='prev_target', self_collision='hand')   # hand_visual: rgba only, no physics
        self.env = train_env.TrainEnv(obj_cat=obj_cat, obj_name=obj_name, seed=seed, **self.kw)
        self.rng = np.random.default_rng(seed)
        self.sample_start, self.uniform, self.contact_mode = sample_start, uniform_start, contact
        self.realtime = False
        self.obj = obj_name

    def connect(self):
        pass

    def reset(self):
        """New episode: object placed at a random start; the robot at the env's own IK pre-grasp (training rule)."""
        st = self.sample_start(self.rng, self.uniform)
        self.env.reset_to(*st)
        self.z0 = float(self.env.d.xpos[self.env.obj_body][2])
        return self.env.pregrasp_info

    def q(self):
        return self.env.d.qpos[self.env.robot_q].copy()

    def object_points(self):
        d = self.env.d
        Ro = d.xmat[self.env.obj_body].reshape(3, 3)
        return self.env.aff_pcd @ Ro.T + d.xpos[self.env.obj_body]

    def contact(self):
        if self.contact_mode != 'sensor':
            return None
        imp = np.linalg.norm(self.env.imp_obj, axis=1)
        return (imp > 0.01).astype(float), imp

    def send_target(self, target):
        env = self.env
        a = (np.asarray(target, float) - env.action_mean) / env.std   # step() reproduces exactly this target
        env.step(a)

    def move_to(self, q, duration=None):
        """Teleport (episode start) keeping the object where it is."""
        env = self.env
        pos = env.d.qpos[env.obj_q:env.obj_q + 3].copy()
        wxyz = env.d.qpos[env.obj_q + 3:env.obj_q + 7].copy()
        env.reset(obj_pose=(pos, np.r_[wxyz[1:], wxyz[0]]), robot_q=np.asarray(q, float))

    def object_lift(self):
        return float(self.env.d.xpos[self.env.obj_body][2]) - self.z0

    def hand_effort(self):
        """Hand actuator torques (Nm, 16, policy joint order): the sim PD output, force-limited to +-1 Nm."""
        env = self.env
        out = np.full(16, np.nan)
        for i, a in env.act_cols:
            if i >= 6:
                out[i - 6] = env.d.actuator_force[a]
        return out

    def hold(self):
        pass

    def close(self):
        pass


class RealBackend:
    ARM_SPEED, HAND_SPEED = 0.25, 0.6        # rad/s caps for move_to (pre-grasp / release)

    def __init__(self, namespace='right', cloud_endpoint='tcp://127.0.0.1:5571', rate_hz=100.0, allow_live=False,
                 tactile_contact=False, cloud_max_age=2.0):
        self.ns, self.rate, self.allow_live = namespace, rate_hz, allow_live
        self.cloud_endpoint = cloud_endpoint
        self.tactile_contact, self.cloud_max_age = tactile_contact, cloud_max_age
        self.realtime = True
        self.arm = self.hand = None
        self._lock = threading.Lock()
        self._seg = None          # (q_from, q_to, t0, duration)
        self._cmd = None          # last commanded 22
        self._stop = False
        self.n_sent = 0

    # ------------------------------------------------------------------ connection
    def connect(self):
        from .cloud import CloudClient
        self.cloud = CloudClient(self.cloud_endpoint)
        from paradex.io.robot_controller.allegro_v5_controller_ros2 import AllegroController
        from paradex.io.robot_controller.xarm_controller_ros import XArmControllerROS
        print('[backend] waiting for the object cloud on %s ...' % self.cloud_endpoint, flush=True)
        self.cloud.wait(30.0)
        print('[backend] cloud ok; creating the xArm controller (ROS2 services /%s/xarm/*, servo_api=angle_j) ...' % self.ns, flush=True)
        self.arm = self._with_timeout(lambda: XArmControllerROS(namespace=self.ns, servo_api='angle_j', fps=int(self.rate)),
                                      30.0, 'xArm controller (are the xArm ROS2 driver and its services up?)')
        print('[backend] xArm ok; creating the Allegro controller (/%s/allegroHand_0/*) ...' % self.ns, flush=True)
        self.hand = self._with_timeout(lambda: AllegroController(namespace=self.ns, tactile=self.tactile_contact), 30.0, 'Allegro controller')
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0 and not self.hand.connection_event.is_set():
            time.sleep(0.01)
        if not self.hand.connection_event.is_set():
            raise RuntimeError('no Allegro joint states received (is the allegro_hand driver running?)')
        q = self.q()
        if q.shape != (22,) or not np.all(np.isfinite(q)):
            raise RuntimeError('robot state not ready: %r' % q)
        self._cmd = q.copy()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def reset(self):
        return None

    @staticmethod
    def _with_timeout(fn, timeout, what):
        """Run a blocking constructor in a helper thread; a hang (e.g. ROS service never answering) becomes an error
        instead of an un-interruptible wait."""
        box = {}

        def target():
            try:
                box['v'] = fn()
            except BaseException as e:   # noqa: BLE001
                box['e'] = e
        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise RuntimeError('%s did not come up within %.0fs' % (what, timeout))
        if 'e' in box:
            raise box['e']
        return box['v']

    # ------------------------------------------------------------------ state
    def q(self):
        a, h = self.arm.get_data(), self.hand.get_data()
        return np.r_[np.asarray(a.get('qpos'), float).reshape(-1), np.asarray(h.get('qpos'), float).reshape(-1)]

    def state_age(self):
        a, h = self.arm.get_data(), self.hand.get_data()
        now = time.monotonic()
        ts = [a.get('state_monotonic_time'), h.get('state_monotonic_time')]
        return max(now - float(t) for t in ts if t is not None) if all(t is not None for t in ts) else np.inf

    def error(self):
        return bool(self.arm.is_error() or self.hand.is_error())

    def object_points(self):
        header, xyz, _ = self.cloud.poll() if self.cloud.latest is not None else self.cloud.wait()
        if self.cloud.age() > self.cloud_max_age:
            raise RuntimeError('object cloud is %.1fs old' % self.cloud.age())
        return xyz

    def hand_effort(self):
        """Allegro desired joint torque (Nm, 16) as published by the BHand driver in /joint_states effort, or None."""
        e = self.hand.get_data().get('effort')
        return None if e is None else np.asarray(e, float).reshape(-1)

    def arm_torque(self):
        """xArm joint torque estimate from /joint_states effort (Nm, 6) or None."""
        t = getattr(self.arm, 'latest_torque', None)
        return None if t is None else np.asarray(t, float).copy()

    def tactile_levels(self):
        t = self.hand.get_data().get('tactile')
        if t is None:
            return None
        v = np.abs(np.asarray(t, float).reshape(-1))
        return np.array([b.max() for b in np.split(v, 4)]) if v.size % 4 == 0 and v.size else None

    def contact(self):
        return None   # proxy; the runner may OR tactile fingertips in (see runner)

    # ------------------------------------------------------------------ commands
    def _loop(self):
        period = 1.0 / self.rate
        while not self._stop:
            t = time.monotonic()
            with self._lock:
                seg = self._seg
                if seg is not None:
                    q_from, q_to, t0, dur = seg
                    f = 1.0 if dur <= 0 else min(1.0, (t - t0) / dur)
                    cmd = q_from + f * (q_to - q_from)
                    self._cmd = cmd
                    if f >= 1.0:
                        self._seg = None
                else:
                    cmd = self._cmd
            if cmd is not None and self.allow_live:
                self.arm.move(cmd[:6])
                self.hand.move(cmd[6:])
                self.n_sent += 1
            time.sleep(max(0.0, period - (time.monotonic() - t)))

    def _start_segment(self, q_to, duration):
        with self._lock:
            q_from = self._cmd.copy()
            self._seg = (q_from, np.asarray(q_to, float).copy(), time.monotonic(), float(duration))

    def send_target(self, target):
        """Interpolate from the previous target to `target` over one control period (training's target_interp)."""
        self._start_segment(target, CONTROL_DT)

    def move_to(self, q, duration=None):
        cur = self._cmd.copy()
        dq = np.abs(np.asarray(q, float) - cur)
        need = max(dq[:6].max() / self.ARM_SPEED, dq[6:].max() / self.HAND_SPEED, 0.5)
        dur = need if duration is None else max(duration, need)
        self._start_segment(q, dur)
        time.sleep(dur + 0.1)

    def commanded(self):
        with self._lock:
            return None if self._cmd is None else self._cmd.copy()

    def hold(self):
        with self._lock:
            self._seg = None
            self._cmd = self.q() if self._cmd is None else self._cmd.copy()

    def close(self, timeout=6.0):
        """Stop our interpolation thread, then the controllers. Their loops are told to exit *first* (exit_event) so no
        servo call is issued after rclpy shuts down; end() runs in a helper thread with a timeout because the ROS
        executor shutdown can hang -- the runner then force-exits the process (daemon threads cannot keep it alive)."""
        self._stop = True
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        for c in (self.arm, self.hand):
            ev = getattr(c, 'exit_event', None)
            if ev is not None:
                ev.set()
        for name, c in (('hand', self.hand), ('arm', self.arm)):
            if c is None:
                continue
            t = threading.Thread(target=lambda c=c: c.end(), daemon=True)
            t.start()
            t.join(timeout)
            if t.is_alive():
                print('[backend] %s controller end() did not return in %.0fs; continuing' % (name, timeout), flush=True)


class SimRenderer:
    """Two-view mp4 (wide + object close-up) of SimBackend episodes with the phase / object lift / contact flags."""
    W, H, FPS, SUB = 560, 420, 25, 5

    def __init__(self, be, out):
        import os
        os.environ.setdefault('MUJOCO_GL', 'egl')
        import imageio.v2 as imageio
        import mujoco as mj
        self.be, self.mj = be, mj
        m = be.env.m
        m.vis.headlight.ambient[:] = 0.4; m.vis.headlight.diffuse[:] = 0.6
        self.rend, self.opt = mj.Renderer(m, self.H, self.W), mj.MjvOption()
        self.wide = mj.MjvCamera(); self.wide.lookat[:] = [0.45, 0.0, 0.2]
        self.wide.distance, self.wide.azimuth, self.wide.elevation = 1.35, 140, -22
        self.cam_close = mj.MjvCamera(); self.cam_close.distance, self.cam_close.azimuth, self.cam_close.elevation = 0.45, 200, -20
        self.writer = imageio.get_writer(out, fps=self.FPS, codec='libx264', quality=7, pixelformat='yuv420p', macro_block_size=8)
        self.every = be.env.nsub // self.SUB
        self.phase, self.step, self.con, self.title = 'pre-grasp', 0, None, ''
        be.env.on_substep = self._grab

    def begin(self, idx, obj, contact_mode):
        self.title = '#%d %s   contact=%s' % (idx, obj, contact_mode)
        self.o0 = self.be.env.d.xpos[self.be.env.obj_body].copy()
        self.phase, self.step = 'grasp', 0

    def status(self, phase, step, con):
        self.phase, self.step, self.con = phase, step, con

    def _label(self, img, lines):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        for i, (t, c) in enumerate(lines):
            dr.text((8, 6 + 16 * i), t, fill=c)
        return np.asarray(im)

    def _grab(self, k):
        if (k + 1) % self.every:
            return
        env, d = self.be.env, self.be.env.d
        self.cam_close.lookat[:] = 0.5 * (d.xpos[env.obj_body] + self.o0) + [0, 0, 0.04]
        imgs = []
        for cam in (self.wide, self.cam_close):
            self.rend.update_scene(d, cam, self.opt); imgs.append(self.rend.render())
        lift = 100 * self.be.object_lift()
        names = ['palm'] + ['%s%d' % (f, j) for f in ('ix', 'md', 'rg', 'th') for j in (1, 2, 3)]
        flags = '-' if self.con is None else (' '.join(n for n, c in zip(names, self.con) if c > 0.5) or 'none')
        lines = [(self.title, (255, 255, 255)),
                 ('%s step %3d   object lift %5.1f cm' % (self.phase, self.step, lift), (150, 255, 150) if lift > 10 else (255, 255, 255)),
                 ('contact flags (policy input): ' + flags, (255, 230, 120))]
        self.writer.append_data(np.concatenate([self._label(imgs[0], lines), imgs[1]], axis=1))

    def close(self):
        self.be.env.on_substep = None
        self.writer.close()
        self.rend.close()
