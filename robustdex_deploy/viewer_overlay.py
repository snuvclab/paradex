"""Viser overlay of the policy runner's state inside the stereo_track viewer (same process as the tracker, runs in the
base env): hand probe frames (red = contact flag on), frame -> nearest object point vectors (yellow), the predicted
frames of the commanded target (blue), pre-grasp palm target (green) and a status label. Data arrives over ZMQ from
robustdex_deploy.runner (--state-pub), so the two processes never block each other."""
import numpy as np

from .cloud import StateClient

FRAME_NAMES = ['wrist'] + ['%s%d' % (f, j) for f in ('ix', 'md', 'rg', 'th') for j in (1, 2, 3, 4)]
# 17 frames -> 13 contact slots (wrist->palm; link1,2,3,tip -> slots 1,2,3,3 of each finger)
FRAME_SLOT = [0] + sum([[1 + 3 * f, 2 + 3 * f, 3 + 3 * f, 3 + 3 * f] for f in range(4)], [])


ROBOT_URDF = 'rsc/robot/xarm_allegro_v5.urdf'   # paradex xArm6 + Allegro v5 (same joint names as the policy's 22-D q)


class PolicyOverlay:
    def __init__(self, server, endpoint='tcp://127.0.0.1:5572', urdf=ROBOT_URDF):
        self.server = server
        self.client = StateClient(endpoint)
        self.h = {}
        self.status = server.gui.add_text('Policy', initial_value='waiting for runner', disabled=True)
        self.show = server.gui.add_checkbox('Show policy overlay', initial_value=True)
        self.show_robot = server.gui.add_checkbox('Show policy robot (ghost)', initial_value=True)
        self.robot = None
        try:   # ghost robot posed at the runner's q (virtual rollout in dry-run, measured q live)
            import yourdfpy
            from viser.extras import ViserUrdf
            self.robot_root = server.scene.add_frame('/policy/robot', show_axes=False)
            self.robot = ViserUrdf(server, yourdfpy.URDF.load(urdf, load_meshes=True, build_scene_graph=True,
                                                               load_collision_meshes=False), root_node_name='/policy/robot',
                                   mesh_color_override=(0.3, 0.6, 1.0))
            self.robot_joints = list(self.robot.get_actuated_joint_names())
            self.robot.update_cfg(np.zeros(len(self.robot_joints)))   # ViserUrdf leaves joint frames unset until the
            self.robot_root.visible = False          # first update_cfg (links pile up at the base); hidden until data

            @self.show_robot.on_update
            def _(_):
                self.robot_root.visible = self.show_robot.value
        except Exception as exc:
            print(f'[robustdex] overlay: no ghost robot ({exc!r})', flush=True)

    def update(self):
        st = self.client.poll()
        if st is None or not self.show.value:
            return
        sc = self.server.scene
        if self.robot is not None and st.get('q') is not None:
            self.robot.update_cfg(np.asarray(self._order_q(st['q']), float))
            self.robot_root.visible = self.show_robot.value
        frames = np.asarray(st['frames'], np.float32)
        con = st.get('contact')
        colors = np.full((len(frames), 3), 230, np.uint8)
        if con is not None:
            hit = np.array([con[s] > 0.5 for s in FRAME_SLOT])
            colors[hit] = (255, 40, 40)
        self._pc('/policy/frames', frames, colors, 0.012)
        tf = st.get('target_frames')
        if tf is not None:
            self._pc('/policy/target_frames', np.asarray(tf, np.float32), np.array([[60, 120, 255]] * len(tf), np.uint8), 0.006)
        near = st.get('nearest')
        if near is not None:
            seg = np.stack([frames, np.asarray(near, np.float32)], 1)
            if '/policy/af_vec' in self.h:
                self.h['/policy/af_vec'].points = seg
            else:
                self.h['/policy/af_vec'] = sc.add_line_segments('/policy/af_vec', points=seg, colors=(255, 220, 60), line_width=2)
        pg = st.get('pregrasp') or {}
        if pg.get('palm_pos') is not None:
            self._pc('/policy/pregrasp_palm', np.asarray([pg['palm_pos']], np.float32), np.array([[40, 220, 80]], np.uint8), 0.02)
        md = st.get('min_dist')
        self.status.value = '%s%s step %s | contact %s | min frame-object dist %.3f m | |a| %.2f' % (
            'VIRTUAL(dry-run) ' if st.get('virtual') else '', st.get('phase'), st.get('step'), ''.join(str(int(c)) for c in con) if con is not None else '-',
            float(np.min(md)) if md is not None else float('nan'), st.get('action_norm') or 0.0)

    POLICY_JOINTS = ['joint%d' % i for i in range(1, 7)] + ['joint_%d_0' % i for i in range(16)]

    def _order_q(self, q):
        """Policy order (arm 1-6, joint_0_0..15_0) -> the URDF's actuated-joint order."""
        idx = {n: i for i, n in enumerate(self.POLICY_JOINTS)}
        return [q[idx[n]] if n in idx else 0.0 for n in self.robot_joints]

    def _pc(self, name, pts, colors, size):
        if name in self.h:
            self.h[name].points = pts
            self.h[name].colors = colors
        else:
            self.h[name] = self.server.scene.add_point_cloud(name, points=pts, colors=colors, point_size=size,
                                                              point_shape='circle', precision='float32')

    def close(self):
        for h in self.h.values():
            try:
                h.remove()
            except Exception:
                pass
