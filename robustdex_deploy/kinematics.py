"""Kinematics-only MuJoCo model of xArm6 + Allegro v5 (exported by export_deploy.py): FK of the 17 hand probe
frames / 6 arm links / wrist, and damped-least-squares IK on the 6 arm joints (same solver as training)."""
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R


def raisim_euler(mat):
    """raisim::RotmatToEuler == scipy intrinsic XYZ (the convention of the policy's wrist observations)."""
    return R.from_matrix(mat).as_euler('XYZ')


class FKModel:
    def __init__(self, xml_path, meta):
        self.m = mj.MjModel.from_xml_path(xml_path)
        self.d = mj.MjData(self.m)
        self.joints = list(meta['joints'])
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in self.joints])
        self.arm_dofs = np.array([self.m.jnt_dofadr[self.m.joint(j).id] for j in self.joints[:6]])
        bid = lambda n: mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, n)
        self.frame_ids = np.array([bid(n) for n in meta['frame_bodies']])
        self.arm_frame_ids = np.array([bid(n) for n in meta['arm_frame_bodies']])
        self.wrist = bid(meta['wrist_body'])
        self.lo, self.hi = np.array(meta['joint_lo']), np.array(meta['joint_hi'])
        self.hand_center = np.array(meta['hand_center_in_wrist'])
        assert self.m.nq == 22, self.m.nq

    def fk(self, q):
        """q (22,) -> dict(frames (17,3), arm (6,3), Rw (3,3), wrist (3,), hand_center (3,))"""
        self.d.qpos[self.qadr] = q
        mj.mj_kinematics(self.m, self.d)
        Rw = self.d.xmat[self.wrist].reshape(3, 3).copy()
        frames = self.d.xpos[self.frame_ids].copy()
        return dict(frames=frames, arm=self.d.xpos[self.arm_frame_ids].copy(), Rw=Rw, wrist=self.d.xpos[self.wrist].copy(),
                    hand_center=frames[0] + Rw @ self.hand_center)

    def solve_ik(self, target_pos, target_R, q_arm_init, q_hand, iters=150, tol_p=0.005, tol_r=0.05, max_dq=0.2):
        """Arm joints placing the wrist body at (target_pos, target_R); the hand joints only affect nothing here but are
        set for completeness. Returns (q_arm, converged)."""
        q = np.asarray(q_arm_init, float).copy()
        lo, hi = self.lo[:6], self.hi[:6]
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        self.d.qpos[self.qadr[6:]] = q_hand
        quat, er = np.zeros(4), np.zeros(3)
        for _ in range(iters):
            self.d.qpos[self.qadr[:6]] = q
            mj.mj_kinematics(self.m, self.d)
            mj.mj_comPos(self.m, self.d)
            Rw = self.d.xmat[self.wrist].reshape(3, 3)
            ep = target_pos - self.d.xpos[self.wrist]
            mj.mju_mat2Quat(quat, (target_R @ Rw.T).ravel())
            mj.mju_quat2Vel(er, quat, 1.0)
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                return q, True
            mj.mj_jacBody(self.m, self.d, jp, jr, self.wrist)
            J = np.vstack([jp[:, self.arm_dofs], jr[:, self.arm_dofs]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[ep, er])
            q = np.clip(q + np.clip(dq, -max_dq, max_dq), lo, hi)
        return q, False
