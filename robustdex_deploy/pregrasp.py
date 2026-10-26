"""IK pre-grasp start pose (RobustDexGrasp App. B.1 / upstream initial_pose_final.py), as used to start every training
episode: object centre c (from the visible points), approach direction dir = (CAMERA - c)/|.|, palm centre at
c + 0.25 dir, 10 wrist rotations about dir, IK from theta0, pick the candidate with the smallest object width along the
finger axis (< 0.18 m preferred), slight preference for staying near theta0. The training-time collision test against
table/object is replaced by a height test of all hand frames (no collision model on the robot side)."""
import numpy as np


def sample_rot_mats(hdir, n, points):
    ref = hdir.reshape(3)
    temp = np.array([1, 0, 0]) if abs(ref[0]) < 0.9 else np.array([0, 1, 0])
    p1 = np.cross(ref, temp); p1 /= np.linalg.norm(p1)
    p2 = np.cross(ref, p1); p2 /= np.linalg.norm(p2)
    perps = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        v = p1 * np.cos(th) + p2 * np.sin(th)
        v /= np.linalg.norm(v)
        if v[0] > 0:          # fingers (= -perp) point away from the robot (+x)
            v = -v
        perps.append(v)
    perps = np.array(perps)
    centred = points - points.mean(0)
    proj = centred @ perps.T
    width = proj.max(0) - proj.min(0)
    ys = np.cross(ref[None], perps)
    ys /= np.linalg.norm(ys, axis=1, keepdims=True)
    return perps, ys, width


def visible_points(points, viewpoint, radius_factor=100.0):
    """Points visible from `viewpoint` (Katz et al. hidden point removal by spherical flipping + convex hull). Stands in
    for training's ray cast of the object mesh from the camera; works on any cloud (sim points or the tracked model)."""
    from scipy.spatial import ConvexHull
    p = np.asarray(points, float) - viewpoint
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    R = norm.max() * radius_factor
    flipped = p + 2 * (R - norm) * p / norm
    hull = ConvexHull(np.vstack([flipped, np.zeros(3)]))
    idx = np.unique(hull.vertices)
    idx = idx[idx < len(p)]
    return np.asarray(points, float)[idx] if len(idx) >= 4 else np.asarray(points, float)


def pregrasp_arm(fk, points, q_hand, meta, min_frame_z=None, return_all=False):
    """Returns (q_arm or None, info). `points`: object points in the robot frame (the policy's 200-point set is fine)."""
    P = meta['pregrasp']
    cam, palm, theta0 = np.array(P['camera']), np.array(P['palm_in_wrist']), np.array(meta['theta0'])
    pts = visible_points(points, cam)       # training: ray-cast visible points from CAMERA
    c = pts.mean(0)
    hdir = cam - c
    hdir /= np.linalg.norm(hdir)
    pos = c + P['standoff'] * hdir
    perps, ys, width = sample_rot_mats(hdir, P['n_rot'], pts)
    zmin = meta['table_height'] + 0.01 if min_frame_z is None else min_frame_z
    cands = []
    for j in range(P['n_rot']):
        Rt = -np.stack([hdir, ys[j], perps[j]], axis=1)   # Allegro wrist: palm normal +x, fingers +z
        q, ok = fk.solve_ik(pos - Rt @ palm, Rt, theta0, q_hand)
        if not ok:
            continue
        k = fk.fk(np.r_[q, q_hand])
        if k['frames'][:, 2].min() < zmin or k['arm'][:, 2].min() < zmin:
            continue
        cands.append((j, q))
    if not cands:
        return None, dict(feasible=0, centre=c.tolist())
    w = np.array([width[j] for j, _ in cands])
    score = np.where(w < 0.18, w * 5.0, 1e4 + w) + np.array([0.1 * np.linalg.norm(q - theta0) for _, q in cands])
    best = int(np.argmin(score))
    info = dict(feasible=len(cands), width=float(w[best]), centre=c.tolist(), palm_pos=pos.tolist(), rot_index=cands[best][0])
    if return_all:
        info['candidates'] = [q for _, q in cands]
    return cands[best][1], info
