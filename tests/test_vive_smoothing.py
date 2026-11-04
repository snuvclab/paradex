from pathlib import Path
import importlib.util
import numpy as np
from scipy.spatial.transform import Rotation as R
p=Path(__file__).resolve().parents[1]/'paradex/dataset_acqusition/pose_smoothing.py'
if not p.exists():p=Path(__file__).with_name('pose_smoothing.py')
spec=importlib.util.spec_from_file_location('pose',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_low_pass_reduces_jitter_and_preserves_rigid_rotation():
    s=m.PoseSmoother(np.eye(4),0,.08);out=[]
    for i in range(1,101):
        p=np.eye(4);p[0,3]=.002*(-1)**i;p[:3,:3]=R.from_euler('z',(-1)**i,degrees=True).as_matrix()
        v=s.filter(p,i*.01);out.append(v[0,3]);assert np.allclose(v[:3,:3].T@v[:3,:3],np.eye(3));assert np.isclose(np.linalg.det(v[:3,:3]),1)
    assert np.std(out[30:])<.0002

def test_convergence_frame_rate_invariance_and_zero_opt_out():
    p=np.eye(4);p[0,3]=.1;p[:3,:3]=R.from_euler('z',30,degrees=True).as_matrix();results=[]
    for dt in (.005,.01,.02):
        s=m.PoseSmoother(np.eye(4),0,.08)
        for t in np.arange(dt,.4+dt/2,dt):v=s.filter(p,t)
        results.append(v)
    for v in results:assert np.allclose(v,results[0],atol=1e-10)
    assert np.linalg.norm(v[:3,3]-p[:3,3])<.001
    s=m.PoseSmoother(np.eye(4),0,0);assert np.allclose(s.filter(p,.01),p)

def test_shortest_rotation_and_fresh_reset():
    a=np.eye(4);a[:3,:3]=R.from_euler('z',179,degrees=True).as_matrix();b=a.copy();b[:3,:3]=R.from_euler('z',-179,degrees=True).as_matrix()
    s=m.PoseSmoother(a,0,.08);v=s.filter(b,.01);assert np.degrees((R.from_matrix(a[:3,:3]).inv()*R.from_matrix(v[:3,:3])).magnitude())<1
    new=m.PoseSmoother(np.eye(4),10,.08);assert np.allclose(new.filter(np.eye(4),10.01),np.eye(4))

def test_repeated_cached_frames_do_not_shrink_guard_interval():
    from paradex.dataset_acqusition.capture import _PoseCommandLimiter
    limiter=_PoseCommandLimiter(np.eye(4),0,smoothing_tau_s=.08)
    for t in np.arange(.001,.01,.001):limiter.filter(np.eye(4),t)
    target=np.eye(4);target[0,3]=.008
    filtered,_,_=limiter.filter(target,.01)
    assert filtered is not None and 0<filtered[0,3]<.008


def test_smoothing_never_admits_a_raw_tracking_jump():
    from paradex.dataset_acqusition.capture import _PoseCommandLimiter
    limiter=_PoseCommandLimiter(np.eye(4),0,smoothing_tau_s=.08)
    target=np.eye(4);target[0,3]=.1
    filtered,_,_=limiter.filter(target,.01)
    assert filtered is None and np.allclose(limiter.smoother.pose,np.eye(4))
