"""Object point cloud transport: stereo_track.py publishes the tracked object model (robot base frame) over ZMQ PUB,
the policy runner subscribes. Also the farthest-point subsampling to the policy's 200 points."""
import json
import time

import numpy as np

TOPIC = b'cloud'


def farthest_point_sample(pts, n, seed=0, max_candidates=1024):
    """FPS to n points; large clouds are first randomly thinned to max_candidates (FPS cost is n * candidates)."""
    pts = np.asarray(pts, float)
    if len(pts) <= n:
        return pts
    rng = np.random.default_rng(seed)
    if len(pts) > max_candidates:
        pts = pts[rng.choice(len(pts), max_candidates, replace=False)]
    idx = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(n - 1):
        i = int(d.argmax())
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return pts[idx]


class CloudPublisher:
    """Duck-types OneShotGraspRetrieval.observe(xyz, rgb, record) so stereo_track can call it per frame."""

    def __init__(self, endpoint='tcp://127.0.0.1:5571', max_points=4096, seed=0):
        import zmq
        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.bind(endpoint)
        self.max_points, self.rng, self.n = max_points, np.random.default_rng(seed), 0

    def observe(self, xyz, rgb=None, record=None):
        xyz = np.asarray(xyz, np.float32)
        if len(xyz) > self.max_points:
            xyz = xyz[self.rng.choice(len(xyz), self.max_points, replace=False)]
        rec = record or {}
        tf = rec.get('transform')
        header = dict(time=time.time(), mono=time.monotonic(), n=int(len(xyz)), frame=rec.get('frame'),
                      accepted=bool(rec.get('accepted', True)), status=rec.get('status'),
                      transform=None if tf is None else np.asarray(tf, float).reshape(16).tolist())
        self.sock.send_multipart([TOPIC, json.dumps(header).encode(), xyz.tobytes()])
        self.n += 1

    def close(self):
        self.sock.close(0)


class CloudClient:
    def __init__(self, endpoint='tcp://127.0.0.1:5571'):
        import zmq
        self.zmq = zmq
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self.sock.connect(endpoint)
        self.latest = None   # (header, xyz, recv_monotonic)

    def poll(self):
        while True:
            try:
                parts = self.sock.recv_multipart(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
            header = json.loads(parts[1].decode())
            xyz = np.frombuffer(parts[2], np.float32).reshape(-1, 3).astype(float)
            self.latest = (header, xyz, time.monotonic())
        return self.latest

    def wait(self, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.poll() is not None:
                return self.latest
            time.sleep(0.02)
        raise TimeoutError('no object cloud received from stereo_track')

    def age(self):
        return None if self.latest is None else time.monotonic() - self.latest[2]


STATE_TOPIC = b'policy'


class StatePublisher:
    """Runner -> viewer: per-step policy state (frames, nearest points, contact flags, phase) for the Viser overlay."""

    def __init__(self, endpoint='tcp://127.0.0.1:5572'):
        import zmq
        self.sock = zmq.Context.instance().socket(zmq.PUB)
        self.sock.bind(endpoint)

    def send(self, **state):
        self.sock.send_multipart([STATE_TOPIC, json.dumps(dict(state, time=time.time()),
                                                           default=lambda v: np.asarray(v).round(5).tolist()).encode()])

    def close(self):
        self.sock.close(0)


class StateClient:
    def __init__(self, endpoint='tcp://127.0.0.1:5572'):
        import zmq
        self.zmq = zmq
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, STATE_TOPIC)
        self.sock.connect(endpoint)
        self.latest = None

    def poll(self):
        while True:
            try:
                parts = self.sock.recv_multipart(flags=self.zmq.NOBLOCK)
            except self.zmq.Again:
                break
            self.latest = json.loads(parts[1].decode())
        return self.latest
