import http.client
import json
import re
import threading
import time
from types import SimpleNamespace

import pytest

from paradex.io.camera_system.timestamp_preview import (
    AravisPreviewCamera, PreviewController, make_server,
)


class FakeCamera:
    def __init__(self, fail=False, frames=True):
        self.fail, self.frames = fail, frames
        self.opened = self.closed = False
        self.frame = 0

    def open(self):
        self.opened = True
        if self.fail:
            raise RuntimeError("camera busy")

    def read(self):
        time.sleep(0.005)
        if not self.frames:
            return None
        self.frame += 1
        return b"\xff\xd8fake-jpeg\xff\xd9", {
            "frame_id": self.frame, "received_at": time.time(),
            "camera_timestamp_ns": "123456789", "width": 960, "height": 720,
        }

    def close(self):
        self.closed = True


def wait_state(controller, state):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if controller.status()["state"] == state:
            return controller.status()
        time.sleep(0.005)
    raise AssertionError(controller.status())


def test_camera_is_lazy_and_stop_releases_it_and_clears_jpeg():
    camera = FakeCamera()
    c = PreviewController(camera_factory=lambda: camera)
    assert not camera.opened
    try:
        c.start()
        wait_state(c, "running")
        c.start()  # Idempotent; must not open a second camera.
        assert c.stop()["state"] == "stopped"
        assert camera.closed and c._jpeg is None
    finally:
        c.stop()


def test_partial_open_failure_is_cleaned_up_and_restart_works():
    cameras = [FakeCamera(fail=True), FakeCamera()]
    c = PreviewController(camera_factory=lambda: cameras.pop(0))
    failed = cameras[0]
    try:
        c.start()
        assert wait_state(c, "error")["error"] == "camera busy"
        assert failed.closed
        c.start()
        wait_state(c, "running")
    finally:
        c.stop()


def test_lost_browser_heartbeat_releases_camera():
    camera = FakeCamera()
    c = PreviewController(camera_factory=lambda: camera, idle_timeout=0.12)
    c.start()
    wait_state(c, "running")
    time.sleep(0.07)
    c.heartbeat()
    time.sleep(0.07)
    assert c.status()["state"] == "running"
    status = wait_state(c, "stopped")
    assert "disconnected" in status["stop_reason"] and camera.closed


def test_missing_frames_reports_error_and_releases_camera():
    camera = FakeCamera(frames=False)
    c = PreviewController(camera_factory=lambda: camera, frame_timeout=0.05)
    c.start()
    assert "No camera frames" in wait_state(c, "error")["error"]
    assert camera.closed


def test_original_features_restored_even_after_partial_configuration_failure():
    class Device:
        values = {"TriggerMode": "On", "PixelFormat": "Mono8"}

        def get_string_feature_value(self, name):
            return self.values[name]

        def set_string_feature_value(self, name, value):
            self.values[name] = value
            if value == "BayerRG8":
                raise RuntimeError("configuration failure")

    camera = AravisPreviewCamera("22684253")
    device = Device()
    camera.device = device
    camera._set("string", "TriggerMode", "Off")
    with pytest.raises(RuntimeError):
        camera._set("string", "PixelFormat", "BayerRG8")
    camera.close()
    assert device.values == {"TriggerMode": "On", "PixelFormat": "Mono8"}
    assert camera.camera is None and camera.device is None


def test_live_bayer_conversion_works_on_deployed_opencv_and_returns_buffer():
    import cv2
    import numpy as np

    raw = np.zeros((32, 32), dtype=np.uint8)
    raw[::2, ::2] = 255  # RGGB sensor with a pure red scene.
    buffer = SimpleNamespace(
        get_status=lambda: 0, get_image_width=lambda: 32, get_image_height=lambda: 32,
        get_data=lambda: raw.tobytes(), get_frame_id=lambda: 42, get_timestamp=lambda: 123,
    )
    returned = []
    camera = AravisPreviewCamera("22684253")
    camera.cv2, camera.np = cv2, np
    camera.aravis = SimpleNamespace(BufferStatus=SimpleNamespace(SUCCESS=0))
    camera.stream = SimpleNamespace(timeout_pop_buffer=lambda _: buffer, push_buffer=returned.append)
    jpeg, metadata = camera.read()
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image[:, :, 2].mean() > 240 and image[:, :, 0].mean() < 10
    assert metadata["frame_id"] == 42 and returned == [buffer]


def test_http_controls_authorization_jpeg_stream_and_stop():
    camera = FakeCamera()
    c = PreviewController(camera_factory=lambda: camera)
    server = make_server(c, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port

    def request(method, path, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        status, body = response.status, response.read()
        connection.close()
        return status, body

    try:
        status, html = request("GET", "/")
        assert status == 200 and not camera.opened
        token = re.search(rb"const token='([^']+)'", html).group(1).decode()
        headers = {"X-Preview-Token": token}
        assert request("GET", "/frame.jpg")[0] == 409
        assert request("POST", "/api/start")[0] == 403
        assert request("GET", "/", {"Host": "evil.example"})[0] == 403
        assert request("POST", "/api/start", dict(headers, Origin="http://evil.example"))[0] == 403
        assert request("POST", "/api/start", dict(headers, **{"Sec-Fetch-Site": "cross-site"}))[0] == 403
        assert request("POST", "/api/start", headers)[0] == 200
        wait_state(c, "running")
        assert request("GET", "/frame.jpg") == (200, b"\xff\xd8fake-jpeg\xff\xd9")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("GET", "/stream.mjpg")
        response = connection.getresponse()
        assert response.status == 200
        assert "multipart/x-mixed-replace" in response.getheader("Content-Type")
        assert response.readline() == b"--frame\r\n"
        assert response.readline() == b"Content-Type: image/jpeg\r\n"
        length = int(response.readline().split(b":")[1])
        assert response.readline() == b"\r\n"
        assert response.read(length) == b"\xff\xd8fake-jpeg\xff\xd9"
        connection.close()
        assert request("POST", "/api/heartbeat", headers)[0] == 200
        assert json.loads(request("POST", "/api/stop", headers)[1])["state"] == "stopped"
        assert camera.closed
        assert request("GET", "/stream.mjpg")[0] == 409
    finally:
        c.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
