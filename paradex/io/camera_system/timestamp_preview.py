"""On-demand, non-recording timestamp-camera preview over a loopback HTTP API.

Run with the system Python that provides Aravis GI, NumPy and OpenCV. The HTTP
server itself has no third-party dependencies and never opens a camera on boot.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import secrets
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


log = logging.getLogger(__name__)


class AravisPreviewCamera:
    """Own one camera while previewing and restore its volatile feature values."""

    def __init__(self, serial, fps=10, width=960, quality=80):
        self.serial, self.fps, self.width, self.quality = serial, fps, width, quality
        self.camera = self.device = self.stream = None
        self._saved = []
        self._acquiring = False

    def _set(self, kind, name, value):
        getter = getattr(self.device, "get_{}_feature_value".format(kind))
        setter = getattr(self.device, "set_{}_feature_value".format(kind))
        previous = getter(name)
        self._saved.append((kind, name, previous))
        setter(name, value)
        actual = getter(name)
        equal = abs(actual - value) < 0.1 if kind == "float" else actual == value
        if not equal:
            raise RuntimeError("{} readback mismatch: {} != {}".format(name, actual, value))

    def open(self):
        import gi
        import cv2
        import numpy as np

        gi.require_version("Aravis", "0.8")
        from gi.repository import Aravis

        self.aravis, self.cv2, self.np = Aravis, cv2, np
        Aravis.update_device_list()
        device_id = next(
            (Aravis.get_device_id(i) for i in range(Aravis.get_n_devices())
             if str(Aravis.get_device_serial_nbr(i)) == self.serial), None
        )
        if device_id is None:
            raise RuntimeError("Camera {} was not discovered".format(self.serial))
        self.camera = Aravis.Camera.new(device_id)
        if self.camera is None:
            raise RuntimeError("Could not open camera {}; it may be busy".format(self.serial))
        self.device = self.camera.get_device()
        self._set("string", "TriggerSelector", "FrameStart")
        self._set("string", "TriggerMode", "Off")
        self._set("string", "AcquisitionMode", "Continuous")
        self._set("string", "PixelFormat", "BayerRG8")
        # Blackfly BFLY uses Enabled; newer models use Enable.
        rate_enable = next(
            (name for name in ("AcquisitionFrameRateEnabled", "AcquisitionFrameRateEnable")
             if self.camera.is_feature_available(name)), None
        )
        if rate_enable is None:
            raise RuntimeError("Camera has no acquisition frame-rate enable feature")
        if self.camera.is_feature_available("AcquisitionFrameRateAuto"):
            self._set("string", "AcquisitionFrameRateAuto", "Off")
        self._set("boolean", rate_enable, True)
        self._set("float", "AcquisitionFrameRate", float(self.fps))
        self.stream = self.camera.create_stream(None, None)
        if self.stream is None:
            raise RuntimeError("Could not create camera stream")
        payload = self.camera.get_payload()
        for _ in range(12):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        self.camera.start_acquisition()
        self._acquiring = True

    def read(self):
        buffer = self.stream.timeout_pop_buffer(250000)
        if buffer is None:
            return None
        try:
            if buffer.get_status() != self.aravis.BufferStatus.SUCCESS:
                return None
            width, height = buffer.get_image_width(), buffer.get_image_height()
            data = buffer.get_data()
            # Older GI versions expose a sequence; newer versions expose bytes.
            raw = self.np.asarray(data, dtype=self.np.uint8) if not isinstance(data, bytes) else self.np.frombuffer(data, dtype=self.np.uint8)
            image = raw[:width * height].reshape(height, width)
            # OpenCV BG means sensor RGGB. The long RGGB alias is missing in
            # mainpc's OpenCV 4.2, so use its equivalent legacy constant.
            image = self.cv2.cvtColor(image, self.cv2.COLOR_BayerBG2BGR)
            if width > self.width:
                image = self.cv2.resize(image, (self.width, round(height * self.width / width)), interpolation=self.cv2.INTER_AREA)
            ok, jpeg = self.cv2.imencode(".jpg", image, [self.cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            return jpeg.tobytes(), {
                "frame_id": int(buffer.get_frame_id()),
                "camera_timestamp_ns": str(buffer.get_timestamp()),
                "received_at": time.time(),
                "width": image.shape[1], "height": image.shape[0],
            }
        finally:
            self.stream.push_buffer(buffer)

    def close(self):
        errors = []
        if self._acquiring:
            try:
                self.camera.stop_acquisition()
            except Exception as exc:
                errors.append("stop acquisition: {}".format(exc))
        self._acquiring = False
        self.stream = None
        for kind, name, value in reversed(self._saved):
            try:
                getattr(self.device, "set_{}_feature_value".format(kind))(name, value)
            except Exception as exc:
                errors.append("restore {}: {}".format(name, exc))
        self._saved.clear()
        self.device = self.camera = None
        if errors:
            raise RuntimeError("; ".join(errors))


class PreviewController:
    def __init__(self, serial="22684253", camera_factory=None, idle_timeout=15, frame_timeout=5):
        self.serial = serial
        self.camera_factory = camera_factory or (lambda: AravisPreviewCamera(serial))
        self.idle_timeout, self.frame_timeout = idle_timeout, frame_timeout
        self.condition = threading.Condition()
        self._operations = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._state = "stopped"
        self._error = None
        self._reason = None
        self._jpeg = None
        self._sequence = 0
        self._metadata = {}
        self._heartbeat = 0

    def status(self):
        with self.condition:
            age = time.time() - self._metadata["received_at"] if self._metadata else None
            return dict(self._metadata, serial=self.serial, state=self._state,
                        error=self._error, stop_reason=self._reason, frames=self._sequence,
                        frame_age_seconds=age)

    def heartbeat(self):
        with self.condition:
            self._heartbeat = time.monotonic()

    def start(self):
        with self._operations, self.condition:
            if self._thread and self._thread.is_alive():
                return self.status()
            self._stop.clear()
            self._state, self._error, self._reason = "starting", None, None
            self._jpeg, self._sequence, self._metadata = None, 0, {}
            self._heartbeat = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="timestamp-preview", daemon=True)
            self._thread.start()
            return self.status()

    def stop(self):
        with self._operations:
            with self.condition:
                self._stop.set()
                if self._thread and self._thread.is_alive():
                    self._state = "stopping"
                self.condition.notify_all()
            if self._thread:
                self._thread.join(timeout=10)
                if self._thread.is_alive():
                    raise RuntimeError("Camera is still stopping; control has not been released")
            return self.status()

    def _run(self):
        camera = None
        failure = None
        try:
            camera = self.camera_factory()
            camera.open()
            last_frame = time.monotonic()
            while not self._stop.is_set():
                now = time.monotonic()
                with self.condition:
                    if now - self._heartbeat > self.idle_timeout:
                        self._reason = "Browser disconnected; camera released"
                        break
                frame = camera.read()
                if frame is None:
                    if time.monotonic() - last_frame > self.frame_timeout:
                        raise RuntimeError("No camera frames received for {} seconds".format(self.frame_timeout))
                    continue
                last_frame = time.monotonic()
                with self.condition:
                    if self._stop.is_set():
                        break
                    self._jpeg, self._metadata = frame
                    self._sequence += 1
                    self._state = "running"
                    self.condition.notify_all()
        except Exception as exc:
            log.exception("Timestamp camera preview failed")
            failure = str(exc)
        finally:
            if camera:
                try:
                    camera.close()
                except Exception as exc:
                    log.exception("Camera cleanup failed")
                    failure = "{}; {}".format(failure, exc) if failure else str(exc)
            with self.condition:
                self._state = "error" if failure else "stopped"
                self._error = failure
                self._jpeg = None
                self.condition.notify_all()


def make_server(controller, host="127.0.0.1", port=8765):
    token = secrets.token_urlsafe(32)
    html = Path(__file__).with_name("templates").joinpath("timestamp_preview.html").read_text(encoding="utf-8")
    html = html.replace("__CONTROL_TOKEN__", token).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def send_body(self, code, content_type, body):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def json(self, code, payload):
            self.send_body(code, "application/json", json.dumps(payload).encode("utf-8"))

        def authorized(self):
            # A page on another origin cannot operate the local camera service.
            origin = self.headers.get("Origin")
            return (self.headers.get("X-Preview-Token") == token
                    and self.headers.get("Sec-Fetch-Site", "same-origin") in ("same-origin", "none")
                    and (origin is None or urlsplit(origin).netloc == self.headers.get("Host")))

        def do_POST(self):
            if not self.local_host() or not self.authorized():
                self.json(403, {"error": "Invalid preview control token or origin"})
                return
            try:
                path = urlsplit(self.path).path
                if path == "/api/start":
                    self.json(200, controller.start())
                elif path == "/api/stop":
                    self.json(200, controller.stop())
                elif path == "/api/heartbeat":
                    controller.heartbeat()
                    self.json(200, controller.status())
                else:
                    self.json(404, {"error": "Not found"})
            except Exception as exc:
                self.json(500, {"error": str(exc)})

        def do_GET(self):
            if not self.local_host():
                self.json(403, {"error": "Use localhost or 127.0.0.1 through an SSH tunnel"})
                return
            path = urlsplit(self.path).path
            if path == "/":
                self.send_body(200, "text/html; charset=utf-8", html)
            elif path == "/api/status":
                self.json(200, controller.status())
            elif path == "/frame.jpg":
                with controller.condition:
                    jpeg = controller._jpeg
                if jpeg:
                    self.send_body(200, "image/jpeg", jpeg)
                else:
                    self.json(409, {"error": "No live frame"})
            elif path == "/stream.mjpg":
                self.stream()
            else:
                self.json(404, {"error": "Not found"})

        def local_host(self):
            try:
                return urlsplit("//" + self.headers.get("Host", "")).hostname in ("127.0.0.1", "localhost")
            except ValueError:
                return False

        def stream(self):
            with controller.condition:
                if controller._state not in ("starting", "running"):
                    self.json(409, {"error": "Preview is stopped"})
                    return
            self.connection.settimeout(3)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            previous = 0
            try:
                while True:
                    with controller.condition:
                        controller.condition.wait_for(
                            lambda: controller._sequence != previous or controller._state not in ("starting", "running"), timeout=1)
                        if controller._state not in ("starting", "running"):
                            return
                        jpeg, sequence = controller._jpeg, controller._sequence
                    if jpeg and sequence != previous:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        self.wfile.flush()
                        previous = sequence
            except (OSError, TimeoutError):
                pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="22684253")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=int, choices=range(1, 31), default=10)
    parser.add_argument("--width", type=int, choices=range(160, 2049), default=960)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    controller = PreviewController(args.serial, camera_factory=lambda: AravisPreviewCamera(args.serial, args.fps, args.width))
    server = make_server(controller, port=args.port)

    def shutdown(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log.info("Timestamp camera %s Web UI: http://127.0.0.1:%s (camera stopped)", args.serial, args.port)
    try:
        server.serve_forever()
    finally:
        controller.stop()
        server.server_close()


if __name__ == "__main__":
    main()
