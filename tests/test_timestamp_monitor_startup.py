from threading import Thread

import pytest

from paradex.io.camera_system import aravis_timestamp, timestamp_monitor


@pytest.mark.parametrize("cam_type", ["aravis", "unsupported"])
def test_camera_connection_error_reaches_caller_instead_of_hanging(monkeypatch, cam_type):
    def fail(_name):
        raise OSError("camera unavailable")

    monkeypatch.setattr(aravis_timestamp, "load_timestamp_monitor", fail)
    errors = []

    def connect():
        try:
            timestamp_monitor.TimestampMonitor(cam_type=cam_type, name="test")
        except Exception as exc:
            errors.append(exc)

    # Daemon wrapper makes a regression failure bounded even if the constructor hangs.
    caller = Thread(target=connect, daemon=True)
    caller.start()
    caller.join(timeout=2.0)
    assert not caller.is_alive(), "TimestampMonitor constructor hung after connection failure"
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "test" in str(errors[0])
    assert isinstance(errors[0].__cause__, (OSError, NotImplementedError))


def test_successful_connection_still_releases_camera(monkeypatch):
    released = []

    class Camera:
        def release(self):
            released.append(True)

    monkeypatch.setattr(aravis_timestamp, "load_timestamp_monitor", lambda _name: Camera())
    monitor = timestamp_monitor.TimestampMonitor(name="test")
    monitor.end()
    assert released == [True]
    assert not monitor.capture_thread.is_alive()
