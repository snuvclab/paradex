"""Read-only browser gateway to already-running capture-PC camera daemons.

Only GET /cameras, /preview/<serial>, and /frame/<serial> are sent upstream.
No camera SDK, controller, trigger or recording code is imported here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, unquote, urlsplit
from urllib.request import ProxyHandler, build_opener


class ReadFailure(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class CachedRead:
    """Coalesce concurrent viewers, including failed requests, per resource."""
    def __init__(self):
        self.lock = threading.Lock()
        self.until = 0.0
        self.value = None
        self.error = None

    def get(self, fetch, ttl):
        with self.lock:
            if time.monotonic() >= self.until:
                try:
                    self.value, self.error = fetch(), None
                except Exception as exc:
                    self.value, self.error = None, exc
                self.until = time.monotonic() + ttl
            if self.error is not None:
                raise self.error
            return self.value


class ViewerGateway:
    def __init__(self, pcs, daemon_port=5484, timeout=1.5, fps=5):
        self.pcs = pcs
        self.daemon_port = int(daemon_port)
        self.timeout = timeout
        self.fps = fps
        self._status_cache = CachedRead()
        # Fixed allowlist prevents arbitrary upstream URLs and unbounded caches.
        self._frames = {
            (pc, str(serial), quality): CachedRead()
            for pc, config in pcs.items()
            for serial in config['cam_list']
            for quality in ('preview', 'full')
        }

    def _read(self, pc, path):
        url = 'http://{}:{}{}'.format(self.pcs[pc]['ip'], self.daemon_port, path)
        try:
            # Lab addresses must not go through HTTP_PROXY on the main PC.
            with build_opener(ProxyHandler({})).open(url, timeout=self.timeout) as response:
                body = response.read(16 * 1024 * 1024 + 1)
                if len(body) > 16 * 1024 * 1024:
                    raise ReadFailure(502, 'Oversized daemon response')
                return body, response.headers.get('X-Frame-Id')
        except HTTPError as exc:
            status = 503 if exc.code == 404 else 502
            exc.close()
            raise ReadFailure(status, 'Frame unavailable; waiting for camera') from None
        except (OSError, ValueError) as exc:
            raise ReadFailure(502, 'Capture PC unavailable: {}'.format(exc)) from None

    def _pc_status(self, pc):
        try:
            raw, _ = self._read(pc, '/cameras')
            data = json.loads(raw)
            cameras = {str(c['name']): c for c in data['cameras']}
            return {
                'pc': pc, 'online': True, 'backend': data.get('backend'),
                'state': data.get('state'),
                'cameras': [dict(cameras.get(str(s), {'name': str(s), 'state': 'MISSING'}))
                            for s in self.pcs[pc]['cam_list']],
            }
        except Exception as exc:
            return {'pc': pc, 'online': False, 'error': str(exc), 'cameras': [
                {'name': str(s), 'state': 'OFFLINE'} for s in self.pcs[pc]['cam_list']]}

    def status(self):
        def fetch():
            with ThreadPoolExecutor(max_workers=min(6, max(1, len(self.pcs)))) as pool:
                return {'pcs': list(pool.map(self._pc_status, self.pcs)), 'max_fps': self.fps}
        return self._status_cache.get(fetch, 0.75)

    def frame(self, pc, serial, quality):
        entry = self._frames.get((pc, serial, quality))
        if entry is None:
            raise ReadFailure(404, 'Unknown PC, camera or quality')

        def fetch():
            path = '/{}/{}'.format('frame' if quality == 'full' else 'preview', quote(serial, safe=''))
            body, frame_id = self._read(pc, path)
            if not body.startswith(b'\xff\xd8'):
                raise ReadFailure(502, 'Daemon did not return a JPEG')
            return body, frame_id
        return entry.get(fetch, 1.0 / self.fps)


def make_server(gateway, host='127.0.0.1', port=8766):
    page = (Path(__file__).parent / 'templates' / 'live_viewer.html').read_bytes()
    script = (Path(__file__).parent / 'templates' / 'live_viewer.mjs').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def send_body(self, status, body, content_type, frame_id=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if frame_id is not None:
                self.send_header('X-Frame-Id', str(frame_id))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                if url.path == '/':
                    self.send_body(200, page, 'text/html; charset=utf-8')
                elif url.path == '/viewer.mjs':
                    self.send_body(200, script, 'text/javascript; charset=utf-8')
                elif url.path == '/api/cameras':
                    self.send_body(200, json.dumps(gateway.status()).encode(), 'application/json')
                elif url.path.startswith('/api/frame/'):
                    parts = url.path.split('/')
                    if len(parts) != 5:
                        raise ReadFailure(404, 'Unknown camera path')
                    pc, serial = map(unquote, parts[3:])
                    quality = parse_qs(url.query).get('quality', ['preview'])[0]
                    body, frame_id = gateway.frame(pc, serial, quality)
                    self.send_body(200, body, 'image/jpeg', frame_id)
                else:
                    raise ReadFailure(404, 'Not found')
            except ReadFailure as exc:
                self.send_body(exc.status, json.dumps({'error': str(exc)}).encode(), 'application/json')
            except (OSError, TimeoutError):
                pass

        def log_message(self, _format, *_args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[3] / 'system/current/pc.json')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--daemon-port', type=int, default=5484)
    parser.add_argument('--fps', type=float, default=5)
    parser.add_argument('--timeout', type=float, default=1.5)
    parser.add_argument('--stop-on-stdin-eof', action='store_true',
                        help='Shut down when the owning SSH connection closes')
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 0 < args.fps <= 10:
        parser.error('--fps must be in (0, 10]')
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 10:
        parser.error('--timeout must be in (0, 10]')
    gateway = ViewerGateway(json.loads(args.config.read_text()), args.daemon_port, args.timeout, args.fps)
    server = make_server(gateway, args.host, args.port)
    if args.stop_on_stdin_eof:
        def watch_stdin():
            while sys.stdin.buffer.read(1):
                pass
            server.shutdown()
        threading.Thread(target=watch_stdin, daemon=True).start()
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    print('Read-only camera viewer: http://{}:{}'.format(args.host, server.server_port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
