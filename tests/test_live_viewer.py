"""Exercise real HTTP gateways with an instrumented capture daemon substitute."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from paradex.io.camera_system.live_viewer import ViewerGateway, make_server


@pytest.fixture
def rig():
    state = {'phase': 'idle', 'count': 1, 'requests': [], 'available': True}

    class Daemon(BaseHTTPRequestHandler):
        def do_GET(self):
            state['requests'].append(self.path)
            if self.path == '/cameras':
                body = json.dumps({'backend': 'aravis-gstreamer', 'state': state['phase'],
                                   'cameras': [{'name': '123', 'state': 'CAPTURING',
                                                'frame_count': state['count']}]}).encode()
            elif self.path in ('/preview/123', '/frame/123'):
                if not state['available']:
                    self.send_error(404)
                    return
                body = b'\xff\xd8' + str(state['count']).encode() + b'\xff\xd9'
            else:
                pytest.fail('Unexpected daemon request: ' + self.path)
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Frame-Id', str(state['count']))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Daemon)
    gateway = ViewerGateway({'capture12': {'ip': '127.0.0.1', 'cam_list': ['123', 'missing']}},
                            daemon_port=upstream.server_port, timeout=0.2, fps=10)
    server = make_server(gateway, port=0)
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, server)]
    for thread in threads:
        thread.start()
    yield state, gateway, 'http://127.0.0.1:' + str(server.server_port)
    for s in (server, upstream):
        s.shutdown()
        s.server_close()


def test_viewing_idle_capture_and_transition_never_sends_control(rig):
    state, gateway, base = rig
    for phase in ('idle', 'preparing', 'capturing', 'stopping', 'idle'):
        state['phase'] = phase
        state['count'] += 1
        gateway._status_cache.until = 0
        with urlopen(base + '/api/cameras') as r:
            data = json.load(r)
        assert data['pcs'][0]['state'] == phase
        assert data['pcs'][0]['cameras'][1]['state'] == 'MISSING'
        for quality in ('preview', 'full'):
            gateway._frames[('capture12', '123', quality)].until = 0
            with urlopen(base + '/api/frame/capture12/123?quality=' + quality) as r:
                assert r.read().startswith(b'\xff\xd8')
                assert int(r.headers['X-Frame-Id']) == state['count']
        assert state['phase'] == phase
    assert set(state['requests']) == {'/cameras', '/preview/123', '/frame/123'}


def test_multiple_viewers_share_fetch_and_failed_fetch_recovers(rig):
    state, gateway, _ = rig
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: gateway.frame('capture12', '123', 'preview'), range(8)))
    assert len(set(results)) == 1
    assert state['requests'].count('/preview/123') == 1
    time.sleep(0.11)
    state['available'] = False
    from paradex.io.camera_system.live_viewer import ReadFailure
    for _ in range(3):
        with pytest.raises(ReadFailure) as exc:
            gateway.frame('capture12', '123', 'preview')
        assert exc.value.status == 503
    assert state['requests'].count('/preview/123') == 2
    time.sleep(0.11)
    state['available'] = True
    state['count'] = 99
    assert b'99' in gateway.frame('capture12', '123', 'preview')[0]


def test_gateway_rejects_unknown_sources_and_write_requests(rig):
    state, _, base = rig
    for path in ('/api/frame/attacker/123', '/api/frame/capture12/999',
                 '/api/frame/capture12/123?quality=anything', '/api/frame/capture12/123/extra'):
        with pytest.raises(HTTPError) as exc:
            urlopen(base + path)
        assert exc.value.code == 404
    from urllib.request import Request
    with pytest.raises(HTTPError) as exc:
        urlopen(Request(base + '/api/cameras', data=b'{}'))
    assert exc.value.code == 501
    assert not state['requests']


def test_grid_page_and_module_do_not_contact_camera_daemons(rig):
    state, _, base = rig
    with urlopen(base + '/?view=grid') as response:
        page = response.read().decode()
    assert 'value="grid"' in page
    assert 'src="/viewer.mjs"' in page
    with urlopen(base + '/viewer.mjs') as response:
        assert response.headers['Content-Type'].startswith('text/javascript')
        script = response.read().decode()
    assert 'export class FrameScheduler' in script
    assert not state['requests']


def test_offline_pc_does_not_hide_other_cameras(rig):
    _, gateway, _ = rig
    gateway.pcs['offline'] = {'ip': '127.0.0.2', 'cam_list': ['456']}
    data = gateway.status()
    assert data['pcs'][0]['online']
    assert not data['pcs'][1]['online']
    assert data['pcs'][1]['cameras'][0]['state'] == 'OFFLINE'
