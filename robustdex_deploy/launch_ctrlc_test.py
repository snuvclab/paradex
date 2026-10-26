"""Shutdown test for robustdex_deploy/launch.py with mock tracker / runner processes (run on mainpc from ~/paradex):
  python robustdex_deploy/tests/launch_ctrlc_test.py            # all scenarios
Scenarios: single Ctrl-C; three rapid Ctrl-C; runner stuck in shutdown (ignores SIGINT/SIGTERM).
Expectation: launcher exits, no mock tracker / runner / 'sleep 600' child left behind."""
import os
import signal
import subprocess
import sys
import time

PY = sys.executable
TRACKER = '''
import subprocess, sys, time, numpy as np
sys.path.insert(0, ".")
from robustdex_deploy.cloud import CloudPublisher
child = subprocess.Popen(["sleep", "600"])          # stands in for the SAM3 worker subprocess
pub = CloudPublisher("tcp://127.0.0.1:5571")
try:
    while True:
        pub.observe(np.random.rand(300, 3), None, dict(accepted=True, frame=1, transform=np.eye(4))); time.sleep(0.3)
except KeyboardInterrupt:
    print("TRACKER: SIGINT, cleaning up", flush=True); time.sleep(1.0); print("TRACKER: done", flush=True)
'''
RUNNER = '''
import sys, time, signal
hang = "hang" in sys.argv
if hang:
    signal.signal(signal.SIGINT, signal.SIG_IGN); signal.signal(signal.SIGTERM, signal.SIG_IGN)   # simulates a stuck ROS shutdown
print("RUNNER: started (hang=%s)" % hang, flush=True)
try:
    while True: time.sleep(0.2)
except KeyboardInterrupt:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print("RUNNER: SIGINT -> hold robot, close controllers", flush=True); time.sleep(2.0); print("RUNNER: closed", flush=True)
'''
MT, MR = '/tmp/rdx_mock_tracker.py', '/tmp/rdx_mock_runner.py'
open(MT, 'w').write(TRACKER)
open(MR, 'w').write(RUNNER)
PAT = 'rdx_mock|launch.py --dry-run|sleep 600'


def procs():
    out = subprocess.run(['pgrep', '-af', PAT], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if 'pgrep' not in l and 'launch_ctrlc_test' not in l]


def scenario(name, runner_extra, ctrlc_times):
    print('\n===== scenario:', name, flush=True)
    env = dict(os.environ, RDX_TRACKER_CMD=f'{PY} {MT}', RDX_RUNNER_CMD=f'{PY} {MR} robustdex_deploy.runner {runner_extra}')
    out = open('/tmp/rdx_launch.out', 'w')
    p = subprocess.Popen([PY, 'robustdex_deploy/launch.py', '--dry-run'], env=env, stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    time.sleep(6)
    print('before: %d processes' % len(procs()))
    t0 = time.time()
    for i, dt in enumerate(ctrlc_times):
        time.sleep(dt)
        os.killpg(p.pid, signal.SIGINT)       # == Ctrl-C on the terminal (launcher + runner process group)
        print('  Ctrl-C #%d at +%.1fs' % (i + 1, time.time() - t0), flush=True)
    rc = p.wait(90)
    print('launcher exit %s after %.1fs' % (rc, time.time() - t0))
    time.sleep(1)
    print(open('/tmp/rdx_launch.out').read())
    left = procs()
    print('left:', '\n      '.join(left) if left else '(none)')
    for l in left:
        try:
            os.kill(int(l.split()[0]), signal.SIGKILL)
        except Exception:
            pass
    return not left


ok = [scenario('single Ctrl-C', '', [0]),
      scenario('three rapid Ctrl-C', '', [0, 0.3, 0.3]),
      scenario('runner stuck in shutdown (ignores SIGINT/SIGTERM)', 'hang', [0])]
for f in (MT, MR):
    os.remove(f)
print('\nRESULT:', 'ALL CLEAN' if all(ok) else 'LEFTOVERS in %d scenario(s)' % (len(ok) - sum(ok)))
