"""One-command launcher on mainpc: starts the stereo object tracker (base anaconda env, GPUs) as a background process
with the cloud publisher + policy overlay enabled, waits for the first object cloud, then runs the policy runner in the
foreground inside the ROS2 `robot` env (prompts / Esc work on this terminal). The two processes talk over ZMQ only, so
the 3 Hz tracker and the 5 Hz policy never block each other; both draw into the same Viser viewer (port 8080).

    python robustdex_deploy/launch.py [--prompt OBJECT_PROMPT] [runner args...] [-- extra stereo_track args...]
    e.g. python robustdex_deploy/launch.py --prompt red_cup --dry-run
         python robustdex_deploy/launch.py --prompt yellow_mustard_bottle --allow-live --scale 0.5 --release -- --fusion-every 2

Shutdown: Ctrl-C reaches the launcher and the runner (same process group); the runner holds the robot and closes its
controllers while the launcher waits for it (up to 20 s, then SIGTERM), then the tracker's whole process group (incl.
its SAM3 worker) gets SIGINT, 20 s, then SIGKILL. Any other exit path (error, tracker crash) ends the same way.
Tracker stdout -> logs/robustdex_deploy/stereo_track_*.log.  --keep-tracker leaves the tracker running."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]
BASE_PY = HOME / 'anaconda3/bin/python'
ROS_PY = HOME / 'anaconda3/envs/robot/bin/python'
ROS_SETUPS = [HOME / 'ros2_humble/install/local_setup.bash', HOME / 'xarm_ws/install/local_setup.bash',
              HOME / 'allegro_ws/install/local_setup.bash']
TRACKER = REPO / 'src/capture/camera/stereo_track.py'
CLOUD, STATE = 'tcp://127.0.0.1:5571', 'tcp://127.0.0.1:5572'
DEFAULT_TRACKER_ARGS = ['--local-four-pair', '--stereo-backend', 'fast-foundation-trt',
                        '--fast-root', 'output/stereo_tracking/optimization_ab_20260922/assets/Fast-FoundationStereo',
                        '--fast-engine-dir', 'output/stereo_tracking/fast_trt_ab_20260923/twostage_trt103', '--fast-trt-cuda-graph',
                        '--device', '1', '--cutie-device', '0', '--sam3-device', '1', '--tracking-model', 'stable',
                        '--icp-method', 'robust_multiscale', '--fusion-every', '3', '--frames', '100000',
                        '--visualize', '--viser-port', '8080', '--keep-viewer-open']
GRACE = 20.0


def log(msg):
    print('[launch] ' + msg, flush=True)


def stop_group(proc, name, grace=GRACE):
    """SIGINT the process group (children such as the SAM3 worker included), wait, then SIGKILL what is left."""
    if proc.poll() is not None:
        return
    log(f'stopping {name} (pid {proc.pid}) with SIGINT')
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(grace)
        log(f'{name} exited ({proc.returncode})')
    except subprocess.TimeoutExpired:
        log(f'{name} still alive after {grace:.0f}s -> SIGKILL')
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(5)


def build_commands(argv):
    runner_args, tracker_args = (argv[:argv.index('--')], argv[argv.index('--') + 1:]) if '--' in argv else (argv, [])
    keep = '--keep-tracker' in runner_args
    runner_args = [a for a in runner_args if a != '--keep-tracker']
    prompt = 'yellow_mustard_bottle'                       # --prompt NAME: SAM3 init + segment prompt of the object
    if '--prompt' in runner_args:
        i = runner_args.index('--prompt')
        prompt = runner_args[i + 1]
        del runner_args[i:i + 2]
    if '--export' not in runner_args:
        runner_args = ['--export', str(REPO / 'robustdex_deploy/export_v5_3650')] + runner_args
    if '--backend' not in runner_args:
        runner_args = ['--backend', 'real'] + runner_args
    if not any(a.startswith('--init-prompt') or a.startswith('--reference-object') for a in tracker_args):
        tracker_args = ['--init-prompt', prompt, '--segment-prompt', prompt] + tracker_args
    if '--stereo-backend' not in tracker_args:
        tracker_args = DEFAULT_TRACKER_ARGS + tracker_args
    tracker_args += ['--cloud-pub', CLOUD, '--policy-sub', STATE]
    tracker_cmd = [str(BASE_PY), str(TRACKER)] + tracker_args
    src = ' && '.join(f'source "{p}"' for p in ROS_SETUPS if p.is_file())
    runner_cmd = ['bash', '-c', f'{src} && exec "{ROS_PY}" -u -m robustdex_deploy.runner "$@"', 'bash'] + runner_args + \
                 ['--cloud-endpoint', CLOUD, '--state-pub', STATE]
    # test hook: RDX_TRACKER_CMD / RDX_RUNNER_CMD replace the real programs (shell strings)
    if os.environ.get('RDX_TRACKER_CMD'):
        tracker_cmd = ['bash', '-c', os.environ['RDX_TRACKER_CMD']]
    if os.environ.get('RDX_RUNNER_CMD'):
        runner_cmd = ['bash', '-c', os.environ['RDX_RUNNER_CMD']]
    return tracker_cmd, runner_cmd, keep


def main():
    tracker_cmd, runner_cmd, keep = build_commands(sys.argv[1:])
    logdir = REPO / 'logs/robustdex_deploy'
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = open(logdir / time.strftime('stereo_track_%Y%m%d_%H%M%S.log'), 'w')
    env = dict(os.environ, OMP_NUM_THREADS='4')
    # own session/process group: the terminal's Ctrl-C does not reach the tracker; we stop it ourselves, in order
    tracker = subprocess.Popen(tracker_cmd, cwd=REPO, stdout=logfile, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    log(f'tracker pid {tracker.pid}, log {logfile.name}; waiting for the first object cloud ...')
    runner = None
    rc = 1
    try:
        sys.path.insert(0, str(REPO))
        from robustdex_deploy.cloud import CloudClient
        client = CloudClient(CLOUD)
        t0 = time.monotonic()
        while True:
            if tracker.poll() is not None:
                log(f'tracker exited early ({tracker.returncode}); see {logfile.name}')
                return 1
            try:
                client.wait(5.0)
                break
            except TimeoutError:
                log('... still waiting for the tracker (%.0f s)' % (time.monotonic() - t0))
        log('cloud ok (%d points); starting the policy runner' % len(client.latest[1]))
        # same process group as the launcher: Ctrl-C reaches the runner directly and it does its own hold/cleanup
        runner_log = logdir / time.strftime('runner_%Y%m%d_%H%M%S.log')
        # stdout/stderr go to the terminal AND a log file (tee), stdin stays on the terminal for the prompts
        # tee must survive the terminal's Ctrl-C (it is in the same process group) or the runner's shutdown messages
        # would hit a broken pipe
        runner = subprocess.Popen(f'{{ {shell_join(runner_cmd)} ; }} 2>&1 | {{ trap "" INT; exec tee "{runner_log}"; }}',
                                  shell=True, cwd=REPO, executable='/bin/bash')
        log(f'runner log {runner_log}')
        try:
            rc = runner.wait()
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)      # one Ctrl-C starts the shutdown; more are ignored
            log('Ctrl-C: waiting for the runner to hold the robot and exit (up to %.0fs) ...' % GRACE)
            try:
                rc = runner.wait(GRACE)
            except subprocess.TimeoutExpired:
                log('runner did not exit in %.0fs -> SIGTERM' % GRACE)
                rc = _force_stop_runner(runner)
        log(f'runner exited ({rc})')
        return rc
    except KeyboardInterrupt:
        log('Ctrl-C before the runner started')
        return 130
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if runner is not None and (runner.poll() is None or _runner_pids()):
                _force_stop_runner(runner)
        except Exception as e:
            log(f'runner cleanup error: {e!r}')
        try:
            if keep:
                log(f'leaving the tracker running (pid {tracker.pid})')
            else:
                stop_group(tracker, 'tracker')
        except Exception as e:
            log(f'tracker cleanup error: {e!r}')
        logfile.close()


def shell_join(cmd):
    import shlex
    return ' '.join(shlex.quote(c) for c in cmd)


def _runner_pids():
    """Python runner processes (the tee pipeline hides the real pid behind a bash wrapper, so look them up)."""
    out = subprocess.run(['pgrep', '-af', 'robustdex_deploy.runne[r]'], capture_output=True, text=True).stdout
    return [int(l.split()[0]) for l in out.splitlines() if 'python' in l.split()[1] and 'bash' not in l.split()[1]]


def _force_stop_runner(runner):
    """SIGTERM the python runner(s) (-> their hold/cleanup path), wait, SIGKILL whatever is left; reap the pipeline."""
    for sig, wait_s in ((signal.SIGTERM, 8.0), (signal.SIGKILL, 5.0)):
        pids = _runner_pids()
        if not pids:
            break
        log(f'sending {sig.name} to runner pid(s) {pids}')
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        t0 = time.monotonic()
        while time.monotonic() - t0 < wait_s and _runner_pids():
            time.sleep(0.2)
    try:
        return runner.wait(5)
    except subprocess.TimeoutExpired:
        runner.kill()
        return runner.wait(5)


if __name__ == '__main__':
    sys.exit(main())
