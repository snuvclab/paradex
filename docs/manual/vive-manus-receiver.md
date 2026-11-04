# VIVE / MANUS callback execution

HRI `capture_robot.py` defaults to `--teleop-receiver-mode split`.
When both inputs are enabled, VIVE subscriptions use one ROS node and
SingleThreadedExecutor thread; MANUS subscriptions use a different node and
executor thread. MANUS frame reconstruction no longer occupies the executor
that dispatches VIVE callbacks. Completed input frames remain protected by
the existing short state lock. Recording and retargeting consume that state
through the existing receiver API.

For comparison with the previous callback scheduling, pass
`--teleop-receiver-mode shared`. Restart capture to change modes. The startup
log and teleop recording metadata include the selected callback mode.
VIVE-only (`--hand None`) and MANUS-only (`--no-vive`) sessions create just
one executor thread. Shutdown stops and joins both threads before destroying
their nodes and releasing an owned ROS context.

HRI also defaults to `--teleop-control-mode split`. A dedicated arm target
thread reads only fresh VIVE poses, maps wrists, applies the existing motion
limiter/smoothing option, and updates the arm drivers. A different hand target
thread reads MANUS frames in their wrist frame, retargets fingers, reads hand
feedback and updates hand drivers. Each loop waits 10ms between iterations;
the configured hand command rate cap still applies. Driver publish rates are
unchanged. There is no FIFO of old targets between these loops.

MANUS loss holds hand targets while fresh VIVE can continue moving arms;
VIVE loss holds arm targets while fresh MANUS can continue moving fingers.
The pedal controls both loops. Releasing it invalidates computations already
in flight; re-enabling rebases the arm mapping to actual arm feedback. A loop
exception is reported and holds that lane while the other lane continues.
The next session refuses to start if an old control thread has not stopped.
Use `--teleop-control-mode shared` for the previous joint target loop.

The coordinator retains keyboard/session transitions and synchronized input
recording. The combined recording `get_data()` still requires all configured
streams to be fresh within 250ms, so a source outage creates gaps in the
synchronized input recording even though the other command loop continues.

These are threads in the same Python process. Python GIL, CPU, ROS and OS
resource contention can still occur; this is independent scheduling and
input validity, not complete process/resource isolation. No hardware latency
improvement is claimed without a hardware measurement. Camera preparation
still completes before teleoperation starts.
