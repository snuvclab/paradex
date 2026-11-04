# VIVE arm smoothing

HRI capture_robot.py now defaults to --vive-arm-smoothing-ms 80 (time constant; 0 disables). Position uses exponential smoothing; rotation uses shortest-arc SO(3) interpolation with the same time-based gain. Larger values add smoothing and delay. No hand-retargeting changes.

The raw tracker jump/speed guard runs BEFORE smoothing and retains its existing limits. Small jitter is smoothed; large raw jumps still print Rejected VIVE arm command and hold. The guard timestamp now advances only for changed tracker poses, so repeated reads of a cached frame cannot make a normal next frame look implausibly fast.

The smoother initializes from the measured arm home pose at each teleop start, separately for both arms, and reinitializes on pedal re-enable along with the existing VIVE rebase. Per-call filter dt is capped at 50ms so a long pause cannot directly snap the filter to its target. No running controller was restarted or robot command issued during deployment. Restart capture_robot.py to apply.

52 targeted tests passed, including existing capture/pedal guard tests, position jitter suppression, SO(3) validity/shortest rotation, frame-rate invariance, convergence, fresh reset, disabled smoothing, duplicate-frame guard timing and raw jump rejection. Controller ret=1/HAS_ERROR requires separate diagnosis; smoothing does not clear robot errors.
