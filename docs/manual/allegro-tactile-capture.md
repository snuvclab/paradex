# Allegro V5 tactile capture

capture_robot.py --hand allegro_v5 now enables tactile recording by default; --tactile is still accepted and --no-tactile opts out (real-time tactile visualization still requires the tactile subscription). Start/stop with the usual capture keys. Restart an already-running capture program to load this code. No camera-PC changes are needed.

Under each episode raw/hand/ (or raw/hand_left and raw/hand_right for bimanual):
- tactile_raw.npy: independent received sensor packets, int32 N x channels, channel order unchanged.
- tactile_time.npy: corresponding host receive Unix seconds, N.
- tactile_monotonic_time.npy: corresponding perf_counter seconds, N.
- tactile_info.json: topic, sample count, clock descriptions, malformed packet count.
- tactile.npy: existing command-sampled cache retained for compatibility. Use tactile_raw/time for synchronization; they are independent of the command/position time.npy sampling.

Int32MultiArray has no sensor acquisition timestamp: receive times are not hardware-synchronized timestamps. Recording continues while the teleoperation pedal/command loop is paused, while the capture session remains active. No packets before/after the episode are included. Empty or inconsistent channel-width packets are counted as malformed. A zero-sample episode emits a warning and sample count 0; this change does not start or modify the tactile driver.

Verification: callback/start/stop tests without ROS/hardware cover independent high-rate packets, idle command loop, episode boundaries, missing packets and malformed packets. CLI default/override parsing checked. No motion or actual capture was launched.
