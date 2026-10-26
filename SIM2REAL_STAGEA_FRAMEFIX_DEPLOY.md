# Sim-to-real Stage A + object-frame deployment

This bundle is a repo-root overlay for an existing Paradex checkout. It contains
only the three H2R runners, their new shared helper, documentation, and focused
tests. Existing robot configs, meshes, calibration, and controller files are not
included.

From the Paradex repository root:

```bash
tar -xzf paradex_sim2real_stagea_framefix_20260914.tar.gz
python -m pytest -q tests/test_hrdex_replay_common.py
```

The target checkout must already provide
`paradex/io/robot_controller/xarm_controller_ros.py` with
`move_joint_trajectory()`. Verify before connecting hardware:

```bash
rg "def move_joint_trajectory" paradex/io/robot_controller/xarm_controller_ros.py
```

Do the first pipeline run without `--yes`, inspect the complete Viser trajectory,
and use an empty robot workspace for the first physical low-speed replay.

