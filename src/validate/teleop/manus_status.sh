#!/usr/bin/env bash
set -eo pipefail
source "$HOME/ros2_humble/install/setup.bash"
source "$HOME/vive-teleop/ros2_ws/install/setup.bash"
cd "$HOME/paradex"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$HOME/anaconda3/envs/robot/bin/python" -u src/validate/teleop/manus_status.py "$@"
