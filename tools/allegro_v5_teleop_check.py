"""Check live MANUS/VIVE -> V5 DexPilot and ROS feedback without commanding.

Run from the repo: python -m tools.allegro_v5_teleop_check --duration 15
"""

import argparse
import json
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=15.)
    parser.add_argument("--scale", type=float, default=1.15)
    parser.add_argument("--no-vive", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    from paradex.retargetor.allegro_v5_dexpilot import AsyncAllegroV5DexPilotRetargeter
    from paradex.io.teleop.vive.receiver import ViveManusROSReceiver
    from paradex.io.robot_controller.allegro_v5_controller_ros2 import AllegroController

    retargeter = receiver = hand = None
    samples = 0
    connected = False
    latencies = []
    try:
        retargeter = AsyncAllegroV5DexPilotRetargeter(scaling_factor=args.scale)
        receiver = ViveManusROSReceiver(use_vive=not args.no_vive, require_left_control=False)
        hand = AllegroController(hand_side="right", command_enabled=False)
        print("READ ONLY: joint command and library command publishers are disabled.", flush=True)
        deadline = time.monotonic() + args.duration
        report_at = 0.
        while time.monotonic() < deadline:
            frame = receiver.get_data()
            feedback = hand.get_data()
            connected = connected or feedback["is_connected"]
            if frame is not None and frame.get("Right") is not None:
                start = time.monotonic()
                command = retargeter(frame["Right"])
                latencies.append(time.monotonic() - start)
                samples += 1
                if time.monotonic() >= report_at:
                    print(json.dumps({
                        "samples": samples, "connected": feedback["is_connected"],
                        "target_rad": command.round(3).tolist(),
                        "measured_rad": feedback["qpos"].round(3).tolist(),
                    }), flush=True)
                    report_at = time.monotonic() + 2.
            time.sleep(1. / 30.)
        print(json.dumps({
            "samples": samples, "allegro_connected": connected,
            "call_latency_p95_ms": float(np.percentile(latencies, 95) * 1000) if latencies else None,
            "manus": receiver.get_diagnostics(),
        }, indent=2), flush=True)
        return 0 if samples and connected else 2
    finally:
        if retargeter is not None:
            retargeter.close()
        # Receiver owns the rclpy context; destroy hand subscribers first.
        if hand is not None:
            hand.end()
        if receiver is not None:
            receiver.end()


if __name__ == "__main__":
    raise SystemExit(main())
