#!/usr/bin/env python3
"""Sequential, feedback-checked Allegro v5 range-of-motion diagnostic.

Use this for a manufacturer support video, not during normal teleoperation.
It moves one joint at a time: current pose -> near extension -> near flexion ->
the starting pose.  All inactive joints remain at their measured start pose.

The script is intentionally dry-run by default.  It only publishes to the
physical hand when both ``--execute`` and ``--confirm-physical-clear`` are
present.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from paradex.io.robot_controller.allegro_v5_controller_ros2 import AllegroController


# allegro_right_A.urdf limits, in the hardware driver's index/middle/ring/thumb
# order: joint_0_0 through joint_15_0.
PHYSICAL_LOWER = np.array(
    [-0.47, -0.196, -0.174, -0.227] * 3 + [0.0, -0.26, 0.0, -0.09],
    dtype=np.float64,
)
PHYSICAL_UPPER = np.array(
    [0.47, 1.8, 1.709, 1.618] * 3 + [1.78, 1.78, 1.9, 1.8],
    dtype=np.float64,
)
JOINT_COUNT = 16


def _parse_joints(value: str) -> list[int]:
    try:
        joints = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--joints must be comma-separated integers") from error
    if not joints or any(index < 0 or index >= JOINT_COUNT for index in joints):
        raise argparse.ArgumentTypeError("--joints must contain values from 0 through 15")
    if len(set(joints)) != len(joints):
        raise argparse.ArgumentTypeError("--joints must not contain duplicates")
    return joints


def _wait_for_feedback(hand: AllegroController, timeout_seconds: float) -> np.ndarray:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        data = hand.get_data()
        if data["is_connected"] and data["state_monotonic_time"] is not None:
            return np.asarray(data["qpos"], dtype=np.float64)
        time.sleep(0.02)
    raise TimeoutError(f"No Allegro feedback received within {timeout_seconds:.1f} s")


def _move_and_verify(
    hand: AllegroController,
    target: np.ndarray,
    joint_index: int,
    speed_rad_s: float,
    tolerance_rad: float,
    timeout_margin_seconds: float,
) -> tuple[bool, float | None]:
    """Issue a target and wait for the selected joint to approach it."""
    current = _wait_for_feedback(hand, timeout_seconds=1.0)
    estimated_seconds = abs(target[joint_index] - current[joint_index]) / speed_rad_s
    deadline = time.monotonic() + estimated_seconds + timeout_margin_seconds
    hand.move(target)
    while time.monotonic() < deadline:
        feedback = _wait_for_feedback(hand, timeout_seconds=1.0)
        error = float(feedback[joint_index] - target[joint_index])
        if abs(error) <= tolerance_rad:
            return True, error
        time.sleep(0.02)
    feedback = _wait_for_feedback(hand, timeout_seconds=1.0)
    return False, float(feedback[joint_index] - target[joint_index])


def _target_from_start(start: np.ndarray, joint_index: int, value: float) -> np.ndarray:
    target = start.copy()
    target[joint_index] = value
    return target


def _restore_start_pose(
    hand: AllegroController,
    start: np.ndarray,
    speed_rad_s: float,
    tolerance_rad: float,
    timeout_margin_seconds: float,
) -> tuple[bool, float]:
    """Return to the initial whole-hand pose before the publisher is stopped."""
    current = _wait_for_feedback(hand, timeout_seconds=1.0)
    estimated_seconds = float(np.max(np.abs(start - current))) / speed_rad_s
    deadline = time.monotonic() + estimated_seconds + timeout_margin_seconds
    hand.move(start)
    while time.monotonic() < deadline:
        feedback = _wait_for_feedback(hand, timeout_seconds=1.0)
        max_error = float(np.max(np.abs(feedback - start)))
        if max_error <= tolerance_rad:
            return True, max_error
        time.sleep(0.02)
    feedback = _wait_for_feedback(hand, timeout_seconds=1.0)
    return False, float(np.max(np.abs(feedback - start)))


def _ensure_exclusive_command_publisher(hand: AllegroController) -> None:
    """Refuse to fight a capture/teleop process on the same command topic."""
    publishers = hand._state_node.get_publishers_info_by_topic(hand.command_topic)
    # This diagnostic contributes one publisher itself.  Any additional publisher
    # can overwrite its held pose or, worse, resume the unsafe teleop stream.
    if len(publishers) != 1:
        raise RuntimeError(
            f"Expected exactly this diagnostic publisher on {hand.command_topic}; "
            f"found {len(publishers)}. Stop all teleop/capture processes first."
        )


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="right", help="Allegro ROS namespace (default: right)")
    parser.add_argument("--joints", type=_parse_joints, default=list(range(JOINT_COUNT)))
    parser.add_argument("--speed-rad-s", type=float, default=0.20)
    parser.add_argument("--dwell-seconds", type=float, default=0.75)
    parser.add_argument("--limit-margin-rad", type=float, default=0.03)
    parser.add_argument("--tolerance-rad", type=float, default=0.08)
    parser.add_argument("--feedback-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-margin-seconds", type=float, default=2.0)
    parser.add_argument(
        "--continue-on-timeout",
        action="store_true",
        help="Record an unreachable endpoint and continue with the next joint",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--execute", action="store_true", help="Allow physical motion")
    parser.add_argument(
        "--confirm-physical-clear",
        action="store_true",
        help="Confirm fingers, cables, and surroundings are clear",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.speed_rad_s <= 0.0:
        raise ValueError("--speed-rad-s must be positive")
    if args.dwell_seconds < 0.0 or args.limit_margin_rad < 0.0:
        raise ValueError("dwell and limit margin must be non-negative")
    if args.tolerance_rad <= 0.0 or args.feedback_timeout_seconds <= 0.0:
        raise ValueError("tolerance and feedback timeout must be positive")
    if np.any(PHYSICAL_LOWER + args.limit_margin_rad >= PHYSICAL_UPPER - args.limit_margin_rad):
        raise ValueError("--limit-margin-rad leaves no valid range for at least one joint")


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    low = PHYSICAL_LOWER + args.limit_margin_rad
    high = PHYSICAL_UPPER - args.limit_margin_rad
    print("Joint limits used (physical limits minus margin):")
    for index in args.joints:
        print(f"  joint_{index}_0: {low[index]:+.3f} -> {high[index]:+.3f} rad")
    if not (args.execute and args.confirm_physical_clear):
        print("Dry run only. Add --execute --confirm-physical-clear to publish commands.")
        return 0

    hand = AllegroController(namespace=args.namespace)
    report: dict = {
        "namespace": args.namespace,
        "joint_order": [f"joint_{index}_0" for index in range(JOINT_COUNT)],
        "limit_margin_rad": args.limit_margin_rad,
        "speed_rad_s": args.speed_rad_s,
        "results": [],
    }
    start: np.ndarray | None = None
    restore_ok = None
    try:
        start = _wait_for_feedback(hand, args.feedback_timeout_seconds)
        _ensure_exclusive_command_publisher(hand)
        hand.set_command_slew_rate(args.speed_rad_s, initial_action=start)
        report["start_qpos"] = start.tolist()
        for index in args.joints:
            print(f"\nTesting joint_{index}_0: extend {low[index]:+.3f}, flex {high[index]:+.3f}")
            result = {"joint": index, "start": float(start[index])}
            for label, value in (("extend", low[index]), ("flex", high[index]), ("restore", start[index])):
                target = _target_from_start(start, index, value)
                reached, error = _move_and_verify(
                    hand,
                    target,
                    index,
                    args.speed_rad_s,
                    args.tolerance_rad,
                    args.timeout_margin_seconds,
                )
                result[label] = {"target": float(value), "error": error, "reached": reached}
                print(f"  {label}: {'OK' if reached else 'TIMEOUT'} (error {error:+.3f} rad)")
                if not reached:
                    result["timeout_at"] = label
                    if not args.continue_on_timeout:
                        report["results"].append(result)
                        raise RuntimeError(f"joint_{index}_0 did not reach its {label} target")
                time.sleep(args.dwell_seconds)
            report["results"].append(result)
    except (KeyboardInterrupt, TimeoutError, RuntimeError) as error:
        report["error"] = str(error)
        print(f"\nStopped: {error}", file=sys.stderr)
        return 2
    finally:
        if start is not None:
            try:
                restore_ok, restore_error = _restore_start_pose(
                    hand,
                    start,
                    args.speed_rad_s,
                    args.tolerance_rad,
                    args.timeout_margin_seconds,
                )
                report["final_restore"] = {
                    "reached": restore_ok,
                    "max_error": restore_error,
                }
                if not restore_ok:
                    print(
                        f"Final restore did not converge (max error {restore_error:.3f} rad)",
                        file=sys.stderr,
                    )
            except Exception as restore_error:
                report["final_restore"] = {"reached": False, "error": str(restore_error)}
                print(f"Final restore failed: {restore_error}", file=sys.stderr)
        hand.end()
        if args.report:
            _write_report(args.report, report)
            print(f"Report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
