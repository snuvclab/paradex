from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


DEFAULT_DATASET_ROOT = Path(
    "/home/temp_id/shared_data/capture/eccv2026/v0/allegro_v5"
)
DEFAULT_OUTPUT_NAME = "eef_apex_annotations.json"


def _numeric_episode_dirs(object_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in object_root.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )


def _load_arm_trajectory(episode_root: Path) -> tuple[np.ndarray, np.ndarray]:
    raw_arm = episode_root / "raw" / "arm"
    poses = np.load(raw_arm / "action.npy", allow_pickle=True).astype(np.float64)
    times = np.load(raw_arm / "time.npy", allow_pickle=True).astype(np.float64)
    times = times.reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        raise ValueError(f"empty or malformed arm action array: {poses.shape}")
    count = min(len(poses), len(times))
    if count < 3:
        raise ValueError(f"arm trajectory is too short: {count} frame(s)")
    poses, times = poses[:count], times[:count]
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(times)):
        raise ValueError("arm trajectory contains non-finite values")
    time_deltas = np.diff(times)
    if np.any(time_deltas < 0) or not np.any(time_deltas > 0):
        raise ValueError("arm timestamps must be nondecreasing with positive progress")
    return poses, times


def _odd_window(sample_period: float, duration: float, frame_count: int) -> int:
    window = max(3, int(round(duration / sample_period)))
    if window % 2 == 0:
        window += 1
    largest_odd = frame_count if frame_count % 2 == 1 else frame_count - 1
    return min(window, largest_odd)


def annotate_episode(
    episode_root: Path,
    *,
    smoothing_seconds: float,
    minimum_prominence_m: float,
    minimum_peak_separation_seconds: float,
    search_start_fraction: float,
    search_end_fraction: float,
) -> dict[str, object]:
    poses, times = _load_arm_trajectory(episode_root)
    z = poses[:, 2, 3]
    time_deltas = np.diff(times)
    sample_period = float(np.median(time_deltas[time_deltas > 0]))
    smoothing_window = _odd_window(sample_period, smoothing_seconds, len(z))
    half_window = smoothing_window // 2
    smoothed_z = np.convolve(
        np.pad(z, half_window, mode="edge"),
        np.ones(smoothing_window, dtype=np.float64) / smoothing_window,
        mode="valid",
    )
    minimum_peak_distance = max(
        1, int(round(minimum_peak_separation_seconds / sample_period))
    )
    peaks, properties = find_peaks(
        smoothed_z,
        prominence=minimum_prominence_m,
        distance=minimum_peak_distance,
    )
    search_start = int(np.ceil(search_start_fraction * len(z)))
    search_end = int(np.floor(search_end_fraction * len(z)))
    candidate_indices = [
        index
        for index, frame in enumerate(peaks)
        if search_start <= frame <= search_end
    ]
    if not candidate_indices:
        raise ValueError(
            "no interior EEF height peak satisfies the configured prominence"
        )

    selected = max(
        candidate_indices,
        key=lambda index: float(properties["prominences"][index]),
    )
    frame = int(peaks[selected])
    prominence = float(properties["prominences"][selected])
    left_base = int(properties["left_bases"][selected])
    right_base = int(properties["right_bases"][selected])
    edge_margin = min(frame / len(z), 1.0 - frame / len(z))
    if prominence < 0.03 or edge_margin < 0.12:
        review = "recommended"
    elif prominence < 0.05:
        review = "optional"
    else:
        review = "not_required"

    return {
        "arm_frame_index": frame,
        "arm_timestamp": float(times[frame]),
        "elapsed_seconds": float(times[frame] - times[0]),
        "eef_pose_robot": poses[frame].tolist(),
        "eef_position_robot_m": poses[frame, :3, 3].tolist(),
        "eef_height_m": float(z[frame]),
        "smoothed_eef_height_m": float(smoothed_z[frame]),
        "peak_prominence_m": prominence,
        "left_valley_arm_frame": left_base,
        "right_valley_arm_frame": right_base,
        "trajectory_frame_count": len(z),
        "manual_review": review,
    }


def generate_annotations(args: argparse.Namespace) -> dict[str, object]:
    dataset_root = args.dataset_root.expanduser().resolve()
    annotations: dict[str, dict[str, object]] = {}
    invalid_episodes: list[dict[str, str]] = []
    review_counts = {"not_required": 0, "optional": 0, "recommended": 0}
    episode_count = 0

    for object_root in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        object_annotations: dict[str, object] = {}
        for episode_root in _numeric_episode_dirs(object_root):
            episode_count += 1
            try:
                annotation = annotate_episode(
                    episode_root,
                    smoothing_seconds=args.smoothing_seconds,
                    minimum_prominence_m=args.minimum_prominence_m,
                    minimum_peak_separation_seconds=(
                        args.minimum_peak_separation_seconds
                    ),
                    search_start_fraction=args.search_start_fraction,
                    search_end_fraction=args.search_end_fraction,
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                invalid_episodes.append(
                    {
                        "object": object_root.name,
                        "episode": episode_root.name,
                        "reason": str(exc),
                    }
                )
                continue
            object_annotations[episode_root.name] = annotation
            review_counts[str(annotation["manual_review"])] += 1
        if object_annotations:
            annotations[object_root.name] = object_annotations

    annotated_count = sum(len(value) for value in annotations.values())
    return {
        "schema_version": 1,
        "description": (
            "Automatically detected pick-place apex from robot-base EEF height. "
            "Frame indices refer directly to raw/arm/action.npy and are zero-based."
        ),
        "dataset_root": str(dataset_root),
        "coordinate_frame": "robot_base",
        "source": "raw/arm/action.npy",
        "method": {
            "signal": "eef_pose[2, 3]",
            "selection": (
                "largest-prominence interior local maximum after temporal smoothing"
            ),
            "smoothing_seconds": args.smoothing_seconds,
            "minimum_prominence_m": args.minimum_prominence_m,
            "minimum_peak_separation_seconds": (
                args.minimum_peak_separation_seconds
            ),
            "search_fraction": [
                args.search_start_fraction,
                args.search_end_fraction,
            ],
        },
        "summary": {
            "episode_count": episode_count,
            "annotated_count": annotated_count,
            "invalid_count": len(invalid_episodes),
            "manual_review_counts": review_counts,
        },
        "annotations": annotations,
        "invalid_episodes": invalid_episodes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate the EEF-height apex of each replay episode."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoothing-seconds", type=float, default=0.12)
    parser.add_argument("--minimum-prominence-m", type=float, default=0.01)
    parser.add_argument(
        "--minimum-peak-separation-seconds", type=float, default=0.4
    )
    parser.add_argument("--search-start-fraction", type=float, default=0.15)
    parser.add_argument("--search-end-fraction", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.0 <= args.search_start_fraction < args.search_end_fraction <= 1.0:
        parser.error("search fractions must satisfy 0 <= start < end <= 1")
    if args.smoothing_seconds <= 0 or args.minimum_peak_separation_seconds <= 0:
        parser.error("time parameters must be positive")
    if args.minimum_prominence_m <= 0:
        parser.error("--minimum-prominence-m must be positive")
    return args


def main() -> None:
    args = parse_args()
    output_path = (
        args.output.expanduser()
        if args.output is not None
        else args.dataset_root.expanduser() / DEFAULT_OUTPUT_NAME
    )
    result = generate_annotations(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = result["summary"]
    print(f"Saved EEF apex annotations: {output_path}")
    print(
        f"Annotated {summary['annotated_count']}/{summary['episode_count']} episode(s); "
        f"invalid={summary['invalid_count']}, review={summary['manual_review_counts']}"
    )


if __name__ == "__main__":
    main()
