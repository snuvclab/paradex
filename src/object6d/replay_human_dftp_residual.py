"""Thin remote-PC wrapper for replay_human_dftp.py plus local residual RPC.

The original replay script is imported unchanged.  A complete trajectory is
planned first, sent once to the model PC, validated again after correction,
and only then exposed to the original preview/execute path.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import replay_human_dftp as base


DEFAULT_SERVER = "http://192.168.0.69:5571"
TRAINING_SCALE = 1.5414379142433483


def _get_json(server: str, path: str, *, timeout_s: float, token: str | None) -> dict[str, Any]:
    headers = {"X-HRDEXDB-Token": token} if token else {}
    request = urllib.request.Request(
        server.rstrip("/") + path, headers=headers, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"residual server metadata request failed: {exc}") from exc


def _request(
    server: str,
    values: Mapping[str, np.ndarray],
    *,
    timeout_s: float,
    token: str | None,
) -> dict[str, np.ndarray]:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    headers = {"Content-Type": "application/octet-stream"}
    if token:
        headers["X-HRDEXDB-Token"] = token
    request = urllib.request.Request(
        server.rstrip("/") + "/infer",
        data=buffer.getvalue(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"residual server rejected the plan: {detail.strip()}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"residual server unavailable: {exc}") from exc
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _frame_ids(episode_root: Path) -> np.ndarray:
    files = sorted(
        (episode_root / "hand/mano_params").glob("*.json"),
        key=lambda path: int(path.stem),
    )
    result = np.asarray([int(path.stem) for path in files], dtype=np.int64)
    if len(result) == 0 or not np.array_equal(result, np.arange(result[0], result[0] + len(result))):
        raise ValueError("MANO source frame IDs must be non-empty and contiguous")
    return result


def _safety_limits(safety: Mapping[str, float] | None) -> dict[str, float]:
    result = {
        "min_z_m": 0.03,
        "max_z_m": 1.00,
        "max_radius_m": 0.90,
        "max_linear_step_m": 0.04,
        "max_angular_step_rad": 0.45,
        "max_hand_raw_step": 180.0,
    }
    if safety is not None:
        result.update({key: float(value) for key, value in safety.items()})
    return result


def _wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--residual-server", default=DEFAULT_SERVER)
    parser.add_argument("--residual-timeout-s", type=float, default=30.0)
    parser.add_argument("--residual-token")
    parser.add_argument(
        "--residual-start-fraction",
        type=float,
        default=0.70,
        help="smoothly ramp terminal-trained residual after this trajectory fraction",
    )
    parser.add_argument("--residual-smoothing-alpha", type=float, default=0.20)
    parser.add_argument(
        "--residual-max-abs-rad",
        type=float,
        default=0.35,
        help="per-joint model-residual clip; zero disables clipping",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    residual_args, remaining = _wrapper_parser().parse_known_args(arguments)
    if residual_args.residual_timeout_s <= 0:
        raise SystemExit("--residual-timeout-s must be positive")

    # Match the frozen training retargeter unless the caller explicitly supplied
    # a value.  A mismatched explicit value is rejected by the model server.
    if "--scaling-factor" not in remaining:
        remaining.extend(["--scaling-factor", str(TRAINING_SCALE)])
    if "--low-pass-alpha" not in remaining:
        remaining.extend(["--low-pass-alpha", "1.0"])

    if "--list-objects" not in remaining:
        health = _get_json(
            residual_args.residual_server,
            "/health",
            timeout_s=residual_args.residual_timeout_s,
            token=residual_args.residual_token,
        )
        if health.get("status") != "ready":
            raise RuntimeError(f"residual server is not ready: {health}")
        if abs(float(health["retarget_scaling_factor"]) - TRAINING_SCALE) > 1e-8:
            raise RuntimeError("wrapper/server frozen retarget scales disagree")
        anchor_payload = _get_json(
            residual_args.residual_server,
            "/anchors",
            timeout_s=residual_args.residual_timeout_s,
            token=residual_args.residual_token,
        )
        eligible = {
            (str(row["object_name"]), str(row["episode_id"]))
            for row in anchor_payload["anchors"]
        }
        original_discover = base.discover_episodes

        def discover_paired_episodes(root: Path, object_name: str) -> list[Path]:
            return [
                path
                for path in original_discover(root, object_name)
                if (object_name, path.name) in eligible
            ]

        base.discover_episodes = discover_paired_episodes
        print(
            "[residual] server ready: "
            f"epoch={health['checkpoint_epoch']}, anchors={len(eligible)}, "
            f"device={health['device']}",
            flush=True,
        )

    original_build = base.build_replay_plan
    original_save = base.save_plan
    diagnostics: dict[str, np.ndarray] = {}

    def build_with_residual(
        episode: base.HumanEpisode,
        current_object_robot: np.ndarray,
        **kwargs: Any,
    ) -> base.ReplayPlan:
        plan = original_build(episode, current_object_robot, **kwargs)
        scaling_factor = float(kwargs.get("scaling_factor", 1.15))
        request = {
            "object_name": np.asarray(episode.object_name),
            "episode_id": np.asarray(episode.episode_id),
            "joints_world": np.asarray(episode.joints_world, dtype=np.float64),
            "object_poses_world": np.asarray(episode.object_poses_world, dtype=np.float64),
            "source_frame_ids": _frame_ids(episode.root),
            "target_times_s": np.asarray(plan.times_s, dtype=np.float64),
            "baseline_qpos": np.asarray(plan.hand_qpos, dtype=np.float64),
            "source_hz": np.asarray(float(kwargs.get("source_hz", 30.0))),
            "speed_scale": np.asarray(float(kwargs.get("speed_scale", 1.0))),
            "retarget_scaling_factor": np.asarray(scaling_factor),
            "residual_start_fraction": np.asarray(residual_args.residual_start_fraction),
            "residual_smoothing_alpha": np.asarray(residual_args.residual_smoothing_alpha),
            "residual_max_abs_rad": np.asarray(residual_args.residual_max_abs_rad),
        }
        print(
            f"[residual] requesting {len(plan.times_s)} frame corrections from "
            f"{residual_args.residual_server}",
            flush=True,
        )
        result = _request(
            residual_args.residual_server,
            request,
            timeout_s=residual_args.residual_timeout_s,
            token=residual_args.residual_token,
        )
        corrected = np.asarray(result["corrected_qpos"], dtype=np.float64)
        if corrected.shape != plan.hand_qpos.shape or not np.isfinite(corrected).all():
            raise RuntimeError("residual server returned an invalid corrected trajectory")
        raw = base.hand_qpos_to_dftp_raw(corrected)
        errors = base.validate_plan(
            plan.link6_poses_robot,
            raw,
            **_safety_limits(kwargs.get("safety")),
        )
        diagnostics.clear()
        diagnostics.update(result)
        diagnostics["original_baseline_qpos"] = np.asarray(plan.hand_qpos, np.float32)
        diagnostics["corrected_hand_raw"] = np.asarray(raw, np.float32)
        diagnostics["server_url"] = np.asarray(residual_args.residual_server)
        diagnostics["safety_errors"] = np.asarray(errors)
        magnitude = np.abs(np.asarray(result["delta_applied_rad"], dtype=np.float64))
        print(
            "[residual] correction applied: "
            f"mean_abs={magnitude.mean():.4f} rad, max_abs={magnitude.max():.4f} rad; "
            f"anchor={result['anchor_query_id'].item()} ({result['anchor_split'].item()})",
            flush=True,
        )
        return dataclasses.replace(
            plan,
            hand_qpos=corrected,
            hand_raw=raw,
            safety_errors=errors,
        )

    def save_with_diagnostics(plan: base.ReplayPlan, output_root: Path) -> Path:
        path = original_save(plan, output_root)
        sidecar = path.with_name(path.stem + "_residual.npz")
        with sidecar.open("wb") as handle:
            np.savez_compressed(handle, **diagnostics)
        print(f"[residual] saved diagnostics: {sidecar}", flush=True)
        return path

    base.build_replay_plan = build_with_residual
    base.save_plan = save_with_diagnostics
    return base.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
