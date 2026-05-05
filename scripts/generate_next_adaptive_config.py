#!/usr/bin/env python3
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _number(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value, low, high):
    return max(low, min(high, value))


def _adjust_int(value, delta, low, high):
    return int(_clamp(int(value) + delta, low, high))


def _adjust_float(value, delta, low, high, places=4):
    return round(_clamp(float(value) + delta, low, high), places)


def _adjust_iterations(values, delta, low=1, high=25):
    return [_adjust_int(v, delta, low, high) for v in values]


def _latest_metric(metrics, *keys):
    for key in keys:
        value = _number(metrics.get(key))
        if value is not None:
            return value
    return None


def _quality_is_good_enough(metrics, pt_cfg):
    qj = pt_cfg.get("quality_judgment", {})
    accepted = _number(metrics.get("accepted_frame_ratio"), 0.0)
    segments = _integer(metrics.get("num_segments"), 99)
    drift = _number(metrics.get("head_tail_translation_drift_m"))
    runtime = str(metrics.get("runtime") or "").lower()
    overall = str(metrics.get("overall_run_quality") or "").lower()
    medium_drift = _number(qj.get("medium_drift_m"), 0.3)
    stable = accepted >= 0.95 and segments <= 1
    drift_ok = drift is None or drift <= medium_drift
    runtime_ok = runtime in {"", "fast", "moderate"}
    overall_ok = overall in {"good", "excellent"}
    return stable and drift_ok and runtime_ok and overall_ok


def _is_runtime_high(metrics, pt_cfg):
    qj = pt_cfg.get("quality_judgment", {})
    runtime_per_frame = _number(metrics.get("runtime_per_accepted_frame_sec"))
    moderate_limit = _number(qj.get("moderate_runtime_per_frame_sec"), 2.0)
    runtime_label = str(metrics.get("runtime") or "").lower()
    return runtime_label == "slow" or (
        runtime_per_frame is not None and runtime_per_frame > moderate_limit
    )


def _is_memory_high(metrics, pt_cfg):
    jetson = pt_cfg.get("jetson", {})
    warn_mb = _number(jetson.get("ram_warn_mb"), 4000.0)
    observed = [
        _number(metrics.get(key))
        for key in (
            "final_memory_mb",
            "after_global_lc_memory_mb",
            "after_odometry_memory_mb",
            "after_orb_memory_mb",
        )
    ]
    observed = [value for value in observed if value is not None]
    return bool(observed) and max(observed) >= warn_mb


def _tracking_is_weak(metrics, pt_cfg):
    qj = pt_cfg.get("quality_judgment", {})
    accepted = _number(metrics.get("accepted_frame_ratio"), 0.0)
    segments = _integer(metrics.get("num_segments"), 99)
    good_ratio = _number(qj.get("good_accepted_ratio"), 0.85)
    return accepted < good_ratio or segments > 1


def _loop_closure_is_weak(metrics):
    total_global = _integer(metrics.get("total_global_lc_added"), 0)
    orb_rate = _number(metrics.get("orb_glc_accept_rate"), 0.0)
    fpfh_rate = _number(metrics.get("fpfh_glc_accept_rate"), 0.0)
    local_rate = _number(metrics.get("local_lc_accept_rate"), 0.0)
    return (
        total_global < 20
        or local_rate < 0.75
        or (orb_rate < 0.6 and fpfh_rate < 0.25)
    )


def _increase_tracking_quality(pt_cfg):
    odom = pt_cfg["odometry"]
    odom["scale"] = _adjust_float(odom["scale"], 0.05, 0.2, 0.4)
    odom["fast_iterations"] = _adjust_iterations(odom["fast_iterations"], 1)
    odom["full_iterations"] = _adjust_iterations(odom["full_iterations"], 1)

    seq = pt_cfg["sequential_icp_fallback"]
    seq["enabled"] = True
    seq["fitness_min"] = _adjust_float(seq["fitness_min"], -0.03, 0.2, 0.5)
    seq["max_translation_m"] = _adjust_float(seq["max_translation_m"], 0.1, 0.5, 1.2)
    seq["max_rotation_rad"] = _adjust_float(seq["max_rotation_rad"], 0.1, 0.8, 1.6)


def _increase_loop_quality(pt_cfg):
    local = pt_cfg["local_loop_closure"]
    local["stride"] = _adjust_int(local["stride"], -1, 3, 12)
    gaps = sorted({int(v) for v in local.get("gaps", [])} | {3, 5})
    local["gaps"] = [v for v in gaps if 2 <= v <= 8][:3]

    orb = pt_cfg["orb_global_loop_closure"]
    orb["features_per_frame"] = _adjust_int(orb["features_per_frame"], 150, 300, 1000)
    orb["query_stride"] = _adjust_int(orb["query_stride"], -2, 6, 24)
    orb["max_candidates"] = _adjust_int(orb["max_candidates"], 1, 1, 4)
    orb["prescreen_top_k"] = _adjust_int(orb["prescreen_top_k"], 5, 10, 40)

    fpfh = pt_cfg["fpfh_global_loop_closure"]
    fpfh["enabled"] = True
    fpfh["voxel_size"] = _adjust_float(fpfh["voxel_size"], -0.01, 0.06, 0.14)
    fpfh["query_stride"] = _adjust_int(fpfh["query_stride"], -5, 15, 45)
    fpfh["max_candidates"] = _adjust_int(fpfh["max_candidates"], 1, 1, 3)
    fpfh["spatial_top_k"] = _adjust_int(fpfh["spatial_top_k"], 5, 10, 35)
    fpfh["ransac_max_iterations"] = _adjust_int(
        fpfh["ransac_max_iterations"], 500, 1500, 5000
    )


def _reduce_runtime_cost(pt_cfg):
    local = pt_cfg["local_loop_closure"]
    local["stride"] = _adjust_int(local["stride"], 1, 3, 12)
    if len(local.get("gaps", [])) > 1:
        local["gaps"] = [min(int(v) for v in local["gaps"])]

    orb = pt_cfg["orb_global_loop_closure"]
    orb["features_per_frame"] = _adjust_int(orb["features_per_frame"], -100, 300, 1000)
    orb["query_stride"] = _adjust_int(orb["query_stride"], 2, 6, 24)
    orb["max_candidates"] = _adjust_int(orb["max_candidates"], -1, 1, 4)
    orb["prescreen_top_k"] = _adjust_int(orb["prescreen_top_k"], -5, 10, 40)

    fpfh = pt_cfg["fpfh_global_loop_closure"]
    fpfh["query_stride"] = _adjust_int(fpfh["query_stride"], 5, 15, 45)
    fpfh["voxel_size"] = _adjust_float(fpfh["voxel_size"], 0.01, 0.06, 0.14)
    fpfh["max_candidates"] = _adjust_int(fpfh["max_candidates"], -1, 1, 3)
    fpfh["ransac_max_iterations"] = _adjust_int(
        fpfh["ransac_max_iterations"], -500, 1500, 5000
    )


def _reduce_memory_pressure(pt_cfg):
    caches = pt_cfg.get("caches", {})
    minimums = {
        "rgbd": 20,
        "pcd": 15,
        "pcd_level": 30,
        "pnp_depth": 8,
    }
    for key, minimum in minimums.items():
        if key in caches:
            caches[key] = max(minimum, int(int(caches[key]) * 0.8))


def generate_next_config(previous_config, metrics, adaptive_index, mode):
    if mode != "metric_conservative":
        raise ValueError(f"Unsupported adaptive mode: {mode}")

    next_config = copy.deepcopy(previous_config)
    pt_cfg = next_config["pose_tracking"]
    reasons = []
    tracking_weak = _tracking_is_weak(metrics, pt_cfg)
    loop_weak = _loop_closure_is_weak(metrics)
    good_enough = _quality_is_good_enough(metrics, pt_cfg)
    runtime_high = _is_runtime_high(metrics, pt_cfg)
    memory_high = _is_memory_high(metrics, pt_cfg)

    if tracking_weak:
        _increase_tracking_quality(pt_cfg)
        reasons.append("increased odometry and ICP fallback quality")

    if loop_weak:
        _increase_loop_quality(pt_cfg)
        reasons.append("increased local/ORB/FPFH loop closure coverage")

    if good_enough and runtime_high and not tracking_weak and not loop_weak:
        _reduce_runtime_cost(pt_cfg)
        reasons.append("reduced loop-closure cost after acceptable quality")

    if memory_high:
        _reduce_memory_pressure(pt_cfg)
        reasons.append("reduced cache sizes after high memory use")

    if not reasons:
        reasons.append("kept parameters stable; previous run met conservative thresholds")

    next_config["adaptive_metadata"] = {
        "adaptive_index": adaptive_index,
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "previous_metrics": {
            "accepted_frame_ratio": metrics.get("accepted_frame_ratio"),
            "num_segments": metrics.get("num_segments"),
            "head_tail_translation_drift_m": metrics.get(
                "head_tail_translation_drift_m"
            ),
            "runtime_per_accepted_frame_sec": metrics.get(
                "runtime_per_accepted_frame_sec"
            ),
            "final_memory_mb": metrics.get("final_memory_mb"),
            "overall_run_quality": metrics.get("overall_run_quality"),
        },
        "reasons": reasons,
    }
    return next_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate the next conservative adaptive pose-tracking config."
    )
    parser.add_argument("--previous-config", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--adaptive-index", type=int, required=True)
    parser.add_argument("--mode", default="metric_conservative")
    args = parser.parse_args()

    previous_config = _load_json(args.previous_config)
    metrics = _load_json(args.metrics)
    if "pose_tracking" not in previous_config:
        raise ValueError("previous config is missing pose_tracking")

    next_config = generate_next_config(
        previous_config, metrics, args.adaptive_index, args.mode
    )
    _write_json(args.output_config, next_config)
    print(f"Wrote adaptive config: {args.output_config}")


if __name__ == "__main__":
    main()
