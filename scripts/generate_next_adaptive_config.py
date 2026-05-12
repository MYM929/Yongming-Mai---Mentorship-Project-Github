#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CONSERVATIVE_MODE = "metric_conservative"
BAYES_MODE = "bayes_opt"
BAYES_FAILED_SCORE = -1000.0
BAYES_MIN_TRIALS = 12
BAYES_PATIENCE = 8
BAYES_MIN_DELTA = 0.002
BAYES_DUPLICATE_RETRY_LIMIT = 32
BAYES_GAP_CHOICES = ("3", "3_5", "3_5_7")


def _require_optuna():
    try:
        import optuna
        from optuna.distributions import (
            CategoricalDistribution,
            FloatDistribution,
            IntDistribution,
        )
    except ImportError as exc:
        raise ImportError(
            "ADAPTIVE_MODE=bayes_opt requires optuna. "
            "Install dependencies from requirements.txt."
        ) from exc
    return optuna, CategoricalDistribution, FloatDistribution, IntDistribution


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


def _gap_choice_from_config(local_cfg):
    values = sorted({int(v) for v in local_cfg.get("gaps", [])})
    if values == [3, 5, 7]:
        return "3_5_7"
    if values == [3, 5]:
        return "3_5"
    return "3"


def _gaps_from_choice(choice):
    if choice == "3_5_7":
        return [3, 5, 7]
    if choice == "3_5":
        return [3, 5]
    return [3]


def _bayes_distributions():
    _, CategoricalDistribution, FloatDistribution, IntDistribution = _require_optuna()
    return {
        "odometry_scale": FloatDistribution(0.2, 0.4, step=0.01),
        "fast_iter_0": IntDistribution(4, 12),
        "fast_iter_1": IntDistribution(3, 10),
        "fast_iter_2": IntDistribution(2, 8),
        "full_iter_0": IntDistribution(6, 16),
        "full_iter_1": IntDistribution(4, 12),
        "full_iter_2": IntDistribution(2, 8),
        "seq_fitness_min": FloatDistribution(0.2, 0.5, step=0.01),
        "seq_max_translation_m": FloatDistribution(0.5, 1.2, step=0.05),
        "seq_max_rotation_rad": FloatDistribution(0.8, 1.6, step=0.05),
        "local_stride": IntDistribution(3, 12),
        "local_gap_choice": CategoricalDistribution(BAYES_GAP_CHOICES),
        "orb_features_per_frame": IntDistribution(300, 1000, step=50),
        "orb_query_stride": IntDistribution(6, 24),
        "orb_max_candidates": IntDistribution(1, 4),
        "orb_prescreen_top_k": IntDistribution(10, 40),
        "fpfh_voxel_size": FloatDistribution(0.06, 0.14, step=0.01),
        "fpfh_query_stride": IntDistribution(15, 45),
        "fpfh_max_candidates": IntDistribution(1, 3),
        "fpfh_spatial_top_k": IntDistribution(10, 35),
        "fpfh_ransac_max_iterations": IntDistribution(1500, 5000, step=500),
    }


def extract_bayes_params_from_config(config):
    pt_cfg = config["pose_tracking"] if "pose_tracking" in config else config
    odom = pt_cfg["odometry"]
    seq = pt_cfg["sequential_icp_fallback"]
    local = pt_cfg["local_loop_closure"]
    orb = pt_cfg["orb_global_loop_closure"]
    fpfh = pt_cfg["fpfh_global_loop_closure"]
    fast_iterations = list(odom["fast_iterations"])
    full_iterations = list(odom["full_iterations"])
    return {
        "odometry_scale": round(float(odom["scale"]), 2),
        "fast_iter_0": int(fast_iterations[0]),
        "fast_iter_1": int(fast_iterations[1]),
        "fast_iter_2": int(fast_iterations[2]),
        "full_iter_0": int(full_iterations[0]),
        "full_iter_1": int(full_iterations[1]),
        "full_iter_2": int(full_iterations[2]),
        "seq_fitness_min": round(float(seq["fitness_min"]), 2),
        "seq_max_translation_m": round(float(seq["max_translation_m"]) / 0.05) * 0.05,
        "seq_max_rotation_rad": round(float(seq["max_rotation_rad"]) / 0.05) * 0.05,
        "local_stride": int(local["stride"]),
        "local_gap_choice": _gap_choice_from_config(local),
        "orb_features_per_frame": int(round(int(orb["features_per_frame"]) / 50) * 50),
        "orb_query_stride": int(orb["query_stride"]),
        "orb_max_candidates": int(orb["max_candidates"]),
        "orb_prescreen_top_k": int(orb["prescreen_top_k"]),
        "fpfh_voxel_size": round(float(fpfh["voxel_size"]), 2),
        "fpfh_query_stride": int(fpfh["query_stride"]),
        "fpfh_max_candidates": int(fpfh["max_candidates"]),
        "fpfh_spatial_top_k": int(fpfh["spatial_top_k"]),
        "fpfh_ransac_max_iterations": int(
            round(int(fpfh["ransac_max_iterations"]) / 500) * 500
        ),
    }


def _normalize_bayes_params(params):
    distributions = _bayes_distributions()
    normalized = {}
    for key, distribution in distributions.items():
        value = params[key]
        low = getattr(distribution, "low", None)
        high = getattr(distribution, "high", None)
        if low is not None and high is not None:
            value = _clamp(value, low, high)
        if distribution.__class__.__name__ == "IntDistribution":
            value = int(round(value))
            step = getattr(distribution, "step", 1)
            if step and step > 1:
                value = int(round((value - low) / step) * step + low)
                value = int(_clamp(value, low, high))
        elif distribution.__class__.__name__ == "FloatDistribution":
            value = float(value)
            step = getattr(distribution, "step", None)
            if step:
                value = round(round((value - low) / step) * step + low, 6)
                value = _clamp(value, low, high)
        normalized[key] = value
    return normalized


def apply_bayes_params_to_config(config, params):
    next_config = copy.deepcopy(config)
    params = _normalize_bayes_params(params)
    pt_cfg = next_config["pose_tracking"]
    odom = pt_cfg["odometry"]
    odom["scale"] = round(float(params["odometry_scale"]), 2)
    odom["fast_iterations"] = [
        int(params["fast_iter_0"]),
        int(params["fast_iter_1"]),
        int(params["fast_iter_2"]),
    ]
    odom["full_iterations"] = [
        int(params["full_iter_0"]),
        int(params["full_iter_1"]),
        int(params["full_iter_2"]),
    ]

    seq = pt_cfg["sequential_icp_fallback"]
    seq["enabled"] = True
    seq["fitness_min"] = round(float(params["seq_fitness_min"]), 2)
    seq["max_translation_m"] = round(float(params["seq_max_translation_m"]), 2)
    seq["max_rotation_rad"] = round(float(params["seq_max_rotation_rad"]), 2)

    local = pt_cfg["local_loop_closure"]
    local["stride"] = int(params["local_stride"])
    local["gaps"] = _gaps_from_choice(params["local_gap_choice"])

    orb = pt_cfg["orb_global_loop_closure"]
    orb["features_per_frame"] = int(params["orb_features_per_frame"])
    orb["query_stride"] = int(params["orb_query_stride"])
    orb["max_candidates"] = int(params["orb_max_candidates"])
    orb["prescreen_top_k"] = int(params["orb_prescreen_top_k"])

    fpfh = pt_cfg["fpfh_global_loop_closure"]
    fpfh["enabled"] = True
    fpfh["voxel_size"] = round(float(params["fpfh_voxel_size"]), 2)
    fpfh["query_stride"] = int(params["fpfh_query_stride"])
    fpfh["max_candidates"] = int(params["fpfh_max_candidates"])
    fpfh["spatial_top_k"] = int(params["fpfh_spatial_top_k"])
    fpfh["ransac_max_iterations"] = int(params["fpfh_ransac_max_iterations"])

    return next_config


def evaluate_bayes_objective(metrics, pt_cfg):
    if not metrics:
        return {
            "objective_score": BAYES_FAILED_SCORE,
            "constraint_status": "failed_or_missing_metrics",
            "quality_score": 0.0,
            "speed_score": 0.0,
        }

    qj = pt_cfg.get("quality_judgment", {})
    jetson = pt_cfg.get("jetson", {})
    accepted = _number(metrics.get("accepted_frame_ratio"), 0.0)
    segments = _integer(metrics.get("num_segments"), 99)
    drift = _number(metrics.get("head_tail_translation_drift_m"))
    runtime = _number(metrics.get("runtime_per_accepted_frame_sec"))
    memory = max(
        [
            value
            for value in (
                _number(metrics.get("final_memory_mb")),
                _number(metrics.get("after_global_lc_memory_mb")),
                _number(metrics.get("after_odometry_memory_mb")),
                _number(metrics.get("after_orb_memory_mb")),
            )
            if value is not None
        ]
        or [0.0]
    )
    global_lc = _number(metrics.get("total_global_lc_added"), 0.0)
    local_rate = _number(metrics.get("local_lc_accept_rate"), 0.0)
    orb_rate = _number(metrics.get("orb_glc_accept_rate"), 0.0)
    fpfh_rate = _number(metrics.get("fpfh_glc_accept_rate"), 0.0)

    good_ratio = _number(qj.get("good_accepted_ratio"), 0.85)
    low_drift = _number(qj.get("low_drift_m"), 0.1)
    medium_drift = _number(qj.get("medium_drift_m"), 0.3)
    moderate_runtime = _number(qj.get("moderate_runtime_per_frame_sec"), 2.0)
    memory_warn = _number(jetson.get("ram_warn_mb"), 4000.0)

    constraint_violations = []
    if accepted < good_ratio:
        constraint_violations.append("weak_tracking")
    if segments > 1:
        constraint_violations.append("multiple_segments")
    if drift is not None and drift > medium_drift:
        constraint_violations.append("high_drift")
    if memory >= memory_warn:
        constraint_violations.append("high_memory")

    segment_score = 1.0 if segments <= 1 else max(0.0, 1.0 - 0.25 * (segments - 1))
    if drift is None:
        drift_score = 1.0
    elif drift <= low_drift:
        drift_score = 1.0
    elif drift <= medium_drift:
        drift_score = 1.0 - ((drift - low_drift) / (medium_drift - low_drift)) * 0.5
    else:
        drift_score = max(0.0, 0.5 - min((drift - medium_drift) / medium_drift, 1.0) * 0.5)
    loop_score = min(
        1.0,
        (global_lc / 20.0) * 0.4
        + local_rate * 0.25
        + orb_rate * 0.2
        + fpfh_rate * 0.15,
    )
    quality_score = (
        accepted * 0.4
        + segment_score * 0.25
        + drift_score * 0.25
        + loop_score * 0.1
    )

    if runtime is None or runtime <= 0:
        speed_score = 0.0
    elif runtime <= 0.5:
        speed_score = 1.0
    elif runtime >= moderate_runtime:
        speed_score = max(0.0, moderate_runtime / runtime * 0.5)
    else:
        speed_score = 1.0 - ((runtime - 0.5) / (moderate_runtime - 0.5)) * 0.4

    score = quality_score * 0.85 + speed_score * 0.15
    if constraint_violations:
        score = -100.0 + quality_score * 10.0 + speed_score
    return {
        "objective_score": round(score, 4),
        "constraint_status": (
            "ok" if not constraint_violations else ",".join(constraint_violations)
        ),
        "quality_score": round(quality_score, 4),
        "speed_score": round(speed_score, 4),
    }


def should_stop_bayes_early(trial_records, min_trials=BAYES_MIN_TRIALS,
                            patience=BAYES_PATIENCE, min_delta=BAYES_MIN_DELTA):
    completed = [
        record for record in trial_records
        if _number(record.get("objective_score")) is not None
    ]
    if len(completed) < min_trials:
        return {
            "should_stop": False,
            "reason": f"need at least {min_trials} completed trials",
            "trials_used": len(completed),
            "best_score": None,
            "best_sequence_number": None,
        }

    best_score = None
    best_sequence = None
    improvements = []
    for record in completed:
        score = _number(record.get("objective_score"), BAYES_FAILED_SCORE)
        previous_best = best_score
        if best_score is None or score > best_score:
            best_score = score
            best_sequence = record.get("sequence_number")
        if previous_best is None:
            improvements.append(float("inf"))
        else:
            improvements.append(max(0.0, best_score - previous_best))

    recent = improvements[-patience:]
    plateaued = len(recent) == patience and all(value < min_delta for value in recent)
    if plateaued:
        reason = (
            f"best score improved by less than {min_delta} for "
            f"{patience} consecutive trials"
        )
    else:
        reason = "improvement still above early-stop threshold"
    return {
        "should_stop": plateaued,
        "reason": reason,
        "trials_used": len(completed),
        "best_score": round(best_score, 4) if best_score is not None else None,
        "best_sequence_number": best_sequence,
    }


def _seed_from_parts(*parts):
    raw = "::".join(str(part or "") for part in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def _suggest_bayes_params(trial):
    return {
        "odometry_scale": trial.suggest_float("odometry_scale", 0.2, 0.4, step=0.01),
        "fast_iter_0": trial.suggest_int("fast_iter_0", 4, 12),
        "fast_iter_1": trial.suggest_int("fast_iter_1", 3, 10),
        "fast_iter_2": trial.suggest_int("fast_iter_2", 2, 8),
        "full_iter_0": trial.suggest_int("full_iter_0", 6, 16),
        "full_iter_1": trial.suggest_int("full_iter_1", 4, 12),
        "full_iter_2": trial.suggest_int("full_iter_2", 2, 8),
        "seq_fitness_min": trial.suggest_float("seq_fitness_min", 0.2, 0.5, step=0.01),
        "seq_max_translation_m": trial.suggest_float(
            "seq_max_translation_m", 0.5, 1.2, step=0.05
        ),
        "seq_max_rotation_rad": trial.suggest_float(
            "seq_max_rotation_rad", 0.8, 1.6, step=0.05
        ),
        "local_stride": trial.suggest_int("local_stride", 3, 12),
        "local_gap_choice": trial.suggest_categorical(
            "local_gap_choice", list(BAYES_GAP_CHOICES)
        ),
        "orb_features_per_frame": trial.suggest_int(
            "orb_features_per_frame", 300, 1000, step=50
        ),
        "orb_query_stride": trial.suggest_int("orb_query_stride", 6, 24),
        "orb_max_candidates": trial.suggest_int("orb_max_candidates", 1, 4),
        "orb_prescreen_top_k": trial.suggest_int("orb_prescreen_top_k", 10, 40),
        "fpfh_voxel_size": trial.suggest_float("fpfh_voxel_size", 0.06, 0.14, step=0.01),
        "fpfh_query_stride": trial.suggest_int("fpfh_query_stride", 15, 45),
        "fpfh_max_candidates": trial.suggest_int("fpfh_max_candidates", 1, 3),
        "fpfh_spatial_top_k": trial.suggest_int("fpfh_spatial_top_k", 10, 35),
        "fpfh_ransac_max_iterations": trial.suggest_int(
            "fpfh_ransac_max_iterations", 1500, 5000, step=500
        ),
    }


def _bayes_param_key(params):
    return json.dumps(
        _normalize_bayes_params(params),
        sort_keys=True,
        separators=(",", ":"),
    )


def _suggest_unique_bayes_params(study, tried_param_keys):
    fallback = None
    duplicate_attempts = 0
    for _ in range(BAYES_DUPLICATE_RETRY_LIMIT):
        trial = study.ask()
        params = _suggest_bayes_params(trial)
        key = _bayes_param_key(params)
        fallback = params
        if key not in tried_param_keys:
            return params, duplicate_attempts
        duplicate_attempts += 1
        study.tell(trial, BAYES_FAILED_SCORE)
    return fallback, duplicate_attempts


def _load_trial_history(path):
    if not path:
        return []
    history_path = Path(path)
    if not history_path.exists():
        return []
    value = _load_json(history_path)
    if isinstance(value, dict):
        return value.get("trials", [])
    return value if isinstance(value, list) else []


def _generate_bayes_config(previous_config, metrics, adaptive_index,
                           trial_history, dataset_name, batch_run_id):
    optuna, _, _, _ = _require_optuna()
    distributions = _bayes_distributions()
    seed = _seed_from_parts(batch_run_id, dataset_name, adaptive_index)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=8)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    trials = list(trial_history or [])
    if not trials:
        objective = evaluate_bayes_objective(metrics, previous_config["pose_tracking"])
        trials.append({
            "sequence_number": adaptive_index,
            "config_name": Path(str(previous_config.get("config_path", "base"))).name,
            "params": extract_bayes_params_from_config(previous_config),
            "objective_score": objective["objective_score"],
            "constraint_status": objective["constraint_status"],
            "quality_score": objective["quality_score"],
            "speed_score": objective["speed_score"],
        })

    tried_param_keys = set()
    for record in trials:
        params = record.get("params")
        value = _number(record.get("objective_score"))
        if not params or value is None:
            continue
        try:
            normalized_params = _normalize_bayes_params(params)
            trial = optuna.trial.create_trial(
                params=normalized_params,
                distributions=distributions,
                value=value,
            )
            study.add_trial(trial)
            tried_param_keys.add(_bayes_param_key(normalized_params))
        except ValueError:
            continue

    params, duplicate_attempts = _suggest_unique_bayes_params(study, tried_param_keys)
    next_config = apply_bayes_params_to_config(previous_config, params)
    next_config["adaptive_metadata"] = {
        "adaptive_index": adaptive_index,
        "mode": BAYES_MODE,
        "optimizer": "optuna_tpe",
        "objective": "quality_first",
        "trial_number": len(trials) + 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "history_trials": len(trials),
        "suggested_params": params,
        "duplicate_suggestion_retries": duplicate_attempts,
        "previous_best": should_stop_bayes_early(trials),
        "reasons": ["suggested next configuration with Optuna TPE"],
    }
    return next_config


def generate_next_config(previous_config, metrics, adaptive_index, mode,
                         trial_history=None, dataset_name=None, batch_run_id=None):
    if mode == BAYES_MODE:
        return _generate_bayes_config(
            previous_config,
            metrics,
            adaptive_index,
            trial_history,
            dataset_name,
            batch_run_id,
        )
    if mode != CONSERVATIVE_MODE:
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
    parser.add_argument("--trial-history")
    parser.add_argument("--dataset-name")
    parser.add_argument("--batch-run-id")
    args = parser.parse_args()

    previous_config = _load_json(args.previous_config)
    metrics = _load_json(args.metrics)
    if "pose_tracking" not in previous_config:
        raise ValueError("previous config is missing pose_tracking")

    trial_history = _load_trial_history(args.trial_history)
    next_config = generate_next_config(
        previous_config,
        metrics,
        args.adaptive_index,
        args.mode,
        trial_history=trial_history,
        dataset_name=args.dataset_name,
        batch_run_id=args.batch_run_id,
    )
    _write_json(args.output_config, next_config)
    print(f"Wrote adaptive config: {args.output_config}")


if __name__ == "__main__":
    main()
