import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import boto3

from scripts.generate_next_adaptive_config import (
    BAYES_FAILED_SCORE,
    BAYES_MODE,
    evaluate_bayes_objective,
    extract_bayes_params_from_config,
    should_stop_bayes_early,
)


APP_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = APP_DIR / "configs"
DATASETS_DIR = APP_DIR / "datasets"
EXPERIMENTS_DIR = APP_DIR / "experiments"
DEFAULT_CONFIG_NAME = "dataset_1_bedroom.json"
ADAPTIVE_CONFIG_SCRIPT = APP_DIR / "scripts" / "generate_next_adaptive_config.py"


@dataclass(frozen=True)
class S3Uri:
    bucket: str
    key: str


def _get_required_env(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _parse_s3_uri(value):
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an s3:// URI, got: {value}")
    return S3Uri(bucket=parsed.netloc, key=parsed.path.lstrip("/").rstrip("/"))


def _join_s3_key(*parts):
    clean_parts = [str(part).strip("/") for part in parts if str(part).strip("/")]
    return posixpath.join(*clean_parts) if clean_parts else ""


def _sanitize_token(value, name):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip(".-")
    if not cleaned:
        raise ValueError(f"Invalid {name}: {value}")
    return cleaned


def _get_experiment_count(default_value="1"):
    value = os.environ.get("EXPERIMENT_COUNT", default_value)
    try:
        count = int(value)
    except ValueError as exc:
        raise ValueError(f"EXPERIMENT_COUNT must be an integer, got: {value}") from exc
    if count < 1:
        raise ValueError("EXPERIMENT_COUNT must be at least 1")
    return count


def _default_batch_run_id():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = os.environ.get("AWS_BATCH_JOB_ID") or os.environ.get("HOSTNAME") or "local"
    return _sanitize_token(f"{stamp}-{job_id}", "BATCH_RUN_ID")


def _safe_child_path(root, relative_key):
    target = (root / relative_key).resolve()
    root_resolved = root.resolve()
    if root_resolved != target and root_resolved not in target.parents:
        raise ValueError(f"Refusing to write outside {root}: {relative_key}")
    return target


def _download_object(s3, source_uri, dest_path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading s3://{source_uri.bucket}/{source_uri.key} -> {dest_path}")
    s3.download_file(source_uri.bucket, source_uri.key, str(dest_path))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _download_prefix(s3, source_uri, dest_dir):
    prefix = source_uri.key.rstrip("/")
    if prefix:
        prefix = f"{prefix}/"

    print(f"Syncing s3://{source_uri.bucket}/{prefix} -> {dest_dir}")
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0

    for page in paginator.paginate(Bucket=source_uri.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            relative_key = key[len(prefix):] if prefix else key
            if not relative_key:
                continue
            dest_path = _safe_child_path(dest_dir, relative_key)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(source_uri.bucket, key, str(dest_path))
            downloaded += 1

    if downloaded == 0:
        raise FileNotFoundError(
            f"No S3 objects found under s3://{source_uri.bucket}/{prefix}"
        )

    print(f"Downloaded {downloaded} object(s).")


def _upload_prefix(s3, source_dir, target_uri):
    if not source_dir.exists():
        raise FileNotFoundError(f"Experiment output directory not found: {source_dir}")

    uploaded = 0
    target_prefix = target_uri.key.rstrip("/")
    print(f"Uploading {source_dir} -> s3://{target_uri.bucket}/{target_prefix}/")

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source_dir).as_posix()
        target_key = _join_s3_key(target_prefix, relative_path)
        s3.upload_file(str(path), target_uri.bucket, target_key)
        uploaded += 1

    if uploaded == 0:
        print(f"No experiment files found to upload in {source_dir}.")
    else:
        print(f"Uploaded {uploaded} object(s).")


def _upload_object(s3, source_path, target_uri):
    if not source_path.exists():
        raise FileNotFoundError(f"Upload source file not found: {source_path}")
    print(f"Uploading {source_path} -> s3://{target_uri.bucket}/{target_uri.key}")
    s3.upload_file(str(source_path), target_uri.bucket, target_uri.key)


def _load_dataset_name(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    dataset_name = config.get("dataset_name") or config.get("active_dataset")
    if not dataset_name:
        raise ValueError("dataset_name missing from config")
    if "/" in dataset_name or "\\" in dataset_name or dataset_name in {".", ".."}:
        raise ValueError(f"Invalid dataset_name in config: {dataset_name}")
    return dataset_name


def _run_dirs(dataset_name):
    dataset_dir = EXPERIMENTS_DIR / dataset_name
    if not dataset_dir.exists():
        return []
    run_dirs = []
    for path in dataset_dir.iterdir():
        match = re.fullmatch(r"run(\d+)", path.name)
        if path.is_dir() and match:
            run_dirs.append((int(match.group(1)), path))
    return [path for _, path in sorted(run_dirs)]


def _latest_metrics_json(run_dir):
    metric_paths = sorted(
        run_dir.glob("metrics_*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not metric_paths:
        raise FileNotFoundError(f"No metrics_*.json file found in {run_dir}")
    return metric_paths[-1]


def _run_pose_tracking(config_path):
    command = [
        sys.executable,
        "1.pose_tracking.py",
        "--config",
        str(config_path),
    ]
    result = subprocess.run(command, cwd=str(APP_DIR), check=False)
    return result.returncode


def _prepare_sequence_run_dir(dataset_name, sequence_number, local_run_dir):
    target_dir = EXPERIMENTS_DIR / dataset_name / f"batch_sequence_run{sequence_number}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(local_run_dir, target_dir)
    return target_dir


def _upload_sequence_run(s3, outputs_uri, dataset_name, batch_run_id,
                         sequence_number, local_run_dir):
    sequence_dir = _prepare_sequence_run_dir(
        dataset_name, sequence_number, local_run_dir
    )
    target = S3Uri(
        bucket=outputs_uri.bucket,
        key=_join_s3_key(
            outputs_uri.key,
            dataset_name,
            batch_run_id,
            f"run{sequence_number}",
        ),
    )
    _upload_prefix(s3, sequence_dir, target)


def _number(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_run(metrics):
    accepted = _number(metrics.get("accepted_frame_ratio"), 0.0)
    segments = _number(metrics.get("num_segments"), 99.0)
    drift = _number(metrics.get("head_tail_translation_drift_m"))
    runtime = _number(metrics.get("runtime_per_accepted_frame_sec"))
    global_lc = _number(metrics.get("total_global_lc_added"), 0.0)
    local_rate = _number(metrics.get("local_lc_accept_rate"), 0.0)
    orb_rate = _number(metrics.get("orb_glc_accept_rate"), 0.0)
    fpfh_rate = _number(metrics.get("fpfh_glc_accept_rate"), 0.0)

    segment_score = 1.0 if segments <= 1 else max(0.0, 1.0 - 0.15 * (segments - 1))
    drift_score = 1.0 if drift is None else max(0.0, 1.0 - min(drift, 1.0))
    loop_score = min(1.0, (global_lc / 20.0) * 0.45
                     + local_rate * 0.25 + orb_rate * 0.2 + fpfh_rate * 0.1)
    quality_score = (
        accepted * 0.45
        + segment_score * 0.2
        + drift_score * 0.2
        + loop_score * 0.15
    )

    if runtime is None or runtime <= 0:
        speed_score = 0.0
    elif runtime <= 0.5:
        speed_score = 1.0
    elif runtime >= 2.0:
        speed_score = max(0.0, 2.0 / runtime * 0.5)
    else:
        speed_score = 1.0 - ((runtime - 0.5) / 1.5) * 0.4

    balanced_score = quality_score * 0.7 + speed_score * 0.3
    return {
        "quality_score": round(quality_score, 4),
        "speed_score": round(speed_score, 4),
        "balanced_score": round(balanced_score, 4),
    }


def _report_value(value, digits=3):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _failed_scores():
    return {
        "quality_score": 0.0,
        "speed_score": 0.0,
        "balanced_score": BAYES_FAILED_SCORE,
        "objective_score": BAYES_FAILED_SCORE,
        "constraint_status": "failed_or_missing_metrics",
    }


def _write_batch_report(dataset_name, batch_run_id, completed_runs,
                        adaptive_mode="metric_conservative",
                        early_stop_info=None):
    report_dir = EXPERIMENTS_DIR / dataset_name / f"batch_report_{batch_run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)

    ranking_key = (
        lambda item: item["scores"].get("objective_score", BAYES_FAILED_SCORE)
        if adaptive_mode == BAYES_MODE
        else item["scores"]["balanced_score"]
    )
    ranked_runs = sorted(
        completed_runs,
        key=ranking_key,
        reverse=True,
    )
    best = ranked_runs[0] if ranked_runs else None
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    early_stop_info = early_stop_info or {
        "should_stop": False,
        "reason": "not evaluated",
        "trials_used": len(completed_runs),
        "best_score": None,
        "best_sequence_number": None,
    }
    summary = {
        "dataset_name": dataset_name,
        "batch_run_id": batch_run_id,
        "adaptive_mode": adaptive_mode,
        "generated_at": generated_at,
        "actual_experiment_count": len(completed_runs),
        "early_stopping": early_stop_info,
        "scoring": {
            "balanced_score": "quality_score * 0.7 + speed_score * 0.3",
            "quality_score": (
                "accepted frame ratio, segment count, head-tail drift, "
                "and loop-closure strength"
            ),
            "speed_score": "runtime per accepted frame; fast is <= 0.5 sec",
            "objective_score": (
                "quality-first Bayesian objective with hard penalties for "
                "weak tracking, multiple segments, high drift, or high memory"
            ),
        },
        "best_run": best,
        "runs": ranked_runs,
    }
    _write_json(report_dir / "adaptive_batch_report.json", summary)

    csv_fields = [
        "rank",
        "sequence_number",
        "config_name",
        "balanced_score",
        "objective_score",
        "quality_score",
        "speed_score",
        "constraint_status",
        "failed",
        "accepted_frame_ratio",
        "num_segments",
        "head_tail_translation_drift_m",
        "runtime_per_accepted_frame_sec",
        "total_wall_time_sec",
        "overall_run_quality",
    ]
    csv_path = report_dir / "adaptive_batch_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(csv_fields) + "\n")
        for rank, run in enumerate(ranked_runs, start=1):
            metrics = run.get("metrics") or {}
            scores = run["scores"]
            row = {
                "rank": rank,
                "sequence_number": run["sequence_number"],
                "config_name": run["config_name"],
                "balanced_score": scores.get("balanced_score"),
                "objective_score": scores.get("objective_score"),
                "quality_score": scores.get("quality_score"),
                "speed_score": scores.get("speed_score"),
                "constraint_status": scores.get("constraint_status"),
                "failed": run.get("failed", False),
                "accepted_frame_ratio": metrics.get("accepted_frame_ratio"),
                "num_segments": metrics.get("num_segments"),
                "head_tail_translation_drift_m": metrics.get(
                    "head_tail_translation_drift_m"
                ),
                "runtime_per_accepted_frame_sec": metrics.get(
                    "runtime_per_accepted_frame_sec"
                ),
                "total_wall_time_sec": metrics.get("total_wall_time_sec"),
                "overall_run_quality": metrics.get("overall_run_quality"),
            }
            handle.write(",".join(str(row.get(field, "")) for field in csv_fields) + "\n")

    md_lines = [
        "# Adaptive Batch Report",
        "",
        f"- Dataset: {dataset_name}",
        f"- Batch run id: {batch_run_id}",
        f"- Generated at: {generated_at}",
    ]
    if best:
        md_lines.extend([
            f"- Best balanced config: {best['config_name']}",
            f"- Best run: run{best['sequence_number']}",
            f"- Balanced score: {best['scores']['balanced_score']}",
        ])
    md_lines.extend([
        "",
        f"- Adaptive mode: {adaptive_mode}",
        "Scoring balances quality and speed: 70% quality, 30% speed.",
        f"- Actual experiments used: {len(completed_runs)}",
        f"- Early stopping: {early_stop_info.get('reason')}",
        "",
        "| Rank | Run | Config | Objective | Balanced | Quality | Speed | Constraints | Accepted | Segments | Drift m | Sec/frame | Overall |",
        "|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ])
    for rank, run in enumerate(ranked_runs, start=1):
        metrics = run.get("metrics") or {}
        scores = run["scores"]
        md_lines.append(
            "| "
            f"{rank} | "
            f"{run['sequence_number']} | "
            f"{run['config_name']} | "
            f"{_report_value(scores.get('objective_score'))} | "
            f"{_report_value(scores.get('balanced_score'))} | "
            f"{_report_value(scores.get('quality_score'))} | "
            f"{_report_value(scores.get('speed_score'))} | "
            f"{scores.get('constraint_status', 'N/A')} | "
            f"{_report_value(metrics.get('accepted_frame_ratio'))} | "
            f"{_report_value(metrics.get('num_segments'), 0)} | "
            f"{_report_value(metrics.get('head_tail_translation_drift_m'))} | "
            f"{_report_value(metrics.get('runtime_per_accepted_frame_sec'))} | "
            f"{metrics.get('overall_run_quality', 'N/A')} |"
        )
    md_lines.append("")
    (report_dir / "adaptive_batch_report.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    return report_dir


def _upload_batch_report(s3, outputs_uri, dataset_name, batch_run_id, report_dir):
    target = S3Uri(
        bucket=outputs_uri.bucket,
        key=_join_s3_key(outputs_uri.key, dataset_name, batch_run_id, "report"),
    )
    _upload_prefix(s3, report_dir, target)


def _generate_adaptive_config(previous_config_path, metrics_path, dataset_name,
                              adaptive_index, mode, batch_run_id,
                              trial_history_path=None):
    if not ADAPTIVE_CONFIG_SCRIPT.exists():
        raise FileNotFoundError(
            f"Adaptive config generator not found: {ADAPTIVE_CONFIG_SCRIPT}"
        )

    config_name = f"{dataset_name}_adaptive_{adaptive_index:02d}.json"
    output_path = CONFIGS_DIR / "adaptive" / batch_run_id / config_name
    command = [
        sys.executable,
        str(ADAPTIVE_CONFIG_SCRIPT),
        "--previous-config",
        str(previous_config_path),
        "--metrics",
        str(metrics_path),
        "--output-config",
        str(output_path),
        "--adaptive-index",
        str(adaptive_index),
        "--mode",
        mode,
        "--dataset-name",
        dataset_name,
        "--batch-run-id",
        batch_run_id,
    ]
    if trial_history_path is not None:
        command.extend(["--trial-history", str(trial_history_path)])
    result = subprocess.run(command, cwd=str(APP_DIR), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return output_path, config_name


def _run_adaptive_sequence(s3, configs_uri, outputs_uri, dataset_name,
                           initial_config_path, experiment_count,
                           batch_run_id, adaptive_mode):
    current_config_path = initial_config_path
    completed_runs = 0
    completed_run_records = []
    early_stop_info = None
    trial_history_path = (
        CONFIGS_DIR / "adaptive" / batch_run_id / f"{dataset_name}_bayes_trials.json"
        if adaptive_mode == BAYES_MODE
        else None
    )

    for sequence_number in range(1, experiment_count + 1):
        current_config_name = current_config_path.name
        with current_config_path.open("r", encoding="utf-8") as handle:
            current_config = json.load(handle)
        print(
            f"Running experiment {sequence_number}/{experiment_count} "
            f"for dataset '{dataset_name}' with {current_config_name}..."
        )
        before_runs = set(_run_dirs(dataset_name))
        returncode = _run_pose_tracking(current_config_path)
        after_runs = _run_dirs(dataset_name)
        new_runs = [path for path in after_runs if path not in before_runs]
        local_run_dir = new_runs[-1] if new_runs else (after_runs[-1] if after_runs else None)
        metrics_path = None
        metrics = None
        scores = None
        failed = returncode != 0

        if local_run_dir is not None:
            try:
                metrics_path = _latest_metrics_json(local_run_dir)
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
                scores = _score_run(metrics)
            except FileNotFoundError as exc:
                print(f"No metrics report found for run{sequence_number}: {exc}")
                scores = _failed_scores()
                failed = True
            if adaptive_mode == BAYES_MODE:
                bayes_scores = evaluate_bayes_objective(
                    metrics,
                    current_config["pose_tracking"],
                )
                if scores is None:
                    scores = _failed_scores()
                scores.update(bayes_scores)
                failed = failed or metrics is None
            shutil.copy2(current_config_path, local_run_dir / "used_config.json")
            _write_json(local_run_dir / "run_manifest.json", {
                "dataset_name": dataset_name,
                "batch_run_id": batch_run_id,
                "sequence_number": sequence_number,
                "config_name": current_config_name,
                "config_path": str(current_config_path),
                "metrics_file": metrics_path.name if metrics_path else None,
                "scores": scores,
                "adaptive_metadata": current_config.get("adaptive_metadata"),
                "failed": failed,
            })
            if metrics is not None and scores is not None:
                completed_run_records.append({
                    "sequence_number": sequence_number,
                    "config_name": current_config_name,
                    "local_run_dir": str(local_run_dir),
                    "metrics_file": metrics_path.name,
                    "metrics": metrics,
                    "scores": scores,
                    "adaptive_metadata": current_config.get("adaptive_metadata"),
                    "failed": failed,
                })
            elif adaptive_mode == BAYES_MODE:
                completed_run_records.append({
                    "sequence_number": sequence_number,
                    "config_name": current_config_name,
                    "local_run_dir": str(local_run_dir),
                    "metrics_file": None,
                    "metrics": {},
                    "scores": scores or _failed_scores(),
                    "adaptive_metadata": current_config.get("adaptive_metadata"),
                    "failed": True,
                })
            _upload_sequence_run(
                s3,
                outputs_uri,
                dataset_name,
                batch_run_id,
                sequence_number,
                local_run_dir,
            )
            completed_runs += 1
        else:
            print("No local run directory was created before process exit.")
            if adaptive_mode == BAYES_MODE:
                scores = _failed_scores()
                completed_run_records.append({
                    "sequence_number": sequence_number,
                    "config_name": current_config_name,
                    "local_run_dir": None,
                    "metrics_file": None,
                    "metrics": {},
                    "scores": scores,
                    "adaptive_metadata": current_config.get("adaptive_metadata"),
                    "failed": True,
                })

        if returncode != 0 and adaptive_mode != BAYES_MODE:
            print(
                f"Experiment {sequence_number} failed with exit code {returncode}; "
                f"uploaded {completed_runs} completed/partial run folder(s)."
            )
            raise SystemExit(returncode)
        if returncode != 0:
            print(
                f"Experiment {sequence_number} failed with exit code {returncode}; "
                "recorded as a poor Bayesian trial and continuing."
            )
        if local_run_dir is None and adaptive_mode != BAYES_MODE:
            raise RuntimeError(
                f"Experiment {sequence_number} completed without a run directory"
            )
        if metrics_path is None and adaptive_mode != BAYES_MODE:
            raise RuntimeError(
                f"Experiment {sequence_number} completed without a metrics report"
            )

        if adaptive_mode == BAYES_MODE:
            for record in completed_run_records:
                if "params" not in record:
                    record["params"] = extract_bayes_params_from_config(current_config)
                if "objective_score" not in record:
                    record["objective_score"] = record["scores"].get(
                        "objective_score", BAYES_FAILED_SCORE
                    )
                    record["constraint_status"] = record["scores"].get(
                        "constraint_status", "unknown"
                    )
                    record["quality_score"] = record["scores"].get("quality_score")
                    record["speed_score"] = record["scores"].get("speed_score")
            _write_json(trial_history_path, {"trials": completed_run_records})
            early_stop_info = should_stop_bayes_early(completed_run_records)
            if early_stop_info.get("should_stop"):
                print(
                    f"Stopping Bayesian optimization after {sequence_number} "
                    f"trial(s): {early_stop_info['reason']}"
                )
                break

        if sequence_number == experiment_count:
            break

        generation_metrics_path = metrics_path
        if generation_metrics_path is None:
            generation_metrics_path = (
                CONFIGS_DIR / "adaptive" / batch_run_id
                / f"{dataset_name}_failed_metrics_{sequence_number:02d}.json"
            )
            _write_json(generation_metrics_path, {})
        next_config_path, next_config_name = _generate_adaptive_config(
            current_config_path,
            generation_metrics_path,
            dataset_name,
            sequence_number,
            adaptive_mode,
            batch_run_id,
            trial_history_path=trial_history_path,
        )
        config_target = S3Uri(
            bucket=configs_uri.bucket,
            key=_join_s3_key(
                configs_uri.key,
                "adaptive",
                batch_run_id,
                next_config_name,
            ),
        )
        _upload_object(s3, next_config_path, config_target)
        current_config_path = next_config_path

    if completed_run_records:
        report_dir = _write_batch_report(
            dataset_name,
            batch_run_id,
            completed_run_records,
            adaptive_mode=adaptive_mode,
            early_stop_info=early_stop_info,
        )
        _upload_batch_report(s3, outputs_uri, dataset_name, batch_run_id, report_dir)


def main():
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    configs_uri = _parse_s3_uri(_get_required_env("S3_CONFIGS_URI"))
    datasets_uri = _parse_s3_uri(_get_required_env("S3_DATASETS_URI"))
    outputs_uri = _parse_s3_uri(_get_required_env("S3_OUTPUTS_URI"))
    adaptive_mode = os.environ.get("ADAPTIVE_MODE", "metric_conservative")
    config_name = os.environ.get("CONFIG_NAME", DEFAULT_CONFIG_NAME)
    experiment_count = _get_experiment_count(
        "24" if adaptive_mode == BAYES_MODE else "1"
    )
    batch_run_id_value = os.environ.get("BATCH_RUN_ID")
    batch_run_id = (
        _sanitize_token(batch_run_id_value, "BATCH_RUN_ID")
        if batch_run_id_value
        else _default_batch_run_id()
    )

    if "/" in config_name or "\\" in config_name or config_name in {".", ".."}:
        raise ValueError(f"Invalid CONFIG_NAME: {config_name}")

    s3 = boto3.client("s3", region_name=region)

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    config_key = _join_s3_key(configs_uri.key, config_name)
    config_path = CONFIGS_DIR / config_name
    _download_object(s3, S3Uri(configs_uri.bucket, config_key), config_path)

    dataset_name = _load_dataset_name(config_path)
    dataset_prefix = _join_s3_key(datasets_uri.key, dataset_name)
    dataset_dir = DATASETS_DIR / dataset_name
    _download_prefix(s3, S3Uri(datasets_uri.bucket, dataset_prefix), dataset_dir)

    if experiment_count == 1 and not batch_run_id_value:
        print(f"Running experiment for dataset '{dataset_name}'...")
        result_code = _run_pose_tracking(config_path)
        if result_code != 0:
            raise SystemExit(result_code)

        output_target = S3Uri(
            bucket=outputs_uri.bucket,
            key=_join_s3_key(outputs_uri.key, dataset_name),
        )
        _upload_prefix(s3, EXPERIMENTS_DIR / dataset_name, output_target)
    else:
        print(
            f"Adaptive Batch run id: {batch_run_id}; "
            f"experiment_count={experiment_count}; mode={adaptive_mode}"
        )
        _run_adaptive_sequence(
            s3,
            configs_uri,
            outputs_uri,
            dataset_name,
            config_path,
            experiment_count,
            batch_run_id,
            adaptive_mode,
        )


if __name__ == "__main__":
    main()
