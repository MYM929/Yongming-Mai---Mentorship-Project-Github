import json
import os
import posixpath
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3


APP_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = APP_DIR / "configs"
DATASETS_DIR = APP_DIR / "datasets"
EXPERIMENTS_DIR = APP_DIR / "experiments"
DEFAULT_CONFIG_NAME = "dataset_1_bedroom.json"


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


def _load_dataset_name(config_path):
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    dataset_name = config.get("dataset_name") or config.get("active_dataset")
    if not dataset_name:
        raise ValueError("dataset_name missing from config")
    if "/" in dataset_name or "\\" in dataset_name or dataset_name in {".", ".."}:
        raise ValueError(f"Invalid dataset_name in config: {dataset_name}")
    return dataset_name


def main():
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    configs_uri = _parse_s3_uri(_get_required_env("S3_CONFIGS_URI"))
    datasets_uri = _parse_s3_uri(_get_required_env("S3_DATASETS_URI"))
    outputs_uri = _parse_s3_uri(_get_required_env("S3_OUTPUTS_URI"))
    config_name = os.environ.get("CONFIG_NAME", DEFAULT_CONFIG_NAME)

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

    command = [
        sys.executable,
        "1.pose_tracking.py",
        "--config",
        str(config_path),
    ]
    print(f"Running experiment for dataset '{dataset_name}'...")
    result = subprocess.run(command, cwd=str(APP_DIR), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    output_target = S3Uri(
        bucket=outputs_uri.bucket,
        key=_join_s3_key(outputs_uri.key, dataset_name),
    )
    _upload_prefix(s3, EXPERIMENTS_DIR / dataset_name, output_target)


if __name__ == "__main__":
    main()
