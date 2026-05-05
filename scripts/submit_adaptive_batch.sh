#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION, for example us-east-1}"
: "${BATCH_JOB_QUEUE:?Set BATCH_JOB_QUEUE, for example pose-tracking-arm64-queue}"
: "${BATCH_JOB_DEFINITION:?Set BATCH_JOB_DEFINITION, for example pose-tracking-arm64}"
: "${S3_CONFIGS_URI:?Set S3_CONFIGS_URI, for example s3://bucket/configs}"
: "${S3_DATASETS_URI:?Set S3_DATASETS_URI, for example s3://bucket/datasets}"
: "${S3_OUTPUTS_URI:?Set S3_OUTPUTS_URI, for example s3://bucket/experiments}"

DATASET_CONFIGS="${DATASET_CONFIGS:-dataset_1_bedroom.json,dataset_2_meeting_room.json}"
EXPERIMENT_COUNT="${EXPERIMENT_COUNT:-3}"
ADAPTIVE_MODE="${ADAPTIVE_MODE:-metric_conservative}"
BATCH_RUN_ID="${BATCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

IFS=',' read -r -a configs <<< "${DATASET_CONFIGS}"

for config_name in "${configs[@]}"; do
  config_name="$(echo "${config_name}" | xargs)"
  if [[ -z "${config_name}" ]]; then
    continue
  fi

  export CONFIG_NAME="${config_name}"
  export AWS_REGION S3_CONFIGS_URI S3_DATASETS_URI S3_OUTPUTS_URI
  export EXPERIMENT_COUNT ADAPTIVE_MODE BATCH_RUN_ID

  safe_dataset="$(
    python - <<'PY'
import os
import re
name = os.environ["CONFIG_NAME"].rsplit(".", 1)[0]
print(re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")[:96])
PY
  )"

  override_path="${tmp_dir}/${safe_dataset}.json"
  python - <<'PY' > "${override_path}"
import json
import os

names = [
    "AWS_REGION",
    "S3_CONFIGS_URI",
    "S3_DATASETS_URI",
    "S3_OUTPUTS_URI",
    "CONFIG_NAME",
    "EXPERIMENT_COUNT",
    "ADAPTIVE_MODE",
    "BATCH_RUN_ID",
]
print(json.dumps({
    "environment": [
        {"name": name, "value": os.environ[name]}
        for name in names
    ]
}))
PY

  job_name="pose-${safe_dataset}-${BATCH_RUN_ID}"
  job_id="$(
    aws batch submit-job \
      --region "${AWS_REGION}" \
      --job-name "${job_name}" \
      --job-queue "${BATCH_JOB_QUEUE}" \
      --job-definition "${BATCH_JOB_DEFINITION}" \
      --container-overrides "file://${override_path}" \
      --query 'jobId' \
      --output text
  )"

  echo "Submitted ${job_name}: ${job_id}"
done

echo "Batch run id: ${BATCH_RUN_ID}"
