#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION, for example us-east-1}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
: "${ECR_REPO:?Set ECR_REPO}"
: "${BATCH_INSTANCE_ROLE_ARN:?Set BATCH_INSTANCE_ROLE_ARN, for example arn:aws:iam::<account>:instance-profile/ecsInstanceRole}"
: "${BATCH_JOB_ROLE_ARN:?Set BATCH_JOB_ROLE_ARN with S3 read/write permissions for the container}"
: "${SUBNET_IDS:?Set SUBNET_IDS as comma-separated subnet ids}"
: "${SECURITY_GROUP_IDS:?Set SECURITY_GROUP_IDS as comma-separated security group ids}"

for name in BATCH_INSTANCE_ROLE_ARN BATCH_JOB_ROLE_ARN SUBNET_IDS SECURITY_GROUP_IDS; do
  value="${!name}"
  if [[ "${value}" =~ \<.*\> || "${value}" =~ (^|[^A-Za-z0-9])xxx([^A-Za-z0-9]|$) || "${value}" =~ (^|[^A-Za-z0-9])aaa([^A-Za-z0-9]|$) || "${value}" =~ (^|[^A-Za-z0-9])yyy([^A-Za-z0-9]|$) ]]; then
    echo "${name} is still a placeholder: ${value}" >&2
    exit 1
  fi
done

if [[ -z "${BATCH_SERVICE_ROLE_ARN:-}" ]]; then
  BATCH_SERVICE_ROLE_ARN="$(
    aws iam get-role \
      --role-name AWSServiceRoleForBatch \
      --query 'Role.Arn' \
      --output text 2>/dev/null || true
  )"
  if [[ -z "${BATCH_SERVICE_ROLE_ARN}" || "${BATCH_SERVICE_ROLE_ARN}" == "None" ]]; then
    aws iam create-service-linked-role \
      --aws-service-name batch.amazonaws.com >/dev/null
    BATCH_SERVICE_ROLE_ARN="$(
      aws iam get-role \
        --role-name AWSServiceRoleForBatch \
        --query 'Role.Arn' \
        --output text
    )"
  fi
  echo "Using Batch service role ${BATCH_SERVICE_ROLE_ARN}"
fi

IMAGE_TAG="${IMAGE_TAG:-pose-arm64-v1}"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

BATCH_COMPUTE_ENV="${BATCH_COMPUTE_ENV:-pose-tracking-arm64-ce}"
BATCH_JOB_QUEUE="${BATCH_JOB_QUEUE:-pose-tracking-arm64-queue}"
BATCH_JOB_DEFINITION="${BATCH_JOB_DEFINITION:-pose-tracking-arm64}"
BATCH_MAX_VCPUS="${BATCH_MAX_VCPUS:-8}"
BATCH_JOB_VCPUS="${BATCH_JOB_VCPUS:-4}"
BATCH_JOB_MEMORY_MB="${BATCH_JOB_MEMORY_MB:-7000}"
BATCH_INSTANCE_TYPES="${BATCH_INSTANCE_TYPES:-c7g.xlarge,c6g.xlarge}"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

export IMAGE_URI BATCH_JOB_ROLE_ARN BATCH_JOB_VCPUS BATCH_JOB_MEMORY_MB
export BATCH_MAX_VCPUS BATCH_INSTANCE_TYPES SUBNET_IDS SECURITY_GROUP_IDS
export BATCH_INSTANCE_ROLE_ARN

batch_instance_profile_name="${BATCH_INSTANCE_ROLE_ARN##*/}"
batch_job_role_name="${BATCH_JOB_ROLE_ARN##*/}"

aws iam get-instance-profile \
  --instance-profile-name "${batch_instance_profile_name}" >/dev/null

aws iam get-role \
  --role-name "${batch_job_role_name}" >/dev/null

IFS=',' read -r -a subnet_id_list <<< "${SUBNET_IDS}"
IFS=',' read -r -a security_group_id_list <<< "${SECURITY_GROUP_IDS}"

aws ec2 describe-subnets \
  --region "${AWS_REGION}" \
  --subnet-ids "${subnet_id_list[@]}" >/dev/null

aws ec2 describe-security-groups \
  --region "${AWS_REGION}" \
  --group-ids "${security_group_id_list[@]}" >/dev/null

python - <<'PY' > "${tmp_dir}/compute_resources.json"
import json
import os

def csv(name):
    return [v.strip() for v in os.environ[name].split(",") if v.strip()]

print(json.dumps({
    "type": "EC2",
    "allocationStrategy": "BEST_FIT_PROGRESSIVE",
    "minvCpus": 0,
    "maxvCpus": int(os.environ["BATCH_MAX_VCPUS"]),
    "desiredvCpus": 0,
    "instanceTypes": csv("BATCH_INSTANCE_TYPES"),
    "subnets": csv("SUBNET_IDS"),
    "securityGroupIds": csv("SECURITY_GROUP_IDS"),
    "instanceRole": os.environ["BATCH_INSTANCE_ROLE_ARN"],
}))
PY

python - <<'PY' > "${tmp_dir}/container_properties.json"
import json
import os

print(json.dumps({
    "image": os.environ["IMAGE_URI"],
    "jobRoleArn": os.environ["BATCH_JOB_ROLE_ARN"],
    "resourceRequirements": [
        {"type": "VCPU", "value": os.environ["BATCH_JOB_VCPUS"]},
        {"type": "MEMORY", "value": os.environ["BATCH_JOB_MEMORY_MB"]},
    ],
}))
PY

compute_env_exists="$(
  aws batch describe-compute-environments \
    --region "${AWS_REGION}" \
    --compute-environments "${BATCH_COMPUTE_ENV}" \
    --query 'length(computeEnvironments)' \
    --output text
)"

if [[ "${compute_env_exists}" == "0" ]]; then
  aws batch create-compute-environment \
    --region "${AWS_REGION}" \
    --compute-environment-name "${BATCH_COMPUTE_ENV}" \
    --type MANAGED \
    --state ENABLED \
    --service-role "${BATCH_SERVICE_ROLE_ARN}" \
    --compute-resources "file://${tmp_dir}/compute_resources.json" >/dev/null
  echo "Created compute environment ${BATCH_COMPUTE_ENV}"
else
  aws batch update-compute-environment \
    --region "${AWS_REGION}" \
    --compute-environment "${BATCH_COMPUTE_ENV}" \
    --state ENABLED \
    --compute-resources "maxvCpus=${BATCH_MAX_VCPUS}" >/dev/null
  echo "Updated compute environment ${BATCH_COMPUTE_ENV}"
fi

queue_exists="$(
  aws batch describe-job-queues \
    --region "${AWS_REGION}" \
    --job-queues "${BATCH_JOB_QUEUE}" \
    --query 'length(jobQueues)' \
    --output text
)"

if [[ "${queue_exists}" == "0" ]]; then
  aws batch create-job-queue \
    --region "${AWS_REGION}" \
    --job-queue-name "${BATCH_JOB_QUEUE}" \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order "order=1,computeEnvironment=${BATCH_COMPUTE_ENV}" >/dev/null
  echo "Created job queue ${BATCH_JOB_QUEUE}"
else
  aws batch update-job-queue \
    --region "${AWS_REGION}" \
    --job-queue "${BATCH_JOB_QUEUE}" \
    --state ENABLED \
    --priority 1 \
    --compute-environment-order "order=1,computeEnvironment=${BATCH_COMPUTE_ENV}" >/dev/null
  echo "Updated job queue ${BATCH_JOB_QUEUE}"
fi

revision="$(
  aws batch register-job-definition \
    --region "${AWS_REGION}" \
    --job-definition-name "${BATCH_JOB_DEFINITION}" \
    --type container \
    --platform-capabilities EC2 \
    --container-properties "file://${tmp_dir}/container_properties.json" \
    --query 'revision' \
    --output text
)"

echo "Registered job definition ${BATCH_JOB_DEFINITION}:${revision}"
echo "Image: ${IMAGE_URI}"
echo "Queue: ${BATCH_JOB_QUEUE}"
