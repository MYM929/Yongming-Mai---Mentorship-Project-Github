#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION, for example us-east-1}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
: "${ECR_REPO:?Set ECR_REPO}"
: "${S3_CONFIGS_URI:?Set S3_CONFIGS_URI, for example s3://bucket/configs}"
: "${S3_DATASETS_URI:?Set S3_DATASETS_URI, for example s3://bucket/datasets}"
: "${S3_OUTPUTS_URI:?Set S3_OUTPUTS_URI, for example s3://bucket/experiments}"

IMAGE_TAG="${IMAGE_TAG:-pose-arm64-v1}"
CONFIG_NAME="${CONFIG_NAME:-dataset_1_bedroom.json}"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker pull "${IMAGE_URI}"

docker run --rm \
  -e AWS_REGION="${AWS_REGION}" \
  -e S3_CONFIGS_URI="${S3_CONFIGS_URI}" \
  -e S3_DATASETS_URI="${S3_DATASETS_URI}" \
  -e S3_OUTPUTS_URI="${S3_OUTPUTS_URI}" \
  -e CONFIG_NAME="${CONFIG_NAME}" \
  "${IMAGE_URI}"
