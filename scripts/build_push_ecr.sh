#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION, for example us-east-1}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
: "${ECR_REPO:?Set ECR_REPO}"

IMAGE_TAG="${IMAGE_TAG:-pose-arm64-v1}"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

aws ecr describe-repositories \
  --repository-names "${ECR_REPO}" \
  --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository \
    --repository-name "${ECR_REPO}" \
    --region "${AWS_REGION}" >/dev/null

aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker buildx build --platform linux/arm64 -t "${IMAGE_URI}" --push .

echo "Pushed ${IMAGE_URI}"
