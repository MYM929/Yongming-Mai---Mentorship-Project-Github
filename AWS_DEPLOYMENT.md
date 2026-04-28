# AWS Docker Deployment

This project runs as a Linux ARM64 Docker image on EC2. The image contains the
Python code and dependencies only; configs, datasets, and experiment outputs are
handled through S3 at runtime.

## S3 Layout

Expected S3 prefixes:

```text
s3://<bucket>/configs/<config-name>.json
s3://<bucket>/datasets/<dataset_name>/...
s3://<bucket>/experiments/<dataset_name>/...
```

The container downloads only the dataset named by `dataset_name` in the selected
config file.

## Build And Push From Local Machine

Set these values first:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=<account-id>
export ECR_REPO=<repo-name>
export IMAGE_TAG=pose-arm64-v1
export IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
```

Create the ECR repository once if it does not already exist:

```bash
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"
```

Log in, build the ARM64 image, and push it:

```bash
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
    "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker buildx build --platform linux/arm64 -t "$IMAGE_URI" --push .
```

Recommended tag format:

```text
<repo>:pose-arm64-v1
```

## Run On EC2

The EC2 instance should be Linux ARM64, such as an AWS Graviton instance, with an
IAM role attached. Do not place AWS keys inside the image or container.
If the container cannot read the instance role credentials, check that the EC2
metadata response hop limit allows containers to reach IMDSv2, commonly `2`.

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=<account-id>
export ECR_REPO=<repo-name>
export IMAGE_TAG=pose-arm64-v1
export IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
    "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker pull "$IMAGE_URI"

docker run --rm \
  -e AWS_REGION="$AWS_REGION" \
  -e S3_CONFIGS_URI="s3://<bucket>/configs" \
  -e S3_DATASETS_URI="s3://<bucket>/datasets" \
  -e S3_OUTPUTS_URI="s3://<bucket>/experiments" \
  -e CONFIG_NAME="dataset_1_bedroom.json" \
  "$IMAGE_URI"
```

## IAM Role Permissions

The EC2 role needs ECR pull permissions plus both bucket-level and object-level
S3 permissions.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ],
      "Resource": "arn:aws:ecr:<region>:<account-id>:repository/<repo-name>"
    },
    {
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::<bucket>",
      "Condition": {
        "StringLike": {
          "s3:prefix": [
            "configs/*",
            "datasets/*",
            "experiments/*"
          ]
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::<bucket>/configs/*",
        "arn:aws:s3:::<bucket>/datasets/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::<bucket>/experiments/*"
    }
  ]
}
```

## Runtime Variables

Required:

```text
AWS_REGION
S3_CONFIGS_URI
S3_DATASETS_URI
S3_OUTPUTS_URI
```

Optional:

```text
CONFIG_NAME=dataset_1_bedroom.json
```
