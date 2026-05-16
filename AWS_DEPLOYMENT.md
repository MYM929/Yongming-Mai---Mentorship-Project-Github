# AWS Docker And Batch Deployment

This project runs as one Linux ARM64 Docker image. The image contains the code
and dependencies only; configs, datasets, generated adaptive configs, and
experiment outputs are handled through S3 at runtime.

## S3 Layout

Expected prefixes:

```text
s3://<bucket>/configs/<config-name>.json
s3://<bucket>/configs/adaptive/<batch_run_id>/<dataset>_adaptive_01.json
s3://<bucket>/configs/adaptive/<batch_run_id>/<dataset>_adaptive_02.json
s3://<bucket>/datasets/<dataset_name>/...
s3://<bucket>/experiments/<dataset_name>/<batch_run_id>/run1/...
s3://<bucket>/experiments/<dataset_name>/<batch_run_id>/run2/...
s3://<bucket>/experiments/<dataset_name>/<batch_run_id>/run3/...
s3://<bucket>/experiments/<dataset_name>/<batch_run_id>/report/adaptive_batch_report.md
s3://<bucket>/experiments/<dataset_name>/<batch_run_id>/report/adaptive_batch_report.json
s3://<bucket>/experiments/<dataset_name>/<batch_run_id>/report/adaptive_batch_report.csv
```

For single-run local/manual container usage without `BATCH_RUN_ID`, the legacy
output layout is preserved:

```text
s3://<bucket>/experiments/<dataset_name>/runN/...
```

## Build And Push From Local Machine

Set these values:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=<account-id>
export ECR_REPO=<repo-name>
export IMAGE_TAG=pose-arm64-v1
```

Build and push the ARM64 image:

```bash
bash scripts/build_push_ecr.sh
```

From PowerShell:

```powershell
.\scripts\build_push_ecr.ps1
```

The image URI will be:

```text
<account-id>.dkr.ecr.<region>.amazonaws.com/<repo-name>:pose-arm64-v1
```

## Single Container Run

The container downloads the config named by `CONFIG_NAME`, reads
`dataset_name`, downloads only that dataset, runs `1.pose_tracking.py`, and
uploads outputs.

```bash
docker run --rm \
  -e AWS_REGION="$AWS_REGION" \
  -e S3_CONFIGS_URI="s3://<bucket>/configs" \
  -e S3_DATASETS_URI="s3://<bucket>/datasets" \
  -e S3_OUTPUTS_URI="s3://<bucket>/experiments" \
  -e CONFIG_NAME="dataset_1_bedroom.json" \
  "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG"
```

Optional adaptive variables:

```text
EXPERIMENT_COUNT=3
BATCH_RUN_ID=<unique-run-id>
ADAPTIVE_MODE=metric_conservative
```

When `EXPERIMENT_COUNT=3`, the container runs three experiments in sequence:

```text
run1: base config from CONFIG_NAME
run2: generated <dataset>_adaptive_01.json
run3: generated <dataset>_adaptive_02.json
```

Each `runN/` upload includes the run outputs plus:

```text
used_config.json
run_manifest.json
```

At the end of the sequence, the container writes a dataset report under
`experiments/<dataset_name>/<batch_run_id>/report/`. The report ranks every
config used in the batch and names the best balanced config. The balanced score
uses 70% quality and 30% speed, where quality includes accepted-frame ratio,
segment count, head-tail drift, and loop-closure strength, and speed uses runtime
per accepted frame.

Set `ADAPTIVE_MODE=bayes_opt` to use Optuna TPE Bayesian optimization instead of
the conservative one-step heuristic. In this mode `EXPERIMENT_COUNT` is treated
as the maximum trial budget, defaulting to 24 in the submit scripts when it is
not set. Each dataset optimizes independently, and the container may stop early
after at least 12 trials when the best quality-first objective improves by less
than 0.002 for 8 consecutive trials. The final report includes the actual number
of experiments used, the objective score, constraint status, and early-stop
reason.

## AWS Batch Setup

The recommended Batch shape is one managed EC2 ARM64 compute environment, one
job queue, one job definition, and two submitted jobs that share the same image.
Each job handles one dataset and runs all three adaptive experiments inside its
container.

Required setup variables:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=<account-id>
export ECR_REPO=<repo-name>
export IMAGE_TAG=pose-arm64-v1

export BATCH_INSTANCE_ROLE_ARN=arn:aws:iam::<account-id>:instance-profile/ecsInstanceRole
export BATCH_JOB_ROLE_ARN=arn:aws:iam::<account-id>:role/<batch-job-role>

export SUBNET_IDS=subnet-aaa,subnet-bbb
export SECURITY_GROUP_IDS=sg-aaa
```

`BATCH_SERVICE_ROLE_ARN` is optional. If it is not set, the setup script uses
or creates the AWS Batch service-linked role `AWSServiceRoleForBatch`.

Optional defaults:

```bash
export BATCH_COMPUTE_ENV=pose-tracking-arm64-ce
export BATCH_JOB_QUEUE=pose-tracking-arm64-queue
export BATCH_JOB_DEFINITION=pose-tracking-arm64
export BATCH_MAX_VCPUS=8
export BATCH_JOB_VCPUS=4
export BATCH_JOB_MEMORY_MB=7000
export BATCH_INSTANCE_TYPES=c7g.xlarge,c6g.xlarge
```

Create or update Batch resources and register the current ECR image:

```bash
bash scripts/setup_aws_batch.sh
```

From PowerShell:

```powershell
.\scripts\setup_aws_batch.ps1
```

## Submit Adaptive Dataset Jobs

Submit two jobs, one for each dataset config, using the same job definition:

```bash
export BATCH_JOB_QUEUE=pose-tracking-arm64-queue
export BATCH_JOB_DEFINITION=pose-tracking-arm64
export S3_CONFIGS_URI=s3://<bucket>/configs
export S3_DATASETS_URI=s3://<bucket>/datasets
export S3_OUTPUTS_URI=s3://<bucket>/experiments

bash scripts/submit_adaptive_batch.sh
```

From PowerShell:

```powershell
.\scripts\submit_adaptive_batch.ps1
```

Defaults:

```text
DATASET_CONFIGS=dataset_1_bedroom.json,dataset_2_meeting_room.json
ADAPTIVE_MODE=metric_conservative
EXPERIMENT_COUNT=3 for metric_conservative, 24 for bayes_opt
BATCH_RUN_ID=<UTC timestamp>
```

Use a fixed run id when you want both jobs grouped under a known S3 prefix:

```bash
export BATCH_RUN_ID=adaptive-20260504-001
bash scripts/submit_adaptive_batch.sh
```

## IAM Role Permissions

The Batch job role needs S3 access to configs, datasets, generated adaptive
configs, and outputs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
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
      "Resource": [
        "arn:aws:s3:::<bucket>/configs/adaptive/*",
        "arn:aws:s3:::<bucket>/experiments/*"
      ]
    }
  ]
}
```

The Batch instance role must allow ECS/Batch instance operations and ECR image
pulls. The setup script assumes the role is provided as an instance profile ARN.

## Runtime Variables

Required:

```text
AWS_REGION
S3_CONFIGS_URI
S3_DATASETS_URI
S3_OUTPUTS_URI
CONFIG_NAME
```

Optional:

```text
EXPERIMENT_COUNT=1 for metric_conservative, 24 for bayes_opt
BATCH_RUN_ID=<generated if not provided>
ADAPTIVE_MODE=metric_conservative or bayes_opt
```
