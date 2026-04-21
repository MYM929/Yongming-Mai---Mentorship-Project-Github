# RGB-D Docker Pipeline

Dockerized pipeline for RGB-D odometry, point cloud generation, and TUM FR1 evaluation with optional AWS S3 input sync and result upload.

## Build

```powershell
docker build --no-cache -t rgbd_docker .
```

## Run (your current command)

```powershell
docker run --rm `
  -v "$env:USERPROFILE\.aws:/root/.aws:ro" `
  -e AWS_SYNC_MODE=on `
  -e AWS_REGION=us-east-1 `
  -e S3_BUCKET=yongming-dataset-bucket `
  -e S3_DATASET_PREFIX=dataset `
  -e S3_CONFIGS_PREFIX=configs `
  rgbd_docker tum-fr1-eval
```

## Run all TUM FR1 steps (pipeline + eval + upload)

```powershell
docker run --rm `
  -v "$env:USERPROFILE\.aws:/root/.aws:ro" `
  -e AWS_SYNC_MODE=on `
  -e AWS_REGION=us-east-1 `
  -e S3_BUCKET=yongming-dataset-bucket `
  -e S3_DATASET_PREFIX=dataset `
  -e S3_CONFIGS_PREFIX=configs `
  -e S3_EXPERIMENTS_PREFIX=experiments `
  -e S3_RESULTS_PREFIX=experiments/tum_results `
  rgbd_docker tum-fr1-all
```

## Modes

- `normal`: `1.make_file_lists.py` -> `2.pose_tracking.py` -> `3.build_pointcloud.py`
- `visualize`: `4.visualize_pointcloud.py`
- `tum-fr1`: TUM FR1 tracking pipeline
- `tum-fr1-eval`: TUM FR1 evaluation (runs evo tools, uploads `tum_results` to S3)
- `tum-fr1-all`: runs `tum-fr1` then `tum-fr1-eval`

## AWS environment variables

- `AWS_SYNC_MODE`: `off` | `auto` | `on`
  - `off`: no S3 download/upload
  - `auto`: use S3 only if AWS CLI + credentials are available
  - `on`: require AWS CLI + credentials, otherwise fail
- `AWS_REGION`: AWS region (default `us-east-1`)
- `S3_BUCKET`: S3 bucket name
- `S3_DATASET_PREFIX`: input dataset prefix (default `dataset`)
- `S3_CONFIGS_PREFIX`: input config prefix (default `configs`)
- `S3_EXPERIMENTS_PREFIX`: input experiments prefix (default `experiments`)
- `S3_RESULTS_PREFIX`: upload prefix for `tum_results` (default `experiments/tum_results`)

## Input/output expectations

- Input data is synced into:
  - `/app/dataset`
  - `/app/configs`
  - `/app/experiments`
- Evaluation artifacts are produced under:
  - `/app/tum_results`
- When AWS upload is enabled, `/app/tum_results` is synced to:
  - `s3://$S3_BUCKET/$S3_RESULTS_PREFIX`
