#!/bin/bash
set -euo pipefail

MODE=${1:-normal}
S3_BUCKET=${S3_BUCKET:-yongming-dataset-bucket}
S3_DATASET_PREFIX=${S3_DATASET_PREFIX:-dataset}
S3_CONFIGS_PREFIX=${S3_CONFIGS_PREFIX:-configs}
S3_EXPERIMENTS_PREFIX=${S3_EXPERIMENTS_PREFIX:-experiments}
S3_RESULTS_PREFIX=${S3_RESULTS_PREFIX:-experiments/tum_results}
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_SYNC_MODE=${AWS_SYNC_MODE:-auto}

sync_path_if_exists() {
    local src="$1"
    local dst="$2"

    if aws s3 ls "$src" >/dev/null 2>&1; then
        echo "Syncing ${src} -> ${dst}"
        aws s3 sync "$src" "$dst"
    else
        echo "Skipping missing S3 path: ${src}"
    fi
}

sync_inputs_from_s3() {
    # Modes:
    # - off: never sync
    # - on : always sync and fail on sync errors
    # - auto: sync only if AWS CLI + credentials are available (default)
    if [ "${AWS_SYNC_MODE}" = "off" ]; then
        echo "AWS sync disabled (AWS_SYNC_MODE=off)."
        return 0
    fi

    if ! command -v aws >/dev/null 2>&1; then
        if [ "${AWS_SYNC_MODE}" = "on" ]; then
            echo "AWS CLI not found but AWS_SYNC_MODE=on."
            exit 1
        fi
        echo "AWS CLI not found, skipping AWS sync (AWS_SYNC_MODE=auto)."
        return 0
    fi

    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        if [ "${AWS_SYNC_MODE}" = "on" ]; then
            echo "AWS credentials unavailable but AWS_SYNC_MODE=on."
            exit 1
        fi
        echo "AWS credentials unavailable, skipping AWS sync (AWS_SYNC_MODE=auto)."
        return 0
    fi

    echo "Syncing inputs from s3://${S3_BUCKET}..."
    export AWS_DEFAULT_REGION="${AWS_REGION}"
    mkdir -p ./dataset ./configs ./experiments
    sync_path_if_exists "s3://${S3_BUCKET}/${S3_DATASET_PREFIX}" "./dataset"
    sync_path_if_exists "s3://${S3_BUCKET}/${S3_CONFIGS_PREFIX}" "./configs"
    sync_path_if_exists "s3://${S3_BUCKET}/${S3_EXPERIMENTS_PREFIX}" "./experiments"

    # Backward compatibility for existing env var usage.
    if [ "${SYNC_FROM_S3:-0}" = "1" ]; then
        echo "SYNC_FROM_S3=1 detected (legacy); AWS sync already executed."
    fi
}

upload_results_to_s3() {
    if [ "${AWS_SYNC_MODE}" = "off" ]; then
        echo "AWS upload disabled (AWS_SYNC_MODE=off)."
        return 0
    fi

    if ! command -v aws >/dev/null 2>&1; then
        if [ "${AWS_SYNC_MODE}" = "on" ]; then
            echo "AWS CLI not found but AWS_SYNC_MODE=on."
            exit 1
        fi
        echo "AWS CLI not found, skipping results upload (AWS_SYNC_MODE=auto)."
        return 0
    fi

    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        if [ "${AWS_SYNC_MODE}" = "on" ]; then
            echo "AWS credentials unavailable but AWS_SYNC_MODE=on."
            exit 1
        fi
        echo "AWS credentials unavailable, skipping results upload (AWS_SYNC_MODE=auto)."
        return 0
    fi

    if [ ! -d "./tum_results" ]; then
        echo "No ./tum_results directory found, skipping upload."
        return 0
    fi

    export AWS_DEFAULT_REGION="${AWS_REGION}"
    echo "Uploading tum_results -> s3://${S3_BUCKET}/${S3_RESULTS_PREFIX}"
    aws s3 sync "./tum_results" "s3://${S3_BUCKET}/${S3_RESULTS_PREFIX}"
}

run_tum_fr1() {
    echo "Running TUM FR1 pipeline..."
    sync_inputs_from_s3
    python 1.make_file_lists.py
    python 2.pose_tracking.py --config configs/tum_fr1_config.json
}

run_tum_fr1_eval() {
    echo "Running TUM FR1 evaluation pipeline..."
    mkdir -p configs tum_results/fr1_room

    sync_inputs_from_s3

    if [ -f "tum_fr1_config.json" ] && [ ! -f "configs/tum_fr1_config.json" ]; then
        cp "tum_fr1_config.json" "configs/tum_fr1_config.json"
    fi

    export MPLBACKEND="${MPLBACKEND:-Agg}"

    python 1.make_file_lists.py
    python 2.pose_tracking.py --config configs/tum_fr1_config.json

    evo_ape tum dataset/groundtruth.txt dataset/pose_trajectory.txt \
        --align --correct_scale \
        --save_plot tum_results/fr1_room/ate_plot.png \
        --save_results tum_results/fr1_room/ate_results.zip -v

    evo_traj tum dataset/groundtruth.txt dataset/pose_trajectory.txt \
        --ref dataset/groundtruth.txt \
        --align --correct_scale \
        --save_plot tum_results/fr1_room/traj_overlay.png -v

    upload_results_to_s3
}

if [ "$MODE" = "normal" ]; then
    echo "Running normal RGB-D pipeline..."
    sync_inputs_from_s3
    python 1.make_file_lists.py
    python 2.pose_tracking.py
    python 3.build_pointcloud.py
elif [ "$MODE" = "visualize" ]; then
    echo "Visualizing point cloud..."
    sync_inputs_from_s3
    python 4.visualize_pointcloud.py
elif [ "$MODE" = "tum-fr1" ]; then
    run_tum_fr1
elif [ "$MODE" = "tum-fr1-eval" ]; then
    run_tum_fr1_eval
elif [ "$MODE" = "tum-fr1-all" ]; then
    echo "Running TUM FR1 pipeline + evaluation..."
    run_tum_fr1
    run_tum_fr1_eval
else
    echo "Unknown mode: $MODE"
    exit 1
fi
