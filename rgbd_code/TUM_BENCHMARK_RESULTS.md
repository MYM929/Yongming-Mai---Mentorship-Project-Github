# TUM RGB-D Benchmark Results

## Overview

This document records the Absolute Trajectory Error (ATE) evaluation of the
pipeline against two TUM RGB-D benchmark sequences, establishing the measurement
infrastructure for automated parameter tuning.

## Sequences Tested

| Sequence | Duration | GT Path Length | Frames Processed | Pipeline Runtime |
|----------|----------|---------------|------------------|-----------------|
| fr1/room | 48.9s | 15.99m | 1348 / 1362 | 276s |
| fr3/long_office_household | 87.1s | 21.46m | 2488 / 2488 | 497s |

## ATE Results (Sim(3) Umeyama alignment with scale correction)

| Metric | fr1/room | fr3/long_office_household |
|--------|----------|--------------------------|
| **RMSE** | **0.1073 m** | **0.0410 m** |
| Mean | 0.0916 m | 0.0365 m |
| Median | 0.0709 m | 0.0324 m |
| Max | 0.2669 m | 0.1188 m |
| Min | 0.0132 m | 0.0073 m |
| Std | 0.0559 m | 0.0186 m |
| Scale correction | 1.034 | 0.996 |

## Interpretation

### What these numbers mean

**ATE RMSE** (Root Mean Square Error) measures the average deviation of the
estimated camera position from the ground truth, after optimally aligning the
two trajectories. It is the single most common metric for comparing SLAM/VO
systems.

- **RMSE** gives the overall "typical" error magnitude. It penalizes large
  outliers more heavily than mean.
- **Mean** is the arithmetic average error — less sensitive to outliers.
- **Median** is the 50th percentile error — the "most representative" single
  number, robust to outlier frames.
- **Max** reveals the worst-case deviation — important for understanding if the
  trajectory ever diverges badly.

### What is "good" vs "poor" for room-scale?

For room-scale sequences (trajectory lengths of 15-25m):

| Quality | ATE RMSE | Interpretation |
|---------|----------|----------------|
| Excellent | < 0.03 m | State-of-the-art SLAM systems |
| Good | 0.03 - 0.10 m | Solid trajectory, suitable for 3D reconstruction |
| Acceptable | 0.10 - 0.30 m | Usable but visible drift, loop closures may be weak |
| Poor | 0.30 - 0.50 m | Significant drift, check depth scale and intrinsics |
| Failed | > 0.50 m | Trajectory diverged, fundamental misconfiguration |

### Analysis of our results

**fr3/long_office_household (RMSE = 0.041 m)**: This is a **good** result. The
estimated trajectory closely tracks the ground truth with errors mostly in the
3-4 cm range. The scale correction factor of 0.996 is nearly 1.0, meaning our
depth scale (5000.0) is correctly configured. The trajectory overlay plot shows
the estimated path (blue) almost perfectly overlapping with ground truth (dashed
gray) throughout the ~87-second, 21-meter sequence. The max error of 0.12m
occurs in a localized region — likely during a fast turn or textureless area.

**fr1/room (RMSE = 0.107 m)**: This is on the border of **acceptable**. The
fr1/room sequence is harder: faster camera motion (avg 0.334 m/s vs fr3's
~0.25 m/s), higher angular velocity (avg ~30 deg/s), and the fr1 Kinect had
uncalibrated distortion (we zero it to preserve depth-RGB registration). The
scale correction of 1.034 suggests a ~3.4% scale offset, which is consistent
with the known fr1 depth scaling factor of 1.035 documented by TUM. The
trajectory overlay shows good overall shape agreement with some deviation in the
desk area where rapid back-and-forth motion occurs. The max error of 0.27m stays
well below the 0.5m divergence threshold.

## Configuration

Both sequences used the full `sample_config.json` parameter set with these
TUM-specific overrides:

- **Resolution**: 640x480 (TUM Kinect native)
- **Depth scale**: 5000.0 (TUM standard, vs pipeline default of 1000.0)
- **MAX_DEPTH**: 5.0m (increased from 3.0m for room-scale coverage)
- **ODO_SCALE**: 0.5 (odometry at 320x240, up from 0.25)
- **MAX_VELOCITY**: 1.0 m/s (relaxed from 0.5 for handheld motion)
- **MAX_ANGULAR_VEL**: 3.0 rad/s (relaxed from 2.5)
- **Intrinsics**: Per-sequence calibrated values with zero distortion
  - fr1: fx=517.3, fy=516.5, cx=318.6, cy=255.3
  - fr3: fx=535.4, fy=539.2, cx=320.1, cy=247.6

## Files Produced

```
tum_results/
  fr1_room/
    pose_trajectory.txt          # Estimated trajectory (TUM format)
    groundtruth.txt              # Ground truth from mocap
    metrics.json                 # Pipeline operational metrics
    ate_results.zip              # evo serialized results
    ate_plot_map.png             # ATE color-mapped on trajectory
    ate_plot_raw.png             # ATE raw error plot
    traj_overlay_trajectories.png # 3D trajectory overlay
    traj_overlay_xyz.png         # Per-axis comparison
    traj_overlay_rpy.png         # Orientation comparison
    traj_overlay_speeds.png      # Speed comparison
  fr3_long_office/
    (same file set)
```

## How to Reproduce

```bash
# 1. Download and extract TUM sequences
curl -L -o tum_data/fr1_room.tgz https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
curl -L -o tum_data/fr3_office.tgz https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz
tar xzf tum_data/fr1_room.tgz -C tum_data
tar xzf tum_data/fr3_office.tgz -C tum_data

# 2. Convert, generate lists, run pipeline
python tum_to_dataset.py tum_data/rgbd_dataset_freiburg1_room --clean
python 1.make_file_lists.py
python 2.pose_tracking.py --config tum_fr1_config.json

# 3. Evaluate
evo_ape tum dataset/groundtruth.txt dataset/pose_trajectory.txt --align --correct_scale -v
```

## Next Steps

The ATE measurement infrastructure built here becomes the quality metric for
every AWS Batch experiment. Issue #2 wires this into the pipeline's output.
Together they form the objective function for parameter tuning.
