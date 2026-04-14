#!/usr/bin/env python3
"""
Convert a TUM RGB-D sequence directory into the pipeline's expected dataset structure.

Usage:
    python tum_to_dataset.py <tum_sequence_dir> [--sequence-type fr1|fr3] [--clean]

Key mapping details:
  - RGB/depth filenames: float seconds -> integer nanoseconds
  - IMU reorder: TUM accelerometer.txt (timestamp ax ay az) ->
    pipeline CSV (timestamp_ns, wx, wy, wz, ax, ay, az) with zero gyro
  - Intrinsics: per-sequence calibration from TUM calibration page
  - Depth scale: 5000.0 (TUM Kinect standard)
"""
import argparse
import json
import os
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

# Per-sequence calibrated intrinsics from TUM.  Distortion is zeroed because
# TUM depth is pre-registered to RGB — undistorting RGB alone would break the
# 1:1 pixel correspondence.
INTRINSICS = {
    "fr1": {
        "width": 640, "height": 480,
        "fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3,
        "k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0,
    },
    "fr3": {
        "width": 640, "height": 480,
        "fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6,
        "k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0,
    },
}


def ts_sec_to_ns(ts_str):
    """Convert a TUM timestamp string (float seconds) to integer nanoseconds."""
    return int(round(float(ts_str) * 1e9))


def parse_tum_list(filepath):
    """Parse a TUM-style list file.  Returns [(timestamp_str, rel_path), ...]."""
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    return entries


def convert_images(tum_dir, entries, target_subdir):
    """Copy images from TUM dir to dataset/, renaming to nanosecond timestamps."""
    out_dir = os.path.join(DATASET_DIR, target_subdir)
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for ts_str, rel_path in entries:
        src = os.path.join(tum_dir, rel_path)
        if not os.path.exists(src):
            print(f"  Warning: missing {src}")
            continue
        ns = ts_sec_to_ns(ts_str)
        ext = os.path.splitext(rel_path)[1]
        dst = os.path.join(out_dir, f"{ns}{ext}")
        shutil.copy2(src, dst)
        count += 1
    return count


def convert_accelerometer(tum_dir):
    """Convert TUM accelerometer.txt to pipeline's IMU CSV format.

    TUM Kinect has only a 3-axis accelerometer (no gyroscope), so gyro
    columns are filled with zeros.  The pipeline's gyro integrator will
    produce identity rotations, effectively disabling IMU-aided tracking.
    """
    accel_path = os.path.join(tum_dir, "accelerometer.txt")
    if not os.path.exists(accel_path):
        print("  No accelerometer.txt found — skipping IMU conversion")
        return

    imu_dir = os.path.join(DATASET_DIR, "imu")
    os.makedirs(imu_dir, exist_ok=True)
    out_path = os.path.join(imu_dir, "data.csv")

    rows = []
    with open(accel_path, "r") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 4:
                t_ns = ts_sec_to_ns(parts[0])
                rows.append(f"{t_ns},0.0,0.0,0.0,{parts[1]},{parts[2]},{parts[3]}")
            elif len(parts) >= 7:
                t_ns = ts_sec_to_ns(parts[0])
                rows.append(f"{t_ns},{parts[4]},{parts[5]},{parts[6]},{parts[1]},{parts[2]},{parts[3]}")

    if not rows:
        print("  No valid IMU data found — skipping")
        return

    imu_dir = os.path.join(DATASET_DIR, "imu")
    os.makedirs(imu_dir, exist_ok=True)
    out_path = os.path.join(imu_dir, "data.csv")
    with open(out_path, "w") as fout:
        fout.write("timestamp_ns,wx,wy,wz,ax,ay,az\n")
        fout.write("\n".join(rows) + "\n")
    print(f"  Wrote {len(rows)} IMU entries to {out_path}")


def write_intrinsics(seq_type):
    """Write rgb_intrinsics.json for the given sequence type."""
    calib_dir = os.path.join(DATASET_DIR, "calib")
    os.makedirs(calib_dir, exist_ok=True)
    out_path = os.path.join(calib_dir, "rgb_intrinsics.json")
    with open(out_path, "w") as f:
        json.dump(INTRINSICS[seq_type], f, indent=2)
    print(f"  Wrote {seq_type} intrinsics to {out_path}")


def detect_sequence_type(tum_dir):
    dirname = os.path.basename(tum_dir.rstrip("/\\")).lower()
    if "freiburg1" in dirname:
        return "fr1"
    if "freiburg3" in dirname:
        return "fr3"
    return None


def clean_dataset():
    """Remove previous dataset contents (images, lists, trajectory)."""
    for subdir in ("rgb", "depth", "calib", "imu"):
        path = os.path.join(DATASET_DIR, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)
    for fname in ("rgb.txt", "depth.txt", "pose_trajectory.txt", "groundtruth.txt"):
        path = os.path.join(DATASET_DIR, fname)
        if os.path.exists(path):
            os.remove(path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a TUM RGB-D sequence to the pipeline's dataset format")
    parser.add_argument("tum_dir", help="Path to extracted TUM sequence directory")
    parser.add_argument(
        "--sequence-type", choices=["fr1", "fr3"],
        help="Sequence type (auto-detected from directory name if omitted)")
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove existing dataset contents before conversion")
    args = parser.parse_args()

    tum_dir = os.path.abspath(args.tum_dir)
    if not os.path.isdir(tum_dir):
        print(f"Error: {tum_dir} is not a directory")
        return 1

    seq_type = args.sequence_type or detect_sequence_type(tum_dir)
    if not seq_type:
        print("Error: could not auto-detect sequence type. "
              "Use --sequence-type fr1|fr3")
        return 1

    print(f"TUM sequence : {tum_dir}")
    print(f"Sequence type: {seq_type}")
    print(f"Output       : {DATASET_DIR}")

    if args.clean:
        print("\nCleaning previous dataset...")
        clean_dataset()

    # --- Parse TUM association files ---
    rgb_list = os.path.join(tum_dir, "rgb.txt")
    depth_list = os.path.join(tum_dir, "depth.txt")
    if not os.path.exists(rgb_list) or not os.path.exists(depth_list):
        print("Error: rgb.txt or depth.txt not found in TUM directory")
        return 1

    print("\nParsing TUM lists...")
    rgb_entries = parse_tum_list(rgb_list)
    depth_entries = parse_tum_list(depth_list)
    print(f"  {len(rgb_entries)} RGB,  {len(depth_entries)} depth entries")

    # --- Copy and rename images ---
    print("\nConverting RGB images...")
    n_rgb = convert_images(tum_dir, rgb_entries, "rgb")
    print(f"  Copied {n_rgb} RGB images")

    print("\nConverting depth images...")
    n_depth = convert_images(tum_dir, depth_entries, "depth")
    print(f"  Copied {n_depth} depth images")

    # --- Intrinsics ---
    print("\nWriting camera intrinsics...")
    write_intrinsics(seq_type)

    # --- IMU ---
    print("\nConverting IMU data...")
    convert_accelerometer(tum_dir)

    # --- Ground truth ---
    gt_src = os.path.join(tum_dir, "groundtruth.txt")
    gt_dst = os.path.join(DATASET_DIR, "groundtruth.txt")
    if os.path.exists(gt_src):
        shutil.copy2(gt_src, gt_dst)
        print(f"\nCopied groundtruth.txt")
    else:
        print("\nWarning: no groundtruth.txt in TUM directory")

    print("\n" + "=" * 50)
    print("Conversion complete!")
    print(f"Next: python 1.make_file_lists.py")
    print(f"Then: python 2.pose_tracking.py --config tum_{seq_type}_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
