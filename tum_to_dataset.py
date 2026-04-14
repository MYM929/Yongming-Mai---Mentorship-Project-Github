"""
Convert a TUM RGB-D benchmark sequence into the pipeline's expected dataset/ layout.

Usage:
    python tum_to_dataset.py <tum_sequence_dir> [--output dataset] [--sequence fr1|fr3]

The script:
  1. Copies RGB/depth images, renaming from decimal TUM timestamps to integer
     nanosecond timestamps (required by 1.make_file_lists.py).
  2. Writes calib/rgb_intrinsics.json with the published TUM calibration.
  3. Converts accelerometer.txt → imu/data.csv in pipeline format.
  4. Writes a per-sequence config JSON (e.g., tum_fr1_config.json) that sets
     DEPTH_SCALE=5000, resolution=640×480, and matching intrinsics.
  5. Copies groundtruth.txt into the output directory for later evo evaluation.
"""

import argparse
import json
import os
import re
import shutil
import sys

# ── TUM published calibration (from cvg.cit.tum.de) ────────────────────────
# OpenCV distortion mapping: d0→k1, d1→k2, d2→p1, d3→p2, d4→k3
# TUM recommends using the ROS default parameter set without undistortion,
# as undistorting the pre-registered depth images is not trivial.  We still
# use the per-sensor focal length / principal point but set distortion to
# zero so the pipeline does not attempt to undistort the depth maps.
CALIBRATIONS = {
    "fr1": {
        "width": 640, "height": 480,
        "fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
    },
    "fr2": {
        "width": 640, "height": 480,
        "fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
    },
    "fr3": {
        "width": 640, "height": 480,
        "fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0,
    },
}

DEPTH_SCALE_TUM = 5000.0


def detect_sequence_type(tum_dir, override=None):
    """Return 'fr1', 'fr2', or 'fr3' by inspecting the directory name."""
    if override:
        return override
    name = os.path.basename(os.path.normpath(tum_dir)).lower()
    for tag in ("freiburg1", "fr1"):
        if tag in name:
            return "fr1"
    for tag in ("freiburg2", "fr2"):
        if tag in name:
            return "fr2"
    for tag in ("freiburg3", "fr3"):
        if tag in name:
            return "fr3"
    sys.exit(f"Cannot detect sequence type from '{name}'. Use --sequence.")


def find_tum_root(tum_dir):
    """Resolve potential nested extraction (e.g. seq/seq/) to the real root."""
    if os.path.isfile(os.path.join(tum_dir, "rgb.txt")):
        return tum_dir
    entries = os.listdir(tum_dir)
    if len(entries) == 1:
        nested = os.path.join(tum_dir, entries[0])
        if os.path.isdir(nested) and os.path.isfile(os.path.join(nested, "rgb.txt")):
            return nested
    sys.exit(f"Cannot locate rgb.txt inside '{tum_dir}'. Check directory structure.")


def parse_tum_list(filepath):
    """Parse a TUM-format list file (rgb.txt / depth.txt).
    Returns [(float_timestamp, relative_path), ...] sorted by timestamp.
    """
    entries = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((float(parts[0]), parts[1]))
    entries.sort(key=lambda x: x[0])
    return entries


def ts_to_ns(ts_sec):
    """Convert float seconds to integer nanoseconds."""
    return int(round(ts_sec * 1e9))


def copy_images(entries, src_root, dst_dir, label):
    """Copy images from TUM layout to pipeline layout, renaming to ns integer stems.
    Returns list of (ns_int, new_filename) for verification.
    """
    os.makedirs(dst_dir, exist_ok=True)
    results = []
    for ts, rel_path in entries:
        src = os.path.join(src_root, rel_path)
        if not os.path.isfile(src):
            print(f"  Warning: missing {src}, skipping")
            continue
        ext = os.path.splitext(rel_path)[1]
        ns = ts_to_ns(ts)
        new_name = f"{ns}{ext}"
        dst = os.path.join(dst_dir, new_name)
        shutil.copy2(src, dst)
        results.append((ns, new_name))
    print(f"  Copied {len(results)} {label} images")
    return results


def write_intrinsics(calib_dict, out_dir):
    """Write calib/rgb_intrinsics.json."""
    calib_dir = os.path.join(out_dir, "calib")
    os.makedirs(calib_dir, exist_ok=True)
    path = os.path.join(calib_dir, "rgb_intrinsics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib_dict, f, indent=2)
    print(f"  Wrote {path}")


def convert_accelerometer(accel_path, out_dir):
    """Convert TUM accelerometer.txt → imu/data.csv.
    TUM format:  timestamp ax ay az  (space-separated, seconds)
    Pipeline format:  timestamp_ns,wx,wy,wz,ax,ay,az  (CSV, nanoseconds)
    TUM accelerometer.txt has no gyro data, so wx=wy=wz=0.
    """
    if not os.path.isfile(accel_path):
        print("  No accelerometer.txt found, skipping IMU conversion")
        return

    rows = []
    with open(accel_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            ts_sec = float(parts[0])
            ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
            ts_ns = ts_to_ns(ts_sec)
            rows.append((ts_ns, 0.0, 0.0, 0.0, ax, ay, az))

    if not rows:
        print("  accelerometer.txt has no data rows, skipping IMU conversion")
        return

    imu_dir = os.path.join(out_dir, "imu")
    os.makedirs(imu_dir, exist_ok=True)
    csv_path = os.path.join(imu_dir, "data.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("timestamp_ns,wx,wy,wz,ax,ay,az\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]}\n")
    print(f"  Wrote {len(rows)} IMU samples to {csv_path}")


def copy_groundtruth(tum_root, out_dir):
    """Copy groundtruth.txt to output for evo evaluation."""
    src = os.path.join(tum_root, "groundtruth.txt")
    if os.path.isfile(src):
        dst = os.path.join(out_dir, "groundtruth.txt")
        shutil.copy2(src, dst)
        print(f"  Copied groundtruth.txt")
    else:
        print("  Warning: no groundtruth.txt found")


def write_pipeline_config(seq_type, calib_dict, out_path):
    """Write a pipeline config JSON with TUM-appropriate settings."""
    config = {
        "_comment": f"Auto-generated config for TUM {seq_type} sequence",
        "WIDTH": calib_dict["width"],
        "HEIGHT": calib_dict["height"],
        "FX": calib_dict["fx"],
        "FY": calib_dict["fy"],
        "CX": calib_dict["cx"],
        "CY": calib_dict["cy"],
        "DEPTH_SCALE": DEPTH_SCALE_TUM,
        "MAX_DEPTH": 5.0,
        "MIN_DEPTH": 0.1,
        "MAX_DEPTH_DIFF": 0.07,
        "ODO_SCALE": 0.5,
        "MAX_VELOCITY": 2.0,
        "MAX_ANGULAR_VEL": 5.0,
        "MAX_CONSECUTIVE_SKIP": 15,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote pipeline config to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a TUM RGB-D sequence into the pipeline dataset layout")
    parser.add_argument("tum_dir",
                        help="Path to the TUM sequence directory")
    parser.add_argument("--output", default="dataset",
                        help="Output directory (default: dataset)")
    parser.add_argument("--sequence", choices=["fr1", "fr2", "fr3"], default=None,
                        help="Override auto-detected sequence type")
    args = parser.parse_args()

    tum_root = find_tum_root(args.tum_dir)
    seq_type = detect_sequence_type(tum_root, args.sequence)
    calib = CALIBRATIONS[seq_type]
    out_dir = args.output

    print(f"TUM sequence root : {tum_root}")
    print(f"Sequence type     : {seq_type}")
    print(f"Output directory  : {os.path.abspath(out_dir)}")
    print()

    # Clean previous pipeline data in output dir
    for subdir in ("rgb", "depth", "calib", "imu"):
        target = os.path.join(out_dir, subdir)
        if os.path.isdir(target):
            shutil.rmtree(target)
    for fname in ("rgb.txt", "depth.txt", "pose_trajectory.txt",
                   "groundtruth.txt", "metrics.json"):
        fpath = os.path.join(out_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    # 1. Copy images with renamed timestamps
    print("[1/5] Copying RGB images...")
    rgb_entries = parse_tum_list(os.path.join(tum_root, "rgb.txt"))
    copy_images(rgb_entries, tum_root, os.path.join(out_dir, "rgb"), "RGB")

    print("[2/5] Copying depth images...")
    depth_entries = parse_tum_list(os.path.join(tum_root, "depth.txt"))
    copy_images(depth_entries, tum_root, os.path.join(out_dir, "depth"), "depth")

    # 2. Write calibration
    print("[3/5] Writing calibration...")
    write_intrinsics(calib, out_dir)

    # 3. Convert IMU data
    print("[4/5] Converting IMU data...")
    convert_accelerometer(os.path.join(tum_root, "accelerometer.txt"), out_dir)

    # 4. Copy ground truth
    print("[5/5] Copying ground truth...")
    copy_groundtruth(tum_root, out_dir)

    # 5. Write pipeline config
    config_name = f"tum_{seq_type}_config.json"
    config_path = os.path.join(os.path.dirname(os.path.abspath(out_dir)), config_name)
    write_pipeline_config(seq_type, calib, config_path)

    print(f"\nConversion complete. Next steps:")
    print(f"  1. python 1.make_file_lists.py")
    print(f"  2. python 2.pose_tracking.py --config {config_name}")


if __name__ == "__main__":
    main()
