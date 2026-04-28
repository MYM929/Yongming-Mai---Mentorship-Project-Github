"""
Generate rgb.txt and depth.txt from the contents of dataset/rgb/ and dataset/depth/.
Run this after copying new RGB and depth images into the dataset folder.
"""
import os
import re

# ================= Configuration =================
# Point to the dataset folder (inner folder containing rgb/, depth/, calib/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")
RGB_DIR = os.path.join(BASE_DIR, "rgb")
DEPTH_DIR = os.path.join(BASE_DIR, "depth")
RGB_TXT = os.path.join(BASE_DIR, "rgb.txt")
DEPTH_TXT = os.path.join(BASE_DIR, "depth.txt")
# Image extensions to consider
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
# =================================================


def get_timestamp_from_filename(filename):
    """
    Extract numeric timestamp from filename (e.g. 20410725000.png -> 20410725000).
    If the stem is a number, use it; otherwise return None (skip file).
    """
    stem, _ = os.path.splitext(filename)
    if re.match(r"^\d+$", stem):
        return int(stem)
    return None


def scan_folder(folder, subdir_name):
    """
    Scan folder for image files, extract timestamps from filenames.
    Returns list of (timestamp, relative_path) sorted by timestamp.
    """
    if not os.path.isdir(folder):
        return []
    entries = []
    for name in os.listdir(folder):
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        ts = get_timestamp_from_filename(name)
        if ts is not None:
            rel_path = f"{subdir_name}/{name}"
            entries.append((ts, rel_path))
    entries.sort(key=lambda x: x[0])
    return entries


def write_list_file(path, entries, comment=None):
    """Write a TUM-style list file: optional comment line, then 'timestamp path' per line."""
    with open(path, "w", encoding="utf-8") as f:
        if comment:
            f.write(comment + "\n")
        for ts, rel_path in entries:
            f.write(f"{ts} {rel_path}\n")
    print(f"  Wrote {len(entries)} entries to {os.path.basename(path)}")


def main():
    print("Scanning rgb and depth folders...")
    rgb_entries = scan_folder(RGB_DIR, "rgb")
    depth_entries = scan_folder(DEPTH_DIR, "depth")

    if not rgb_entries:
        print("Warning: No RGB images found in rgb/")
    else:
        write_list_file(RGB_TXT, rgb_entries)

    if not depth_entries:
        print("Warning: No depth images found in depth/")
    else:
        write_list_file(DEPTH_TXT, depth_entries)

    print("Done.")


if __name__ == "__main__":
    main()
