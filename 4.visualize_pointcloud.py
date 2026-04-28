import open3d as o3d
import numpy as np
import os
import sys

# ================= Configuration =================
# Point to the dataset folder (same as 3. – where pointcloud.ply is saved)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "dataset")
PLY_FILENAME = "pointcloud.ply"
# =================================================

def main():
    ply_path = os.path.join(BASE_DIR, PLY_FILENAME)

    if not os.path.exists(ply_path):
        print(f"Error: Point cloud file not found at: {ply_path}")
        print("Please run '3.build_pointcloud.py' first to generate it.")
        return

    print(f"Loading point cloud from {ply_path}...")
    try:
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
    except Exception as e:
        print(f"Failed to load file using Open3D. Error: {e}")
        return

    if pcd.is_empty():
        print("Warning: The loaded point cloud is empty.")
        return

    n_pts = len(pcd.points)
    print(f"Loaded {n_pts} points.")

    # Remove extreme outliers so the view fits the real scene (not thousands of meters away)
    pts = np.asarray(pcd.points, dtype=np.float64)
    median = np.median(pts, axis=0)
    # Keep points within 2.5 m of median in each axis (scene is ~0.3–3 m)
    max_dist = 2.5
    in_range = np.abs(pts - median) <= max_dist
    mask = in_range[:, 0] & in_range[:, 1] & in_range[:, 2]
    pcd_display = pcd.select_by_index(np.where(mask)[0])
    n_display = len(pcd_display.points)
    if n_display == 0:
        pcd_display = pcd
        print("  No points in range; showing full cloud.")
    else:
        print(f"  Displaying {n_display} points (cropped to ±{max_dist} m of center for view).")

    center = np.asarray(pcd_display.get_center())
    extent = np.asarray(pcd_display.get_max_bound()) - np.asarray(pcd_display.get_min_bound())
    print(f"  View center: {center}, extent (m): {extent}")

    print("\nControls: [Left drag] Rotate  [Ctrl+drag] Pan  [Wheel] Zoom  [R] Reset view  [Q] Quit\n")

    # draw_geometries with lookat at scene center so the cloud is in view; [R] resets to fit
    o3d.visualization.draw_geometries(
        [pcd_display],
        window_name="RGB-D Point Cloud",
        width=1280,
        height=720,
        lookat=center,
        up=[0, -1, 0],
        front=[0, 0, -1],
        zoom=0.6,
    )

if __name__ == "__main__":
    main()