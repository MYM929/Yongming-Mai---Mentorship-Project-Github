# RGB-D Point Cloud Project

This project uses four Python scripts to turn RGB and depth images into a 3D point cloud

## Files in this project

- `1.make_file_lists.py` creates `rgb.txt` and `depth.txt`
- `2.pose_tracking.py` estimates camera motion
- `3.build_pointcloud.py` builds `pointcloud.ply`
- `4.visualize_pointcloud.py` opens the final point cloud

## What to install on a new computer

Install these before running the scripts:

1. Python
2. `pip` for Python packages
3. The Python libraries in `requirements.txt`

Recommended Python version:

- Python 3.10 or Python 3.11

## Python packages used

This project needs these packages:

- `numpy`
- `opencv-python`
- `open3d`

The scripts also use Python built-in modules like `os`, `sys`, `json`, `math`, `time`, `threading`, `re`, and `collections`, so you do not need to install those separately.

## First-time setup on Windows

Open PowerShell in this project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python` does not work, try:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Dataset folder needed

The scripts expect a `dataset` folder like this:

```text
dataset/
  calib/
    rgb_intrinsics.json
    depth_intrinsics.json
  imu/
    data.csv
  rgb/
    ...image files...
  depth/
    ...image files...
```

Important notes:

- RGB and depth image filenames should be numeric timestamps such as `20410725000.png`
- `2.pose_tracking.py` expects `dataset/calib/rgb_intrinsics.json`
- `2.pose_tracking.py` also expects `dataset/imu/data.csv`

## How to run the project

Run the scripts in this order:

```powershell
python 1.make_file_lists.py
python 2.pose_tracking.py
python 3.build_pointcloud.py
python 4.visualize_pointcloud.py
```

## Output files

After running the scripts, these files are created or used:

- `dataset/rgb.txt`
- `dataset/depth.txt`
- `dataset/pose_trajectory.txt`
- `dataset/pointcloud.ply`

## If something does not work

- Make sure the virtual environment is activated
- Make sure all packages were installed successfully
- Make sure the `dataset` folder contains `rgb`, `depth`, `calib`, and `imu`
- Make sure the image names are timestamps
- Make sure `pointcloud.ply` exists before running `4.visualize_pointcloud.py`
