"""
2.pose_tracking.py – Jetson Nano Super Optimized RGB-D Odometry
================================================================
Depth-Enhanced IMU-Aided RGB-D Odometry with Pose-Graph Optimization,
tuned to run within the resource envelope of an NVIDIA Jetson Nano
Super developer kit (simulated on Windows).

JETSON NANO SUPER PROFILE
  Hardware : 8 GB shared CPU/GPU RAM  (~5.5 GB usable)
             6× ARM Cortex-A78AE cores
             NVIDIA GPU (unused – CPU-only pipeline)
  Disk     : NVMe / SD-card (random reads on demand)

OPTIMISATIONS vs DESKTOP VERSION
  ┌──────────────────────┬──────────────┬──────────────────┐
  │ Parameter            │ Desktop      │ Jetson (this)    │
  ├──────────────────────┼──────────────┼──────────────────┤
  │ Frame loading        │ All in RAM   │ On-demand + LRU  │
  │ Odometry resolution  │ 640×360      │ 320×180          │
  │ Sequential odo res   │ 320×180      │ 320×180 (same)   │
  │ Compute workers      │ up to 10     │ 2                │
  │ I/O workers          │ up to 10     │ 3                │
  │ ORB features/frame   │ 1 000        │ 500              │
  │ ORB query stride     │ 8            │ 12               │
  │ FPFH query stride    │ 20           │ 30               │
  │ FPFH voxel size      │ 0.08         │ 0.10             │
  │ Local LC gaps        │ [3, 5]       │ [3]              │
  │ Local LC stride      │ 4            │ 6                │
  │ RGBD cache           │ unlimited    │ 50 frames (LRU)  │
  │ PCD cache            │ unlimited    │ 40 frames (LRU)  │
  │ Peak RAM (est.)      │ 2–6 GB       │ < 500 MB         │
  └──────────────────────┴──────────────┴──────────────────┘

Run AFTER  1.make_file_lists.py
Run BEFORE 3.build_pointcloud.py
"""

import open3d as o3d
import numpy as np
import cv2
import csv
import json
import math
import os
import sys
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    cv2.setNumThreads(1)
except Exception:
    pass

# ====================== Jetson Nano Super Simulation ======================
JETSON_CORES = 6
JETSON_RAM_LIMIT_MB = 5500
JETSON_RAM_WARN_MB = 4000

# ====================== Configuration (Jetson-tuned) ======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")
DATASETS_ROOT = os.path.join(SCRIPT_DIR, "datasets")


def load_config():
    """Load the selected dataset folder from config.json."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"Missing config file: {CONFIG_PATH}. Create it with an 'active_dataset' entry."
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset_name = config.get("active_dataset")
    if not dataset_name:
        raise ValueError(f"'active_dataset' is missing or empty in {CONFIG_PATH}")

    base_dir = os.path.join(DATASETS_ROOT, dataset_name)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"Configured dataset folder does not exist: {base_dir}"
        )

    return dataset_name, base_dir


ACTIVE_DATASET, BASE_DIR = load_config()

# Camera intrinsics (full resolution)
WIDTH, HEIGHT = 1280, 720
FX, FY = 613.584, 613.542
CX, CY = 644.171, 355.251

# ---- Quarter-resolution for ALL odometry (Jetson) ----
ODO_SCALE = 0.25
ODO_W = int(WIDTH * ODO_SCALE)    # 320
ODO_H = int(HEIGHT * ODO_SCALE)   # 180

# Depth
DEPTH_SCALE = 1000.0
MAX_DEPTH   = 3.0
MIN_DEPTH   = 0.1
MAX_DEPTH_DIFF = 0.07

# Motion validation
MAX_VELOCITY    = 0.5    # m/s
MAX_ANGULAR_VEL = 2.5    # rad/s

# Segment detection
MAX_CONSECUTIVE_SKIP = 8

# ---- Local loop closures (reduced for Jetson) ----
LC_STRIDE = 6
LC_GAPS   = [3]

# ---- Global Loop Closure – ORB (reduced for Jetson) ----
GLC_MIN_INTERVAL    = 20
GLC_ORB_FEATURES    = 500
GLC_MATCH_RATIO     = 0.80
GLC_MIN_MATCHES     = 15
GLC_MIN_INLIERS     = 8
GLC_RANSAC_THRESH   = 4.0
GLC_QUERY_STRIDE    = 12
GLC_MAX_CANDIDATES  = 2
GLC_MAX_TRANSLATION = 2.0
GLC_MAX_ROTATION    = 1.5
GLC_ORB_PRESCREEN_K = 20
GLC_FAST_ONLY_ODO   = True

# ---- ICP verification ----
GLC_ICP_FITNESS_MIN = 0.15
GLC_HEAD_TAIL_N     = 6

# ---- Depth-ICP sequential fallback (fewer iterations for Jetson) ----
DEPTH_ICP_FALLBACK = True
ICP_SEQ_VOXELS = [
    (0.05, 0.15, 20),
    (0.03, 0.08, 15),
]
ICP_SEQ_FITNESS_MIN = 0.30

# ---- FPFH geometric loop closure (coarser for Jetson) ----
GLC_FPFH_ENABLED          = True
GLC_FPFH_VOXEL            = 0.10
GLC_FPFH_RADIUS_NORMAL    = 0.20
GLC_FPFH_RADIUS_FEAT      = 0.40
GLC_FPFH_QUERY_STRIDE     = 30
GLC_FPFH_RANSAC_DIST      = 0.06
GLC_FPFH_FITNESS_MIN      = 0.20
GLC_FPFH_MAX_CANDIDATES   = 2
GLC_FPFH_SPATIAL_MAX_DIST = 2.5
GLC_FPFH_SPATIAL_TOPK     = 20
GLC_FPFH_SKIP_ODO_REFINE  = True
GLC_FPFH_ORB_BACKFILL_ONLY = True

# Output / calibration paths
TRAJ_FILE = os.path.join(BASE_DIR, "pose_trajectory.txt")
RGB_INTRINSICS_JSON = os.path.join(BASE_DIR, "calib", "rgb_intrinsics.json")
IMU_CSV = os.path.join(BASE_DIR, "imu", "data.csv")

# ---- Jetson-class worker & cache limits ----
CPU_COUNT    = min(os.cpu_count() or 4, JETSON_CORES)
IO_WORKERS   = min(3, CPU_COUNT)
ODO_WORKERS  = 2
FPFH_WORKERS = 2
GLC_WORKERS  = 2

RGBD_CACHE_SIZE      = 50
PCD_CACHE_SIZE       = 40
PCD_LEVEL_CACHE_SIZE = 80
PNP_DEPTH_CACHE_SIZE = 15

VERBOSE_GLOBAL_EDGE_LOG = False
GLOBAL_EDGE_LOG_LIMIT   = 120

# Reused constants
EYE4    = np.eye(4)
ZERO66  = np.zeros((6, 6))
K_RGB_64 = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)


# ====================== Thread-safe bounded LRU cache ======================
_MISSING = object()


class LRUCache:
    """Thread-safe LRU cache with fixed capacity and hit-rate tracking."""

    def __init__(self, maxsize, name="cache"):
        self._maxsize = maxsize
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._name = name
        self._hits = 0
        self._misses = 0

    def get(self, key):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return _MISSING

    def put(self, key, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def stats(self):
        with self._lock:
            total = self._hits + self._misses
            rate = (self._hits / total * 100) if total else 0
            return (f"{self._name}: {self._hits}/{total} hits "
                    f"({rate:.0f}%), size={len(self._cache)}/{self._maxsize}")


# ====================== On-demand RGBD frame store ======================
class FrameStore:
    """Lazy-loading RGBD frames with bounded LRU eviction.

    Loads images from disk on first access, caches up to *cache_size*
    frames, and evicts least-recently-used entries when full.  This keeps
    peak memory proportional to the cache size (~50 frames ≈ 14 MB at
    320×180) instead of the dataset size (which can exceed available RAM
    on a Jetson for large captures).
    """

    def __init__(self, pairs, base_dir, undist, scale,
                 depth_scale, max_depth, cache_size, name="rgbd"):
        self._pairs = pairs
        self._base_dir = base_dir
        self._undist = undist
        self._scale = scale
        self._depth_scale = depth_scale
        self._max_depth = max_depth
        self._cache = LRUCache(cache_size, name)
        self._invalid = set()
        self._inv_lock = threading.Lock()

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._pairs):
            return None
        with self._inv_lock:
            if idx in self._invalid:
                return None
        cached = self._cache.get(idx)
        if cached is not _MISSING:
            return cached
        rgbd = self._load(idx)
        if rgbd is None:
            with self._inv_lock:
                self._invalid.add(idx)
            return None
        self._cache.put(idx, rgbd)
        return rgbd

    def _load(self, idx):
        _, rp, dp = self._pairs[idx]
        c = cv2.imread(os.path.join(self._base_dir, rp))
        d = cv2.imread(os.path.join(self._base_dir, dp), cv2.IMREAD_UNCHANGED)
        if c is None or d is None:
            return None
        if self._undist is not None:
            c = cv2.remap(c, self._undist[0], self._undist[1], cv2.INTER_LINEAR)
            d = cv2.remap(d, self._undist[0], self._undist[1], cv2.INTER_NEAREST)
        if self._scale != 1.0:
            nw = int(c.shape[1] * self._scale)
            nh = int(c.shape[0] * self._scale)
            c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA)
            d = cv2.resize(d, (nw, nh), interpolation=cv2.INTER_NEAREST)
        c_o3d = o3d.geometry.Image(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        d_o3d = o3d.geometry.Image(d.astype(np.uint16))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            c_o3d, d_o3d,
            depth_scale=self._depth_scale,
            depth_trunc=self._max_depth,
            convert_rgb_to_intensity=False)

    def stats(self):
        return self._cache.stats()


# ====================== Memory monitoring ======================
def get_memory_mb():
    """Current process RSS in MB (requires psutil; returns -1 otherwise)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1


def _mem_check(label=""):
    mem = get_memory_mb()
    if mem <= 0:
        return
    warn = ""
    if mem > JETSON_RAM_WARN_MB:
        warn = "  ** WARNING: approaching Jetson RAM limit **"
    print(f"  [MEM] {label}: {mem:.0f} MB / {JETSON_RAM_LIMIT_MB} MB{warn}")


# ====================== IMU helpers ======================
def load_imu_data():
    """Load IMU CSV.  Returns (data, times) or (None, None)."""
    if not os.path.exists(IMU_CSV):
        print("Warning: IMU data not found –", IMU_CSV)
        return None, None

    raw = {}
    with open(IMU_CSV) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            t_ns = int(parts[0])
            wx, wy, wz = float(parts[1]), float(parts[2]), float(parts[3])
            ax, ay, az = float(parts[4]), float(parts[5]), float(parts[6])
            if t_ns not in raw:
                raw[t_ns] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if abs(ax) + abs(ay) + abs(az) > 1e-9:
                raw[t_ns][3] = ax
                raw[t_ns][4] = ay
                raw[t_ns][5] = az
            if abs(wx) + abs(wy) + abs(wz) > 1e-9:
                raw[t_ns][0] = wx
                raw[t_ns][1] = wy
                raw[t_ns][2] = wz

    sorted_ts = sorted(raw.keys())
    data, times = [], []
    for t_ns in sorted_ts:
        d = raw[t_ns]
        t_sec = t_ns / 1e9
        data.append((t_sec, d[0], d[1], d[2], d[3], d[4], d[5]))
        times.append(t_sec)

    times = np.array(times)
    print(f"[OK] Loaded {len(data)} IMU samples "
          f"({times[0]:.3f}s – {times[-1]:.3f}s)")
    return data, times


def integrate_gyro(imu_data, imu_times, t_start, t_end):
    """Integrate gyroscope from *t_start* to *t_end*.  Returns 3×3 R."""
    if imu_data is None or imu_times is None:
        return np.eye(3)

    i0 = int(np.searchsorted(imu_times, t_start, side='left'))
    i1 = int(np.searchsorted(imu_times, t_end, side='right'))
    if i0 >= i1:
        return np.eye(3)

    R = np.eye(3)
    prev_t = t_start
    for i in range(i0, i1):
        t  = imu_data[i][0]
        wx = imu_data[i][1]
        wy = imu_data[i][2]
        wz = imu_data[i][3]
        dt = t - prev_t
        if dt <= 0 or dt > 0.1:
            prev_t = t
            continue
        norm_w = math.sqrt(wx * wx + wy * wy + wz * wz)
        angle = norm_w * dt
        if angle > 1e-8:
            ax_x, ax_y, ax_z = wx / norm_w, wy / norm_w, wz / norm_w
            K = np.array([
                [0,     -ax_z,  ax_y],
                [ax_z,   0,    -ax_x],
                [-ax_y,  ax_x,  0   ]
            ])
            sa, ca = math.sin(angle), math.cos(angle)
            dR = np.eye(3) + sa * K + (1.0 - ca) * (K @ K)
            R = R @ dR
        prev_t = t
    return R


# ====================== File / calibration helpers ======================
def load_undistort_maps():
    path = RGB_INTRINSICS_JSON
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        calib = json.load(f)
    w, h = int(calib["width"]), int(calib["height"])
    K = np.array([
        [float(calib["fx"]), 0, float(calib["cx"])],
        [0, float(calib["fy"]), float(calib["cy"])],
        [0, 0, 1]
    ], dtype=np.float64)
    dist = np.array([
        calib.get("k1", 0), calib.get("k2", 0),
        calib.get("p1", 0), calib.get("p2", 0),
        calib.get("k3", 0), calib.get("k4", 0),
        calib.get("k5", 0), calib.get("k6", 0),
    ], dtype=np.float64)
    m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
    return (m1, m2)


def load_frame_pairs():
    def read_list(fname):
        data = {}
        fp = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fp):
            print(f"Error: {fp} not found.")
            sys.exit(1)
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line or line[0] == '#':
                    continue
                parts = line.split()
                t = float(parts[0])
                if t > 1e10:
                    t /= 1e9
                data[t] = parts[1]
        return data

    rgb = read_list("rgb.txt")
    dep = read_list("depth.txt")
    rtimes = sorted(rgb.keys())
    dtimes = sorted(dep.keys())
    pairs = []
    j = 0
    for t_r in rtimes:
        while j + 1 < len(dtimes) and abs(dtimes[j + 1] - t_r) < abs(dtimes[j] - t_r):
            j += 1
        if abs(dtimes[j] - t_r) < 0.02:
            pairs.append((t_r, rgb[t_r], dep[dtimes[j]]))
    return pairs


def rotation_angle(R):
    """Rotation angle (rad) from a 3×3 rotation matrix."""
    return np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))


# ====================== ORB extraction worker ======================
def _extract_orb_worker(args):
    """Extract ORB features from a single frame at full resolution.

    Designed for ThreadPoolExecutor: loads one image, extracts, then lets
    the image be garbage-collected.  Peak memory per worker ≈ 3 MB.
    """
    idx, rp, base_dir, undist_maps, n_orb = args
    c = cv2.imread(os.path.join(base_dir, rp))
    if c is None:
        return idx, None, None
    if undist_maps is not None:
        c = cv2.remap(c, undist_maps[0], undist_maps[1], cv2.INTER_LINEAR)
    gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    cache = _extract_orb_worker.__dict__.setdefault("_det", {})
    key = (threading.get_ident(), n_orb)
    det = cache.get(key)
    if det is None:
        det = cv2.ORB_create(nfeatures=n_orb)
        cache[key] = det
    kps, des = det.detectAndCompute(gray, None)
    return idx, kps, des


# ====================== FPFH extraction ======================
def extract_fpfh(pcd, voxel_size=None, radius_normal=None, radius_feat=None):
    """FPFH descriptors for depth-only place recognition."""
    if voxel_size is None:
        voxel_size = GLC_FPFH_VOXEL
    if radius_normal is None:
        radius_normal = GLC_FPFH_RADIUS_NORMAL
    if radius_feat is None:
        radius_feat = GLC_FPFH_RADIUS_FEAT

    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) < 50:
        return None, None

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feat, max_nn=100))
    return pcd_down, fpfh


# ====================== PnP initial guess ======================
def compute_pnp_initial_guess(kps_src, kps_tgt, depth_src_path, undist,
                               good_matches, inlier_mask,
                               depth_lru=None):
    """SE(3) initial guess via PnP with LRU-cached depth maps."""
    depth = None
    if depth_lru is not None:
        cached = depth_lru.get(depth_src_path)
        if cached is not _MISSING:
            depth = cached
    if depth is None:
        d = cv2.imread(os.path.join(BASE_DIR, depth_src_path),
                       cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        if undist is not None:
            d = cv2.remap(d, undist[0], undist[1], cv2.INTER_NEAREST)
        depth = d.astype(np.float32) / DEPTH_SCALE
        if depth_lru is not None:
            depth_lru.put(depth_src_path, depth)

    inlier_matches = [m for m, flag in zip(good_matches, inlier_mask.ravel())
                      if flag]
    pts_3d, pts_2d = [], []
    for m in inlier_matches:
        u, v = kps_src[m.queryIdx].pt
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1]:
            z = depth[vi, ui]
            if MIN_DEPTH < z < MAX_DEPTH:
                pts_3d.append([(u - CX) * z / FX,
                               (v - CY) * z / FY,
                               z])
                pts_2d.append(kps_tgt[m.trainIdx].pt)

    if len(pts_3d) < 6:
        return None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.array(pts_3d, dtype=np.float64),
        np.array(pts_2d, dtype=np.float64),
        K_RGB_64, None,
        reprojectionError=3.0,
        iterationsCount=300,
        flags=cv2.SOLVEPNP_ITERATIVE)

    if not ok or inliers is None or len(inliers) < 6:
        return None

    R, _ = cv2.Rodrigues(rvec)
    T = EYE4.copy()
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


# ====================== Trajectory I/O ======================
def save_tum_trajectory(trajectory, filename):
    with open(filename, 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in trajectory:
            t = T[:3, 3]
            R = T[:3, :3]
            qw = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
            if qw > 1e-8:
                qx = (R[2, 1] - R[1, 2]) / (4 * qw)
                qy = (R[0, 2] - R[2, 0]) / (4 * qw)
                qz = (R[1, 0] - R[0, 1]) / (4 * qw)
            else:
                qx = qy = qz = 0.0
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")


def rotation_angle_deg(R):
    """Rotation angle (degrees) from a 3x3 rotation matrix."""
    try:
        return float(math.degrees(rotation_angle(np.asarray(R))))
    except Exception:
        return None


def safe_float(value):
    """Convert numeric values to JSON-safe floats, returning None if invalid."""
    if value is None:
        return None
    try:
        if isinstance(value, np.generic):
            value = value.item()
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def get_memory_mb_safe():
    """Current process RSS in MB, or None if unavailable."""
    mem = safe_float(get_memory_mb())
    if mem is None or mem <= 0:
        return None
    return mem


def compute_trajectory_motion_metrics(trajectory):
    metrics = {
        "total_path_length_m": 0.0,
        "mean_translation_step_m": None,
        "max_translation_step_m": None,
        "std_translation_step_m": None,
        "mean_rotation_step_deg": None,
        "max_rotation_step_deg": None,
        "std_rotation_step_deg": None,
        "mean_linear_velocity_mps": None,
        "max_linear_velocity_mps": None,
        "mean_angular_velocity_dps": None,
        "max_angular_velocity_dps": None,
        "head_tail_translation_drift_m": None,
        "head_tail_rotation_drift_deg": None,
    }
    if len(trajectory) == 0:
        return metrics

    first_T = np.asarray(trajectory[0][1])
    last_T = np.asarray(trajectory[-1][1])
    metrics["head_tail_translation_drift_m"] = safe_float(
        np.linalg.norm(last_T[:3, 3] - first_T[:3, 3]))
    metrics["head_tail_rotation_drift_deg"] = rotation_angle_deg(
        first_T[:3, :3].T @ last_T[:3, :3])

    if len(trajectory) < 2:
        return metrics

    translation_steps = []
    rotation_steps = []
    linear_velocities = []
    angular_velocities = []

    for (ts_prev, T_prev), (ts_cur, T_cur) in zip(trajectory[:-1],
                                                  trajectory[1:]):
        T_prev = np.asarray(T_prev)
        T_cur = np.asarray(T_cur)
        step_m = safe_float(np.linalg.norm(T_cur[:3, 3] - T_prev[:3, 3]))
        step_deg = rotation_angle_deg(T_prev[:3, :3].T @ T_cur[:3, :3])
        if step_m is not None:
            translation_steps.append(step_m)
        if step_deg is not None:
            rotation_steps.append(step_deg)
        dt = safe_float(ts_cur - ts_prev)
        if dt is not None and dt > 0:
            if step_m is not None:
                linear_velocities.append(step_m / dt)
            if step_deg is not None:
                angular_velocities.append(step_deg / dt)

    def _series_stats(values, prefix):
        if not values:
            return
        arr = np.asarray(values, dtype=np.float64)
        metrics[f"mean_{prefix}"] = safe_float(np.mean(arr))
        metrics[f"max_{prefix}"] = safe_float(np.max(arr))
        metrics[f"std_{prefix}"] = safe_float(np.std(arr))

    metrics["total_path_length_m"] = safe_float(sum(translation_steps)) or 0.0
    _series_stats(translation_steps, "translation_step_m")
    _series_stats(rotation_steps, "rotation_step_deg")
    if linear_velocities:
        arr = np.asarray(linear_velocities, dtype=np.float64)
        metrics["mean_linear_velocity_mps"] = safe_float(np.mean(arr))
        metrics["max_linear_velocity_mps"] = safe_float(np.max(arr))
    if angular_velocities:
        arr = np.asarray(angular_velocities, dtype=np.float64)
        metrics["mean_angular_velocity_dps"] = safe_float(np.mean(arr))
        metrics["max_angular_velocity_dps"] = safe_float(np.max(arr))
    return metrics


def build_parameter_snapshot():
    return {
        "ODO_SCALE": ODO_SCALE,
        "ODO_W": ODO_W,
        "ODO_H": ODO_H,
        "MAX_DEPTH": MAX_DEPTH,
        "MIN_DEPTH": MIN_DEPTH,
        "MAX_DEPTH_DIFF": MAX_DEPTH_DIFF,
        "MAX_VELOCITY": MAX_VELOCITY,
        "MAX_ANGULAR_VEL": MAX_ANGULAR_VEL,
        "LC_STRIDE": LC_STRIDE,
        "LC_GAPS": list(LC_GAPS),
        "GLC_ORB_FEATURES": GLC_ORB_FEATURES,
        "GLC_QUERY_STRIDE": GLC_QUERY_STRIDE,
        "GLC_FPFH_ENABLED": GLC_FPFH_ENABLED,
        "GLC_FPFH_VOXEL": GLC_FPFH_VOXEL,
        "GLC_FPFH_QUERY_STRIDE": GLC_FPFH_QUERY_STRIDE,
        "ICP_SEQ_FITNESS_MIN": ICP_SEQ_FITNESS_MIN,
        "ICP_SEQ_VOXELS": [list(v) for v in ICP_SEQ_VOXELS],
        "RGBD_CACHE_SIZE": RGBD_CACHE_SIZE,
        "PCD_CACHE_SIZE": PCD_CACHE_SIZE,
        "PCD_LEVEL_CACHE_SIZE": PCD_LEVEL_CACHE_SIZE,
        "ODO_WORKERS": ODO_WORKERS,
        "FPFH_WORKERS": FPFH_WORKERS,
        "GLC_WORKERS": GLC_WORKERS,
    }


def make_quality_judgment(metrics):
    accepted_ratio = safe_float(metrics.get("accepted_frame_ratio")) or 0.0
    num_segments = int(metrics.get("num_segments") or 0)
    drift_m = safe_float(metrics.get("head_tail_translation_drift_m"))
    runtime_per_frame = safe_float(metrics.get("runtime_per_accepted_frame_sec"))

    if accepted_ratio >= 0.95 and num_segments == 1:
        tracking = "Excellent"
    elif accepted_ratio >= 0.85:
        tracking = "Good"
    else:
        tracking = "Weak"

    if drift_m is None:
        drift = "Unknown"
    elif drift_m < 0.10:
        drift = "Low"
    elif drift_m < 0.30:
        drift = "Medium"
    else:
        drift = "High"

    if runtime_per_frame is None:
        runtime = "Unknown"
    elif runtime_per_frame < 0.5:
        runtime = "Fast"
    elif runtime_per_frame < 2.0:
        runtime = "Moderate"
    else:
        runtime = "Slow"

    if tracking == "Excellent" and drift == "Low" and runtime != "Slow":
        overall = "Excellent"
    elif tracking != "Weak" and drift not in ("High", "Unknown"):
        overall = "Good"
    else:
        overall = "Needs Review"

    return {
        "tracking_stability": tracking,
        "drift": drift,
        "runtime": runtime,
        "overall_run_quality": overall,
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _fmt(value, decimals=3, suffix=""):
    value = safe_float(value)
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def _fmt_int(value):
    try:
        if value is None:
            return "N/A"
        return str(int(value))
    except (TypeError, ValueError):
        return "N/A"


def save_metrics_report(metrics, base_dir):
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    timestamp = metrics.get("timestamp") or datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S")
    metrics["timestamp"] = timestamp
    file_stamp = timestamp.replace("-", "").replace(":", "").replace(" ", "_")

    metrics["quality_judgment"] = make_quality_judgment(metrics)
    for key, value in metrics["quality_judgment"].items():
        metrics[key] = value

    metrics = _json_ready(metrics)
    json_path = os.path.join(reports_dir, f"metrics_{file_stamp}.json")
    md_path = os.path.join(reports_dir, f"metrics_{file_stamp}.md")
    csv_path = os.path.join(reports_dir, "experiment_summary.csv")

    memory_values = [
        safe_float(metrics.get(k))
        for k in (
            "startup_memory_mb",
            "after_orb_memory_mb",
            "after_odometry_memory_mb",
            "after_global_lc_memory_mb",
            "final_memory_mb",
        )
    ]
    memory_values = [v for v in memory_values if v is not None]
    peak_memory = max(memory_values) if memory_values else None

    params = metrics.get("parameters", {})
    judgment = metrics.get("quality_judgment", {})
    phase_rows = [
        ("ORB extraction", "orb_extraction_time_sec"),
        ("Phase 1 sequential tracking", "phase1_sequential_tracking_time_sec"),
        ("Phase 2 local LC", "phase2_local_lc_time_sec"),
        ("Phase 2b global LC", "phase2b_global_lc_time_sec"),
        ("Phase 3 optimization", "phase3_optimization_time_sec"),
        ("Total wall time", "total_wall_time_sec"),
    ]

    md_lines = [
        "# Pose Trajectory Experiment Report",
        "",
        "## Summary",
        f"- Dataset: {metrics.get('active_dataset', 'N/A')}",
        f"- Timestamp: {timestamp}",
        f"- Accepted frame ratio: {_fmt(metrics.get('accepted_frame_ratio'))}",
        f"- Number of segments: {_fmt_int(metrics.get('num_segments'))}",
        f"- Head-tail drift: {_fmt(metrics.get('head_tail_translation_drift_m'), 3, ' m')} / "
        f"{_fmt(metrics.get('head_tail_rotation_drift_deg'), 2, ' deg')}",
        f"- Total path length: {_fmt(metrics.get('total_path_length_m'), 3, ' m')}",
        f"- Total runtime: {_fmt(metrics.get('total_wall_time_sec'), 2, ' s')}",
        f"- Peak/final memory: {_fmt(peak_memory, 1, ' MB')} / "
        f"{_fmt(metrics.get('final_memory_mb'), 1, ' MB')}",
        f"- Trajectory file: {metrics.get('trajectory_file', 'N/A')}",
        "",
        "## Tracking Quality",
        f"- Successful odometry edges: {_fmt_int(metrics.get('successful_odometry_edges'))}",
        f"- Sequential success rate: {_fmt(metrics.get('sequential_success_rate'))}",
        f"- ICP fallback rate: {_fmt(metrics.get('icp_fallback_rate'))}",
        f"- IMU used count: {_fmt_int(metrics.get('imu_used_count'))}",
        f"- Segments: {_fmt_int(metrics.get('num_segments'))}",
        f"- Largest segment length: {_fmt_int(metrics.get('largest_segment_length'))}",
        "",
        "## Motion Quality",
        f"- Translation step mean/max/std: "
        f"{_fmt(metrics.get('mean_translation_step_m'), 4, ' m')} / "
        f"{_fmt(metrics.get('max_translation_step_m'), 4, ' m')} / "
        f"{_fmt(metrics.get('std_translation_step_m'), 4, ' m')}",
        f"- Rotation step mean/max/std: "
        f"{_fmt(metrics.get('mean_rotation_step_deg'), 3, ' deg')} / "
        f"{_fmt(metrics.get('max_rotation_step_deg'), 3, ' deg')} / "
        f"{_fmt(metrics.get('std_rotation_step_deg'), 3, ' deg')}",
        f"- Linear velocity mean/max: "
        f"{_fmt(metrics.get('mean_linear_velocity_mps'), 4, ' m/s')} / "
        f"{_fmt(metrics.get('max_linear_velocity_mps'), 4, ' m/s')}",
        f"- Angular velocity mean/max: "
        f"{_fmt(metrics.get('mean_angular_velocity_dps'), 3, ' deg/s')} / "
        f"{_fmt(metrics.get('max_angular_velocity_dps'), 3, ' deg/s')}",
        "",
        "## Loop Closure Quality",
        f"- Local LC: {_fmt_int(metrics.get('local_lc_accepted'))}/"
        f"{_fmt_int(metrics.get('local_lc_tried'))}, "
        f"rate {_fmt(metrics.get('local_lc_accept_rate'))}",
        f"- ORB global LC: {_fmt_int(metrics.get('orb_glc_accepted'))}/"
        f"{_fmt_int(metrics.get('orb_glc_tried'))}, "
        f"rate {_fmt(metrics.get('orb_glc_accept_rate'))}",
        f"- FPFH global LC: {_fmt_int(metrics.get('fpfh_glc_accepted'))}/"
        f"{_fmt_int(metrics.get('fpfh_glc_tried'))}, "
        f"rate {_fmt(metrics.get('fpfh_glc_accept_rate'))}",
        f"- Head-tail LC added: {_fmt_int(metrics.get('head_tail_lc_added'))}",
        f"- Total global LC added: {_fmt_int(metrics.get('total_global_lc_added'))}",
        "",
        "## Pose Graph",
        f"- Nodes: {_fmt_int(metrics.get('num_pose_graph_nodes'))}",
        f"- Edges: {_fmt_int(metrics.get('num_pose_graph_edges'))}",
        f"- Odometry edges: {_fmt_int(metrics.get('num_odometry_edges'))}",
        f"- Uncertain edges: {_fmt_int(metrics.get('num_uncertain_edges'))}",
        "",
        "## Runtime",
        "| Phase | Seconds |",
        "|---|---:|",
    ]
    for label, key in phase_rows:
        md_lines.append(f"| {label} | {_fmt(metrics.get(key), 2)} |")

    md_lines.extend([
        "",
        "## Memory and Cache",
        "| Measurement | MB |",
        "|---|---:|",
        f"| Startup | {_fmt(metrics.get('startup_memory_mb'), 1)} |",
        f"| After ORB | {_fmt(metrics.get('after_orb_memory_mb'), 1)} |",
        f"| After odometry | {_fmt(metrics.get('after_odometry_memory_mb'), 1)} |",
        f"| After global LC | {_fmt(metrics.get('after_global_lc_memory_mb'), 1)} |",
        f"| Final | {_fmt(metrics.get('final_memory_mb'), 1)} |",
        "",
        f"- RGBD cache: {metrics.get('rgbd_cache_stats', 'N/A')}",
        f"- PCD cache: {metrics.get('pcd_cache_stats', 'N/A')}",
        f"- PCD level cache: {metrics.get('pcd_level_cache_stats', 'N/A')}",
        f"- PnP depth cache: {metrics.get('pnp_depth_cache_stats', 'N/A')}",
        "",
        "## Parameter Snapshot",
        "| Parameter | Value |",
        "|---|---|",
    ])
    for key in sorted(params.keys()):
        md_lines.append(f"| {key} | `{json.dumps(params[key])}` |")

    md_lines.extend([
        "",
        "## Quick Judgment",
        f"- Tracking stability: {judgment.get('tracking_stability', 'N/A')}",
        f"- Drift: {judgment.get('drift', 'N/A')}",
        f"- Runtime: {judgment.get('runtime', 'N/A')}",
        f"- Overall run quality: {judgment.get('overall_run_quality', 'N/A')}",
        "",
    ])

    csv_fields = [
        "timestamp",
        "active_dataset",
        "total_frames",
        "accepted_frames",
        "accepted_frame_ratio",
        "num_segments",
        "largest_segment_length",
        "head_tail_translation_drift_m",
        "head_tail_rotation_drift_deg",
        "total_path_length_m",
        "successful_odometry_edges",
        "sequential_success_rate",
        "imu_used_count",
        "icp_fallback_count",
        "icp_fallback_rate",
        "local_lc_accept_rate",
        "orb_glc_accept_rate",
        "fpfh_glc_accept_rate",
        "total_global_lc_added",
        "total_wall_time_sec",
        "runtime_per_accepted_frame_sec",
        "final_memory_mb",
        "tracking_stability",
        "drift",
        "runtime",
        "overall_run_quality",
        "trajectory_file",
        "json_report",
        "markdown_report",
    ]

    metrics["json_report"] = json_path
    metrics["markdown_report"] = md_path
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()
        row = {key: metrics.get(key) for key in csv_fields}
        writer.writerow(row)

    return {"json": json_path, "markdown": md_path, "csv": csv_path}


# ==================================================================
def main():
    wall_start = time.time()
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active_dataset": ACTIVE_DATASET,
        "trajectory_file": TRAJ_FILE,
        "parameters": build_parameter_snapshot(),
        "startup_memory_mb": get_memory_mb_safe(),
    }
    print(f"  Active dataset: {ACTIVE_DATASET}")

    print("=" * 62)
    print("  JETSON NANO SUPER – Optimised RGB-D Odometry + Pose Graph")
    print(f"  Simulated constraints: {JETSON_CORES} cores, "
          f"{JETSON_RAM_LIMIT_MB} MB usable RAM")
    print(f"  Odometry resolution : {ODO_W}×{ODO_H} "
          f"(scale {ODO_SCALE})")
    print(f"  Workers : IO={IO_WORKERS}  ODO={ODO_WORKERS}  "
          f"FPFH={FPFH_WORKERS}  GLC={GLC_WORKERS}")
    print(f"  Caches  : RGBD={RGBD_CACHE_SIZE}  "
          f"PCD={PCD_CACHE_SIZE}  PCD_LVL={PCD_LEVEL_CACHE_SIZE}")
    print("=" * 62)
    _mem_check("startup")
    print()

    # ---- Single (quarter-res) intrinsic for all odometry ----
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        ODO_W, ODO_H,
        FX * ODO_SCALE, FY * ODO_SCALE,
        CX * ODO_SCALE, CY * ODO_SCALE)

    undist = load_undistort_maps()
    if undist:
        print("[OK] Loaded lens undistortion maps.")

    imu_data, imu_times = load_imu_data()

    pairs = load_frame_pairs()
    n = len(pairs)
    metrics["total_frames"] = n
    print(f"[OK] {n} RGB-D frame pairs loaded.\n")
    if n < 2:
        print("Need at least 2 frames.")
        metrics.update({
            "accepted_frames": 0,
            "accepted_frame_ratio": 0.0,
            "num_segments": 0,
            "largest_segment_length": 0,
            "successful_odometry_edges": 0,
            "sequential_success_rate": 0.0,
            "total_wall_time_sec": time.time() - wall_start,
        })
        report_paths = save_metrics_report(metrics, SCRIPT_DIR)
        print(f"Reports saved to {report_paths['json']} and "
              f"{report_paths['markdown']}")
        return

    # ---- On-demand RGBD frame store (bounded LRU) ----
    rgbd_store = FrameStore(
        pairs, BASE_DIR, undist, ODO_SCALE,
        DEPTH_SCALE, MAX_DEPTH, RGBD_CACHE_SIZE, "rgbd")

    # ---- Lightweight ORB extraction pass ----
    # Reads each image once at full resolution for ORB quality, stores
    # only the compact keypoints + descriptors (~100 KB/frame), then
    # lets the full-res image be garbage-collected immediately.
    print(f"Pre-extracting ORB features ({GLC_ORB_FEATURES}/frame, "
          f"{IO_WORKERS} workers)...")
    t_orb = time.time()
    orb_kps = {}
    orb_des = {}
    tasks = [(i, pairs[i][1], BASE_DIR, undist, GLC_ORB_FEATURES)
             for i in range(n)]
    done = 0
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        for idx, kps, des in pool.map(_extract_orb_worker, tasks):
            orb_kps[idx] = kps
            orb_des[idx] = des
            done += 1
            if done % 100 == 0 or done == n:
                print(f"  {done}/{n}")
    n_feat = sum(1 for d in orb_des.values() if d is not None)
    orb_extraction_time = time.time() - t_orb
    print(f"  {n_feat}/{n} frames with ORB features  "
          f"({orb_extraction_time:.1f}s)")
    metrics["orb_extraction_time_sec"] = orb_extraction_time
    metrics["after_orb_memory_mb"] = get_memory_mb_safe()
    _mem_check("after ORB extraction")
    print()

    # ---- Bounded point-cloud caches ----
    pcd_cache = LRUCache(PCD_CACHE_SIZE, "pcd_frame")
    pcd_level_cache = LRUCache(PCD_LEVEL_CACHE_SIZE, "pcd_level")
    pnp_depth_lru = LRUCache(PNP_DEPTH_CACHE_SIZE, "pnp_depth")

    def get_pcd_frame(frame_idx):
        cached = pcd_cache.get(frame_idx)
        if cached is not _MISSING:
            return cached
        rgbd = rgbd_store[frame_idx]
        if rgbd is not None:
            pc = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic)
            value = pc if len(pc.points) > 0 else None
        else:
            value = None
        pcd_cache.put(frame_idx, value)
        return value

    def get_pcd_level(frame_idx, voxel_size):
        key = (frame_idx, float(voxel_size))
        cached = pcd_level_cache.get(key)
        if cached is not _MISSING:
            return cached
        base = get_pcd_frame(frame_idx)
        if base is None:
            value = None
        else:
            lvl = base.voxel_down_sample(voxel_size)
            if len(lvl.points) < 50:
                value = None
            else:
                lvl.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=voxel_size * 4, max_nn=30))
                value = lvl
        pcd_level_cache.put(key, value)
        return value

    # ---- Odometry helpers ----
    def _make_odo_option(fast_mode=False):
        opt = o3d.pipelines.odometry.OdometryOption(
            depth_diff_max=MAX_DEPTH_DIFF,
            depth_min=MIN_DEPTH,
            depth_max=MAX_DEPTH,
        )
        if fast_mode:
            opt.iteration_number_per_pyramid_level = (
                o3d.utility.IntVector([6, 4, 3]))
        else:
            opt.iteration_number_per_pyramid_level = (
                o3d.utility.IntVector([10, 5, 3]))
        return opt

    def _get_odo_ctx(fast_mode=False):
        cache = _get_odo_ctx.__dict__.setdefault("cache", {})
        key = (threading.get_ident(), bool(fast_mode))
        if key not in cache:
            cache[key] = (
                _make_odo_option(fast_mode),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            )
        return cache[key]

    def try_odometry(src, tgt, init=EYE4, fast_mode=False):
        src_rgbd = rgbd_store[src]
        tgt_rgbd = rgbd_store[tgt]
        if src_rgbd is None or tgt_rgbd is None:
            return False, EYE4.copy(), ZERO66.copy()
        odo_option, jacobian = _get_odo_ctx(fast_mode)
        ok, T, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            src_rgbd, tgt_rgbd, intrinsic, init, jacobian, odo_option)
        return ok, T, info

    def try_odometry_fast_then_full(src, tgt, init=EYE4, full_retry=True):
        ok, T, info = try_odometry(src, tgt, init, fast_mode=True)
        if ok or not full_retry:
            return ok, T, info
        return try_odometry(src, tgt, init, fast_mode=False)

    def validate_motion(trans, dt):
        if dt < 0.005:
            return False
        t_norm = np.linalg.norm(trans[:3, 3])
        angle = rotation_angle(trans[:3, :3])
        if t_norm / dt > MAX_VELOCITY:
            return False
        if angle / dt > MAX_ANGULAR_VEL:
            return False
        return True

    def imu_init_guess(t_src, t_tgt):
        R_body = integrate_gyro(imu_data, imu_times, t_src, t_tgt)
        init = EYE4.copy()
        init[:3, :3] = R_body.T
        return init

    def try_icp_sequential(src_idx, tgt_idx, init):
        """Point-to-plane ICP fallback for sequential tracking."""
        if get_pcd_frame(src_idx) is None or get_pcd_frame(tgt_idx) is None:
            return False, EYE4.copy(), ZERO66.copy()

        current_T = init.copy()
        fitness = 0.0
        for vs, max_d, max_it in ICP_SEQ_VOXELS:
            src_d = get_pcd_level(src_idx, vs)
            tgt_d = get_pcd_level(tgt_idx, vs)
            if src_d is None or tgt_d is None:
                continue
            result = o3d.pipelines.registration.registration_icp(
                src_d, tgt_d, max_d, current_T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_it))
            current_T = result.transformation
            fitness = result.fitness

        if fitness < ICP_SEQ_FITNESS_MIN:
            return False, EYE4.copy(), ZERO66.copy()

        tn = np.linalg.norm(current_T[:3, 3])
        an = rotation_angle(current_T[:3, :3])
        if tn > 0.8 or an > 1.2:
            return False, EYE4.copy(), ZERO66.copy()

        ok_odo, trans_odo, info_odo = try_odometry(
            src_idx, tgt_idx, current_T, fast_mode=True)
        if ok_odo:
            return True, trans_odo, info_odo

        info = np.eye(6) * max(fitness * 8000, 400)
        return True, current_T, info

    # ==============================================================
    # Phase 1 – Greedy sequential tracking (IMU-aided, ICP fallback)
    # ==============================================================
    print("Phase 1: Sequential tracking "
          f"(Jetson: {ODO_WORKERS} workers, {ODO_W}×{ODO_H})...")
    t_phase1 = time.time()

    def _compute_consec_odo(i):
        init = imu_init_guess(pairs[i - 1][0], pairs[i][0])
        ok, trans, info = try_odometry(i - 1, i, init, fast_mode=True)
        used_icp = False
        if not ok and DEPTH_ICP_FALLBACK:
            ok, trans, info = try_icp_sequential(i - 1, i, init)
            used_icp = ok
        return i, ok, trans, info, used_icp

    # Submit all consecutive pairs; only ODO_WORKERS (2) run at a time
    # so the FrameStore LRU cache (50 frames) easily contains the
    # ~4 concurrently-needed frames.
    submit_list = list(range(1, n))
    consec_results = {}
    with ThreadPoolExecutor(max_workers=ODO_WORKERS) as pool:
        futures = {pool.submit(_compute_consec_odo, i): i
                   for i in submit_list}
        done_count = 0
        for fut in as_completed(futures):
            i, ok, trans, info, used_icp = fut.result()
            consec_results[i] = (ok, trans, info, used_icp)
            done_count += 1
            if done_count % 100 == 0 or done_count == len(submit_list):
                print(f"  odometry: {done_count}/{len(submit_list)} pairs")
    t_odo_parallel = time.time() - t_phase1
    metrics["after_odometry_memory_mb"] = get_memory_mb_safe()
    _mem_check("after parallel odometry")

    # Chain poses sequentially using pre-computed transforms
    segments = [[0]]
    poses = {0: np.eye(4)}
    odo_edges = []
    skip_run = 0
    imu_used = 0
    icp_used = 0

    for i in range(1, n):
        seg = segments[-1]
        connected = False
        last_accepted = seg[-1]

        if last_accepted == i - 1 and i in consec_results:
            ok, trans, info, used_icp = consec_results[i]
            if ok:
                dt = pairs[i][0] - pairs[last_accepted][0]
                if validate_motion(trans, dt):
                    poses[i] = poses[last_accepted] @ np.linalg.inv(trans)
                    seg.append(i)
                    odo_edges.append((last_accepted, i, trans, info))
                    connected = True
                    skip_run = 0
                    if imu_data is not None:
                        imu_used += 1
                    if used_icp:
                        icp_used += 1

        if not connected:
            for back in range(min(3, len(seg))):
                last = seg[-(back + 1)]
                if last == i - 1 and i in consec_results:
                    continue
                init = imu_init_guess(pairs[last][0], pairs[i][0])
                if imu_data is not None:
                    imu_used += 1
                ok, trans, info = try_odometry(last, i, init, fast_mode=True)
                if not ok and DEPTH_ICP_FALLBACK:
                    ok, trans, info = try_icp_sequential(last, i, init)
                    if ok:
                        icp_used += 1
                if not ok:
                    continue
                dt = pairs[i][0] - pairs[last][0]
                if not validate_motion(trans, dt):
                    continue
                poses[i] = poses[last] @ np.linalg.inv(trans)
                seg.append(i)
                odo_edges.append((last, i, trans, info))
                connected = True
                skip_run = 0
                break

        if not connected:
            skip_run += 1
            if skip_run >= MAX_CONSECUTIVE_SKIP:
                poses[i] = np.eye(4)
                segments.append([i])
                skip_run = 0

    total_accepted = sum(len(s) for s in segments)
    successful_odometry_edges = len(odo_edges)
    phase1_time = time.time() - t_phase1
    print(f"  Accepted {total_accepted}/{n} frames in {len(segments)} segment(s)  "
          f"(IMU {imu_used}×, ICP fallback {icp_used}×)  "
          f"({phase1_time:.1f}s, parallel odo {t_odo_parallel:.1f}s)")
    if icp_used > 0:
        print(f"  ** {icp_used} frame(s) recovered by depth-only ICP **")
    print()

    best_seg_idx = max(range(len(segments)), key=lambda k: len(segments[k]))
    best_seg = segments[best_seg_idx]
    if len(segments) > 1:
        print(f"  Using segment {best_seg_idx + 1} ({len(best_seg)} frames).")
        seg_set = set(best_seg)
        odo_edges = [(a, b, t, inf) for a, b, t, inf in odo_edges
                     if a in seg_set and b in seg_set]

    accepted = best_seg
    n_acc = len(accepted)
    metrics.update({
        "accepted_frames": n_acc,
        "accepted_frame_ratio": n_acc / n if n else 0.0,
        "num_segments": len(segments),
        "largest_segment_length": max((len(s) for s in segments), default=0),
        "successful_odometry_edges": successful_odometry_edges,
        "sequential_success_rate": (
            successful_odometry_edges / (n - 1) if n > 1 else 0.0),
        "imu_used_count": imu_used,
        "icp_fallback_count": icp_used,
        "icp_fallback_rate": (
            icp_used / successful_odometry_edges
            if successful_odometry_edges > 0 else 0.0),
        "phase1_sequential_tracking_time_sec": phase1_time,
    })
    print(f"  Building pose graph for {n_acc} accepted frames.\n")

    # ==============================================================
    # Phase 2 – Local loop-closure edges
    # ==============================================================
    local_lc_gaps = list(LC_GAPS)
    if len(segments) == 1 and n_acc >= 600 and total_accepted == n:
        local_lc_gaps = [min(LC_GAPS)]

    print(f"Phase 2: Local loop closures "
          f"(stride={LC_STRIDE}, gaps={local_lc_gaps})...")
    t_phase2 = time.time()
    idx_to_node = {accepted[j]: j for j in range(n_acc)}

    pose_graph = o3d.pipelines.registration.PoseGraph()
    for j in range(n_acc):
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(poses[accepted[j]]))

    for (a, b, trans, info) in odo_edges:
        if a in idx_to_node and b in idx_to_node:
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    idx_to_node[a], idx_to_node[b],
                    trans, info, uncertain=False))

    lc_candidates = []
    for j in range(0, n_acc, LC_STRIDE):
        for gap in local_lc_gaps:
            k = j + gap
            if k >= n_acc:
                continue
            s_idx = accepted[j]
            t_idx = accepted[k]
            init = np.linalg.inv(poses[t_idx]) @ poses[s_idx]
            dt = pairs[t_idx][0] - pairs[s_idx][0]
            lc_candidates.append((j, k, s_idx, t_idx, init, dt))
    lc_tried = len(lc_candidates)

    def _eval_local_lc(cand):
        j, k, s_idx, t_idx, init, dt = cand
        ok, trans, info = try_odometry(s_idx, t_idx, init, fast_mode=True)
        if not ok:
            return None
        if not validate_motion(trans, dt):
            return None
        return (j, k, trans, info)

    lc_edges = []
    with ThreadPoolExecutor(max_workers=ODO_WORKERS) as pool:
        for out in pool.map(_eval_local_lc, lc_candidates):
            if out is not None:
                lc_edges.append(out)

    lc_edges.sort(key=lambda x: (x[0], x[1]))
    for j, k, trans, info in lc_edges:
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                j, k, trans, info, uncertain=True))
    lc_ok = len(lc_edges)
    phase2_time = time.time() - t_phase2
    metrics.update({
        "local_lc_tried": lc_tried,
        "local_lc_accepted": lc_ok,
        "local_lc_accept_rate": lc_ok / lc_tried if lc_tried else 0.0,
        "phase2_local_lc_time_sec": phase2_time,
    })

    print(f"  Local loop closures: {lc_ok}/{lc_tried} accepted  "
          f"({phase2_time:.1f}s)\n")

    # ==============================================================
    # Phase 2b – Global Loop Closure (ORB + FPFH + head-tail ICP)
    # ==============================================================
    print("Phase 2b: Global loop closure (ORB + FPFH depth)...")
    t_phase2b = time.time()

    def get_pcd(node_j):
        return get_pcd_frame(accepted[node_j])

    def try_icp_edge(q_node, t_node, init,
                     refine_with_odometry=True,
                     odometry_full_retry=False):
        pcd_s = get_pcd(q_node)
        pcd_t = get_pcd(t_node)
        if pcd_s is None or pcd_t is None:
            return False, None, None
        s_idx = accepted[q_node]
        t_idx = accepted[t_node]
        current_T = init.copy()
        fitness = 0.0
        for vs, max_d, max_it in [(0.08, 0.40, 20), (0.04, 0.15, 12)]:
            src_d = get_pcd_level(s_idx, vs)
            tgt_d = get_pcd_level(t_idx, vs)
            if src_d is None or tgt_d is None:
                continue
            result = o3d.pipelines.registration.registration_icp(
                src_d, tgt_d, max_d, current_T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_it))
            current_T = result.transformation
            fitness = result.fitness
        T_icp = current_T
        if fitness < GLC_ICP_FITNESS_MIN:
            return False, None, None
        tn = np.linalg.norm(T_icp[:3, 3])
        an = rotation_angle(T_icp[:3, :3])
        if tn > GLC_MAX_TRANSLATION or an > GLC_MAX_ROTATION:
            return False, None, None
        if not refine_with_odometry:
            info_icp = np.eye(6) * max(fitness * 10000, 500)
            return True, T_icp, info_icp
        ok, trans, info = try_odometry_fast_then_full(
            s_idx, t_idx, T_icp, full_retry=odometry_full_retry)
        if ok:
            tn2 = np.linalg.norm(trans[:3, 3])
            an2 = rotation_angle(trans[:3, :3])
            if tn2 <= GLC_MAX_TRANSLATION and an2 <= GLC_MAX_ROTATION:
                return True, trans, info
        info_icp = np.eye(6) * max(fitness * 10000, 500)
        return True, T_icp, info_icp

    # ---- Map pre-extracted ORB features to accepted-frame nodes ----
    print("  Mapping ORB features to accepted frames...")
    kps_db = [None] * n_acc
    des_db = [None] * n_acc
    for j in range(n_acc):
        frame_idx = accepted[j]
        kps_db[j] = orb_kps.get(frame_idx)
        des_db[j] = orb_des.get(frame_idx)
    n_feat_acc = sum(1 for d in des_db if d is not None)
    print(f"  {n_feat_acc}/{n_acc} accepted frames with ORB features")
    del orb_kps, orb_des

    # ---- Compact ORB summaries for fast pre-screening ----
    orb_compact = [None] * n_acc
    compact_list = []
    compact_idx = []
    for j in range(n_acc):
        if des_db[j] is not None:
            csum = des_db[j].astype(np.float32).mean(axis=0)
            orb_compact[j] = csum
            compact_list.append(csum)
            compact_idx.append(j)
    compact_matrix = (np.array(compact_list, dtype=np.float32)
                      if compact_list
                      else np.empty((0, 32), dtype=np.float32))
    compact_idx_arr = (np.array(compact_idx)
                       if compact_idx
                       else np.array([], dtype=int))
    compact_matrix_sqnorm = (
        np.einsum("ij,ij->i", compact_matrix, compact_matrix)
        if compact_matrix.size > 0
        else np.empty((0,), dtype=np.float32))

    def _get_bf_matcher():
        cache = _get_bf_matcher.__dict__.setdefault("cache", {})
        tid = threading.get_ident()
        if tid not in cache:
            cache[tid] = cv2.BFMatcher(cv2.NORM_HAMMING)
        return cache[tid]

    pose_inv = {idx: np.linalg.inv(poses[idx]) for idx in accepted}
    glc_ok = 0
    glc_tried = 0
    glc_edges = []
    orb_closed_queries = set()

    query_nodes = [q for q in range(0, n_acc, GLC_QUERY_STRIDE)
                   if des_db[q] is not None and orb_compact[q] is not None]

    def _eval_orb_query(q):
        bf = _get_bf_matcher()
        local_edges = []
        local_tried = 0
        local_closed = False

        q_vec = orb_compact[q]
        dists = (compact_matrix_sqnorm
                 + float(np.dot(q_vec, q_vec))
                 - 2.0 * (compact_matrix @ q_vec))
        temporal_mask = np.abs(compact_idx_arr - q) >= GLC_MIN_INTERVAL
        dists[~temporal_mask] = np.inf
        k = min(GLC_ORB_PRESCREEN_K, dists.shape[0])
        if k == 0:
            return q, local_tried, local_closed, local_edges
        top_k = np.argpartition(dists, k - 1)[:k]
        candidates = [int(compact_idx_arr[i]) for i in top_k
                      if np.isfinite(dists[i])]

        scored = []
        for t in candidates:
            try:
                raw = bf.knnMatch(des_db[q], des_db[t], k=2)
            except cv2.error:
                continue
            good = [p[0] for p in raw
                    if len(p) == 2
                    and p[0].distance < GLC_MATCH_RATIO * p[1].distance]
            if len(good) >= GLC_MIN_MATCHES:
                scored.append((t, good))

        scored.sort(key=lambda x: len(x[1]), reverse=True)

        for (t_node, good_matches) in scored[:GLC_MAX_CANDIDATES]:
            local_tried += 1

            pts_q = np.float32([kps_db[q][m.queryIdx].pt
                                for m in good_matches])
            pts_t = np.float32([kps_db[t_node][m.trainIdx].pt
                                for m in good_matches])
            F, mask = cv2.findFundamentalMat(
                pts_q, pts_t, cv2.FM_RANSAC, GLC_RANSAC_THRESH)
            if F is None or mask is None:
                continue
            n_inliers = int(mask.ravel().sum())
            if n_inliers < GLC_MIN_INLIERS:
                continue

            s_idx = accepted[q]
            t_idx = accepted[t_node]
            pnp_T = compute_pnp_initial_guess(
                kps_db[q], kps_db[t_node],
                pairs[s_idx][2], undist, good_matches, mask,
                pnp_depth_lru)
            init = (pnp_T if pnp_T is not None
                    else (pose_inv[t_idx] @ poses[s_idx]))

            ok, trans, info = try_odometry_fast_then_full(
                s_idx, t_idx, init,
                full_retry=(not GLC_FAST_ONLY_ODO))
            if ok:
                tn = np.linalg.norm(trans[:3, 3])
                an = rotation_angle(trans[:3, :3])
                if tn <= GLC_MAX_TRANSLATION and an <= GLC_MAX_ROTATION:
                    local_edges.append(
                        (q, t_node, trans, info, n_inliers, "odo"))
                    local_closed = True
                    continue

            ok_icp, T_icp, info_icp = try_icp_edge(
                q, t_node, init,
                refine_with_odometry=True,
                odometry_full_retry=(not GLC_FAST_ONLY_ODO))
            if ok_icp:
                local_edges.append(
                    (q, t_node, T_icp, info_icp, n_inliers, "icp"))
                local_closed = True

        return q, local_tried, local_closed, local_edges

    orb_results = []
    with ThreadPoolExecutor(max_workers=GLC_WORKERS) as pool:
        for out in pool.map(_eval_orb_query, query_nodes):
            orb_results.append(out)

    orb_results.sort(key=lambda x: x[0])
    for q, local_tried, local_closed, local_edges in orb_results:
        glc_tried += local_tried
        if local_closed:
            orb_closed_queries.add(q)
        for a, b, trans, info, n_inliers, method in local_edges:
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    a, b, trans, info, uncertain=True))
            glc_ok += 1
            glc_edges.append((a, b, n_inliers, method))

    print(f"  ORB-based global LC: {glc_ok}/{glc_tried} verified & added")
    _mem_check("after ORB global LC")

    # ---- FPFH geometric loop closure ----
    fpfh_ok = 0
    fpfh_tried = 0

    if GLC_FPFH_ENABLED:
        print("  Extracting FPFH features (depth-based place recognition)...")

        pose_positions = np.array(
            [poses[accepted[j]][:3, 3] for j in range(n_acc)])
        if GLC_FPFH_ORB_BACKFILL_ONLY:
            fpfh_queries = [q for q in range(0, n_acc, GLC_FPFH_QUERY_STRIDE)
                            if q not in orb_closed_queries]
            if len(fpfh_queries) == 0:
                fpfh_queries = [0]
        else:
            fpfh_queries = list(range(0, n_acc, GLC_FPFH_QUERY_STRIDE))

        print(f"  FPFH query nodes: {len(fpfh_queries)} "
              f"(backfill_only={GLC_FPFH_ORB_BACKFILL_ONLY})")
        fpfh_needed = set()
        fpfh_candidate_pool = {}
        node_ids = np.arange(n_acc)
        for q in fpfh_queries:
            fpfh_needed.add(q)
            dists_to_q = np.linalg.norm(
                pose_positions - pose_positions[q], axis=1)
            valid_mask = np.abs(node_ids - q) >= GLC_MIN_INTERVAL
            cand = node_ids[valid_mask]
            if cand.size == 0:
                fpfh_candidate_pool[q] = []
                continue
            order = np.argsort(dists_to_q[cand])
            cand_sorted = cand[order]
            near = cand_sorted[
                dists_to_q[cand_sorted] <= GLC_FPFH_SPATIAL_MAX_DIST]
            if near.size == 0:
                near = cand_sorted[
                    :min(GLC_FPFH_SPATIAL_TOPK, cand_sorted.size)]
            elif near.size > GLC_FPFH_SPATIAL_TOPK:
                near = near[:GLC_FPFH_SPATIAL_TOPK]
            pool_list = [int(x) for x in near]
            fpfh_candidate_pool[q] = pool_list
            fpfh_needed.update(pool_list)
        print(f"  FPFH needed for {len(fpfh_needed)}/{n_acc} frames "
              f"(spatial pre-filter)")

        fpfh_pcd_db = {j: None for j in range(n_acc)}
        fpfh_feat_db = {j: None for j in range(n_acc)}
        fpfh_summary = {j: None for j in range(n_acc)}
        needed_nodes = sorted(fpfh_needed)

        def _extract_fpfh_node(j):
            pcd_j = get_pcd(j)
            if pcd_j is None or len(pcd_j.points) < 50:
                return j, None, None, None
            pcd_down, fpfh_feat = extract_fpfh(pcd_j)
            if fpfh_feat is None:
                return j, None, None, None
            summary = np.asarray(fpfh_feat.data).mean(axis=1)
            return j, pcd_down, fpfh_feat, summary

        fpfh_done = 0
        with ThreadPoolExecutor(max_workers=FPFH_WORKERS) as pool:
            futures = [pool.submit(_extract_fpfh_node, j)
                       for j in needed_nodes]
            for fut in as_completed(futures):
                j, pcd_down, fpfh_feat, summary = fut.result()
                fpfh_pcd_db[j] = pcd_down
                fpfh_feat_db[j] = fpfh_feat
                fpfh_summary[j] = summary
                fpfh_done += 1
                if fpfh_done % 50 == 0 or fpfh_done == len(needed_nodes):
                    print(f"    {fpfh_done}/{len(needed_nodes)}")
        n_fpfh = sum(1 for f in fpfh_feat_db.values() if f is not None)
        print(f"  {n_fpfh}/{n_acc} frames with FPFH features")
        _mem_check("after FPFH extraction")

        existing_edges = set()
        for e in pose_graph.edges:
            existing_edges.add((e.source_node_id, e.target_node_id))
            existing_edges.add((e.target_node_id, e.source_node_id))

        fpfh_queries_valid = [q for q in fpfh_queries
                              if fpfh_summary[q] is not None]

        def _eval_fpfh_query(q):
            local_edges = []
            local_tried = 0

            scored = []
            for t in fpfh_candidate_pool.get(q, []):
                if (q, t) in existing_edges:
                    continue
                if fpfh_summary[t] is None:
                    continue
                dist = np.linalg.norm(fpfh_summary[q] - fpfh_summary[t])
                scored.append((t, dist))

            scored.sort(key=lambda x: x[1])

            for t, _ in scored[:GLC_FPFH_MAX_CANDIDATES]:
                local_tried += 1
                try:
                    ransac_result = (
                        o3d.pipelines.registration
                        .registration_ransac_based_on_feature_matching(
                            fpfh_pcd_db[q], fpfh_pcd_db[t],
                            fpfh_feat_db[q], fpfh_feat_db[t],
                            True,
                            GLC_FPFH_RANSAC_DIST,
                            o3d.pipelines.registration
                            .TransformationEstimationPointToPoint(False),
                            3,
                            [
                                o3d.pipelines.registration
                                .CorrespondenceCheckerBasedOnEdgeLength(0.9),
                                o3d.pipelines.registration
                                .CorrespondenceCheckerBasedOnDistance(
                                    GLC_FPFH_RANSAC_DIST),
                            ],
                            o3d.pipelines.registration
                            .RANSACConvergenceCriteria(2500, 0.999)))
                except TypeError:
                    try:
                        ransac_result = (
                            o3d.pipelines.registration
                            .registration_ransac_based_on_feature_matching(
                                fpfh_pcd_db[q], fpfh_pcd_db[t],
                                fpfh_feat_db[q], fpfh_feat_db[t],
                                GLC_FPFH_RANSAC_DIST,
                                o3d.pipelines.registration
                                .TransformationEstimationPointToPoint(False),
                                3,
                                [
                                    o3d.pipelines.registration
                                    .CorrespondenceCheckerBasedOnEdgeLength(
                                        0.9),
                                    o3d.pipelines.registration
                                    .CorrespondenceCheckerBasedOnDistance(
                                        GLC_FPFH_RANSAC_DIST),
                                ],
                                o3d.pipelines.registration
                                .RANSACConvergenceCriteria(8000, 0.999)))
                    except Exception:
                        continue

                if ransac_result.fitness < GLC_FPFH_FITNESS_MIN:
                    continue

                ok_icp, T_icp, info_icp = try_icp_edge(
                    q, t, ransac_result.transformation,
                    refine_with_odometry=(not GLC_FPFH_SKIP_ODO_REFINE),
                    odometry_full_retry=False)
                if ok_icp:
                    local_edges.append((q, t, T_icp, info_icp))

            return q, local_tried, local_edges

        fpfh_results = []
        with ThreadPoolExecutor(max_workers=GLC_WORKERS) as pool:
            for out in pool.map(_eval_fpfh_query, fpfh_queries_valid):
                fpfh_results.append(out)

        seen_pairs = set()
        fpfh_results.sort(key=lambda x: x[0])
        for _, local_tried, local_edges in fpfh_results:
            fpfh_tried += local_tried
            for q, t, T_icp, info_icp in local_edges:
                key = (min(q, t), max(q, t))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        q, t, T_icp, info_icp, uncertain=True))
                fpfh_ok += 1
                glc_edges.append((q, t, 0, "fpfh"))

        print(f"  FPFH-based global LC: {fpfh_ok}/{fpfh_tried} "
              f"verified & added")
    else:
        print("  FPFH geometric loop closure disabled.")

    # ---- Head-tail ICP (full-circle detection) ----
    ht_ok = 0
    head_nodes = list(range(0, min(GLC_HEAD_TAIL_N, n_acc), 3))
    tail_nodes = list(range(max(0, n_acc - GLC_HEAD_TAIL_N), n_acc, 3))
    ht_possible = (len(tail_nodes) > 0 and len(head_nodes) > 0
                   and tail_nodes[0] - head_nodes[-1] >= GLC_MIN_INTERVAL)

    if ht_possible:
        print("  Checking head-tail loop closure via ICP...")
        existing = set((a, b) for (a, b, *_) in glc_edges)
        for t_n in tail_nodes:
            for h_n in head_nodes:
                if (t_n, h_n) in existing or (h_n, t_n) in existing:
                    continue
                s_idx = accepted[t_n]
                t_idx = accepted[h_n]
                pose_init = np.linalg.inv(poses[t_idx]) @ poses[s_idx]
                best_ok = False
                best_T = best_info = None
                for init_guess in [pose_init, EYE4]:
                    ok_i, T_i, info_i = try_icp_edge(
                        t_n, h_n, init_guess,
                        refine_with_odometry=True,
                        odometry_full_retry=(not GLC_FAST_ONLY_ODO))
                    if ok_i:
                        best_ok, best_T, best_info = True, T_i, info_i
                        break
                if best_ok:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            t_n, h_n, best_T, best_info, uncertain=True))
                    ht_ok += 1
                    glc_edges.append((t_n, h_n, 0, "ht-icp"))
                    existing.add((t_n, h_n))
        print(f"  Head-tail ICP loop closures: {ht_ok} added")
    else:
        print("  Trajectory too short for head-tail ICP check.")

    glc_total = glc_ok + fpfh_ok + ht_ok
    if glc_edges:
        to_show = glc_edges
        if (not VERBOSE_GLOBAL_EDGE_LOG
                and len(glc_edges) > GLOBAL_EDGE_LOG_LIMIT):
            to_show = glc_edges[:GLOBAL_EDGE_LOG_LIMIT]
        for entry in to_show:
            a, b = entry[0], entry[1]
            method = entry[3] if len(entry) > 3 else "?"
            s_ts = pairs[accepted[a]][0]
            t_ts = pairs[accepted[b]][0]
            print(f"    [{method:>6s}] node {a:>4d} <-> {b:>4d}  "
                  f"(dt={abs(t_ts - s_ts):.1f}s)")
        if len(to_show) < len(glc_edges):
            print(f"    ... {len(glc_edges) - len(to_show)} more edges "
                  f"omitted")
    phase2b_time = time.time() - t_phase2b
    metrics.update({
        "orb_glc_tried": glc_tried,
        "orb_glc_accepted": glc_ok,
        "orb_glc_accept_rate": glc_ok / glc_tried if glc_tried else 0.0,
        "fpfh_glc_tried": fpfh_tried,
        "fpfh_glc_accepted": fpfh_ok,
        "fpfh_glc_accept_rate": (
            fpfh_ok / fpfh_tried if fpfh_tried else 0.0),
        "head_tail_lc_added": ht_ok,
        "total_global_lc_added": glc_total,
        "phase2b_global_lc_time_sec": phase2b_time,
        "after_global_lc_memory_mb": get_memory_mb_safe(),
    })
    print(f"  Total global edges: {glc_total} "
          f"(ORB={glc_ok}, FPFH={fpfh_ok}, head-tail={ht_ok})  "
          f"({phase2b_time:.1f}s)\n")
    _mem_check("after global LC")

    # ==============================================================
    # Phase 3 – Global pose-graph optimisation (two-pass)
    # ==============================================================
    print("Phase 3: Pose-graph optimisation (two-pass)...")
    t_phase3 = time.time()

    opt1 = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.10,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        opt1)
    print("  Pass 1 (coarse) done")

    opt2 = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=0.03,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        opt2)
    phase3_time = time.time() - t_phase3
    metrics["phase3_optimization_time_sec"] = phase3_time
    print(f"  Pass 2 (fine) done  ({phase3_time:.1f}s)\n")

    # ==============================================================
    # Save trajectory
    # ==============================================================
    trajectory = []
    for j in range(n_acc):
        T_wc = pose_graph.nodes[j].pose
        trajectory.append((pairs[accepted[j]][0], T_wc))

    save_tum_trajectory(trajectory, TRAJ_FILE)
    elapsed = time.time() - wall_start
    final_memory_mb = get_memory_mb_safe()
    num_uncertain_edges = sum(
        1 for e in pose_graph.edges if getattr(e, "uncertain", False))
    num_pose_graph_edges = len(pose_graph.edges)
    metrics.update(compute_trajectory_motion_metrics(trajectory))
    metrics.update({
        "trajectory_file": TRAJ_FILE,
        "num_pose_graph_nodes": len(pose_graph.nodes),
        "num_pose_graph_edges": num_pose_graph_edges,
        "num_odometry_edges": num_pose_graph_edges - num_uncertain_edges,
        "num_uncertain_edges": num_uncertain_edges,
        "total_wall_time_sec": elapsed,
        "runtime_per_accepted_frame_sec": (
            elapsed / n_acc if n_acc > 0 else None),
        "final_memory_mb": final_memory_mb,
    })
    print(f"Saved {n_acc} poses to {TRAJ_FILE}")
    print(f"Total wall time: {elapsed:.1f}s")
    _mem_check("final")

    rgbd_cache_stats = rgbd_store.stats()
    pcd_cache_stats = pcd_cache.stats()
    pcd_level_cache_stats = pcd_level_cache.stats()
    pnp_depth_cache_stats = pnp_depth_lru.stats()
    metrics.update({
        "rgbd_cache_stats": rgbd_cache_stats,
        "pcd_cache_stats": pcd_cache_stats,
        "pcd_level_cache_stats": pcd_level_cache_stats,
        "pnp_depth_cache_stats": pnp_depth_cache_stats,
    })

    print(f"\n  Cache statistics (Jetson simulation):")
    print(f"    {rgbd_cache_stats}")
    print(f"    {pcd_cache_stats}")
    print(f"    {pcd_level_cache_stats}")
    print(f"    {pnp_depth_cache_stats}")
    report_paths = save_metrics_report(metrics, SCRIPT_DIR)
    print(f"Reports saved to {report_paths['json']} and "
          f"{report_paths['markdown']}")
    print()
    print("Run 3.build_pointcloud.py next.")


if __name__ == "__main__":
    main()
