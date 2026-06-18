"""
sift_triangulation.py
---------------------
SIFT-based reference matching experiment with RANSAC intersection.

Pipeline
--------
  1. Take view_000 of bugNIST900 GH as reference.
  2. Extract SIFT keypoints and descriptors from the reference and all
     other views.
  3. Click on the reference view to select a point.
  4. Find the SIFT keypoint closest to the clicked position.
  5. For each other view, find the best SIFT descriptor match
     (using BFMatcher with L2 norm + Lowe's ratio test).
  6. Plot the reference with both the click point and the selected SIFT
     keypoint highlighted, plus all matched views.
  7. Cast rays through the reference keypoint and all matched keypoints,
     run RANSAC to estimate the 3-D intersection, and show the result
     in a vedo 3-D viewer.

Usage:
  python src/checks/sift_triangulation.py
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ── Make src importable ───────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
SRC_DIR      = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from backprojection import build_perspective_camera
from helper_functions import otsu_threshold

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_ROOT  = PROJECT_ROOT / "data" / "raw"       / "bugNIST_900"
PROC_ROOT = PROJECT_ROOT / "data" / "processed" / "bugNIST_900"

# ── Constants ─────────────────────────────────────────────────────────────────
SPECIMEN   = "GH"
IMAGE_NAME = "gras_9_041"
REF_VIEW   = "view_000"
IMG_SIZE   = 1440

# SIFT parameters
SIFT_N_FEATURES    = 0       # 0 = keep all features
SIFT_N_OCTAVE_LAYERS = 3
SIFT_CONTRAST_THRESH = 0.04
SIFT_EDGE_THRESH     = 10
SIFT_SIGMA           = 1.6

# Lowe's ratio test threshold
LOWE_RATIO = 0.75

# ── RANSAC / intersection constants ───────────────────────────────────────────
RANSAC_N_ITER      = 400
RANSAC_INLIER_FRAC = 0.07
RANSAC_MIN_INLIERS = 3

ESTIMATED_POINT_RADIUS = 55.0   # voxel units


# ═══════════════════════════════════════════════════════════════════════════════
#  SIFT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_sift_features(image_np: np.ndarray):
    """
    Extract SIFT keypoints and descriptors from a numpy image (RGB).

    Returns
    -------
    keypoints   : list of cv2.KeyPoint
    descriptors : np.ndarray of shape (n_keypoints, 128), dtype float32
                  or None if no keypoints found
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=SIFT_N_FEATURES,
        nOctaveLayers=SIFT_N_OCTAVE_LAYERS,
        contrastThreshold=SIFT_CONTRAST_THRESH,
        edgeThreshold=SIFT_EDGE_THRESH,
        sigma=SIFT_SIGMA,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def find_nearest_keypoint(keypoints, u: float, v: float) -> int:
    """
    Find the index of the keypoint nearest to pixel (u, v).
    """
    pts = np.array([kp.pt for kp in keypoints])   # (N, 2) — each is (x, y)
    dists = np.sqrt((pts[:, 0] - u) ** 2 + (pts[:, 1] - v) ** 2)
    return int(np.argmin(dists))


def match_descriptor_to_view(
    ref_descriptor: np.ndarray,
    target_descriptors: np.ndarray,
    target_keypoints,
    ratio: float = LOWE_RATIO,
) -> tuple[int, float] | None:
    """
    Match a single reference descriptor against all descriptors in a target
    view using BFMatcher with Lowe's ratio test.

    Returns
    -------
    (best_kp_idx, best_distance) or None if no good match
    """
    if target_descriptors is None or len(target_descriptors) < 2:
        return None

    # BFMatcher expects arrays of descriptors — wrap the query
    bf = cv2.BFMatcher(cv2.NORM_L2)
    ref_desc = ref_descriptor.reshape(1, -1).astype(np.float32)
    matches = bf.knnMatch(ref_desc, target_descriptors.astype(np.float32), k=2)

    if len(matches) == 0 or len(matches[0]) < 2:
        return None

    m, n = matches[0]
    if m.distance < ratio * n.distance:
        return m.trainIdx, m.distance

    # Fall back to best match without ratio test
    return m.trainIdx, m.distance


# ═══════════════════════════════════════════════════════════════════════════════
#  RAY COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ray(cam_data, u, v):
    """
    Compute world-space ray origin and direction for pixel (u, v).

    Returns
    -------
    origin    : (3,) camera position
    direction : (3,) unit direction in world space
    """
    K_inv  = cam_data["K_inv"]
    right  = cam_data["right"]
    up     = cam_data["up"]
    look   = cam_data["look"]
    origin = cam_data["cam_origin"]

    # Pixel centre in homogeneous coords
    pix = np.array([u + 0.5, v + 0.5, 1.0])
    d_cam = K_inv @ pix

    # World-space direction (VTK convention: -up for image Y)
    d_world = d_cam[0] * right + (-d_cam[1]) * up + d_cam[2] * look
    d_world = d_world / np.linalg.norm(d_world)

    return origin.copy(), d_world


def ray_aabb_intersect(origin, direction, box_min, box_max):
    """
    Compute parametric entry/exit of a ray with an AABB.
    Returns (t_enter, t_exit) or (None, None) if miss.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d = np.where(np.abs(direction) > 1e-12, 1.0 / direction, np.inf)

    t1 = (box_min - origin) * inv_d
    t2 = (box_max - origin) * inv_d

    t_enter = np.max(np.minimum(t1, t2))
    t_exit  = np.min(np.maximum(t1, t2))

    if t_enter > t_exit or t_exit < 0:
        return None, None

    t_enter = max(t_enter, 0.0)
    return t_enter, t_exit


def cast_keypoint_ray(cam_data, kp_x, kp_y, box_min, box_max):
    """
    Cast a single ray through the keypoint pixel and return it as a dict,
    or None if the ray misses the volume.

    Parameters
    ----------
    kp_x, kp_y : float — keypoint pixel coordinates
    """
    u = int(round(kp_x))
    v = int(round(kp_y))
    origin, direction = compute_ray(cam_data, u, v)
    t_enter, t_exit = ray_aabb_intersect(origin, direction, box_min, box_max)
    if t_enter is None:
        return None
    return {
        "origin":    origin,
        "direction": direction,
        "p_enter":   origin + t_enter * direction,
        "p_exit":    origin + t_exit  * direction,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  RANSAC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def closest_point_to_two_rays(o1, d1, o2, d2):
    """
    Midpoint of the shortest segment between two skew lines.
    Returns None when the lines are (near-)parallel.
    """
    cross = np.cross(d1, d2)
    denom = np.dot(cross, cross)
    if denom < 1e-12:
        return None
    w  = o2 - o1
    t1 = np.dot(np.cross(w, d2), cross) / denom
    t2 = np.dot(np.cross(w, d1), cross) / denom
    return 0.5 * ((o1 + t1 * d1) + (o2 + t2 * d2))


def point_to_ray_distances_batch(point, origins, directions):
    """
    Vectorised perpendicular distance from *point* to each of N rays.

    Parameters
    ----------
    point      : (3,)
    origins    : (N, 3)
    directions : (N, 3)  — must be unit vectors

    Returns
    -------
    distances : (N,)
    """
    diff     = point - origins                                   # (N, 3)
    proj_len = np.einsum("ni,ni->n", diff, directions)           # (N,)
    perp     = diff - proj_len[:, np.newaxis] * directions       # (N, 3)
    return np.linalg.norm(perp, axis=1)                          # (N,)


def weighted_least_squares_intersection(origins, directions, weights):
    """
    Find the 3-D point P that minimises the weighted sum of squared
    perpendicular distances to a set of rays.
    """
    A  = np.zeros((3, 3))
    b  = np.zeros(3)
    I3 = np.eye(3)
    for o, d, w in zip(origins, directions, weights):
        M  = w * (I3 - np.outer(d, d))
        A += M
        b += M @ o
    try:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None


def ransac_ray_intersection(
    origins,
    directions,
    weights,
    n_iterations=300,
    inlier_threshold=15.0,
    min_inliers=10,
):
    """
    RANSAC estimation of the 3-D point best described by a bundle of rays.

    Hypothesis generation draws two different rays from the pool.
    Scoring uses the weighted inlier sum.
    """
    N = len(origins)

    if N < 2:
        pt = weighted_least_squares_intersection(origins, directions, weights)
        return pt, np.ones(N, dtype=bool), np.nan

    best_point   = None
    best_score   = -np.inf
    best_inliers = np.zeros(N, dtype=bool)

    indices = np.arange(N)

    for _ in range(n_iterations):
        i1, i2 = np.random.choice(indices, size=2, replace=False)

        candidate = closest_point_to_two_rays(
            origins[i1], directions[i1],
            origins[i2], directions[i2],
        )
        if candidate is None:
            continue

        dists   = point_to_ray_distances_batch(candidate, origins, directions)
        inliers = dists < inlier_threshold
        if inliers.sum() < min_inliers:
            continue

        score = float(weights[inliers].sum())
        if score > best_score:
            best_score   = score
            best_inliers = inliers.copy()

            refined = weighted_least_squares_intersection(
                origins[inliers], directions[inliers], weights[inliers],
            )
            if refined is not None:
                best_point = refined

    if best_point is None:
        return None, best_inliers, np.nan

    # Final refinement
    dists        = point_to_ray_distances_batch(best_point, origins, directions)
    best_inliers = dists < inlier_threshold
    if best_inliers.sum() >= min_inliers:
        refined = weighted_least_squares_intersection(
            origins[best_inliers], directions[best_inliers],
            weights[best_inliers],
        )
        if refined is not None:
            best_point = refined

    inlier_dists = point_to_ray_distances_batch(
        best_point, origins[best_inliers], directions[best_inliers]
    )
    residual = float(np.mean(inlier_dists))

    return best_point, best_inliers, residual


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SIFT-based reference matching with RANSAC intersection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lowe-ratio", type=float, default=LOWE_RATIO,
        help="Lowe's ratio test threshold for SIFT matching.",
    )
    args = parser.parse_args()
    lowe_ratio = args.lowe_ratio

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1: Load images and extract SIFT features
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  EXTRACTING SIFT FEATURES")
    print("=" * 60)

    ref_path = PROC_ROOT / SPECIMEN / IMAGE_NAME / f"{REF_VIEW}.png"
    if not ref_path.exists():
        print(f"[ERROR] Reference view not found: {ref_path}")
        sys.exit(1)

    ref_img = Image.open(ref_path).convert("RGB")
    ref_img_np = np.array(ref_img)
    print(f"  Reference image: {ref_path.relative_to(PROJECT_ROOT)}")
    print(f"  Image size     : {ref_img.size}")

    ref_kps, ref_descs = extract_sift_features(ref_img_np)
    print(f"  Reference SIFT keypoints: {len(ref_kps)}")
    if ref_descs is None or len(ref_kps) == 0:
        print("[ERROR] No SIFT features found in the reference image.")
        sys.exit(1)

    # ── Load camera data ──────────────────────────────────────────────────────
    import pickle
    cam_pkl = PROC_ROOT / SPECIMEN / IMAGE_NAME / "cameras.pkl"
    print(f"Loading cameras: {cam_pkl.relative_to(PROJECT_ROOT)}")
    with open(cam_pkl, "rb") as f:
        cameras = pickle.load(f)

    volume_shape = cameras[REF_VIEW]["volume_shape"]
    Vx, Vy, Vz = volume_shape
    print(f"Volume shape: {volume_shape}")

    # ── Discover other views ──────────────────────────────────────────────────
    view_dir = PROC_ROOT / SPECIMEN / IMAGE_NAME
    all_views = sorted(
        p.stem for p in view_dir.glob("view_*.png")
    )
    other_view_names = [vn for vn in all_views if vn != REF_VIEW]
    print(f"  Target views: {other_view_names}")

    # ── Extract SIFT for other views ──────────────────────────────────────────
    other_sift: dict[str, tuple] = {}      # vn -> (keypoints, descriptors)
    other_view_images: dict[str, np.ndarray] = {}

    for vn in other_view_names:
        vp = PROC_ROOT / SPECIMEN / IMAGE_NAME / f"{vn}.png"
        img_np = np.array(Image.open(vp).convert("RGB"))
        other_view_images[vn] = img_np

        kps, descs = extract_sift_features(img_np)
        other_sift[vn] = (kps, descs)
        n_kps = len(kps) if kps else 0
        print(f"    {vn}: {n_kps} keypoints")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2: Click on the reference view to select a point
    # ══════════════════════════════════════════════════════════════════════════

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(ref_img_np)

    # Draw SIFT keypoints as small dots so the user can see them
    kp_xs = [kp.pt[0] for kp in ref_kps]
    kp_ys = [kp.pt[1] for kp in ref_kps]
    ax.scatter(kp_xs, kp_ys, s=3, c="cyan", alpha=0.3, zorder=2)

    ax.set_title(
        f"Click on {REF_VIEW} to select a point\n"
        f"(cyan dots = SIFT keypoints · closest will be picked)\n"
        f"Close window when done",
    )
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")

    clicked_points: list[tuple[float, float]] = []

    def on_click(event):
        if event.inaxes != ax:
            return
        u, v = event.xdata, event.ydata
        clicked_points.append((u, v))
        ax.plot(u, v, "r+", markersize=20, markeredgewidth=2)
        fig.canvas.draw()
        print(f"  Clicked: (u={u:.1f}, v={v:.1f})")

    fig.canvas.mpl_connect("button_press_event", on_click)
    print("\n>>> Click on the reference view to select a pixel. "
          "Close the window when done.\n")
    plt.show()

    if not clicked_points:
        print("No points clicked. Exiting.")
        return

    # ── Find nearest SIFT keypoint to the click ───────────────────────────────
    u_click, v_click = clicked_points[-1]
    nearest_idx = find_nearest_keypoint(ref_kps, u_click, v_click)
    nearest_kp  = ref_kps[nearest_idx]
    kp_u, kp_v  = nearest_kp.pt       # (x, y) in pixel coords
    kp_dist     = math.sqrt((kp_u - u_click) ** 2 + (kp_v - v_click) ** 2)

    print(f"\nClicked pixel : ({u_click:.1f}, {v_click:.1f})")
    print(f"Nearest SIFT  : ({kp_u:.1f}, {kp_v:.1f})  "
          f"[distance = {kp_dist:.1f} px,  scale = {nearest_kp.size:.1f}]")

    ref_descriptor = ref_descs[nearest_idx]   # (128,)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3: Match the selected SIFT descriptor to all other views
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  SIFT MATCHING TO OTHER VIEWS")
    print("=" * 60)

    # match_results[vn] = (kp_x, kp_y, distance, kp_idx)  or None
    match_results: dict[str, tuple[float, float, float, int] | None] = {}

    for vn in other_view_names:
        t_kps, t_descs = other_sift[vn]
        result = match_descriptor_to_view(
            ref_descriptor, t_descs, t_kps, ratio=lowe_ratio,
        )
        if result is not None:
            kp_idx, dist = result
            mx, my = t_kps[kp_idx].pt
            match_results[vn] = (mx, my, dist, kp_idx)
            print(f"  {vn}:  ({mx:.1f}, {my:.1f})  dist={dist:.2f}")
        else:
            match_results[vn] = None
            print(f"  {vn}:  no good match")

    matched_views = [vn for vn in other_view_names
                     if match_results[vn] is not None]
    print(f"\n  Matched {len(matched_views)} / {len(other_view_names)} views")

    if len(matched_views) == 0:
        print("[WARNING] No SIFT matches found. Try lowering --lowe-ratio.")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4: RANSAC intersection estimation
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  RANSAC INTERSECTION")
    print("=" * 60)

    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([Vx, Vy, Vz], dtype=np.float64)

    ref_cam_data = build_perspective_camera(cameras[REF_VIEW])

    # ── Cast reference ray through the selected SIFT keypoint ─────────────────
    print(f"\n  Casting reference ray through SIFT keypoint "
          f"({kp_u:.1f}, {kp_v:.1f}) …")
    ref_ray = cast_keypoint_ray(ref_cam_data, kp_u, kp_v,
                                 box_min, box_max)
    if ref_ray is None:
        print("  [ERROR] Reference ray misses the volume. Exiting.")
        return
    print(f"    1 valid ray")

    # ── Cast matched-view rays ────────────────────────────────────────────────
    print("  Casting matched-view rays …")
    match_rays_by_view: dict[str, tuple[dict, float]] = {}

    for vn in matched_views:
        mx, my, dist, _ = match_results[vn]
        cam_data_other = build_perspective_camera(cameras[vn])
        mray = cast_keypoint_ray(cam_data_other, mx, my,
                                  box_min, box_max)
        if mray is None:
            print(f"    {vn}: ({mx:.1f},{my:.1f}), ray misses volume — skipped")
            continue
        # Convert distance to a similarity-like weight (lower dist = higher weight)
        weight = 1.0 / max(dist, 1e-6)
        match_rays_by_view[vn] = (mray, weight)
        print(f"    {vn}: ({mx:.1f},{my:.1f}), 1 ray, "
              f"dist={dist:.2f}, w={weight:.4f}")

    # ── Build unified ray arrays for RANSAC ───────────────────────────────────
    # ── Build ray pool from matched views only (no reference ray) ─────────────
    rans_origins_list    = []
    rans_directions_list = []
    rans_weights_list    = []

    for vn in matched_views:
        if vn not in match_rays_by_view:
            continue
        mray, weight = match_rays_by_view[vn]
        rans_origins_list.append(mray["origin"])
        rans_directions_list.append(mray["direction"])
        rans_weights_list.append(weight)

    if len(rans_origins_list) < 2:
        print("  [ERROR] Not enough matched-view rays for RANSAC "
              f"(need ≥ 2, have {len(rans_origins_list)}). Exiting.")
        return

    rans_origins  = np.array(rans_origins_list,    dtype=np.float64)
    rans_dirs     = np.array(rans_directions_list, dtype=np.float64)
    rans_weights  = np.array(rans_weights_list,    dtype=np.float64)

    n_match_rays = len(rans_origins)

    print(f"\n  Rays in pool: {n_match_rays} matched-view rays  "
          f"(reference ray excluded)")

    # ── Adaptive inlier threshold ─────────────────────────────────────────────
    vol_diag         = float(np.linalg.norm(box_max - box_min))
    inlier_threshold = max(10.0, RANSAC_INLIER_FRAC * vol_diag)
    print(f"  Volume diagonal: {vol_diag:.1f} vox  →  "
          f"inlier threshold: {inlier_threshold:.1f} vox")

    # ── Run RANSAC ────────────────────────────────────────────────────────────
    best_pt, inlier_mask, residual = ransac_ray_intersection(
        rans_origins, rans_dirs, rans_weights,
        n_iterations     = RANSAC_N_ITER,
        inlier_threshold = inlier_threshold,
        min_inliers      = RANSAC_MIN_INLIERS,
    )

    n_inliers = int(inlier_mask.sum())

    if best_pt is not None:
        print(f"  ✓ Intersection estimated at: "
              f"({best_pt[0]:.1f}, {best_pt[1]:.1f}, {best_pt[2]:.1f})")
        print(f"    Inliers: {n_inliers}/{len(rans_origins)}  "
              f"|  mean residual: {residual:.2f} vox")
    else:
        print("  ✗ RANSAC failed to find a consensus intersection.")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5: 2D Visualisation
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  PLOTTING (2D)")
    print("=" * 60)

    from matplotlib.patches import Circle, FancyArrowPatch
    from matplotlib.colors import to_rgba

    # ── Dark theme ────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor":  "#1a1a2e",
        "axes.facecolor":    "#16213e",
        "text.color":        "#e0e0e0",
        "axes.labelcolor":   "#e0e0e0",
        "xtick.color":       "#e0e0e0",
        "ytick.color":       "#e0e0e0",
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Helvetica", "Arial"],
    })

    # ── Layout: reference on top-left, matched views in the rest ──────────────
    n_matched = len(matched_views)
    total_panels = 1 + n_matched
    n_cols = min(5, total_panels)
    n_rows = math.ceil(total_panels / n_cols)

    fig, axes_flat = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.4),
        gridspec_kw={"hspace": 0.40, "wspace": 0.20},
    )
    if n_rows == 1 and n_cols == 1:
        axes_flat = np.array([[axes_flat]])
    elif n_rows == 1:
        axes_flat = axes_flat[np.newaxis, :]
    elif n_cols == 1:
        axes_flat = axes_flat[:, np.newaxis]

    all_axes = axes_flat.flatten()

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "SIFT Triangulation",
        fontsize=20, fontweight="bold", color="white", y=0.97,
    )
    fig.text(
        0.5, 0.935,
        f"Query: {REF_VIEW} · SIFT keypoint ({kp_u:.0f}, {kp_v:.0f}) · "
        f"{SPECIMEN}/{IMAGE_NAME}  ·  "
        f"matched {len(matched_views)}/{len(other_view_names)} views",
        ha="center", fontsize=11, color="#aaaaaa",
    )

    # ── Panel 0: Reference image ──────────────────────────────────────────────
    ax_ref = all_axes[0]
    ax_ref.imshow(ref_img_np)

    # Show where the user clicked (red cross)
    ax_ref.plot(u_click, v_click, "r+", markersize=18, markeredgewidth=2.5,
                zorder=5, label="Click")

    # Show the selected SIFT keypoint (green circle)
    kp_circle = Circle(
        (kp_u, kp_v), radius=nearest_kp.size,
        linewidth=2.5, edgecolor="#2ed573", facecolor=to_rgba("#2ed573", 0.2),
        zorder=5,
    )
    ax_ref.add_patch(kp_circle)
    ax_ref.plot(kp_u, kp_v, "o", color="#2ed573", markersize=5, zorder=6)

    # Arrow from click to keypoint
    if kp_dist > 3:
        arrow = FancyArrowPatch(
            (u_click, v_click), (kp_u, kp_v),
            arrowstyle="->", color="yellow", linewidth=1.5,
            mutation_scale=12, zorder=5,
        )
        ax_ref.add_patch(arrow)

    ax_ref.set_title(
        f"REF {REF_VIEW}",
        fontsize=11, fontweight="bold", color="#ff4757", pad=6,
    )
    ax_ref.text(
        0.5, -0.04,
        f"click ({u_click:.0f},{v_click:.0f}) → "
        f"SIFT ({kp_u:.0f},{kp_v:.0f})  Δ={kp_dist:.1f}px",
        transform=ax_ref.transAxes, ha="center", va="top",
        fontsize=8, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="#ff4757", edgecolor="none", alpha=0.85),
    )
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    for spine in ax_ref.spines.values():
        spine.set_edgecolor("#ff4757")
        spine.set_linewidth(3.0)

    # ── Remaining panels: matched views ───────────────────────────────────────
    match_color = "#1e90ff"
    for idx, vn in enumerate(matched_views):
        ax = all_axes[1 + idx]
        ax.imshow(other_view_images[vn])

        mx, my, dist, _ = match_results[vn]

        # Matched keypoint circle
        kp_size = other_sift[vn][0][match_results[vn][3]].size
        kp_c = Circle(
            (mx, my), radius=kp_size,
            linewidth=2.5, edgecolor=match_color,
            facecolor=to_rgba(match_color, 0.25), zorder=5,
        )
        ax.add_patch(kp_c)
        ax.plot(mx, my, "o", color=match_color, markersize=5, zorder=6)

        # Crosshairs
        ax.axhline(my, color=match_color, linewidth=0.5, alpha=0.45,
                    linestyle="--")
        ax.axvline(mx, color=match_color, linewidth=0.5, alpha=0.45,
                    linestyle="--")

        ax.set_title(
            vn.replace("_", " ").upper(),
            fontsize=10, fontweight="bold", color=match_color, pad=6,
        )
        ax.text(
            0.5, -0.04,
            f"({mx:.0f},{my:.0f})  dist={dist:.2f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=match_color, edgecolor="none", alpha=0.85),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
            spine.set_linewidth(1.0)

    # Hide unused panels
    for idx in range(1 + n_matched, len(all_axes)):
        all_axes[idx].axis("off")

    # ── Legend ─────────────────────────────────────────────────────────────────
    fig.text(0.25, 0.02, "✚ Click position", fontsize=11, fontweight="bold",
             color="#ff4757", ha="center")
    fig.text(0.50, 0.02, "● Selected SIFT keypoint", fontsize=11,
             fontweight="bold", color="#2ed573", ha="center")
    fig.text(0.75, 0.02, "● Matched keypoint", fontsize=11,
             fontweight="bold", color=match_color, ha="center")

    plt.subplots_adjust(top=0.90, bottom=0.06)
    print("\n>>> Showing 2D results. "
          "Close the window to proceed to intersection view.")
    plt.show()
    plt.rcdefaults()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6: Dedicated intersection point viewer
    # ══════════════════════════════════════════════════════════════════════════

    if best_pt is not None:
        from vedo import (Volume as VedoVolume, Sphere as VedoSphere,
                          Plotter, Text2D)

        print("\n>>> Loading 3D volume for intersection viewer …")
        tiff_path = RAW_ROOT / SPECIMEN / f"{IMAGE_NAME}.tif"
        original_vol = VedoVolume(str(tiff_path))
        vol_data = original_vol.tonumpy().astype(np.float32)
        binary_mask = otsu_threshold(vol_data, log_scale=True)
        vol_data[~binary_mask] = 0
        print(f"  Loaded. Foreground voxels: {int((vol_data > 0).sum()):,}")

        print(">>> Opening intersection point viewer. Close when done.")

        pt_actors = []

        vol_ctx = VedoVolume(vol_data, spacing=original_vol.spacing(),
                             origin=original_vol.origin())
        vol_ctx.mode(0).cmap("bone").alpha([0, 0, 0.3, 0.6])
        pt_actors.append(vol_ctx)

        halo_pt = VedoSphere(best_pt, r=ESTIMATED_POINT_RADIUS * 1.4)
        halo_pt.color("white").alpha(0.20)
        halo_pt.name = "Estimated point halo"
        pt_actors.append(halo_pt)

        est_sphere = VedoSphere(best_pt, r=ESTIMATED_POINT_RADIUS)
        est_sphere.color("deeppink").alpha(0.92)
        est_sphere.name = "Estimated intersection"
        pt_actors.append(est_sphere)

        info_pt = Text2D(
            f"SIFT Triangulation\n"
            f"  Estimated intersection:\n"
            f"  ({best_pt[0]:.1f}, {best_pt[1]:.1f}, {best_pt[2]:.1f})\n"
            f"  radius={ESTIMATED_POINT_RADIUS:.1f} vox\n"
            f"  residual={residual:.2f} vox\n"
            f"  inliers={n_inliers}/{len(rans_origins)}\n"
            f"  matched views: {len(matched_views)}",
            pos="top-left", font="Mono", s=0.7, bg="black", alpha=0.75,
        )
        pt_actors.append(info_pt)

        plt_pt = Plotter(
            axes=1,
            title=f"SIFT Intersection — {SPECIMEN}/{IMAGE_NAME}",
        )
        plt_pt.show(*pt_actors, interactive=True)
    else:
        print("\n  Skipping intersection point viewer — no RANSAC result.")

    print("Done.")


if __name__ == "__main__":
    main()
