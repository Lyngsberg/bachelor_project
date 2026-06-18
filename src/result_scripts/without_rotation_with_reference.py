"""
interactive_ray_test.py
-----------------------
Interactive pipeline to verify ray geometry + embedding similarity:

  1. Load a 3D TIFF volume + a 2D rendered view.
  2. Show the 2D view in a matplotlib window — click a pixel.
  3. Compute the ray and show it in a vedo 3D viewer.
  4. Close the 3D viewer → find the most similar embedding patch
     in every other view (cosine similarity).
  5. Show all views in a grid with the matched patches highlighted.

Usage:
  python src/checks/interactive_ray_test.py

  # Custom specimen / image / view:
  python src/checks/interactive_ray_test.py \
      --specimen GH --image gras_9_043 --view view_000
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Make src importable
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from backprojection import build_perspective_camera
from helper_functions import otsu_threshold

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_ROOT  = PROJECT_ROOT / "data" / "raw"       / "bugNIST_900"
PROC_ROOT = PROJECT_ROOT / "data" / "processed" / "bugNIST_900"
EMB_ROOT  = PROJECT_ROOT / "data" / "embeddings" / "bugNIST_900"

# ── Constants ─────────────────────────────────────────────────────────────────
VTK_DEFAULT_FOV_DEG = 30.0
IMG_W      = 1440
IMG_H      = 1440
PATCH_SIZE = 16
GRID_SIZE  = IMG_H // PATCH_SIZE   # 90


def patch_idx_to_rc(idx: int) -> tuple[int, int]:
    """Convert a flat patch index to (row, col)."""
    return (idx // GRID_SIZE, idx % GRID_SIZE)


def find_most_similar_patches(
    query_embedding: np.ndarray,
    embeddings: dict,
    query_view: str,
    top_k: int = 1,
) -> dict:
    """
    For each view (except *query_view*), find the *top_k* patches most
    similar to *query_embedding* by cosine similarity.

    Returns
    -------
    dict[str, list[tuple[int, float]]]
        ``{view_name: [(patch_idx, similarity), ...]}``
    """
    q = query_embedding.astype(np.float64)
    q_norm = q / (np.linalg.norm(q) + 1e-12)

    results: dict[str, list[tuple[int, float]]] = {}
    for vn, emb_matrix in embeddings.items():
        if vn == query_view:
            continue
        T = emb_matrix.astype(np.float64)
        T_norms = np.linalg.norm(T, axis=1, keepdims=True)
        T_norms = np.where(T_norms < 1e-12, 1.0, T_norms)
        sims = (T / T_norms) @ q_norm            # (n_patches,)
        top_indices = np.argsort(sims)[::-1][:top_k]
        results[vn] = [(int(idx), float(sims[idx])) for idx in top_indices]

    return results

# ── RANSAC / intersection constants ───────────────────────────────────────────
# Weight assigned to every ray belonging to the reference (query) tube.
# Matched-view rays receive weight = cosine_similarity of their view (0–1).
# Raising REFERENCE_RAY_WEIGHT makes the query tube dominate both the
# inlier score and the final weighted-least-squares refinement step.
REFERENCE_RAY_WEIGHT = 1.0

RANSAC_N_ITER      = 300   # number of hypothesis-generate-and-score iterations
RANSAC_INLIER_FRAC = 0.04  # inlier threshold as fraction of volume diagonal
RANSAC_MIN_INLIERS = 10    # reject hypotheses with fewer inliers than this

# ── Estimated-point visualisation radius ──────────────────────────────────────
# Controls the size of the sphere drawn for the estimated intersection point
# in the dedicated 3D viewer at the end of the script.  Increase for visibility.
ESTIMATED_POINT_RADIUS = 55.0  # voxel units


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


# ── RANSAC helpers ────────────────────────────────────────────────────────────

def closest_point_to_two_rays(o1, d1, o2, d2):
    """
    Midpoint of the shortest segment between two skew lines.

    Uses the closed-form formula via the cross-product of the two directions.
    Returns None when the lines are (near-)parallel (cross-product ≈ 0).
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

    The closed-form solution is:

        A = Σ_i  w_i · (I − d_i d_i^T)
        b = Σ_i  w_i · (I − d_i d_i^T) · o_i
        P = A⁺ b          (least-squares pseudo-inverse for robustness)

    Parameters
    ----------
    origins    : (N, 3)
    directions : (N, 3)  — unit vectors
    weights    : (N,)    — non-negative

    Returns
    -------
    point : (3,) or None if the system is degenerate
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
    ref_mask,
    n_iterations=300,
    inlier_threshold=15.0,
    min_inliers=10,
):
    """
    RANSAC estimation of the 3-D point best described by a bundle of rays.

    Design choices
    --------------
    * **Hypothesis generation** — one ray is always drawn from the reference
      tube and one from the matched-view pool.  This anchors every candidate
      to the query view and avoids wasting iterations on hypotheses that
      ignore the reference entirely.

    * **Inlier scoring** — uses the *sum of weights* of inliers rather than
      a plain count.  Reference rays therefore contribute proportionally more
      to the consensus score (controlled by REFERENCE_RAY_WEIGHT).

    * **Refinement** — after the best inlier set is identified, a final
      weighted least-squares fit is run over all inliers so the result uses
      more than just the minimal two-ray sample.

    Parameters
    ----------
    origins          : (N, 3)
    directions       : (N, 3)  — unit vectors
    weights          : (N,)    — reference rays carry REFERENCE_RAY_WEIGHT,
                                 matched rays carry their cosine similarity
    ref_mask         : (N,) bool — True for reference-tube rays
    n_iterations     : int
    inlier_threshold : float   — max perpendicular distance (voxel units)
    min_inliers      : int

    Returns
    -------
    best_point   : (3,) ndarray or None
    inlier_mask  : (N,) bool
    residual     : float — mean inlier perpendicular distance (voxels)
    """
    N         = len(origins)
    ref_idx   = np.where( ref_mask)[0]
    other_idx = np.where(~ref_mask)[0]

    # Degenerate cases: fall back to plain WLS over everything
    if len(ref_idx) == 0 or len(other_idx) == 0 or N < 2:
        pt = weighted_least_squares_intersection(origins, directions, weights)
        return pt, np.ones(N, dtype=bool), np.nan

    best_point   = None
    best_score   = -np.inf
    best_inliers = np.zeros(N, dtype=bool)

    for _ in range(n_iterations):
        # ── Minimal sample: 1 reference ray + 1 matched-view ray ─────────
        i1 = int(np.random.choice(ref_idx))
        i2 = int(np.random.choice(other_idx))

        candidate = closest_point_to_two_rays(
            origins[i1], directions[i1],
            origins[i2], directions[i2],
        )
        if candidate is None:           # parallel rays — skip
            continue

        # ── Score hypothesis ──────────────────────────────────────────────
        dists   = point_to_ray_distances_batch(candidate, origins, directions)
        inliers = dists < inlier_threshold
        if inliers.sum() < min_inliers:
            continue

        # Weighted score — reference rays count more
        score = float(weights[inliers].sum())
        if score > best_score:
            best_score   = score
            best_inliers = inliers.copy()

            # Immediate refinement on the current inlier set
            refined = weighted_least_squares_intersection(
                origins[inliers],
                directions[inliers],
                weights[inliers],
            )
            if refined is not None:
                best_point = refined

    if best_point is None:
        return None, best_inliers, np.nan

    # ── Final refinement pass on the best inlier set ─────────────────────
    dists        = point_to_ray_distances_batch(best_point, origins, directions)
    best_inliers = dists < inlier_threshold
    if best_inliers.sum() >= min_inliers:
        refined  = weighted_least_squares_intersection(
            origins[best_inliers],
            directions[best_inliers],
            weights[best_inliers],
        ) 
        if refined is not None:
            best_point = refined

    # Residual = mean inlier distance after final refinement
    inlier_dists = point_to_ray_distances_batch(
        best_point, origins[best_inliers], directions[best_inliers]
    )
    residual = float(np.mean(inlier_dists))

    return best_point, best_inliers, residual


def main():
    parser = argparse.ArgumentParser(
        description="Interactive click-to-ray 3D visualizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--specimen", default="GH", help="Specimen name.")
    parser.add_argument("--image", default="gras_9_041", help="Image name (no ext).")
    parser.add_argument("--view", default="view_000", help="View name.")
    args = parser.parse_args()

    specimen   = args.specimen
    image_name = args.image
    view_name  = args.view

    # ── Load camera data ──────────────────────────────────────────────────────
    cam_pkl = PROC_ROOT / specimen / image_name / "cameras.pkl"
    print(f"Loading cameras: {cam_pkl.relative_to(PROJECT_ROOT)}")
    with open(cam_pkl, "rb") as f:
        cameras = pickle.load(f)

    cam_raw = cameras[view_name]
    cam_data = build_perspective_camera(cam_raw)
    volume_shape = cam_raw["volume_shape"]
    Vx, Vy, Vz = volume_shape
    print(f"Volume shape: {volume_shape}")

    # ── Load the rendered 2D view image ───────────────────────────────────────
    # The MIP image is stored in cameras.pkl
    img = cam_raw["mip"]   # (H, W, C) uint8
    print(f"View image shape: {img.shape}")

    # ── Load 3D volume for visualization ──────────────────────────────────────
    tiff_path = RAW_ROOT / specimen / f"{image_name}.tif"
    print(f"Loading 3D volume: {tiff_path.relative_to(PROJECT_ROOT)} ...")
    from vedo import Volume as VedoVolume
    original_vol = VedoVolume(str(tiff_path))
    vol_data = original_vol.tonumpy().astype(np.float32)
    binary_mask = otsu_threshold(vol_data, log_scale=True)
    vol_data[~binary_mask] = 0
    print(f"  Loaded. Foreground voxels: {int((vol_data > 0).sum()):,}")

    # ── Show 2D view and let user click ───────────────────────────────────────
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(f"Click on {view_name} to cast a ray\n(close window when done)")
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")

    clicked_points = []

    def on_click(event):
        if event.inaxes != ax:
            return
        u, v = event.xdata, event.ydata
        clicked_points.append((u, v))
        ax.plot(u, v, "r+", markersize=20, markeredgewidth=2)
        fig.canvas.draw()
        print(f"  Clicked: (u={u:.1f}, v={v:.1f})")

    fig.canvas.mpl_connect("button_press_event", on_click)
    print("\n>>> Click on the 2D view to select a pixel. Close the window when done.\n")
    plt.show()

    if not clicked_points:
        print("No points clicked. Exiting.")
        return

    # ── Compute rays for the ENTIRE 16x16 patch ───────────────────────────────
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([Vx, Vy, Vz], dtype=np.float64)

    rays = []
    # Use the LAST clicked point to define the patch
    u_click, v_click = clicked_points[-1]

    patch_col = int(u_click) // PATCH_SIZE
    patch_row = int(v_click) // PATCH_SIZE
    patch_col = min(patch_col, GRID_SIZE - 1)
    patch_row = min(patch_row, GRID_SIZE - 1)

    u_start = patch_col * PATCH_SIZE
    v_start = patch_row * PATCH_SIZE
    
    print(f"\nGenerating rays for patch (row={patch_row}, col={patch_col})...")

    # Loop over all 16x16 pixels in the patch
    for du in range(PATCH_SIZE):
        for dv in range(PATCH_SIZE):
            u = u_start + du
            v = v_start + dv

            origin, direction = compute_ray(cam_data, u, v)
            t_enter, t_exit = ray_aabb_intersect(origin, direction, box_min, box_max)

            if t_enter is None:
                continue

            p_enter = origin + t_enter * direction
            p_exit  = origin + t_exit  * direction

            rays.append({
                "u": u, "v": v,
                "origin": origin,
                "direction": direction,
                "p_enter": p_enter,
                "p_exit": p_exit,
            })
            
    print(f"  Created {len(rays)} valid rays for the patch.")

    # ── Build 3D actors (shown at the end, after embedding similarity) ────────
    from vedo import Volume as VedoVolume, Line, Point, Plotter, Text2D

    print("\nBuilding 3D actors for all patch rays ...")

    # Create semi-transparent volume
    vol_3d = VedoVolume(vol_data, spacing=original_vol.spacing(),
                        origin=original_vol.origin())
    vol_3d.mode(0).cmap("bone").alpha([0, 0, 0.3, 0.6])

    # Create ray lines and camera point
    actors = [vol_3d]

    cam_point = Point(cam_data["cam_origin"], c="yellow", r=15)
    cam_point.name = "Camera"
    actors.append(cam_point)

    # Use a single color for the whole patch beam for clarity
    beam_color = "cyan"

    for i, ray in enumerate(rays):
        # Very faint line from camera to volume entry
        line_pre = Line(ray["origin"], ray["p_enter"],
                        c=beam_color, alpha=0.1, lw=1)
        line_pre.name = f"Ray {i} pre-entry"

        # Brighter line through the volume
        line_vol = Line(ray["p_enter"], ray["p_exit"],
                        c=beam_color, alpha=0.3, lw=2)
        line_vol.name = f"Ray {i} through volume"

        actors.extend([line_pre, line_vol])

        # Mark entry/exit only at the 4 corners of the patch
        if i in (0, PATCH_SIZE - 1,
                 PATCH_SIZE * (PATCH_SIZE - 1), PATCH_SIZE * PATCH_SIZE - 1):
            actors.append(Point(ray["p_enter"], c="green", r=8))
            actors.append(Point(ray["p_exit"],  c="red",   r=8))

    print(f"  Query tube built — {len(rays)} rays, {len(actors)} actors so far.")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2: Embedding similarity — find best matching patch in each view
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  EMBEDDING SIMILARITY")
    print("=" * 60)

    # ── Load embeddings ───────────────────────────────────────────────────────
    emb_pkl = EMB_ROOT / specimen / f"{image_name}.pkl"
    if not emb_pkl.exists():
        print(f"  [SKIP] Embeddings not found: {emb_pkl}")
        print("  Run generate_embeddings.py first. Exiting.")
        return

    with open(emb_pkl, "rb") as f:
        embeddings = pickle.load(f)
    print(f"  Loaded embeddings for {len(embeddings)} views.")

    # Use the LAST clicked point for the similarity search
    u_click, v_click = clicked_points[-1]

    # Convert pixel (u, v) to patch (row, col)
    patch_col = int(u_click) // PATCH_SIZE
    patch_row = int(v_click) // PATCH_SIZE
    patch_col = min(patch_col, GRID_SIZE - 1)
    patch_row = min(patch_row, GRID_SIZE - 1)
    patch_idx = patch_row * GRID_SIZE + patch_col

    print(f"  Clicked pixel: ({u_click:.1f}, {v_click:.1f})")
    print(f"  → Patch (row={patch_row}, col={patch_col}), flat index={patch_idx}")

    if view_name not in embeddings:
        print(f"  [ERROR] Query view '{view_name}' not in embeddings. "
              f"Available: {sorted(embeddings.keys())}")
        return

    query_embedding = embeddings[view_name][patch_idx]  # (hidden_dim,)
    print(f"  Query embedding shape: {query_embedding.shape}")

    # ── Find most similar patch in each other view ────────────────────────────
    matches = find_most_similar_patches(
        query_embedding, embeddings, view_name, top_k=1
    )

    print(f"\n  Best matches:")
    for vn, match_list in matches.items():
        for idx, sim in match_list:
            rc = patch_idx_to_rc(idx)
            print(f"    {vn}: patch {idx} (row={rc[0]}, col={rc[1]})  "
                  f"similarity={sim:.4f}")

    # ── Add tube rays for every matched view into the 3D scene ───────────────
    # Colors mirror the 2D similarity grid (index 0 = query = red is already
    # cyan in 3D; matched views start at index 1 in plot_colors).
    view_3d_colors = ["lime", "magenta", "orange", "yellow", "violet",
                      "pink", "gold", "tomato", "aquamarine", "coral"]

    print("\n  Computing patch-ray tubes for matched views ...")

    # Accumulates every matched-view ray for the RANSAC intersection step.
    # Each entry: {"origin": (3,), "direction": (3,), "weight": float}
    all_ransac_match_rays = []
    total_match_rays = 0
    for j, vn in enumerate(sorted(matches.keys())):
        match_idx, sim = matches[vn][0]
        m_row, m_col = patch_idx_to_rc(match_idx)
        ray_color = view_3d_colors[j % len(view_3d_colors)]

        # Camera for this view
        cam_data_other = build_perspective_camera(cameras[vn])

        # Camera-origin marker in the same colour as its rays
        other_cam_pt = Point(cam_data_other["cam_origin"], c=ray_color, r=15)
        other_cam_pt.name = f"Camera {vn}"
        actors.append(other_cam_pt)

        # Cast 16×16 rays for the matched patch
        u0 = m_col * PATCH_SIZE
        v0 = m_row * PATCH_SIZE
        match_rays = []

        for du in range(PATCH_SIZE):
            for dv in range(PATCH_SIZE):
                origin, direction = compute_ray(cam_data_other,
                                                u0 + du, v0 + dv)
                t_enter, t_exit = ray_aabb_intersect(
                    origin, direction, box_min, box_max)
                if t_enter is None:
                    continue
                match_rays.append({
                    "origin":    origin,
                    "direction": direction,           # needed for RANSAC
                    "p_enter":   origin + t_enter * direction,
                    "p_exit":    origin + t_exit  * direction,
                })

        # Weight matched rays by their cosine similarity (0–1).
        # This means a view with sim=0.95 has nearly full weight while a
        # weaker match contributes proportionally less to RANSAC consensus.
        for ray in match_rays:
            all_ransac_match_rays.append({
                "origin":    ray["origin"],
                "direction": ray["direction"],
                "weight":    float(sim),
            })

        for k, ray in enumerate(match_rays):
            actors.append(Line(ray["origin"], ray["p_enter"],
                               c=ray_color, alpha=0.1, lw=1))
            actors.append(Line(ray["p_enter"], ray["p_exit"],
                               c=ray_color, alpha=0.3, lw=2))
            # Corner entry/exit dots only
            if k in (0, PATCH_SIZE - 1,
                     PATCH_SIZE * (PATCH_SIZE - 1),
                     PATCH_SIZE * PATCH_SIZE - 1):
                actors.append(Point(ray["p_enter"], c=ray_color, r=8))
                actors.append(Point(ray["p_exit"],  c=ray_color, r=6))

        total_match_rays += len(match_rays)
        print(f"    {vn}: patch ({m_row},{m_col}), {len(match_rays)} rays, "
              f"color={ray_color}, sim={sim:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3: RANSAC intersection estimation
    # ══════════════════════════════════════════════════════════════════════════
    #
    # Pool all rays from the reference tube (high weight) and every matched
    # view (weight = cosine similarity).  RANSAC finds the 3-D point that
    # is closest to the largest weighted subset of rays, then refines with
    # a weighted least-squares fit over all inliers.
    # ──────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  RANSAC INTERSECTION")
    print("=" * 60)

    # ── Build unified ray arrays ──────────────────────────────────────────────
    ransac_origins_list    = []
    ransac_directions_list = []
    ransac_weights_list    = []
    ransac_ref_flags       = []

    # Reference tube — every ray gets REFERENCE_RAY_WEIGHT
    for ray in rays:
        ransac_origins_list.append(ray["origin"])
        ransac_directions_list.append(ray["direction"])
        ransac_weights_list.append(REFERENCE_RAY_WEIGHT)
        ransac_ref_flags.append(True)

    # Matched-view rays — weight = cosine similarity of the view
    for entry in all_ransac_match_rays:
        ransac_origins_list.append(entry["origin"])
        ransac_directions_list.append(entry["direction"])
        ransac_weights_list.append(entry["weight"])
        ransac_ref_flags.append(False)

    rans_origins  = np.array(ransac_origins_list,    dtype=np.float64)
    rans_dirs     = np.array(ransac_directions_list, dtype=np.float64)
    rans_weights  = np.array(ransac_weights_list,    dtype=np.float64)
    rans_ref_mask = np.array(ransac_ref_flags,       dtype=bool)

    n_ref_rays   = int(rans_ref_mask.sum())
    n_match_rays = int((~rans_ref_mask).sum())

    print(f"  Rays in pool: {len(rans_origins)} total  "
          f"({n_ref_rays} reference @ w={REFERENCE_RAY_WEIGHT:.1f}  |  "
          f"{n_match_rays} matched @ w=sim)")

    # ── Adaptive inlier threshold — RANSAC_INLIER_FRAC of volume diagonal ────
    vol_diag         = float(np.linalg.norm(box_max - box_min))
    inlier_threshold = max(10.0, RANSAC_INLIER_FRAC * vol_diag)
    print(f"  Volume diagonal: {vol_diag:.1f} vox  →  "
          f"inlier threshold: {inlier_threshold:.1f} vox")

    # ── Run RANSAC ────────────────────────────────────────────────────────────
    best_pt, inlier_mask, residual = ransac_ray_intersection(
        rans_origins,
        rans_dirs,
        rans_weights,
        rans_ref_mask,
        n_iterations     = RANSAC_N_ITER,
        inlier_threshold = inlier_threshold,
        min_inliers      = RANSAC_MIN_INLIERS,
    )

    n_inliers     = int(inlier_mask.sum())
    n_ref_inliers = int((inlier_mask & rans_ref_mask).sum())
    n_mat_inliers = int((inlier_mask & ~rans_ref_mask).sum())

    if best_pt is not None:
        print(f"  ✓ Intersection estimated at: "
              f"({best_pt[0]:.1f}, {best_pt[1]:.1f}, {best_pt[2]:.1f})")
        print(f"    Inliers: {n_inliers}/{len(rans_origins)}  "
              f"({n_ref_inliers} ref + {n_mat_inliers} match)  "
              f"|  mean residual: {residual:.2f} vox")

        # ── Add intersection visualisation to 3D scene ────────────────────
        from vedo import Sphere as VedoSphere, Points as VedoPoints

        # Sphere radius scaled to ~1.5 % of volume diagonal for visibility
        sphere_r = max(5.0, vol_diag * 0.015)

        # Outer semi-transparent halo for contrast against the volume
        halo = VedoSphere(best_pt, r=sphere_r * 1.35)
        halo.color("white").alpha(0.18)
        halo.name = "RANSAC halo"
        actors.append(halo)

        # Inner solid sphere — deep pink so it reads clearly in any colormap
        isect_sphere = VedoSphere(best_pt, r=sphere_r)
        isect_sphere.color("deeppink").alpha(0.90)
        isect_sphere.name = "RANSAC intersection"
        actors.append(isect_sphere)

        # Small dots at each inlier ray's closest-approach point to best_pt.
        # These show which part of the ray bundle agreed on the location.
        closest_pts = []
        for i in np.where(inlier_mask)[0]:
            t = float(np.dot(best_pt - rans_origins[i], rans_dirs[i]))
            t = max(0.0, t)
            closest_pts.append(rans_origins[i] + t * rans_dirs[i])
        if closest_pts:
            inlier_cloud = VedoPoints(
                np.array(closest_pts), r=4, c="deeppink", alpha=0.35
            )
            inlier_cloud.name = "RANSAC inlier closest points"
            actors.append(inlier_cloud)

        ransac_legend = (
            f"RANSAC ●  "
            f"({best_pt[0]:.0f}, {best_pt[1]:.0f}, {best_pt[2]:.0f})  "
            f"res={residual:.1f} vox  "
            f"inliers={n_inliers} ({n_ref_inliers}r+{n_mat_inliers}m)  "
            f"[deeppink]"
        )
    else:
        print("  ✗ RANSAC failed to find a consensus intersection.")
        ransac_legend = "RANSAC: no consensus found"

    # Combined info label — added last so it covers all views
    legend_lines = [
        f"Volume: {specimen}/{image_name}",
        f"Query  {view_name}: patch ({patch_row},{patch_col})  "
        f"{len(rays)} rays  [cyan]",
    ]
    for j, vn in enumerate(sorted(matches.keys())):
        match_idx, sim = matches[vn][0]
        m_row, m_col = patch_idx_to_rc(match_idx)
        legend_lines.append(
            f"Match  {vn}: patch ({m_row},{m_col})  sim={sim:.3f}  "
            f"[{view_3d_colors[j % len(view_3d_colors)]}]"
        )
    legend_lines.append("Yellow=cam origin  filled dots=entry/exit")
    legend_lines.append(ransac_legend)
    info = Text2D("\n".join(legend_lines),
                  pos="top-left", font="Mono", s=0.65, bg="black", alpha=0.7)
    actors.append(info)
    print(f"\n  Total 3D actors: {len(actors)}  "
          f"({len(rays)} query rays + {total_match_rays} match rays + RANSAC sphere)")

    # ── Plot all views with matched patches highlighted ───────────────────────
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import to_rgba

    # ── Dark theme & typography ──────────────────────────────────────────────
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

    # Collect all views to display (query + matched)
    all_views_to_show = [view_name] + sorted(matches.keys())
    n_views = len(all_views_to_show)

    # ── Fixed palette — distinct, vivid colours ──────────────────────────────
    plot_colors = {
        view_name: "#ff4757",  # query = coral-red
    }
    match_palette = ["#2ed573", "#1e90ff", "#ffa502", "#ff6bff",
                     "#7bed9f", "#70a1ff", "#ffda79", "#ff4757",
                     "#a29bfe", "#fd79a8"]
    for j, vn in enumerate(sorted(matches.keys())):
        plot_colors[vn] = match_palette[j % len(match_palette)]

    # Grid layout: aim for a square-ish arrangement
    import math
    n_cols = math.ceil(math.sqrt(n_views))
    n_rows = math.ceil(n_views / n_cols)

    # Cap figure to fit on screen (max ~14" wide, ~9" tall)
    MAX_W, MAX_H = 14, 9
    cell_w = min(MAX_W / n_cols, 5.5)
    cell_h = min(MAX_H / n_rows, 6.2)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_w * n_cols, cell_h * n_rows),
        gridspec_kw={"hspace": 0.35, "wspace": 0.12},
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Embedding Similarity",
        fontsize=20, fontweight="bold", color="white",
        y=0.97,
    )
    fig.text(
        0.5, 0.935,
        f"Query: {view_name}  ·  patch ({patch_row}, {patch_col})"
        f"  ·  {specimen}/{image_name}",
        ha="center", fontsize=12, color="#aaaaaa",
    )


    for i, vn in enumerate(all_views_to_show):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]

        # Get the view image
        view_img = cameras[vn]["mip"]
        ax.imshow(view_img)

        color = plot_colors[vn]
        color_rgba = to_rgba(color, alpha=0.30)

        if vn == view_name:
            # Query view
            pr, pc = patch_row, patch_col
            label = "QUERY"
            sim_val = None
        else:
            match_idx, sim_val = matches[vn][0]
            pr, pc = patch_idx_to_rc(match_idx)
            label = f"sim = {sim_val:.4f}"

        px = pc * PATCH_SIZE
        py = pr * PATCH_SIZE

        # ── Filled highlight ─────────────────────────────────────────────
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=0, facecolor=color_rgba,
        ))
        # ── Crisp border ─────────────────────────────────────────────────
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=2.5, edgecolor=color, facecolor="none",
        ))
        # ── Crosshair lines (full-width / full-height, very thin) ────────
        cx = px + PATCH_SIZE / 2
        cy = py + PATCH_SIZE / 2
        ax.axhline(cy, color=color, linewidth=0.5, alpha=0.45, linestyle="--")
        ax.axvline(cx, color=color, linewidth=0.5, alpha=0.45, linestyle="--")

        # ── Title ────────────────────────────────────────────────────────
        ax.set_title(
            vn.replace("_", " ").upper(),
            fontsize=12, fontweight="bold", color=color, pad=8,
        )

        # ── Badge below image with similarity / label ────────────────────
        ax.text(
            0.5, -0.04, label,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=color, edgecolor="none", alpha=0.85,
            ),
        )


        # ── Clean up main axes ───────────────────────────────────────────
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
            spine.set_linewidth(1.0)

    # Hide unused axes
    for i in range(n_views, n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes[row_idx, col_idx].axis("off")

    plt.subplots_adjust(top=0.90, bottom=0.05)
    print("\n>>> Showing 2D similarity grid. Close the window to proceed to 3D view.")
    plt.show()

    # Reset rcParams so the dark theme doesn't bleed into other plots
    plt.rcdefaults()

    # ── 3D visualization — shown last, after all tube rays are cast ───────────
    print("\n>>> Opening 3D viewer with query + matched patch-ray tubes. Close when done.")
    plt_3d = Plotter(axes=1, title=f"Patch Ray Tubes — {specimen}/{image_name} | query={view_name}")
    plt_3d.show(*actors, interactive=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4: Dedicated 3D view of the estimated intersection point
    # ══════════════════════════════════════════════════════════════════════════
    if best_pt is not None:
        from vedo import Sphere as VedoSphere

        print("\n>>> Opening 3D viewer with estimated intersection point. Close when done.")

        pt_actors = []

        # Semi-transparent volume for spatial context
        vol_ctx = VedoVolume(vol_data, spacing=original_vol.spacing(),
                             origin=original_vol.origin())
        vol_ctx.mode(0).cmap("bone").alpha([0, 0, 0.3, 0.6])
        pt_actors.append(vol_ctx)

        # Outer glow / halo
        halo_pt = VedoSphere(best_pt, r=ESTIMATED_POINT_RADIUS * 1.4)
        halo_pt.color("white").alpha(0.20)
        halo_pt.name = "Estimated point halo"
        pt_actors.append(halo_pt)

        # Main sphere at the estimated intersection
        est_sphere = VedoSphere(best_pt, r=ESTIMATED_POINT_RADIUS)
        est_sphere.color("deeppink").alpha(0.92)
        est_sphere.name = "Estimated intersection"
        pt_actors.append(est_sphere)

        # Info label
        info_pt = Text2D(
            f"Estimated intersection\n"
            f"  ({best_pt[0]:.1f}, {best_pt[1]:.1f}, {best_pt[2]:.1f})\n"
            f"  radius={ESTIMATED_POINT_RADIUS:.1f} vox\n"
            f"  residual={residual:.2f} vox\n"
            f"  inliers={n_inliers}/{len(rans_origins)}",
            pos="top-left", font="Mono", s=0.7, bg="black", alpha=0.75,
        )
        pt_actors.append(info_pt)

        plt_pt = Plotter(
            axes=1,
            title=f"Estimated Intersection — {specimen}/{image_name}",
        )
        plt_pt.show(*pt_actors, interactive=True)
    else:
        print("\n  Skipping intersection point viewer — no RANSAC result.")

    print("Done.")


if __name__ == "__main__":
    main()