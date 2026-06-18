"""
raycast_rotated_reference.py
----------------------------
Rotation-augmented reference matching experiment with RANSAC intersection.

Pipeline
--------
  1. Take view_000 of bugNIST900 GH as reference.
  2. Create 3 rotated copies (90°, 180°, 270° CCW) of the reference.
  3. Embed all 4 rotated versions through DINOv3
     (cached in  data/rotation_experiment/exp1/).
  4. Click on the original reference view to select a patch.
  5. Automatically map the clicked patch to the corresponding position
     in each rotated version (known from the rotation geometry).
  6. For each of the other views (view_001 … view_014), find the single
     best matching patch across all 4 reference embeddings.
  7. Plot 2D grid: 4 reference views on the left with patches highlighted,
     matched views on the right coloured by whichever reference produced
     the best match.
  8. Cast rays through the reference patch and all matched patches, run
     RANSAC to estimate the 3-D intersection, and show the result in
     a vedo 3-D viewer.

Usage:
  python src/checks/raycast_rotated_reference.py
  python src/checks/raycast_rotated_reference.py --device cuda
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

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
EMB_ROOT  = PROJECT_ROOT / "data" / "embeddings" / "bugNIST_900"
MODEL_PATH = PROJECT_ROOT / "models" / "vitb16"
EXP_DIR   = PROJECT_ROOT / "data" / "rotation_experiment" / "exp1"

# ── Constants ─────────────────────────────────────────────────────────────────
SPECIMEN   = "GH"
IMAGE_NAME = "gras_9_041"
REF_VIEW   = "view_000"
PATCH_SIZE = 16
IMG_SIZE   = 1440
GRID_SIZE  = IMG_SIZE // PATCH_SIZE   # 90

N_PREFIX_TOKENS = 5   # 1 [CLS] + 4 register tokens (DINOv3)

# Rotation settings
ROTATION_ANGLES = [0, 90, 180, 270]
ROTATION_KEYS   = ["rot_000", "rot_090", "rot_180", "rot_270"]
ROTATION_LABELS = ["0°", "90°", "180°", "270°"]
ROTATION_COLORS = ["#ff4757", "#1e90ff", "#2ed573", "#ffa502"]

# ── RANSAC / intersection constants ───────────────────────────────────────────

RANSAC_N_ITER      = 400
RANSAC_INLIER_FRAC = 0.04
RANSAC_MIN_INLIERS = 10

ESTIMATED_POINT_RADIUS = 55.0   # voxel units


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING & EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(device: torch.device):
    """Load the local DINOv3 model and image processor (resize disabled)."""
    print(f"Loading DINOv3 model from: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
    processor.do_resize = False          # keep native 1440×1440 resolution
    model = AutoModel.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()
    print(f"  Model loaded on {device}.")
    return processor, model


@torch.no_grad()
def embed_image(pil_img: Image.Image, processor, model, device) -> np.ndarray:
    """
    Run a single PIL image through DINOv3 and return patch embeddings.

    Returns
    -------
    np.ndarray, shape (n_patches, hidden_dim), dtype float32
    """
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state                       # (1, n_tokens, hidden)
    patch_tokens = hidden[0, N_PREFIX_TOKENS:, :]            # (n_patches, hidden)
    return patch_tokens.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE ROTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def rotate_image(pil_img: Image.Image, angle: int) -> Image.Image:
    """Rotate a PIL image by *angle* degrees CCW using exact transposition."""
    if angle == 0:
        return pil_img.copy()
    elif angle == 90:
        return pil_img.transpose(Image.Transpose.ROTATE_90)
    elif angle == 180:
        return pil_img.transpose(Image.Transpose.ROTATE_180)
    elif angle == 270:
        return pil_img.transpose(Image.Transpose.ROTATE_270)
    raise ValueError(f"Unsupported angle: {angle}")


def map_patch_to_rotation(row: int, col: int, angle: int,
                          G: int = GRID_SIZE - 1) -> tuple[int, int]:
    """
    Given a patch (row, col) in the **original** image, return the
    (row, col) where the same content appears in the image rotated by
    *angle* degrees CCW.

    Derivation (for a square grid of size G+1):
        0°:   (r, c)       → (r,     c    )
       90°:   (r, c)       → (G − c, r    )
      180°:   (r, c)       → (G − r, G − c)
      270°:   (r, c)       → (c,     G − r)
    """
    if angle == 0:
        return (row, col)
    elif angle == 90:
        return (G - col, row)
    elif angle == 180:
        return (G - row, G - col)
    elif angle == 270:
        return (col, G - row)
    raise ValueError(f"Unsupported angle: {angle}")


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING GENERATION FOR ROTATED REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rotated_embeddings(device: torch.device) -> dict:
    """
    Create 3 rotated copies of the reference view, embed all 4, and save
    to ``EXP_DIR / "rotated_embeddings.pkl"``.

    Returns the dict ``{rot_key: np.ndarray(n_patches, hidden)}``.
    """
    print("\n" + "=" * 60)
    print("  GENERATING ROTATED REFERENCE EMBEDDINGS")
    print("=" * 60)

    ref_path = PROC_ROOT / SPECIMEN / IMAGE_NAME / f"{REF_VIEW}.png"
    if not ref_path.exists():
        print(f"[ERROR] Reference view not found: {ref_path}")
        sys.exit(1)

    ref_img = Image.open(ref_path).convert("RGB")
    print(f"  Reference image : {ref_path.relative_to(PROJECT_ROOT)}")
    print(f"  Image size      : {ref_img.size}")

    processor, model = load_model(device)

    rotated_embeddings: dict[str, np.ndarray] = {}
    for angle, key in zip(ROTATION_ANGLES, ROTATION_KEYS):
        print(f"\n  Rotating {angle}° and embedding …")
        rot_img = rotate_image(ref_img, angle)
        emb = embed_image(rot_img, processor, model, device)
        rotated_embeddings[key] = emb
        print(f"    Embedding shape: {emb.shape}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXP_DIR / "rotated_embeddings.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(rotated_embeddings, f)
    print(f"\n  ✓ Saved → {out_path.relative_to(PROJECT_ROOT)}")

    return rotated_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
#  MATCHING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_match_across_references(
    ref_embeddings: list[np.ndarray],
    target_matrix: np.ndarray,
) -> tuple[int, float, int]:
    """
    For a single target view, find the best matching patch across
    multiple reference embeddings.

    Parameters
    ----------
    ref_embeddings : list of (hidden_dim,) vectors — one per reference
    target_matrix  : (n_patches, hidden_dim) — all patches in target view

    Returns
    -------
    best_patch_idx : int   — flat patch index in the target view
    best_sim       : float — cosine similarity of the best match
    best_ref_idx   : int   — which reference (0–3) produced the best match
    """
    T = target_matrix.astype(np.float64)
    T_norms = np.linalg.norm(T, axis=1, keepdims=True)
    T_norms = np.where(T_norms < 1e-12, 1.0, T_norms)
    T_normed = T / T_norms

    best_patch_idx = -1
    best_sim       = -np.inf
    best_ref_idx   = -1

    for ref_idx, ref_emb in enumerate(ref_embeddings):
        q = ref_emb.astype(np.float64)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        sims = T_normed @ q_norm                         # (n_patches,)
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])
        if top_sim > best_sim:
            best_sim       = top_sim
            best_patch_idx = top_idx
            best_ref_idx   = ref_idx

    return best_patch_idx, best_sim, best_ref_idx


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


def cast_patch_rays(cam_data, patch_row, patch_col, box_min, box_max):
    """
    Cast 16×16 rays for one patch and return the list of ray dicts.

    Each dict has keys: origin, direction, p_enter, p_exit.
    """
    rays = []
    u_start = patch_col * PATCH_SIZE
    v_start = patch_row * PATCH_SIZE

    for du in range(PATCH_SIZE):
        for dv in range(PATCH_SIZE):
            u = u_start + du
            v = v_start + dv
            origin, direction = compute_ray(cam_data, u, v)
            t_enter, t_exit = ray_aabb_intersect(origin, direction,
                                                  box_min, box_max)
            if t_enter is None:
                continue
            rays.append({
                "origin":    origin,
                "direction": direction,
                "p_enter":   origin + t_enter * direction,
                "p_exit":    origin + t_exit  * direction,
            })
    return rays


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
        description="Rotation-augmented reference matching experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device", default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for DINOv3 forward passes.",
    )
    args = parser.parse_args()

    # ── Resolve device ────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 1: Load / generate rotated reference embeddings
    # ══════════════════════════════════════════════════════════════════════════

    emb_path = EXP_DIR / "rotated_embeddings.pkl"

    if emb_path.exists():
        print(f"Rotated embeddings found: {emb_path.relative_to(PROJECT_ROOT)}")
        with open(emb_path, "rb") as f:
            rotated_embs = pickle.load(f)
        print(f"  Keys: {list(rotated_embs.keys())}")
        first_key = next(iter(rotated_embs))
        print(f"  Shape per rotation: {rotated_embs[first_key].shape}")
    else:
        print(f"Rotated embeddings NOT found at {emb_path.relative_to(PROJECT_ROOT)}")
        rotated_embs = generate_rotated_embeddings(device)

    # ── Load regular embeddings for the other views ───────────────────────────
    reg_emb_path = EMB_ROOT / SPECIMEN / f"{IMAGE_NAME}.pkl"
    if not reg_emb_path.exists():
        print(f"[ERROR] Regular embeddings not found: {reg_emb_path}")
        print("  Run  python src/generate_embeddings.py  first.")
        sys.exit(1)

    with open(reg_emb_path, "rb") as f:
        regular_embs = pickle.load(f)
    print(f"Loaded regular embeddings for {len(regular_embs)} views.")

    # Other views = everything except the reference
    other_view_names = sorted(vn for vn in regular_embs if vn != REF_VIEW)
    print(f"  Target views: {other_view_names}")

    # ── Load camera data ──────────────────────────────────────────────────────
    cam_pkl = PROC_ROOT / SPECIMEN / IMAGE_NAME / "cameras.pkl"
    print(f"Loading cameras: {cam_pkl.relative_to(PROJECT_ROOT)}")
    with open(cam_pkl, "rb") as f:
        cameras = pickle.load(f)

    volume_shape = cameras[REF_VIEW]["volume_shape"]
    Vx, Vy, Vz = volume_shape
    print(f"Volume shape: {volume_shape}")

    # ── Load reference image + rotated versions for display ───────────────────
    ref_path = PROC_ROOT / SPECIMEN / IMAGE_NAME / f"{REF_VIEW}.png"
    ref_img = Image.open(ref_path).convert("RGB")
    ref_img_np = np.array(ref_img)

    rotated_images_np: dict[int, np.ndarray] = {}
    for angle in ROTATION_ANGLES:
        rotated_images_np[angle] = np.array(rotate_image(ref_img, angle))

    # ── Load view images for the other views ──────────────────────────────────
    other_view_images: dict[str, np.ndarray] = {}
    for vn in other_view_names:
        vp = PROC_ROOT / SPECIMEN / IMAGE_NAME / f"{vn}.png"
        other_view_images[vn] = np.array(Image.open(vp).convert("RGB"))

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2: Click on the reference view to select a patch
    # ══════════════════════════════════════════════════════════════════════════

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(ref_img_np)
    ax.set_title(f"Click on {REF_VIEW} to select a patch\n(close window when done)")
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

    # ── Convert last click to patch coordinates ───────────────────────────────
    u_click, v_click = clicked_points[-1]
    patch_col = min(int(u_click) // PATCH_SIZE, GRID_SIZE - 1)
    patch_row = min(int(v_click) // PATCH_SIZE, GRID_SIZE - 1)
    print(f"\nClicked pixel: ({u_click:.1f}, {v_click:.1f})")
    print(f"  → Patch (row={patch_row}, col={patch_col})")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3: Map clicked patch to all 4 rotated reference versions
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  REFERENCE PATCH MAPPING")
    print("=" * 60)

    ref_patch_info: list[dict] = []          # one entry per rotation
    ref_query_embeddings: list[np.ndarray] = []

    for i, (angle, key, label) in enumerate(
        zip(ROTATION_ANGLES, ROTATION_KEYS, ROTATION_LABELS)
    ):
        r_row, r_col = map_patch_to_rotation(patch_row, patch_col, angle)
        flat_idx = r_row * GRID_SIZE + r_col
        emb = rotated_embs[key][flat_idx]    # (hidden_dim,)

        ref_patch_info.append({
            "angle":    angle,
            "key":      key,
            "label":    label,
            "row":      r_row,
            "col":      r_col,
            "flat_idx": flat_idx,
        })
        ref_query_embeddings.append(emb)

        print(f"  {label:>4s}  →  patch (row={r_row}, col={r_col}), "
              f"flat_idx={flat_idx}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4: For each other view, find the best match across 4 references
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  BEST MATCHES (across 4 rotated references)")
    print("=" * 60)

    # match_results[view_name] = (best_patch_idx, best_sim, best_ref_idx)
    match_results: dict[str, tuple[int, float, int]] = {}

    for vn in other_view_names:
        target_matrix = regular_embs[vn]
        patch_idx, sim, ref_idx = find_best_match_across_references(
            ref_query_embeddings, target_matrix,
        )
        match_results[vn] = (patch_idx, sim, ref_idx)

        m_row = patch_idx // GRID_SIZE
        m_col = patch_idx % GRID_SIZE
        print(f"  {vn}:  patch ({m_row:2d},{m_col:2d})  "
              f"sim={sim:.4f}  ← ref {ROTATION_LABELS[ref_idx]}")

    # ── Count how many views matched each reference ───────────────────────────
    ref_counts = [0] * len(ROTATION_ANGLES)
    for _, (_, _, ref_idx) in match_results.items():
        ref_counts[ref_idx] += 1
    print("\n  Reference usage:")
    for i, (label, cnt) in enumerate(zip(ROTATION_LABELS, ref_counts)):
        print(f"    {label}: {cnt} views")

    # ── Determine the winning reference (most matches, lowest index on tie) ──
    max_count = max(ref_counts)
    winning_ref_idx = ref_counts.index(max_count)    # first with max → lowest
    winning_label = ROTATION_LABELS[winning_ref_idx]
    winning_color = ROTATION_COLORS[winning_ref_idx]
    print(f"\n  ★ Winning reference: {winning_label} "
          f"({ref_counts[winning_ref_idx]} matches)")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5: RANSAC intersection estimation
    # ══════════════════════════════════════════════════════════════════════════
    #
    # Pool only matched-view rays (reference rays excluded).
    # RANSAC finds the 3-D point closest to the largest weighted subset
    # of rays, then refines with a weighted least-squares fit.
    # ──────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  RANSAC INTERSECTION")
    print("=" * 60)

    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([Vx, Vy, Vz], dtype=np.float64)

    ref_cam_data = build_perspective_camera(cameras[REF_VIEW])

    # ── Cast reference tube rays (kept for visualisation only) ───────────────
    print(f"\n  Casting reference tube: {REF_VIEW} patch "
          f"({patch_row},{patch_col}) …")
    ref_rays = cast_patch_rays(ref_cam_data, patch_row, patch_col,
                               box_min, box_max)
    print(f"    {len(ref_rays)} valid rays")

    # ── Cast matched-view tube rays ───────────────────────────────────────────
    print("  Casting matched-view tubes …")
    # match_rays_by_view[vn] = (ray_list, sim, ref_idx)
    match_rays_by_view: dict[str, tuple[list, float, int]] = {}

    for vn in other_view_names:
        patch_idx, sim, ref_idx = match_results[vn]
        m_row = patch_idx // GRID_SIZE
        m_col = patch_idx % GRID_SIZE

        cam_data_other = build_perspective_camera(cameras[vn])
        mrays = cast_patch_rays(cam_data_other, m_row, m_col,
                                box_min, box_max)
        match_rays_by_view[vn] = (mrays, sim, ref_idx)
        print(f"    {vn}: patch ({m_row},{m_col}), {len(mrays)} rays, "
              f"sim={sim:.4f}")

    # ── Build ray pool from matched views only (no reference rays) ────────────
    rans_origins_list    = []
    rans_directions_list = []
    rans_weights_list    = []

    total_match_rays = 0
    for vn in other_view_names:
        mrays, sim, _ = match_rays_by_view[vn]
        for ray in mrays:
            rans_origins_list.append(ray["origin"])
            rans_directions_list.append(ray["direction"])
            rans_weights_list.append(float(sim))
        total_match_rays += len(mrays)

    rans_origins  = np.array(rans_origins_list,    dtype=np.float64)
    rans_dirs     = np.array(rans_directions_list, dtype=np.float64)
    rans_weights  = np.array(rans_weights_list,    dtype=np.float64)

    n_match_rays = len(rans_origins)

    print(f"\n  Rays in pool: {n_match_rays} matched-view rays  "
          f"(reference rays excluded)")

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
    #  STEP 6: 2D Visualisation
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  PLOTTING (2D)")
    print("=" * 60)

    from matplotlib.patches import Rectangle
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

    # ── Layout: 4 reference cols on the left  |  grid of matched views ────────
    n_matched = len(other_view_names)
    match_cols = 5
    match_rows = math.ceil(n_matched / match_cols)

    total_rows = 1 + match_rows
    total_cols = max(4, match_cols)

    fig, axes = plt.subplots(
        total_rows, total_cols,
        figsize=(total_cols * 2.8, total_rows * 3.0),
        gridspec_kw={"hspace": 0.40, "wspace": 0.15},
    )
    if total_rows == 1:
        axes = axes[np.newaxis, :]

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "Rotated Reference Matching",
        fontsize=20, fontweight="bold", color="white", y=0.97,
    )
    fig.text(
        0.5, 0.935,
        f"Query: {REF_VIEW} · patch ({patch_row}, {patch_col}) · "
        f"{SPECIMEN}/{IMAGE_NAME}  ·  "
        f"★ winning ref: {winning_label}",
        ha="center", fontsize=12, color="#aaaaaa",
    )

    # ── Row 0: four reference images ──────────────────────────────────────────
    for i, info in enumerate(ref_patch_info):
        ax = axes[0, i]
        angle = info["angle"]
        color = ROTATION_COLORS[i]

        ax.imshow(rotated_images_np[angle])

        pr, pc = info["row"], info["col"]
        px = pc * PATCH_SIZE
        py = pr * PATCH_SIZE

        # Filled highlight
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=0, facecolor=to_rgba(color, alpha=0.35),
        ))
        # Crisp border
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=3, edgecolor=color, facecolor="none",
        ))
        # Crosshairs
        cx = px + PATCH_SIZE / 2
        cy = py + PATCH_SIZE / 2
        ax.axhline(cy, color=color, linewidth=0.5, alpha=0.45, linestyle="--")
        ax.axvline(cx, color=color, linewidth=0.5, alpha=0.45, linestyle="--")

        title_str = f"REF {info['label']}"
        if i == winning_ref_idx:
            title_str += " ★"
        ax.set_title(
            title_str,
            fontsize=11, fontweight="bold", color=color, pad=6,
        )
        # Badge
        badge_text = f"patch ({pr},{pc})"
        if i == winning_ref_idx:
            badge_text += f"  ({ref_counts[i]} matches)"
        ax.text(
            0.5, -0.04, badge_text,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=color, edgecolor="none", alpha=0.85),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0 if i != winning_ref_idx else 3.5)

    # Hide extra cols in the reference row (if total_cols > 4)
    for j in range(4, total_cols):
        axes[0, j].axis("off")

    # ── Rows 1+: matched views ───────────────────────────────────────────────
    for idx, vn in enumerate(other_view_names):
        r = 1 + idx // match_cols
        c = idx % match_cols
        ax = axes[r, c]

        ax.imshow(other_view_images[vn])

        patch_idx, sim, ref_idx = match_results[vn]
        m_row = patch_idx // GRID_SIZE
        m_col = patch_idx % GRID_SIZE
        color = ROTATION_COLORS[ref_idx]

        px = m_col * PATCH_SIZE
        py = m_row * PATCH_SIZE

        # Filled highlight
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=0, facecolor=to_rgba(color, alpha=0.30),
        ))
        # Border
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=2.5, edgecolor=color, facecolor="none",
        ))
        # Crosshairs
        cx = px + PATCH_SIZE / 2
        cy = py + PATCH_SIZE / 2
        ax.axhline(cy, color=color, linewidth=0.5, alpha=0.45, linestyle="--")
        ax.axvline(cx, color=color, linewidth=0.5, alpha=0.45, linestyle="--")

        ax.set_title(
            vn.replace("_", " ").upper(),
            fontsize=10, fontweight="bold", color=color, pad=6,
        )
        ax.text(
            0.5, -0.04,
            f"sim={sim:.4f}  ← ref {ROTATION_LABELS[ref_idx]}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=color, edgecolor="none", alpha=0.85),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
            spine.set_linewidth(1.0)

    # Hide unused cells
    for idx in range(n_matched, match_rows * match_cols):
        r = 1 + idx // match_cols
        c = idx % match_cols
        axes[r, c].axis("off")

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_y = 0.02
    for i, (label, color) in enumerate(zip(ROTATION_LABELS, ROTATION_COLORS)):
        marker = " ★" if i == winning_ref_idx else ""
        fig.text(
            0.15 + i * 0.20, legend_y,
            f"■ ref {label}{marker}",
            fontsize=11, fontweight="bold", color=color,
            ha="center",
        )

    plt.subplots_adjust(top=0.90, bottom=0.06)
    print("\n>>> Showing 2D results. Close the window to proceed to intersection view.")
    plt.show()
    plt.rcdefaults()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 8: Dedicated intersection point viewer
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
            f"Estimated intersection\n"
            f"  ({best_pt[0]:.1f}, {best_pt[1]:.1f}, {best_pt[2]:.1f})\n"
            f"  radius={ESTIMATED_POINT_RADIUS:.1f} vox\n"
            f"  residual={residual:.2f} vox\n"
            f"  inliers={n_inliers}/{len(rans_origins)}\n"
            f"  winning ref: {winning_label}",
            pos="top-left", font="Mono", s=0.7, bg="black", alpha=0.75,
        )
        pt_actors.append(info_pt)

        plt_pt = Plotter(
            axes=1,
            title=f"Estimated Intersection — {SPECIMEN}/{IMAGE_NAME}",
        )
        plt_pt.show(*pt_actors, interactive=True)
    else:
        print("\n  Skipping intersection point viewer — no RANSAC result.")

    print("Done.")


if __name__ == "__main__":
    main()
