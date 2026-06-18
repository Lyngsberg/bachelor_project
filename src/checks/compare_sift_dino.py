"""
compare_sift_dino.py
--------------------
Side-by-side comparison of SIFT vs DINOv3 (rotation-augmented) matching
and triangulation from a single click on the reference view.

Pipeline
--------
  1. Load the reference image and extract SIFT features.
  2. Load/generate DINOv3 rotated-reference embeddings + regular view embeddings.
  3. Click once on the reference view.
  4. SIFT path: find nearest keypoint, match descriptor to all other views.
  5. DINOv3 path: map click to patch, use 4 rotated embeddings, find best
     match in each other view.
  6. Combined 2D plot:
       Row 0  — Reference image (click + SIFT keypoint shown)
       Row 1  — SIFT matched views
       Row 2  — DINOv3 matched views
  7. RANSAC on both ray bundles independently.
  8. Side-by-side 3D intersection viewers (SIFT left, DINOv3 right).

Usage:
  python src/checks/compare_sift_dino.py
  python src/checks/compare_sift_dino.py --device cuda
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import cv2
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
RAW_ROOT   = PROJECT_ROOT / "data" / "raw"       / "bugNIST_900"
PROC_ROOT  = PROJECT_ROOT / "data" / "processed" / "bugNIST_900"
EMB_ROOT   = PROJECT_ROOT / "data" / "embeddings" / "bugNIST_900"
MODEL_PATH = PROJECT_ROOT / "models" / "dinov3_local"
EXP_DIR    = PROJECT_ROOT / "data" / "rotation_experiment" / "exp1"

# ── Constants ─────────────────────────────────────────────────────────────────
SPECIMEN   = "GH"
IMAGE_NAME = "gras_9_043"
REF_VIEW   = "view_000"
IMG_SIZE   = 1440

# DINOv3 constants
PATCH_SIZE      = 16
GRID_SIZE       = IMG_SIZE // PATCH_SIZE   # 90
N_PREFIX_TOKENS = 5   # 1 [CLS] + 4 register tokens

# Rotation settings (DINOv3 only)
ROTATION_ANGLES = [0, 90, 180, 270]
ROTATION_KEYS   = ["rot_000", "rot_090", "rot_180", "rot_270"]
ROTATION_LABELS = ["0°", "90°", "180°", "270°"]
ROTATION_COLORS = ["#ff4757", "#1e90ff", "#2ed573", "#ffa502"]

# SIFT parameters
SIFT_N_FEATURES      = 0
SIFT_N_OCTAVE_LAYERS = 3
SIFT_CONTRAST_THRESH = 0.04
SIFT_EDGE_THRESH     = 10
SIFT_SIGMA           = 1.6
LOWE_RATIO           = 0.75

# Ray / RANSAC constants
REFERENCE_RAY_WEIGHT   = 1.0
RAY_PATCH_HALF         = 8
RANSAC_N_ITER          = 400
RANSAC_INLIER_FRAC     = 0.04
RANSAC_MIN_INLIERS     = 10
ESTIMATED_POINT_RADIUS = 55.0


# ═══════════════════════════════════════════════════════════════════════════════
#  DINOv3 MODEL & EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dino_model(device: torch.device):
    """Load the local DINOv3 model and image processor (resize disabled)."""
    print(f"Loading DINOv3 model from: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
    processor.do_resize = False
    model = AutoModel.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()
    print(f"  Model loaded on {device}.")
    return processor, model


@torch.no_grad()
def embed_image(pil_img: Image.Image, processor, model, device) -> np.ndarray:
    """Run a single PIL image through DINOv3, return patch embeddings."""
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    hidden = outputs.last_hidden_state
    patch_tokens = hidden[0, N_PREFIX_TOKENS:, :]
    return patch_tokens.cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  IMAGE ROTATION HELPERS  (DINOv3)
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
    """Map a patch (row, col) in the original image to the rotated version."""
    if angle == 0:
        return (row, col)
    elif angle == 90:
        return (G - col, row)
    elif angle == 180:
        return (G - row, G - col)
    elif angle == 270:
        return (col, G - row)
    raise ValueError(f"Unsupported angle: {angle}")


def generate_rotated_embeddings(device: torch.device) -> dict:
    """Create 4 rotated copies of reference, embed all, and cache."""
    print("\n" + "=" * 60)
    print("  GENERATING ROTATED REFERENCE EMBEDDINGS")
    print("=" * 60)

    ref_path = PROC_ROOT / SPECIMEN / IMAGE_NAME / f"{REF_VIEW}.png"
    ref_img = Image.open(ref_path).convert("RGB")
    processor, model = load_dino_model(device)

    rotated_embeddings: dict[str, np.ndarray] = {}
    for angle, key in zip(ROTATION_ANGLES, ROTATION_KEYS):
        print(f"  Rotating {angle}° and embedding …")
        rot_img = rotate_image(ref_img, angle)
        emb = embed_image(rot_img, processor, model, device)
        rotated_embeddings[key] = emb
        print(f"    Embedding shape: {emb.shape}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXP_DIR / "rotated_embeddings.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(rotated_embeddings, f)
    print(f"  ✓ Saved → {out_path.relative_to(PROJECT_ROOT)}")
    return rotated_embeddings


# ═══════════════════════════════════════════════════════════════════════════════
#  DINOv3 MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_match_across_references(
    ref_embeddings: list[np.ndarray],
    target_matrix: np.ndarray,
) -> tuple[int, float, int]:
    """
    For a single target view, find the best matching patch across multiple
    reference embeddings (one per rotation).
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
        sims = T_normed @ q_norm
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])
        if top_sim > best_sim:
            best_sim       = top_sim
            best_patch_idx = top_idx
            best_ref_idx   = ref_idx

    return best_patch_idx, best_sim, best_ref_idx


# ═══════════════════════════════════════════════════════════════════════════════
#  SIFT EXTRACTION & MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_sift_features(image_np: np.ndarray):
    """Extract SIFT keypoints and descriptors from a numpy RGB image."""
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
    """Find the index of the keypoint nearest to pixel (u, v)."""
    pts = np.array([kp.pt for kp in keypoints])
    dists = np.sqrt((pts[:, 0] - u) ** 2 + (pts[:, 1] - v) ** 2)
    return int(np.argmin(dists))


def match_descriptor_to_view(
    ref_descriptor: np.ndarray,
    target_descriptors: np.ndarray,
    ratio: float = LOWE_RATIO,
) -> tuple[int, float] | None:
    """Match a single ref descriptor to a target view via BFMatcher + ratio test."""
    if target_descriptors is None or len(target_descriptors) < 2:
        return None

    bf = cv2.BFMatcher(cv2.NORM_L2)
    ref_desc = ref_descriptor.reshape(1, -1).astype(np.float32)
    matches = bf.knnMatch(ref_desc, target_descriptors.astype(np.float32), k=2)

    if len(matches) == 0 or len(matches[0]) < 2:
        return None

    m, n = matches[0]
    # Always return best match (ratio test just for info)
    return m.trainIdx, m.distance


# ═══════════════════════════════════════════════════════════════════════════════
#  RAY COMPUTATION (shared)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ray(cam_data, u, v):
    """Compute world-space ray origin and direction for pixel (u, v)."""
    K_inv  = cam_data["K_inv"]
    right  = cam_data["right"]
    up     = cam_data["up"]
    look   = cam_data["look"]
    origin = cam_data["cam_origin"]

    pix = np.array([u + 0.5, v + 0.5, 1.0])
    d_cam = K_inv @ pix
    d_world = d_cam[0] * right + (-d_cam[1]) * up + d_cam[2] * look
    d_world = d_world / np.linalg.norm(d_world)
    return origin.copy(), d_world


def ray_aabb_intersect(origin, direction, box_min, box_max):
    """Parametric entry/exit of a ray with an AABB."""
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
    """Cast 16×16 rays for one DINOv3 patch."""
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
                "origin": origin, "direction": direction,
                "p_enter": origin + t_enter * direction,
                "p_exit":  origin + t_exit  * direction,
            })
    return rays


def cast_keypoint_rays(cam_data, kp_x, kp_y, box_min, box_max,
                       half_size=RAY_PATCH_HALF):
    """Cast rays in a square window around a SIFT keypoint."""
    rays = []
    u_start = max(0, int(kp_x) - half_size)
    v_start = max(0, int(kp_y) - half_size)
    u_end   = min(IMG_SIZE, int(kp_x) + half_size)
    v_end   = min(IMG_SIZE, int(kp_y) + half_size)
    for u in range(u_start, u_end):
        for v in range(v_start, v_end):
            origin, direction = compute_ray(cam_data, u, v)
            t_enter, t_exit = ray_aabb_intersect(origin, direction,
                                                  box_min, box_max)
            if t_enter is None:
                continue
            rays.append({
                "origin": origin, "direction": direction,
                "p_enter": origin + t_enter * direction,
                "p_exit":  origin + t_exit  * direction,
            })
    return rays


# ═══════════════════════════════════════════════════════════════════════════════
#  RANSAC HELPERS (shared)
# ═══════════════════════════════════════════════════════════════════════════════

def closest_point_to_two_rays(o1, d1, o2, d2):
    """Midpoint of shortest segment between two skew lines."""
    cross = np.cross(d1, d2)
    denom = np.dot(cross, cross)
    if denom < 1e-12:
        return None
    w  = o2 - o1
    t1 = np.dot(np.cross(w, d2), cross) / denom
    t2 = np.dot(np.cross(w, d1), cross) / denom
    return 0.5 * ((o1 + t1 * d1) + (o2 + t2 * d2))


def point_to_ray_distances_batch(point, origins, directions):
    """Vectorised perpendicular distance from *point* to each ray."""
    diff     = point - origins
    proj_len = np.einsum("ni,ni->n", diff, directions)
    perp     = diff - proj_len[:, np.newaxis] * directions
    return np.linalg.norm(perp, axis=1)


def weighted_least_squares_intersection(origins, directions, weights):
    """WLS intersection point minimising weighted perpendicular distances."""
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
    origins, directions, weights, ref_mask,
    n_iterations=300, inlier_threshold=15.0, min_inliers=10,
):
    """RANSAC estimation of the 3-D intersection point for a ray bundle."""
    N         = len(origins)
    ref_idx   = np.where( ref_mask)[0]
    other_idx = np.where(~ref_mask)[0]

    if len(ref_idx) == 0 or len(other_idx) == 0 or N < 2:
        pt = weighted_least_squares_intersection(origins, directions, weights)
        return pt, np.ones(N, dtype=bool), np.nan

    best_point   = None
    best_score   = -np.inf
    best_inliers = np.zeros(N, dtype=bool)

    for _ in range(n_iterations):
        i1 = int(np.random.choice(ref_idx))
        i2 = int(np.random.choice(other_idx))
        candidate = closest_point_to_two_rays(
            origins[i1], directions[i1], origins[i2], directions[i2],
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
        best_point, origins[best_inliers], directions[best_inliers],
    )
    residual = float(np.mean(inlier_dists))
    return best_point, best_inliers, residual


def build_ransac_arrays(ref_rays, match_rays_by_view, matched_views,
                        ref_weight=REFERENCE_RAY_WEIGHT):
    """Build unified ray arrays for RANSAC from ref + matched-view rays."""
    origins_list    = []
    directions_list = []
    weights_list    = []
    ref_flags       = []

    for ray in ref_rays:
        origins_list.append(ray["origin"])
        directions_list.append(ray["direction"])
        weights_list.append(ref_weight)
        ref_flags.append(True)

    for vn in matched_views:
        mrays, weight = match_rays_by_view[vn]
        for ray in mrays:
            origins_list.append(ray["origin"])
            directions_list.append(ray["direction"])
            weights_list.append(weight)
            ref_flags.append(False)

    return (
        np.array(origins_list,    dtype=np.float64),
        np.array(directions_list, dtype=np.float64),
        np.array(weights_list,    dtype=np.float64),
        np.array(ref_flags,       dtype=bool),
    )


def run_ransac_pipeline(label, ref_rays, match_rays_by_view, matched_views,
                        box_min, box_max):
    """Run RANSAC for one method and return results dict."""
    origins, dirs, weights, ref_mask = build_ransac_arrays(
        ref_rays, match_rays_by_view, matched_views,
    )

    n_ref   = int(ref_mask.sum())
    n_match = int((~ref_mask).sum())
    print(f"\n  [{label}]  Rays: {len(origins)} total  "
          f"({n_ref} ref + {n_match} match)")

    vol_diag = float(np.linalg.norm(box_max - box_min))
    inlier_threshold = max(10.0, RANSAC_INLIER_FRAC * vol_diag)

    best_pt, inlier_mask, residual = ransac_ray_intersection(
        origins, dirs, weights, ref_mask,
        n_iterations=RANSAC_N_ITER,
        inlier_threshold=inlier_threshold,
        min_inliers=RANSAC_MIN_INLIERS,
    )

    n_inliers     = int(inlier_mask.sum())
    n_ref_inliers = int((inlier_mask &  ref_mask).sum())
    n_mat_inliers = int((inlier_mask & ~ref_mask).sum())

    if best_pt is not None:
        print(f"  ✓ [{label}] Intersection: "
              f"({best_pt[0]:.1f}, {best_pt[1]:.1f}, {best_pt[2]:.1f})")
        print(f"    Inliers: {n_inliers}/{len(origins)}  "
              f"({n_ref_inliers} ref + {n_mat_inliers} match)  "
              f"|  residual: {residual:.2f} vox")
    else:
        print(f"  ✗ [{label}] RANSAC failed.")

    return {
        "label":      label,
        "best_pt":    best_pt,
        "inliers":    inlier_mask,
        "residual":   residual,
        "n_total":    len(origins),
        "n_inliers":  n_inliers,
        "n_ref_inl":  n_ref_inliers,
        "n_mat_inl":  n_mat_inliers,
        "n_matched":  len(matched_views),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compare SIFT vs DINOv3 matching and triangulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device", default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for DINOv3 forward passes.",
    )
    parser.add_argument(
        "--lowe-ratio", type=float, default=LOWE_RATIO,
        help="Lowe's ratio test threshold for SIFT matching.",
    )
    args = parser.parse_args()
    lowe_ratio = args.lowe_ratio

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
    #  STEP 1: Load images, cameras, SIFT features, DINOv3 embeddings
    # ══════════════════════════════════════════════════════════════════════════

    ref_path = PROC_ROOT / SPECIMEN / IMAGE_NAME / f"{REF_VIEW}.png"
    if not ref_path.exists():
        print(f"[ERROR] Reference view not found: {ref_path}")
        sys.exit(1)

    ref_img = Image.open(ref_path).convert("RGB")
    ref_img_np = np.array(ref_img)
    print(f"Reference image: {ref_path.relative_to(PROJECT_ROOT)}")

    # ── Camera data ───────────────────────────────────────────────────────────
    cam_pkl = PROC_ROOT / SPECIMEN / IMAGE_NAME / "cameras.pkl"
    with open(cam_pkl, "rb") as f:
        cameras = pickle.load(f)
    volume_shape = cameras[REF_VIEW]["volume_shape"]
    Vx, Vy, Vz = volume_shape
    print(f"Volume shape: {volume_shape}")

    # ── Discover other views ──────────────────────────────────────────────────
    view_dir = PROC_ROOT / SPECIMEN / IMAGE_NAME
    all_views = sorted(p.stem for p in view_dir.glob("view_*.png"))
    other_view_names = [vn for vn in all_views if vn != REF_VIEW]
    print(f"Target views: {other_view_names}")

    # ── Load other view images ────────────────────────────────────────────────
    other_view_images: dict[str, np.ndarray] = {}
    for vn in other_view_names:
        vp = PROC_ROOT / SPECIMEN / IMAGE_NAME / f"{vn}.png"
        other_view_images[vn] = np.array(Image.open(vp).convert("RGB"))

    # ── SIFT extraction ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SIFT FEATURE EXTRACTION")
    print("=" * 60)

    ref_kps, ref_descs = extract_sift_features(ref_img_np)
    print(f"  Reference: {len(ref_kps)} keypoints")

    other_sift: dict[str, tuple] = {}
    for vn in other_view_names:
        kps, descs = extract_sift_features(other_view_images[vn])
        other_sift[vn] = (kps, descs)
        print(f"    {vn}: {len(kps) if kps else 0} keypoints")

    # ── DINOv3 embeddings ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DINOv3 EMBEDDINGS")
    print("=" * 60)

    # Rotated reference embeddings
    emb_path = EXP_DIR / "rotated_embeddings.pkl"
    if emb_path.exists():
        print(f"  Rotated embeddings found: {emb_path.relative_to(PROJECT_ROOT)}")
        with open(emb_path, "rb") as f:
            rotated_embs = pickle.load(f)
    else:
        rotated_embs = generate_rotated_embeddings(device)

    # Regular view embeddings
    reg_emb_path = EMB_ROOT / SPECIMEN / f"{IMAGE_NAME}.pkl"
    if not reg_emb_path.exists():
        print(f"[ERROR] Regular embeddings not found: {reg_emb_path}")
        print("  Run  python src/generate_embeddings.py  first.")
        sys.exit(1)
    with open(reg_emb_path, "rb") as f:
        regular_embs = pickle.load(f)
    print(f"  Regular embeddings loaded for {len(regular_embs)} views.")

    # Rotated reference images for display
    rotated_images_np: dict[int, np.ndarray] = {}
    for angle in ROTATION_ANGLES:
        rotated_images_np[angle] = np.array(rotate_image(ref_img, angle))

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 2: Click on the reference view
    # ══════════════════════════════════════════════════════════════════════════

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(ref_img_np)

    # Show SIFT keypoints as faint dots
    kp_xs = [kp.pt[0] for kp in ref_kps]
    kp_ys = [kp.pt[1] for kp in ref_kps]
    ax.scatter(kp_xs, kp_ys, s=3, c="cyan", alpha=0.3, zorder=2)

    ax.set_title(
        f"Click on {REF_VIEW} to select a point\n"
        f"(cyan dots = SIFT keypoints · used by both pipelines)\n"
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
    print("\n>>> Click on the reference view. Close window when done.\n")
    plt.show()

    if not clicked_points:
        print("No points clicked. Exiting.")
        return

    u_click, v_click = clicked_points[-1]
    print(f"\nClicked pixel: ({u_click:.1f}, {v_click:.1f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3a: SIFT — find nearest keypoint & match
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  SIFT MATCHING")
    print("=" * 60)

    nearest_idx = find_nearest_keypoint(ref_kps, u_click, v_click)
    nearest_kp  = ref_kps[nearest_idx]
    sift_u, sift_v = nearest_kp.pt
    sift_dist = math.sqrt((sift_u - u_click) ** 2 + (sift_v - v_click) ** 2)
    print(f"  Nearest SIFT: ({sift_u:.1f}, {sift_v:.1f})  "
          f"[Δ={sift_dist:.1f} px, scale={nearest_kp.size:.1f}]")

    ref_descriptor = ref_descs[nearest_idx]

    # sift_matches[vn] = (kp_x, kp_y, distance, kp_idx)  or None
    sift_matches: dict[str, tuple | None] = {}
    for vn in other_view_names:
        t_kps, t_descs = other_sift[vn]
        result = match_descriptor_to_view(ref_descriptor, t_descs,
                                          ratio=lowe_ratio)
        if result is not None:
            kp_idx, dist = result
            mx, my = t_kps[kp_idx].pt
            sift_matches[vn] = (mx, my, dist, kp_idx)
            print(f"    {vn}: ({mx:.1f}, {my:.1f})  dist={dist:.2f}")
        else:
            sift_matches[vn] = None
            print(f"    {vn}: no match")

    sift_matched_views = [vn for vn in other_view_names
                          if sift_matches[vn] is not None]
    print(f"  SIFT matched {len(sift_matched_views)}/{len(other_view_names)}")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 3b: DINOv3 — map click to patch & match across rotations
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  DINOv3 MATCHING (rotation-augmented)")
    print("=" * 60)

    patch_col = min(int(u_click) // PATCH_SIZE, GRID_SIZE - 1)
    patch_row = min(int(v_click) // PATCH_SIZE, GRID_SIZE - 1)
    print(f"  Click → DINOv3 patch (row={patch_row}, col={patch_col})")

    # Collect query embeddings from all 4 rotations
    dino_ref_info: list[dict] = []
    dino_query_embs: list[np.ndarray] = []

    for i, (angle, key, label) in enumerate(
        zip(ROTATION_ANGLES, ROTATION_KEYS, ROTATION_LABELS)
    ):
        r_row, r_col = map_patch_to_rotation(patch_row, patch_col, angle)
        flat_idx = r_row * GRID_SIZE + r_col
        emb = rotated_embs[key][flat_idx]

        dino_ref_info.append({
            "angle": angle, "key": key, "label": label,
            "row": r_row, "col": r_col, "flat_idx": flat_idx,
        })
        dino_query_embs.append(emb)
        print(f"    {label:>4s}  →  patch ({r_row},{r_col})")

    # Match each other view
    # dino_matches[vn] = (patch_idx, sim, ref_idx)
    dino_matches: dict[str, tuple[int, float, int]] = {}
    for vn in other_view_names:
        target_matrix = regular_embs[vn]
        pidx, sim, ridx = find_best_match_across_references(
            dino_query_embs, target_matrix,
        )
        dino_matches[vn] = (pidx, sim, ridx)
        m_row = pidx // GRID_SIZE
        m_col = pidx % GRID_SIZE
        print(f"    {vn}: patch ({m_row},{m_col})  "
              f"sim={sim:.4f}  ← ref {ROTATION_LABELS[ridx]}")

    dino_matched_views = list(other_view_names)   # DINOv3 always produces a match

    # Winning reference
    ref_counts = [0] * len(ROTATION_ANGLES)
    for _, (_, _, ridx) in dino_matches.items():
        ref_counts[ridx] += 1
    winning_ref_idx = ref_counts.index(max(ref_counts))
    winning_label   = ROTATION_LABELS[winning_ref_idx]
    print(f"\n  ★ Winning reference: {winning_label} "
          f"({ref_counts[winning_ref_idx]} matches)")

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 4: RANSAC on both
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  RANSAC INTERSECTION (both methods)")
    print("=" * 60)

    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([Vx, Vy, Vz], dtype=np.float64)
    ref_cam_data = build_perspective_camera(cameras[REF_VIEW])

    # ── SIFT rays ─────────────────────────────────────────────────────────────
    sift_ref_rays = cast_keypoint_rays(ref_cam_data, sift_u, sift_v,
                                        box_min, box_max)
    sift_match_rays: dict[str, tuple[list, float]] = {}
    for vn in sift_matched_views:
        mx, my, dist, _ = sift_matches[vn]
        cam_other = build_perspective_camera(cameras[vn])
        mrays = cast_keypoint_rays(cam_other, mx, my, box_min, box_max)
        weight = 1.0 / max(dist, 1e-6)
        sift_match_rays[vn] = (mrays, weight)

    sift_result = run_ransac_pipeline(
        "SIFT", sift_ref_rays, sift_match_rays, sift_matched_views,
        box_min, box_max,
    )

    # ── DINOv3 rays ───────────────────────────────────────────────────────────
    dino_ref_rays = cast_patch_rays(ref_cam_data, patch_row, patch_col,
                                     box_min, box_max)
    dino_match_rays: dict[str, tuple[list, float]] = {}
    for vn in dino_matched_views:
        pidx, sim, _ = dino_matches[vn]
        m_row = pidx // GRID_SIZE
        m_col = pidx % GRID_SIZE
        cam_other = build_perspective_camera(cameras[vn])
        mrays = cast_patch_rays(cam_other, m_row, m_col, box_min, box_max)
        dino_match_rays[vn] = (mrays, float(sim))

    dino_result = run_ransac_pipeline(
        "DINOv3", dino_ref_rays, dino_match_rays, dino_matched_views,
        box_min, box_max,
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 5: Combined 2D Visualisation
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  PLOTTING (2D)")
    print("=" * 60)

    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
    from matplotlib.colors import to_rgba

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

    # Layout:
    #   Row 0: Reference image (col 0) + DINOv3 4 rotated refs (cols 1-4)
    #   Row 1: SIFT matched views  (label on left)
    #   Row 2: DINOv3 matched views (label on left)
    n_views   = len(other_view_names)
    view_cols = min(n_views, 5)
    sift_rows = math.ceil(len(sift_matched_views) / view_cols) if sift_matched_views else 1
    dino_rows = math.ceil(n_views / view_cols)

    total_rows = 1 + sift_rows + dino_rows
    total_cols = max(5, view_cols)

    fig, axes = plt.subplots(
        total_rows, total_cols,
        figsize=(total_cols * 2.8, total_rows * 3.0),
        gridspec_kw={"hspace": 0.45, "wspace": 0.15},
    )
    if total_rows == 1:
        axes = axes[np.newaxis, :]

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        "SIFT vs DINOv3 Comparison",
        fontsize=20, fontweight="bold", color="white", y=0.98,
    )

    sift_pt_str = (f"({sift_result['best_pt'][0]:.0f}, "
                   f"{sift_result['best_pt'][1]:.0f}, "
                   f"{sift_result['best_pt'][2]:.0f})"
                   if sift_result["best_pt"] is not None else "failed")
    dino_pt_str = (f"({dino_result['best_pt'][0]:.0f}, "
                   f"{dino_result['best_pt'][1]:.0f}, "
                   f"{dino_result['best_pt'][2]:.0f})"
                   if dino_result["best_pt"] is not None else "failed")

    fig.text(
        0.5, 0.955,
        f"Click: ({u_click:.0f}, {v_click:.0f}) · {SPECIMEN}/{IMAGE_NAME}  |  "
        f"SIFT → {sift_pt_str}  |  DINOv3 → {dino_pt_str}",
        ha="center", fontsize=11, color="#aaaaaa",
    )

    # ── Row 0, col 0: Reference image with click + SIFT keypoint ──────────────
    ax_ref = axes[0, 0]
    ax_ref.imshow(ref_img_np)

    # Click marker (red cross)
    ax_ref.plot(u_click, v_click, "r+", markersize=18, markeredgewidth=2.5,
                zorder=5)
    # SIFT keypoint (green circle)
    kp_circle = Circle(
        (sift_u, sift_v), radius=nearest_kp.size * 2,
        linewidth=2.5, edgecolor="#2ed573",
        facecolor=to_rgba("#2ed573", 0.2), zorder=5,
    )
    ax_ref.add_patch(kp_circle)
    ax_ref.plot(sift_u, sift_v, "o", color="#2ed573", markersize=4, zorder=6)

    # DINOv3 patch (yellow rectangle)
    px_dino = patch_col * PATCH_SIZE
    py_dino = patch_row * PATCH_SIZE
    ax_ref.add_patch(Rectangle(
        (px_dino, py_dino), PATCH_SIZE, PATCH_SIZE,
        linewidth=2.5, edgecolor="#ffa502",
        facecolor=to_rgba("#ffa502", 0.2), zorder=4,
    ))

    # Arrow from click to SIFT keypoint
    if sift_dist > 3:
        arrow = FancyArrowPatch(
            (u_click, v_click), (sift_u, sift_v),
            arrowstyle="->", color="yellow", linewidth=1.5,
            mutation_scale=12, zorder=5,
        )
        ax_ref.add_patch(arrow)

    ax_ref.set_title("REFERENCE", fontsize=11, fontweight="bold",
                     color="white", pad=6)
    ax_ref.text(
        0.5, -0.04,
        f"click ({u_click:.0f},{v_click:.0f})\n"
        f"SIFT ({sift_u:.0f},{sift_v:.0f}) · "
        f"patch ({patch_row},{patch_col})",
        transform=ax_ref.transAxes, ha="center", va="top",
        fontsize=7, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="#555577", edgecolor="none", alpha=0.85),
    )
    ax_ref.set_xticks([]); ax_ref.set_yticks([])
    for spine in ax_ref.spines.values():
        spine.set_edgecolor("white"); spine.set_linewidth(2.5)

    # ── Row 0, cols 1-4: DINOv3 rotated references ───────────────────────────
    for i, info in enumerate(dino_ref_info):
        ax = axes[0, 1 + i]
        angle = info["angle"]
        color = ROTATION_COLORS[i]

        ax.imshow(rotated_images_np[angle])

        pr, pc = info["row"], info["col"]
        px = pc * PATCH_SIZE
        py = pr * PATCH_SIZE

        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=0, facecolor=to_rgba(color, alpha=0.35),
        ))
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=3, edgecolor=color, facecolor="none",
        ))
        cx = px + PATCH_SIZE / 2
        cy = py + PATCH_SIZE / 2
        ax.axhline(cy, color=color, linewidth=0.5, alpha=0.45, linestyle="--")
        ax.axvline(cx, color=color, linewidth=0.5, alpha=0.45, linestyle="--")

        title_str = f"REF {info['label']}"
        if i == winning_ref_idx:
            title_str += " ★"
        ax.set_title(title_str, fontsize=10, fontweight="bold",
                     color=color, pad=6)
        ax.text(
            0.5, -0.04, f"patch ({pr},{pc})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=color, edgecolor="none", alpha=0.85),
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0 if i != winning_ref_idx else 3.5)

    # Hide unused cols in row 0
    for j in range(5, total_cols):
        axes[0, j].axis("off")

    # ── Row(s) 1+: SIFT matched views ────────────────────────────────────────
    sift_color = "#2ed573"
    sift_row_start = 1

    # Row label
    fig.text(0.01, 1.0 - (sift_row_start + 0.5 * sift_rows) / total_rows,
             "SIFT", fontsize=14, fontweight="bold", color=sift_color,
             rotation=90, va="center", ha="left")

    for idx, vn in enumerate(sift_matched_views):
        r = sift_row_start + idx // view_cols
        c = idx % view_cols
        ax = axes[r, c]
        ax.imshow(other_view_images[vn])

        mx, my, dist, kp_idx = sift_matches[vn]
        kp_size = other_sift[vn][0][kp_idx].size

        kp_c = Circle(
            (mx, my), radius=kp_size * 2,
            linewidth=2.5, edgecolor=sift_color,
            facecolor=to_rgba(sift_color, 0.25), zorder=5,
        )
        ax.add_patch(kp_c)
        ax.plot(mx, my, "o", color=sift_color, markersize=4, zorder=6)
        ax.axhline(my, color=sift_color, linewidth=0.5, alpha=0.45,
                    linestyle="--")
        ax.axvline(mx, color=sift_color, linewidth=0.5, alpha=0.45,
                    linestyle="--")

        ax.set_title(vn.replace("_", " ").upper(), fontsize=9,
                     fontweight="bold", color=sift_color, pad=4)
        ax.text(
            0.5, -0.04, f"dist={dist:.1f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=sift_color, edgecolor="none", alpha=0.85),
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355"); spine.set_linewidth(1.0)

    # Hide unused SIFT cells
    for idx in range(len(sift_matched_views), sift_rows * view_cols):
        r = sift_row_start + idx // view_cols
        c = idx % view_cols
        if r < total_rows and c < total_cols:
            axes[r, c].axis("off")

    # Hide extra columns in SIFT rows
    for r in range(sift_row_start, sift_row_start + sift_rows):
        for c in range(view_cols, total_cols):
            if r < total_rows and c < total_cols:
                axes[r, c].axis("off")

    # ── Row(s) 2+: DINOv3 matched views ──────────────────────────────────────
    dino_row_start = sift_row_start + sift_rows

    # Row label
    fig.text(0.01, 1.0 - (dino_row_start + 0.5 * dino_rows) / total_rows,
             "DINOv3", fontsize=14, fontweight="bold", color="#ffa502",
             rotation=90, va="center", ha="left")

    for idx, vn in enumerate(other_view_names):
        r = dino_row_start + idx // view_cols
        c = idx % view_cols
        ax = axes[r, c]
        ax.imshow(other_view_images[vn])

        pidx, sim, ridx = dino_matches[vn]
        m_row = pidx // GRID_SIZE
        m_col = pidx % GRID_SIZE
        color = ROTATION_COLORS[ridx]

        px = m_col * PATCH_SIZE
        py = m_row * PATCH_SIZE

        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=0, facecolor=to_rgba(color, alpha=0.30),
        ))
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=2.5, edgecolor=color, facecolor="none",
        ))
        cx = px + PATCH_SIZE / 2
        cy = py + PATCH_SIZE / 2
        ax.axhline(cy, color=color, linewidth=0.5, alpha=0.45, linestyle="--")
        ax.axvline(cx, color=color, linewidth=0.5, alpha=0.45, linestyle="--")

        ax.set_title(vn.replace("_", " ").upper(), fontsize=9,
                     fontweight="bold", color=color, pad=4)
        ax.text(
            0.5, -0.04,
            f"sim={sim:.4f} ← {ROTATION_LABELS[ridx]}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor=color, edgecolor="none", alpha=0.85),
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355"); spine.set_linewidth(1.0)

    # Hide unused DINOv3 cells
    for idx in range(n_views, dino_rows * view_cols):
        r = dino_row_start + idx // view_cols
        c = idx % view_cols
        if r < total_rows and c < total_cols:
            axes[r, c].axis("off")

    # Hide extra columns in DINOv3 rows
    for r in range(dino_row_start, dino_row_start + dino_rows):
        for c in range(view_cols, total_cols):
            if r < total_rows and c < total_cols:
                axes[r, c].axis("off")

    # ── Legend ────────────────────────────────────────────────────────────────
    fig.text(0.15, 0.015, "✚ Click", fontsize=10, fontweight="bold",
             color="#ff4757", ha="center")
    fig.text(0.30, 0.015, "● SIFT keypoint", fontsize=10,
             fontweight="bold", color="#2ed573", ha="center")
    fig.text(0.47, 0.015, "■ DINOv3 patch", fontsize=10,
             fontweight="bold", color="#ffa502", ha="center")
    for i, (label, color) in enumerate(zip(ROTATION_LABELS, ROTATION_COLORS)):
        fig.text(0.62 + i * 0.10, 0.015, f"■ {label}",
                 fontsize=10, fontweight="bold", color=color, ha="center")

    plt.subplots_adjust(top=0.92, bottom=0.04, left=0.04, right=0.98)
    print("\n>>> Showing 2D comparison. "
          "Close the window to proceed to 3D viewers.")
    plt.show()
    plt.rcdefaults()

    # ══════════════════════════════════════════════════════════════════════════
    #  STEP 6: Side-by-side 3D intersection viewers
    # ══════════════════════════════════════════════════════════════════════════

    sift_pt = sift_result["best_pt"]
    dino_pt = dino_result["best_pt"]

    if sift_pt is None and dino_pt is None:
        print("\n  Both RANSAC runs failed. Skipping 3D viewer.")
        print("Done.")
        return

    from vedo import (Volume as VedoVolume, Sphere as VedoSphere,
                      Plotter, Text2D)

    print("\n>>> Loading 3D volume for intersection viewers …")
    tiff_path = RAW_ROOT / SPECIMEN / f"{IMAGE_NAME}.tif"
    original_vol = VedoVolume(str(tiff_path))
    vol_data = original_vol.tonumpy().astype(np.float32)
    binary_mask = otsu_threshold(vol_data, log_scale=True)
    vol_data[~binary_mask] = 0
    print(f"  Loaded. Foreground voxels: {int((vol_data > 0).sum()):,}")

    # ── Build actors helper ────────────────────────────────────────────────────
    def make_vol_actor():
        v = VedoVolume(vol_data, spacing=original_vol.spacing(),
                       origin=original_vol.origin())
        v.mode(0).cmap("bone").alpha([0, 0, 0.3, 0.6])
        return v

    def make_sphere_actors(pt, color, label, res):
        actors = []
        halo = VedoSphere(pt, r=ESTIMATED_POINT_RADIUS * 1.4)
        halo.color("white").alpha(0.20)
        halo.name = f"{label} halo"
        actors.append(halo)

        sphere = VedoSphere(pt, r=ESTIMATED_POINT_RADIUS)
        sphere.color(color).alpha(0.92)
        sphere.name = f"{label} intersection"
        actors.append(sphere)

        info = Text2D(
            f"{label}\n"
            f"  ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})\n"
            f"  residual={res['residual']:.2f} vox\n"
            f"  inliers={res['n_inliers']}/{res['n_total']}\n"
            f"  matched={res['n_matched']} views",
            pos="top-left", font="Mono", s=0.7, bg="black", alpha=0.75,
        )
        actors.append(info)
        return actors

    # ── Viewer 1: SIFT ────────────────────────────────────────────────────────
    if sift_pt is not None:
        print(">>> Opening SIFT intersection viewer. Close to proceed to DINOv3.")
        sift_actors = [make_vol_actor()]
        sift_actors += make_sphere_actors(sift_pt, "limegreen", "SIFT",
                                          sift_result)
        plt_sift = Plotter(
            axes=1,
            title=f"SIFT Intersection — {SPECIMEN}/{IMAGE_NAME}",
        )
        plt_sift.show(*sift_actors, interactive=True)
    else:
        print("  Skipping SIFT viewer — RANSAC failed.")

    # ── Viewer 2: DINOv3 ─────────────────────────────────────────────────────
    if dino_pt is not None:
        print(">>> Opening DINOv3 intersection viewer. Close when done.")
        dino_actors = [make_vol_actor()]
        dino_actors += make_sphere_actors(dino_pt, "deeppink", "DINOv3",
                                          dino_result)
        # Also show SIFT point as faint reference if available
        if sift_pt is not None:
            ghost = VedoSphere(sift_pt, r=ESTIMATED_POINT_RADIUS * 0.6)
            ghost.color("limegreen").alpha(0.25)
            ghost.name = "SIFT (ghost)"
            dino_actors.append(ghost)
        plt_dino = Plotter(
            axes=1,
            title=f"DINOv3 Intersection — {SPECIMEN}/{IMAGE_NAME}",
        )
        plt_dino.show(*dino_actors, interactive=True)
    else:
        print("  Skipping DINOv3 viewer — RANSAC failed.")

    # ── Print comparison summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    if sift_pt is not None and dino_pt is not None:
        delta = np.linalg.norm(sift_pt - dino_pt)
        print(f"  SIFT  intersection: ({sift_pt[0]:.1f}, {sift_pt[1]:.1f}, "
              f"{sift_pt[2]:.1f})  residual={sift_result['residual']:.2f}")
        print(f"  DINOv3 intersection: ({dino_pt[0]:.1f}, {dino_pt[1]:.1f}, "
              f"{dino_pt[2]:.1f})  residual={dino_result['residual']:.2f}")
        print(f"  Distance between estimates: {delta:.1f} voxels")
    elif sift_pt is not None:
        print(f"  SIFT succeeded, DINOv3 failed.")
    elif dino_pt is not None:
        print(f"  DINOv3 succeeded, SIFT failed.")

    print("Done.")


if __name__ == "__main__":
    main()
