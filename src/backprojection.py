"""
backprojection.py
-----------------
Pipeline (per processed image folder):
  1. Load cameras.pkl  (camera geometry for every view).
  2. Load the corresponding embeddings .pkl
     { "view_000": np.ndarray(n_patches, hidden), ... }.
  3. For every view, cast one **perspective** ray per pixel in the image:
       • Each pixel (u, v) gets a unique ray direction radiating from the
         camera's optical centre through that pixel's position on the image
         plane.
       • Use an analytic slab test (ray-AABB intersection) to find
         where each ray enters the volume bounding box.
       • Step one voxel-width past the entry point → first-hit voxel.
  4. For each 16×16 patch, collect **every unique voxel** hit by any of
     its 256 pixel rays and append the patch embedding to all of them
     (one copy per unique voxel, avoiding duplicates within the same
     patch+view).
  5. Save the result as a pickle at
       data/infered/bugNIST_900/{specimen}/{image_name}.pkl

Output format
-------------
The saved pickle is a dict:
  {
    (ix, iy, iz): [emb_a, emb_b, …],   # list of float32 (hidden_dim,) arrays
    …
  }
Each list collects every embedding (across all views) whose ray first hit
that voxel.  Multiple views hitting the same voxel accumulate independently,
so you can later average, PCA-visualise, or cluster the per-voxel lists.

Perspective camera model
--------------------------
vedo/VTK places the camera at position *t* (world coords) looking at
*vol_center*.  The default vertical field-of-view is 30°.

For each pixel (u, v), the ray direction in camera space is:
    d_cam = K⁻¹ @ [u, v, 1]ᵀ
where K is the intrinsic matrix with focal length derived from the FOV:
    f = (img_h / 2) / tan(FOV / 2)

The camera-to-world rotation R_c2w is constructed from the actual camera
axes (right, up, -look) derived from t, vol_center, and the view-up hint.
The world-space ray direction is then:
    d_world = R_c2w @ d_cam   (normalised)

Usage examples:
  # Backproject all specimens, all images
  python src/backprojection.py

  # Backproject only GH, first 2 images
  python src/backprojection.py --specimens GH --n-images 2

  # Backproject GH and SL, all images
  python src/backprojection.py --specimens GH SL --n-images all
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Make the src package importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helper_functions import otsu_threshold

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_ROOT     = PROJECT_ROOT / "data" / "raw"       / "bugNIST_900"
PROC_ROOT    = PROJECT_ROOT / "data" / "processed" / "bugNIST_900"
EMB_ROOT     = PROJECT_ROOT / "data" / "embeddings" / "bugNIST_900"
INF_ROOT     = PROJECT_ROOT / "data" / "infered"  / "bugNIST_900"

# ── ViT / image constants ─────────────────────────────────────────────────────
PATCH_SIZE = 16      # ViT patch size in pixels (DINOv3: 16×16)
IMG_W      = 1440
IMG_H      = 1440

# ── VTK camera default ────────────────────────────────────────────────────────
VTK_DEFAULT_FOV_DEG = 30.0   # vedo/VTK default vertical field of view


# ── 3-D volume mask ───────────────────────────────────────────────────────────
def load_volume_mask(specimen: str, image_name: str) -> np.ndarray:
    """
    Load the raw TIFF for *image_name* and return a boolean foreground mask
    produced by the same Otsu threshold used in generate_views.py.

    Parameters
    ----------
    specimen   : e.g. "GH"
    image_name : e.g. "gras_9_042"  (no extension)

    Returns
    -------
    np.ndarray, shape (Vx, Vy, Vz), dtype bool
        True where the voxel is foreground (above Otsu threshold).
    """
    tiff_path = RAW_ROOT / specimen / f"{image_name}.tif"
    if not tiff_path.exists():
        raise FileNotFoundError(
            f"Raw TIFF not found: {tiff_path}\n"
            f"Cannot build 3-D Otsu mask without the source volume."
        )

    # Import vedo here so the rest of the module stays lightweight
    from vedo import Volume
    print(f"  Loading raw TIFF for Otsu mask: {tiff_path.relative_to(PROJECT_ROOT)}")
    vol_data = Volume(str(tiff_path)).tonumpy().astype(np.float32)
    mask = otsu_threshold(vol_data, log_scale=True)   # same call as generate_views.py
    n_fg = int(mask.sum())
    print(f"  Foreground voxels (3-D Otsu): {n_fg:,} / {mask.size:,}")
    return mask


# ── Perspective camera helpers ────────────────────────────────────────────────
def build_perspective_camera(cam: dict) -> dict:
    """
    Reconstruct the VTK perspective camera for one view.

    VTK camera convention (verified numerically against stored camera data):
      look  = normalise(vol_center - t)        camera points toward the volume
      right = normalise(look x up_hint)        screen-right direction
      up    = normalise(right x look)          screen-up direction (world)

    Ray for pixel (u, v):
      d_cam   = K_inv @ [u+0.5, v+0.5, 1]     camera-space direction
      d_world = right*d_cam[0] + (-up)*d_cam[1] + look*d_cam[2]
      (-up because image rows increase downward, world-up points upward)

    The stored K has f=1 (placeholder).  We recompute f from the known
    VTK default vertical FOV of 30 degrees.

    t and vol_center are both in voxel-index space (spacing=1, origin=0).

    Returns
    -------
    dict with keys:
        cam_origin : (3,) camera position in voxel-index space
        right      : (3,) world-space rightward unit vector
        up         : (3,) world-space upward unit vector  
        look       : (3,) world-space forward (into scene) unit vector
        K_inv      : (3, 3) inverse intrinsic matrix (with correct focal length)
    """
    t          = np.asarray(cam["t"],          dtype=np.float64)
    vol_center = np.asarray(cam["vol_center"], dtype=np.float64)

    # ── Camera axes (VTK convention) ─────────────────────────────────────────
    look = vol_center - t
    look = look / np.linalg.norm(look)

    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(look, up_hint)) > 0.99:
        up_hint = np.array([0.0, 1.0, 0.0])

    right = np.cross(look, up_hint)
    right = right / np.linalg.norm(right)

    up = np.cross(right, look)           # reorthogonalised up (world-upward)
    up = up / np.linalg.norm(up)

    # ── Intrinsic matrix (f from VTK 30° vertical FOV) ───────────────────────
    fov_rad = np.radians(VTK_DEFAULT_FOV_DEG)
    f       = (IMG_H / 2.0) / np.tan(fov_rad / 2.0)
    cx, cy  = IMG_W / 2.0, IMG_H / 2.0

    K_inv = np.array([
        [1.0/f,   0.0,  -cx/f],
        [0.0,   1.0/f,  -cy/f],
        [0.0,     0.0,    1.0],
    ], dtype=np.float64)

    return {
        "cam_origin": t,
        "right"     : right,
        "up"        : up,
        "look"      : look,
        "K_inv"     : K_inv,
    }


# ── Vectorised first-hit ray casting (perspective) ────────────────────────────

# Numba JIT-compiled inner loop for performance (~100× faster than Python).
# Falls back to pure numpy/Python if numba is not installed.
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _march_rays(origins, d_unit, t_start, t_exit, vol_mask, Vx, Vy, Vz,
                    active, step_size, max_steps, result):
        for ii in prange(len(active)):
            i = active[ii]
            ox = origins[i, 0]; oy = origins[i, 1]; oz = origins[i, 2]
            dx = d_unit[i, 0];  dy = d_unit[i, 1];  dz = d_unit[i, 2]
            t     = t_start[i] + step_size
            t_max = t_exit[i]
            for _ in range(max_steps):
                if t > t_max:
                    break
                px = ox + t * dx
                py = oy + t * dy
                pz = oz + t * dz
                ix = int(px); iy = int(py); iz = int(pz)
                if ix < 0 or ix >= Vx or iy < 0 or iy >= Vy or iz < 0 or iz >= Vz:
                    break
                if vol_mask[ix, iy, iz]:
                    result[i, 0] = ix
                    result[i, 1] = iy
                    result[i, 2] = iz
                    break
                t += step_size
        return result

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


def first_hit_batch(
    origins:      np.ndarray,   # (N, 3) ray origins
    directions:   np.ndarray,   # (N, 3) per-ray directions (unnormalised ok)
    volume_shape: tuple,        # (Vx, Vy, Vz)
    vol_mask:     np.ndarray = None,  # (Vx, Vy, Vz) bool — foreground mask
) -> np.ndarray:
    """
    For each ray find the first *foreground* voxel it hits inside the volume.

    Strategy
    --------
    1. Normalise each direction to unit length so that every march step
       advances exactly `step_size` voxels regardless of input magnitude.
    2. Use the Williams slab test to find t_enter (parametric entry into
       the AABB).  Rays that miss the box entirely are marked -1.
    3. Starting at t_enter, march along the ray in steps of `step_size`
       voxels.  At each step, check if the current voxel is inside the
       foreground mask.  Stop at the first foreground voxel.
       If vol_mask is None, stop at the first in-bounds voxel (entry face).

    Uses numba (parallel JIT) when available for ~100× speedup; falls back
    to a pure-Python loop otherwise (correct but slow for large images).

    Returns
    -------
    hits : (N, 3) int32.
        hits[i] = (ix, iy, iz)  if ray i hits a foreground voxel, else (-1,-1,-1).
    """
    STEP_SIZE  = 0.5    # march step in voxels (half-voxel for accuracy)
    MAX_STEPS  = 4000   # safety cap (> diagonal of 450×450×900 ≈ 1080 voxels)

    Vx, Vy, Vz = volume_shape
    N = origins.shape[0]

    # ── 1. Normalise directions to unit length ────────────────────────────────
    norms = np.linalg.norm(directions, axis=1, keepdims=True)   # (N, 1)
    norms = np.where(norms < 1e-12, 1.0, norms)
    d_unit = directions / norms                                  # (N, 3) unit

    # ── 2. Slab test to find AABB entry ──────────────────────────────────────
    box_min = np.array([0.0, 0.0, 0.0])
    box_max = np.array([Vx,  Vy,  Vz], dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_d = np.where(np.abs(d_unit) > 1e-12, 1.0 / d_unit, np.inf)

    t1 = (box_min - origins) * inv_d
    t2 = (box_max - origins) * inv_d

    t_enter = np.max(np.minimum(t1, t2), axis=1)   # (N,)
    t_exit  = np.min(np.maximum(t1, t2), axis=1)   # (N,)

    box_hit = (t_enter <= t_exit) & (t_exit > 0.0)
    t_start = np.where(box_hit, np.maximum(t_enter, 0.0), 0.0)

    # ── 3. March to first foreground voxel ───────────────────────────────────
    result = np.full((N, 3), -1, dtype=np.int32)
    active = np.where(box_hit)[0].astype(np.int64)

    if len(active) == 0:
        return result

    # If no mask is provided, just take the first in-bounds voxel at entry
    if vol_mask is None:
        vol_mask_eff = np.ones((Vx, Vy, Vz), dtype=np.bool_)
    else:
        vol_mask_eff = vol_mask.astype(np.bool_)

    if _NUMBA_AVAILABLE:
        result = _march_rays(
            origins, d_unit, t_start, t_exit,
            vol_mask_eff, Vx, Vy, Vz,
            active, STEP_SIZE, MAX_STEPS, result,
        )
    else:
        # Pure-Python fallback — correct but slow for large images
        print("  [WARN] numba not found; ray marching in pure Python (slow). "
              "Install numba for ~100× speedup: pip install numba")
        for i in active:
            ox, oy, oz = origins[i]
            dx, dy, dz = d_unit[i]
            t     = t_start[i] + STEP_SIZE
            t_max = t_exit[i]
            for _ in range(MAX_STEPS):
                if t > t_max:
                    break
                ix = int(ox + t * dx)
                iy = int(oy + t * dy)
                iz = int(oz + t * dz)
                if ix < 0 or ix >= Vx or iy < 0 or iy >= Vy or iz < 0 or iz >= Vz:
                    break
                if vol_mask_eff[ix, iy, iz]:
                    result[i, 0] = ix
                    result[i, 1] = iy
                    result[i, 2] = iz
                    break
                t += STEP_SIZE

    return result


# ── Per-view backprojection (perspective + multi-voxel accumulation) ──────────
def backproject_view(
    embedding:    np.ndarray,   # (n_patches, hidden_dim)
    cam_data:     dict,
    volume_shape: tuple,
    accumulator:  dict,
    vol_mask:     np.ndarray = None,   # (Vx, Vy, Vz) bool — 3-D Otsu foreground mask
    img:          np.ndarray = None,   # rendered view image (H, W, C) or (H, W)
) -> int:
    """
    Backproject all patches from one view into the accumulator dict.

    Strategy
    --------
    1. For every pixel (u, v) in the image, compute a perspective ray:
           d_cam   = K⁻¹ @ [u+0.5, v+0.5, 1]ᵀ        (camera space)
           d_world = R_c2w @ d_cam                     (world/mm space)
    2. Transform the ray origin and directions from world/mm space into
       voxel-index space.
    3. March each ray through the volume (via first_hit_batch) to find
       the first *foreground* voxel it hits — background voxels (Otsu
       mask == False) are skipped, so the march continues past them.
    4. For each 16×16 patch:
         - Skip entirely if every pixel in the patch is 0 (pure black).
         - Collect every *unique* foreground voxel hit by any of its 256
           pixel rays.
         - Append the patch embedding once to every such voxel.

    Parameters
    ----------
    embedding    : (n_patches, hidden_dim) float32
    cam_data     : dict from build_perspective_camera()
    volume_shape : (Vx, Vy, Vz)
    accumulator  : {(ix,iy,iz): [emb, …]} modified in-place
    vol_mask     : (Vx, Vy, Vz) bool — foreground mask from 3-D Otsu
    img          : rendered view image (H, W, C) used for blackness check

    Returns
    -------
    Total number of (patch, voxel) assignments made.
    """
    cam_origin = cam_data["cam_origin"]   # (3,) voxel-index coords
    right      = cam_data["right"]        # (3,) world right unit vector
    up         = cam_data["up"]           # (3,) world up unit vector
    look       = cam_data["look"]         # (3,) world forward unit vector
    K_inv      = cam_data["K_inv"]        # (3, 3)

    n_patches = embedding.shape[0]
    grid_h    = int(np.sqrt(n_patches))
    grid_w    = grid_h

    patch_img_h = grid_h * PATCH_SIZE
    patch_img_w = grid_w * PATCH_SIZE

    # ── 1. Build per-pixel ray directions ────────────────────────────────────
    # For pixel centre (u+0.5, v+0.5):
    #   d_cam   = K_inv @ [u+0.5, v+0.5, 1]
    #   d_world = right * d_cam[0]  +  (-up) * d_cam[1]  +  look * d_cam[2]
    #
    # The -up term flips the Y axis: image rows go downward (+v),
    # but the world up vector points upward, so they are opposite.
    # Verified: centre pixel ray == look direction exactly.

    cols = np.arange(patch_img_w, dtype=np.float64) + 0.5
    rows = np.arange(patch_img_h, dtype=np.float64) + 0.5
    C, R_grid = np.meshgrid(cols, rows)   # (H, W) each

    N = patch_img_h * patch_img_w
    pix_hom = np.stack([C.ravel(), R_grid.ravel(), np.ones(N)], axis=1)  # (N, 3)

    d_cam = (K_inv @ pix_hom.T).T   # (N, 3)  — camera-space directions

    # Build world-space directions explicitly using verified VTK convention
    dx = d_cam[:, 0:1]   # (N, 1)
    dy = d_cam[:, 1:2]
    dz = d_cam[:, 2:3]
    d_world = dx * right[np.newaxis, :] + (-dy) * up[np.newaxis, :] + dz * look[np.newaxis, :]
    # d_world: (N, 3) — unnormalised; first_hit_batch normalises internally

    origins = np.broadcast_to(cam_origin[np.newaxis, :], (N, 3)).copy()

    # ── 3. March each ray to first foreground voxel ──────────────────────────
    # first_hit_batch normalises directions internally, marches in 0.5-voxel
    # steps, and skips background voxels (vol_mask==False) until it finds
    # the first foreground hit. Returns (-1,-1,-1) for rays that miss.
    hit_voxels = first_hit_batch(origins, d_world, volume_shape, vol_mask=vol_mask)
    # hit_voxels: (N, 3), each row is (ix, iy, iz) or (-1, -1, -1)

    # Reshape to (H, W, 3) so we can index by patch region
    hit_grid = hit_voxels.reshape(patch_img_h, patch_img_w, 3)

    # ── 5. Per-patch: append embedding to ALL unique foreground hit voxels ─────
    #
    #  A patch is skipped if ALL its 16×16 pixels in the rendered view image
    #  are zero (pure black = no foreground visible in this patch).
    #  The embedding is appended once to every unique foreground voxel hit by
    #  any of the patch's pixel rays.

    Vx, Vy, Vz = volume_shape
    hits = 0

    for pr in range(grid_h):
        for pc in range(grid_w):
            patch_idx = pr * grid_w + pc

            r0 = pr * PATCH_SIZE
            r1 = r0 + PATCH_SIZE
            c0 = pc * PATCH_SIZE
            c1 = c0 + PATCH_SIZE

            # Skip patch if every pixel in the 16×16 region is black (sum == 0)
            if img is not None:
                patch_pixels = img[r0:r1, c0:c1]
                if patch_pixels.sum() == 0:
                    continue

            emb_vec = embedding[patch_idx]   # (hidden_dim,)

            patch_hits = hit_grid[r0:r1, c0:c1, :]   # (16, 16, 3)
            flat = patch_hits.reshape(-1, 3)          # (256, 3)

            # Keep only valid hits (voxel index ≥ 0)
            valid_mask = flat[:, 0] >= 0
            if not valid_mask.any():
                continue

            valid = flat[valid_mask]   # (k, 3)

            # Encode each voxel as a single int64 for fast de-duplication
            codes = (valid[:, 0].astype(np.int64) * Vy * Vz
                     + valid[:, 1].astype(np.int64) * Vz
                     + valid[:, 2].astype(np.int64))

            # Find every unique voxel hit by this patch (no majority vote)
            unique_codes = np.unique(codes)

            # Decode and append the patch embedding to EACH unique voxel
            for code in unique_codes:
                bx = int(code // (Vy * Vz))
                by = int((code % (Vy * Vz)) // Vz)
                bz = int(code % Vz)
                accumulator.setdefault((bx, by, bz), []).append(emb_vec)
                hits += 1

    return hits


# ── Per-image processing ──────────────────────────────────────────────────────
def process_image(image_dir: Path, emb_path: Path) -> None:
    """
    Backproject embeddings for one processed image folder and save results.

    image_dir  : data/processed/bugNIST_900/{specimen}/{image_name}/
    emb_path   : data/embeddings/bugNIST_900/{specimen}/{image_name}.pkl
    Output     : data/infered/bugNIST_900/{specimen}/{image_name}.pkl
    """
    specimen   = image_dir.parent.name    # e.g. "GH"
    image_name = image_dir.name           # e.g. "gras_9_043"

    out_dir  = INF_ROOT / specimen
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_name}.pkl"

    print(f"\n{'='*60}")
    print(f"  Image : {image_dir.relative_to(PROJECT_ROOT)}")
    print(f"  Embeds: {emb_path.relative_to(PROJECT_ROOT)}")
    print(f"  Output: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"{'='*60}")

    # ── Load cameras ──────────────────────────────────────────────────────────
    cam_pkl = image_dir / "cameras.pkl"
    if not cam_pkl.exists():
        print(f"  [SKIP] cameras.pkl not found in {image_dir}")
        return

    with open(cam_pkl, "rb") as f:
        cameras: dict = pickle.load(f)

    # ── Load embeddings ───────────────────────────────────────────────────────
    if not emb_path.exists():
        print(f"  [SKIP] Embeddings file not found: {emb_path}")
        return

    with open(emb_path, "rb") as f:
        embeddings: dict = pickle.load(f)

    # Reconcile view keys
    cam_views = set(cameras.keys())
    emb_views = set(embeddings.keys())
    shared     = sorted(cam_views & emb_views)
    if not shared:
        print("  [SKIP] No matching view keys between cameras and embeddings.")
        print(f"         cameras: {sorted(cam_views)}")
        print(f"         embeds : {sorted(emb_views)}")
        return
    if cam_views != emb_views:
        print(f"  [WARN] View key mismatch. Proceeding with: {shared}")

    volume_shape = cameras[shared[0]]["volume_shape"]
    print(f"  Volume shape    : {volume_shape}")
    print(f"  Views to process: {len(shared)}")

    # ── Load 3-D Otsu foreground mask ─────────────────────────────────────────
    # Applied identically to generate_views.py so that only voxels that were
    # visible foreground in the rendered views can receive embeddings.
    vol_mask = load_volume_mask(specimen, image_name)

    # ── Backproject all views ─────────────────────────────────────────────────
    accumulator: dict[tuple[int, int, int], list[np.ndarray]] = {}

    for view_name in shared:
        cam = cameras[view_name]
        emb = embeddings[view_name]   # (n_patches, hidden_dim)

        cam_data = build_perspective_camera(cam)

        # cam["mip"] is the raw rendered RGB array — used for the patch-blackness
        # check (sum == 0 over 16×16 pixels → skip patch).
        img_array = cam["mip"]

        print(f"  {view_name}  [{emb.shape[0]} patches, hidden={emb.shape[1]}] … ",
              end="", flush=True)

        hits = backproject_view(
            emb, cam_data, volume_shape, accumulator,
            vol_mask=vol_mask, img=img_array,
        )
        print(f"{hits} voxel assignments")

    n_voxels = len(accumulator)
    n_total  = sum(len(v) for v in accumulator.values())
    print(f"\n  Voxels with ≥1 embedding : {n_voxels:,}")
    print(f"  Total embedding entries  : {n_total:,}")
    print(f"  Avg embeddings per voxel : {n_total / max(n_voxels, 1):.2f}")

    with open(out_path, "wb") as f:
        pickle.dump(accumulator, f)
    print(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Backproject DINOv3 patch embeddings into 3-D voxel space "
            "using first-hit perspective ray casting."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--specimens", "-s",
        nargs="+",
        default=["all"],
        metavar="SPECIMEN",
        help=(
            "Specimen folder(s) inside data/processed/bugNIST_900/ to process "
            "(e.g. GH SL WO).  Use 'all' to process every folder."
        ),
    )
    parser.add_argument(
        "--n-images", "-i",
        default="all",
        metavar="N|all",
        help=(
            "How many image folders to process per specimen. "
            "Use 'all' for every folder, or an integer (e.g. 1 for the first only)."
        ),
    )
    return parser.parse_args()


def resolve_specimens(requested: list[str]) -> list[Path]:
    available = sorted(p for p in PROC_ROOT.iterdir() if p.is_dir())
    if requested == ["all"]:
        return available
    resolved = []
    for name in requested:
        name = name.upper()
        path = PROC_ROOT / name
        if not path.is_dir():
            avail_names = [p.name for p in available]
            print(f"[ERROR] Specimen '{name}' not found in {PROC_ROOT}.")
            print(f"        Available specimens: {avail_names}")
            sys.exit(1)
        resolved.append(path)
    return resolved


def resolve_image_dirs(specimen_dir: Path, n_images: str) -> list[Path]:
    dirs = sorted(d for d in specimen_dir.iterdir() if d.is_dir())
    if not dirs:
        print(f"[WARN] No image folders found in {specimen_dir}")
        return []
    if n_images == "all":
        return dirs
    try:
        n = int(n_images)
    except ValueError:
        print(f"[WARN] --n-images must be an integer or 'all', got '{n_images}'. Using all.")
        return dirs
    return dirs[:n]


def main():
    args = parse_args()

    specimens = resolve_specimens(args.specimens)
    if not specimens:
        print("No valid specimen folders found. Exiting.")
        sys.exit(1)

    image_dirs = []
    for spec_dir in specimens:
        image_dirs.extend(resolve_image_dirs(spec_dir, args.n_images))

    if not image_dirs:
        print("No image folders to process. Exiting.")
        sys.exit(1)

    print(f"\nProcessing {len(image_dirs)} image folder(s).")
    for image_dir in image_dirs:
        specimen   = image_dir.parent.name
        image_name = image_dir.name
        emb_path   = EMB_ROOT / specimen / f"{image_name}.pkl"
        process_image(image_dir, emb_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()