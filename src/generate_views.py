"""
generate_views.py
-----------------
Pipeline (per TIFF):
  1. Load a 3D TIFF volume via vedo (preserves spacing/origin).
  2. Filter low-intensity background noise.
  3. Generate N camera positions uniformly distributed on a sphere.
  4. For each position, render a 2D view using vedo volume rendering
     (composite ray-casting with alpha transfer function — same method
     as 3d_2d_view.py).
  5. Save each view as a PNG and all camera data as one cameras.pkl.

Usage examples:
  # 10 views for the first GH image
  python src/generate_views.py --specimens GH --n-images 1 --n-views 10

  # 5 views for all images in GH and SL
  python src/generate_views.py --specimens GH SL --n-images all --n-views 5

  # 8 views for every image across all specimens
  python src/generate_views.py --specimens all --n-images all --n-views 8
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from vedo import Volume, Plotter

# Make the src package importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helper_functions import uniform_sphere_points, otsu_threshold

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_ROOT     = PROJECT_ROOT / "data" / "raw" / "bugNIST_900"
PROC_ROOT    = PROJECT_ROOT / "data" / "processed" / "bugNIST_900"

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE     = (1440, 1440)  # (width, height) of rendered images

# ── Camera helpers ────────────────────────────────────────────────────────────
def rot_z(deg):
    """Rotation matrix around Z axis (right-hand, active)."""
    r = np.radians(deg)
    return np.array([[ np.cos(r), -np.sin(r), 0],
                     [ np.sin(r),  np.cos(r), 0],
                     [ 0,          0,         1]], dtype=np.float64)

def rot_y(deg):
    """Rotation matrix around Y axis (right-hand, active)."""
    r = np.radians(deg)
    return np.array([[ np.cos(r), 0, np.sin(r)],
                     [ 0,         1, 0        ],
                     [-np.sin(r), 0, np.cos(r)]], dtype=np.float64)


# ── Core processing ───────────────────────────────────────────────────────────
def process_tiff(input_path: Path, n_views: int) -> None:
    """Generate volume-rendered views and camera data for a single TIFF volume."""
    # Derive output dir: data/raw/GH/foo.tif → data/processed/GH/foo/
    rel        = input_path.relative_to(RAW_ROOT)
    output_dir = PROC_ROOT / rel.parent / rel.stem

    print(f"\n{'='*60}")
    print(f"  Input : {input_path.relative_to(PROJECT_ROOT)}")
    print(f"  Output: {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"{'='*60}")

    # 1. Load volume with vedo (preserves TIFF spacing and origin)
    print("Loading volume …")
    original_vol = Volume(str(input_path))
    spacing = original_vol.spacing()
    origin  = original_vol.origin()
    vol_data = original_vol.tonumpy().astype(np.float32)
    volume_shape = vol_data.shape
    print(f"  Shape: {volume_shape}  spacing: {spacing}")

    # 2. Filter background noise (same threshold as 3d_2d_view.py)
    print(f"Filtering intensities using Otsu …")
    binary_mask = otsu_threshold(vol_data, log_scale=True)
    vol_data[~binary_mask] = 0
    n_retained = int((vol_data > 0).sum())
    print(f"  Retained voxels: {n_retained:,} / {vol_data.size:,}")

    # 3. Create vedo Volume with rendering settings matching 3d_2d_view.py
    #    mode(0) = composite ray-casting with alpha compositing
    vol = Volume(vol_data, spacing=spacing, origin=origin)
    vol.mode(0).cmap("bone").alpha([0, 0, 1, 1])

    center = np.array(vol.center())
    vol_center_numpy = np.array(volume_shape, dtype=np.float64) / 2.0
    print(f"  Volume center: {center}")

    # 4. Generate sphere points (camera directions)
    print(f"Generating {n_views} camera directions on the sphere …")
    diag = np.linalg.norm(np.array(volume_shape) * np.array(spacing))
    dynamic_radius = diag * 1.5
    pts = uniform_sphere_points(n_views, radius=dynamic_radius)

    # 5. Prepare output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    print(f"Saving views to {output_dir} …")

    # 6. Create plotter for rendering (reused for all views)
    #    Using interactive=False with the system DISPLAY rather than
    #    offscreen=True, which requires OSMesa/EGL (not always available).
    plt = Plotter(size=IMG_SIZE, bg="black")
    plt.show(vol, interactive=False, resetcam=False)

    cameras = {}

    for idx, (px, py, pz) in enumerate(pts):
        cam_pos = center + np.array([px, py, pz])

        # Robust view-up vector: avoid degenerate case when looking
        # straight along z-axis (dot product ≈ ±1).
        view_dir = center - cam_pos
        view_dir = view_dir / np.linalg.norm(view_dir)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(view_dir, up)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])

        # Position camera
        plt.camera.SetPosition(*cam_pos)
        plt.camera.SetFocalPoint(*center)
        plt.camera.SetViewUp(*up)
        plt.render()

        # Save rendered image
        view_name = f"view_{idx:03d}"
        out_path  = output_dir / f"{view_name}.png"
        plt.screenshot(str(out_path))

        # Get rendered image as numpy array for the cameras dict
        img_array = plt.screenshot(asarray=True)

        # # Convert to grayscale float [0, 1] for backward-compatible 'mip' field
        # if img_array is not None and img_array.ndim == 3:
        #     mip = np.mean(img_array[:, :, :3], axis=2).astype(np.float32) / 255.0
        # else:
        #     mip = np.zeros((IMG_SIZE[1], IMG_SIZE[0]), dtype=np.float32)

        # Camera geometry (kept for downstream compatibility)
        az = np.degrees(np.arctan2(py, px))
        el = np.degrees(np.arcsin(np.clip(pz / dynamic_radius, -1, 1)))
        R  = rot_y(el) @ rot_z(-az)                                       # (3, 3)
        t  = cam_pos.copy()                                                # (3,)


        img_h, img_w, img_d = img_array.shape
        K = np.array([[1.0,  0.0, img_w / 2.0],
                      [0.0,  1.0, img_h / 2.0],
                      [0.0,  0.0, 1.0        ]], dtype=np.float64)

        cameras[view_name] = {
            "image_file"   : f"{view_name}.png",
            "view_index"   : idx,
            "az_deg"       : az,
            "el_deg"       : el,
            "R"            : R,          # world-to-camera rotation
            "t"            : t,          # camera centre in world coords
            "K"            : K,          # intrinsic matrix
            "projection"   : "perspective",
            "volume_shape" : volume_shape,
            "vol_center"   : vol_center_numpy,
            "sphere_radius": dynamic_radius,
            "mip"          : img_array,
        }

        print(f"  [{idx+1:>{len(str(n_views))}}/{n_views}]  {out_path.name}")

    plt.close()

    # Save single camera pickle for this volume
    pkl_path = output_dir / "cameras.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(cameras, f)
    print(f"  Camera data → {pkl_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 2D volume-rendered views from 3D TIFF volumes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--specimens", "-s",
        nargs="+",
        default=["all"],
        metavar="SPECIMEN",
        help=(
            "Specimen folder(s) inside data/raw/ to process "
            "(e.g. GH SL WO). Use 'all' to process every folder."
        ),
    )
    parser.add_argument(
        "--n-images", "-i",
        default="all",
        metavar="N|all",
        help=(
            "How many TIFF images to process per specimen. "
            "Use 'all' for every image, or an integer (e.g. 1 for the first only)."
        ),
    )
    parser.add_argument(
        "--n-views", "-v",
        type=int,
        default=5,
        metavar="N",
        help="Number of camera positions (views) to generate per TIFF.",
    )
    return parser.parse_args()


def resolve_specimens(requested: list[str]) -> list[Path]:
    """Return sorted list of specimen directories to process."""
    available = sorted(p for p in RAW_ROOT.iterdir() if p.is_dir())
    if requested == ["all"]:
        return available
    resolved = []
    for name in requested:
        name = name.upper()   
        path = RAW_ROOT / name
        if not path.is_dir():
            available_names = [p.name for p in available]
            print(f"[ERROR] Specimen '{name}' not found in {RAW_ROOT}.")
            print(f"        Available specimens: {available_names}")
            sys.exit(1)
        resolved.append(path)
    return resolved


def resolve_tiffs(specimen_dir: Path, n_images: str) -> list[Path]:
    """Return the TIFFs to process for a given specimen directory."""
    tiffs = sorted(specimen_dir.glob("*.tif"))
    if not tiffs:
        print(f"[WARN] No .tif files found in {specimen_dir}")
        return []
    if n_images == "all":
        return tiffs
    try:
        n = int(n_images)
    except ValueError:
        print(f"[WARN] --n-images must be an integer or 'all', got '{n_images}'. Using all.")
        return tiffs
    return tiffs[:n]


def main():
    args = parse_args()

    specimens = resolve_specimens(args.specimens)
    if not specimens:
        print("No valid specimen folders found. Exiting.")
        sys.exit(1)

    tiff_list = []
    for spec_dir in specimens:
        tiff_list.extend(resolve_tiffs(spec_dir, args.n_images))

    if not tiff_list:
        print("No TIFF files to process. Exiting.")
        sys.exit(1)

    print(f"Processing {len(tiff_list)} TIFF(s) with {args.n_views} view(s) each.")
    for tiff_path in tiff_list:
        process_tiff(tiff_path, args.n_views)

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
