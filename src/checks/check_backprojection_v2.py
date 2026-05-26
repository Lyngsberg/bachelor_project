import sys
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from vedo import Volume, Plotter

# --- 1. Robust Path Resolution & Imports ---
SCRIPT_DIR = Path(__file__).resolve().parent  # points to src/checks
SRC_DIR = SCRIPT_DIR.parent                   # points to src
PROJECT_ROOT = SRC_DIR.parent                 # points to your project root

sys.path.insert(0, str(SRC_DIR))
from helper_functions import otsu_threshold

def main():
    # --- 2. Configuration ---
    specimen   = "GH"
    image_name = "gras_9_043"
    n_clusters = 5

    raw_tiff_path     = PROJECT_ROOT / "data" / "raw"     / "bugNIST_900" / specimen / f"{image_name}.tif"
    inferred_pkl_path = PROJECT_ROOT / "data" / "infered" / "bugNIST_900" / specimen / f"{image_name}.pkl"

    if not raw_tiff_path.exists():
        raise FileNotFoundError(f"Raw TIFF not found at: {raw_tiff_path}")
    if not inferred_pkl_path.exists():
        raise FileNotFoundError(f"Embeddings not found at: {inferred_pkl_path}")

    # --- 3. Load the Data ---
    print(f"Loading raw TIFF from {raw_tiff_path.relative_to(PROJECT_ROOT)}...")
    raw_vol  = Volume(str(raw_tiff_path))
    raw_data = raw_vol.tonumpy().astype(np.float32)
    spacing  = raw_vol.spacing()
    origin   = raw_vol.origin()
    volume_shape = raw_data.shape
    print(f"  Volume shape: {volume_shape}  spacing: {spacing}")

    print(f"Loading embeddings from {inferred_pkl_path.relative_to(PROJECT_ROOT)}...")
    with open(inferred_pkl_path, "rb") as f:
        voxel_embeddings = pickle.load(f)
    print(f"  Loaded embeddings for {len(voxel_embeddings):,} voxels.")

    # --- 4. Process Embeddings ---
    # Average all embeddings accumulated per voxel (multiple views hitting the
    # same voxel all contribute; take the mean as a single representative vector).
    coords = []
    embs   = []
    for (ix, iy, iz), emb_list in voxel_embeddings.items():
        coords.append((ix, iy, iz))
        embs.append(np.mean(emb_list, axis=0))

    coords = np.array(coords)          # (M, 3)
    embs   = np.array(embs)            # (M, hidden_dim)
    print(f"  Embedding matrix: {embs.shape}")

    # --- 5. Clustering ---
    print(f"Running K-Means clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embs)   # 0 … n_clusters-1
    for c in range(n_clusters):
        print(f"  Cluster {c}: {int((labels == c).sum()):,} voxels")

    # --- 6. Construct 3D label volume ---
    # 0 = background (transparent), 1…n_clusters = cluster labels
    cluster_vol_data = np.zeros(volume_shape, dtype=np.uint8)
    for (ix, iy, iz), label in zip(coords, labels):
        cluster_vol_data[ix, iy, iz] = int(label) + 1

    # Remove any labelled voxels that fall outside the 3D Otsu foreground mask
    print("Applying Otsu mask to remove stray background voxels...")
    mask = otsu_threshold(raw_data, log_scale=True)
    cluster_vol_data[~mask] = 0
    print(f"  Labelled voxels after masking: {int((cluster_vol_data > 0).sum()):,}")

    # --- 7. Rendering ---
    print("Preparing vedo render...")
    clean_vol = Volume(cluster_vol_data, spacing=spacing, origin=origin)

    # Scalar range: 0 (background) through n_clusters (last label).
    # Alpha list: one value per integer level — 0 is transparent, clusters are opaque.
    alpha_values = [0.0] + [0.9] * n_clusters
    clean_vol.mode(0).cmap("jet", vmin=0, vmax=n_clusters).alpha(alpha_values)

    plt = Plotter(axes=1, title="Embedding Clusters Viewer")
    print("Opening viewer... Rotate the model! (Close window to exit)")
    plt.show(clean_vol, interactive=True)

if __name__ == "__main__":
    main()