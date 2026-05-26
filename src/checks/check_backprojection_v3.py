import sys
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from vedo import Volume, Plotter, Points

# --- 1. Robust Path Resolution & Imports ---
SCRIPT_DIR = Path(__file__).resolve().parent  # points to src/checks
SRC_DIR = SCRIPT_DIR.parent                   # points to src
PROJECT_ROOT = SRC_DIR.parent                 # points to your project root

sys.path.insert(0, str(SRC_DIR))
# otsu_threshold is no longer strictly needed for rendering, 
# but kept if you want to use it for data filtering later.
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
    raw_data = raw_vol.tonumpy().astype(np.float32) # Need the actual data array now
    spacing  = raw_vol.spacing()
    origin   = raw_vol.origin()
    volume_shape = raw_vol.dimensions()
    print(f"  Volume shape: {volume_shape}  spacing: {spacing}")

    print(f"Loading embeddings from {inferred_pkl_path.relative_to(PROJECT_ROOT)}...")
    with open(inferred_pkl_path, "rb") as f:
        voxel_embeddings = pickle.load(f)
    print(f"  Loaded embeddings for {len(voxel_embeddings):,} voxels.")

    # --- 4. Process & Filter Embeddings ---
    coords = []
    embs   = []
    FILTER_VALUE = 60 # Using your threshold from 3d_2d_view.py

    for (ix, iy, iz), emb_list in voxel_embeddings.items():
        # ONLY keep the voxel if the raw CT density is above the cylinder noise
        if raw_data[ix, iy, iz] >= FILTER_VALUE:
            coords.append((ix, iy, iz))
            embs.append(np.mean(emb_list, axis=0))

    coords = np.array(coords)          
    embs   = np.array(embs)            
    print(f"  Embeddings kept after intensity filter (>= {FILTER_VALUE}): {len(coords):,}")

    if len(coords) == 0:
        print("Error: The filter removed all points! Check your raw data scale.")
        sys.exit(1)

    # --- 5. Clustering (Runs only on the filtered insect points) ---
    print(f"Running K-Means clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embs)

    # --- 4. Process Embeddings ---
    # You nailed this part: Averaging the list of embeddings per voxel.
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

    # --- 6. Construct Point Cloud (Replaces Volume Rendering) ---
    print("Preparing vedo Point Cloud render...")
    
    # Convert discrete voxel indices back to physical world coordinates 
    # so the 3D aspect ratio is completely correct.
    real_coords = coords * np.array(spacing) + np.array(origin)

    # Create a PointCloud from the coordinates. 
    # r=4 sets the point size. You can adjust this up or down to make it look solid.
    pts = Points(real_coords, r=4)
    
    # Apply the KMeans labels directly as a colormap
    pts.cmap("jet", labels)

    # --- 7. Rendering ---
    plt = Plotter(axes=1, title="Embedding Point Cloud Viewer")
    print("Opening viewer... Rotate the model! (Close window to exit)")
    plt.show(pts, interactive=True)

if __name__ == "__main__":
    main()