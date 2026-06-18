"""
interactive_ray_test.py
-----------------------
Interactive pipeline to verify embedding similarity across views:

  1. Load the rendered 2D views for a specimen/image.
  2. Show the reference view in a matplotlib window — click a pixel to pick
     a reference patch.
  3. Find the most similar embedding patch in every other view (cosine
     similarity).
  4. Plot the reference view and every other view in a clean grid, with the
     matched patch highlighted in each, and save the figure as a
     print-ready PDF for the report.

Usage:
  python src/checks/interactive_ray_test.py

  # Custom specimen / image / view:
  python src/checks/interactive_ray_test.py \
      --specimen GH --image gras_9_043 --view view_000
"""

import argparse
import math
import pickle
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
SRC_DIR      = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

RAW_ROOT  = PROJECT_ROOT / "data" / "raw"        / "bugNIST_900"
PROC_ROOT = PROJECT_ROOT / "data" / "processed"  / "bugNIST_900"
EMB_ROOT  = PROJECT_ROOT / "data" / "embeddings" / "bugNIST_900"
FIG_ROOT  = PROJECT_ROOT / "figures"             # report-ready PDFs land here

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_W      = 1440
IMG_H      = 1440
PATCH_SIZE = 16
GRID_SIZE  = IMG_H // PATCH_SIZE   # 90


def patch_idx_to_rc(idx: int) -> tuple[int, int]:
    """Convert a flat patch index to (row, col)."""
    return (idx // GRID_SIZE, idx % GRID_SIZE)


def find_most_similar_patches(
    reference_embedding: np.ndarray,
    embeddings: dict,
    reference_view: str,
    top_k: int = 1,
) -> dict:
    """
    For each view (except *reference_view*), find the *top_k* patches most
    similar to *reference_embedding* by cosine similarity.

    Returns
    -------
    dict[str, list[tuple[int, float]]]
        ``{view_name: [(patch_idx, similarity), ...]}``
    """
    q = reference_embedding.astype(np.float64)
    q_norm = q / (np.linalg.norm(q) + 1e-12)

    results: dict[str, list[tuple[int, float]]] = {}
    for vn, emb_matrix in embeddings.items():
        if vn == reference_view:
            continue
        T = emb_matrix.astype(np.float64)
        T_norms = np.linalg.norm(T, axis=1, keepdims=True)
        T_norms = np.where(T_norms < 1e-12, 1.0, T_norms)
        sims = (T / T_norms) @ q_norm            # (n_patches,)
        top_indices = np.argsort(sims)[::-1][:top_k]
        results[vn] = [(int(idx), float(sims[idx])) for idx in top_indices]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Interactive click-to-patch embedding similarity viewer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--specimen", default="GH", help="Specimen name.")
    parser.add_argument("--image", default="gras_9_041", help="Image name (no ext).")
    parser.add_argument("--view", default="view_000", help="View name.")
    args = parser.parse_args()

    specimen   = args.specimen
    image_name = args.image
    view_name  = args.view

    # ── Load the rendered 2D views ──────────────────────────────────────────
    cam_pkl = PROC_ROOT / specimen / image_name / "cameras.pkl"
    print(f"Loading cameras: {cam_pkl.relative_to(PROJECT_ROOT)}")
    with open(cam_pkl, "rb") as f:
        cameras = pickle.load(f)

    img = cameras[view_name]["mip"]   # (H, W, C) uint8
    print(f"View image shape: {img.shape}")

    # ── Show the reference view and let the user click a pixel ─────────────────
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(f"Click on {view_name} to select a reference patch\n(close window when done)")
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

    # ── Convert the clicked pixel to a patch (row, col) ─────────────────────
    u_click, v_click = clicked_points[-1]
    patch_col = min(int(u_click) // PATCH_SIZE, GRID_SIZE - 1)
    patch_row = min(int(v_click) // PATCH_SIZE, GRID_SIZE - 1)
    patch_idx = patch_row * GRID_SIZE + patch_col

    print(f"  Clicked pixel: ({u_click:.1f}, {v_click:.1f})")
    print(f"  \u2192 Patch (row={patch_row}, col={patch_col}), flat index={patch_idx}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Embedding similarity — find the best matching patch in each view
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  EMBEDDING SIMILARITY")
    print("=" * 60)

    emb_pkl = EMB_ROOT / specimen / f"{image_name}.pkl"
    if not emb_pkl.exists():
        print(f"  [SKIP] Embeddings not found: {emb_pkl}")
        print("  Run generate_embeddings.py first. Exiting.")
        return

    with open(emb_pkl, "rb") as f:
        embeddings = pickle.load(f)
    print(f"  Loaded embeddings for {len(embeddings)} views.")

    if view_name not in embeddings:
        print(f"  [ERROR] Reference view '{view_name}' not in embeddings. "
              f"Available: {sorted(embeddings.keys())}")
        return

    reference_embedding = embeddings[view_name][patch_idx]  # (hidden_dim,)
    print(f"  Reference embedding shape: {reference_embedding.shape}")

    matches = find_most_similar_patches(
        reference_embedding, embeddings, view_name, top_k=1
    )

    print("\n  Best matches:")
    for vn, match_list in matches.items():
        for idx, sim in match_list:
            rc = patch_idx_to_rc(idx)
            print(f"    {vn}: patch {idx} (row={rc[0]}, col={rc[1]})  "
                  f"similarity={sim:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Report figure — reference view + all matched views, patches highlighted
    # ══════════════════════════════════════════════════════════════════════════
    from matplotlib.patches import Rectangle
    from matplotlib.colors import to_rgba

    # ── Light, print-friendly theme ──────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "text.color":        "#1a1a1a",
        "axes.labelcolor":   "#1a1a1a",
        "xtick.color":       "#1a1a1a",
        "ytick.color":       "#1a1a1a",
        "font.family":       "sans-serif",
        "font.sans-serif":   ["DejaVu Sans", "Helvetica", "Arial"],
    })

    # Collect all views to display (reference + matched)
    all_views_to_show = [view_name] + sorted(matches.keys())
    n_views = len(all_views_to_show)

    # ── Fixed, colour-blind-friendly palette (tab10-derived) ─────────────────
    plot_colors = {
        view_name: "#d62728",  # reference = red
    }
    match_palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd",
                      "#17becf", "#8c564b", "#bcbd22", "#e377c2",
                      "#7f7f7f", "#393b79"]
    for j, vn in enumerate(sorted(matches.keys())):
        plot_colors[vn] = match_palette[j % len(match_palette)]

    # Grid layout: aim for a square-ish arrangement
    n_cols = math.ceil(math.sqrt(n_views))
    n_rows = math.ceil(n_views / n_cols)

    # Sized for a printed page, not a screen — modest per-panel footprint
    cell_w, cell_h = 3.3, 3.7
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

    for i, vn in enumerate(all_views_to_show):
        row_idx, col_idx = divmod(i, n_cols)
        ax = axes[row_idx, col_idx]

        ax.imshow(cameras[vn]["mip"])

        color = plot_colors[vn]
        color_rgba = to_rgba(color, alpha=0.25)

        if vn == view_name:
            pr, pc = patch_row, patch_col
            label = "REFERENCE"
        else:
            match_idx, sim_val = matches[vn][0]
            pr, pc = patch_idx_to_rc(match_idx)
            label = f"sim = {sim_val:.4f}"

        px = pc * PATCH_SIZE
        py = pr * PATCH_SIZE

        # ── Filled highlight ───────────────────────────────────────────────
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=0, facecolor=color_rgba,
        ))
        # ── Crisp border ──────────────────────────────────────────────────
        ax.add_patch(Rectangle(
            (px, py), PATCH_SIZE, PATCH_SIZE,
            linewidth=2.0, edgecolor=color, facecolor="none",
        ))
        # ── Thin crosshair through the patch centre ─────────────────────────
        cx = px + PATCH_SIZE / 2
        cy = py + PATCH_SIZE / 2
        ax.axhline(cy, color=color, linewidth=0.5, alpha=0.4, linestyle="--")
        ax.axvline(cx, color=color, linewidth=0.5, alpha=0.4, linestyle="--")

        # ── Title ─────────────────────────────────────────────────────────
        ax.set_title(
            vn.replace("_", " ").upper(),
            fontsize=11, fontweight="bold", color=color, pad=6,
        )

        # ── Badge below image with similarity / label ───────────────────────
        ax.text(
            0.5, -0.05, label,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9.5, fontweight="bold", color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=color, edgecolor="none", alpha=0.9,
            ),
        )

        # ── Clean up axes ─────────────────────────────────────────────────
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(1.0)

    # Hide unused axes
    for i in range(n_views, n_rows * n_cols):
        row_idx, col_idx = divmod(i, n_cols)
        axes[row_idx, col_idx].axis("off")

    plt.subplots_adjust(top=0.94, bottom=0.06)

    # ── Save a print-ready vector copy for the report ───────────────────────
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = FIG_ROOT / f"similarity_{specimen}_{image_name}_{view_name}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"\nSaved report figure to: {out_path}")

    print("\n>>> Showing similarity grid. Close the window when done.")
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()