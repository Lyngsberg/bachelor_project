"""
rotation_check.py
-----------------
Checks two properties of the local DINOv3 model on bugNIST900 GH views:

  1. **Determinism** – Run the same 5 views through the model twice and
     compare patch embeddings.  If the model is deterministic the cosine
     similarity between run-1 and run-2 embeddings should be exactly 1.0
     for every patch.

  2. **Rotation equivariance** – Rotate each view 180° and run through
     the model again.  If the model is perfectly rotation-equivariant,
     the patch at position (r, c) in the rotated image should have the
     same embedding as the patch at position (R-1-r, C-1-c) in the
     original image (i.e. the spatial order reverses).

Usage:
  python src/checks/rotation_check.py
  python src/checks/rotation_check.py --device cuda
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
SRC_DIR      = SCRIPT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
PROC_ROOT    = PROJECT_ROOT / "data" / "processed" / "bugNIST_900"
MODEL_PATH   = PROJECT_ROOT / "models" / "dinov3_local"

# ── Constants ─────────────────────────────────────────────────────────────────
SPECIMEN   = "GH"
IMAGE_NAME = "gras_9_043"
N_VIEWS    = 5                 # use view_000 .. view_004
N_PREFIX_TOKENS = 5            # 1 [CLS] + 4 register tokens


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def cosine_sim_paired(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine similarity between two (N, D) arrays.

    Returns
    -------
    np.ndarray of shape (N,)
    """
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    dot  = np.sum(a64 * b64, axis=1)
    norm = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1)
    return dot / np.maximum(norm, 1e-12)


def print_stats(label: str, sims: np.ndarray) -> None:
    """Pretty-print min / max / mean / std of a similarity array."""
    print(f"  {label}")
    print(f"    min  = {sims.min():.8f}")
    print(f"    max  = {sims.max():.8f}")
    print(f"    mean = {sims.mean():.8f}")
    print(f"    std  = {sims.std():.8f}")
    n_perfect = int((sims >= 1.0 - 1e-6).sum())
    print(f"    patches with sim ≥ 0.999999: {n_perfect}/{len(sims)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Check DINOv3 embedding determinism and rotation equivariance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device", default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run the model on.",
    )
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ── Load model ────────────────────────────────────────────────────────────
    processor, model = load_model(device)

    # ── Discover view images ──────────────────────────────────────────────────
    image_dir = PROC_ROOT / SPECIMEN / IMAGE_NAME
    view_names = [f"view_{i:03d}" for i in range(N_VIEWS)]
    view_paths = [image_dir / f"{vn}.png" for vn in view_names]

    for vp in view_paths:
        if not vp.exists():
            print(f"[ERROR] View image not found: {vp}")
            sys.exit(1)

    print(f"\nUsing {N_VIEWS} views from {image_dir.relative_to(PROJECT_ROOT)}:")
    for vn in view_names:
        print(f"  - {vn}")

    # ── Load PIL images ───────────────────────────────────────────────────────
    pil_images = {}
    for vn, vp in zip(view_names, view_paths):
        pil_images[vn] = Image.open(vp).convert("RGB")
    img_w, img_h = pil_images[view_names[0]].size
    print(f"\nImage size: {img_w} × {img_h}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 1: Determinism check — run the same images twice
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 1: DETERMINISM CHECK  (same images, two forward passes)")
    print("=" * 70)

    embeddings_run1 = {}
    embeddings_run2 = {}

    for vn in view_names:
        img = pil_images[vn]
        embeddings_run1[vn] = embed_image(img, processor, model, device)
        embeddings_run2[vn] = embed_image(img, processor, model, device)

    print(f"\n  Patch embedding shape: {embeddings_run1[view_names[0]].shape}")

    print("\n  Per-view cosine similarity between run-1 and run-2:")
    all_det_sims = []
    for vn in view_names:
        sims = cosine_sim_paired(embeddings_run1[vn], embeddings_run2[vn])
        all_det_sims.append(sims)
        print_stats(vn, sims)

    all_det_sims = np.concatenate(all_det_sims)
    print("\n  ── Aggregated across all views ──")
    print_stats("All patches", all_det_sims)

    if all_det_sims.min() >= 1.0 - 1e-6:
        print("\n  ✓ Model appears DETERMINISTIC — all embeddings are identical.")
    else:
        print("\n  ✗ Model shows RANDOMNESS — embeddings differ between runs!")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART 2: Rotation equivariance — rotate 180° and compare
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 2: ROTATION EQUIVARIANCE  (original vs 180° rotated)")
    print("=" * 70)

    # The model with do_resize=False on 1440×1440 images produces a
    # 90×90 = 8100 patch grid.  A 180° rotation should reverse the
    # spatial order: original patch[i] ↔ rotated patch[n_patches - 1 - i].
    n_patches = embeddings_run1[view_names[0]].shape[0]
    grid_side = int(np.sqrt(n_patches))
    print(f"\n  n_patches = {n_patches}  (grid {grid_side}×{grid_side})")
    print(f"  Comparing: original patch[i] vs rotated patch[{n_patches - 1} - i]")

    embeddings_rotated = {}
    for vn in view_names:
        img_rot = pil_images[vn].rotate(180)       # PIL 180° rotation
        embeddings_rotated[vn] = embed_image(img_rot, processor, model, device)

    print("\n  Per-view cosine similarity (original[i] vs rotated[n-1-i]):")
    all_rot_sims = []
    for vn in view_names:
        emb_orig = embeddings_run1[vn]                   # (n_patches, hidden)
        emb_rot  = embeddings_rotated[vn]                # (n_patches, hidden)

        # Reverse the rotated embeddings so index i matches i
        emb_rot_reversed = emb_rot[::-1].copy()

        sims = cosine_sim_paired(emb_orig, emb_rot_reversed)
        all_rot_sims.append(sims)
        print_stats(vn, sims)

    all_rot_sims = np.concatenate(all_rot_sims)
    print("\n  ── Aggregated across all views ──")
    print_stats("All patches", all_rot_sims)

    if all_rot_sims.min() >= 1.0 - 1e-6:
        print("\n  ✓ Model appears perfectly ROTATION-EQUIVARIANT at 180°.")
    elif all_rot_sims.mean() >= 0.99:
        print(f"\n  ~ Model is NEARLY rotation-equivariant "
              f"(mean sim = {all_rot_sims.mean():.6f}).")
        print("    Small deviations likely come from positional embeddings.")
    else:
        print(f"\n  ✗ Model is NOT rotation-equivariant at 180° "
              f"(mean sim = {all_rot_sims.mean():.6f}).")
        print("    Positional embeddings or asymmetric processing breaks equivariance.")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Check':<35} {'Mean sim':>10} {'Std':>10} {'Min':>10}")
    print(f"  {'─' * 35} {'─' * 10} {'─' * 10} {'─' * 10}")
    print(f"  {'Determinism (run1 vs run2)':<35} "
          f"{all_det_sims.mean():>10.6f} {all_det_sims.std():>10.6f} "
          f"{all_det_sims.min():>10.6f}")
    print(f"  {'Rotation (orig vs 180° reversed)':<35} "
          f"{all_rot_sims.mean():>10.6f} {all_rot_sims.std():>10.6f} "
          f"{all_rot_sims.min():>10.6f}")
    print("=" * 70)

    print("\nDone.")


if __name__ == "__main__":
    main()
