"""
generate_embeddings.py
----------------------
Pipeline (per processed image folder):
  1. Discover all view_*.png files inside data/processed/{specimen}/{image_name}/.
  2. Load each view through the DINOv3 image processor.
  3. Run a forward pass with the local DINOv3ViT model.
  4. Extract the patch-token embeddings (last_hidden_state, all tokens except
     the [CLS] and register tokens).
  5. Save a single dict  { "view_000": np.ndarray(shape=[n_patches, hidden]),
                            "view_001": ..., ... }
     as a pickle at  data/embeddings/{specimen}/{image_name}.pkl

Usage examples:
  # Embed all specimens, all images
  python src/generate_embeddings.py

  # Embed only GH, first 2 images
  python src/generate_embeddings.py --specimens GH --n-images 2

  # Embed GH and SL, all images
  python src/generate_embeddings.py --specimens GH SL --n-images all
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROC_ROOT    = PROJECT_ROOT / "data" / "processed" / "bugNIST_900"
EMB_ROOT     = PROJECT_ROOT / "data" / "embeddings" / "bugNIST_900"
MODEL_PATH   = PROJECT_ROOT / "models" / "vitb16"

# ── DINOv3 constants ──────────────────────────────────────────────────────────
# The model has 1 [CLS] token + num_register_tokens register tokens prepended
# before the patch tokens.  We skip those when extracting patch embeddings.
N_PREFIX_TOKENS = 5   # 1 CLS + 4 register tokens (see config.json)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(device: torch.device):
    """Load the local DINOv3 model and image processor."""
    print(f"Loading DINOv3 model from: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    processor = AutoImageProcessor.from_pretrained(str(MODEL_PATH))
    processor.do_resize = False
    model     = AutoModel.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()
    print(f"  Model loaded on {device}.")
    return processor, model


# ── Embedding extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def embed_views(image_dir: Path, processor, model, device: torch.device) -> dict:
    """
    Compute DINOv3 patch embeddings for every view_*.png in *image_dir*.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are view names (e.g. 'view_000').
        Values are float32 arrays of shape (n_patches, hidden_size).
    """
    view_paths = sorted(image_dir.glob("view_*.png"))
    if not view_paths:
        print(f"  [WARN] No view_*.png found in {image_dir}")
        return {}

    embeddings = {}
    for vp in view_paths:
        view_name = vp.stem   # e.g. "view_000"

        # Load image → PIL RGB (processor expects RGB)
        img = Image.open(vp).convert("RGB")

        # Preprocess: resize to 224×224, normalise, return torch tensor
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        outputs = model(**inputs)

        # last_hidden_state: (1, n_tokens, hidden_size)
        # Token layout: [CLS, reg_0, reg_1, reg_2, reg_3, patch_0, patch_1, …]
        hidden = outputs.last_hidden_state  # (1, n_tokens, hidden)
        patch_tokens = hidden[0, N_PREFIX_TOKENS:, :]   # (n_patches, hidden)

        embeddings[view_name] = patch_tokens.cpu().numpy().astype(np.float32)

    return embeddings


# ── Per-image processing ──────────────────────────────────────────────────────
def process_image(image_dir: Path, processor, model, device: torch.device) -> None:
    """
    Generate and save patch embeddings for one processed image folder.

    image_dir example: data/processed/GH/gras_9_042
    Output path:       data/embeddings/GH/gras_9_042.pkl
    """
    specimen   = image_dir.parent.name    # e.g. "GH"
    image_name = image_dir.name           # e.g. "gras_9_042"

    out_dir = EMB_ROOT / specimen
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_name}.pkl"

    print(f"\n{'='*60}")
    print(f"  Image : {image_dir.relative_to(PROJECT_ROOT)}")
    print(f"  Output: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"{'='*60}")

    embeddings = embed_views(image_dir, processor, model, device)

    if not embeddings:
        print("  [SKIP] No embeddings produced.")
        return

    # Report shape for first view
    first_key  = next(iter(embeddings))
    first_arry = embeddings[first_key]
    print(f"  Views embedded: {len(embeddings)}")
    print(f"  Patch embedding shape (per view): {first_arry.shape}")

    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate DINOv3 patch embeddings for processed MIP views.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--specimens", "-s",
        nargs="+",
        default=["all"],
        metavar="SPECIMEN",
        help=(
            "Specimen folder(s) inside data/processed/ to process "
            "(e.g. GH SL WO). Use 'all' to process every folder."
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
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run the model on. 'auto' picks CUDA > MPS > CPU.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


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
    t_start = time.time()

    args    = parse_args()
    device  = resolve_device(args.device)
    print(f"Device: {device}")

    processor, model = load_model(device)

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
        process_image(image_dir, processor, model, device)

    total_time = time.time() - t_start
    n_views = 15
    avg_per_view = total_time / n_views
    print(f"\n{'='*60}")
    print(f"  Total time:          {total_time:.2f} s")
    print(f"  Average per view:    {avg_per_view:.2f} s  ({n_views} views)")
    print(f"{'='*60}")
    print("\nAll done.")


if __name__ == "__main__":
    main()
