"""
camera_inspection.py
--------------------
Load and display the contents of one cameras.pkl file.
"""

import pickle
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKL_PATH = PROJECT_ROOT / "data" / "processed" / "GH" / "gras_9_042" / "cameras.pkl"

# ── Load ──────────────────────────────────────────────────────────────────────
with open(PKL_PATH, "rb") as f:
    cameras = pickle.load(f)

print(f"Loaded: {PKL_PATH.relative_to(PROJECT_ROOT)}")
print(f"Number of views: {len(cameras)}\n")

# ── Inspect first view ────────────────────────────────────────────────────────
first_key = next(iter(cameras))
cam = cameras[first_key]

print(f"=== {first_key} ===")
for key, value in cam.items():
    if isinstance(value, np.ndarray):
        print(f"  {key:15s}: shape={value.shape}  dtype={value.dtype}")
        print(f"  {'':<15s}  {value}")
    else:
        print(f"  {key:15s}: {value}")
