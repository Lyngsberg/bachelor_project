import numpy as np
from pathlib import Path
from vedo import Volume, Plotter, Text2D

# 1. Setup paths robustly
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
full_path = PROJECT_ROOT / "data" / "raw" / "bugNIST_900" / "GH" / "gras_9_043.tif"

if not full_path.exists():
    print(f"Error: Could not find file at {full_path}")
    exit()

filter_value = 60
print("Loading 3D Volume...")
vol = Volume(str(full_path))

# 2. Filter the background noise
print(f"Filtering intensities below {filter_value}...")
vol_data = vol.tonumpy()
print(f"Volume shape: {vol_data.shape}")
vol_data[vol_data < filter_value] = 0
clean_vol = Volume(vol_data, spacing=vol.spacing(), origin=vol.origin())
clean_vol.mode(0).cmap("jet").alpha([0, 0.2, 0.8, 1])

# 3. Create Plotter and the HUD overlay
plt = Plotter(axes=1, title="Live Math HUD")

# Setup the 2D text object (top-left corner, monospaced font for clean matrices)
hud_text = Text2D(
    "Move camera to initialize matrices...", 
    pos="top-left", 
    font="Mono", 
    s=0.7, 
    bg="black", 
    alpha=0.6
)

# 4. Define the callback function that runs on every mouse movement
def update_hud(event):
    cam = plt.camera
    pos = np.array(cam.GetPosition())
    focal_point = np.array(cam.GetFocalPoint())
    viewup = np.array(cam.GetViewUp())
    
    window_width, window_height = plt.window.GetSize()
    fov_degrees = cam.GetViewAngle()
    
    # --- R Matrix ---
    forward = focal_point - pos
    norm_f = np.linalg.norm(forward)
    if norm_f == 0: return # Prevent division by zero errors on initialization
    forward = forward / norm_f
    
    right = np.cross(forward, viewup)
    norm_r = np.linalg.norm(right)
    if norm_r == 0: return
    right = right / norm_r
    
    true_up = np.cross(right, forward)
    true_up = true_up / np.linalg.norm(true_up)
    
    R = np.vstack([right, true_up, forward])
    
    # --- K Matrix ---
    fov_radians = np.radians(fov_degrees)
    focal_length = (window_height / 2.0) / np.tan(fov_radians / 2.0)
    cx = window_width / 2.0
    cy = window_height / 2.0
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    # Format the NumPy arrays so the decimals line up beautifully on screen
    r_str = np.array2string(R, formatter={'float_kind':lambda x: f"{x:6.3f}"})
    k_str = np.array2string(K, formatter={'float_kind':lambda x: f"{x:8.1f}"})
    
    # Update the text on the screen
    hud_str = (
        f"--- LIVE CAMERA MATRICES ---\n\n"
        f"Rotation Matrix (R):\n{r_str}\n\n"
        f"Intrinsic Matrix (K):\n{k_str}\n\n"
        f"Distance to target: {cam.GetDistance():.1f}"
    )
    hud_text.text(hud_str)

# 5. Attach the callback to the mouse interaction
plt.add_callback("InteractionEvent", update_hud)

# 6. Show the volume and the HUD
print("Opening viewer... Rotate the grasshopper to see the matrices update live!")
plt.show(clean_vol, hud_text, interactive=True)