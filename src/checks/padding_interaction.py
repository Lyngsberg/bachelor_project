from transformers import AutoImageProcessor, AutoModel
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

import matplotlib.pyplot as plt

# 1. Load Model and Processor
local_model_path = "models/dinov3_local"

processor = AutoImageProcessor.from_pretrained(local_model_path, local_files_only=True)
model = AutoModel.from_pretrained(local_model_path, local_files_only=True)
model.eval() # Ensure model is in evaluation mode

# Helper function to extract the global embedding
def get_embedding(img, height=224, width=224):
    inputs = processor(
        images=img, 
        return_tensors="pt",
        do_resize=True,
        size={"height": height, "width": width},
        do_center_crop=False # <- Crucial to keep your padding intact!
    )
    with torch.no_grad():
        outputs = model(**inputs)
        
    cls_embedding = outputs.last_hidden_state[:, 0, :] 
    return cls_embedding

# 2. Establish the Baseline (0 padding)
image = Image.open("data/examples/racecars_start.jpg").convert("RGB")
baseline_emb = get_embedding(image, 512, 512)

# Pre-calculate the original image area for percentage math
orig_area = image.width * image.height

# 3. Test Robustness to Zero Padding
padding_amounts = [10, 50, 100, 200, 500]
print("\n--- Testing Zero Padding Robustness ---")

# --- VISUALIZATION SETUP ---
num_images = len(padding_amounts) + 1
fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

# Plot Baseline
# Resize to 512x512 just for the visual plot to match the model input
axes[0].imshow(image.resize((512, 512)))
axes[0].set_title("Baseline (0px)\nOrig Area: 100.0%\nSim: 1.0000")
axes[0].axis('off')

for i, pad in enumerate(padding_amounts):
    padded_image = ImageOps.expand(image, border=pad, fill=0)
    
    new_area = padded_image.width * padded_image.height
    pct_original = (orig_area / new_area) * 100
    
    padded_emb = get_embedding(padded_image, 512, 512)

    similarity = F.cosine_similarity(baseline_emb, padded_emb).item()
    print(f"Padding: {pad:>3}px | Orig Area: {pct_original:>5.1f}% | Sim: {similarity:.4f}")
    
    # Plot Padded Image
    ax = axes[i + 1]
    # Resize to 512x512 to ensure all plot boxes are identical squares
    ax.imshow(padded_image.resize((512, 512)))
    ax.set_title(f"Pad: {pad}px\nOrig Area: {pct_original:.1f}%\nSim: {similarity:.4f}")
    ax.axis('off')

# Display the final plot
plt.tight_layout()
plt.show()