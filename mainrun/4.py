import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nrbm_model import RBM  # Make sure rbm_model.py is in your path

# --- CONFIGURATION ---
# According to the checkpoint, the model was trained with:
num_visible = 320   # Total number of visible units
num_hidden = 80     # Total number of hidden units
checkpoint_path = "checkpoint.pth"

# --- LOAD THE MODEL ---
model = RBM(num_visible=num_visible, num_hidden=num_hidden)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
rbm_state = checkpoint["rbm_state_dict"]

# Manually load weights adjusting for shape mismatches:
# Checkpoint's weight matrix 'W' has shape [320, 80] (visible x hidden)
# but your model expects shape [80, 320], so we transpose.
model.W.data = rbm_state["W"].t()

# The checkpoint's visible bias 'a' is scalar (shape [1]); expand it to match num_visible.
model.a.data = rbm_state["a"].expand(num_visible)
# Hidden bias 'b' can be loaded as is.
model.b.data = rbm_state["b"]

# --- EXTRACT AND NORMALIZE WEIGHTS ---
# Get W, which now has shape [80, 320]: each row is a filter for one hidden unit.
W = model.W.detach().numpy()
# Normalize each filter (row) using z-score normalization:
W_norm = (W - W.mean(axis=1, keepdims=True)) / W.std(axis=1, keepdims=True)
# Sort the filters by variance (descending order)
sorted_idx = np.argsort(np.var(W_norm, axis=1))[::-1]
W_sorted = W_norm[sorted_idx]

# --- CREATE A HEATMAP (Optional) ---
plt.figure(figsize=(10, 6))
sns.heatmap(W_sorted, cmap="coolwarm", cbar=True)
plt.title("Normalized & Sorted RBM Filters (W) — Paper-style")
plt.xlabel("Visible Units")
plt.ylabel("Sorted Hidden Units")
plt.tight_layout()
plt.savefig("W_heatmap_paper_style.png", dpi=300)
plt.close()

# --- CREATE A 2D MONTAGE OF FILTERS ---
# We'll reshape each filter (of length 320) into a 2D array.
# Choose shape factors such that rows * cols = 320. For example, 16 x 20.
filter_rows, filter_cols = 16, 20

num_filters = W_sorted.shape[0]  # should be 80
# We'll arrange the filters in a grid montage, e.g., 10 rows x 8 columns (10*8 = 80)
montage_rows, montage_cols = 10, 8

fig, axes = plt.subplots(montage_rows, montage_cols, figsize=(montage_cols*1.5, montage_rows*1.5))
for i in range(num_filters):
    filt = W_sorted[i]  # shape (320,)
    # Reshape to 2D
    filt_2d = filt.reshape(filter_rows, filter_cols)
    ax = axes[i // montage_cols, i % montage_cols]
    im = ax.imshow(filt_2d, cmap="coolwarm", interpolation="nearest")
    ax.axis("off")
    
fig.suptitle("2D Reshaped RBM Filters (Sorted)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("W_montage_2d.png", dpi=300)
plt.show()

