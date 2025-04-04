import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nrbm_model import RBM  # Ensure your rbm_model.py is accessible

# --- CONFIGURATION ---
# According to the checkpoint shapes, the model was trained with:
num_visible = 320
num_hidden = 80
checkpoint_path = "checkpoint.pth"

# --- LOAD THE MODEL ---
model = RBM(num_visible=num_visible, num_hidden=num_hidden)
# Load the checkpoint with full weights
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
rbm_state = checkpoint["rbm_state_dict"]

# --- MANUALLY LOAD WEIGHTS ---
# The checkpoint's 'W' shape is [320, 80] but your model expects [80, 320],
# so we transpose the weight matrix.
model.W.data = rbm_state["W"].t()

# The checkpoint's 'a' is a scalar (shape [1]); expand it to match num_visible.
model.a.data = rbm_state["a"].expand(num_visible)

# Load hidden biases as is
model.b.data = rbm_state["b"]

# --- EXTRACT AND PROCESS WEIGHTS ---
W = model.W.detach().numpy()  # Now shape [80, 320]
# Normalize each filter (each row represents a filter)
W_norm = (W - W.mean(axis=1, keepdims=True)) / W.std(axis=1, keepdims=True)
# Sort filters by their variance (descending)
sorted_idx = np.argsort(np.var(W_norm, axis=1))[::-1]
W_sorted = W_norm[sorted_idx]

# --- PLOT THE HEATMAP ---
plt.figure(figsize=(10, 6))
sns.heatmap(W_sorted, cmap="coolwarm", cbar=True)
plt.title("Normalized & Sorted RBM Filters (W) — Paper-style")
plt.xlabel("Visible Units")
plt.ylabel("Sorted Hidden Units")
plt.tight_layout()
plt.savefig("W_heatmap_paper_style.png", dpi=300)
plt.show()

