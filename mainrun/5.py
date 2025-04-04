import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nrbm_model import RBM  # Ensure rbm_model.py is in your working directory

# =========================
#   1) LOAD & PREP MODEL
# =========================
checkpoint_path = "checkpoint.pth"

# According to the checkpoint shapes, the model was trained with:
num_visible = 320  # e.g., 320 visible units
num_hidden = 80    # e.g., 80 hidden units

model = RBM(num_visible=num_visible, num_hidden=num_hidden)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
rbm_state = checkpoint["rbm_state_dict"]

# Manually fix shape mismatches:
# Checkpoint's 'W' is saved as [320, 80] (visible x hidden), but our model expects [80, 320]
model.W.data = rbm_state["W"].t()
# The checkpoint's visible bias 'a' is a scalar (shape [1]); expand it to a vector of size num_visible.
model.a.data = rbm_state["a"].expand(num_visible)
# Load hidden bias 'b' as is.
model.b.data = rbm_state["b"]

# =========================
#   2) EXTRACT & NORMALIZE
# =========================
# Now, model.W has shape [80, 320]: each row corresponds to one hidden unit's filter.
W = model.W.detach().numpy()
# Normalize each filter using z-score normalization.
W_norm = (W - W.mean(axis=1, keepdims=True)) / W.std(axis=1, keepdims=True)
# Sort filters by variance in descending order.
sorted_idx = np.argsort(np.var(W_norm, axis=1))[::-1]
W_sorted = W_norm[sorted_idx]

# ------------------------------------------------------------
# Helper function: Add filter label in top-left of subplot.
def label_filter(ax, f_index):
    ax.text(
        0.02, 0.15, 
        rf"$W^{{({f_index+1})}}$", 
        transform=ax.transAxes,
        fontsize=10, color="black", fontweight="bold"
    )

# ================================
#   3) FIGURE A: "Ising 1D"
# ================================
# Pick 4 filters for Ising 1D (for demonstration)
ising_1d_filters = W_sorted[:4]

figA, axesA = plt.subplots(nrows=4, ncols=1, figsize=(6, 6))
figA.suptitle("Ising 1D (Paper-style)", fontsize=16)

for i, ax in enumerate(axesA):
    # Reshape each filter into a 1x320 image.
    row_img = ising_1d_filters[i].reshape(1, -1)
    im = ax.imshow(
        row_img,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.15, vmax=0.15  # Set color range to match paper
    )
    ax.set_xticks([])
    ax.set_yticks([])
    label_filter(ax, i)

# Add a single colorbar for all subplots.
cbarA = figA.colorbar(im, ax=axesA.ravel().tolist(), fraction=0.02, pad=0.04)
cbarA.set_label("Weight Value")

figA.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figureA_ising_1d.png", dpi=300)
plt.close(figA)

# ================================
#   4) FIGURE B: "Heisenberg 1D"
# ================================
# Pick the next 4 filters for Heisenberg 1D (as an example)
heis_1d_filters = W_sorted[4:8]

figB, axesB = plt.subplots(nrows=4, ncols=1, figsize=(6, 6))
figB.suptitle("Heisenberg 1D (Paper-style)", fontsize=16)

for i, ax in enumerate(axesB):
    row_img = heis_1d_filters[i].reshape(1, -1)
    im = ax.imshow(
        row_img,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.15, vmax=0.15  # Same color range as Ising 1D
    )
    ax.set_xticks([])
    ax.set_yticks([])
    label_filter(ax, i + 4)
    
cbarB = figB.colorbar(im, ax=axesB.ravel().tolist(), fraction=0.02, pad=0.04)
cbarB.set_label("Weight Value")

figB.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figureB_heisenberg_1d.png", dpi=300)
plt.close(figB)

# =================================
#   5) FIGURE C: "Heisenberg 2D"
# =================================
# Pick 16 filters for Heisenberg 2D (example)
heis_2d_filters = W_sorted[8:8+16]

# We'll reshape each filter (length 320) into a 2D array.
# For example, 16 rows x 20 columns (16*20=320).
filter_rows, filter_cols = 16, 20

figC, axesC = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
figC.suptitle("Heisenberg 2D (Paper-style)", fontsize=16)

for i in range(16):
    ax = axesC[i // 4, i % 4]
    filt_2d = heis_2d_filters[i].reshape(filter_rows, filter_cols)
    im = ax.imshow(
        filt_2d,
        cmap="RdBu_r",
        vmin=-0.3, vmax=0.3  # Set color range for 2D filters as in the paper
    )
    ax.set_xticks([])
    ax.set_yticks([])
    label_filter(ax, i + 8)

# Add a colorbar for the 2D montage.
cbarC = figC.colorbar(im, ax=axesC.ravel().tolist(), fraction=0.02, pad=0.04)
cbarC.set_label("Weight Value")

figC.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figureC_heisenberg_2d.png", dpi=300)
plt.show()

