import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nrbm_model import RBM  # Make sure rbm_model.py is accessible

# =========================
#   1) LOAD & PREP MODEL
# =========================
checkpoint_path = "checkpoint.pth"

# As discovered, the checkpoint has W shape [320,80], so we do:
num_visible = 320
num_hidden = 80

model = RBM(num_visible=num_visible, num_hidden=num_hidden)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
rbm_state = checkpoint["rbm_state_dict"]

# Manually fix shape mismatches:
model.W.data = rbm_state["W"].t()                 # shape => [80, 320]
model.a.data = rbm_state["a"].expand(num_visible) # shape => [320]
model.b.data = rbm_state["b"]                     # shape => [80]

# =========================
#   2) EXTRACT & NORMALIZE
# =========================
W = model.W.detach().numpy()  # shape: [80, 320]
# Normalize each filter (row) with z-score
W_norm = (W - W.mean(axis=1, keepdims=True)) / W.std(axis=1, keepdims=True)
# Sort filters by variance (descending)
sorted_idx = np.argsort(np.var(W_norm, axis=1))[::-1]
W_sorted = W_norm[sorted_idx]

# ------------------------------------------------------------
#   HELPER: Label subplots with W^(i) in top-left corner
# ------------------------------------------------------------
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
# We'll just pick 4 filters to mimic the paper's style
ising_1d_filters = W_sorted[:4]

figA, axesA = plt.subplots(nrows=4, ncols=1, figsize=(5, 5))
figA.suptitle("Ising 1D (Paper-style)")

for i, ax in enumerate(axesA):
    # Reshape each filter into a single row
    row_img = ising_1d_filters[i].reshape(1, -1)  # shape: [1, 320]
    im = ax.imshow(
        row_img,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.15, vmax=0.15  # paper color range
    )
    ax.set_xticks([])
    ax.set_yticks([])
    label_filter(ax, i)

figA.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figureA_ising_1d.png", dpi=300)
plt.close(figA)

# ================================
#   4) FIGURE B: "Heisenberg 1D"
# ================================
# Next 4 filters (just an example)
heis_1d_filters = W_sorted[4:8]

figB, axesB = plt.subplots(nrows=4, ncols=1, figsize=(5, 5))
figB.suptitle("Heisenberg 1D (Paper-style)")

for i, ax in enumerate(axesB):
    row_img = heis_1d_filters[i].reshape(1, -1)
    im = ax.imshow(
        row_img,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.15, vmax=0.15  # same scale as the paper's 1D
    )
    ax.set_xticks([])
    ax.set_yticks([])
    label_filter(ax, i)

figB.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figureB_heisenberg_1d.png", dpi=300)
plt.close(figB)

# =================================
#   5) FIGURE C: "Heisenberg 2D"
# =================================
# Next 16 filters for a 4x4 montage
heis_2d_filters = W_sorted[8:8+16]

figC, axesC = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
figC.suptitle("Heisenberg 2D (Paper-style)")

for i in range(16):
    ax = axesC[i // 4, i % 4]
    # Reshape each filter into 16x20 (like your 2D montage)
    filt_2d = heis_2d_filters[i].reshape(16, 20)
    im = ax.imshow(
        filt_2d,
        cmap="RdBu_r",
        vmin=-0.3, vmax=0.3  # paper's 2D scale
    )
    ax.set_xticks([])
    ax.set_yticks([])
    label_filter(ax, i)

figC.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figureC_heisenberg_2d.png", dpi=300)
plt.show()

