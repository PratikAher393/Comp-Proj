#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import torch
from nrbm_model import RBM  # using your RBM implementation from nrbm_model.py

# ---------------------------
# Load the checkpoint
# ---------------------------
# If you need to allow safe globals, you can uncomment the following lines:
# with torch.serialization.safe_globals(["numpy._core.multiarray.scalar"]):
#     checkpoint = torch.load("checkpoint.pth", weights_only=True)
checkpoint = torch.load("checkpoint.pth", weights_only=True)
print("Checkpoint keys:", checkpoint.keys())

# ---------------------------
# Set model configuration matching the checkpoint
# The checkpoint indicates:
#   - visible bias shape: [1]     -> shift_invariant mode (single bias)
#   - hidden bias shape: [320] and W shape: [320, 80]
# So, we use:
num_visible = 80    # number of visible units
num_hidden = 320    # number of hidden units
shift_invariant = True

# ---------------------------
# Instantiate and load the model
# ---------------------------
model = RBM(num_visible, num_hidden, shift_invariant=shift_invariant)
model.load_state_dict(checkpoint["rbm_state_dict"])
model.eval()
print("Model loaded from checkpoint with matching configuration.")
print("Visible biases:", model.a)
print("Hidden biases:", model.b)

# ---------------------------
# Load simulation data
# ---------------------------
energy_history = np.loadtxt("energy_history_ground.txt")
magnetization_history = np.loadtxt("magnetization_ground.txt")

# ---------------------------
# Plot Energy Convergence
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(energy_history, label="Energy")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Ground State Energy Convergence")
plt.legend()
plt.grid(True)
plt.savefig("energy_convergence_ground.png")
print("Saved energy_convergence_ground.png")
plt.close()

# ---------------------------
# Plot Magnetization Convergence
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(magnetization_history, label="Magnetization", color="red")
plt.xlabel("Iteration")
plt.ylabel("Magnetization")
plt.title("Magnetization Convergence")
plt.legend()
plt.grid(True)
plt.savefig("magnetization_convergence_ground.png")
print("Saved magnetization_convergence_ground.png")
plt.close()

