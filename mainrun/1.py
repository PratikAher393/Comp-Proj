# %% [code]
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch

# %% [code]
# Load energy and magnetization data from text files
energy_history = np.loadtxt("energy_history_parallel.txt")
magnetization_history = np.loadtxt("magnetization_parallel.txt")

# Plot the energy convergence over iterations
plt.figure(figsize=(8, 6))
plt.plot(energy_history, label="Energy")
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Ground State Energy Convergence")
plt.legend()
plt.grid(True)
plt.savefig("energy_convergence_ground.png")
plt.show()

# Plot the magnetization convergence over iterations
plt.figure(figsize=(8, 6))
plt.plot(magnetization_history, label="Magnetization", color="red")
plt.xlabel("Iteration")
plt.ylabel("Magnetization")
plt.title("Magnetization Convergence")
plt.legend()
plt.grid(True)
plt.show()

# %% [code]
# Load a checkpoint from a saved model (checkpoint.pth)
# This assumes your checkpoint was saved as a dictionary containing at least a key 'model_state_dict'.
checkpoint = torch.load("checkpoint.pth", weights_only=False)
print("Checkpoint keys:", checkpoint.keys())

# For example, if your model is an RBM, you can restore it as follows:
from nrbm_model import RBM

# Use the same configuration as the checkpoint
num_visible = 80    # visible spins
num_hidden = 320    # hidden units
shift_invariant = True  # since a has shape [1] in the checkpoint

model = RBM(num_visible, num_hidden, shift_invariant=shift_invariant)
model.load_state_dict(checkpoint["rbm_state_dict"])
model.eval()
print("Model loaded from checkpoint with matching configuration.")


# %% [code]
# Optionally, inspect some model parameters or use the model for further analysis
print("Visible biases:", model.a)
print("Hidden biases:", model.b)

