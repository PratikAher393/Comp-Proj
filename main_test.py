#!/usr/bin/env python3
"""
main_test.py

A minimal test-run script (serial version) to verify the code functionality.
It runs a short simulation using the TFI model.
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from rbm_model import RBM, metropolis_sample, local_energy_tfi, generate_neighbor_pairs_1D
from observables import compute_magnetization

def main():
    # Test parameters: reduced for a quick run
    num_spins = 10
    num_hidden = 5
    h_field = 1.0
    num_samples = 100
    num_iterations = 10
    learning_rate = 0.01

    rbm = RBM(num_spins, num_hidden).double()
    optimizer = optim.Adam(rbm.parameters(), lr=learning_rate)
    neighbor_pairs = generate_neighbor_pairs_1D(num_spins)
    
    np.random.seed(42)
    torch.manual_seed(42)
    v0 = torch.tensor(np.random.choice([-1, 1], size=num_spins), dtype=torch.double)
    
    energy_history = []
    magnetization_history = []

    for it in range(num_iterations):
        samples = metropolis_sample(rbm, v0, num_samples)
        E_locals = [local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]
        E_mean = np.mean(E_locals)
        energy_history.append(E_mean)
        
        mag_avg, _ = compute_magnetization(samples)
        magnetization_history.append(mag_avg)
        
        loss = sum([rbm.psi(v) * local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]) / len(samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Iteration {it:2d}: Energy = {E_mean:.4f}, Magnetization = {mag_avg:.4f}")

    np.savetxt("energy_history_test.txt", np.array(energy_history))
    np.savetxt("magnetization_history_test.txt", np.array(magnetization_history))
    
    plt.figure(figsize=(6,4))
    plt.plot(energy_history, marker='o', label="Energy")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Test Run: Energy Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_convergence_test.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(magnetization_history, marker='o', label="Magnetization", color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Magnetization")
    plt.title("Test Run: Magnetization Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("magnetization_convergence_test.png")
    plt.close()

if __name__ == "__main__":
    main()
