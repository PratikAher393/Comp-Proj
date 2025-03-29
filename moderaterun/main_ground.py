#!/usr/bin/env python3
"""
main_ground.py

Ground state simulation using variational Monte Carlo (VMC) with an RBM ansatz.
Usage:
    python main_ground.py [model]
where [model] is either "tfi" or "heisenberg".
"""

import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from rbm_model import RBM, metropolis_sample, local_energy_tfi, local_energy_heisenberg, generate_neighbor_pairs_1D
from observables import compute_magnetization, compute_correlation

def main():
    if len(sys.argv) < 2:
        print("Usage: python main_ground.py [tfi|heisenberg]")
        sys.exit(1)
    model = sys.argv[1].lower()
    if model not in ["tfi", "heisenberg"]:
        print("Model must be either 'tfi' or 'heisenberg'")
        sys.exit(1)
    
    # Simulation parameters
    num_spins = 20
    num_hidden = 10
    h_field = 1.0  # Only used for TFI
    num_samples = 5000
    num_iterations = 200
    learning_rate = 0.01

    rbm = RBM(num_spins, num_hidden).double()
    optimizer = optim.Adam(rbm.parameters(), lr=learning_rate)
    neighbor_pairs = generate_neighbor_pairs_1D(num_spins)
    
    np.random.seed(42)
    torch.manual_seed(42)
    v0 = torch.tensor(np.random.choice([-1, 1], size=num_spins), dtype=torch.double)
    
    energy_history = []
    magnetization_history = []
    correlation_history = []

    for it in range(num_iterations):
        samples = metropolis_sample(rbm, v0, num_samples)
        if model == "tfi":
            E_locals = [local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]
        else:
            E_locals = [local_energy_heisenberg(rbm, v, neighbor_pairs) for v in samples]
        E_mean = np.mean(E_locals)
        energy_history.append(E_mean)
        
        mag_avg, _ = compute_magnetization(samples)
        magnetization_history.append(mag_avg)
        
        correlations = compute_correlation(samples, num_spins)
        correlation_history.append(correlations)
        
        # Simplified loss estimator (for VMC optimization)
        if model == "tfi":
            loss = sum([rbm.psi(v) * local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]) / len(samples)
        else:
            loss = sum([rbm.psi(v) * local_energy_heisenberg(rbm, v, neighbor_pairs) for v in samples]) / len(samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if it % 10 == 0:
            print(f"Iteration {it:3d}: Energy = {E_mean:.6f}, Magnetization = {mag_avg:.4f}")
    
    # Save output files and plots
    np.savetxt("energy_history_ground.txt", np.array(energy_history))
    np.savetxt("magnetization_ground.txt", np.array(magnetization_history))
    np.savetxt("correlations_ground.txt", correlation_history[-1])
    
    plt.figure(figsize=(6,4))
    plt.plot(energy_history, label="Energy")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Ground State Energy Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_convergence_ground.png")
    plt.close()
    
    plt.figure(figsize=(6,4))
    plt.plot(magnetization_history, label="Magnetization", color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Magnetization")
    plt.title("Magnetization Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("magnetization_convergence_ground.png")
    plt.close()
    
    plt.figure(figsize=(6,4))
    plt.plot(correlation_history[-1], marker='o')
    plt.xlabel("Distance")
    plt.ylabel("Correlation")
    plt.title("Spin-Spin Correlation Function")
    plt.grid(True)
    plt.savefig("correlations_ground.png")
    plt.close()

if __name__ == "__main__":
    main()
