#!/usr/bin/env python3
"""
main_unitary.py

Time-dependent (unitary) evolution using a simplified time-dependent VMC approach.
This script simulates the evolution of the RBM wave function over time for a given quantum model.
Usage:
    python main_unitary.py [tfi|heisenberg]
"""

import sys
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from rbm_model import RBM, metropolis_sample, local_energy_tfi, local_energy_heisenberg, generate_neighbor_pairs_1D
from observables import compute_magnetization

def main():
    if len(sys.argv) < 2:
        print("Usage: python main_unitary.py [tfi|heisenberg]")
        sys.exit(1)
    model = sys.argv[1].lower()
    if model not in ["tfi", "heisenberg"]:
        print("Model must be either 'tfi' or 'heisenberg'")
        sys.exit(1)
        
    # Simulation parameters
    num_spins = 20
    num_hidden = 10
    h_field = 1.0  # Used for TFI; for Heisenberg, this is not used.
    num_samples = 2000
    num_time_steps = 100
    dt = 0.01  # time step
    learning_rate = 0.005

    rbm = RBM(num_spins, num_hidden).double()
    optimizer = optim.Adam(rbm.parameters(), lr=learning_rate)
    neighbor_pairs = generate_neighbor_pairs_1D(num_spins)
    
    np.random.seed(42)
    torch.manual_seed(42)
    v0 = torch.tensor(np.random.choice([-1, 1], size=num_spins), dtype=torch.double)
    
    energy_history = []
    magnetization_history = []
    times = []

    # Time evolution loop (simplified Euler integration)
    for t in range(num_time_steps):
        samples = metropolis_sample(rbm, v0, num_samples)
        if model == "tfi":
            E_locals = [local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]
        else:
            E_locals = [local_energy_heisenberg(rbm, v, neighbor_pairs) for v in samples]
        E_mean = np.mean(E_locals)
        energy_history.append(E_mean)
        mag_avg, _ = compute_magnetization(samples)
        magnetization_history.append(mag_avg)
        times.append(t * dt)
        
        # Simplified time derivative update (this is only demonstrative)
        if model == "tfi":
            loss = sum([rbm.psi(v) * local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]) / len(samples)
        else:
            loss = sum([rbm.psi(v) * local_energy_heisenberg(rbm, v, neighbor_pairs) for v in samples]) / len(samples)
        optimizer.zero_grad()
        loss.backward()
        # Update parameters with a time step dt (this is a simplified integration)
        for param in rbm.parameters():
            param.data -= dt * param.grad.data
        optimizer.step()
        
        if t % 10 == 0:
            print(f"Time {t*dt:.3f}: Energy = {E_mean:.6f}, Magnetization = {mag_avg:.4f}")
    
    # Save output files and plots
    np.savetxt("energy_history_unitary.txt", np.array(energy_history))
    np.savetxt("magnetization_unitary.txt", np.array(magnetization_history))
    np.savetxt("times.txt", np.array(times))
    
    plt.figure(figsize=(6,4))
    plt.plot(times, energy_history, label="Energy")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.title("Unitary Evolution: Energy vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_vs_time.png")
    plt.close()
    
    plt.figure(figsize=(6,4))
    plt.plot(times, magnetization_history, label="Magnetization", color='red')
    plt.xlabel("Time")
    plt.ylabel("Magnetization")
    plt.title("Unitary Evolution: Magnetization vs. Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("magnetization_vs_time.png")
    plt.close()

if __name__ == "__main__":
    main()
