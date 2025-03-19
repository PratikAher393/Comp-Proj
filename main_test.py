#!/usr/bin/env python3
"""
main_test.py

A modified serial test-run script with higher sampling, longer burn-in, more iterations,
and a smaller learning rate to achieve smoother convergence.
"""

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from rbm_model import RBM, metropolis_sample, local_energy_tfi, generate_neighbor_pairs_1D
from observables import compute_magnetization

def metropolis_sample_with_diagnostics(rbm, v, num_samples, burn_in=1000):
    """
    Metropolis sampling that tracks acceptance rate.
    Increased burn-in and sample size to reduce fluctuations.
    """
    samples = []
    current_v = v.clone()
    accepted_moves = 0
    total_moves = 0
    total_steps = num_samples + burn_in
    
    for step in range(total_steps):
        i = np.random.randint(0, rbm.num_visible)
        proposed_v = current_v.clone()
        proposed_v[i] *= -1  # flip spin
        psi_current = torch.exp(2 * rbm.forward(current_v))
        psi_proposed = torch.exp(2 * rbm.forward(proposed_v))
        acceptance = min(1, (psi_proposed / psi_current).item())
        total_moves += 1
        if np.random.rand() < acceptance:
            current_v = proposed_v
            accepted_moves += 1
        if step >= burn_in:
            samples.append(current_v.clone())
    
    acceptance_rate = accepted_moves / total_moves
    return samples, acceptance_rate

def main():
    # Adjusted parameters for improved stability
    num_spins = 10
    num_hidden = 5
    h_field = 1.0
    
    # Increased sampling
    num_samples = 1000      
    burn_in = 1000          
    
    # More VMC iterations
    num_iterations = 50     
    
    # Smaller learning rate for smoother updates
    learning_rate = 0.002   

    rbm = RBM(num_spins, num_hidden).double()
    optimizer = optim.Adam(rbm.parameters(), lr=learning_rate)
    neighbor_pairs = generate_neighbor_pairs_1D(num_spins)
    
    np.random.seed(42)
    torch.manual_seed(42)
    v0 = torch.tensor(np.random.choice([-1, 1], size=num_spins), dtype=torch.double)
    
    energy_history = []
    magnetization_history = []

    for it in range(num_iterations):
        samples, acc_rate = metropolis_sample_with_diagnostics(rbm, v0, num_samples, burn_in=burn_in)
        
        # Compute local energies
        E_locals = [local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]
        E_mean = np.mean(E_locals)
        energy_history.append(E_mean)
        
        # Compute magnetization
        mag_avg, _ = compute_magnetization(samples)
        magnetization_history.append(mag_avg)
        
        # Simplified VMC loss
        loss = sum([rbm.psi(v) * local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]) / len(samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print iteration info
        print(f"Iteration {it:2d}: Energy = {E_mean:.4f}, Magnetization = {mag_avg:.4f}, Acceptance Rate = {acc_rate:.2f}")

    # Save results
    np.savetxt("energy_history_test.txt", np.array(energy_history))
    np.savetxt("magnetization_history_test.txt", np.array(magnetization_history))
    
    # Plot Energy Convergence
    plt.figure(figsize=(6,4))
    plt.plot(energy_history, marker='o', label="Energy")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Test Run: Energy Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_convergence_test.png")
    plt.close()

    # Plot Magnetization Convergence
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
