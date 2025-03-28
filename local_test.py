#!/usr/bin/env python3

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from nrbm_model import RBM, metropolis_sample_with_diagnostics, local_energy_tfi, generate_neighbor_pairs_1D, visualize_filters
from nobservables import compute_magnetization

def main():
    # Test parameters for a quick local run:
    num_spins = 10
    num_hidden = 5
    h_field = 1.0
    num_samples = 100    # Reduced sample count
    burn_in = 100        # Reduced burn-in steps
    num_iterations = 5   # Fewer iterations for a quick test
    learning_rate = 0.002

    # Instantiate RBM in shift-invariant mode
    rbm = RBM(num_spins, num_hidden, shift_invariant=True).double()
    optimizer = optim.Adam(rbm.parameters(), lr=learning_rate)
    neighbor_pairs = generate_neighbor_pairs_1D(num_spins)
    
    np.random.seed(42)
    torch.manual_seed(42)
    v0 = torch.tensor(np.random.choice([-1, 1], size=num_spins), dtype=torch.double)
    
    energy_history = []
    magnetization_history = []

    for it in range(num_iterations):
        samples, acc_rate = metropolis_sample_with_diagnostics(rbm, v0, num_samples, burn_in=burn_in)
        E_locals = [local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]
        E_mean = np.mean(E_locals)
        energy_history.append(E_mean)
        
        mag_avg, _ = compute_magnetization(samples)
        magnetization_history.append(mag_avg)
        
        loss = sum([rbm.psi(v) * local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]) / len(samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Iteration {it:2d}: Energy = {E_mean:.4f}, Magnetization = {mag_avg:.4f}, Acceptance Rate = {acc_rate:.2f}")

    # Save energy and magnetization histories to text files
    np.savetxt("energy_history_test.txt", np.array(energy_history))
    np.savetxt("magnetization_history_test.txt", np.array(magnetization_history))
    
    # Plot energy convergence
    plt.figure(figsize=(6,4))
    plt.plot(energy_history, marker='o', label="Energy")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Test Run: Energy Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_convergence_test.png")
    plt.close()

    # Plot magnetization convergence
    plt.figure(figsize=(6,4))
    plt.plot(magnetization_history, marker='o', label="Magnetization", color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Magnetization")
    plt.title("Test Run: Magnetization Convergence")
    plt.legend()
    plt.grid(True)
    plt.savefig("magnetization_convergence_test.png")
    plt.close()

    # Visualize the learned filters: montage and heatmap
    visualize_filters(rbm, filename_prefix='test_filter', montage=True)

    # Plot hidden biases as a bar chart
    hidden_biases = rbm.b.detach().cpu().numpy()
    plt.figure()
    plt.bar(range(len(hidden_biases)), hidden_biases)
    plt.title("Hidden Biases")
    plt.xlabel("Hidden Unit Index")
    plt.ylabel("Bias Value")
    plt.savefig("hidden_biases.png")
    plt.close()

    # Print visible bias (for shift-invariant RBM, there's only one scalar)
    if rbm.shift_invariant:
        print(f"Visible bias a: {rbm.a.item()}")

    # Plot a histogram of all filter weights
    filters = rbm.W.detach().cpu().numpy().flatten()
    plt.figure()
    plt.hist(filters, bins=20, edgecolor='black')
    plt.title("Distribution of Filter Weights")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.savefig("filter_histogram.png")
    plt.close()

if __name__ == "__main__":
    main()

