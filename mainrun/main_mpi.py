#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import torch
import torch.optim as optim
from nrbm_model import RBM, metropolis_sample, local_energy_tfi, generate_neighbor_pairs_1D, visualize_filters
from nobservables import compute_magnetization
import matplotlib.pyplot as plt
from datetime import datetime
import sys, os

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Simulation parameters for a moderate run on one node
    num_spins = 80
    num_hidden = 320
    h_field = 1.0
    
    num_samples = 5000     # Increased sample count per iteration
    burn_in = 10000        # Increased burn-in steps
    num_iterations = 500   # Total iterations set to 500
    learning_rate = 0.001  # Lower learning rate

    # Instantiate RBM in shift-invariant mode
    rbm = RBM(num_spins, num_hidden, shift_invariant=True).double()
    optimizer = optim.Adam(rbm.parameters(), lr=learning_rate)
    neighbor_pairs = generate_neighbor_pairs_1D(num_spins)
    
    # Use different seeds per MPI rank for independent sampling
    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    v0 = torch.tensor(np.random.choice([-1, 1], size=num_spins), dtype=torch.double)
    
    energy_history = []
    magnetization_history = []
    
    checkpoint_file = "checkpoint.pth"
    start_iteration = 0
    # Rank 0 checks for a checkpoint and loads it if present
    if rank == 0 and os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        start_iteration = checkpoint["iteration"] + 1
        rbm.load_state_dict(checkpoint["rbm_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        energy_history = checkpoint["energy_history"]
        magnetization_history = checkpoint["magnetization_history"]
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        sys.stdout.flush()
    # Broadcast the starting iteration to all ranks
    start_iteration = comm.bcast(start_iteration, root=0)

    # Run iterations from start_iteration to num_iterations
    for it in range(start_iteration, num_iterations):
        samples = metropolis_sample(rbm, v0, num_samples, burn_in=burn_in)
        E_locals = [local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]
        E_mean = np.mean(E_locals)
        energy_history.append(E_mean)
        
        mag_avg, _ = compute_magnetization(samples)
        magnetization_history.append(mag_avg)
        
        loss = sum([rbm.psi(v) * local_energy_tfi(rbm, v, h_field, neighbor_pairs) for v in samples]) / len(samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Every 10 iterations, print progress with time stamp and flush output
        if it % 10 == 0:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Rank {rank:2d}] Iteration {it:3d} at {now}: Energy = {E_mean:.4f}, Magnetization = {mag_avg:.4f}")
            sys.stdout.flush()

        # Save checkpoint every 50 iterations
        if it % 50 == 0:
            if rank == 0:
                checkpoint = {
                    "iteration": it,
                    "rbm_state_dict": rbm.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "energy_history": energy_history,
                    "magnetization_history": magnetization_history
                }
                torch.save(checkpoint, checkpoint_file)
                print(f"Saved checkpoint at iteration {it}")
                sys.stdout.flush()
    
    # Gather results from all MPI processes (should be 56 tasks on one node)
    energies_all = comm.gather(energy_history, root=0)
    mags_all = comm.gather(magnetization_history, root=0)

    if rank == 0:
        avg_energy = np.mean(np.array(energies_all), axis=0)
        avg_mag = np.mean(np.array(mags_all), axis=0)
        np.savetxt("energy_history_parallel.txt", avg_energy)
        np.savetxt("magnetization_parallel.txt", avg_mag)
        print("Parallel moderate run completed on one node. Outputs saved.")

        # Plot energy convergence
        plt.figure(figsize=(6,4))
        plt.plot(avg_energy, marker='o', label="Energy")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Energy Convergence (MPI)")
        plt.legend()
        plt.grid(True)
        plt.savefig("mpi_energy_convergence.png")
        plt.close()

        # Plot magnetization convergence
        plt.figure(figsize=(6,4))
        plt.plot(avg_mag, marker='o', label="Magnetization", color='red')
        plt.xlabel("Iteration")
        plt.ylabel("Magnetization")
        plt.title("Magnetization Convergence (MPI)")
        plt.legend()
        plt.grid(True)
        plt.savefig("mpi_magnetization_convergence.png")
        plt.close()

        # Visualize learned filters: montage and heatmap
        visualize_filters(rbm, filename_prefix='mpi_filter', montage=True)
        
        # Plot hidden biases as a bar chart
        hidden_biases = rbm.b.detach().cpu().numpy()
        plt.figure()
        plt.bar(range(len(hidden_biases)), hidden_biases)
        plt.title("Hidden Biases")
        plt.xlabel("Hidden Unit Index")
        plt.ylabel("Bias Value")
        plt.savefig("mpi_hidden_biases.png")
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
        plt.savefig("mpi_filter_histogram.png")
        plt.close()

if __name__ == "__main__":
    main()

