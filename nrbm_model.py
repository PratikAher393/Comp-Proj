#!/usr/bin/env python3
"""
rbm_model.py

This module implements the Restricted Boltzmann Machine (RBM) used as a variational ansatz 
for neural-network quantum states. It supports two modes:
  - Standard RBM: Each visible unit has its own bias and weight.
  - Shift-Invariant RBM: Enforces translational invariance; the visible bias is scalar and 
    the weight matrix is given as a set of filters that are shared across sites.
It also provides functions for:
  - Metropolis Monte Carlo sampling.
  - Local energy evaluation for the transverse-field Ising (TFI) model.
  - Generating nearest-neighbor pairs for a 1D chain with periodic boundary conditions.
  - Visualization of the learned filters.
"""

import numpy as np
import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, shift_invariant=False):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.shift_invariant = shift_invariant
        if self.shift_invariant:
            # In shift-invariant mode, visible bias is a single scalar.
            self.a = nn.Parameter(torch.zeros(1, dtype=torch.double))
            # Hidden biases remain a vector.
            self.b = nn.Parameter(torch.zeros(num_hidden, dtype=torch.double))
            # Weight filters: shape (num_hidden, num_visible)
            self.W = nn.Parameter(torch.randn(num_hidden, num_visible, dtype=torch.double) * 0.01)
        else:
            # Standard RBM: visible bias is a vector.
            self.a = nn.Parameter(torch.zeros(num_visible, dtype=torch.double))
            self.b = nn.Parameter(torch.zeros(num_hidden, dtype=torch.double))
            # Weight matrix: shape (num_visible, num_hidden)
            self.W = nn.Parameter(torch.randn(num_visible, num_hidden, dtype=torch.double) * 0.01)
    
    def forward(self, v):
        """
        Compute the logarithm of the wave function amplitude for configuration v.
        v: a torch tensor of shape (num_visible,) with values ±1.
        """
        if self.shift_invariant:
            # Visible term: same bias for all sites.
            visible_term = self.a * torch.sum(v)
            # Hidden activation: for each hidden unit f, compute b_f + sum_j W[f, j] * v[j]
            hidden_activation = self.b + torch.matmul(self.W, v)
            hidden_term = torch.sum(torch.log(2 * torch.cosh(hidden_activation)))
            return visible_term + hidden_term
        else:
            visible_term = torch.dot(self.a, v)
            hidden_activation = self.b + torch.matmul(v, self.W)
            hidden_term = torch.sum(torch.log(2 * torch.cosh(hidden_activation)))
            return visible_term + hidden_term

    def psi(self, v):
        """Return the amplitude of the wave function for configuration v."""
        return torch.exp(self.forward(v))

def metropolis_sample(rbm, v, num_samples, burn_in=1000):
    """
    Perform Metropolis Monte Carlo sampling of spin configurations.
    
    Parameters:
      rbm: An instance of RBM.
      v: Initial configuration (torch tensor).
      num_samples: Number of samples to collect after burn-in.
      burn_in: Number of initial steps to discard.
      
    Returns:
      A list of torch tensors representing sampled spin configurations.
    """
    samples = []
    current_v = v.clone()
    for step in range(num_samples + burn_in):
        i = np.random.randint(0, rbm.num_visible)
        proposed_v = current_v.clone()
        proposed_v[i] *= -1  # flip spin i
        psi_current = torch.exp(2 * rbm.forward(current_v))
        psi_proposed = torch.exp(2 * rbm.forward(proposed_v))
        acceptance = min(1, (psi_proposed / psi_current).item())
        if np.random.rand() < acceptance:
            current_v = proposed_v
        if step >= burn_in:
            samples.append(current_v.clone())
    return samples

def metropolis_sample_with_diagnostics(rbm, v, num_samples, burn_in=1000):
    """
    Modified Metropolis sampling that tracks the acceptance rate.
    
    Returns:
      samples: List of sampled configurations.
      acceptance_rate: Fraction of accepted moves.
    """
    samples = []
    current_v = v.clone()
    accepted_moves = 0
    total_moves = 0
    total_steps = num_samples + burn_in

    for step in range(total_steps):
        i = np.random.randint(0, rbm.num_visible)
        proposed_v = current_v.clone()
        proposed_v[i] *= -1
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

def local_energy_tfi(rbm, v, h, neighbor_pairs):
    """
    Compute the local energy for the transverse-field Ising (TFI) model:
       H = -h * sum_i σ^x_i - sum_{<ij>} σ^z_i σ^z_j
       
    For each spin configuration v, the off-diagonal term is evaluated by flipping one spin.
    
    Parameters:
      rbm: RBM instance.
      v: Spin configuration (torch tensor).
      h: Transverse field strength.
      neighbor_pairs: List of nearest-neighbor index pairs.
      
    Returns:
      Local energy (float) for configuration v.
    """
    # Diagonal term: interaction energy between neighbors
    E_diag = 0.0
    for (i, j) in neighbor_pairs:
        E_diag += - v[i].item() * v[j].item()
    # Off-diagonal term: transverse field term
    E_off = 0.0
    for i in range(rbm.num_visible):
        v_flip = v.clone()
        v_flip[i] *= -1
        log_ratio = rbm.forward(v_flip) - rbm.forward(v)
        ratio = torch.exp(log_ratio)
        E_off += -h * ratio.item()
    return E_diag + E_off

def generate_neighbor_pairs_1D(num_spins):
    """
    Generate nearest-neighbor pairs for a 1D chain with periodic boundary conditions.
    
    Parameters:
      num_spins: Total number of spins.
      
    Returns:
      A list of tuples (i, j) of neighboring spin indices.
    """
    return [(i, (i+1) % num_spins) for i in range(num_spins)]

def visualize_filters(rbm, filename_prefix='filter'):
    """
    Visualize the learned weight filters.
    For shift-invariant RBM, each hidden unit has a filter of length num_visible.
    For standard RBM, visualize the full weight matrix.
    
    Parameters:
      rbm: An instance of RBM.
      filename_prefix: Prefix for the output image files.
    """
    import matplotlib.pyplot as plt
    if rbm.shift_invariant:
        # Plot each hidden unit's filter as a line plot.
        num_filters = rbm.num_hidden
        for f in range(num_filters):
            plt.figure()
            plt.plot(rbm.W[f].detach().cpu().numpy())
            plt.title(f"Hidden Unit {f} Filter")
            plt.xlabel("Visible Index")
            plt.ylabel("Weight")
            plt.savefig(f"{filename_prefix}_{f}.png")
            plt.close()
    else:
        # Visualize the weight matrix as an image.
        plt.figure(figsize=(8, 6))
        plt.imshow(rbm.W.detach().cpu().numpy(), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("RBM Weight Matrix")
        plt.xlabel("Hidden Unit Index")
        plt.ylabel("Visible Unit Index")
        plt.savefig(f"{filename_prefix}_matrix.png")
        plt.close()
