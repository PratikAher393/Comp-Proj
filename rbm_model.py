#!/usr/bin/env python3
"""
rbm_model.py

This module implements the RBM used as a variational ansatz for neural-network quantum states.
It provides:
  - RBM: The neural network model.
  - metropolis_sample: Metropolis Monte Carlo sampling.
  - local_energy_tfi: Local energy evaluation for the transverse-field Ising (TFI) model.
  - local_energy_heisenberg: Local energy evaluation for the antiferromagnetic Heisenberg model.
  - generate_neighbor_pairs_1D: Utility to generate nearest-neighbor pairs for a 1D chain (with periodic boundary conditions).
"""

import numpy as np
import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        # Variational parameters: visible bias (a), hidden bias (b), and weights (W)
        self.a = nn.Parameter(torch.zeros(num_visible, dtype=torch.double))
        self.b = nn.Parameter(torch.zeros(num_hidden, dtype=torch.double))
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden, dtype=torch.double) * 0.01)
        
    def forward(self, v):
        """
        Compute the logarithm of the wave function amplitude for configuration v.
        v: a torch tensor of shape (num_visible,) with values ±1.
        """
        linear_term = torch.dot(self.a, v)
        activation = self.b + torch.matmul(v, self.W)
        hidden_term = torch.sum(torch.log(2 * torch.cosh(activation)))
        return linear_term + hidden_term

    def psi(self, v):
        """Return the amplitude of the wave function for configuration v."""
        return torch.exp(self.forward(v))

def metropolis_sample(rbm, v, num_samples, burn_in=100):
    """
    Perform Metropolis Monte Carlo sampling of spin configurations.
    
    Parameters:
      rbm: An instance of RBM.
      v: Initial configuration (torch tensor).
      num_samples: Number of samples to collect (after burn-in).
      burn_in: Number of initial samples to discard.
      
    Returns:
      A list of torch tensors representing sampled spin configurations.
    """
    samples = []
    current_v = v.clone()
    for step in range(num_samples + burn_in):
        i = np.random.randint(0, rbm.num_visible)
        proposed_v = current_v.clone()
        proposed_v[i] *= -1  # Flip the selected spin
        psi_current = torch.exp(2 * rbm.forward(current_v))
        psi_proposed = torch.exp(2 * rbm.forward(proposed_v))
        acceptance = min(1, (psi_proposed / psi_current).item())
        if np.random.rand() < acceptance:
            current_v = proposed_v
        if step >= burn_in:
            samples.append(current_v.clone())
    return samples

def local_energy_tfi(rbm, v, h, neighbor_pairs):
    """
    Compute the local energy for the transverse-field Ising (TFI) model:
       H = -h * sum_i σ^x_i - sum_{<ij>} σ^z_i σ^z_j
       
    The off-diagonal σ^x term is treated by flipping a single spin.
    
    Parameters:
      rbm: RBM instance.
      v: Spin configuration (torch tensor).
      h: Transverse field strength.
      neighbor_pairs: List of nearest-neighbor pairs.
      
    Returns:
      The local energy (float) for configuration v.
    """
    # Diagonal term: interaction energy (using σ^z = ±1)
    E_diag = 0.0
    for (i, j) in neighbor_pairs:
        E_diag += - v[i].item() * v[j].item()
    # Off-diagonal term: transverse field (σ^x) term
    E_off = 0.0
    for i in range(rbm.num_visible):
        v_flip = v.clone()
        v_flip[i] *= -1
        log_ratio = rbm.forward(v_flip) - rbm.forward(v)
        ratio = torch.exp(log_ratio)
        E_off += -h * ratio.item()
    return E_diag + E_off

def local_energy_heisenberg(rbm, v, neighbor_pairs):
    """
    Compute the local energy for the antiferromagnetic Heisenberg model:
       H = sum_{<ij>} (σ^x_i σ^x_j + σ^y_i σ^y_j + σ^z_i σ^z_j)
       
    In the S^z basis, the σ^z part is diagonal.
    For simplicity, we approximate the off-diagonal contributions by considering a simultaneous flip of a spin pair.
    
    Parameters:
      rbm: RBM instance.
      v: Spin configuration (torch tensor).
      neighbor_pairs: List of nearest-neighbor pairs.
      
    Returns:
      The local energy (float) for configuration v.
    """
    # Diagonal term: only the σ^zσ^z part
    E_diag = 0.0
    for (i, j) in neighbor_pairs:
        E_diag += v[i].item() * v[j].item()
    
    # Off-diagonal term (simplified): for each neighbor pair, flip both spins and compute ratio
    E_off = 0.0
    for (i, j) in neighbor_pairs:
        v_flip = v.clone()
        v_flip[i] *= -1
        v_flip[j] *= -1
        log_ratio = rbm.forward(v_flip) - rbm.forward(v)
        ratio = torch.exp(log_ratio)
        # The Heisenberg off-diagonal prefactor is taken as 0.5 (a simplified factor)
        E_off += 0.5 * ratio.item()
    return E_diag + E_off

def generate_neighbor_pairs_1D(num_spins):
    """
    Generate a list of nearest-neighbor pairs for a 1D chain with periodic boundary conditions.
    
    Parameters:
      num_spins: Total number of spins.
      
    Returns:
      A list of tuples (i, j) of neighboring spin indices.
    """
    return [(i, (i+1) % num_spins) for i in range(num_spins)]
