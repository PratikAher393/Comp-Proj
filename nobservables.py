#!/usr/bin/env python3
"""
observables.py

This module provides functions to compute observables from Monte Carlo samples:
  - compute_magnetization: Computes the average magnetization and standard deviation.
  - compute_correlation: Computes the two-point spin-spin correlation function.
"""

import numpy as np

def compute_magnetization(samples):
    """
    Calculate the average magnetization and its standard deviation.
    
    Parameters:
      samples: List of torch tensors representing spin configurations.
      
    Returns:
      (mean_magnetization, std_magnetization)
    """
    mags = [v.float().mean().item() for v in samples]
    return np.mean(mags), np.std(mags)

def compute_correlation(samples, num_spins):
    """
    Compute the two-point spin-spin correlation function.
    
    For each distance d, computes the average correlation between spins separated by d.
    
    Parameters:
      samples: List of torch tensors representing spin configurations.
      num_spins: Number of spins per configuration.
      
    Returns:
      A numpy array of correlation values for distances 0 to num_spins-1.
    """
    correlations = np.zeros(num_spins)
    for v in samples:
        v_np = v.numpy()
        for d in range(num_spins):
            correlations[d] += np.mean(v_np * np.roll(v_np, d))
    correlations /= len(samples)
    return correlations
