# Code Documentation for Neural-Network Quantum States Simulation

## Overview
This package implements a variational Monte Carlo (VMC) simulation of quantum many-body systems using Restricted Boltzmann Machines (RBMs) as neural-network quantum states. Two models are considered:
- **Transverse-Field Ising (TFI) Model**
- **Antiferromagnetic Heisenberg Model**

A simplified time-dependent (unitary) evolution simulation is also included.

## File Descriptions

- **rbm_model.py**  
  Contains the RBM class and functions:
  - `RBM`: Defines the neural network.
  - `metropolis_sample`: Generates spin configurations using the Metropolis algorithm.
  - `local_energy_tfi`: Computes the local energy for the TFI Hamiltonian.
  - `local_energy_heisenberg`: Computes the local energy for the Heisenberg Hamiltonian (using a simplified off-diagonal approximation).
  - `generate_neighbor_pairs_1D`: Utility function for 1D periodic boundary conditions.

- **observables.py**  
  Provides functions to compute:
  - `compute_magnetization`: Average magnetization and its standard deviation.
  - `compute_correlation`: Two-point spin-spin correlation function.

- **main_ground.py**  
  Main driver for ground state simulations. Accepts a command-line argument (`tfi` or `heisenberg`) to select the model. It performs VMC optimization and outputs energy, magnetization, and correlation data along with plots.

- **main_unitary.py**  
  A simplified driver for time-dependent unitary evolution simulation using a basic Euler integration scheme. It outputs energy and magnetization as a function of time.

- **main_test.py**  
  A minimal serial test-run file for verifying the code functionality with reduced parameters.

- **run_nqs_mpi.slurm**  
  SLURM job submission script to run the MPI-enabled simulation on ISAAC HPC.

## Running the Code

### Local Test
Run the minimal test script:
```bash
python main_test.py
