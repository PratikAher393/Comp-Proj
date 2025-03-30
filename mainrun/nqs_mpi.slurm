#!/bin/bash
#SBATCH --job-name=nqs
#SBATCH --output=nqs_mpi_moderate_%j.out
#SBATCH --error=nqs_mpi_moderate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --partition=condo-phys573
#SBATCH --qos=condo
#SBATCH --account=ISAAC-UTK0350
#SBATCH --time=40:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=paher@vols.utk.edu

echo "Job started on $(date)"

source /nfs/home/paher/venv/bin/activate


# Run the MPI-enabled simulation using 56 tasks (one node)
srun -n 56 python3 main_mpi.py

echo "Job ended on $(date)"

