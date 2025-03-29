#!/bin/bash
#SBATCH --job-name=nqs_mode
#SBATCH --output=nqs_mpi_moderate_%j.out
#SBATCH --error=nqs_mpi_moderate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56
#SBATCH --partition=condo-phys573
#SBATCH --qos=condo
#SBATCH --account=ISAAC-UTK0350
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=paher@vols.utk.edu

echo "Job started on $(date)"
module load python/3.9.10
module load mpi4py

source /nfs/home/paher/venv/bin/activate


# Run the MPI-enabled simulation using 56 tasks (one node)
srun -n 56 python main_mpi.py

echo "Job ended on $(date)"

