#!/bin/bash
#SBATCH --job-name=afterpulse_simple
#SBATCH --partition=icecube
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=55G
#SBATCH --time=01:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/slurm_afterpulse_simple_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/slurm_afterpulse_simple_%j.err

set -euo pipefail

PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python"
export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export OMP_NUM_THREADS=8

cd /groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
"${PYTHON}" plot_afterpulse_simple.py
echo "Finished at $(date)"
