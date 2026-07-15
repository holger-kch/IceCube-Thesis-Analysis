#!/bin/bash
#SBATCH --job-name=pulse_merge_data
#SBATCH --partition=icecube
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/slurm_merge_data_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/slurm_merge_data_%j.err

set -euo pipefail

PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python"
export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export OMP_NUM_THREADS=4

cd /groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
"${PYTHON}" pulse_merger.py --data-only
echo "Finished at $(date)"
