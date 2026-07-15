#!/bin/bash
#SBATCH --job-name=vmf_pole_collapse
#SBATCH --partition=icecube
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/slurm_vmf_pole_collapse_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/slurm_vmf_pole_collapse_%j.err

set -euo pipefail

export TMPDIR=/tmp
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python"
export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export OMP_NUM_THREADS=4

cd /groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
"${PYTHON}" plot_vmf_pole_collapse_evidence.py
echo "Finished at $(date)"
