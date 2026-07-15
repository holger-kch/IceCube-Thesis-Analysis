#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube
#SBATCH --job-name=afterpulse_reset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=6:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.err

set -euo pipefail

# Same cvmfs python 3.11 used on the login node (has matplotlib + pandas +
# pyarrow; system latex at /usr/bin is used for usetex rendering).
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python"
export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLBACKEND=Agg

cd /groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Cores: ${SLURM_CPUS_PER_TASK}"
"${PYTHON}" -u plot_afterpulse_reset_dom_time.py
echo "Finished at $(date)"
