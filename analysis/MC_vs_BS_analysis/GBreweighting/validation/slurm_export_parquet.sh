#!/bin/bash
#SBATCH --job-name=export_parquet
#SBATCH --partition=icecube
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/slurm_export_parquet_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/slurm_export_parquet_%j.err

set -euo pipefail

PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python"
export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export OMP_NUM_THREADS=4

cd /groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
# Full re-export: all pulses sorted (mc + data), and MC truth re-exported.
# Skip data truth (placeholder columns only). --force overwrites everything.
# Pass 1: all 8 pulse parquets
"${PYTHON}" export_to_parquet.py --force --no-truth
# Pass 2: MC truth only (2 files), force-overwrite the existing truth files
"${PYTHON}" export_to_parquet.py --force --no-pulses --sources mc
"${PYTHON}" verify_parquet.py
echo "Finished at $(date)"
