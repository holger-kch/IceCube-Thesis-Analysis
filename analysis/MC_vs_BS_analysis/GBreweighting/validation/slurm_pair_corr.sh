#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube
#SBATCH --job-name=pair_corr
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.err

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation"
mkdir -p "${SCRIPT_DIR}/logs"

echo "=== pair-correlation diagnostic ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "==================================="

cd "${SCRIPT_DIR}"
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"
"${PYTHON}" plot_pair_correlation_diagnostic.py

echo "Done: $(date)"
