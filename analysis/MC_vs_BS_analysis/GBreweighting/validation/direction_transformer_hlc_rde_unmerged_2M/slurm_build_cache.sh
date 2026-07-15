#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube
#SBATCH --nodelist=node187
#SBATCH --job-name=build_dir_cache_hlc_rde
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=1-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/direction_transformer_hlc_rde_unmerged_2M"
mkdir -p "${SCRIPT_DIR}/logs"
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"
export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Build unmerged 2M HLC/RDE direction cache ==="
echo "Node: $(hostname)"
echo "Python: ${PYTHON}"
echo "Date: $(date)"

"${PYTHON}" "${SCRIPT_DIR}/build_cache.py"

echo "Done: $(date)"
