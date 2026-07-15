#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube
#SBATCH --job-name=dynedge_perm_xyt
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.err

# Usage:
#   sbatch slurm_eval_quantized.sh                 # stopped only
#   sbatch slurm_eval_quantized.sh --classes stopped through

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation"
mkdir -p "${SCRIPT_DIR}/logs"

echo "=== DynEdge event-level eval (float32-quantized dom_xyz) ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Args: $@"
echo "============================================================"

cd "${SCRIPT_DIR}"
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"
"${PYTHON}" eval_dynedge_event_perm_keep_xyt.py --cpu --num-workers 4 "$@"

echo "Done: $(date)"
