#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=muon_infer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/logs/%x_%j.err

# Ensure correct libstdc++ from cvmfs
export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

TARGET="${1:?Usage: sbatch slurm_inference.sh <target> <ckpt_path>}"
CKPT="${2:?Usage: sbatch slurm_inference.sh <target> <ckpt_path>}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction"

echo "=== Muon Reconstruction Inference ==="
echo "Target:    ${TARGET}"
echo "Ckpt:      ${CKPT}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "======================================="

python "${SCRIPT_DIR}/run_inference.py" \
    --target "${TARGET}" \
    --ckpt "${CKPT}" \
    --batch-size 256 \
    --num-workers 2

echo "Done: $(date)"
