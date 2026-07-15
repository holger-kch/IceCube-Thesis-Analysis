#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=icecube_gpu
#SBATCH --job-name=muon_recon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/logs/%x_%j.err

# Usage: sbatch slurm_train.sh <target>
#   target: direction | energy | position_z

# Ensure correct libstdc++ from cvmfs (compute nodes may lack GLIBCXX_3.4.30)
export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

TARGET="${1:?Usage: sbatch slurm_train.sh <direction|energy|position_z>}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction"

echo "=== Muon Reconstruction Training ==="
echo "Target:    ${TARGET}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "======================================"

python "${SCRIPT_DIR}/train_reconstruction.py" \
    --target "${TARGET}" \
    --epochs 30 \
    --batch-size 128 \
    --lr 1e-3 \
    --num-workers 2 \
    --early-stopping 5

echo "Done: $(date)"
