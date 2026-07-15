#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=dynedge720k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/dynedge_720k/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/dynedge_720k/logs/%x_%j.err

# DynEdge on full 720k MuonGun sample
# Same split as direction transformer (500k/110k/110k)
# Usage: sbatch slurm_train.sh <energy|position_z>

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

TARGET="${1:?Usage: sbatch slurm_train.sh <energy|position_z>}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/dynedge_720k"

echo "=== DynEdge 720k — ${TARGET} ==="
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "============================================="

python "${SCRIPT_DIR}/train_dynedge_720k.py" \
    --target "${TARGET}" \
    --epochs 50 \
    --batch-size 128 \
    --lr 1e-3 \
    --early-stopping 8

echo "Done: $(date)"
