#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=transformer_scalar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon/scripts/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon/scripts/logs/%x_%j.err

# Usage: sbatch slurm_train_scalar.sh <energy|position_z>
# Bigger model config for scalar regression targets.

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

TARGET="${1:?Usage: sbatch slurm_train_scalar.sh <energy|position_z>}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Inars_zenith_azimuth_transformer_recon/scripts"
DB_PATH="/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"

echo "=== Transformer Scalar Reconstruction (bigger model) ==="
echo "Target:    ${TARGET}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "==========================================================="

python "${SCRIPT_DIR}/train_transformer.py" \
    --db-path "${DB_PATH}" \
    --target "${TARGET}" \
    --run-name "transformer_720k_${TARGET}_v2" \
    --d-model 256 \
    --num-layers 6 \
    --num-heads 8 \
    --ffn-dim 512 \
    --head-hidden-dim 512 \
    --dropout 0.05 \
    --epochs 50 \
    --batch-size 256 \
    --lr 5e-4 \
    --num-workers 2 \
    --early-stopping 8

echo "Done: $(date)"
