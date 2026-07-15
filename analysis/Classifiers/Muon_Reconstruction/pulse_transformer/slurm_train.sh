#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=pulse_transformer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/pulse_transformer/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/pulse_transformer/logs/%x_%j.err

# Pulse-level transformer: each pulse = one token, no DOM grouping.
# Usage: sbatch slurm_train.sh <direction|energy|position_z>

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

TARGET="${1:?Usage: sbatch slurm_train.sh <direction|energy|position_z>}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/pulse_transformer"
DB_PATH="/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"

echo "=== Pulse Transformer — ${TARGET} ==="
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "============================================="

python "${SCRIPT_DIR}/train_pulse_transformer.py" \
    --db-path "${DB_PATH}" \
    --target "${TARGET}" \
    --run-name "pulse_transformer_720k_${TARGET}" \
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
    --early-stopping 8 \
    --max-pulses 256

echo "Done: $(date)"
