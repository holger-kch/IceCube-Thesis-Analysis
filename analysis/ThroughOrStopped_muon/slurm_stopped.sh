#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=stopped_transformer_2M
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon/logs/%x_%j.err

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon"
DB_PATH="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/Data/muons_1305k_130000_720k_139008.db"

echo "=== Stopped/Through Transformer — 2M ==="
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "DB:        ${DB_PATH}"
echo "Date:      $(date)"
echo "============================================="

python "${SCRIPT_DIR}/train_stopped_transformer.py" \
    --db-path "${DB_PATH}" \
    --run-name "stopped_transformer_2M" \
    --epochs 50 \
    --batch-size 256 \
    --lr 5e-4 \
    --d-model 256 \
    --num-layers 6 \
    --num-heads 8 \
    --ffn-dim 512 \
    --head-hidden-dim 256 \
    --dropout 0.1 \
    --num-workers 2 \
    --early-stopping 10 \
    --max-pulses 256

echo "Done: $(date)"
