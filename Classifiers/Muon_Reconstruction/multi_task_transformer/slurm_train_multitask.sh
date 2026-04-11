#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=transformer_multitask
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/multi_task_transformer/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/multi_task_transformer/logs/%x_%j.err

# Multi-Task Transformer: direction + energy + position_z jointly
# Usage: sbatch slurm_train_multitask.sh

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/Classifiers/Muon_Reconstruction/multi_task_transformer"
DB_PATH="/groups/icecube/janikh/PREP/Transformer_Muon_Track_Reco/data/muons_139008.db"

echo "=== Multi-Task Transformer (direction + energy + position_z) ==="
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date:      $(date)"
echo "================================================================="

python "${SCRIPT_DIR}/multi_training_transformer.py" \
    --db-path "${DB_PATH}" \
    --run-name "transformer_multitask_720k_v1" \
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
    --uncertainty-weighting

echo "Done: $(date)"
