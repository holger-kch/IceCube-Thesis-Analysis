#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=data_vs_mc_hlcfb_fw
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/logs/%x_%A_%a.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new/logs/%x_%A_%a.err

set -euo pipefail

CLASSES=(stopped through)
CLASS_NAME="${CLASSES[$SLURM_ARRAY_TASK_ID]}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/Data_vs_MC_new"
PARQUET_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/data_parquet_v2"
OUT_DIR="${SCRIPT_DIR}/results/transformer_data_vs_mc_${CLASS_NAME}_hlc_rde_merged_v2_finalweight_transformer_hlcflip_best"
mkdir -p "${SCRIPT_DIR}/logs" "${OUT_DIR}"

PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"
export PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin:${PATH}"
export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH:-}"

echo "=== Data-vs-MC transformer: ${CLASS_NAME}, merged v2 + Transformer HLC best flip, HLC/RDE, final_weight ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "GPU memory:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv 2>/dev/null || true
echo "Python: ${PYTHON}"
echo "Parquet dir: ${PARQUET_DIR}"
echo "Output: ${OUT_DIR}"
echo "Date: $(date)"

"${PYTHON}" "${SCRIPT_DIR}/train_mcdata_parquet.py" \
    --class-name "${CLASS_NAME}" \
    --out-dir "${OUT_DIR}" \
    --parquet-dir "${PARQUET_DIR}" \
    --parquet-suffix v2 \
    --pulse-file-template "{source}_SplitInIcePulses_{cls}_merged_{parquet_suffix}_transformer_hlcflip_best.parquet" \
    --weight-template "GB_and_base_weights_{cls}_2M_v2.csv" \
    --weight-column final_weight \
    --epochs 20 \
    --early-stopping 5 \
    --batch-size 256 \
    --num-workers 8 \
    --prefetch-factor 4 \
    --lr 5e-4

echo "Done: $(date)"
