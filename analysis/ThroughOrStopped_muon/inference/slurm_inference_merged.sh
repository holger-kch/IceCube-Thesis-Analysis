#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=stopped_inf_merged
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon/inference/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/ThroughOrStopped_muon/inference/logs/%x_%j.err

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

ROOT="/groups/icecube/holgerkc/Thesis_Analysis"
INF_DIR="${ROOT}/ThroughOrStopped_muon/inference"
CHECKPOINT="${ROOT}/ThroughOrStopped_muon/results/stopped_transformer_2M/best_model.pt"

MC_DB="${ROOT}/MC_vs_BS_analysis/MC/muons_1305k_130000_720k_139008_with_SplitInIcePulses_merged_0.3PE.db"
DATA_DB="${ROOT}/MC_vs_BS_analysis/Data/data_IC86.21_withrates_with_SplitInIcePulses_merged_0.3PE.db"
PID_CSV="${ROOT}/MC_vs_BS_analysis/Data/data_IC86_2021_with_subrun_PID_888826.csv"
PULSEMAP="SplitInIcePulses_merged"

echo "=== Stopped/Through Inference (merged pulses) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Pulsemap: ${PULSEMAP}"
echo "Date: $(date)"
echo "================================="

echo
echo ">>> MC inference"
python "${INF_DIR}/run_inference.py" \
    --mode mc \
    --db-path "${MC_DB}" \
    --checkpoint "${CHECKPOINT}" \
    --pulsemap "${PULSEMAP}" \
    --output "${INF_DIR}/output/stopped_recon_mc_muons_1305k_130000_720k_139008.csv" \
    --batch-size 512 \
    --num-workers 4

echo
echo ">>> Data inference (pid_muon_logit_data > 5)"
python "${INF_DIR}/run_inference.py" \
    --mode data \
    --db-path "${DATA_DB}" \
    --csv-path "${PID_CSV}" \
    --logit-threshold 5.0 \
    --checkpoint "${CHECKPOINT}" \
    --pulsemap "${PULSEMAP}" \
    --output "${INF_DIR}/output/stopped_recon_data_IC86_2021_pid_muon_logit_data_gt5.csv" \
    --batch-size 512 \
    --num-workers 4

echo
echo "Done: $(date)"
