#!/bin/bash
#SBATCH --account=icecube
#SBATCH --partition=gr10_gpu
#SBATCH --job-name=replot_full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --time=02:00:00
#SBATCH --output=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.out
#SBATCH --error=/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation/logs/%x_%j.err

# Re-runs the five dynedge-dependent analysis scripts against the new
# 8-feature model (dynedge_event_full / dynedge_pulse_full). All output
# PNGs are written with a "_full" suffix so the 7-feature originals are
# preserved.
#
# Submit AFTER the two training jobs complete:
#   sbatch --dependency=afterok:JOBID_EVENT:JOBID_PULSE slurm_replot_full.sh
# (or just sbatch directly once both are done — pre-flight checks below
# verify the trained models exist before running anything.)

export LD_LIBRARY_PATH="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/spack/opt/spack/linux-almalinux9-x86_64_v2/gcc-11.3.1/gcc-13.1.0-fa6vr33ioxgsp2rkkog45hckfbaumvef/lib64:${LD_LIBRARY_PATH}"

SCRIPT_DIR="/groups/icecube/holgerkc/Thesis_Analysis/MC_vs_BS_analysis/GBreweighting/validation"
mkdir -p "${SCRIPT_DIR}/logs"
cd "${SCRIPT_DIR}"
PYTHON="/cvmfs/icecube.opensciencegrid.org/py3-v4.3.0/RHEL_9_x86_64/bin/python3.11"

echo "=== replot_full ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "==================="

# Pre-flight: confirm both 8-feature models exist
fail=0
for d in dynedge_event_full/stopped dynedge_event_full/through \
         dynedge_pulse_full/stopped dynedge_pulse_full/through; do
    if [[ ! -f "${d}/results.csv" || ! -f "${d}/state_dict.pth" ]]; then
        echo "MISSING: ${d}/{results.csv,state_dict.pth}" >&2
        fail=1
    fi
done
if [[ $fail -ne 0 ]]; then
    echo "Aborting: trained 8-feature model(s) not found." >&2
    exit 1
fi

run_step() {
    local name=$1
    shift
    echo ""
    echo "----- ${name} -----"
    if "$@"; then
        echo "OK: ${name}"
    else
        echo "FAILED: ${name}" >&2
    fi
}

# 1. eval_dynedge_event_quantized — float32 quantization on dom_xyz
run_step "eval_dynedge_event_quantized" \
    "${PYTHON}" eval_dynedge_event_quantized.py \
    --suffix _full --classes stopped through

# 2. eval_dynedge_event_perm_keep_xyt — keep (xyzt) vs keep (charge,rde,pmt_area,hlc)
run_step "eval_dynedge_event_perm_keep_xyt" \
    "${PYTHON}" eval_dynedge_event_perm_keep_xyt.py \
    --suffix _full --classes stopped through

# 3. plot_high_score_mc_vs_data — needs both event + pulse models
run_step "plot_high_score_mc_vs_data" \
    "${PYTHON}" plot_high_score_mc_vs_data.py --suffix _full

# 4. plot_pair_correlation_diagnostic
run_step "plot_pair_correlation_diagnostic" \
    "${PYTHON}" plot_pair_correlation_diagnostic.py --suffix _full

# 5. plot_qmax_position_residual
run_step "plot_qmax_position_residual" \
    "${PYTHON}" plot_qmax_position_residual.py --suffix _full

echo ""
echo "Done: $(date)"
