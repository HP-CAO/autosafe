#!/usr/bin/env bash
set -euo pipefail

SEEDS=(1 2 3 4 5)
MAX_STEPS=100000
LOG_ROOT="outputs"
ENV_NAME='glucose'
run() {
  local job=$1; shift
  local exp=$1; shift
  local seed=$1; shift
  local extra=("$@")
  local out="${LOG_ROOT}/${ENV_NAME}/${job}/seed_${seed}"
  mkdir -p "$out"
  echo "[RUN] ${job} (seed=${seed})"
  python job/glucose.py \
    JobParams.job_name="$job" \
    JobParams.experiment_name="$exp" \
    JobParams.seed="$seed" \
    SACParams.total_training_steps="$MAX_STEPS" \
    "${extra[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log" &
}

for SEED in "${SEEDS[@]}"; do
#   launch all variants in parallel for this SEED
#  run model_based           model_based     "$SEED"
#  run sac_base              sac_base        "$SEED"
#  run sac_autosafe_opt      sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=opt
#  run sac_autosafe_linear   sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=linear
#  run sac_autosafe_exp      sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=exp
#  run sac_residual          sac_residual    "$SEED"
#  run sac_simplex           sac_simplex     "$SEED"
#  run sac_lyapunov          sac_lyapunov    "$SEED"
#  run sac_lag               sac_lag         "$SEED"
#  run sac_lam_opt           sac_lam         "$SEED"  SACParams.sac_lam_mode=opt
#  run sac_lam_linear        sac_lam         "$SEED"  SACParams.sac_lam_mode=linear
#  run sac_lam_exp           sac_lam         "$SEED"  SACParams.sac_lam_mode=exp
   run sac_cbf               sac_cbf        "$SEED"

  wait
  echo "[DONE] All experiments for seed=${SEED}"
done

echo "All seeds done."
