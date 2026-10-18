#!/usr/bin/env bash
set -euo pipefail

SEEDS=(1 2 3)
MAX_STEPS=200000
LOG_ROOT="outputs"
ENV_NAME='cartpole'
run() {
  local job=$1; shift
  local exp=$1; shift
  local seed=$1; shift
  local extra=("$@")
  local out="${LOG_ROOT}/${ENV_NAME}/${job}/seed_${seed}"
  mkdir -p "$out"
  echo "[RUN] ${job} (seed=${seed})"
  python job/cartpole.py \
    JobParams.job_name="$job" \
    JobParams.experiment_name="$exp" \
    JobParams.seed="$seed" \
    SACParams.total_training_steps="$MAX_STEPS" \
    "${extra[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log" &
}

for SEED in "${SEEDS[@]}"; do
  # launch all variants in parallel for this SEED
#  run model_based           model_based     "$SEED"
#  run sac_base              sac_base        "$SEED"
#   run sac_autosafe_opt_delta_0       sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=opt SACParams.delta_min=0.0
#   run sac_autosafe_opt_delta_25      sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=opt SACParams.delta_min=0.25
#   run sac_autosafe_opt_delta_50      sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=opt SACParams.delta_min=0.5
#   run sac_autosafe_opt_delta_75      sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=opt SACParams.delta_min=0.75
#   run sac_autosafe_opt_delta_100     sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=opt SACParams.delta_min=1.0
#   run sac_cbf               sac_cbf        "$SEED"
#  run sac_autosafe_linear   sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=linear
#  run sac_autosafe_exp      sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=exp
#  run sac_residual          sac_residual    "$SEED"
#  run sac_simplex           sac_simplex     "$SEED"
#  run sac_lyapunov          sac_lyapunov    "$SEED"
#  run sac_lag               sac_lag         "$SEED"
#  run sac_lam_opt           sac_lam         "$SEED"  SACParams.sac_lam_mode=opt
#  run sac_lam_linear        sac_lam         "$SEED"  SACParams.sac_lam_mode=linear
#  run sac_lam_exp           sac_lam         "$SEED"  SACParams.sac_lam_mode=exp
#   run sac_simplex_delta_0       sac_simplex    "$SEED"  SACParams.delta_min=0.0
#   run sac_simplex_delta_25      sac_simplex    "$SEED"  SACParams.delta_min=0.25
#   run sac_simplex_delta_50      sac_simplex    "$SEED"  SACParams.delta_min=0.5
#   run sac_simplex_delta_75      sac_simplex    "$SEED"  SACParams.delta_min=0.75
#   run sac_simplex_delta_100     sac_simplex    "$SEED"  SACParams.delta_min=1.0
   run sac_autosafe_vary sac_autosafe_vary_cartpole "$SEED"
  wait
  echo "[DONE] All experiments for seed=${SEED}"
done

echo "All seeds done."
