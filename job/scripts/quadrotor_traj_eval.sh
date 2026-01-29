#!/usr/bin/env bash
set -euo pipefail

SEEDS=(1)
MAX_STEPS=200000
LOG_ROOT="outputs"
ENV_NAME='quadrotor'
run() {
  local job=$1; shift
  local exp=$1; shift
  local seed=$1; shift
  local extra=("$@")
  local out="${LOG_ROOT}/${ENV_NAME}/${job}/seed_${seed}"
  mkdir -p "$out"
  echo "[RUN] ${job} (seed=${seed})"
  python job/quadrotor.py \
    JobParams.job_name="$job" \
    JobParams.experiment_name="$exp" \
    JobParams.seed="$seed" \
    SACParams.total_training_steps="$MAX_STEPS" \
    "${extra[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log" &
}

for SEED in "${SEEDS[@]}"; do
   run autosafe_quadrotor_eval_before    autosafe_traj_eval    "$SEED"
   run autosafe_quadrotor_eval_after    autosafe_traj_eval    "$SEED" SACParams.model_path="results/data_source/quadrotor/sac_autosafe_opt/seed_1/20250911-192945/agent_model"
   run simplex_quadrotor_eval_before    simplex_traj_eval    "$SEED"
   run simplex_quadrotor_traj_eval_after     simplex_traj_eval    "$SEED" SACParams.model_path="results/data_source/quadrotor/sac_simplex/seed_1/20250912-123707/agent_model"
  wait
  echo "[DONE] All experiments for seed=${SEED}"
done

echo "All seeds done."
