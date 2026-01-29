#!/usr/bin/env bash
set -euo pipefail

SEEDS=(1)
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
   run autosafe_cartpole_eval_before    autosafe_traj_eval    "$SEED" GymParams.TaskParams.control_goal_x=0.2
   run autosafe_cartpole_eval_after    autosafe_traj_eval    "$SEED" SACParams.model_path="results/data_source/cartpole/sac_autosafe_opt_0.2/seed_1/20251126-163315/agent_model" GymParams.TaskParams.control_goal_x=0.2
   run simplex_cartpole_eval_before    simplex_traj_eval    "$SEED"  GymParams.TaskParams.control_goal_x=0.2
   run simplex_cartpole_traj_eval_after     simplex_traj_eval    "$SEED"  GymParams.TaskParams.control_goal_x=0.2 SACParams.model_path="results/data_source/cartpole/sac_simplex_0.2/seed_1/20251128-103909/agent_model"
  wait
  echo "[DONE] All experiments for seed=${SEED}"
done

echo "All seeds done."
