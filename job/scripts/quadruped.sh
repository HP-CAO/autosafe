##!/usr/bin/env bash
#set -euo pipefail
#
#SEEDS=(1,2,3,4,5)
#MAX_STEPS=1000000
#LOG_ROOT="outputs"
#ENV_NAME='quadrotor'
#run() {
#  local job=$1; shift
#  local exp=$1; shift
#  local seed=$1; shift
#  local extra=("$@")
#  local out="${LOG_ROOT}/${ENV_NAME}/${job}/seed_${seed}"
#  mkdir -p "$out"
#  echo "[RUN] ${job} (seed=${seed})"
#  python job/quadrotor.py \
#    JobParams.job_name="$job" \
#    JobParams.experiment_name="$exp" \
#    JobParams.seed="$seed" \
#    SACParams.total_training_steps="$MAX_STEPS" \
#    "${extra[@]}" \
#    >"$out/stdout.log" 2>"$out/stderr.log" &
#}
#
#for SEED in "${SEEDS[@]}"; do
##   launch all variants in parallel for this SEED
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

#
#  wait
#  echo "[DONE] All experiments for seed=${SEED}"
#done
#
#echo "All seeds done."


#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
SEEDS=(1 2 3 4 5)                     # no commas in bash arrays
MAX_STEPS=500000
LOG_ROOT="outputs"
ENV_NAME="quadruped"

# Each entry: "<job_name> <experiment_name> [extra args...]"
JOBS=(
#  "model_based         model_based"
#  "sac_base            sac_base"
#  "sac_autosafe_opt    sac_autosafe    edge.SACParams.autosafe_lam_mode=opt"
#  "sac_autosafe_linear sac_autosafe    edge.SACParams.autosafe_lam_mode=linear"
#  "sac_autosafe_exp    sac_autosafe    edge.SACParams.autosafe_lam_mode=exp"
#  "sac_residual        sac_residual"
#  "sac_simplex         sac_simplex"
#  "sac_lyapunov        sac_lyapunov"
#  "sac_lag             sac_lag"
#  "sac_lam_opt         sac_lam         edge.SACParams.lam_mode=opt"
#  "sac_lam_linear      sac_lam         edge.SACParams.lam_mode=linear"
#  "sac_lam_exp         sac_lam         edge.SACParams.lam_mode=exp"
    "sac_cbf            sac_cbf"
)

run() {
  local job=$1; shift
  local exp=$1; shift
  local seed=$1; shift
  local extra=("$@")
  local out="${LOG_ROOT}/${ENV_NAME}/${job}/seed_${seed}"
  mkdir -p "$out"
  echo "[RUN] ${job} (seed=${seed})"
  python job/quadruped.py \
    edge.JobParams.job_name="$job" \
    edge.JobParams.experiment_name="$exp" \
    edge.JobParams.seed="$seed" \
    edge.SACParams.total_training_steps="$MAX_STEPS" \
    "${extra[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log" &
}

# ---- launch: parallel across seeds per job ----
for spec in "${JOBS[@]}"; do
  # split spec into an array: first=job, second=exp, rest=extra args
  read -r -a parts <<< "$spec"
  job="${parts[0]}"
  exp="${parts[1]}"
  extras=("${parts[@]:2}")

  echo "[TASK] Launching '${job}' for seeds: ${SEEDS[*]}"
  for SEED in "${SEEDS[@]}"; do
    run "$job" "$exp" "$SEED" "${extras[@]}"
  done

  # wait until all seeds for this job finish before moving to next job
  wait
  echo "[DONE] All seeds finished for job='${job}'"
done

echo "All jobs across all seeds done."

