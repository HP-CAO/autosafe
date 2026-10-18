##!/usr/bin/env bash
#set -euo pipefail
#
#SEEDS=(1)
#MAX_STEPS=50000
#LOG_ROOT="outputs"
#ENV_NAME='cartpole'
#run() {
#  local job=$1; shift
#  local exp=$1; shift
#  local seed=$1; shift
#  local extra=("$@")
#  local out="${LOG_ROOT}/${ENV_NAME}/${job}/seed_${seed}"
#  mkdir -p "$out"
#  echo "[RUN] ${job} (seed=${seed})"
#  python job/cartpole.py \
#    JobParams.job_name="$job" \
#    JobParams.experiment_name="$exp" \
#    JobParams.seed="$seed" \
#    SACParams.total_training_steps="$MAX_STEPS" \
#    "${extra[@]}" \
#    >"$out/stdout.log" 2>"$out/stderr.log" &
#}
#
#for SEED in "${SEEDS[@]}"; do
#  # launch all variants in parallel for this SEED
#  run sac_base_wall_time              sac_base        "$SEED"
#  run sac_cbf_wall_time               sac_cbf        "$SEED"
#  run sac_autosafe_opt_wall_time      sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=opt
#  run sac_residual_wall_time          sac_residual    "$SEED"
#  run sac_simplex_wall_time           sac_simplex     "$SEED"
#  run sac_lyapunov_wall_time          sac_lyapunov    "$SEED"
#  run sac_lag_wall_time               sac_lag         "$SEED"
#  run sac_lam_opt_wall_time           sac_lam         "$SEED"  SACParams.sac_lam_mode=opt
#  wait
#  echo "[DONE] All experiments for seed=${SEED}"
#done
#
#echo "All seeds done."
#
#
#
##!/usr/bin/env bash
#set -euo pipefail
#SEEDS=(1)
#MAX_STEPS=50000
#LOG_ROOT="outputs"
#ENV_NAME='glucose'
#run() {
#  local job=$1; shift
#  local exp=$1; shift
#  local seed=$1; shift
#  local extra=("$@")
#  local out="${LOG_ROOT}/${ENV_NAME}/${job}/seed_${seed}"
#  mkdir -p "$out"
#  echo "[RUN] ${job} (seed=${seed})"
#  python job/glucose.py \
#    JobParams.job_name="$job" \
#    JobParams.experiment_name="$exp" \
#    JobParams.seed="$seed" \
#    SACParams.total_training_steps="$MAX_STEPS" \
#    "${extra[@]}" \
#    >"$out/stdout.log" 2>"$out/stderr.log" &
#}
#
#for SEED in "${SEEDS[@]}"; do
#  run sac_base_wall_time              sac_base        "$SEED"
#  run sac_cbf_wall_time               sac_cbf        "$SEED"
#  run sac_autosafe_opt_wall_time      sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=opt
#  run sac_residual_wall_time          sac_residual    "$SEED"
#  run sac_simplex_wall_time           sac_simplex     "$SEED"
#  run sac_lyapunov_wall_time          sac_lyapunov    "$SEED"
#  run sac_lag_wall_time               sac_lag         "$SEED"
#  run sac_lam_opt_wall_time           sac_lam         "$SEED"  SACParams.sac_lam_mode=opt
#  wait
#  echo "[DONE] All experiments for seed=${SEED}"
#done
#
#echo "All seeds done."
##
##
##
##!/usr/bin/env bash
#set -euo pipefail
#SEEDS=(1)
#MAX_STEPS=50000
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
#  run sac_base_wall_time              sac_base        "$SEED"
#  run sac_cbf_wall_time               sac_cbf         "$SEED"
#  run sac_autosafe_opt_wall_time      sac_autosafe    "$SEED"  SACParams.autosafe_lam_mode=opt
#  run sac_residual_wall_time          sac_residual    "$SEED"
#  run sac_simplex_wall_time           sac_simplex     "$SEED"
#  run sac_lyapunov_wall_time          sac_lyapunov    "$SEED"
#  run sac_lag_wall_time               sac_lag         "$SEED"
#  run sac_lam_opt_wall_time           sac_lam         "$SEED"  SACParams.sac_lam_mode=opt
#  wait
#  echo "[DONE] All experiments for seed=${SEED}"
#done
#
#echo "All seeds done."


#
#
#!/usr/bin/env bash
set -euo pipefail
SEEDS=(1)
MAX_STEPS=50000
LOG_ROOT="outputs"
ENV_NAME='quadruped'
run() {
  local job=$1; shift
  local exp=$1; shift
  local seed=$1; shift
  local extra=("$@")
  local out="${LOG_ROOT}/${ENV_NAME}/${job}/seed_${seed}"
  mkdir -p "$out"
  echo "[RUN] ${job} (seed=${seed})"
  /home/ippc/anaconda3/envs/runtime/bin/python  job/quadruped.py \
    edge.JobParams.job_name="$job" \
    edge.JobParams.experiment_name="$exp" \
    edge.JobParams.seed="$seed" \
    edge.SACParams.total_training_steps="$MAX_STEPS" \
    "${extra[@]}" \
    >"$out/stdout.log" 2>"$out/stderr.log" &
}

for SEED in "${SEEDS[@]}"; do
#  run sac_base_wall_time              sac_base        "$SEED"
#  run sac_cbf_wall_time               sac_cbf         "$SEED"
#  run sac_autosafe_opt_wall_time      sac_autosafe    "$SEED"  edge.SACParams.autosafe_lam_mode=opt
#  run sac_residual_wall_time          sac_residual    "$SEED"
#  run sac_simplex_wall_time           sac_simplex     "$SEED"
#  run sac_lyapunov_wall_time          sac_lyapunov    "$SEED"
#  run sac_lag_wall_time               sac_lag         "$SEED"
  run sac_lam_opt_wall_time           sac_lam         "$SEED"  edge.SACParams.sac_lam_mode=opt
  wait
  echo "[DONE] All experiments for seed=${SEED}"
done

echo "All seeds done."
##
##
#
##!/usr/bin/env bash
#set -euo pipefail
#
## ---- config ----
#SEEDS=(1)                     # no commas in bash arrays
#MAX_STEPS=50000
#LOG_ROOT="outputs"
#ENV_NAME="quadruped"
#
## Each entry: "<job_name> <experiment_name> [extra args...]"
#JOBS=(
#   "sac_base_wall_time              sac_base"
#   "sac_cbf_wall_time               sac_cbf "
#   "sac_autosafe_opt_wall_time      sac_autosafe     edge.SACParams.autosafe_lam_mode=opt"
#   "sac_residual_wall_time          sac_residual"
#   "sac_simplex_wall_time           sac_simplex"
#   "sac_lyapunov_wall_time          sac_lyapunov"
#   "sac_lag_wall_time               sac_lag "
#   "sac_lam_opt_wall_time           sac_lam         edge.SACParams.sac_lam_mode=opt"
#)
#
#run() {
#  local job=$1; shift
#  local exp=$1; shift
#  local seed=$1; shift
#  local extra=("$@")
#  local out="${LOG_ROOT}/${ENV_NAME}/${job}/seed_${seed}"
#  mkdir -p "$out"
#  echo "[RUN] ${job} (seed=${seed})"
#  /home/ippc/anaconda3/envs/runtime/bin/python job/quadruped.py \
#    edge.JobParams.job_name="$job" \
#    edge.JobParams.experiment_name="$exp" \
#    edge.JobParams.seed="$seed" \
#    edge.SACParams.total_training_steps="$MAX_STEPS" \
#    "${extra[@]}" \
#    >"$out/stdout.log" 2>"$out/stderr.log" &
#}
#
## ---- launch: parallel across seeds per job ----
#for spec in "${JOBS[@]}"; do
#  # split spec into an array: first=job, second=exp, rest=extra args
#  read -r -a parts <<< "$spec"
#  job="${parts[0]}"
#  exp="${parts[1]}"
#  extras=("${parts[@]:2}")
#
#  echo "[TASK] Launching '${job}' for seeds: ${SEEDS[*]}"
#  for SEED in "${SEEDS[@]}"; do
#    run "$job" "$exp" "$SEED" "${extras[@]}"
#  done
#
#  # wait until all seeds for this job finish before moving to next job
#  wait
#  echo "[DONE] All seeds finished for job='${job}'"
#done
#
#echo "All jobs across all seeds done."
