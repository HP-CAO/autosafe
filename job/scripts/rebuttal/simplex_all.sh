#!/usr/bin/env bash
set -euo pipefail

SEEDS=(1 2 3 4 5)
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
   run sac_simplex_delta_0       sac_simplex    "$SEED"  SACParams.delta_min=0.0
   run sac_simplex_delta_25      sac_simplex    "$SEED"  SACParams.delta_min=0.25
   run sac_simplex_delta_50      sac_simplex    "$SEED"  SACParams.delta_min=0.5
   run sac_simplex_delta_75      sac_simplex    "$SEED"  SACParams.delta_min=0.75
   run sac_simplex_delta_100     sac_simplex    "$SEED"  SACParams.delta_min=1.0
  wait
  echo "[DONE] All experiments for seed=${SEED}"
done

echo "All seeds done."


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
   run sac_simplex_delta_0       sac_simplex    "$SEED"  SACParams.delta_min=0.0
   run sac_simplex_delta_25      sac_simplex    "$SEED"  SACParams.delta_min=0.25
   run sac_simplex_delta_50      sac_simplex    "$SEED"  SACParams.delta_min=0.5
   run sac_simplex_delta_75      sac_simplex    "$SEED"  SACParams.delta_min=0.75
   run sac_simplex_delta_100     sac_simplex    "$SEED"  SACParams.delta_min=1.0
  wait
  echo "[DONE] All experiments for seed=${SEED}"
done

echo "All seeds done."

#!/usr/bin/env bash
set -euo pipefail

SEEDS=(1 2 3 4 5)
MAX_STEPS=1000000
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
   run sac_simplex_delta_0       sac_simplex    "$SEED"  SACParams.delta_min=0.0
   run sac_simplex_delta_25      sac_simplex    "$SEED"  SACParams.delta_min=0.25
   run sac_simplex_delta_50      sac_simplex    "$SEED"  SACParams.delta_min=0.5
   run sac_simplex_delta_75      sac_simplex    "$SEED"  SACParams.delta_min=0.75
   run sac_simplex_delta_100     sac_simplex    "$SEED"  SACParams.delta_min=1.0
   wait
  echo "[DONE] All experiments for seed=${SEED}"
done

echo "All seeds done."

