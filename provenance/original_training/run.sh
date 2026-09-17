#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_name=${1:-pd_3motions_v2_65536}
iterations=${2:-200}
num_envs=${3:-65536}
checkpoint=${4:-}
motion_manifest=${5:-}
max_hours=${6:-0}
if [[ -e "runs/$run_name/state.txt" ]]; then
  echo "Run already exists; choose a new run name to preserve its evidence." >&2
  exit 4
fi
mkdir -p "runs/$run_name"
exec 9>"runs/$run_name/process.lock"
flock -n 9 || exit 3
printf '%s\n' "$$" > "runs/$run_name/launcher.pid"
printf '%s running\n' "$(date -Is)" > "runs/$run_name/state.txt"
monitor_pid=''
cleanup() {
  code=$?
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  printf '%s exit=%s\n' "$(date -Is)" "$code" > "runs/$run_name/state.txt"
}
trap cleanup EXIT
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 5 > "runs/$run_name/gpu.csv" &
monitor_pid=$!
extra=()
if [[ -n "$checkpoint" ]]; then extra+=(--checkpoint "$checkpoint"); fi
manifest_extra=()
eval_envs=3
if [[ -n "$motion_manifest" ]]; then
  cp "$motion_manifest" "runs/$run_name/frozen_manifest.json"
  motion_manifest="runs/$run_name/frozen_manifest.json"
  manifest_extra+=(--motion_manifest "$motion_manifest")
  eval_envs=$(/root/isaac-sonic-venv/bin/python -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m["complete"]; print(len(m["accepted"]))' "$motion_manifest")
fi
mkdir -p "runs/$run_name/source"
cp training/train.py training/human39_env.py training/run.sh "runs/$run_name/source/"
/root/isaac-sonic-venv/bin/python training/train.py --headless --device cuda:0 --num_envs "$num_envs" --iterations "$iterations" --run "$run_name" --max_hours "$max_hours" "${extra[@]}" "${manifest_extra[@]}"
latest_checkpoint=$(find "runs/$run_name" -maxdepth 1 -name 'model_*.pt' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
printf '%s evaluating\n' "$(date -Is)" > "runs/$run_name/state.txt"
/root/isaac-sonic-venv/bin/python training/train.py --headless --device cuda:0 --num_envs "$eval_envs" --run "${run_name}_eval" --checkpoint "$latest_checkpoint" --eval_steps 400 "${manifest_extra[@]}"
