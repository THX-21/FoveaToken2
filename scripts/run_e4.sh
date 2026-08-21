#!/usr/bin/env bash
set -euo pipefail

# Set, for example, to "0,2,3"; leave empty for automatic detection.
VISIBLE_GPUS=""
if [[ -n "$VISIBLE_GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_count=$(nvidia-smi --list-gpus | wc -l)
  if (( gpu_count == 0 )); then
    echo "No visible CUDA GPUs found" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((gpu_count - 1)))"
else
  IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  gpu_count=${#visible_gpus[@]}
fi

if (( gpu_count == 0 )); then
  echo "CUDA_VISIBLE_DEVICES does not contain any GPU" >&2
  exit 1
fi

config=experiments/e4/configs/mcp.yaml
conditions=(
  full
  lowres8
  dynamic8_all_heads_native
  dynamic8_top8_native
)

for condition in "${conditions[@]}"; do
  torchrun --standalone --nproc-per-node="$gpu_count" \
    -m experiments.e4 run \
    --config "$config" \
    --model qwen35 \
    --suite formal \
    --condition "$condition"
done
