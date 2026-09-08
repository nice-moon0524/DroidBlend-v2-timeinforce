#!/usr/bin/env bash
set -euo pipefail

: "${DROIDBLEND_BATCH_TOKEN_RATIOS:=0.10,0.20}"
: "${DROIDBLEND_BATCH_OUTPUT_ROOT:=results/hybrid_by_ratio}"
: "${DROIDBLEND_BATCH_RESUME:=1}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DROIDBLEND_BATCH_OUTPUT_ROOT"

IFS=',' read -r -a ratios <<< "$DROIDBLEND_BATCH_TOKEN_RATIOS"
for raw_ratio in "${ratios[@]}"; do
  ratio="$(echo "$raw_ratio" | xargs)"
  if [[ -z "$ratio" ]]; then
    continue
  fi
  ratio_label="${ratio//./p}"
  ratio_output_dir="$DROIDBLEND_BATCH_OUTPUT_ROOT/ratio_${ratio_label}"

  if [[ "$DROIDBLEND_BATCH_RESUME" == "1" && -f "$ratio_output_dir/summary.json" ]]; then
    echo "[DroidBlend] skip ratio=$ratio because $ratio_output_dir/summary.json already exists"
    continue
  fi

  echo "[DroidBlend] running token ratio=$ratio -> $ratio_output_dir"
  DROIDBLEND_TOKEN_RATIO="$ratio" \
  DROIDBLEND_TOKEN_RATIOS="$ratio" \
  DROIDBLEND_OUTPUT_DIR="$ratio_output_dir" \
    bash "$script_dir/run_all.sh" "$@"
  echo "[DroidBlend] finished token ratio=$ratio; saved under $ratio_output_dir"
done