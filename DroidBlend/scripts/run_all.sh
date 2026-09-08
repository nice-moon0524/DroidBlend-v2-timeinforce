#!/usr/bin/env bash
set -euo pipefail

: "${DROIDBLEND_SENDER:=/opt/hhy/models/Mistral-7B-v0.1}"
: "${DROIDBLEND_RECEIVER:=/opt/hhy/models/mistrallite}"
: "${DROIDBLEND_DEVICE:=cuda}"
: "${DROIDBLEND_DATA_DIR:=data/processed}"
: "${DROIDBLEND_PROFILE:=../DroidSpeak-new/profiling_results.json}"
: "${DROIDBLEND_MAX_SAMPLES:=0}"
: "${DROIDBLEND_MAX_NEW_TOKENS:=64}"
: "${DROIDBLEND_IMPORTANT_FRACTION:=0.2}"
: "${DROIDBLEND_TOP_K:=0}"

common_args=(
  --sender "$DROIDBLEND_SENDER"
  --receiver "$DROIDBLEND_RECEIVER"
  --profiling-results "$DROIDBLEND_PROFILE"
  --data-dir "$DROIDBLEND_DATA_DIR"
  --device "$DROIDBLEND_DEVICE"
  --max-new-tokens "$DROIDBLEND_MAX_NEW_TOKENS"
  --important-fraction "$DROIDBLEND_IMPORTANT_FRACTION"
)

if [[ "$DROIDBLEND_MAX_SAMPLES" != "0" ]]; then
  common_args+=(--max-samples "$DROIDBLEND_MAX_SAMPLES")
fi
if [[ "$DROIDBLEND_TOP_K" != "0" ]]; then
  common_args+=(--top-k "$DROIDBLEND_TOP_K")
fi

python -m experiments.run_droidblend_experiment "${common_args[@]}" "$@"
python -m scripts.plot_results \
  --droidblend results/droidblend_quality_latency.json \
  --token-selection results/droidblend_token_selection.json \
  --output-dir results/figures
