#!/usr/bin/env bash
set -euo pipefail

: "${DROIDBLEND_SENDER:=/opt/hhy/models/Mistral-7B-v0.1}"
: "${DROIDBLEND_RECEIVER:=/opt/hhy/models/mistrallite}"
: "${DROIDBLEND_DEVICE:=cuda}"
: "${DROIDBLEND_SENDER_DEVICE:=}"
: "${DROIDBLEND_RECEIVER_DEVICE:=}"
: "${DROIDBLEND_DTYPE:=auto}"
: "${DROIDBLEND_DATA_DIR:=data/processed}"
: "${DROIDBLEND_PROFILE:=../DroidSpeak-new/profiling_results.json}"
: "${DROIDBLEND_MAX_SAMPLES:=0}"
: "${DROIDBLEND_MAX_NEW_TOKENS:=20}"
: "${DROIDBLEND_MAX_PROMPT_TOKENS:=0}"
: "${DROIDBLEND_OUTPUT_DIR:=results/hybrid}"
: "${DROIDBLEND_TOKEN_RATIO:=0.15}"
: "${DROIDBLEND_RECOMPUTE_SPANS:=pareto}"
: "${DROIDBLEND_TOKEN_RATIOS:=0.10,0.20,0.30,0.40,0.50}"
: "${DROIDBLEND_MAX_F1_DROP:=0.0}"
: "${DROIDBLEND_LOCAL_FILES_ONLY:=0}"
: "${DROIDBLEND_TRUST_REMOTE_CODE:=0}"

common_args=(
  --sender "$DROIDBLEND_SENDER"
  --receiver "$DROIDBLEND_RECEIVER"
  --profiling-results "$DROIDBLEND_PROFILE"
  --data-dir "$DROIDBLEND_DATA_DIR"
  --output-dir "$DROIDBLEND_OUTPUT_DIR"
  --device "$DROIDBLEND_DEVICE"
  --dtype "$DROIDBLEND_DTYPE"
  --max-new-tokens "$DROIDBLEND_MAX_NEW_TOKENS"
  --token-recompute-ratio "$DROIDBLEND_TOKEN_RATIO"
  --recompute-spans "$DROIDBLEND_RECOMPUTE_SPANS"
  --token-ratios "$DROIDBLEND_TOKEN_RATIOS"
)

if [[ -n "$DROIDBLEND_SENDER_DEVICE" ]]; then
  common_args+=(--sender-device "$DROIDBLEND_SENDER_DEVICE")
fi
if [[ -n "$DROIDBLEND_RECEIVER_DEVICE" ]]; then
  common_args+=(--receiver-device "$DROIDBLEND_RECEIVER_DEVICE")
fi
if [[ "$DROIDBLEND_MAX_SAMPLES" != "0" ]]; then
  common_args+=(--max-samples "$DROIDBLEND_MAX_SAMPLES")
fi
if [[ "$DROIDBLEND_MAX_PROMPT_TOKENS" != "0" ]]; then
  common_args+=(--max-prompt-tokens "$DROIDBLEND_MAX_PROMPT_TOKENS")
fi
if [[ "$DROIDBLEND_LOCAL_FILES_ONLY" == "1" ]]; then
  common_args+=(--local-files-only)
fi
if [[ "$DROIDBLEND_TRUST_REMOTE_CODE" == "1" ]]; then
  common_args+=(--trust-remote-code)
fi

python -m experiments.run_hybrid_experiment "${common_args[@]}" "$@"
python -m scripts.select_hybrid_config \
  --summary "$DROIDBLEND_OUTPUT_DIR/summary.json" \
  --output "$DROIDBLEND_OUTPUT_DIR/selected_config.json" \
  --max-f1-drop "$DROIDBLEND_MAX_F1_DROP"
python -m scripts.plot_hybrid_results \
  --summary "$DROIDBLEND_OUTPUT_DIR/summary.json" \
  --output-dir "$DROIDBLEND_OUTPUT_DIR/figures"


