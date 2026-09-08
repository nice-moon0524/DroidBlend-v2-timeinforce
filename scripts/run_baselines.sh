#!/usr/bin/env bash
set -euo pipefail
export DROIDBLEND_OUTPUT_DIR="${DROIDBLEND_OUTPUT_DIR:-results/hybrid_baselines}"
"$(dirname "$0")/run_all.sh" --mode baselines --output-dir "$DROIDBLEND_OUTPUT_DIR" "$@"

