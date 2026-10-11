#!/usr/bin/env bash
set -euo pipefail
TASK_BASE=/mnt/vggt-data/vggt-slam2
TASK_B="$TASK_BASE/experiments"
TASK_EXTRA=()
case "${1:-normal}" in
 normal) TASK_OUTPUT="$TASK_B/vggt-v50-hybrid-normal-r1" ;;
 profile) TASK_OUTPUT="$TASK_B/vggt-v50-hybrid-profile-r1"; TASK_EXTRA=(--collect-perf-detail) ;;
 *) exit 2 ;;
esac
cd "$TASK_BASE"
. src/deploy/rk3588/env.sh
exec venv/bin/python -B "$TASK_B/vggt-v50-hybrid-runtime-source/run-v50-hybrid-pair-r2-board.py" \
 --parent "$TASK_B/vggt-v49-runtime-source/run-v49-original-pair-precision-board.py" \
 --contracts "$TASK_B/vggt-v49-runtime-source/v49_quantized_pair_runtime_contracts.py" \
 --compiler-source "$TASK_B/vggt-v50-flash-source/compile-v50-flash-pair-board.py" \
 --compile-dir "$TASK_B/vggt-v50-flash-compile-r1" \
 --compile-sha256 d19b34f90ee1c7d41724fd36fb4ba808a880863f57ce58f053cc292d7e705d34 \
 --hybrid-compile-dir "$TASK_B/vggt-v50-hybrid-step2-r2" \
 --hybrid-compile-sha256 60e046c89442a5e0fc2cb5b793cf7184d48ec38810a2e46ba9215ae549bfd50c \
 --hybrid-model-sha256 a502ceb2ae936550d56efe94c4fa4ff385b6d4530a0eaa62ac7ba0d57830b16d \
 --hybrid-contracts "$TASK_B/vggt-v50-hybrid-source/v50_hybrid_step2_r2_contracts.py" \
 --hybrid-source "$TASK_B/vggt-v50-hybrid-source/compile-v50-hybrid-step2-r2-board.py" \
 --capture-source "$TASK_B/vggt-v49-capture-source/v49_capture_contracts.py" \
 --capture-dir "$TASK_B/vggt-v49-capture-r1" \
 --runtime-helper "$TASK_BASE/src/deploy/rk3588/validation/run-v12-rknn-block-board.py" \
 --sdk-header "$TASK_BASE/src/deploy/rk3588/reference/rknn_api_v2.3.2.h" \
 --output-dir "$TASK_OUTPUT" "${TASK_EXTRA[@]}"
