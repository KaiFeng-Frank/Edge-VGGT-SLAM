#!/usr/bin/env bash
set -euo pipefail
TASK_BASE=/mnt/vggt-data/vggt-slam2
TASK_B="$TASK_BASE/experiments"
TASK_RUNTIME="$TASK_B/vggt-v49-runtime-source"
TASK_QUANT="$TASK_B/vggt-v49-quant-source"
TASK_MODE="${1:-normal}"
TASK_EXTRA=()
case "$TASK_MODE" in
  normal) TASK_OUTPUT="$TASK_B/vggt-v49-precision-normal-r1" ;;
  profile) TASK_OUTPUT="$TASK_B/vggt-v49-precision-profile-r1"; TASK_EXTRA=(--collect-perf-detail) ;;
  *) exit 2 ;;
esac
cd "$TASK_BASE"
. src/deploy/rk3588/env.sh
exec "$TASK_BASE/venv/bin/python" -B "$TASK_RUNTIME/run-v49-original-pair-precision-board.py" \
  --contracts "$TASK_RUNTIME/v49_quantized_pair_runtime_contracts.py" \
  --contracts-sha256 44c0a6a1dcca66ee09b2f15bb0e2f0ddc7b7eb3f6cb44fdc3b576e9f75463bc2 \
  --capture-contracts "$TASK_B/vggt-v49-capture-source/v49_capture_contracts.py" \
  --capture-contracts-sha256 6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8 \
  --capture-dir "$TASK_B/vggt-v49-capture-r1" \
  --capture-sha256 bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc \
  --ptq-compiler "$TASK_QUANT/compile-v49-pair-ptq-board.py" \
  --ptq-compiler-sha256 09c6e33be328f6f0c74265c3bdb2fa2c281569ffb76bbbd61db9cf509f81da24 \
  --ptq-contracts "$TASK_QUANT/v49_quant_contracts.py" \
  --ptq-contracts-sha256 47237b1a1e1614822c75180efd15be3e5c708b610f587f698fe9b37fcf1bc649 \
  --ptq-compile-receipt "$TASK_B/vggt-v49-pair-ptq-r1/compile.json" \
  --ptq-compile-sha256 6c88b6271496ccdb6b8ff9db4e638fe47e69ca0c7374e90353ea51220a17e88a \
  --ptq-model "$TASK_B/vggt-v49-pair-ptq-r1/pair-int8.rknn" \
  --ptq-model-sha256 005c803d5fa6b4fa7664e583a6c8c188554c676172d955528f2617525fae18fa \
  --nonquant-compile-receipt "$TASK_B/vggt-v48-pair-compile-r1/compile.json" \
  --nonquant-model "$TASK_B/vggt-v48-pair-compile-r1/compiler/block.rknn" \
  --runtime-helper "$TASK_BASE/src/deploy/rk3588/validation/run-v12-rknn-block-board.py" \
  --sdk-header "$TASK_BASE/src/deploy/rk3588/reference/rknn_api_v2.3.2.h" \
  --diagnostic-allow-FP32-rejected --output-dir "$TASK_OUTPUT" "${TASK_EXTRA[@]}"
