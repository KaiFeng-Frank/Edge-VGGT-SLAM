#!/usr/bin/env bash
set -euo pipefail

# Root alone schedules this after the current compiler job, before SDK evaluation.
TASK_BASE=/mnt/vggt-data/vggt-slam2
TASK_QUANT_SOURCE="${VGGT_QUANT_SOURCE:-$TASK_BASE/experiments/vggt-v49-quant-source}"
TASK_CAPTURE_SOURCE="${VGGT_CAPTURE_SOURCE:-$TASK_BASE/experiments/vggt-v49-capture-source}"
TASK_TOOLCHAIN="$TASK_BASE/experiments/vggt-v49-toolchain-venv-r1"
TASK_OUTPUT="${VGGT_HELDOUT_ORT_OUTPUT:-$TASK_BASE/experiments/vggt-v49-heldout-pair-ort-r1}"

cd "$TASK_BASE"
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH="$TASK_BASE/sysroot/usr/lib/aarch64-linux-gnu" \
  "$TASK_TOOLCHAIN/bin/python" -B "$TASK_QUANT_SOURCE/check-v49-heldout-pair-onnx-board.py" \
  --contracts "$TASK_QUANT_SOURCE/v49_quant_contracts.py" \
  --contracts-sha256 47237b1a1e1614822c75180efd15be3e5c708b610f587f698fe9b37fcf1bc649 \
  --pair-contracts "$TASK_BASE/experiments/vggt-v48-pair-source/v47_pair_contracts.py" \
  --pair-dir "$TASK_BASE/experiments/vggt-v48-pair" \
  --capture-contracts "$TASK_CAPTURE_SOURCE/v49_capture_contracts.py" \
  --capture-contracts-sha256 6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8 \
  --capture-dir "$TASK_BASE/experiments/vggt-v49-capture-r1" \
  --capture-sha256 bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc \
  --toolchain-prefix "$TASK_TOOLCHAIN" --output-dir "$TASK_OUTPUT"
