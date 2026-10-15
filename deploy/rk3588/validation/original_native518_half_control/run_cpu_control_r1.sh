#!/bin/sh
# Original-model component precision diagnosis; no frequency writes or SDK calls.
set -u
base=/mnt/vggt-data/vggt-slam2
run=/tmp/vggt-v116-half-control-r1
control=/tmp/vggt-v116-cpu-control-r1
mkdir "$control" || exit 1
cat /sys/class/devfreq/fdab0000.npu/cur_freq > "$control/npu-hz-before.txt" || exit 1
date -u +%Y-%m-%dT%H:%M:%SZ > "$control/start-utc.txt"
timeout -k 10s -s TERM 180s env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH="$base/sysroot/usr/lib/aarch64-linux-gnu" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES='' \
  "$base/venv/bin/python" -B -u /tmp/vggt-v116-source-r1/run_dino0_half_control_board.py \
  --repo "$base/src" --output-dir "$run" > "$control/worker.log" 2>&1
code=$?
printf '%s\n' "$code" > "$control/child-returncode.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$control/end-utc.txt"
cat /sys/class/devfreq/fdab0000.npu/cur_freq > "$control/npu-hz-after.txt" || exit 1
cat "$control/worker.log"
exit "$code"
