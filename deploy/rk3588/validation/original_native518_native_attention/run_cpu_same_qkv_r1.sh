#!/bin/sh
# CPU-only reference controls; no system policy writes or thermal trace.
cat /sys/class/devfreq/fdab0000.npu/cur_freq
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= \
  timeout -s TERM -k 10 180 /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u \
  /tmp/vggt-v118-source-r1/cpu_sdpa_same_qkv_board.py \
  --output-dir /tmp/vggt-v118-same-qkv-r1
code=$?
cat /sys/class/devfreq/fdab0000.npu/cur_freq
exit "$code"
