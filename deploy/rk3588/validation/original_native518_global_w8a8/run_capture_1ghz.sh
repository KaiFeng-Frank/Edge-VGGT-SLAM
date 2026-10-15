#!/bin/sh
# Root wraps this child in the existing180s/75C/restoring1GHz controller.
# This launcher and capture do not write clocks, governors or system settings.
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u \
  /mnt/vggt-data/vggt-slam2/experiments/vggt-v115-source-r1/capture_native_global0_inputs.py \
  --quality-receipt /mnt/vggt-data/vggt-slam2/experiments/vggt-v115-source-r1/quality-host-reference.json \
  --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v115-capture-r1 \
  --timeout-seconds 170
