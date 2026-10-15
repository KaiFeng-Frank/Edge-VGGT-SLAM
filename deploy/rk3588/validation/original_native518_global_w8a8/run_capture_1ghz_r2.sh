#!/bin/sh
# Root wraps this child in the existing180s/75C/restoring1GHz controller.
# R1 source, inputs and failed receipt remain preserved in their R1 directories.
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u \
  /mnt/vggt-data/vggt-slam2/experiments/vggt-v115-source-r2/capture_native_global0_inputs.py \
  --quality-receipt /mnt/vggt-data/vggt-slam2/experiments/vggt-v115-source-r2/quality-host-reference.json \
  --output-dir /tmp/vggt-v115-capture-r2 \
  --timeout-seconds 170
