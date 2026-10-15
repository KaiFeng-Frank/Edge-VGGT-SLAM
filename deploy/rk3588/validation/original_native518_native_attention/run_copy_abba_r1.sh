#!/bin/sh
# The external restoring 1 GHz / 75 C / 180 s controller owns NPU policy.
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= \
  /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u \
  /tmp/vggt-v118-source-r1/run_native_attention_copy_abba.py \
  --output-dir /tmp/vggt-v118-copy-abba-r1
