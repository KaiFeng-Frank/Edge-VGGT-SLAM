#!/bin/sh
# External restoring controller owns the bounded NPU 1 GHz setting.
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= \
  /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u \
  /tmp/vggt-v122-source-r2/run_native_dino_board.py \
  --source-manifest /tmp/vggt-v122-source-r2/sources.json \
  --source-manifest-sha256 356bdbf3ef4d6f4b46a245b2ba7a037757362bcd73772fdefdb362d98be7e10f \
  --output-dir /tmp/vggt-v122-block-r2
