#!/bin/sh
# External restoring controller owns the bounded NPU 1 GHz setting.
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= \
  /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u \
  /tmp/vggt-v122-source-r1/run_native_dino_board.py \
  --source-manifest /tmp/vggt-v122-source-r1/sources.json \
  --source-manifest-sha256 94b22b72ed259971e67d6df8d878754d54ebb2cae682d4e1886ce449237b8685 \
  --output-dir /tmp/vggt-v122-block-r1
