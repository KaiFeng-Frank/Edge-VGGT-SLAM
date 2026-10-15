#!/bin/sh
# External restoring controller owns the bounded NPU 1 GHz setting.
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= \
  /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u \
  /tmp/vggt-v121-source-r1/run_native_dino_board.py \
  --source-manifest /tmp/vggt-v121-source-r1/sources.json \
  --source-manifest-sha256 d93dc04823aea225122e98ed73ba1b173123460a4d66db9b368002b40774e7c1 \
  --output-dir /tmp/vggt-v121-block-r1
