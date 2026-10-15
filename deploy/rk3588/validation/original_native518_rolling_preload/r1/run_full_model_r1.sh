#!/bin/sh
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u \
  /tmp/vggt-v123-source-r1/run_rolling_full_model_board.py \
  --source-manifest /tmp/vggt-v123-source-r1/sources.json \
  --source-manifest-sha256 9a65445736f7c5a6cfa8c66053c746c05a4e9e3c313b218562ef7f3e72f0f84a \
  --output-dir /tmp/vggt-v123-full-r1
