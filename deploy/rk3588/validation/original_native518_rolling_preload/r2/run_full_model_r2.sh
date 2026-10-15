#!/bin/sh
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u -X faulthandler \
  /tmp/vggt-v123-source-r2/run_rolling_full_model_board.py \
  --source-manifest /tmp/vggt-v123-source-r2/sources.json \
  --source-manifest-sha256 232eb98e124b895fa25d4ba5083bb157d5adac206e923283edb8b86f9073ad77 \
  --output-dir /tmp/vggt-v123-full-r2
