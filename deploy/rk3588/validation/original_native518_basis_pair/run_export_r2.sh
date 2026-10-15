#!/bin/sh
set -eu
cd /dev/shm/edge-vggt-v126-source-r1
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= TMPDIR=/dev/shm/edge-vggt-v126-source-r1/tmp \
  taskset -c 4-7 /mnt/vggt-data/vggt-slam2/experiments/vggt-v49-toolchain-venv-r1/bin/python -B -u \
  /dev/shm/edge-vggt-v126-source-r1/export_basis_pair_board.py \
  --mode export \
  --config /dev/shm/edge-vggt-v126-source-r1/basis-pair-inputs-r1.json \
  --config-sha256 442e22d5f8aaa033604610c92d5ce6a8b889005601f3d9f1cf78394c4f41842a \
  --output-dir /tmp/edge-vggt-v126-basis-export-r2
