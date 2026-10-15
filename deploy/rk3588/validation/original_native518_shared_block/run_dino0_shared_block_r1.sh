#!/bin/sh
# The external restoring 1 GHz / 75 C / 180 s controller owns NPU policy.
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= \
  /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u \
  /tmp/vggt-v120-source-r1/run_dino0_shared_block_board.py \
  --export /tmp/vggt-v120-export-r1/export.json \
  --export-sha256 4290248a74a73e129b931192203a3aac01473542b26314b1b22f81f3afc0e4f7 \
  --check /tmp/vggt-v120-check-r1/result.json \
  --check-sha256 362b1b9152af635beea2c08904a39903809485643eb8160e2e5e86fccfade48b \
  --compile /tmp/vggt-v120-compile-r1/result.json \
  --compile-sha256 cf7ff2fc4d7fd27af7cf5666d995a9382587ea2ec1c46ea643daedc921589d0a \
  --output-dir /tmp/vggt-v120-block-r1
