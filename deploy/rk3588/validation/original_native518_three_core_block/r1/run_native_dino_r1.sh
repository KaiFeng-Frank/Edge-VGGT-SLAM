#!/bin/sh
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
 LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
 /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u -X faulthandler \
 /tmp/vggt-v124-source-r1/run_native_dino_board.py \
 --source-manifest /tmp/vggt-v124-source-r1/sources.json \
 --source-manifest-sha256 b2e383e03c4c514ca332b55185bdfc4e0c8be71c4b50eff1180c98ca093e1cf2 \
 --output-dir /tmp/vggt-v124-block-r1
