#!/bin/sh
set -eu
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
 LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /mnt/vggt-data/vggt-slam2/experiments/vggt-v49-toolchain-venv-r1/bin/python -B -u \
 /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-source-r1/compile_pair_attention_custom_board.py \
 --export /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-pair-attention-custom-export-r1/export.json \
 --export-sha256 f1404c05303be91d368dc6f978fee391416e37802bfade5ceead6a3e9d635653 \
 --builder-source /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-source-r1/build_pair_attention_custom_board.py \
 --builder-source-sha256 2050cf3c51986e787f3661e787925a4639f6008091584dcf8f2050797a11bc27 \
 --registration-form instance \
 --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-pair-attention-custom-compile-r1
