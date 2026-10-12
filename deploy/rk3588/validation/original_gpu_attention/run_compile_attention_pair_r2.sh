#!/bin/sh
set -eu
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
 LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /mnt/vggt-data/vggt-slam2/experiments/vggt-v49-toolchain-venv-r1/bin/python -B -u \
 /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-source-r1/compile_pair_attention_custom_board_r2.py \
 --export /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-pair-attention-custom-export-r1/export.json \
 --export-sha256 f1404c05303be91d368dc6f978fee391416e37802bfade5ceead6a3e9d635653 \
 --builder-source /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-source-r1/build_pair_attention_custom_board_r2.py \
 --builder-source-sha256 a081e542ea33f8278e33baedaf4d6567719522658726df6ce671ab7da1640899 \
 --registration-form instance \
 --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-pair-attention-custom-compile-r2
