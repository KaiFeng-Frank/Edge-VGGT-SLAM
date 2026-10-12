#!/bin/sh
set -eu
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/vggt-data/vggt-slam2/experiments/vggt-v49-toolchain-venv-r1/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v84-source-r1/build_pair_attention_head_major_board.py --parent-manifest /mnt/vggt-data/vggt-slam2/experiments/vggt-v55-pair-graphs-r1/compiled.json --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v84-head-major-export-r1
