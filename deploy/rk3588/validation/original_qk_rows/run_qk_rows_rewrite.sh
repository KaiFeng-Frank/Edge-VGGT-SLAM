#!/bin/sh
set -eu
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/vggt-data/vggt-slam2/experiments/vggt-v49-toolchain-venv-r1/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v86-source-r1/rewrite_frame_qk_rows_board.py --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v86-qk-rows-export-r1
