#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-source-r1/probe_callback_shadow_board.py --capture /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-attention-callback-capture-r2/capture.json --capture-sha256 f21539a34e1b54f66494f90ec58f27ac57b3052e6dab8acc5d90ce06ffc032b4 --input-domain gpu --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-attention-callback-shadow-r1
