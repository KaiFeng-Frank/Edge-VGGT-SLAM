#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v76-source-r1/probe_retained_profiles_board.py --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v76-retained-profile-1ghz-r1 --expected-npu-hz 1000000000
