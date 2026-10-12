#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v81-source-r1/probe_native_ac_mlp_core1_board.py --expected-npu-hz 1000000000 --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v81-native-abc-r2
