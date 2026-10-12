#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v81-source-r1/probe_fc2_three_cores_board.py --native-helper /mnt/vggt-data/vggt-slam2/experiments/vggt-v81-source-r1/probe_native_gpu_mlp_board_r2.py --expected-npu-hz 1000000000 --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v81-fc2-three-cores-r1
