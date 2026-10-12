#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v76-source-r1/probe_retained_fc2_group_board.py --candidate-result-sha256 d1ef4a8fb38fdbcf8499baed8536e1cd82061e283d98fc576fcc6f4fa951da06 --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v76-retained-k4-normal-1ghz-r1 --expected-npu-hz 1000000000
