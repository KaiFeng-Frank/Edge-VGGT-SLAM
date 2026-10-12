#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v82-source-r1/probe_camera_int8_board.py --candidate-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v82-camera-int8-r1 --candidate-sha256 4d4094c76570dd497af3553b6dfcbc768ebd65147717db7aca8b20bc4c0efc89 --compiler-source /mnt/vggt-data/vggt-slam2/experiments/vggt-v82-source-r1/compile_camera_int8_board.py --compiler-sha256 2f935b2937b7465a140a8d1a408cdb0a7205a6a89e8d54833271cf0e51e4f374 --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v82-camera-int8-probe-1ghz-r1 --expected-npu-hz 1000000000
