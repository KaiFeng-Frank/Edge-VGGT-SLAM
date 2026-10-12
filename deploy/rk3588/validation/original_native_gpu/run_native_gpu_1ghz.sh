#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v81-source-r1/probe_native_gpu_mlp_observed_board.py --native-core-mask 1 --gpu-bridge-source /mnt/vggt-data/vggt-slam2/experiments/vggt-v81-source-r1/native_erf_gelu_opencl.py --gpu-bridge-sha256 f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2 --gpu-library /mnt/vggt-data/vggt-slam2/experiments/libmali-valhall-g610-g6p0-dummy.so --gpu-library-sha256 d57e53240b5078eee5cfbc79e103544c090a7825c09fdf3877fee45084897dda --expected-npu-hz 1000000000 --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v81-native-gpu-r1
