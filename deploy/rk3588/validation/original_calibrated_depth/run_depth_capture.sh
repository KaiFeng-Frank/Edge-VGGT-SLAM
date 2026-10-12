#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v85-source-r1/capture_depth_calibration_board.py --depth-manifest /mnt/vggt-data/vggt-slam2/experiments/vggt-v54-depth-graphs-r1/compiled.json --original-depth-capture /mnt/vggt-data/vggt-slam2/experiments/vggt-v54-depth-capture-r1/capture.json --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v85-depth-calibration-r1
