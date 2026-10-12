#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v82-source-r1/capture_camera_calibration_board.py --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v82-camera-calibration-r1
