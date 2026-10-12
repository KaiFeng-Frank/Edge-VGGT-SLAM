#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-source-r1/capture_original_attention_tail_board.py --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v83-attention-tail-capture-r1
