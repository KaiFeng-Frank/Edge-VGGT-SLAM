#!/bin/sh
set -eu
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /mnt/vggt-data/vggt-slam2/venv/bin/python -B -u /mnt/vggt-data/vggt-slam2/experiments/vggt-v78-source-r1/probe_smoothed_pair_board.py --checked /mnt/vggt-data/vggt-slam2/experiments/vggt-v78-pair-power2-smooth-check-r1/result.json --checked-sha256 e9092d0aa3789d5b19719f87b7e2ed5312899f46df02efe00323e6049ce0a9dd --compiled /mnt/vggt-data/vggt-slam2/experiments/vggt-v78-pair-power2-smooth-compile-r1/compile.json --compiled-sha256 1855b487b08734bef44a2a1cb63e6638e4dcb0caf0697f2ef91e54a61150affd --output-dir /mnt/vggt-data/vggt-slam2/experiments/vggt-v78-smooth-profile-r1 --rounds 6 --frequency-hz 1000000000 --profile
