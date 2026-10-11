#!/bin/bash
set -eu
cd /mnt/vggt-data/vggt-slam2
. src/deploy/rk3588/env.sh
exec python -B experiments/vggt-v48-ab-source/run-v48-frame-global-ab-board.py \
  --ab-contracts experiments/vggt-v48-ab-source/v48_ab_contracts.py \
  --ab-contracts-sha256 6b63e117445a453ac383c0d8fa76cc831c9d81af4b2f33f982459aa5fe0a8802 \
  --pair-contracts experiments/vggt-v48-pair-source/v47_pair_contracts.py \
  --pair-dir experiments/vggt-v48-pair \
  --runtime-helper src/deploy/rk3588/validation/run-v12-rknn-block-board.py \
  --sdk-header src/deploy/rk3588/reference/rknn_api_v2.3.2.h \
  --frame0-export-receipt experiments/vggt-v19-frame336-export-r3/export.json \
  --frame0-ort-receipt experiments/vggt-v19-frame336-ort-r3/check.json \
  --frame0-compile-receipt experiments/vggt-v19-frame336-compile-r1/compile.json \
  --frame0-model experiments/vggt-v19-frame336-compile-r1/block.rknn \
  --global0-export-receipt experiments/vggt-v17-global336-export-r3/export.json \
  --global0-ort-receipt experiments/vggt-v17-global336-ort-r3/check.json \
  --global0-compile-receipt experiments/vggt-v17-global336-compile-r3/compile.json \
  --global0-model experiments/vggt-v17-global336-compile-r3/block.rknn \
  --pair-compile-receipt experiments/vggt-v48-pair-compile-r1/compile.json \
  --pair-compile-sha256 5084e00143c4ab5fce94ceae81247c11939ddb7783d80d1b937bffcbdc0bcfb9 \
  --pair-ort-receipt experiments/vggt-v48-pair-ort-r2/check.json \
  --pair-ort-sha256 f93e1939be9628b7e28df8f91cf06d28002f0b3871a0d0a46adca1780ffa5fc0 \
  --pair-model experiments/vggt-v48-pair-compile-r1/compiler/block.rknn \
  --pair-model-sha256 5ba3440634a51ae56fdc46eb1c07bf5847a30640a7c2603d322b44af585d037b \
  --diagnostic-allow-FP32-rejected \
  --output-dir experiments/vggt-v48-pair-ab-r1
