#!/bin/sh
set -eu
vggt_base=/mnt/vggt-data/vggt-slam2
vggt_exp="$vggt_base/experiments"
. "$vggt_base/src/deploy/rk3588/env.sh"
vggt_pair_sha=$(sha256sum "$vggt_exp/vggt-v55-pair-graphs-r1/compiled.json" | cut -d ' ' -f 1)
exec python -B -u "$vggt_base/src/deploy/rk3588/validation/original_full_aggregator/compare_complete_model.py"  --manifest "$vggt_exp/vggt-v55-pair-graphs-r1/compiled.json" --manifest-sha256 "$vggt_pair_sha"  --paired-aggregator --grouped-dino --half-io-library "$vggt_exp/vggt-v56-neon-build-r1/fp16_neon.so" --half-io-library-sha256 d08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9 --model-threads 1 --output "$vggt_exp/vggt-v59-1ghz-complete-r1"  --dino-manifest "$vggt_exp/vggt-v58-dino-groups-r1/compiled.json"  --dino-manifest-sha256 a602bcdaab3f2e1ab51d49feb300ff3af76c07fe334489c983d647859f3f9493  --depth-manifest "$vggt_exp/vggt-v54-depth-graphs-r1/compiled.json"  --depth-manifest-sha256 9a6e5435df543b8484a4fc50da4b065995f32153233548829234dd149fbc7135  --reference-run "$vggt_exp/vggt-v53-complete-ab-r1"
