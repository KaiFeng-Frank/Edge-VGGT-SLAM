#!/bin/sh
# Run on the board after staging python/, wheelhouse/, src/ on its data partition.
set -eu
VGGT_DEPLOY_ROOT=${VGGT_DEPLOY_ROOT:-/mnt/vggt-data/vggt-slam2}
export VGGT_DEPLOY_ROOT
if [ "$(uname -m)" != aarch64 ]; then
    echo 'This offline environment is for Linux aarch64.' >&2
    exit 1
fi
test -x "$VGGT_DEPLOY_ROOT/python/bin/python3.11"
test -f "$VGGT_DEPLOY_ROOT/wheelhouse/requirements-board.txt"
if [ -f "$VGGT_DEPLOY_ROOT/wheelhouse/SHA256SUMS" ]; then
    (cd "$VGGT_DEPLOY_ROOT/wheelhouse" && sha256sum -c SHA256SUMS)
fi
export TMPDIR="$VGGT_DEPLOY_ROOT/tmp"
mkdir -p "$TMPDIR" "$VGGT_DEPLOY_ROOT/cache"
if [ -f "$VGGT_DEPLOY_ROOT/debs/SHA256SUMS" ]; then
    (cd "$VGGT_DEPLOY_ROOT/debs" && sha256sum -c SHA256SUMS)
    mkdir -p "$VGGT_DEPLOY_ROOT/sysroot"
    for package in "$VGGT_DEPLOY_ROOT/debs/"*.deb; do
        dpkg-deb -x "$package" "$VGGT_DEPLOY_ROOT/sysroot"
    done
fi
if [ ! -x "$VGGT_DEPLOY_ROOT/venv/bin/python" ]; then
    "$VGGT_DEPLOY_ROOT/python/bin/python3.11" -m venv "$VGGT_DEPLOY_ROOT/venv"
fi
"$VGGT_DEPLOY_ROOT/venv/bin/python" -m pip install --no-index --no-cache-dir \
    --find-links "$VGGT_DEPLOY_ROOT/wheelhouse" \
    -r "$VGGT_DEPLOY_ROOT/wheelhouse/requirements-board.txt"
. "$VGGT_DEPLOY_ROOT/src/deploy/rk3588/env.sh"
python -m pip check
python "$VGGT_DEPLOY_ROOT/src/scripts/preflight.py" \
    --output "$VGGT_DEPLOY_ROOT/outputs/preflight.json"

