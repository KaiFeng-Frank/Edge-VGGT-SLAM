#!/bin/sh
# Source this after installing the isolated environment on the data partition.
VGGT_DEPLOY_ROOT=${VGGT_DEPLOY_ROOT:-/mnt/vggt-data/vggt-slam2}
export VGGT_DEPLOY_ROOT
export PYTHONPATH="$VGGT_DEPLOY_ROOT/src:$VGGT_DEPLOY_ROOT/src/third_party/vggt${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$VGGT_DEPLOY_ROOT/venv/bin:$PATH"
export LD_LIBRARY_PATH="$VGGT_DEPLOY_ROOT/sysroot/usr/lib/aarch64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_HOME="$VGGT_DEPLOY_ROOT/cache/torch"
export HF_HOME="$VGGT_DEPLOY_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export XDG_CACHE_HOME="$VGGT_DEPLOY_ROOT/cache"

