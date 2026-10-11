#!/usr/bin/env bash
set -euo pipefail
source_dir="${1:-/mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-source}"
output_dir="${2:-/mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-step2-r1}"
work_dir="${3:-/dev/shm/vggt-v50-hybrid-step2-r1}"
toolchain_prefix=/mnt/vggt-data/vggt-slam2/experiments/vggt-v49-toolchain-venv-r1
cd /mnt/vggt-data/vggt-slam2
exec env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH=/mnt/vggt-data/vggt-slam2/sysroot/usr/lib/aarch64-linux-gnu \
  "${toolchain_prefix}/bin/python" -B "${source_dir}/compile-v50-hybrid-step2-board.py" \
  --toolchain-prefix "${toolchain_prefix}" --output-dir "${output_dir}" --work-dir "${work_dir}"
