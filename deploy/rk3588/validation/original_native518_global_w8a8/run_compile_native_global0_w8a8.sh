#!/bin/sh
# Source-only launcher: root supplies actual capture/check receipt pins and a fresh /tmp output.
# Invoke under the existing bounded board controller; this script writes no clocks.
set -eu
BASE=/mnt/vggt-data/vggt-slam2
S115="$BASE/experiments/vggt-v115-source-r1"
SCRIPT="$S115/compile_native_global0_w8a8.py"
SCRIPT_SHA=8e9aa958acf9f6a876e7ce6caee956f92c879b617604414a8dc4635548e9592e
MODE="${1:?First argument must be check or compile}"
shift
case "$MODE" in
  check) PY="$BASE/venv/bin/python" ;;
  compile) PY="$BASE/experiments/vggt-v49-toolchain-venv-r1/bin/python" ;;
  *) echo 'First argument must be check or compile' >&2; exit 2 ;;
esac
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV LD_LIBRARY_PATH="$BASE/sysroot/usr/lib/aarch64-linux-gnu" \
  "$PY" -B -c 'import hashlib,pathlib,sys; assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()==sys.argv[2], "Frozen compiler source differs"' "$SCRIPT" "$SCRIPT_SHA"
if [ "$MODE" = check ]; then
  set -- --verify-only "$@"
fi
exec timeout -s TERM 180 env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  LD_LIBRARY_PATH="$BASE/sysroot/usr/lib/aarch64-linux-gnu" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  "$PY" -B -u "$SCRIPT" "$@"
