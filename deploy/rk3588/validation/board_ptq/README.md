# Board-only PTQ on RK3588S (v35 preparation, v36 execution)

**v36: the isolated ARM environment is installed and real board-CPU calibration/build completed. Deployment inference, INT8 coverage, accuracy and real-time performance remain unvalidated.** [Actual receipts and preserved failures](../../../../results/board_ptq_v36.json). The existing production environment and historical FP16 results remain unchanged. These MIT-licensed project helpers prepare a DPVO encoder experiment; DPVO attribution is retained, and each external SDK/package keeps its own licence. SDK binaries, official source, weights, PDFs and Python wheels are not included here.

RKNN `build(do_quantization=True)` uses real calibration samples to estimate activation ranges. The official guide describes this work in §6.2.3, and the installed SDK's quantizer/session references point to CPU inference. Calling this API on the development host would violate the project's no-host-neural-inference constraint. The measured route runs calibration **on the RK3588 board CPU**, then separately measures NPU inference. Both installer and PTQ entrypoint require Linux, aarch64, and an RK3588/S device-tree identity before any installation/calibration work. [Official guide](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/doc/02_Rockchip_RKNPU_User_Guide_RKNN_SDK_V2.3.2_EN.pdf)

The official ARM64 cp311 requirements constrain Torch to ≤2.2.0. This plan uses a **new venv without system-site-packages**, with Torch 2.2.0, NumPy 1.26.4, ONNX 1.16.1, ORT 1.16.0 and protobuf 3.20.3. It does not replace the existing Torch 2.3.1 environment. All 32 selected packages, their exact official PyPI URLs and archive SHA256 values are in [arm64-dependency-plan.json](arm64-dependency-plan.json) and [arm64-downloads.json](arm64-downloads.json). The selected wheels require glibc ≥2.24. The actual r2 installation and subsequent board PTQ are separately recorded; metadata preflight alone does not verify shared-library imports. [Official ARM requirements](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknn-toolkit2/packages/arm64/arm64_requirements_cp311.txt)

The [official ARM64 Toolkit wheel](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknn-toolkit2/packages/arm64/rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl) is 37,740,634 bytes. Its pinned official Git blob is `0313b8181d121207f12c5a745427d817f18aeeee`; the downloaded full wheel SHA256 is `2793b0f4c008c13aae1d19c7cd9af891222d1bbe9e012ced05ec49d53240ed9c`. The installer also verifies the official Git blob and the wheel's metadata hash.

onnxoptimizer 0.3.8 has no matching PyPI ARM64 wheel. Its [official source archive](https://files.pythonhosted.org/packages/a2/c2/8fcbecf89c52ca3beaccbed4aedb05eb9aa2bf232e18987942b3e1d3f227/onnxoptimizer-0.3.8.tar.gz) includes ONNX, protobuf/protoc and pybind11 sources, so a separate git-submodule download is unnecessary. The candidate installer uses the previously verified isolated v27 GCC 11, CMake ≥3.22, Ninja, matching Python 3.11 headers, external pybind11 2.11.1, and `MAX_JOBS=2`. It builds the unchanged optimizer source in a new experiment directory. The r1 assembler failed to load `libopcodes-2.38-system.so`. The r2 installer adds only a subprocess-local sysroot runtime-library path; the actual native build then succeeded in 371.356 s. All 37 installation commands and `pip check` succeeded. [Static archive review](onnxoptimizer-sdist-source-review.json)

The commands below reproduce the board stages with new output locations. Supply the verified SDK wheel and existing historical artifacts; every output directory and the venv must be new. The installer streams one pinned package at a time or accepts an offline wheelhouse. It never installs into the old environment, invokes RKNN build, or performs a neural forward. Installation success establishes package/build readiness; actual PTQ has a separate receipt.

```bash
PTQ_PROJECT=/mnt/vggt-data/vggt-slam2
PTQ_SRC="$PTQ_PROJECT/src/deploy/rk3588/validation/board_ptq"
PTQ_ENV="$PTQ_PROJECT/experiments/vggt-v36-arm-toolchain-venv-repro"
SDK_WHEEL="${SDK_WHEEL:?Set the path of the verified ARM64 cp311 Toolkit wheel}"
SDK_SHA="${SDK_SHA:?Set the SHA256 of that downloaded wheel}"
PREFLIGHT_SHA="${PREFLIGHT_SHA:?Set the frozen preflight source SHA256}"
"$PTQ_PROJECT/venv/bin/python" -B "$PTQ_SRC/install-v36-arm-toolchain-board.py" \
  --dependency-plan "$PTQ_SRC/arm64-dependency-plan.json" \
  --base-python "$PTQ_PROJECT/venv/bin/python" \
  --sdk-wheel "$SDK_WHEEL" --sdk-wheel-sha256 "$SDK_SHA" \
  --preflight-script "$PTQ_SRC/preflight-v35-arm-toolchain.py" \
  --preflight-script-sha256 "$PREFLIGHT_SHA" \
  --gcc-sysroot "$PTQ_PROJECT/experiments/vggt-v27-toolchain/root" \
  --venv-path "$PTQ_ENV" \
  --output-dir "$PTQ_PROJECT/experiments/vggt-v36-arm-install-repro" \
  --download-wheels
```

Replace `--download-wheels` with `--wheelhouse /verified/offline/packages` for an offline installation. It must contain all pinned wheel files and the optimizer source archive. No automatic package-version substitution is permitted. Installation failures preserve the new partial venv and logs.

The PTQ candidate targets only the frozen **original** v33 sparse-context encoder graph. It pins the v33 CPU suffix acceptance and v34 compiler/editor lineage, preserves all original learned parameters in the floating graph, and leaves the already rejected spatial parent outside this experiment. The calibration text file contains one `image.npy indices.npy` pair per line in graph input order. Images must be once-normalized BGR FP32 `[1,3,464,608]`; indices must remain INT64 `[96]`, including their order and repeats. The script never changes resolution, casts indices to float, or freezes them into the graph. Range/shape checks alone do not prove preprocessing provenance or calibration representativeness. The operator must retain source frame IDs/hashes and original x-then-y-then-depth RNG ordering separately. [Official multi-input example](https://github.com/airockchip/rknn-toolkit2/blob/59a913d172e7f5ff03c9076e2ec7b1b1288ffd08/rknn-toolkit2/examples/functions/multi_input/test.py)

```bash
ORIGINAL_SPARSE_DIR="${ORIGINAL_SPARSE_DIR:?Set the frozen original v33 rewrite directory}"
V33_REFERENCE_DIR="${V33_REFERENCE_DIR:?Set the frozen v33 CPU ORT result directory}"
PARENT_ENCODER_ONNX="${PARENT_ENCODER_ONNX:?Set the original v26 encoder ONNX path}"
CALIBRATION_LIST="${CALIBRATION_LIST:?Set the source-bound calibration dataset text path}"
CALIBRATION_SHA="${CALIBRATION_SHA:?Set its SHA256}"
"$PTQ_ENV/bin/python" -B "$PTQ_SRC/compile-v35-dpvo-int8-board.py" \
  --dependency-plan "$PTQ_SRC/arm64-dependency-plan.json" \
  --dependency-plan-sha256 a32e1a00c8bdbf74241de3de236cdaef050567b26b9d591a7f57c65ab247e2b4 \
  --preflight-script "$PTQ_SRC/preflight-v35-arm-toolchain.py" \
  --preflight-script-sha256 "$PREFLIGHT_SHA" \
  --frozen-sparse-compiler "$PTQ_PROJECT/src/deploy/rk3588/validation/dpvo_sparse_context_sdk/compile-v34-dpvo-sparse-context.py" \
  --rewrite-dir "$ORIGINAL_SPARSE_DIR" --reference-dir "$V33_REFERENCE_DIR" \
  --parent-onnx "$PARENT_ENCODER_ONNX" \
  --editor "$PTQ_PROJECT/src/deploy/rk3588/validation/dpvo_sparse_context/rewrite-v33-dpvo-sparse-context.py" \
  --dataset "$CALIBRATION_LIST" --dataset-sha256 "$CALIBRATION_SHA" \
  --output-dir "$PTQ_PROJECT/experiments/vggt-v36-sparse-original-int8-repro"
```

The requested experiment is RK3588 / optimization 3 / W8A8 / normal calibration / per-channel weights, with identity mean/std and no RGB conversion or automatic hybrid tuning. Actual calibration/build succeeded with both inputs preserved. Internal precision and placement still need runtime evidence: documentation allows INT8 convolution/Gather data but lists RK3588 `exNorm` as FP16-only. The fixed 32-frame calibration policy and input-provenance audit are in [board_calibration](../board_calibration/README.md). Board calibration/build took 60.291 s, peak RSS was 1,170 MiB, and the exported model has 12,491,378 bytes. These values measure compilation, not deployment inference. Compilation success remains unaccepted for model accuracy and performance.

The future check must query actual input types, preserve integer Gather choices, compare outputs against the same CPU/FP16 inputs under the unchanged `rtol=1e-4, atol=1e-5` gate, and record rejection honestly. Separate full-trajectory GT/coverage/ATE/RPE and complete I/O/latency/RSS measurements are needed to assess an approximate INT8 candidate; a new accuracy diagnosis must not convert an existing strict FP32 rejection into equivalence or a 20 Hz claim. Calibration/evaluation separation and monocular alignment must be disclosed. [Scope and source evidence](int8-feasibility-review.json)

Host checks use synthetic arrays and fake SDK methods only:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_v35_board_ptq.py
```

No host real images, original-weight forward, actual SDK import/build, installation or board job is used by these tests.

## Native IO query after actual board calibration

`query-v36-int8-io-board.py` performs only guarded `rknn_init`, `rknn_query` and `rknn_destroy`. The actual query completed with zero inference calls and clean destruction; the saved query audit passed 60 checks. It binds the compile receipt/model SHA, frozen ABI/header/runtime and all source hashes. Use `--help` for the required explicit paths and hashes; the output directory must be a new board experiment directory. This entrypoint does not submit inputs or retrieve outputs.

The image port is INT8 NHWC, indices remain INT64 with neutral scale/zero-point, and both feature ports are INT8. Native fmap storage is 2,375,680 bytes with stride versus 2,256,896 logical INT8 bytes. Future `want_float=1` outputs require 9,175,040 total logical FP32 bytes; native storage length is not the converted output length. No complete deployment inference has been measured by this query. [Query source](query-v36-int8-io-board.py), [saved query audit](io-query-independent-review-r1.json), [all stage receipts](../../../../results/board_ptq_v36.json).
