"""Research-only dense spatial RKNN DPVO encoder; no CPU encoder fallback.

The frozen v27 InstanceNorm rewrite failed the original FP32 gate. This bridge
retains that rejection and requires explicit diagnostic authorization. DPVO
RANDOM Patchifier/normalization are reused byte-for-byte from v30 (MIT, Princeton
Vision & Learning Lab, 2022, commit 0ac95b656d1fda91c271d2a106460d19ad966fc7).
See the accompanying DPVO MIT license. No default model or VO acceptance change.

The SDK context is persistent. Every forward packs the native input layout,
executes synchronously, copies both dense logical FP32 outputs before release,
checks native E RKNN logs and finite/nonzero values, then samples original M
patches. Source hashes/statistics are outside the SDK timing. Returned arrays
are stable owned snapshots; SDK profiling is disabled for this trajectory probe.
"""
from __future__ import annotations

import copy
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import threading
import time

import numpy as np

sys.dont_write_bytecode = True
MODEL_SHA = "b4bd0b8ce7e09a3ab9d8dffaa7ccacd58c33842f9be5d0885b113d2fdc6a9869"
COMPILE_SHA = "7206700a64fb1da400d3f85b6b6f005d0ab883802504ce9f74c3fc8e60d7e733"
COMPILER_SHA = "364d3493f9fe13789b449c0836c37d0eedca71cdbde71e10e28723ec88d1b003"
SDK_HELPER_SHA = "646f875d1b5eea561c0d6aa4d1b9078351014a39a62976ec895a814cd466b387"
PARENT_RUNTIME_SHA = "895ac2902a66c68048c03e98f353300c7065e7568047479db4fee2f9a80a33d3"
ENCODER_MODULE_SHA = "fc6fc662613a720cfb779a360a63906a593a96ff08addecb9d01cfd8767ca7a8"
NATIVE_LOG_HELPER_SHA = "322021ade54fb9035b7ce80652ec90169938a417fd2192528d28a805a83ed25e"
REWRITE_SHA = "66eb8d135165f7538c97733d0c83198fa5a674e0cd2bdf64eb190c24819c5109"
FAILED_ORT_SHA = "a517f822a2e4afd01aff87df303183a3dc5850313134548891dc13c1e36b896b"
INPUT_SHAPE = (1, 3, 464, 608)
OUTPUT_SHAPES = {"fmap": (1, 128, 116, 152), "imap": (1, 384, 116, 152)}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def board_guard():
    require(platform.system() == "Linux" and platform.machine() == "aarch64"
            and b"rockchip,rk3588" in Path("/proc/device-tree/compatible").read_bytes(),
            "Actual RKNN encoder execution is RK3588 board-only")


def _native_ok(record):
    require(record.get("truncated") is False, "Truncated SDK log cannot be fully checked")
    require(record.get("RKNN_error_detected") is False,
            "Native E RKNN diagnostic is fatal even when C API returned zero")


def _array_info(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                nonfinite_elements=int(value.size - np.count_nonzero(np.isfinite(value))))


class BoardRKNNEncoder:
    """Same encode/RNG/patch contract as v30, with one dense NPU call per frame.

    No global affinity/OMP/Torch settings are changed. All SDK access is serial;
    native fd capture is suitable only for this synchronous isolated diagnostic.
    Runtime/source checks precede context creation. Failure destroys the context,
    saves diagnostic metadata/raw returned arrays, and never invokes CPU fallback.
    """
    def __init__(self, model_path, compile_receipt, rewrite_receipt, failed_ort_receipt,
                 *, compiler_source, sdk_helper, parent_runtime, abi_helper, header,
                 library, encoder_module, native_log_helper, log_directory,
                 diagnostic_allow_fp32_rejected=False):
        require(diagnostic_allow_fp32_rejected is True,
                "Explicit --diagnostic-allow-fp32-rejected is required")
        board_guard()  # Before graph/helper loads, writes or SDK context creation.
        self._lock = threading.RLock()
        self._context = None
        self._closed = False
        self._pins = {}
        self._cleanup = []
        self._activity = {"sdk_run_api_calls": 0}
        self._context_record = {}
        self._directory = Path(log_directory).resolve(strict=True)
        require(self._directory.is_dir(), "Existing isolated log directory required")
        self._metrics = dict(encoder_backend="v32_dense_spatial_RKNN_diagnostic",
            CPU_model_execution=False, NPU_used=False, CPU_encoder_fallback=False,
            candidate_accepted=False, historical_FP32_gate_passed=False,
            FP32_thresholds_relaxed=False, output_clipping=False,
            fixed_input_shape=list(INPUT_SHAPE), original_checkpoint_weights=True,
            feature_outputs_scaled_by_four_in_graph=True, completed_frames=0,
            contexts_created=0, contexts_reused=True, sdk_run_api_calls=0,
            successful_sdk_frames=0, normalization_seconds=0., SDK_outer_seconds=0.,
            sampling_seconds=0., total_encode_seconds=0., calls=[], cleanup_errors=[],
            closed=False, array_ownership="Owned logical FP32 copies before SDK release",
            collect_perf_detail=False, source_rechecks_inside_forward=False)
        self._log_helper = None
        try:
            self._pin(__file__, sha(__file__))
            base = load(self._pin(parent_runtime, PARENT_RUNTIME_SHA), "_v35_frozen_v26_runtime")
            self._encoder = load(self._pin(encoder_module, ENCODER_MODULE_SHA), "_v35_frozen_v30_patchifier")
            self._log_helper = load(self._pin(native_log_helper, NATIVE_LOG_HELPER_SHA), "_v35_frozen_native_log")
            compiler = load(self._pin(compiler_source, COMPILER_SHA), "_v35_frozen_v32_compiler")
            sdk = load(self._pin(sdk_helper, SDK_HELPER_SHA), "_v35_frozen_v34_context")
            rewrite = json.loads(self._pin(rewrite_receipt, REWRITE_SHA).read_text())
            failed = json.loads(self._pin(failed_ort_receipt, FAILED_ORT_SHA).read_text())
            ancestor = compiler.validate_lineage(rewrite, failed)
            compiled = json.loads(self._pin(compile_receipt, COMPILE_SHA).read_text())
            compiler.validate_compilation(compiled, ancestor, expected_script_sha256=COMPILER_SHA)
            model = self._pin(model_path, MODEL_SHA)
            require(compiled["output"]["sha256"] == MODEL_SHA and
                    model.stat().st_size == compiled["output"]["size"] == 3831208,
                    "Exact measured dense spatial model required")
            self._abi = base.load_abi(abi_helper, self._pins)
            self._pin(header, base.HEADER_SHA)
            self._pin(library, base.LIBRARY_SHA)
            self._metrics.update(model_sha256=MODEL_SHA, compile_sha256=COMPILE_SHA,
                                 ancestor=ancestor, configuration=compiled["configuration"],
                                 ABI_sha256=base.ABI_SHA, header_sha256=base.HEADER_SHA,
                                 native_runtime_sha256=base.LIBRARY_SHA,
                                 parent_runtime_sha256=PARENT_RUNTIME_SHA,
                                 sdk_helper_sha256=SDK_HELPER_SHA,
                                 normalization_patchifier_sha256=ENCODER_MODULE_SHA)
            api = self._abi.bind_api(ct.CDLL(str(Path(library).resolve())))
            with self._log_helper._native_log(self._directory) as native:
                self._context = sdk.Context(base, self._abi, api, model, False, False,
                                            self._cleanup, self._context_record)
            self._metrics["initial_native_log"] = copy.deepcopy(native)
            _native_ok(native)
            self._metrics["contexts_created"] = 1
            self._metrics["initial_source_checks_passed"] = True
        except Exception as exc:
            self._metrics["startup_error"] = repr(exc)
            if "native" in locals():
                self._metrics["initial_native_log"] = copy.deepcopy(native)
            self.close()
            self._save_failure(exc)
            raise

    def _pin(self, path, expected):
        path = Path(path).resolve(strict=True)
        require(sha(path) == expected, "Pinned encoder artifact changed: " + path.name)
        require(str(path) not in self._pins or self._pins[str(path)] == expected,
                "Conflicting encoder artifact pin")
        self._pins[str(path)] = expected
        return path

    def _save_failure(self, exc, actual=None):
        # No nonfinite Python floats are written to JSON. Raw invalid arrays stay
        # in NPZ for diagnosis; they never enter the VO stream.
        failure = dict(error=repr(exc), metrics=self.metrics())
        if actual:
            path = self._directory / "encoder-failure-outputs.npz"
            if not path.exists():
                with path.open("xb") as stream:
                    np.savez(stream, **actual)
                failure["raw_output_artifact"] = dict(file=path.name, bytes=path.stat().st_size,
                    sha256=sha(path), arrays={k: _array_info(v) for k, v in actual.items()})
        path = self._directory / "encoder-failure.json"
        if not path.exists():
            path.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")

    def encode(self, image_bgr, *, centers=None, generator=None, patch_count=96,
               inverse_depths=None, return_colors=False):
        with self._lock:
            require(not self._closed and self._context is not None, "Encoder is closed")
            require(isinstance(image_bgr, np.ndarray) and image_bgr.dtype == np.uint8
                    and image_bgr.shape == (464, 608, 3), "Fixed cropped BGR uint8 required")
            require(centers is None or generator is None, "Provide centers or generator, not both")
            begin = time.monotonic()
            normalized = self._encoder.normalize_bgr(image_bgr)
            normalized_end = time.monotonic()
            actual, timing, native, failure_exc = None, None, None, None
            try:
                with self._log_helper._native_log(self._directory) as native:
                    actual, timing = self._context.execute(normalized, None, self._activity)
                require(timing.get("output_buffers_released") is True and
                        timing.get("synchronous") is True, "Complete synchronous copy/release required")
                _native_ok(native)
                require(set(actual) == set(OUTPUT_SHAPES), "Both dense outputs required")
                for name, value in actual.items():
                    require(isinstance(value, np.ndarray) and value.dtype == np.float32
                            and value.shape == OUTPUT_SHAPES[name]
                            and np.isfinite(value).all() and np.any(value != 0),
                            "Nonfinite, all-zero or incompatible dense output: " + name)
                sdk_end = time.monotonic()
                if centers is None:
                    centers = self._encoder.random_centers(152, 116, patch_count, generator)
                result = self._encoder.features_to_patches(actual["fmap"], actual["imap"], centers,
                    inverse_depths=inverse_depths, normalized_bgr=normalized[0] if return_colors else None)
                end = time.monotonic()
                values = dict(normalization_seconds=normalized_end-begin,
                              SDK_outer_seconds=sdk_end-normalized_end,
                              sampling_seconds=end-sdk_end, total_encode_seconds=end-begin)
                result["timing"] = values
                # Caller uses these only for the first-three dense CPU reference
                # check, then discards them before process_encoded; no extra copy.
                result["encoder_dense_outputs"] = actual
                self._metrics["completed_frames"] += 1
                self._metrics["successful_sdk_frames"] += 1
                self._metrics["last_patch_count"] = len(centers)
                self._metrics["last_frame_timings"] = dict(values)
                for key, value in values.items():
                    self._metrics[key] += value
                return result
            except Exception as exc:
                failure_exc = exc
                self._metrics["last_error"] = repr(exc)
                self.close()
                raise
            finally:
                self._metrics["sdk_run_api_calls"] = self._activity["sdk_run_api_calls"]
                self._metrics["NPU_used"] = self._activity["sdk_run_api_calls"] > 0
                self._metrics["calls"].append(dict(index=len(self._metrics["calls"]),
                    SDK_timing=copy.deepcopy(timing), native_log=copy.deepcopy(native),
                    completed=self._metrics.get("last_error") is None))
                if failure_exc is not None:
                    self._save_failure(failure_exc, actual)

    def check_sources(self):
        with self._lock:
            require(all(sha(path) == digest for path, digest in self._pins.items()),
                    "Frozen encoder source/model/ancestor changed")
            self._metrics["source_recheck_passed"] = True
            return dict(self._pins)

    def metrics(self):
        with self._lock:
            value = copy.deepcopy(self._metrics)
            value["sdk_run_api_calls"] = self._activity["sdk_run_api_calls"]
            value["NPU_used"] = self._activity["sdk_run_api_calls"] > 0
            value["context"] = copy.deepcopy(self._context_record)
            value["cleanup_errors"] = list(self._cleanup)
            value["source_pins"] = dict(self._pins)
            return value

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._metrics["closed"] = True
            if self._context is not None:
                with self._log_helper._native_log(self._directory) as native:
                    self._context.close()
                self._metrics["close_native_log"] = copy.deepcopy(native)
                if native.get("RKNN_error_detected") or native.get("truncated"):
                    self._cleanup.append("RKNN destroy native log failed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
