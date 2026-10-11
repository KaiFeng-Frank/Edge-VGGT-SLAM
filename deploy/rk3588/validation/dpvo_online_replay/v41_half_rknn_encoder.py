"""Diagnostic-only online half-resolution INT8 RKNN DPVO encoder adapter.

Reuses the unchanged v39 HalfContext, v34 typed SDK transport, v37 queried
attribute validator, and original v30 normalization/RANDOM patch sampler.
The original FP32 rejection is retained; this adapter does not adopt the model.
DPVO: copyright (c) 2022 Princeton Vision & Learning Lab, MIT license.
See the accompanying LICENSE. No neural execution or native library load occurs
on import. Real construction is restricted to RK3588/S Linux aarch64.
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


MODEL_SHA = "060d6bc6ae8f448c5e61c9eb65353c5141eac7e0198b7beebee023d322352952"
MODEL_BYTES = 1521328
COMPILE_SHA = "5fb3cc31867c1bb59905e5fe0dccd77d60a1586087dd96c8f12e06ce59d0127e"
PROBE_SHA = "bdf0feea251e478d761a7bf5c9635e46fee39cb0e9a9720b0949bb54b18e7c8a"
HALF_SDK_SHA = "8850b1c10572ecc5e8255b1c5c195bf2a8aceb467cc7250c0ef41f1bb2932bb6"
QUERY_SHA = "c61a377ede3d33b5a2412b1c7f526ea75deff2f18ed185f0f937792e41b1aca2"
TRANSPORT_SHA = "646f875d1b5eea561c0d6aa4d1b9078351014a39a62976ec895a814cd466b387"
ABI_SHA = "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608"
HEADER_SHA = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
LIBRARY_SHA = "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
SAMPLING_SHA = "fc6fc662613a720cfb779a360a63906a593a96ff08addecb9d01cfd8767ca7a8"
NATIVE_LOG_SHA = "322021ade54fb9035b7ce80652ec90169938a417fd2192528d28a805a83ed25e"
INPUT_SHAPE = (1, 3, 232, 304)
IMAGE_SHAPE = (232, 304, 3)
OUTPUT_SHAPES = {"fmap": (1, 128, 58, 76), "context": (1, 16, 384)}
PATCH_COUNT = 16


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
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def board_guard():
    require(platform.system() == "Linux" and platform.machine() == "aarch64",
            "Actual half RKNN encoder execution requires RK3588/S Linux aarch64")
    compatible = Path("/proc/device-tree/compatible").read_bytes().split(b"\0")
    require(b"rockchip,rk3588" in compatible or b"rockchip,rk3588s" in compatible,
            "Actual half RKNN encoder execution requires RK3588/S")


def _native_ok(record):
    require(record.get("truncated") is False, "Truncated SDK log cannot be checked")
    require(record.get("RKNN_error_detected") is False,
            "Native E RKNN error is fatal even when SDK returned zero")


def features_to_sparse_patches(fmap, context, centers, sampling):
    """Original integer 3x3 patch math with already gathered [1,M,D] context.

    No dense imap allocation, extra feature scaling, color/depth RNG, or neural
    execution. Sampling is the separately pinned original v30 module. Spatial
    coordinates and unit depth match its features_to_patches exactly. The SDK
    caller validates the fixed half-resolution output shapes before entry.
    """
    f = sampling._feature(fmap, "fmap")
    height, width = f.shape[-2:]
    require(isinstance(centers, np.ndarray) and centers.dtype == np.int64
            and centers.ndim == 2 and centers.shape[1] == 2 and len(centers) > 0,
            "centers must be nonempty INT64 [M,2]")
    require(bool(((centers[:, 0] >= 1) & (centers[:, 0] < width - 1)
                  & (centers[:, 1] >= 1) & (centers[:, 1] < height - 1)).all()),
            "Patch centers must be interior")
    count = len(centers)
    require(isinstance(context, np.ndarray) and context.dtype == np.float32
            and context.ndim == 3 and context.shape[:2] == (1, count)
            and context.shape[2] > 0 and np.isfinite(context).all(),
            "Sparse context must be finite FP32 [1,M,D]")
    delta = np.arange(-1, 2, dtype=np.float32)
    patches = np.empty((count, 3, 3, 3), dtype=np.float32)
    patches[:, 0] = centers[:, 0, None, None].astype(np.float32) + delta[None, None, :]
    patches[:, 1] = centers[:, 1, None, None].astype(np.float32) + delta[None, :, None]
    patches[:, 2] = np.float32(1)
    return dict(fmap=np.array(f, copy=True, order="C"),
                gmap=sampling._integer_patchify(f, centers, 1),
                context=np.array(context[0], copy=True, order="C"),
                patches=patches, centers=np.array(centers, copy=True, order="C"))


class BoardHalfRKNNEncoder:
    """Persistent serial M16 SDK context, matching v39 Torch encode contract.

    encode(image, generator=shared_cpu_generator, patch_count=16) returns owned
    fmap [128,58,76], gmap [16,128,3,3], context [16,384], patches [16,3,3,3],
    centers INT64 [16,2], and timing. Caller retains the same generator for later
    depth draws. No global affinity/thread settings, hidden warmup, or CPU model
    fallback. Actual new-model attributes are queried by frozen HalfContext.
    """
    def __init__(self, model_path, compile_receipt, probe_receipt, *, half_sdk,
                 query_helper, transport_helper, abi_helper, header, library,
                 encoder_module, native_log_helper, log_directory,
                 diagnostic_allow_fp32_rejected=False, source_files=()):
        require(diagnostic_allow_fp32_rejected is True,
                "Explicit diagnostic_allow_fp32_rejected=True is required")
        board_guard()  # Before file writes, helper loads, library or SDK init.
        self._lock = threading.RLock()
        self._context = None
        self._closed = False
        self._pins = {}
        self._cleanup = []
        self._activity = {"sdk_run_api_calls": 0}
        self._context_record = {}
        self._log_helper = None
        self._directory = Path(log_directory).resolve(strict=True)
        require(self._directory.is_dir(), "Existing isolated log directory required")
        self._data = dict(backend="v39_half_M16_INT8_RKNN_diagnostic",
            encoder_backend="v39_half_M16_INT8_RKNN_diagnostic",
            CPU_model_execution=False, CPU_encoder_fallback=False, NPU_used=False,
            diagnostic_only=True, diagnostic_allow_fp32_rejected=True,
            candidate_accepted=False, adopted=False, production_changed=False,
            historical_parent_FP32_gate_passed=False,
            historical_parent_rejection_retained=True, FP32_thresholds_relaxed=False,
            original_FP32_gate=dict(rtol=1e-4, atol=1e-5), output_clipping=False,
            original_checkpoint_weights=True, no_weights_changed=True,
            fixed_input_shape=list(INPUT_SHAPE), input_shape=[1, 1, 3, 232, 304],
            feature_outputs_scaled_by_four_in_graph=True,
            additional_feature_scaling=False, patches_per_frame=16,
            context_is_sparse=True, dense_imap_allocated=False,
            contexts_created=0, contexts_reused=True, collect_perf_detail=False,
            completed_frames=0, successful_sdk_frames=0,
            normalization_seconds=0., forward_seconds=0., sampling_seconds=0.,
            total_encode_seconds=0., calls=[], closed=False,
            RNG_semantics="Shared Torch CPU generator; original x16 then y16; depth draws remain with caller",
            array_ownership="Owned C-contiguous snapshots after SDK output copy/release",
            source_rechecks_inside_forward=False)
        try:
            self._pin(__file__, sha(__file__))
            sdk = load(self._pin(half_sdk, HALF_SDK_SHA), "_v41_frozen_half_sdk")
            self._sampling = load(self._pin(encoder_module, SAMPLING_SHA), "_v41_frozen_sampling")
            self._log_helper = load(self._pin(native_log_helper, NATIVE_LOG_SHA), "_v41_frozen_encoder_native_log")
            abi = load(self._pin(abi_helper, ABI_SHA), "_v41_frozen_encoder_abi")
            validator = load(self._pin(query_helper, QUERY_SHA), "_v41_frozen_encoder_attr_guard")
            transport = load(self._pin(transport_helper, TRANSPORT_SHA), "_v41_frozen_encoder_transport")
            compiled = json.loads(self._pin(compile_receipt, COMPILE_SHA).read_text())
            probe = json.loads(self._pin(probe_receipt, PROBE_SHA).read_text())
            sdk.unchanged(probe)
            require(probe["schema"] == "v39_DPVO_half_encoder_board_probe"
                    and probe["status"] == "completed_half_M16_CPU_teacher_and_ORT_diagnostic_not_VO"
                    and probe["script_sha256"] == sdk.PROBE_DRIVER_SHA
                    and probe["strict_gate_each_method"] == {"original": True, "reciprocal": True}
                    and probe["historical_parent_FP32_gate_passed"] is False
                    and probe["historical_parent_rejection_retained"] is True
                    and probe["candidate_accepted"] is False,
                    "Frozen half-size original-weight CPU teacher lineage required")
            model = self._pin(model_path, MODEL_SHA)
            require(model.stat().st_size == MODEL_BYTES, "Frozen half INT8 model byte count differs")
            sdk.validate_compile(compiled, sdk.COMPILER_SHA, probe, MODEL_SHA, MODEL_BYTES)
            self._pin(header, HEADER_SHA)
            self._pin(library, LIBRARY_SHA)
            entries = source_files.items() if isinstance(source_files, dict) else ((p, sha(p)) for p in source_files)
            for path, digest in entries:
                self._pin(path, digest)
            self._data.update(model_sha256=MODEL_SHA, model_bytes=MODEL_BYTES,
                compile_receipt_sha256=COMPILE_SHA, probe_receipt_sha256=PROBE_SHA,
                source_graph_sha256=compiled["input"]["sha256"],
                configuration=copy.deepcopy(compiled["configuration"]),
                half_sdk_sha256=HALF_SDK_SHA, transport_sha256=TRANSPORT_SHA,
                query_helper_sha256=QUERY_SHA, sampling_sha256=SAMPLING_SHA,
                native_runtime_sha256=LIBRARY_SHA, ABI_sha256=ABI_SHA,
                ABI_layout=abi.abi_record(), input_contract=dict(normalized_bgr=list(INPUT_SHAPE), patch_indices=[16]),
                output_contract={k: list(v) for k, v in OUTPUT_SHAPES.items()})
            api = abi.bind_api(ct.CDLL(str(Path(library).resolve())))
            with self._log_helper._native_log(self._directory) as native:
                self._context = sdk.HalfContext(api, abi, validator, transport, model, False,
                                                self._context_record, self._cleanup)
            self._data["initial_native_log"] = copy.deepcopy(native)
            _native_ok(native)
            self._data.update(contexts_created=1, initial_source_checks_passed=True)
        except Exception as exc:
            self._data["startup_error"] = repr(exc)
            if "native" in locals():
                self._data["initial_native_log"] = copy.deepcopy(native)
            self.close()
            self._save_failure(exc)
            raise

    def _pin(self, path, expected):
        path = Path(path).resolve(strict=True)
        require(sha(path) == expected, "Frozen encoder artifact changed: " + path.name)
        require(str(path) not in self._pins or self._pins[str(path)] == expected,
                "Conflicting encoder artifact pin")
        require(not path.is_relative_to(self._directory), "Log directory overlaps encoder source")
        self._pins[str(path)] = expected
        return path

    def _save_failure(self, exc, actual=None):
        try:
            record = dict(error=repr(exc), metrics=self.metrics())
            if actual:
                path = self._directory / "encoder-failure-outputs.npz"
                if not path.exists():
                    with path.open("xb") as stream:
                        np.savez(stream, **actual)
                    record["raw_output_artifact"] = dict(file=path.name, bytes=path.stat().st_size,
                        sha256=sha(path), arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                        nonfinite_elements=int(v.size - np.count_nonzero(np.isfinite(v)))) for k, v in actual.items()})
            path = self._directory / "encoder-failure.json"
            if not path.exists():
                with path.open("x") as stream:
                    stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        except Exception as save_exc:
            self._cleanup.append("failure artifact: " + repr(save_exc))

    def encode(self, image, *, generator, patch_count=16):
        with self._lock:
            require(not self._closed and self._context is not None, "Encoder is closed")
            require(isinstance(image, np.ndarray) and image.dtype == np.uint8
                    and image.shape == IMAGE_SHAPE and type(patch_count) is int
                    and patch_count == PATCH_COUNT, "Declared half-resolution M16 BGR input required")
            begin = time.monotonic()
            normalized = self._sampling.normalize_bgr(image)
            normalized_end = time.monotonic()
            # These are the only RNG calls in this adapter. Keep the stream's
            # shared generator, with original x16 then y16 before depth draws.
            height, width = OUTPUT_SHAPES["fmap"][-2:]
            centers = self._sampling.random_centers(width, height, patch_count, generator)
            indices = np.ascontiguousarray(centers[:, 1] * width + centers[:, 0])
            centers_end = time.monotonic()
            actual, timing, native, failure = None, None, None, None
            completed = False
            try:
                with self._log_helper._native_log(self._directory) as native:
                    actual, timing = self._context.execute(normalized, indices, self._activity)
                require(timing.get("output_buffers_released") is True
                        and timing.get("synchronous") is True, "Complete synchronous SDK copy/release required")
                _native_ok(native)
                require(set(actual) == set(OUTPUT_SHAPES), "Both sparse encoder outputs required")
                for name, value in actual.items():
                    require(isinstance(value, np.ndarray) and value.dtype == np.float32
                            and value.shape == OUTPUT_SHAPES[name]
                            and np.isfinite(value).all() and np.any(value != 0),
                            "Nonfinite, all-zero or incompatible sparse encoder output: " + name)
                sdk_end = time.monotonic()
                result = features_to_sparse_patches(actual["fmap"], actual["context"], centers, self._sampling)
                end = time.monotonic()
                values = dict(normalization_seconds=normalized_end - begin,
                    forward_seconds=sdk_end - centers_end,
                    sampling_seconds=(centers_end - normalized_end) + (end - sdk_end),
                    total_encode_seconds=end - begin)
                result["timing"] = values
                self._data["completed_frames"] += 1
                self._data["successful_sdk_frames"] += 1
                self._data["last_frame_timings"] = dict(values)
                for key, value in values.items():
                    self._data[key] += value
                completed = True
                return result
            except Exception as exc:
                failure = exc
                self._data["last_error"] = repr(exc)
                self.close()
                raise
            finally:
                self._data["calls"].append(dict(index=len(self._data["calls"]),
                    SDK_timing=copy.deepcopy(timing or self._activity.get("last_sdk_attempt")),
                    native_log=copy.deepcopy(native), completed=completed))
                if failure is not None:
                    self._save_failure(failure, actual)

    def check_sources(self):
        with self._lock:
            require(all(sha(path) == digest for path, digest in self._pins.items()),
                    "Frozen encoder source/model/lineage changed")
            self._data["source_recheck_passed"] = True
            return dict(self._pins)

    def metrics(self):
        with self._lock:
            data = copy.deepcopy(self._data)
            data.update(sdk_run_api_calls=self._activity["sdk_run_api_calls"],
                        NPU_used=self._activity["sdk_run_api_calls"] > 0,
                        context=copy.deepcopy(self._context_record),
                        cleanup_errors=list(self._cleanup), source_pins=dict(self._pins))
            return data

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._data["closed"] = True
            if self._context is not None:
                try:
                    with self._log_helper._native_log(self._directory) as native:
                        self._context.close()
                    self._data["close_native_log"] = copy.deepcopy(native)
                    _native_ok(native)
                except Exception as exc:
                    self._cleanup.append("context close/native log: " + repr(exc))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
