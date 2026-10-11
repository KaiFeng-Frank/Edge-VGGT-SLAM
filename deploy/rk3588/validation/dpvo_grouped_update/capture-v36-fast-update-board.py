#!/usr/bin/env python3
"""Board-only read-only capture around the frozen v35 FAST original Update.

This launcher instruments, never replaces, the original neural callback. Exactly
one original call supplies the gold. Extra SHA/copy/save work makes the parent
replay an input-capture diagnostic, not an ordinary performance benchmark.
Derived DPVO lifecycle references retain the adjacent Princeton MIT LICENSE.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
STREAM_SHA256 = "295233513c1b670c953db25bbf4bfad8bdb72b2001f40ac3e1bf5d17083f9bde"
PARENT_SHA256 = "84b8f02e14c0695c79d5c47319ff47ec4c7a8dcf53a3b38d905e03bc1efde6a8"
BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def meta(x):
    return dict(shape=list(x.shape), dtype=str(x.dtype), bytes=x.nbytes,
                tensor_sha256=hashlib.sha256(x.tobytes(order="C")).hexdigest())


def digest_arrays(values):
    return hashlib.sha256(json.dumps({k: meta(v) for k, v in values.items()},
                                    sort_keys=True).encode()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CaptureRecorder:
    """Read-only selected complete Update inputs, not feature/model inference."""
    def __init__(self, output_dir, observations, *, patches=48, channels=384,
                 corr_channels=882, max_total_bytes=512 * 1024 * 1024):
        require(all(type(x) is int and x >= 0 for x in observations)
                and len(set(observations)) == len(observations) and observations,
                "Unique nonnegative observation capture plan required")
        require(all(type(v) is int and v > 0 for v in
                    (patches, channels, corr_channels, max_total_bytes)),
                "Positive capture dimensions and byte budget required")
        self.output_dir = Path(output_dir)
        self.selected = tuple(observations)
        self.patches, self.channels, self.corr_channels = patches, channels, corr_channels
        self.byte_limit = max_total_bytes
        self.saved_bytes = 0
        self.records = []
        self.original_calls = 0
        self.original_successes = 0
        self.fingerprint_seconds = 0.
        self.step_ordinals = {}
        self.last_digest = {k: None for k in ("graph", "geometry", "features")}
        self.serial = {k: 0 for k in self.last_digest}
        self.instances = 0

    def _revision(self, name, digest):
        if self.last_digest[name] != digest:
            self.serial[name] += 1
            self.last_digest[name] = digest
        return dict(revision=self.serial[name], sha256=digest)

    def _save(self, name, arrays):
        bytes_needed = sum(v.nbytes for v in arrays.values())
        require(self.saved_bytes + bytes_needed + 65536 <= self.byte_limit,
                "Explicit total capture artifact byte budget exceeded")
        f = self.output_dir / name
        with f.open("xb") as stream:
            np.savez(stream, **arrays)
        self.saved_bytes += f.stat().st_size
        return dict(file=name, bytes=f.stat().st_size, sha256=sha(f),
                    arrays={k: meta(v) for k, v in arrays.items()})

    def invoke(self, original, state, probe, phase, inputs):
        self.original_calls += 1
        call_index = self.original_calls - 1
        if probe or state is None:
            result = original(*inputs)
            self.original_successes += 1
            return result
        net, inp, corr, ii, jj, kk = inputs
        t = time.monotonic()
        frames = state.frames
        contexts = np.stack([f["context"] for f in frames])
        frame_ids = [dict(observation_id=int(f["observation_id"]), frame_id=f["frame_id"],
                          timestamp=float(f["timestamp"]),
                          captured_monotonic=float(f["captured_monotonic"])) for f in frames]
        graph = self._revision("graph", digest_arrays(dict(ii=ii, jj=jj, kk=kk)))
        geometry = self._revision("geometry", digest_arrays(dict(
            poses=state.poses, patches=state.patches, intrinsics=state.intrinsics)))
        features_digest = hashlib.sha256((digest_arrays(dict(contexts=contexts))
            + json.dumps(frame_ids, sort_keys=True)).encode()).hexdigest()
        features = self._revision("features", features_digest)
        self.fingerprint_seconds += time.monotonic() - t
        observation = int(frames[-1]["observation_id"])
        key = (phase, observation)
        ordinal = self.step_ordinals.get(key, 0)
        self.step_ordinals[key] = ordinal + 1
        capture = phase == "online" and observation in self.selected and ordinal == 0
        if not capture:
            result = original(*inputs)
            self.original_successes += 1
            return result
        capture_begin = time.monotonic()
        require(not any(r["observation_id"] == observation for r in self.records),
                "Selected observation already captured")
        require(net.dtype == inp.dtype == corr.dtype == np.float32
            and net.shape == inp.shape == (1, len(ii), self.channels)
            and corr.shape == (1, len(ii), self.corr_channels),
            "Actual original full Update input shape/dtype differs")
        require(all(x.dtype == np.int64 and x.shape == ii.shape for x in (ii, jj, kk))
                and np.array_equal(ii, kk // self.patches), "Actual graph ownership differs")
        require(np.array_equal(ii, state.ii) and np.array_equal(jj, state.jj)
                and np.array_equal(kk, state.kk), "Captured neural call not bound to current state graph")
        expected_context = np.ascontiguousarray(contexts.reshape(-1, self.channels)[kk][None])
        require(meta(expected_context) == meta(inp), "Actual original context[kk] differs")
        arrays = {k: np.array(v, order="C", copy=True)
                  for k, v in zip(("net", "inp", "corr", "ii", "jj", "kk"), inputs)}
        arrays.update(poses=np.array(state.poses, copy=True),
                      patches=np.array(state.patches, copy=True),
                      intrinsics=np.array(state.intrinsics, copy=True),
                      frame_contexts=np.array(contexts, copy=True),
                      retained_observation_ids=np.array([f["observation_id"] for f in frames], np.int64))
        require(all(np.isfinite(v).all() for v in arrays.values()),
                "Finite actual captured state and inputs required")
        original_input_meta = {k: meta(v) for k, v in arrays.items() if k in ("net", "inp", "corr", "ii", "jj", "kk")}
        pre_snapshot_seconds = time.monotonic() - capture_begin
        original_begin = time.monotonic()
        result = original(*inputs)  # Only original invocation; no repeat/reference fallback.
        original_seconds = time.monotonic() - original_begin
        self.original_successes += 1
        require(isinstance(result, dict) and set(result) == {"net", "delta", "weight"},
                "Original gold output names differ")
        require({k: meta(v) for k, v in zip(("net", "inp", "corr", "ii", "jj", "kk"), inputs)}
                == original_input_meta, "Original neural input mutated during capture")
        gold = {k: np.array(v, order="C", copy=True) for k, v in result.items()}
        require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in gold.values())
                and gold["net"].shape == net.shape
                and gold["delta"].shape == gold["weight"].shape == (1, len(ii), 2)
                and np.all((gold["weight"] >= 0) & (gold["weight"] <= 1)),
                "Original gold output finite/domain/shape differs")
        require(self.saved_bytes + sum(v.nbytes for v in arrays.values())
                + sum(v.nbytes for v in gold.values()) + 131072 <= self.byte_limit,
                "Explicit total capture artifact byte budget exceeded")
        prefix = "capture-%02d-obs-%d" % (len(self.records), observation)
        record = dict(observation_id=observation, source=frame_ids[-1], phase=phase,
            step_within_observation=ordinal, actual_original_call_index=call_index,
            patches_per_frame=self.patches, edges=len(ii), retained_frames=frame_ids,
            original_state_updates_before=int(state.updates),
            snapshot_identity=dict(graph=graph, geometry=geometry, features=features,
                recurrent_revision=call_index,
                revision_origin="capture-derived serials bound to actual arrays; not exposed upstream revision counters"),
            inputs_artifact=self._save(prefix + "-inputs.npz", arrays),
            original_gold_artifact=self._save(prefix + "-gold.npz", gold),
            original_callback_seconds=original_seconds,
            selected_pre_snapshot_copy_validation_seconds=pre_snapshot_seconds,
            selected_capture_overhead_seconds=time.monotonic() - capture_begin - original_seconds,
            original_called_once=True, callback_outputs_returned_unmodified=True,
            before_BA=True, fresh_pose_or_trajectory_validated=False)
        self.records.append(record)
        return result

    def decorate_stream(self, base):
        recorder = self
        class CapturedStream(base):
            def __init__(self, config=None, **kwargs):
                require(recorder.instances == 0, "Single original stream owner required")
                recorder.instances += 1
                original_update = kwargs["update"]
                self._capture_phase, self._capture_active = "outside", None
                def update(*args):
                    active = self._capture_active
                    require(active is not None, "Update not under original _neural context")
                    state, probe = active
                    return recorder.invoke(original_update, state, probe, self._capture_phase, args)
                kwargs["update"] = update
                super().__init__(config, **kwargs)
                require(self.config.patches_per_frame == recorder.patches, "Original M48 config required")
            def _neural(self, state, ii, jj, kk, net, *, probe):
                require(self._capture_active is None, "Unexpected nested original neural call")
                self._capture_active = (state, probe)
                try:
                    return super()._neural(state, ii, jj, kk, net, probe=probe)
                finally:
                    self._capture_active = None
            def process_encoded(self, record):
                self._capture_phase = "online"
                try:
                    return super().process_encoded(record)
                finally:
                    self._capture_phase = "outside"
            def terminate(self):
                self._capture_phase = "EOF"
                try:
                    return super().terminate()
                finally:
                    self._capture_phase = "outside"
        return CapturedStream


def require_board():
    require(platform.system() == "Linux" and platform.machine() == "aarch64",
            "Actual capture is board-only")
    require(b"rockchip,rk3588" in Path("/proc/device-tree/compatible").read_bytes(),
            "RK3588 device tree required")


def parent_output_dir(argv):
    values = []
    for i, argument in enumerate(argv):
        if argument == "--output-dir":
            require(i + 1 < len(argv), "Parent --output-dir needs a value")
            values.append(argv[i + 1])
        elif argument.startswith("--output-dir="):
            values.append(argument.split("=", 1)[1])
    require(len(values) == 1 and values[0], "Exactly one explicit parent --output-dir required")
    return Path(values[0]).resolve()


def verify_parent_receipt(folder, recorder):
    path = folder / "run.json"
    value = json.loads(path.read_text())
    require(value.get("schema") == "v35_DPVO_original_FAST_dense_spatial_RKNN_encoder_diagnostic_replay"
            and value.get("script_sha256") == PARENT_SHA256
            and value.get("status", "").startswith("completed_"), "Original parent receipt differs")
    require(value.get("cleanup_errors") == [] and value.get("sources_and_inputs_unchanged") is True
            and value.get("original_weights_unchanged") is True, "Original parent cleanup/source/weights failed")
    calls = value.get("update_calls", [])
    require(len(calls) == recorder.original_calls == recorder.original_successes,
            "Captured/original callback accounting differs")
    for record in recorder.records:
        index = record["actual_original_call_index"]
        require(index < len(calls) and calls[index]["index"] == index
                and calls[index]["edges"] == record["edges"], "Captured gold callback identity differs")
    manifest = folder / "input-manifest.json"
    require(sha(manifest) == value["input_manifest_sha256"], "Original input manifest SHA differs")
    source = json.loads(manifest.read_text())["frames"]
    for record in recorder.records:
        for frame in record["retained_frames"]:
            original = source[frame["observation_id"]]
            require(frame["frame_id"] == original["frame_id"] and frame["timestamp"] == original["timestamp"],
                    "Retained source identity differs from original ordered observations")
    return dict(run_file=str(path), run_sha256=sha(path), run_bytes=path.stat().st_size,
                input_manifest_sha256=sha(manifest), original_update_calls=len(calls),
                frozen_parent_status=value["status"], original_sources_and_weights_unchanged=True,
                parent_encoder_FP32_rejection_retained=value["historical_encoder_FP32_gate_passed"] is False,
                ordinary_performance_claim=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--parent-runner", type=Path, required=True)
    parser.add_argument("--parent-runner-sha256", required=True)
    parser.add_argument("--capture-output-dir", type=Path, required=True)
    parser.add_argument("--capture-observations", type=int, nargs="+", default=[13, 127, 191])
    parser.add_argument("--capture-max-total-bytes", type=int, default=512 * 1024 * 1024)
    args, rest = parser.parse_known_args(argv)
    if rest[:1] == ["--"]: rest = rest[1:]
    require_board()  # Before loading parent, creating output or reading real state.
    require(args.parent_runner_sha256 == PARENT_SHA256
            and sha(args.parent_runner) == PARENT_SHA256, "Frozen FAST runner SHA differs")
    parent_out = parent_output_dir(rest)
    out = args.capture_output_dir.resolve()
    require(out.is_relative_to(BASE) and out != BASE, "New isolated board capture output required")
    require(not out.exists() and not args.parent_runner.resolve().is_relative_to(out),
            "Capture output must be new and must not contain parent source")
    require(not parent_out.is_relative_to(out) and not out.is_relative_to(parent_out),
            "Original parent and auxiliary capture outputs must not overlap")
    out.mkdir()
    recorder = CaptureRecorder(out, args.capture_observations,
                               max_total_bytes=args.capture_max_total_bytes)
    parent, original_loader = None, None
    restored = []
    report = dict(schema="v36_full_M48_original_Update_capture", status="failed",
        script_sha256=sha(__file__), parent_runner_sha256=args.parent_runner_sha256,
        parent_stream_sha256=STREAM_SHA256, capture_observations=args.capture_observations,
        model_on_host=False, original_Update_replaced=False, original_weights_or_RNG_changed=False,
        timing_is_ordinary_performance=False, extra_SHA_copy_save_overhead_disclosed=True,
        EOF_captured_as_fresh=False, capture_context_is_before_BA=True,
        upstream_revision_counters_exposed=False, candidate_accepted=False,
        full_grouped_Update_or_NPU_candidate_executed=False, cleanup_errors=[])
    def module(path, name):
        loaded = original_loader(path, name)
        if name == "_v30_stream":
            require(sha(path) == STREAM_SHA256, "Original stream module SHA differs")
            restored.append((loaded, loaded.DPVOStream))
            loaded.DPVOStream = recorder.decorate_stream(loaded.DPVOStream)
        return loaded
    started = time.monotonic()
    try:
        parent = load(args.parent_runner, "_v36_frozen_v35_FAST")
        require(hasattr(parent, "STREAM_MODULE_SHA256") and parent.STREAM_MODULE_SHA256 == STREAM_SHA256,
                "Frozen stream source identity differs")
        original_loader = parent.module
        parent.module = module
        code = parent.main(rest)
        report["parent_exit_code"] = code
        require(code == 0, "Frozen parent replay failed; retain its receipt")
        require(set(r["observation_id"] for r in recorder.records) == set(args.capture_observations),
                "Requested complete source-observation snapshots not all observed")
        report["original_parent_receipt"] = verify_parent_receipt(parent_out, recorder)
        report["status"] = "completed_original_M48_capture_not_performance_or_candidate"
    except Exception as e:
        report["error"] = repr(e)
    finally:
        if parent is not None and original_loader is not None:
            parent.module = original_loader
        for loaded, original_class in restored:
            loaded.DPVOStream = original_class
        report.update(records=recorder.records, original_call_attempts=recorder.original_calls,
            original_call_successes=recorder.original_successes, captured=len(recorder.records),
            saved_artifact_bytes=recorder.saved_bytes,
            all_complete_call_fingerprint_seconds=recorder.fingerprint_seconds,
            original_loader_and_stream_class_restored=True,
            parent_runner_source_unchanged=sha(args.parent_runner) == args.parent_runner_sha256,
            elapsed_capture_launch_seconds=time.monotonic() - started)
        if not report["parent_runner_source_unchanged"]:
            report.update(status="failed", error="Frozen parent source changed")
        (out / "capture.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "captured", "error")}), flush=True)
    return 0 if report["status"].startswith("completed_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
