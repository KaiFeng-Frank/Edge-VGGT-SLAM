#!/usr/bin/env python3
"""Run the frozen full FAST replay with only its correlation cache replaced.

Actual neural execution remains on RK3588/S. The original NPU encoder's
failed FP32 gate stays visible and still requires its diagnostic CLI flag.
No model parameters, original Update, geometry, frame selection or RNG change.
DPVO-derived research helper: MIT; see adjacent LICENSE.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import traceback

PARENT_SHA = '84b8f02e14c0695c79d5c47319ff47ec4c7a8dcf53a3b38d905e03bc1efde6a8'
PARENT_LIBRARY_SHA = '829d1b42fa24d28b1e280b882c0a9e8556b0e7cdbe02f7489eeef15ba3581bca'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    for name in ('replay-driver', 'frame-cache-helper', 'frame-cache-library', 'frame-cache-build', 'frame-cache-validation'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('frame-cache-helper-sha256', 'frame-cache-library-sha256', 'frame-cache-build-sha256', 'frame-cache-validation-sha256'):
        parser.add_argument('--' + name, required=True)
    args, remaining = parser.parse_known_args(argv)
    if (platform.machine() != 'aarch64' or platform.system() != 'Linux' or
            not any(v in (b'rockchip,rk3588', b'rockchip,rk3588s')
                    for v in Path('/proc/device-tree/compatible').read_bytes().split(b'\0'))):
        raise ValueError('Full replay requires the RK3588/S board')
    expected = {args.replay_driver: PARENT_SHA,
                args.frame_cache_helper: args.frame_cache_helper_sha256,
                args.frame_cache_library: args.frame_cache_library_sha256,
                args.frame_cache_build: args.frame_cache_build_sha256,
                args.frame_cache_validation: args.frame_cache_validation_sha256}
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError('Frozen candidate source/artifact mismatch: ' + str(path))
    build = json.loads(args.frame_cache_build.read_text())
    if build.get('output', {}).get('sha256') != args.frame_cache_library_sha256:
        raise ValueError('Native library is not bound by the supplied build receipt')
    if (build.get('status') != 'compiled_and_synthetic_smoke_passed' or
            build.get('cleanup_errors') != [] or not build.get('sources_and_inputs_unchanged') or
            build.get('candidate_source_sha256', {}).get('dpvo_frame_corr.py') != args.frame_cache_helper_sha256):
        raise ValueError('A successful, source-matched native build is required')
    expected_parent = {
        'dpvo_corr.cpp': '3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990',
        'dpvo_corr.h': '1866c747265b401f8d29d896e5f634ac104f238704e5fe443aa683a7395ab932'}
    if build.get('parent_source_sha256') != expected_parent:
        raise ValueError('Native correlation parent arithmetic changed')
    validation = json.loads(args.frame_cache_validation.read_text())
    if (validation.get('status') != 'completed_saved_array_correlation_AB_probe' or
            validation.get('all_fixed_FP32_gates_passed') is not True or
            validation.get('old_new_bitwise_equal') is not True or
            validation.get('incremental_preparation_checks_passed') is not True or
            validation.get('candidate_build_sha256') != args.frame_cache_build_sha256 or
            validation.get('candidate_library_sha256') != args.frame_cache_library_sha256 or
            validation.get('cleanup_errors') != [] or
            validation.get('sources_and_inputs_unchanged') is not True):
        raise ValueError('Successful saved-real-array AB validation required')
    parent = module(args.replay_driver, '_v36_frozen_fast_replay')
    helper = module(args.frame_cache_helper, '_v36_incremental_corr')
    original = parent.CompactCorrelation
    created = []

    def construct(factory, library, patch_count):
        if sha(library) != PARENT_LIBRARY_SHA or patch_count != 48:
            raise ValueError('Frozen parent/M48 identity changed')
        value = helper.PerFrameCorrelation(args.frame_cache_library,
            library_sha256=args.frame_cache_library_sha256, patch_count=patch_count,
            threads=4, max_frames=64, max_bytes=512 * 1024 * 1024)
        created.append(value)
        return value

    out_index = remaining.index('--output-dir')
    output = Path(remaining[out_index + 1]).resolve()
    if output.exists():
        raise ValueError('New full-replay output required')
    parent.CompactCorrelation = construct
    status = 'failed'
    report = dict(schema='v36_full_FAST_immutable_frame_cache_wrapper',
                  status=status, script_sha256=sha(__file__), parent_driver_sha256=PARENT_SHA,
                  source_pins={str(p.resolve()): s for p, s in expected.items()},
                  original_weights_changed=False, original_Update_changed=False,
                  original_BA_changed=False, RNG_changed=False,
                  encoder_FP32_rejection_retained=True, realtime_validated=False,
                  host_model_execution=False, no_additional_capture_instrumentation=True)
    try:
        code = parent.main(remaining)
        parent_path = output / 'run.json'
        raw = parent_path.read_bytes()
        run = json.loads(raw)
        report['parent_raw_receipt_sha256'] = hashlib.sha256(raw).hexdigest()
        if code != 0 or not run['status'].startswith('completed_unpaced_') or run['cleanup_errors']:
            raise RuntimeError('Frozen replay failed: ' + str(run.get('error')))
        if len(created) != 1 or not created[0].metrics()['closed']:
            raise RuntimeError('Exactly one closed per-frame cache required')
        if created[0].metrics()['current_context_bytes'] != 0:
            raise RuntimeError('Native frame cache was not fully released')
        after = {str(p.resolve()): sha(p) for p in expected}
        report['source_hashes_after'] = after
        report['sources_unchanged'] = after == report['source_pins']
        if not report['sources_unchanged']:
            raise RuntimeError('Candidate source/artifact changed during replay')
        (output / 'parent-run.json').write_bytes(raw)
        run.update(schema='v36_DPVO_FAST_NPU_encoder_per_frame_native_cache_diagnostic_replay',
                   status='completed_unpaced_FAST_incremental_correlation_diagnostic_not_realtime',
                   wrapper_script_sha256=report['script_sha256'],
                   native_correlation_implementation='v36 immutable per-frame cache; frozen v27 arithmetic',
                   wrapper_source_pins=report['source_pins'], wrapper_sources_unchanged=True,
                   parent_raw_receipt_sha256=report['parent_raw_receipt_sha256'])
        parent_path.write_text(json.dumps(run, indent=2, allow_nan=False) + '\n')
        report.update(status='completed_full_replay_with_per_frame_cache',
                      replay_receipt_sha256=sha(parent_path), frames_completed=run['frames_completed'],
                      original_FP32_gate_passed=run['first_three_FP32_gate_passed'])
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        parent.CompactCorrelation = original
        report['parent_constructor_restored'] = parent.CompactCorrelation is original
        if output.is_dir():
            (output / 'wrapper.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'frames_completed')}), flush=True)
    return 0 if report['status'] == 'completed_full_replay_with_per_frame_cache' else 1


if __name__ == '__main__':
    raise SystemExit(main())
