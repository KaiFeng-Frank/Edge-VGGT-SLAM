#!/usr/bin/env python3
"""Run the unchanged original-model entrypoint with an explicit ownership contract.

Only installs the optional guard policy after every original backend and cache
is ready. This does not alter the model, graphs, inputs, precision or SL4.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
ENTRYPOINTS = {
    'deploy/rk3588/validation/original_full_aggregator/compare_complete_model.py':
        '75b81c6b8dd2c9fcb96f76bdb5fecb5d15e5c7794a161deee8aa4f323d020e7c',
    'scripts/replay_original_vggt.py':
        'eb793e40e126753c4c4714547b0cae9e102c4f3da7f93d6071dbb8474f3821e4',
}
CACHE_SHA = '3a8cc8c5a68c3bb934969e86053538f0d2026da1846274f4e6085297474a1e4b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('baseline', 'immutable'), required=True)
    parser.add_argument('--policy-source', type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--entrypoint', choices=tuple(ENTRYPOINTS), required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
    assert not args.receipt.exists()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    entrypoint = BASE/'src'/args.entrypoint
    assert sha(entrypoint) == ENTRYPOINTS[args.entrypoint]
    assert sha(args.policy_source) == args.policy_sha256
    assert sha(BASE/'src/vggt_slam/patch_cache.py') == CACHE_SHA
    spec = importlib.util.spec_from_file_location('_original_immutable_policy', args.policy_source)
    policy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = policy
    spec.loader.exec_module(policy)
    sys.path.insert(0, str(BASE/'src'))
    import vggt_slam.patch_cache as cache_module
    original_installer = cache_module.install_patch_cache
    records = []
    report = dict(schema='v77_original_immutable_execution_entrypoint', status='running',
        mode=args.mode, original_model_only=True, model_graph_or_weights_changed=False,
        full_mutable_model_guard_semantics_preserved=args.mode == 'baseline',
        exclusive_immutable_model_contract=args.mode == 'immutable',
        entrypoint=args.entrypoint, entrypoint_sha256=sha(entrypoint),
        script_sha256=sha(__file__), policy_source_sha256=args.policy_sha256,
        cache_source_sha256=CACHE_SHA, cleanup_errors=[], sessions=[])
    old_argv = sys.argv
    started = time.monotonic()

    def install(model, *positional, **keywords):
        cache = original_installer(model, *positional, **keywords)
        if args.mode == 'baseline':
            records.append(dict(cache=cache, session=None))
            return cache
        session = None
        try:
            session = policy.install_immutable_session(model, cache,
                model.aggregator._vggt_rknn_blocks_owner,
                model.aggregator.patch_embed._vggt_rknn_dino_owner,
                contract='exclusive_immutable_model')
            original_close, original_metrics = cache.close, cache.metrics
            record = dict(cache=cache, session=session, original_close=original_close,
                original_metrics=original_metrics, close_called=False)
            records.append(record)

            def close():
                if record['close_called']:
                    return original_close()
                record['close_called'] = True
                try:
                    session.close()
                finally:
                    # Closing SDK owners must remain possible even after a
                    # failed final model validation.
                    original_close()

            def metrics():
                result = original_metrics()
                result['immutable_execution_session'] = session.metrics()
                return result

            cache.close, cache.metrics = close, metrics
            return cache
        except BaseException:
            try:
                if session is not None:
                    session.close()
            finally:
                cache.close()
            raise

    try:
        cache_module.install_patch_cache = install
        child_args = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        sys.argv = [str(entrypoint), *child_args]
        try:
            runpy.run_path(str(entrypoint), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
            report['entrypoint_exit_code'] = 0
        report['status'] = 'completed_original_entrypoint_pending_saved_output_review'
    except BaseException as exc:
        report.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        for record in reversed(records):
            try:
                record['cache'].close()
            except BaseException as exc:
                report['cleanup_errors'].append(type(exc).__name__ + ': ' + str(exc))
            session = record['session']
            if session is not None:
                report['sessions'].append(session.metrics())
            if 'original_close' in record:
                del record['cache'].close
                del record['cache'].metrics
        cache_module.install_patch_cache = original_installer
        sys.argv = old_argv
        report.update(installer_restored=cache_module.install_patch_cache is original_installer,
            entrypoint_unchanged=sha(entrypoint) == ENTRYPOINTS[args.entrypoint],
            policy_source_unchanged=sha(args.policy_source) == args.policy_sha256,
            cache_source_unchanged=sha(BASE/'src/vggt_slam/patch_cache.py') == CACHE_SHA,
            seconds=time.monotonic()-started)
        if report['cleanup_errors'] or not all(report[k] for k in
                ('installer_restored', 'entrypoint_unchanged', 'policy_source_unchanged', 'cache_source_unchanged')):
            report['status'] = 'failed_cleanup_or_identity'
        args.receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: report.get(k) for k in ('status', 'mode', 'cleanup_errors')}), flush=True)
    return 0 if report['status'].startswith('completed_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
