#!/usr/bin/env python3
"""Assemble four checked original retained INT8 builds; no model or SDK calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

BASE = Path('/mnt/vggt-data/vggt-slam2')
CONTRACT_SHA = '54445659a031f4109ed41f1e738658033cf12d20560bb1e29d1e3f320adc6746'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path, expected, watched):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, 'Frozen file SHA differs: ' + str(path))
    watched[path] = expected
    return path


def artifact(root, row, watched):
    root, relative = Path(root).resolve(strict=True), Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative artifact required')
    path = pin(root / relative, row['sha256'], watched)
    path.relative_to(root)
    require(path.stat().st_size == row['bytes'], 'Frozen artifact byte size differs')
    return path


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Existing directory containing compile0..3')
    parser.add_argument('--compiler-source', type=Path, required=True)
    parser.add_argument('--compiler-r2-source', type=Path, required=True)
    parser.add_argument('--contract-source', type=Path, default=Path(__file__).with_name('calibrated_retained_contract.py'))
    parser.add_argument('--manifest', type=Path, default=BASE / 'experiments/vggt-v67-retained-groups-r1/compiled.json')
    parser.add_argument('--capture', type=Path, default=BASE / 'experiments/vggt-v56-calibration-r1/capture.json')
    args = parser.parse_args()
    directory = args.output_dir.resolve(strict=True)
    require(not (directory / 'compiled.json').exists() and not (directory / 'assembly.json').exists(), 'New assembly outputs required')
    report = dict(schema='v80_original_retained_INT8_assembly', status='running', script_sha256=sha(Path(__file__)),
        contract_source_sha256=CONTRACT_SHA, checked_groups=[], model_ORT_SDK_execution=False, cleanup_errors=[])
    watched, failure = {}, None
    try:
        pin(Path(__file__), report['script_sha256'], watched)
        contract_path = pin(args.contract_source, CONTRACT_SHA, watched)
        spec = importlib.util.spec_from_file_location('_v80_calibrated_retained_contract', contract_path)
        C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
        pin(args.compiler_source, C.COMPILER_SHA, watched)
        pin(args.compiler_r2_source, C.COMPILER_R2_SHA, watched)
        parent_path = pin(args.manifest, C.PARENT_SHA, watched)
        capture_path = pin(args.capture, C.CAPTURE_SHA, watched)
        parent, capture = json.loads(parent_path.read_text()), json.loads(capture_path.read_text())
        manifest = dict(schema=C.SCHEMA, status='compiled_not_quality_accepted', candidate_accepted=False,
            quality_accepted=False, realtime_validated=False, original_model_only=True, compiled_count=4,
            parent_manifest_sha256=C.PARENT_SHA, parent_manifest_directory=str(parent_path.parent),
            capture_sha256=C.CAPTURE_SHA, calibration_directory=str(capture_path.parent),
            compiler_source_sha256=C.COMPILER_SHA, contract_source_sha256=CONTRACT_SHA,
            compiler_source_sha256_by_group=list(C.COMPILER_SHA_BY_GROUP),
            assembler_source_sha256=report['script_sha256'], checkpoint=parent['checkpoint'],
            configuration=C.CONFIG, all_original_24_pairs_retained=True, original_ONNX_modified=False,
            actual_SDK_descriptors_or_trajectory_quality_validated=False, blocks=[])
        for index, (start, end) in enumerate(C.RANGES):
            receipt_path = directory / ('compile%d' % index) / 'result.json'
            receipt = json.loads(receipt_path.read_text()); receipt_sha = sha(receipt_path)
            watched[receipt_path] = receipt_sha
            rknn = C.validate_compile(receipt, parent, capture, index, receipt_path.parent)
            require(receipt['parent_manifest_directory'] == str(parent_path.parent)
                and receipt['calibration_directory'] == str(capture_path.parent)
                and receipt['calibration']['capture']['bytes'] == capture_path.stat().st_size,
                'Compile parent/capture artifact root differs')
            artifact(receipt_path.parent, rknn, watched)
            dataset = artifact(receipt_path.parent, receipt['calibration']['dataset'], watched)
            expected_paths = [artifact(capture_path.parent, sample['artifact'], watched)
                              for sample in receipt['calibration']['samples']]
            require(dataset.read_text().splitlines() == [str(path) for path in expected_paths], 'Eight dataset row identities differ')
            metadata = dict(file=str(receipt_path.relative_to(directory)), bytes=receipt_path.stat().st_size, sha256=receipt_sha)
            manifest['blocks'].append(dict(index=index, indices=list(range(start, end + 1)), compile_receipt=metadata))
            report['checked_groups'].append(index); write(directory / 'assembly.json', report)
        require(all(sha(path) == digest for path, digest in watched.items()), 'Pinned source/input/artifact changed during assembly')
        manifest['source_and_inputs_unchanged'] = True
        write(directory / 'compiled.json', manifest)
        report.update(status='assembled_not_quality_accepted', compiled_manifest_sha256=sha(directory / 'compiled.json'),
                      source_and_inputs_unchanged=True)
    except BaseException as error:
        failure = error; report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
    finally:
        write(directory / 'assembly.json', report)
    if failure is not None:
        raise failure
    print(json.dumps(dict(status=report['status'], compiled_manifest_sha256=report['compiled_manifest_sha256'])), flush=True)


if __name__ == '__main__':
    main()
