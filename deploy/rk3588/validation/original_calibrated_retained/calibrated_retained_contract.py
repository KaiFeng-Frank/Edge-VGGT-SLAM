"""Pure metadata contract for diagnostic INT8 builds of original retained graphs."""
from pathlib import Path

PARENT_SHA = '771a9b82007d2feefd65ebbf3f4b0fb27070b79ff7f6d0faab1a8c866da90e43'
CAPTURE_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
COMPILER_SHA = 'fae89ae5de8434081cc581b364b94ad31d5e74ebde3e836b4b3ae4032eb9e251'
COMPILER_R2_SHA = '13061f83535dadc1b951604556df1f260c168cf86fe14868a8ec734fd507139e'
COMPILER_SHA_BY_GROUP = (COMPILER_SHA, COMPILER_R2_SHA, COMPILER_R2_SHA, COMPILER_R2_SHA)
SCHEMA = 'v80_original_VGGT_calibrated_retained_RKNN_manifest'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False,
    quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
RANGES = ((0, 4), (5, 11), (12, 17), (18, 23))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_compile(report, parent, capture, index, directory):
    """Validate saved metadata only; callers must pin JSON, NPY and RKNN files."""
    require(type(index) is int and 0 <= index < 4, 'Original group ordinal required')
    row, start = parent['blocks'][index], RANGES[index][0]
    require(report['schema'] == 'v80_original_VGGT_retained_INT8_compile'
        and report['status'] == 'compiled_not_quality_accepted'
        and report['script_sha256'] == COMPILER_SHA_BY_GROUP[index]
        and report['parent_manifest_sha256'] == PARENT_SHA and report['capture_sha256'] == CAPTURE_SHA
        and report['group_index'] == index and report['indices'] == row['indices']
        and report['parent_group'] == row and report['checkpoint'] == parent['checkpoint'] == capture['checkpoint']
        and report['source_hashes'] == parent['source_hashes']
        and report['graph_sha256'] == row['sha256']
        and report['graph'] == dict(file=str(Path(report['parent_manifest_directory']) / row['file']),
                                  bytes=row['bytes'], sha256=row['sha256'])
        and report['prior_check_receipt'] == row['check_receipt']
        and report['prior_FP16_compile_receipt'] == row['compile_receipt'], 'Original graph/parent/build lineage differs')
    require(report['original_model_only'] is True and report['original_ONNX_modified'] is False
        and report['ORT_rerun'] is False and report['candidate_accepted'] is False
        and report['quality_accepted'] is False and report['realtime_validated'] is False,
        'Diagnostic original-only precision scope required')
    require(report.get('compile_cpus', [4, 5, 6, 7]) == ([4, 5, 6, 7] if index == 0 else [4, 6]),
        'Frozen compiler CPU affinity differs')
    require(report['configuration'] == CONFIG
        and report['build_options'] == dict(do_quantization=True,
            dataset=str(Path(directory).resolve() / 'dataset.txt'), auto_hybrid=False), 'Exact W8A8 configuration required')
    operations = report['operations']
    require([o['name'] for o in operations] == ['config', 'load_onnx', 'build', 'export_rknn']
        and all(o['status'] == 'completed' and o['code'] in (None, 0) for o in operations)
        and report['released'] is True and report['release_code'] in (None, 0)
        and report['cleanup_errors'] == [] and report['environment_restored'] is True
        and report['affinity_restored'] is True and report['source_and_inputs_unchanged'] is True,
        'Completed build and successful cleanup/identity evidence required')
    group = next(g for g in capture['groups']['aggregator'] if g['index'] == start)
    calibration = report['calibration']
    require(calibration['group_start_index'] == start and calibration['samples'] == group['samples']
        and len(calibration['samples']) == 8
        and calibration['source_images'] == [s['images'] for s in capture['samples']]
        and calibration['capture']['file'] == 'capture.json'
        and calibration['capture']['sha256'] == CAPTURE_SHA
        and calibration['dataset']['file'] == 'dataset.txt'
        and calibration['dataset']['input_count'] == 1 and calibration['dataset']['line_count'] == 8,
        'Eight actual original FP32 group-start samples required')
    require(report['rknn']['file'] == 'group.rknn' and report['rknn']['bytes'] > 0
        and len(report['rknn']['sha256']) == 64, 'New local RKNN artifact required')
    return report['rknn']
