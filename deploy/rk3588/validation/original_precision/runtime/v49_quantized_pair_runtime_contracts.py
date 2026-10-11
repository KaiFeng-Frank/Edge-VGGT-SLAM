"""Original VGGT pair provenance and logical eight-bit/floating RKNN IO.

This module only reads hashes, saved JSON/arrays and generated descriptors.
It imports no actual model, ORT, compiler or SDK runtime.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

BASE = Path('/mnt/vggt-data/vggt-slam2')
INPUT_SHAPE = (2,437,1024)
OUTPUT_SHAPE = (1,874,1024)
ELEMENTS = 894976
FP32_IO_BYTES = ELEMENTS * 4
RTOL, ATOL = 1e-4, 1e-5
PAIR_EXPORT_SHA = '149450ffc675ce59950c6f89cf84adb41e388cd050eeb731b6d09a6fe16e9fe1'
PAIR_GRAPH_SHA = '2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc'
PAIR_ORT_SHA = 'f93e1939be9628b7e28df8f91cf06d28002f0b3871a0d0a46adca1780ffa5fc0'
PAIR_CONTRACTS_SHA = '4ec6c0e827ab9669c942b589d2c504f0c9de306f76f4f4be0799af14ae322cd7'
PAIR_COMPILER_SHA = '459bbbb81aa43b3ca7562e1f9dc5c946086b276b89f68730123383df84e4ff8a'
PARENT_RUNTIME_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
SDK_LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
SDK_HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
NONQUANT_MODEL_SHA = '5ba3440634a51ae56fdc46eb1c07bf5847a30640a7c2603d322b44af585d037b'
NONQUANT_COMPILE_SHA = '5084e00143c4ab5fce94ceae81247c11939ddb7783d80d1b937bffcbdc0bcfb9'
PTQ_CONFIGURATION = {'target_platform':'rk3588','float_dtype':'float16','optimization_level':3,
                     'quantized_dtype':'w8a8','quantized_algorithm':'normal',
                     'quantized_method':'channel','do_quantization':True}
PTQ_INPUT_TRANSFORM = {'mean_values':None,'std_values':None,'quant_img_RGB2BGR':False,
                       'transpose':False,'extra_batch_dimension':False}
PTQ_BUILD_OPTIONS = {'do_quantization':True,'auto_hybrid':False}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def exact(left,right):
    """JSON contract equality also distinguishes a boolean from numeric 0/1."""
    return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)


def pin(path,expected,pins):
    path=Path(path).resolve(strict=True)
    if not valid_sha(expected) or not path.is_file() or sha(path)!=expected:
        raise ValueError(f'Frozen original pair SHA mismatch: {path}')
    pins[str(path)]=expected
    return path


def source(path,expected,pins):
    path=pin(path,expected,pins)
    spec=importlib.util.spec_from_file_location('_v49_private_'+path.stem.replace('-','_'),path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recheck(pins):
    after={path:sha(path) if Path(path).is_file() else None for path in pins}
    return after,after==pins


def raw_sha(value):
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


def validate_sdk_attr(record,direction):
    """Logical rank3 IO only. API conversion does not use manual quantization.

    Enum values match the independently SHA-pinned RKNN 2.3.2 header:
    FLOAT32=0, FLOAT16=1, INT8=2, UINT8=3; NONE=0, AFFINE=2;
    format UNDEFINED=3. Unsupported fixed-point/packed descriptors reject.
    """
    if direction not in ('input','output'):
        raise ValueError('Unknown original pair SDK IO direction')
    shape=INPUT_SHAPE if direction=='input' else OUTPUT_SHAPE
    fields=('index','n_elems','size','format','type','quantization_type','zero_point','pass_through')
    if any(type(record.get(k)) is not int for k in fields):
        raise ValueError('Actual SDK integer descriptor fields required')
    scale=record.get('scale')
    if (record['index']!=0 or record.get('name')!=direction or record.get('shape')!=list(shape)
            or record['n_elems']!=ELEMENTS or record['format']!=3 or record['pass_through']!=0
            or not isinstance(scale,(int,float)) or isinstance(scale,bool) or not math.isfinite(scale)):
        raise ValueError(f'Unsupported logical rank-three SDK descriptor: {record}')
    dtype,qnt=record['type'],record['quantization_type']
    if dtype in (0,1):
        identity=(dtype==1 and qnt==2 and record['zero_point']==0 and scale==1.0)
        if qnt!=0 and not identity:
            raise ValueError('Floating SDK IO requires NONE or exact FP16 identity affine')
        item_bytes=4 if dtype==0 else 2
        kind='floating'
    elif dtype in (2,3):
        low,high=(-128,127) if dtype==2 else (0,255)
        if qnt!=2 or scale<=0 or not low<=record['zero_point']<=high:
            raise ValueError('Eight-bit SDK IO requires positive finite affine scale and representable zero point')
        item_bytes=1
        kind='INT8_affine' if dtype==2 else 'UINT8_affine'
    else:
        raise ValueError('Unsupported SDK IO type: only FP32/FP16/INT8/UINT8 are defined here')
    if record['size']!=ELEMENTS*item_bytes:
        raise ValueError('SDK logical descriptor size differs from shape/type')
    return {'direction':direction,'descriptor_kind':kind,'logical_shape':list(shape),
            'logical_dtype_enum':dtype,'logical_element_bytes':item_bytes,
            'logical_bytes':record['size'],'quantization_type':qnt,
            'scale':scale,'zero_point':record['zero_point'],
            'API_input_type':'FP32','API_input_pass_through':0,
            'API_output_want_float':1,'API_FP32_bytes':FP32_IO_BYTES,
            'manual_quantization_or_dequantization':False,
            'SDK_owns_conversion':True,'native_stride_memory_not_accessed':True,
            'output_float_size_checked_again_after_outputs_get':True}


def validate_inputs(value,reference):
    import numpy as np
    if (value.shape!=INPUT_SHAPE or reference.shape!=OUTPUT_SHAPE
            or value.dtype!=np.float32 or reference.dtype!=np.float32
            or value.nbytes!=FP32_IO_BYTES or reference.nbytes!=FP32_IO_BYTES
            or not value.flags.c_contiguous or not reference.flags.c_contiguous
            or not np.isfinite(value).all() or not np.isfinite(reference).all()):
        raise ValueError('Finite contiguous original pair FP32 input/reference with exact logical shapes required')


def compare(actual,reference):
    import numpy as np
    if (actual.dtype!=np.float32 or reference.dtype!=np.float32 or actual.shape!=reference.shape
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError('Finite FP32 output and exact original reference shape required')
    a,r=actual.astype(np.float64),reference.astype(np.float64)
    delta=a-r;absolute=np.abs(delta);tolerance=ATOL+RTOL*np.abs(r)
    rn,an=float(np.linalg.norm(r)),float(np.linalg.norm(a))
    return dict(finite=True,array_equal=bool(np.array_equal(actual,reference)),
                max_abs_error=float(absolute.max()),rmse=float(np.sqrt(np.mean(delta*delta))),
                relative_l2=float(np.linalg.norm(delta)/rn) if rn else None,
                cosine_similarity=float(np.sum(a*r)/(rn*an)) if rn and an else None,
                rtol=RTOL,atol=ATOL,strict_FP32_gate_passed=bool(np.all(absolute<=tolerance)),
                outside_tolerance_elements=int(np.count_nonzero(absolute>tolerance)),
                max_tolerance_ratio=float(np.max(absolute/tolerance)),output_raw_sha256=raw_sha(actual))


def check_quantized_compile(row,compiler_sha,capture_sha,model_sha,
                            compiler_contracts_sha=None,capture_contracts_sha=None):
    """Require the new completed board PTQ receipt; old nonquant receipt rejects."""
    if not all(valid_sha(h) for h in (compiler_sha,capture_sha,model_sha)):
        raise ValueError('Exact new compiler/capture/model SHAs required')
    expected={'schema':'v49_original_VGGT_pair_board_PTQ_compile',
              'status':'compiled_not_runtime_validated','script_sha256':compiler_sha,
              'pair_export_sha256':PAIR_EXPORT_SHA,'pair_graph_sha256':PAIR_GRAPH_SHA,
              'passed_board_ORT_receipt_sha256':PAIR_ORT_SHA,'capture_manifest_sha256':capture_sha,
              'calibration_executed_on_board_only':True,'calibration_executed':True,
              'model_inference_on_host':False,'candidate_accepted':False,'released':True,
              'runtime_inference_API_called':False,'production_python_excluded':True,
              'calibration_build_api_invoked':True,'calibration_may_have_executed':True,
              'source_and_inputs_unchanged':True,'cwd_restored':True,'environment_restored':True,
              'configuration':PTQ_CONFIGURATION,'input_transform':PTQ_INPUT_TRANSFORM,
              'build_options':PTQ_BUILD_OPTIONS,'cleanup_errors':[]}
    if any(not exact(row.get(k),v) for k,v in expected.items()):
        raise ValueError('New original-pair board-only PTQ compilation contract mismatch')
    if ('release_code' not in row or not (row['release_code'] is None or
            type(row['release_code']) is int and row['release_code']==0)):
        raise ValueError('Actual successful toolkit release required')
    for key,expected_sha in (('contracts_sha256',compiler_contracts_sha),
                             ('capture_contracts_sha256',capture_contracts_sha)):
        if expected_sha is not None and (not valid_sha(expected_sha) or row.get(key)!=expected_sha):
            raise ValueError('Actual compilation used other source contracts')
    if row.get('pair_contracts_sha256')!=PAIR_CONTRACTS_SHA:
        raise ValueError('Frozen original pair source contracts required')
    board=row.get('board',{})
    if (board.get('machine','').lower() not in ('aarch64','arm64')
            or not isinstance(board.get('device_tree_compatible'),str)
            or 'rk3588' not in board['device_tree_compatible']
            or not isinstance(board.get('python'),str) or not board['python'].startswith('3.11.')
            or any(row.get(k)!=board.get(k) for k in ('machine','device_tree_compatible','python'))):
        raise ValueError('Actual isolated CPython3.11 RK3588 board compilation identity required')
    if (row.get('versions',{}).get('rknn-toolkit2')!='2.3.2'
            or row.get('versions',{}).get('torch','').split('+')[0]!='2.2.0'):
        raise ValueError('Frozen RKNN2.3.2 / Torch2.2.0 board compiler required')
    if not valid_sha(row.get('capture_script_sha256')):
        raise ValueError('Actual original capture script identity required')
    output=row.get('output',{})
    if (output.get('file')!='pair-int8.rknn' or output.get('sha256')!=model_sha
            or type(output.get('size')) is not int or not 0<output['size']<=256*1024*1024):
        raise ValueError('Quantized compiled original pair artifact identity mismatch')
    ops=row.get('operations',[])
    if ([r.get('name') for r in ops]!=['config','load_onnx','build','export_rknn']
            or any(r.get('status')!='completed' or 'code' not in r
                   or not (r['code'] is None or type(r['code']) is int and r['code']==0)
                   or not isinstance(r.get('seconds'),(int,float)) or isinstance(r['seconds'],bool)
                   or not math.isfinite(r['seconds']) or r['seconds']<0 for r in ops)):
        raise ValueError('All four actual board compiler operations must complete')
    dataset=row.get('dataset',{})
    rows=dataset.get('rows',[])
    if (dataset.get('file')!='dataset.txt' or not valid_sha(dataset.get('sha256'))
            or type(dataset.get('bytes')) is not int or dataset['bytes']<=0
            or dataset.get('input_count')!=1 or dataset.get('line_count')!=8 or len(rows)!=8):
        raise ValueError('Exactly eight original calibration input records required')
    ids=[r.get('sample_id') for r in rows]
    heldout=row.get('heldout_sample_ids',[])
    if (any(not isinstance(i,(str,int)) or isinstance(i,bool) for i in ids+heldout)
            or len(set(ids))!=8 or len(heldout)!=2 or len(set(heldout))!=2 or set(ids)&set(heldout)
            or any(not valid_sha(r.get('input_sha256')) or not valid_sha(r.get('input_raw_sha256'))
                   or not isinstance(r.get('path'),str) or not r['path'] for r in rows)):
        raise ValueError('Calibration and heldout identities must be explicit and disjoint')
    before=row.get('source_and_input_hashes',{})
    if (not before or before!=row.get('source_and_input_hashes_after')
            or any(not isinstance(path,str) or not path or not valid_sha(digest) for path,digest in before.items())
            or not {compiler_sha,capture_sha,PAIR_EXPORT_SHA,PAIR_GRAPH_SHA,PAIR_ORT_SHA}<=set(before.values())):
        raise ValueError('Completed board compilation source/input rehash required')
    return row


def bind_compilation_to_capture(compiled,captured,samples):
    """Bind all calibration rows to the frozen capture, without executing it."""
    if (not exact(samples,captured.get('samples')) or not exact(compiled.get('capture_samples'),samples)
            or compiled.get('capture_script_sha256')!=captured.get('script_sha256')
            or captured.get('checkpoint',{}).get('sha256')!=CHECKPOINT_SHA
            or compiled.get('original_checkpoint',{}).get('sha256')!=CHECKPOINT_SHA
            or compiled.get('original_checkpoint',{}).get('bytes')!=5026874952):
        raise ValueError('PTQ receipt and actual original capture lineage differ')
    calib=[s for s in samples if s.get('split')=='calibration']
    evals=[s for s in samples if s.get('split')=='evaluation']
    if len(samples)!=10 or len(calib)!=8 or len(evals)!=2:
        raise ValueError('Use the unchanged 8 calibration / 2 evaluation capture')
    dataset=compiled['dataset']
    if (compiled['heldout_sample_ids']!=[s['sample_id'] for s in evals]
            or dataset.get('heldout_samples_excluded')!=compiled['heldout_sample_ids']):
        raise ValueError('Exactly the two original heldout pairs must be excluded from calibration')
    for row,sample in zip(dataset['rows'],calib):
        item=sample['artifacts']['input.npy']
        suffix=Path(item['file']).parts
        path=Path(row['path'])
        if (row['sample_id']!=sample['sample_id'] or row['input_sha256']!=item['sha256']
                or row['input_raw_sha256']!=item['raw_sha256'] or row.get('input_shape')!=list(INPUT_SHAPE)
                or not path.is_absolute() or not path.is_relative_to(BASE/'experiments')
                or path.parts[-len(suffix):]!=suffix or any(c.isspace() for c in row['path'])
                or item['shape']!=list(INPUT_SHAPE) or item['dtype']!='float32'
                or item['tensor_bytes']!=FP32_IO_BYTES):
            raise ValueError('Each calibrated row must be its unchanged original captured tensor')
    return evals


def check_nonquantized_compile(row):
    """The SHA-pinned v48 nonquant candidate is a comparator, never a reference."""
    expected={'schema':'v47_original_VGGT_pair_RKNN_compilation',
              'status':'compiled_not_runtime_validated','script_sha256':PAIR_COMPILER_SHA,
              'contracts_sha256':PAIR_CONTRACTS_SHA,'pair_export_sha256':PAIR_EXPORT_SHA,
              'ORT_receipt_sha256':PAIR_ORT_SHA,'model_inference_executed':False,
              'candidate_accepted':False,'cleanup_errors':[],'released':True,
              'source_and_inputs_unchanged':True,'board_ORT_gate_passed':True,
              'configuration':{'target_platform':'rk3588','float_dtype':'float16',
                               'optimization_level':3,'do_quantization':False}}
    if (any(not exact(row.get(k),v) for k,v in expected.items())
            or row.get('input',{}).get('sha256')!=PAIR_GRAPH_SHA
            or row.get('output',{}).get('sha256')!=NONQUANT_MODEL_SHA
            or row.get('output',{}).get('size')!=55280948
            or row.get('source_hashes')!=row.get('source_hashes_after') or not row.get('source_hashes')):
        raise ValueError('Frozen original nonquantized pair compile receipt required')
    ops=row.get('operations',[])
    if ([r.get('name') for r in ops]!=['config','load_onnx','build','export_rknn']
            or any('code' not in r or type(r['code']) is not int or r['code']!=0 for r in ops)):
        raise ValueError('Frozen v48 completed compiler operations required')
    # The frozen historical schema did not contain per-operation status. Its
    # complete receipt SHA is fixed; do not invent v49 status metadata for it.
    return row
