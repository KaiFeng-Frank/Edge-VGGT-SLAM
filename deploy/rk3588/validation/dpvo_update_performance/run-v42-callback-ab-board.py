#!/usr/bin/env python3
"""Board RKNN ABBA on three immutable saved states, without recurrent feedback.

DPVO Princeton University (2022), MIT; use the adjacent LICENSE. A is the frozen
v41 callback, B changes only its recorded-NPU-activity metadata query. Two
independent factories/caches use the same historical eight models. Saved NPU
inputs are not a CPU teacher. No Torch/ORT/BA forwards, performance profiling,
frontend, trajectory commit or change of the historical strict FP32 rejection.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np

sys.dont_write_bytecode = True
RUN_SHA256 = '78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b'
INDEX_SHA256 = '8c3f1c3b92e14a6742a984f750a5eb4c1bbf6397ca31440b96b0768333dc21fa'
FACTORY_SHA256 = 'e942d769dd02f37ad0e7eb154f44f1a48e7ab748044f7acff9f0f49f8f7be2c4'
CALLBACK_SHA256 = dict(A='c7caef0c76673cd4511aa1a4c8a8a5427bfee691e1964227d782d0d6f26554eb',
                       B='4152c78b9f5942115fd3f754744fff234c788550806296e16dfb53b062038054')
INPUT_NAMES = ('net','inp','corr','ii','jj','kk')
OUTPUT_NAMES = ('net','delta','weight')
OBSERVATIONS = (34,127,191)
MODEL_KEYS = frozenset(f'B{b}_D{d}_M16' for b in (1,2) for d in range(4,8))


def require(value, reason):
    if not value: raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1 << 20), b''): digest.update(value)
    return digest.hexdigest()


def metadata(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and sha(path) == expected, 'Pinned file SHA differs: '+str(path))
    prior = pins.get(str(path))
    require(prior is None or prior == expected, 'Conflicting source pin')
    pins[str(path)] = expected
    return path


def load_module(path, expected, label, pins):
    path = pin(path,expected,pins)
    name = '_v42_AB_'+label+'_'+uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None); raise
    return module


def board_guard():
    require(platform.system() == 'Linux' and platform.machine() == 'aarch64', 'RK3588 board only')
    require(b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588 device tree required')


def owned_inputs(values, edges):
    require(set(INPUT_NAMES) <= set(values), 'All six full Update inputs required')
    for name, channels in zip(INPUT_NAMES[:3],(384,384,882)):
        v = values[name]
        require(v.dtype == np.float32 and v.shape == (1,edges,channels) and np.isfinite(v).all(), 'Full FP32 input differs: '+name)
    for name in INPUT_NAMES[3:]:
        v = values[name]
        require(v.dtype == np.int64 and v.shape == (edges,), 'Ordered int64 topology differs: '+name)
    require(np.array_equal(values['ii'],values['kk']//16), 'Original M16 source/patch relation differs')
    require(sum(values[n].nbytes for n in INPUT_NAMES) <= 256 << 20, 'Snapshot exceeds original bounded size')
    result = {}
    for name in INPUT_NAMES:
        value = np.array(values[name],copy=True,order='C'); value.flags.writeable = False
        result[name] = value
    return result


def load_inputs(directory, pins):
    root = Path(directory).resolve(strict=True)
    run = json.loads(pin(root/'run.json',RUN_SHA256,pins).read_text())
    require(run['frames_completed'] == 192 and run['candidate_accepted'] is False
            and run['historical_SDK_strict_FP32_rejected'] is True
            and run['source_pins'] == run['source_hashes_after'] and run['cleanup_errors'] == [],
            'Frozen rejected-feedback source run required')
    captures = run['update_captures']
    require([r['frame_index'] for r in captures] == list(OBSERVATIONS), 'Fixed three observed captures required')
    states = []
    for row in captures:
        artifact = row['input']; name = artifact['file']
        require(Path(name).name == name and name == f'update-{row["frame_index"]:03d}-input.npz', 'Root capture filename differs')
        path = pin(root/name,artifact['sha256'],pins)
        require(path.stat().st_size == artifact['bytes'] and row['entire_active_graph'] is True
                and row['same_call_original_Torch'] is False and row['reference_supplied'] is False,
                'Full saved NPU-input provenance differs')
        with np.load(path,allow_pickle=False) as bundle:
            require(set(bundle.files) == set(artifact['arrays']), 'Capture artifact descriptor keys differ')
            raw = {n:np.array(bundle[n],copy=True,order='C') for n in bundle.files}
        for n,v in raw.items():
            require(metadata(v) == artifact['arrays'][n] and np.isfinite(v).all(), 'Original capture raw metadata differs: '+n)
        values = owned_inputs(raw,row['edges'])
        states.append(dict(observation_id=row['frame_index'],saved_update_index=row['update_index'],edges=row['edges'],
                           input_artifact=artifact,selected_inputs=list(INPUT_NAMES),
                           ignored_verified_artifact_keys=sorted(set(raw)-set(INPUT_NAMES)),values=values,
                           before={n:metadata(v) for n,v in values.items()}))
    return run,states


def schedule():
    result = []
    for observation in OBSERVATIONS:
        seen = dict(A=0,B=0)
        for cycle in range(2):
            for position,method in enumerate(('A','B','B','A')):
                repeat = seen[method]; seen[method] += 1
                result.append(dict(observation_id=observation,cycle=cycle,position=position,method=method,
                                   state_method_repeat_index=repeat,excluded_first_state_method_call=repeat == 0))
    return result


def validate_output(values, inputs, edges):
    require(isinstance(values,dict) and set(values) == set(OUTPUT_NAMES), 'Exactly three canonical full outputs required')
    for name,channels in zip(OUTPUT_NAMES,(384,2,2)):
        v = values[name]
        require(isinstance(v,np.ndarray) and v.dtype == np.float32 and v.shape == (1,edges,channels)
                and v.flags.c_contiguous and v.flags.owndata and np.isfinite(v).all(), 'Owned finite canonical output differs: '+name)
        require(all(not np.shares_memory(v, x) for x in inputs.values()), 'Output aliases immutable original input')
    require(((values['weight'] >= 0)&(values['weight'] <= 1)).all(), 'Unclipped weight domain failed')
    require(any(np.any(v) for v in values.values()), 'All-zero full output rejected')


def nested_timing(record, wall):
    require(record['status'] == 'completed_diagnostic_not_adopted' and record['route'] == 'full'
            and record['NPU_used'] is True and record['Torch_fallback'] is False and record['fullgraph_returned'] is True
            and record['candidate_accepted'] is False and record['strict_FP32_gate_passed'] is None,
            'Actual full RKNN diagnostic callback required')
    details = record['SDK_execution']; groups = details['groups']
    require(details['full_graph_returned'] is True and details['padding'] is False and details['dropped_edges'] is False
            and details['sequential_groups'] == len(groups) and set(record['model_keys']) <= MODEL_KEYS
            and record['model_keys'] == [g['model_key'] for g in groups],
            'Complete original groups and eight-model catalogue required')
    fields = ('forward_seconds','pack_seconds','input_set_seconds','run_api_seconds',
              'outputs_get_seconds','owned_copy_seconds','outputs_release_seconds')
    sums = {field:0.0 for field in fields}
    for group in groups:
        native = group['native_log']
        require(group['model_key'] in MODEL_KEYS and group['execution_status'] == 'completed' and group['mode'] == 'normal'
                and group['outputs_finite'] is True and group['weight_domain_valid'] is True
                and group['all_outputs_zero'] is False and group['outputs_released'] is True
                and group['timing']['synchronous'] is True
                and native['truncated'] is False and native['RKNN_error_detected'] is False, 'SDK output/release/log protocol failed')
        for field in fields:
            value = group['timing'][field]
            require(np.isfinite(value) and value >= 0, 'Finite SDK interval required'); sums[field] += value
    clock,full = record['callback_seconds'],details['full_graph_seconds']
    require(0 <= sums['forward_seconds'] <= full+1e-6 and full <= clock+1e-6 and clock <= wall+1e-6, 'Nested timing order differs')
    return dict(actual_callback_wall_seconds=wall,internal_callback_seconds=clock,full_graph_seconds=full,
                sum_SDK_intervals=sums,SDK_calls=len(groups),
                boundary='Nested clocks must not be added. Actual wall includes callback return/record-copy work; internal callback clock ends before normal metadata/record copies.')


def run_calls(owners, records, states, save, report, platform_sample=lambda: None):
    first = {}; by_observation = {s['observation_id']:s for s in states}
    report['calls'] = []
    for order,item in enumerate(schedule()):
        state = by_observation[item['observation_id']]; method = item['method']; values = state['values']
        record_count = len(records[method]); sample = platform_sample()
        started = time.monotonic()
        output = owners[method](*(values[n] for n in INPUT_NAMES))
        wall = time.monotonic()-started
        require(len(records[method]) == record_count+1, 'Exactly one callback metadata record required')
        raw_record = records[method][-1]
        artifact = save(f'call-{order:03d}-{method}-obs{item["observation_id"]:03d}.npz',output)
        validate_output(output,values,state['edges'])
        timing = nested_timing(raw_record,wall)
        raw_bytes = {n:v.tobytes(order='C') for n,v in output.items()}
        prior = first.setdefault(item['observation_id'],raw_bytes)
        equal = raw_bytes == prior
        row = dict(item,output_artifact=artifact,timing=timing,callback_record=raw_record,
                   bitwise_equal_first_A_for_state=equal,platform_sample_before=sample,
                   candidate_accepted=False,CPU_reference_supplied=False,original_strict_FP32_gate_passed=False)
        report['calls'].append(row)
        require(equal, 'A/B or repeat differs bytewise for identical immutable saved state')
        require(all(metadata(v) == state['before'][n] for n,v in values.items()), 'Callback mutated an original snapshot')
    report['A_B_all_saved_outputs_bitwise_equal'] = True
    report['warm_statistics'] = []
    for observation in OBSERVATIONS:
        summary = dict(observation_id=observation,methods={})
        for method in ('A','B'):
            selected = [r for r in report['calls'] if r['observation_id'] == observation and r['method'] == method
                        and not r['excluded_first_state_method_call']]
            summary['methods'][method] = dict(warm_calls=len(selected),
                actual_callback_wall_mean_seconds=float(np.mean([r['timing']['actual_callback_wall_seconds'] for r in selected])),
                internal_callback_mean_seconds=float(np.mean([r['timing']['internal_callback_seconds'] for r in selected])),
                full_graph_mean_seconds=float(np.mean([r['timing']['full_graph_seconds'] for r in selected])),
                sum_SDK_forward_mean_seconds=float(np.mean([r['timing']['sum_SDK_intervals']['forward_seconds'] for r in selected])))
        report['warm_statistics'].append(summary)


def closed_lifecycle(state, expected_calls=None):
    cache = state['cache']
    require(state['closed'] is True and cache['closed'] is True and cache['resident_contexts'] == 0
            and cache['current_DMA_bytes'] == cache['current_model_host_bytes'] == 0 and cache['cleanup_errors'] == [],
            'Callback/cache close or allocation release failed')
    histories = cache['histories']
    for row in histories:
        m = row['metrics']
        require(m['context_destroyed'] is True and m['closed'] is True and m['cleanup_errors'] == []
                and m['SDK_outputs_release_attempts'] == m['SDK_outputs_releases']
                and m['sources_and_inputs_unchanged'] is True, 'Historical context destroy/release/source check failed')
    sdk = sum(row['metrics']['sdk_run_api_calls'] for row in histories)
    released = sum(row['metrics']['SDK_outputs_releases'] for row in histories)
    if expected_calls is not None:
        require(state['failed'] is False and sdk == released == expected_calls
                and sum(row['metrics']['completed_calls'] for row in histories) == sdk, 'Expected SDK completion/release totals differ')
    return dict(contexts_destroyed=len(histories),sdk_run_api_calls=sdk,outputs_released=released,
                current_DMA_bytes=0,current_model_host_bytes=0,closed=True)


def platform_sample():
    sample = dict(monotonic=time.monotonic(),temperatures={},cpu_frequency_kHz={})
    for p in sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp')):
        try: sample['temperatures'][p.parent.name] = dict(type=(p.parent/'type').read_text().strip(),milli_celsius=int(p.read_text()))
        except (OSError,ValueError): pass
    for p in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_cur_freq')):
        try: sample['cpu_frequency_kHz'][p.parent.name] = int(p.read_text())
        except (OSError,ValueError): pass
    return sample


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('factory','config','callback-a','callback-b','inputs-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('factory','config','callback-a','callback-b'):
        parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--diagnostic-allow-fp32-rejected',action='store_true')
    args = parser.parse_args(argv)
    require(args.diagnostic_allow_fp32_rejected, 'Explicit rejected-FP32 diagnostic authorization required')
    board_guard()  # Before imports of runtime/factory/SDK or any output-directory write.
    out = args.output_dir.resolve(); require(not out.exists(), 'New isolated output directory required')
    out.mkdir(parents=True,exist_ok=False)
    started,prior = time.monotonic(),os.sched_getaffinity(0)
    owners,records,pins = {},dict(A=[],B=[]),{}
    report = dict(schema='v42_saved_three_state_callback_metadata_ABBA_diagnostic',status='failed_execution',
        script_sha256=sha(__file__),source_run_sha256=RUN_SHA256,compile_index_sha256=INDEX_SHA256,
        candidate_accepted=False,historical_SDK_strict_FP32_rejected=True,strict_FP32_equivalence_validated=False,
        CPU_teacher_used=False,recurrent_feedback=False,BA_executed=False,VO_state_committed=False,
        realtime_validated=False,collect_perf_detail=False,cleanup_errors=[],calls=[],
        schedule=schedule(),cold_definition='Exclude the first call of each method for each saved state, not a claim that every excluded call creates a context.',
        limits=['Fixed saved NPU-feedback states, no new observation/trajectory or CPU-gold equivalence test.',
                'ABBA ordering and observed temperature/frequency do not establish thermal control or statistical significance.',
                'Only metadata-query overhead changes; unchanged FP16 SDK numerical rejection remains visible.'])
    def save(name,values):
        path = out/name
        with path.open('xb') as stream: np.savez(stream,**values)
        return dict(file=name,bytes=path.stat().st_size,sha256=sha(path),arrays={n:metadata(v) for n,v in values.items()})
    def no_motion(*_args): raise RuntimeError('This fixed full-graph experiment cannot invoke a CPU motion probe')
    try:
        require(args.factory_sha256 == FACTORY_SHA256 and args.callback_a_sha256 == CALLBACK_SHA256['A']
                and args.callback_b_sha256 == CALLBACK_SHA256['B'], 'Only frozen factory and exact A/B callback pair supported')
        pin(__file__,report['script_sha256'],pins)
        config = json.loads(pin(args.config,args.config_sha256,pins).read_text())
        require(config['compile_index_sha256'] == INDEX_SHA256 and config.get('max_contexts',8) == 8
                and config['callback_sha256'] == CALLBACK_SHA256['A'], 'Unchanged old-eight factory configuration required')
        factory = load_module(args.factory,FACTORY_SHA256,'factory',pins)
        for method,path in (('A',args.callback_a),('B',args.callback_b)): pin(path,CALLBACK_SHA256[method],pins)
        run,states = load_inputs(args.inputs_dir,pins)
        for path,digest in run['source_pins'].items(): pin(path,digest,pins)
        report['input_captures'] = [{k:v for k,v in state.items() if k not in ('values','before')} for state in states]
        require({4,5,6,7} <= prior,'Four A76 CPUs required'); os.sched_setaffinity(0,{4,5,6,7})
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        for method,path in (('A',args.callback_a),('B',args.callback_b)):
            directory = out/('callback-'+method); directory.mkdir()
            selected = dict(config,callback_path=str(path),callback_sha256=CALLBACK_SHA256[method])
            owner = factory.open_board_online_from_index(**selected,failure_dir=directory,motion_probe=no_motion,
                diagnostic_allow_fp32_rejected=True,on_record=records[method].append)
            owners[method] = owner
            require(set(owner.cache.specs) == MODEL_KEYS and owner.route == 'full', 'Only full graph old-eight cache required')
            for p,digest in owner.source_pins.items(): pin(p,digest,pins)
        require(owners['A'] is not owners['B'] and owners['A'].cache is not owners['B'].cache,
                'Two independent persistent callback caches required')
        report['factory_sha256'] = FACTORY_SHA256; report['callback_sha256'] = dict(CALLBACK_SHA256)
        run_calls(owners,records,states,save,report,platform_sample)
        report['status'] = 'completed_AB_bitwise_equal_diagnostic_not_adopted'
    except BaseException as error:
        report.update(status='failed_execution',error=repr(error))
        if hasattr(error,'record'): report['callback_failure_record'] = copy.deepcopy(error.record)
    finally:
        report['final_callbacks'] = {}; report['lifecycle'] = {}
        for method,owner in owners.items():
            try: owner.close()
            except BaseException as error: report['cleanup_errors'].append(method+': '+repr(error))
            try:
                state = owner.metrics(); report['final_callbacks'][method] = state
                expected = sum(r['timing']['SDK_calls'] for r in report['calls'] if r['method'] == method) if report['status'].startswith('completed_') else None
                report['lifecycle'][method] = closed_lifecycle(state,expected)
            except BaseException as error: report['cleanup_errors'].append(method+': '+repr(error))
        try:
            after = {p:sha(p) for p in pins}
            report.update(source_pins=pins,source_hashes_after=after,sources_and_inputs_unchanged=after == pins)
            require(after == pins,'Frozen source/input/model changed')
        except BaseException as error: report['cleanup_errors'].append(repr(error))
        try:
            os.sched_setaffinity(0,prior);report['affinity_restored'] = os.sched_getaffinity(0) == prior
            require(report['affinity_restored'],'Affinity restore failed')
        except BaseException as error: report['cleanup_errors'].append(repr(error))
        if report['cleanup_errors']: report['status'] = 'failed_execution'
        report['actual_experiment_seconds'] = time.monotonic()-started
        with (out/'run.json').open('x') as stream: stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','A_B_all_saved_outputs_bitwise_equal','error','cleanup_errors')}),flush=True)
    return int(not report['status'].startswith('completed_'))


if __name__ == '__main__': raise SystemExit(main())
