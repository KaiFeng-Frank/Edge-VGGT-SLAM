#!/usr/bin/env python3
"""Counterfactual saved-output groups through original SL4; no model execution.

Mixed predictions are diagnostic only, not outputs of an executable model.
Unmixed endpoint homographies and poses must reproduce the original R111 runs.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

BASE=Path('/mnt/vggt-data/vggt-slam2')
ROOT=Path('/dev/shm/edge-vggt-v111-native')
SOURCE=BASE/'experiments/vggt-v111-source-r1'
CONSUMER_SHA='a0d37e54b1b1655bf4443f5fa80c07043a1ada9c12dbf1d0cb851b41dc7af271'
MANIFESTS={
 'host-fp32-r1/result.json':'37a7856a6037fe16b29dc601dcf488c12a513d457dcdca8bb2e747ed43f1f77c',
 'batch0-r1/result.json':'2eb6f63d2c832c92631a55e527f4b49e3bf0b197685f3bdea8f76008aada1081',
 'batch4-r1/result.json':'a71c820518bee44fea580da5aecf75ec5acfb55acb10f5f82c117c3089a5ee6a',
 'batch8-r1/result.json':'749684550c1226ec347373a1d7d44599403e66c882e6a18f9a8c3828c0428698',
 'batch12-r1/result.json':'71ca8391f2b39e9a5fbe48114e5966cdfb7544c70fd33ef1df7d7aa91e19fc97',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert platform.machine()=='aarch64'
    assert b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes()
    out=ROOT/'head-group-diagnosis-r1'
    out.mkdir(exist_ok=False)
    pins={}
    def pin(path,digest):
        path=path.resolve(strict=True)
        assert sha(path)==digest,str(path)
        pins[path]=digest
        return path
    source=pin(SOURCE/'replay_saved_native518_sl4.py',CONSUMER_SHA)
    spec=importlib.util.spec_from_file_location('frozen_consumer',source)
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    for relative,digest in frozen.SOURCE_PINS.items():pin(BASE/'src'/relative,digest)
    sys.path[:0]=[str(BASE/'src'),str(BASE/'src/third_party/vggt')]
    import numpy as np
    import torch
    from vggt_slam.solver import Solver
    from vggt_slam.model_execution import ExecutionCapabilities
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    initial_affinity=os.sched_getaffinity(0)
    os.sched_setaffinity(0,{4,5,6,7})
    started=time.monotonic()
    report=dict(status='running',script_sha256=sha(__file__),counterfactual_saved_outputs_only=True,
                model_loaded=False,NN_SDK_GPU_executed=False,performance_or_quality_accepted=False,
                system_clock_writes=False,routes=[],source_and_input_sha256={})
    try:
        records={'cpu':{},'native':{}}
        for relative,digest in MANIFESTS.items():
            path=pin(ROOT/relative,digest);receipt=json.loads(path.read_text())
            assert receipt['status'].startswith('completed_') and not receipt['cleanup_errors']
            kind='cpu' if relative.startswith('host-') else 'native'
            for row in receipt['pairs']:
                assert row['index'] not in records[kind]
                records[kind][row['index']]=(row,path.parent)
        assert all(set(rows)==set(range(16)) for rows in records.values())
        sources=receipt['source_images']
        for row in sources:pin(Path(row['file']),row['sha256'])
        for i in range(16):
            c=records['cpu'][i][0];n=records['native'][i][0]
            assert c['preprocessed_images_raw_sha256']==n['prepared_pair_raw_sha256']
            assert c['source_indices']==n['source_indices']==[32+24*i,56+24*i]
        routes=[('native',()),('cpu',tuple(frozen.SHAPES)),
                ('cpu_camera',('pose_enc','extrinsic','intrinsic')),
                ('cpu_depth_confidence',('depth','depth_conf')),
                ('cpu_depth',('depth',)),('cpu_confidence',('depth_conf',))]
        class Adapter:
            capabilities=ExecutionCapabilities('Explicit saved-output-group counterfactual, no inference',False)
            device='cpu'
            def __init__(self,cpu_keys):self.cpu_keys=cpu_keys;self.count=0
            def predict(self,images,*,compute_similarity=False):
                assert not compute_similarity and images.shape==(2,3,392,518)
                native_row=records['native'][self.count][0]
                assert hashlib.sha256(images.tobytes()).hexdigest()==native_row['prepared_pair_raw_sha256']
                prediction={}
                for name,shape in frozen.SHAPES.items():
                    kind='cpu' if name in self.cpu_keys else 'native'
                    row,directory=records[kind][self.count];metadata=row['outputs'][name]
                    path=pin(directory/metadata['file'],metadata['sha256'])
                    value=np.load(path,allow_pickle=False)
                    assert value.shape==shape and value.dtype==np.float32 and np.isfinite(value).all()
                    assert hashlib.sha256(value.tobytes()).hexdigest()==metadata['raw_sha256']
                    prediction[name]=value.copy()
                prediction.update(images=images,image_match_ratio=None)
                self.count+=1
                return prediction
        for name,keys in routes:
            assert time.monotonic()-started<150
            route=out/name;route.mkdir()
            solver=Solver(init_conf_threshold=25.,device='cpu',image_size=518,max_loops=0,offline=True,enable_viewer=False)
            adapter=Adapter(keys)
            for i in range(16):
                prediction=solver.run_predictions([v['file'] for v in sources[i:i+2]],adapter,max_loops=0)
                solver.add_points(prediction);solver.graph.optimize()
                assert solver.map.get_num_submaps()==i+1 and solver.graph.get_num_loops()==0
                del prediction
            homographies=solver.map.get_all_homographies(solver.graph)
            assert adapter.count==16 and homographies.shape==(32,4,4) and np.isfinite(homographies).all()
            np.save(route/'homographies.npy',homographies,allow_pickle=False)
            solver.map.write_poses_to_file(str(route/'poses-tum.txt'),solver.graph,kitti_format=False)
            poses=np.loadtxt(route/'poses-tum.txt')
            assert poses.shape==(17,8) and np.isfinite(poses).all()
            endpoint=None
            if name in ('cpu','native'):
                reference=ROOT/('sl4-fp32-r1' if name=='cpu' else 'sl4-native-r1')
                for file in ('homographies.npy','poses-tum.txt'):
                    pin(reference/file,sha(reference/file))
                    assert sha(route/file)==sha(reference/file),name+' endpoint differs'
                endpoint=True
            report['routes'].append(dict(name=name,cpu_output_keys=list(keys),native_output_keys=[k for k in frozen.SHAPES if k not in keys],
                actual_model_output=name in ('cpu','native'),mixed_route_not_deployable=name not in ('cpu','native'),
                endpoint_byte_identity=endpoint,pairs=16,poses=17,nodes=32,factors=int(solver.graph.graph.size()),max_loops=0,
                trajectory_file=f'{name}/poses-tum.txt',trajectory_sha256=sha(route/'poses-tum.txt'),homographies_sha256=sha(route/'homographies.npy')))
            print(json.dumps(report['routes'][-1]),flush=True)
            del adapter,solver
        report['status']='completed_saved_head_group_counterfactual_with_exact_unmixed_endpoints'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        os.sched_setaffinity(0,initial_affinity)
        report.update(source_and_input_sha256={str(p):h for p,h in pins.items()},source_inputs_unchanged=all(sha(p)==h for p,h in pins.items()),
                      affinity_restored=os.sched_getaffinity(0)==initial_affinity,total_seconds=time.monotonic()-started)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
