"""Board-only dynamic CPU ORT Update; research, not numerical/VO acceptance.

Original DPVO parameters live in external ONNX, never embedded here. The ten
runtime topology inputs preserve v28 semantics; only terminal Sigmoid uses the
v29 stable identity. Each request owns feeds/results and accounts for topology,
input copies, runtime, output copies and validation. No clipping or fallback.
"""
import hashlib
import copy
import json
from pathlib import Path
import platform
import threading
import time
import numpy as np

INPUT_NAMES=('net','inp','corr','previous_indices','previous_valid','following_indices','following_valid',
             'kk_indices','kk_valid','kk_inverse','ij_indices','ij_valid','ij_inverse')
OUTPUT_NAMES=('net_out','delta','weight')
SHAPES={'net':[1,'E',384],'inp':[1,'E',384],'corr':[1,'E',882],
        'previous_indices':['E'],'previous_valid':[1,'E',1],'following_indices':['E'],'following_valid':[1,'E',1],
        'kk_indices':['Gkk','Lkk'],'kk_valid':['Gkk','Lkk'],'kk_inverse':['E'],
        'ij_indices':['Gij','Lij'],'ij_valid':['Gij','Lij'],'ij_inverse':['E'],
        'net_out':[1,'E',384],'delta':[1,'E',2],'weight':[1,'E',2]}
INT_NAMES={n for n in INPUT_NAMES if n.endswith('_indices') or n.endswith('_inverse')}

def require(ok,reason):
    if not ok:raise ValueError(reason)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(1<<20),b''):h.update(value)
    return h.hexdigest()
def board_guard():
    require(platform.system()=='Linux' and platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'Actual ORT model execution is RK3588 board-only')
def validate_contract(manifest):
    require(manifest['schema']=='v31_DPVO_dynamic_Update_export' and manifest['status']=='exported_dynamic_ONNX_numerical_validation_pending','Expected frozen dynamic export receipt')
    require(manifest['original_weights_unchanged'] and manifest['sources_and_inputs_unchanged'] and not manifest['cleanup_errors'],'Export sources/weights/cleanup failed')
    require(manifest['source_pins']==manifest['source_hashes_after'] and len(manifest['original_weights_before'])==94 and manifest['original_weights_before']==manifest['original_weights_after'],'Original94 parameters differ')
    require(len(manifest['original_Update_weights_before'])==50 and manifest['original_Update_weights_before']==manifest['original_Update_weights_after'],'Original50 Update parameters differ')
    require(all(manifest['rewrite'][k] is True for k in ('terminal_weight_only','original_initializers_byte_unchanged','other_nodes_byte_unchanged','IO_byte_unchanged')),'Only verified terminal weight rewrite allowed')
    model=manifest['models']['final'];require(model['role']=='dynamic_Update_terminal_stable_sigmoid' and model['source_onnx_sha256']==manifest['models']['raw']['sha256'],'Candidate ancestry differs')
    for key,names in [('inputs',INPUT_NAMES),('outputs',OUTPUT_NAMES)]:
        require([r['name'] for r in model[key]]==list(names),'I/O names/order changed')
        for r in model[key]:require(r['shape']==SHAPES[r['name']] and r['dtype']==(7 if r['name'] in INT_NAMES else 1),'Symbolic/fixed axes or type changed')
    return model

class DynamicORTUpdate:
    def __init__(self,manifest_path,manifest_sha256,topology_module,*,max_edges=98304):
        board_guard();self._lock=threading.RLock();self.session=None;self.closed=False
        self.manifest_path=Path(manifest_path).resolve();require(sha(self.manifest_path)==manifest_sha256,'Manifest SHA differs')
        self.manifest=json.loads(self.manifest_path.read_text());self.model=validate_contract(self.manifest)
        require(Path(self.model['file']).name==self.model['file'],'Model basename required')
        self.model_path=self.manifest_path.parent/self.model['file'];require(sha(self.model_path)==self.model['sha256'] and self.model_path.stat().st_size==self.model['bytes'],'Dynamic model artifact differs')
        require(type(max_edges) is int and 0<max_edges<=98304,'Bounded positive edge capacity required')
        require(tuple(topology_module.INPUT_NAMES)==INPUT_NAMES,'Topology input contract differs')
        self.max_edges=max_edges;self.cache=topology_module.TopologyCache(max_entries=2,max_bytes=32<<20,max_table_bytes=32<<20)
        self.calls=[];self.attempted=0;self.failed=0;self.pins={str(self.manifest_path):manifest_sha256,str(self.model_path):self.model['sha256']}
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        begin=time.monotonic()
        try:
            self.session=ort.InferenceSession(str(self.model_path),sess_options=options,providers=['CPUExecutionProvider'])
            require(self.session.get_providers()==['CPUExecutionProvider'],'CPU ORT provider only')
            for nodes,names in [(self.session.get_inputs(),INPUT_NAMES),(self.session.get_outputs(),OUTPUT_NAMES)]:
                require([x.name for x in nodes]==list(names),'Session I/O names changed')
                for x in nodes:require(x.shape==SHAPES[x.name] and x.type==('tensor(int64)' if x.name in INT_NAMES else 'tensor(float)'),'Session symbolic shape/type differs')
        except Exception:
            self.close();raise
        self.load_seconds=time.monotonic()-begin;self.ort_version=ort.__version__
    def __call__(self,net,context,corr,ii,jj,kk):
        with self._lock:
            require(not self.closed and self.session is not None,'Dynamic Update is closed');self.attempted+=1;begin=time.monotonic()
            try:
                require(all(isinstance(x,np.ndarray) and x.dtype==np.int64 and x.ndim==1 for x in (ii,jj,kk)),'Owned int64 graph arrays required')
                E=len(kk);require(0<E<=self.max_edges and ii.shape==jj.shape==kk.shape,'Edge capacity/shape differs')
                for x,shape in [(net,(1,E,384)),(context,(1,E,384)),(corr,(1,E,882))]:
                    require(isinstance(x,np.ndarray) and x.dtype==np.float32 and x.shape==shape and np.isfinite(x).all(),'Finite FP32 neural input shape differs')
                topology_begin=time.monotonic();topology=self.cache.get(ii,jj,kk);raw_arrays=topology.arrays()
                require(isinstance(raw_arrays,(tuple,list)) and len(raw_arrays)==10,'Exactly ten topology arrays required')
                arrays=dict(zip(INPUT_NAMES[3:],raw_arrays));topology_end=time.monotonic()
                feed={'net':net.copy(order='C'),'inp':context.copy(order='C'),'corr':corr.copy(order='C'),**arrays};ready=time.monotonic()
                values=self.session.run(list(OUTPUT_NAMES),feed);runtime_end=time.monotonic()
                require(isinstance(values,(tuple,list)) and len(values)==3,'Exactly three Update outputs required')
                result={}
                for key,value,shape in zip(('net','delta','weight'),values,((1,E,384),(1,E,2),(1,E,2))):
                    require(isinstance(value,np.ndarray) and value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all(),'Dynamic Update output invalid')
                    result[key]=value.copy(order='C')
                require(np.all((result['weight']>=0)&(result['weight']<=1)),'Dynamic confidence outside[0,1]')
                end=time.monotonic();self.calls.append(dict(index=len(self.calls),edges=E,total_seconds=end-begin,input_validation_seconds=topology_begin-begin,
                    topology_seconds=topology_end-topology_begin,feed_copy_seconds=ready-topology_end,ORT_seconds=runtime_end-ready,output_copy_check_seconds=end-runtime_end,
                    weight_min=float(result['weight'].min()),weight_max=float(result['weight'].max()),group_shapes={k:list(arrays[k+'_indices'].shape) for k in ('kk','ij')}))
                return result
            except Exception:
                self.failed+=1;raise
    def check_sources(self):
        result={p:sha(p) for p in self.pins};require(result==self.pins,'Dynamic artifact changed');return result
    def close(self):
        with self._lock:
            self.session=None;self.closed=True
            if hasattr(self,'cache'):self.cache.close()
    def metrics(self):
        with self._lock:
            return dict(attempted_calls=self.attempted,completed_calls=len(self.calls),failed_calls=self.failed,calls=copy.deepcopy(self.calls),closed=self.closed,
                topology_cache=self.cache.metrics(),ORT_version=getattr(self,'ort_version',None),session_load_seconds=getattr(self,'load_seconds',None),
                providers=['CPUExecutionProvider'],threads=4,interop_threads=1,optimization='disabled',model_sha256=self.model['sha256'],
                manifest_sha256=self.pins[str(self.manifest_path)],numerical_acceptance_implied=False,NPU_used=False)
