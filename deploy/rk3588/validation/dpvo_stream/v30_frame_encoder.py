"""Research-only frozen DPVO CPU ORT encoder and RANDOM Patchifier adapter.

Adapted from princeton-vl/DPVO commit 0ac95b656d1fda91c271d2a106460d19ad966fc7
net.py Patchifier and dpvo.py preprocessing/color semantics (MIT).
Copyright (c) 2022 Princeton Vision & Learning Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Input is already undistorted/cropped BGR uint8 HWC. No decode, crop, resize,
channel reorder or disparity RNG occurs here. Original encoder outputs include
/4. CPU random draws preserve the original x-then-y distribution/order but do
not claim CUDA RNG byte equivalence. Returned arrays are owned, stable snapshots.
Only a frozen CPU ORT encoder is executed; this is not a complete VO or NPU path.
"""
import copy
import hashlib
import json
import math
import numbers
from pathlib import Path
import platform
import threading
import time
import numpy as np

MODEL_SHA='527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82'
EXPORT_SHA='45fa589fa3a25a9722dc0832f5d14ffcec662e70d1dff135b86be641eeeba4b2'
ORT_SHA='153da765b1145d7bb74bc635c1982ae74c652dac28cf99be56cb40534c44b667'
CAPTURE_SHA='e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088'
CHECKPOINT_SHA='30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
PROBE_SHA='e2f56fac7a6432761bf67ea2703a7b72b9d695577302d8f02d5cc8af0c0db9a5'
SOURCE_PINS={'net.py':'1aee33d54f0b1cc1f3f842b60790ff1299ef9c1e20bca6467ce7114dca802acc',
    'dpvo_cpu_ops.py':'15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32',
    'dpvo_probe_common.py':'ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9'}
INPUT_SHAPE=(1,3,464,608)
OUTPUT_SHAPES=((1,128,116,152),(1,384,116,152))


def _require(ok,message):
    if not ok:raise ValueError(message)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _board_guard():
    _require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
             'Actual encoder session is RK3588 board-only')


def normalize_bgr(image):
    """CPU FP32 NCHW normalization, exactly ordered division/multiply/subtract."""
    _require(isinstance(image,np.ndarray) and image.dtype==np.uint8 and image.ndim==3 and image.shape[2]==3,
             'Expected cropped HWC BGR uint8')
    value=np.ascontiguousarray(image.transpose(2,0,1),dtype=np.float32)[None]
    return (np.float32(2)*(value/np.float32(255))-np.float32(.5)).astype(np.float32,copy=False)


def frame_input_metadata(image):
    """Optional audit metadata; call outside model/encode timing, no inference."""
    normalized=normalize_bgr(image)
    def row(value):
        return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,
                    tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    return dict(cropped_BGR_uint8=row(image),normalized_BGR_FP32=row(normalized),
                preprocessing='Already undistorted/cropped BGR uint8 HWC -> FP32 NCHW 2*(image/255)-.5; no geometry or color reorder.')


def random_centers(width,height,count,generator):
    """Original RANDOM branch distribution, explicit persistent Torch CPU RNG."""
    _require(all(isinstance(x,numbers.Integral) and not isinstance(x,bool) for x in (width,height,count))
             and width>=3 and height>=3 and count>0,'Positive count and interior feature grid required')
    import torch
    _require(isinstance(generator,torch.Generator) and generator.device.type=='cpu','Explicit Torch CPU Generator required')
    x=torch.randint(1,int(width)-1,(1,int(count)),generator=generator,device='cpu')
    y=torch.randint(1,int(height)-1,(1,int(count)),generator=generator,device='cpu')
    return torch.stack((x,y),dim=-1)[0].numpy().copy()


def _feature(value,name):
    _require(isinstance(value,np.ndarray) and value.dtype==np.float32,name+' must be FP32 NumPy')
    if value.ndim==5:
        _require(value.shape[:2]==(1,1),name+' must contain exactly one view');value=value[0,0]
    elif value.ndim==4:
        _require(value.shape[0]==1,name+' must contain exactly one batch');value=value[0]
    _require(value.ndim==3 and min(value.shape)>0 and np.isfinite(value).all(),name+' must be finite [C,H,W]')
    return value


def _integer_patchify(features,centers,radius):
    """Integer-coordinate subset of altcorr's dot-free four-term bilinear path.

    Extra +1 neighbors and zero padding preserve original FP32 zero arithmetic;
    no pointer/index outside a feature allocation is used.
    """
    _,height,width=features.shape
    offset=np.arange(-radius,radius+2,dtype=np.int64)
    xs=centers[:,0,None,None]+offset[None,None,:]
    ys=centers[:,1,None,None]+offset[None,:,None]
    valid=(xs>=0)&(xs<width)&(ys>=0)&(ys<height)
    sampled=features[:,ys.clip(0,height-1),xs.clip(0,width-1)].transpose(1,0,2,3)
    sampled=np.where(valid[:,None],sampled,np.float32(0))
    zero=np.float32(0)
    return np.ascontiguousarray(sampled[...,:-1,:-1]+zero*sampled[...,:-1,1:]+zero*sampled[...,1:,:-1]+zero*sampled[...,1:,1:])


def features_to_patches(fmap,imap,centers,*,inverse_depths=None,normalized_bgr=None):
    """Pure NumPy one-view Patchifier. No RNG, inference or depth prior.

    centers: int64 [M,2] in xy feature pixels, 1<=x<W-1, 1<=y<H-1.
    inverse_depths: optional FP32 [M] applied uniformly to each 3x3 patch;
    otherwise 1, matching Patchifier's pre-lifecycle default. Caller initializes
    random/median depth separately. Optional colors follow online RGB uint8.
    """
    f=_feature(fmap,'fmap');c=_feature(imap,'imap');height,width=f.shape[-2:]
    _require(c.shape[-2:]==(height,width),'fmap/imap spatial dimensions differ')
    _require(isinstance(centers,np.ndarray) and centers.dtype==np.int64 and centers.ndim==2 and centers.shape[1]==2 and len(centers)>0,
             'centers must be nonempty int64 [M,2]')
    _require(bool(((centers[:,0]>=1)&(centers[:,0]<width-1)&(centers[:,1]>=1)&(centers[:,1]<height-1)).all()),'Patch centers must be interior')
    count=len(centers)
    if inverse_depths is None:inverse_depths=np.ones(count,dtype=np.float32)
    _require(isinstance(inverse_depths,np.ndarray) and inverse_depths.dtype==np.float32 and inverse_depths.shape==(count,)
             and np.isfinite(inverse_depths).all() and (inverse_depths>=0).all(),'inverse_depths must be finite nonnegative FP32 [M]')
    delta=np.arange(-1,2,dtype=np.float32);patches=np.empty((count,3,3,3),dtype=np.float32)
    patches[:,0]=centers[:,0,None,None].astype(np.float32)+delta[None,None,:]
    patches[:,1]=centers[:,1,None,None].astype(np.float32)+delta[None,:,None]
    patches[:,2]=inverse_depths[:,None,None]
    result=dict(fmap=np.array(f,copy=True,order='C'),gmap=_integer_patchify(f,centers,1),
                context=np.array(_integer_patchify(c,centers,0)[:,:,0,0],copy=True,order='C'),
                patches=patches,centers=centers.copy())
    if normalized_bgr is not None:
        image=_feature(normalized_bgr,'normalized_bgr')
        _require(image.shape==(3,4*height,4*width),'Color input must be same original normalized BGR at 4x resolution')
        pixel_centers=4*centers+2
        clr=_integer_patchify(image,pixel_centers,0)[:,:,0,0]
        # Original dpvo.py converts normalized BGR to RGB and truncates uint8.
        result['colors']=np.ascontiguousarray(((clr[:,[2,1,0]]+np.float32(.5))*np.float32(255./2)).astype(np.uint8))
    return result


def _create_session(model_bytes):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session=ort.InferenceSession(model_bytes,sess_options=options,providers=['CPUExecutionProvider'])
    session._v30_ORT_version=ort.__version__
    return session


class BoardORTEncoder:
    """Synchronous fixed-shape CPU encoder with explicit original provenance.

    No affinity/OMP/global Torch thread changes. Session creation is board-only;
    tiny tests replace the private factory/guard, never weaken these checks.
    """
    def __init__(self,model_path,export_receipt,ort_receipt,*,model_sha256=MODEL_SHA,export_sha256=EXPORT_SHA,ort_sha256=ORT_SHA):
        _board_guard() # Before loading any graph or session.
        self._lock=threading.RLock();self._session=None;self._closed=False
        self._metrics=dict(encoder_backend='original_DPVO_CPU_ORT',CPU_model_execution=True,NPU_used=False,ORB_used=False,
            fixed_input_shape=list(INPUT_SHAPE),feature_outputs_scaled_by_four_in_graph=True,completed_frames=0,
            normalization_seconds=0.,ORT_seconds=0.,sampling_seconds=0.,total_encode_seconds=0.,closed=False,
            RNG_semantics='Explicit Torch CPU generator, original x-then-y randint bounds; no CUDA byte claim; no depth draws.',
            crop_undistort_done_by_caller=True,script_sha256=_sha(__file__))
        _require((model_sha256,export_sha256,ort_sha256)==(MODEL_SHA,EXPORT_SHA,ORT_SHA),'Only frozen accepted v26 encoder artifacts supported')
        files={Path(model_path):model_sha256,Path(export_receipt):export_sha256,Path(ort_receipt):ort_sha256}
        self._pins={str(p.resolve()):v for p,v in files.items()}
        for p,value in files.items():_require(_sha(p)==value,'Encoder artifact SHA differs: '+p.name)
        exported=json.loads(Path(export_receipt).read_text());verified=json.loads(Path(ort_receipt).read_text())
        _require(exported['status']=='exported_not_runtime_validated' and exported['script_sha256']==PROBE_SHA
            and exported['capture_sha256']==CAPTURE_SHA and exported['onnx']['sha256']==MODEL_SHA
            and Path(model_path).stat().st_size==exported['onnx']['bytes'] and exported['input_shape']==list(INPUT_SHAPE)
            and exported['output_shapes']==[list(x) for x in OUTPUT_SHAPES],'Frozen graph capture/shape binding differs')
        _require(verified['status']=='completed_CPU_ORT_gate' and verified['script_sha256']==PROBE_SHA and verified['capture_sha256']==CAPTURE_SHA
            and verified['export_sha256']==EXPORT_SHA and verified['onnx']==exported['onnx'] and verified['exact_reference_gate_passed']
            and verified['providers']==['CPUExecutionProvider'] and verified['graph_optimization']=='ORT_DISABLE_ALL','Accepted v26 CPU ORT gate required')
        for receipt in (exported,verified):
            _require(receipt['checkpoint_sha256']==CHECKPOINT_SHA and receipt['state_unchanged'] and receipt['source_pins']==receipt['source_hashes_after']
                and receipt['sources_and_inputs_unchanged'] and receipt['cleanup_errors']==[],'Original weights/source lifecycle differs')
            for name,digest in SOURCE_PINS.items():
                _require([v for p,v in receipt['source_pins'].items() if Path(p).name==name]==[digest],'Original implementation pin differs: '+name)
        try:
            session=_create_session(Path(model_path).read_bytes());self._session=session
            _require(session.get_providers()==['CPUExecutionProvider'],'CPU ORT only')
            inputs=session.get_inputs();outputs=session.get_outputs()
            _require(len(inputs)==1 and inputs[0].name=='normalized_bgr' and inputs[0].shape==list(INPUT_SHAPE) and inputs[0].type=='tensor(float)','ORT input binding differs')
            _require([x.name for x in outputs]==['fmap','imap'] and all(x.shape==list(shape) and x.type=='tensor(float)' for x,shape in zip(outputs,OUTPUT_SHAPES)) and len(outputs)==2,'ORT output binding differs')
            self._metrics.update(model_sha256=MODEL_SHA,export_sha256=EXPORT_SHA,ORT_receipt_sha256=ORT_SHA,capture_sha256=CAPTURE_SHA,checkpoint_sha256=CHECKPOINT_SHA,
                ORT_threads=4,ORT_interop_threads=1,ORT_optimization='disabled',ORT_version=getattr(session,'_v30_ORT_version','synthetic_factory'),
                initial_source_checks_passed=True,source_artifact_bytes={p.name:p.stat().st_size for p in files},
                array_ownership='Returned C-contiguous owned snapshots; no borrowed ORT output retained.')
        except Exception:
            self.close();raise

    def encode(self,image_bgr,*,centers=None,generator=None,patch_count=96,inverse_depths=None,return_colors=False):
        with self._lock:
            _require(not self._closed and self._session is not None,'Encoder is closed')
            _require(isinstance(image_bgr,np.ndarray) and image_bgr.dtype==np.uint8 and image_bgr.shape==(464,608,3),'Fixed cropped BGR uint8 [464,608,3] required')
            begin=time.monotonic();normalized=normalize_bgr(image_bgr);normalized_end=time.monotonic()
            outputs=self._session.run(['fmap','imap'],{'normalized_bgr':normalized});forward_end=time.monotonic()
            _require(len(outputs)==2 and all(isinstance(x,np.ndarray) and x.dtype==np.float32 and x.shape==shape for x,shape in zip(outputs,OUTPUT_SHAPES)), 'Encoder returned incompatible output')
            if centers is None:centers=random_centers(152,116,patch_count,generator)
            else:_require(generator is None,'Provide either centers or generator, not both')
            result=features_to_patches(outputs[0],outputs[1],centers,inverse_depths=inverse_depths,
                                       normalized_bgr=normalized[0] if return_colors else None)
            end=time.monotonic();timing=dict(normalization_seconds=normalized_end-begin,ORT_seconds=forward_end-normalized_end,sampling_seconds=end-forward_end,total_encode_seconds=end-begin)
            result['timing']=timing;self._metrics['completed_frames']+=1
            self._metrics['last_frame_timings']=dict(timing);self._metrics['last_patch_count']=len(centers)
            for key,value in timing.items():self._metrics[key]+=value
            return result

    def check_sources(self):
        with self._lock:
            _require(all(_sha(p)==digest for p,digest in self._pins.items()),'Frozen encoder artifacts changed')
            self._metrics['source_recheck_passed']=True
            return dict(self._pins)

    def metrics(self):
        with self._lock:return copy.deepcopy(self._metrics)

    def close(self):
        with self._lock:
            self._session=None;self._closed=True;self._metrics['closed']=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
