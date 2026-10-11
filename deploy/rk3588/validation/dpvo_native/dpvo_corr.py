"""Owned cached FP32 CPU DPVO correlation; research, no Torch/BA/model.

Derived from DPVO MIT commit 0ac95b656d1fda91c271d2a106460d19ad966fc7;
Copyright (c) 2022 Princeton Vision & Learning Lab. See accompanying LICENSE.

Context preparation copies values: changing original feature ndarrays does not
update it. A new feature generation requires a new PreparedCorrelation and new
explicit cache_key. Keys identify caller-managed generations, not data_ptrs.
Every call must supply the matching key; no content hashing inside timed calls.
Each output is owned and remains valid after another call or context close.
One context is serialised; no global affinity/OpenMP/thread changes are made.
"""
import copy
import ctypes as ct
import hashlib
import threading
import time
from pathlib import Path

import numpy as np

ABI_VERSION = 1
FIXED_RADIUS = 3
MEMORY_LIMIT = 512 * 1024 * 1024
SOURCE_PINS = {
    "dpvo/altcorr/correlation_kernel.cu": "85667383d879c9093cee5b88ae231f562d98e751b4a549917ebda105a1a291e0",
    "dpvo/altcorr/correlation.py": "367fe95eff8c076b4ec3c4830ff9a9626a8bca4abadc9042a05bd3f58d577f4a",
    "dpvo/utils.py": "5bc5820014045b6979d9979ae80a2b519c64f9491ae83e909a77731152751a4a",
}


class CorrelationError(RuntimeError):
    pass


class NativeStats(ct.Structure):
    _fields_ = [("abi_version",ct.c_uint32),("reserved",ct.c_uint32),
                ("context_bytes",ct.c_uint64),("output_elements",ct.c_uint64),
                ("scan_seconds",ct.c_double),("layout_seconds",ct.c_double),
                ("pool_seconds",ct.c_double),("kernel_seconds",ct.c_double),
                ("total_seconds",ct.c_double),("threads",ct.c_int32),("neon",ct.c_int32)]


def _plain(stats):
    return {key:getattr(stats,key) for key,_ in stats._fields_}


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def load_library(path, expected_sha256=None):
    path=Path(path).resolve()
    actual=_sha(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise CorrelationError("Native library SHA256 mismatch")
    api=ct.CDLL(str(path))
    api.dpvo_corr_v27_abi_version.restype=ct.c_uint32
    if api.dpvo_corr_v27_abi_version() != ABI_VERSION:
        raise CorrelationError("Native correlation ABI mismatch")
    api.dpvo_corr_v27_stats_size.restype=ct.c_uint64
    if api.dpvo_corr_v27_stats_size() != ct.sizeof(NativeStats):
        raise CorrelationError("Native statistics ABI size mismatch")
    api.dpvo_corr_v27_build_info.restype=ct.c_char_p
    fptr=ct.POINTER(ct.c_float); iptr=ct.POINTER(ct.c_int64)
    stats=ct.POINTER(NativeStats); error=ct.POINTER(ct.c_char)
    api.dpvo_corr_v27_prepare.argtypes=[fptr,ct.c_uint64,fptr,ct.c_uint64,*([ct.c_int64]*8),
                                      ct.c_int32,ct.POINTER(ct.c_void_p),stats,error,ct.c_size_t]
    api.dpvo_corr_v27_prepare.restype=ct.c_int
    api.dpvo_corr_v27_correlate.argtypes=[ct.c_void_p,fptr,ct.c_uint64,iptr,iptr,ct.c_uint64,
                                        ct.c_int64,ct.c_int32,fptr,ct.c_uint64,stats,error,ct.c_size_t]
    api.dpvo_corr_v27_correlate.restype=ct.c_int
    api.dpvo_corr_v27_destroy.argtypes=[ct.POINTER(ct.c_void_p),error,ct.c_size_t]
    api.dpvo_corr_v27_destroy.restype=ct.c_int
    return api,actual


def _array(value,dtype,name,ndim):
    if not isinstance(value,np.ndarray) or value.dtype != np.dtype(dtype) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D NumPy {np.dtype(dtype)} ndarray")
    if value.nbytes > MEMORY_LIMIT:
        raise ValueError(f"{name} exceeds 512MiB safety budget")
    return np.ascontiguousarray(value)


def _pointer(value,ctype):
    return value.ctypes.data_as(ct.POINTER(ctype))


class PreparedCorrelation:
    def __init__(self,library,gmap,fmap,*,threads=4,cache_key,library_sha256=None):
        begin=time.monotonic()
        self._lock=threading.RLock()
        self._context=ct.c_void_p()
        self._closed=False
        self._calls=0;self._attempts=0;self._failures=0
        self._time_sum=0.0;self._kernel_sum=0.0;self._time_max=0.0
        self.last_call=None
        if not isinstance(cache_key,str) or not cache_key:
            raise ValueError("cache_key must be a nonempty immutable generation string")
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer in [1,64]")
        gmap=_array(gmap,np.float32,"gmap",5)
        fmap=_array(fmap,np.float32,"fmap",5)
        b,p,c,ph,pw=gmap.shape
        if fmap.shape[0] != b or fmap.shape[2] != c:
            raise ValueError("gmap/fmap batch/channel mismatch")
        _,v,_,h,w=fmap.shape
        if min(b,p,v,c,ph,pw,h,w)<1 or c%8 or min(h,w)<4:
            raise ValueError("positive dimensions, C divisible by8, H/W>=4 required")
        context_bytes=(gmap.size+fmap.size+b*v*c*(h//4)*(w//4))*4
        if context_bytes > MEMORY_LIMIT:
            raise ValueError("context exceeds 512MiB safety budget")
        self._api,self.library_sha256=load_library(library,library_sha256)
        self.build_info=self._api.dpvo_corr_v27_build_info().decode("ascii")
        self.shape=(b,p,v,c,ph,pw,h,w)
        self._cache_key=cache_key
        self.threads=threads
        stats=NativeStats();error=ct.create_string_buffer(512)
        status=self._api.dpvo_corr_v27_prepare(_pointer(gmap,ct.c_float),gmap.size,
                   _pointer(fmap,ct.c_float),fmap.size,*self.shape,threads,
                   ct.byref(self._context),ct.byref(stats),error,len(error))
        if status:
            self.close()
            raise CorrelationError(error.value.decode("utf8",errors="replace"))
        self._preparation={"total_seconds":time.monotonic()-begin,"native":_plain(stats),
                          "context_bytes":context_bytes,"cache_key":cache_key,
                          "feature_inputs_copied":True,"source_array_mutations_do_not_update_cache":True}

    @property
    def cache_key(self):
        return self._cache_key

    @property
    def preparation(self):
        return copy.deepcopy(self._preparation)

    def correlate(self,coords,kk,jj,*,cache_key,radius=3):
        """Return (owned result, timings). Total includes packing/lock/allocation.

        Preparation is amortised separately; use correlate_once when comparing
        against v26's per-call pyramid construction. No old result fallback.
        """
        begin=time.monotonic()
        with self._lock:
            self._attempts += 1
            if self._closed:
                self._failures += 1
                raise CorrelationError("correlation context closed")
            try:
                if cache_key != self.cache_key:
                    raise ValueError("feature generation/cache_key mismatch: prepare a new context")
                if type(radius) is not int or radius != FIXED_RADIUS:
                    raise ValueError("only original radius3 supported")
                coords=_array(coords,np.float32,"coords",5)
                kk=_array(kk,np.int64,"kk",1);jj=_array(jj,np.int64,"jj",1)
                b,_,_,_,ph,pw,_,_=self.shape
                edges=kk.size
                if jj.shape != kk.shape or coords.shape != (b,edges,2,ph,pw):
                    raise ValueError("coordinate/index shape mismatch")
                count=b*edges*98*ph*pw
                if count*4 > MEMORY_LIMIT:
                    raise ValueError("output exceeds 512MiB safety budget")
                packing_end=time.monotonic()
                output=np.empty((b,edges,98*ph*pw),np.float32)
                stats=NativeStats();error=ct.create_string_buffer(512)
                native_begin=time.monotonic()
                status=self._api.dpvo_corr_v27_correlate(self._context,_pointer(coords,ct.c_float),coords.size,
                         _pointer(kk,ct.c_int64),_pointer(jj,ct.c_int64),edges,edges,radius,
                         _pointer(output,ct.c_float),output.size,ct.byref(stats),error,len(error))
                native_end=time.monotonic()
                if status:
                    raise CorrelationError(error.value.decode("utf8",errors="replace"))
                timing={"total_seconds":time.monotonic()-begin,"packing_and_lock_seconds":packing_end-begin,
                        "allocation_and_ffi_setup_seconds":native_begin-packing_end,
                        "native_call_seconds":native_end-native_begin,"native":_plain(stats),
                        "includes_prepare":False,"cache_key":self.cache_key}
                self._calls+=1
                self._time_sum+=timing["total_seconds"]
                self._kernel_sum+=stats.kernel_seconds
                self._time_max=max(self._time_max,timing["total_seconds"])
                self.last_call=copy.deepcopy(timing)
                return output,timing
            except Exception:
                self._failures+=1
                raise

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._context.value:
                error=ct.create_string_buffer(512)
                status=self._api.dpvo_corr_v27_destroy(ct.byref(self._context),error,len(error))
                if status:
                    raise CorrelationError(error.value.decode("utf8",errors="replace"))
            self._closed=True

    def metrics(self):
        with self._lock:
            return {"abi_version":ABI_VERSION,"build_info":self.build_info,"library_sha256":self.library_sha256,
                    "threads":self.threads,"cache_key":self.cache_key,"preparation":self.preparation,
                    "attempts":self._attempts,"completed_calls":self._calls,"failed_calls":self._failures,
                    "correlate_total_seconds_sum":self._time_sum,"kernel_seconds_sum":self._kernel_sum,
                    "correlate_total_seconds_max":self._time_max,"closed":self._closed,
                    "current_context_bytes":0 if self._closed else self._preparation["context_bytes"],
                    "Torch_used":False,"CUDA_used":False,"NPU_used":False,
                    "VO_or_BA_implemented":False,"realtime_validated":False}

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()


def correlate_once(library,gmap,fmap,coords,kk,jj,*,threads=4,cache_key,library_sha256=None):
    """Full one-shot wall time includes library/hash, packing/pool, call, destroy."""
    begin=time.monotonic()
    with PreparedCorrelation(library,gmap,fmap,threads=threads,cache_key=cache_key,
                             library_sha256=library_sha256) as prepared:
        result,call=prepared.correlate(coords,kk,jj,cache_key=cache_key)
        preparation=prepared.preparation.copy()
    return result,{"total_seconds":time.monotonic()-begin,"preparation":preparation,
                   "correlation":call,"includes_prepare_and_close":True,"closed":True}
