"""Owned native C2=8 FP16 packing only; no model/SDK calls or FP state writes.

Input is stable, C-contiguous native FP32 CF [B,C,D,16]. Keep that input
unchanged during this synchronous call. The returned uint16 array owns IEEE
binary16 bits with shape [B,ceil(C/8),D,16,8]. Its owner must retain it through
SDK inputs_set/run/outputs_get/release. On error no partial buffer is returned.
"""
import ctypes as ct
import errno
import hashlib
import math
import operator
from pathlib import Path
import platform
import sys

import numpy as np

ABI_VERSION = 1
SOURCE_QUERY_SHA256 = '89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642'
FPCR_UNSUPPORTED_MASK = ((3 << 22) | (1 << 24) | (1 << 19) | (1 << 26)
                         | (1 << 25) | (31 << 8) | (1 << 15) | 3)


class NativePackLibrary:
    """Bind once; a fake API can be supplied for generated host tests."""
    def __init__(self, api, *, sha256=None):
        self.api = api
        self.sha256 = sha256
        self.pack = api.v43_pack_cf_f32_to_nc1hwc2_f16
        self.pack.argtypes = [ct.POINTER(ct.c_float), ct.c_size_t,
                              ct.POINTER(ct.c_uint16), ct.c_size_t,
                              ct.c_uint32, ct.c_uint32, ct.c_uint32, ct.c_uint32]
        self.pack.restype = ct.c_int
        self.version = api.v43_pack_abi_version
        self.version.argtypes = []
        self.version.restype = ct.c_uint32
        self.read_fpcr = api.v43_pack_read_fpcr
        self.read_fpcr.argtypes = []
        self.read_fpcr.restype = ct.c_uint64
        if self.version() != ABI_VERSION:
            raise ValueError('Native pack ABI version differs')


def load_native_library(path, expected_sha256=None):
    """Load a board-built library after hashing; callers can require its pin."""
    if platform.machine().lower() not in ('aarch64', 'arm64') or sys.byteorder != 'little':
        raise OSError(errno.ENOTSUP, 'Native pack requires little endian AArch64')
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Native pack library SHA256 differs')
    return NativePackLibrary(ct.CDLL(str(path)), sha256=digest)


def _library(value):
    if isinstance(value, NativePackLibrary):
        return value
    if isinstance(value, (str, Path)):
        return load_native_library(value)
    if isinstance(value, ct.CDLL):
        return NativePackLibrary(value)
    raise TypeError('NativePackLibrary, ctypes.CDLL or library path required')


def fpcr_record(native_library):
    value = int(_library(native_library).read_fpcr())
    return dict(raw=value, hex=f'0x{value:016x}', rounding=(value >> 22) & 3,
                FZ=bool(value & (1 << 24)), FZ16=bool(value & (1 << 19)),
                AHP=bool(value & (1 << 26)), DN=bool(value & (1 << 25)),
                alternative_handling=bool(value & 3),
                enabled_traps=bool(value & ((31 << 8) | (1 << 15))),
                unsupported_mask=FPCR_UNSUPPORTED_MASK,
                supported=(value & FPCR_UNSUPPORTED_MASK) == 0,
                FPCR_modified_by_packer=False)


def _field(attribute, name):
    if isinstance(attribute, dict):
        if name not in attribute:
            raise ValueError('Missing native attribute field: ' + name)
        return attribute[name]
    try:
        return getattr(attribute, name)
    except AttributeError as error:
        raise ValueError('Missing native attribute field: ' + name) from error


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Integer native attribute required: ' + name)
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError('Integer native attribute required: ' + name) from error


def validate_native_attribute(value, attribute):
    """Validate the measured command8/10 contract, not logical command1."""
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype('float32'):
        raise ValueError('A native FP32 NumPy array is required; dtype conversion is not implicit')
    if value.ndim != 4 or not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError('Aligned C-contiguous CF [B,C,D,16] is required')
    B, C, D, M = value.shape
    if B not in (1, 2) or C not in (384, 882) or D not in range(3, 8) or M != 16:
        raise ValueError('Unsupported CF shape')
    shape = (B, (C + 7) // 8, D, M, 8)
    dims = tuple(_integer(v, 'dims') for v in _field(attribute, 'dims'))
    if len(dims) != 16 or dims[:5] != shape or any(dims[5:]):
        raise ValueError('Actual native dims/padding differ from C2=8 contract')
    if isinstance(attribute, dict) and 'shape' in attribute and tuple(attribute['shape']) != shape:
        raise ValueError('Native shape and dims differ')
    count = math.prod(shape)
    expected = dict(n_dims=5, fmt=2, type=1, qnt_type=2, fl=0, zp=0,
                    w_stride=16, h_stride=0, pass_through=0,
                    n_elems=count, size=count * 2, size_with_stride=count * 2)
    for name, wanted in expected.items():
        if _integer(_field(attribute, name), name) != wanted:
            raise ValueError('Actual native attribute differs: ' + name)
    scale = float(_field(attribute, 'scale'))
    if not math.isfinite(scale) or scale != 1.0:
        raise ValueError('Native scale must be the queried neutral value 1.0')
    if isinstance(attribute, dict) and 'scale_finite' in attribute and attribute['scale_finite'] is not True:
        raise ValueError('Native scale_finite differs')
    index = _integer(_field(attribute, 'index'), 'index')
    name = _field(attribute, 'name')
    if isinstance(name, bytes):
        name = name.split(b'\0', 1)[0].decode('ascii')
    if (C == 384 and (index, name) not in ((0, 'net'), (1, 'inp'))) or (C == 882 and (index, name) != (2, 'corr')):
        raise ValueError('Native input name/index/channel contract differs')
    return shape


def pack_blocked_fp16(value, attribute, native_library):
    """Full pack: attribute checks, owned allocation, finite/overflow checks, NEON.

    No whole-array astype or separate Python finite scan occurs. Numeric errors
    are detected within C; a failed call's partial destination is discarded.
    No clipping, scaling or input mutation is performed.
    """
    shape = validate_native_attribute(value, attribute)
    library = _library(native_library)
    owned = np.empty(shape, dtype=np.uint16, order='C')
    code = int(library.pack(value.ctypes.data_as(ct.POINTER(ct.c_float)), value.size,
                            owned.ctypes.data_as(ct.POINTER(ct.c_uint16)), owned.nbytes,
                            *value.shape))
    if code:
        descriptions = {errno.EDOM: 'Nonfinite FP32 input rejected',
                        errno.ERANGE: 'FP16 rounding overflow rejected',
                        errno.ENOTSUP: 'Unsupported FPCR state rejected',
                        errno.ENOSPC: 'Native output capacity too small',
                        errno.EINVAL: 'Invalid dimensions, source size or pointer alias',
                        errno.EOVERFLOW: 'Pointer/size arithmetic overflow'}
        raise OSError(code, descriptions.get(code, 'Native pack failed'))
    return owned
