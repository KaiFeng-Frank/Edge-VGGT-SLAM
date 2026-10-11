"""Generated/fake ABI only; no native compilation, model, SDK or board execution."""
import ctypes as ct
import errno
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


pack = load('_v43_pack_fake_test', 'v43_native_fp16_pack.py')
check = load('_v43_pack_conversion_fake_test', 'check-v43-native-pack-board.py')


class Function:
    def __init__(self, call):
        self.call = call
    def __call__(self, *args):
        return self.call(*args)


class FakeNative:
    """A reference surrogate for wrapper/driver protocol, not the C kernel."""
    def __init__(self, fpcr=0):
        self.fpcr = fpcr
        self.calls = []
        self.v43_pack_abi_version = Function(lambda: 1)
        self.v43_pack_read_fpcr = Function(lambda: self.fpcr)
        self.v43_pack_cf_f32_to_nc1hwc2_f16 = Function(self.pack)

    def pack(self, src, src_elements, dst, capacity, B, C, D, M):
        self.calls.append((B, C, D, M, src_elements, capacity))
        if B not in (1, 2) or C not in (384, 882) or D not in range(3, 8) or M != 16:
            return errno.EINVAL
        shape = (B, (C + 7) // 8, D, M, 8)
        count = int(np.prod(shape))
        if src_elements != B * C * D * M:
            return errno.EINVAL
        if capacity < count * 2:
            return errno.ENOSPC
        s, t = ct.cast(src, ct.c_void_p).value, ct.cast(dst, ct.c_void_p).value
        if s < t + count * 2 and t < s + src_elements * 4:
            return errno.EINVAL
        if self.fpcr & pack.FPCR_UNSUPPORTED_MASK:
            return errno.ENOTSUP
        value = np.ctypeslib.as_array(src, shape=(src_elements,)).reshape(B, C, D, M)
        if not np.isfinite(value).all():
            return errno.EDOM
        try:
            reference = check.reference_bits(value)
        except ValueError:
            return errno.ERANGE
        np.ctypeslib.as_array(dst, shape=(count,))[:] = reference.reshape(-1)
        return 0


def contract(B=1, D=3):
    result = dict(shapes={}, attrs={})
    for index, (name, C) in enumerate((('net', 384), ('inp', 384), ('corr', 882))):
        shape = [B, (C + 7) // 8, D, 16, 8]
        count = int(np.prod(shape))
        result['shapes'][name] = (B, C, D, 16)
        result['attrs'][name] = dict(index=index, name=name, n_dims=5, dims=shape + [0] * 11,
            shape=shape, n_elems=count, size=count * 2, size_with_stride=count * 2,
            fmt=2, type=1, qnt_type=2, fl=0, zp=0, scale=1.0, scale_finite=True,
            w_stride=16, h_stride=0, pass_through=0)
    return result


def test_all_supported_shapes_mapping_tail_and_owned_lifetime():
    api = FakeNative()
    library = pack.NativePackLibrary(api)
    for B in (1, 2):
        for D in range(3, 8):
            spec = contract(B, D)
            for name in ('net', 'corr'):
                value = np.arange(np.prod(spec['shapes'][name]), dtype=np.float32).reshape(spec['shapes'][name])
                value *= np.float32(0.001)
                before = value.tobytes()
                result = pack.pack_blocked_fp16(value, spec['attrs'][name], library)
                assert result.flags.owndata and result.flags.c_contiguous and result.dtype == np.uint16
                assert not np.shares_memory(result, value)
                assert np.array_equal(result, check.reference_bits(value))
                if name == 'corr':
                    assert np.all(result[:, -1, :, :, 2:] == 0)
                result.reshape(-1)[0] = 123
                assert value.tobytes() == before
    del library, api
    assert result.flags.owndata and result.reshape(-1)[0] == 123


def test_finite_boundary_rounding_and_rejected_numeric_values():
    spec = contract()
    library = pack.NativePackLibrary(FakeNative())
    value = np.zeros(spec['shapes']['net'], dtype=np.float32)
    values = check.finite_boundary_values()
    value.reshape(-1)[:len(values)] = values
    actual = pack.pack_blocked_fp16(value, spec['attrs']['net'], library)
    assert np.array_equal(actual, check.reference_bits(value))
    boundary_rows = check.check_boundary_bits(pack, library, spec)
    assert len(boundary_rows) == 18 and all(row['passed'] for row in boundary_rows)
    rows = check.check_rejections(pack, library, spec)
    assert len(rows) == 6 and all(row['rejected'] and row['input_unchanged'] for row in rows)


@pytest.mark.parametrize('bit', [0, 1, 8, 9, 10, 11, 12, 15, 19, 22, 23, 24, 25, 26])
def test_unsupported_fpcr_rejected_without_state_mutation(bit):
    api = FakeNative(1 << bit)
    library = pack.NativePackLibrary(api)
    spec = contract()
    value = np.zeros(spec['shapes']['net'], dtype=np.float32)
    with pytest.raises(OSError) as error:
        pack.pack_blocked_fp16(value, spec['attrs']['net'], library)
    assert error.value.errno == errno.ENOTSUP and api.fpcr == 1 << bit
    assert pack.fpcr_record(library)['supported'] is False


def test_attribute_mismatch_rejected_before_C_call():
    api = FakeNative()
    library = pack.NativePackLibrary(api)
    spec = contract()
    value = np.zeros(spec['shapes']['net'], dtype=np.float32)
    for field, wrong in [('n_dims', 4), ('fmt', 1), ('type', 0), ('qnt_type', 0),
                          ('scale', 0.5), ('w_stride', 32), ('h_stride', 3),
                          ('size_with_stride', 0), ('pass_through', 1), ('index', 2), ('name', 'corr')]:
        bad = dict(spec['attrs']['net'])
        bad[field] = wrong
        with pytest.raises(ValueError):
            pack.pack_blocked_fp16(value, bad, library)
    assert not api.calls


def test_no_implicit_dtype_or_layout_conversion():
    api = FakeNative()
    library = pack.NativePackLibrary(api)
    spec = contract()
    value = np.zeros(spec['shapes']['net'], dtype=np.float32)
    with pytest.raises(ValueError):
        pack.pack_blocked_fp16(value.astype(np.float64), spec['attrs']['net'], library)
    with pytest.raises(ValueError):
        pack.pack_blocked_fp16(value[:, :, :, ::-1], spec['attrs']['net'], library)
    assert not api.calls


def test_C_ABI_capacity_alias_guards_with_surrogate():
    rows = check.check_c_capacity_and_alias(pack, pack.NativePackLibrary(FakeNative()), contract())
    assert len(rows) == 5 and rows[-1]['passed']


def test_conversion_driver_timings_and_reference_surrogate():
    spec = contract()
    rng = np.random.default_rng(4301)
    feed = {name: rng.standard_normal(shape, dtype=np.float32) for name, shape in spec['shapes'].items()}
    rows, outputs = check.check_feed(pack, pack.NativePackLibrary(FakeNative()), feed, spec, repeat=2)
    assert len(rows) == 3 and set(outputs) == set(feed)
    assert all(len(row['complete_pack_seconds']) == 2 and row['different_reference_bits'] == 0 for row in rows)
    assert all('finite/half-overflow validation' in row['timing_includes'] for row in rows)


def test_C_source_has_only_read_FPCR_and_neon_tile_conversion():
    source = (HERE / 'v43_native_fp16_pack.c').read_text()
    assert 'mrs %0, fpcr' in source and 'msr' not in source
    assert 'p += 4' in source and 'lane < 8' in source
    assert 'vcvt_f16_f32' in source and 'vtrnq_f32' in source
    assert '__FAST_MATH__' in source and 'return ENOTSUP' in source
