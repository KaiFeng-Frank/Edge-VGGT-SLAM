"""Generated modules only; no VGGT weights, inference, board or SDK execution."""
from collections import OrderedDict

import pytest
import torch
from torch import nn

from vggt_slam import patch_cache as cache


class Plain(nn.Module):
    def forward(self, value):
        return value


def install(probe):
    model = Plain()
    model.aggregator = Plain()
    model.aggregator.patch_embed = Plain()
    model.probe = probe
    return cache.install_patch_cache(model.eval())


def scalars(handle):
    return next(row[-1] for row in handle._fingerprint()[0] if row[0] == 'probe')


def require_stale(handle):
    with torch.no_grad(), pytest.raises(cache.PatchCacheStaleState):
        handle._guard()


def test_instance_scalar_add_change_delete():
    probe = Plain()
    handle = install(probe)
    try:
        assert scalars(handle) == ()
        probe.eps = .1
        assert scalars(handle) == (('eps', .1),)
        require_stale(handle)
        probe.eps = .2
        assert scalars(handle) == (('eps', .2),)
        require_stale(handle)
        del probe.eps
        assert scalars(handle) == ()
        with torch.no_grad():
            handle._guard()
    finally:
        handle.close()


def test_class_and_mro_scalar_changes_are_seen_on_next_fingerprint():
    class Base(Plain):
        pass
    class Child(Base):
        pass
    handle = install(Child())
    try:
        assert scalars(handle) == ()
        Base.eps = .1
        assert scalars(handle) == (('eps', .1),)
        require_stale(handle)
        Base.eps = .2
        assert scalars(handle) == (('eps', .2),)
        del Base.eps
        assert scalars(handle) == ()
        with torch.no_grad():
            handle._guard()
    finally:
        handle.close()


def test_property_retains_hasattr_then_getattr_and_attributeerror():
    class Probe(Plain):
        def __init__(self):
            super().__init__()
            self.calls, self.fail_at = 0, None

        @property
        def eps(self):
            self.calls += 1
            if self.calls == self.fail_at:
                raise AttributeError('generated descriptor failure')
            return .125

    probe = Probe()
    handle = install(probe)
    try:
        probe.calls = 0
        assert scalars(handle) == (('eps', .125),)
        assert probe.calls == 2
        probe.calls, probe.fail_at = 0, 1
        assert scalars(handle) == ()
        assert probe.calls == 1
        probe.calls, probe.fail_at = 0, 2
        with pytest.raises(AttributeError):
            handle._fingerprint()
        assert probe.calls == 2
    finally:
        handle.close()


def test_custom_getattr_retains_two_reads():
    class Probe(Plain):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def __getattr__(self, name):
            if name == 'eps':
                self.calls += 1
                return .25
            return super().__getattr__(name)

    probe = Probe()
    handle = install(probe)
    try:
        assert cache._class_presence_plan(type(probe)) is None
        probe.calls = 0
        assert scalars(handle) == (('eps', .25),)
        assert probe.calls == 2
    finally:
        handle.close()


def test_custom_getattribute_and_metaclass_use_original_lookup():
    class AccessProbe(Plain):
        def __getattribute__(self, name):
            return super().__getattribute__(name)
    class Meta(type):
        pass
    class MetaProbe(Plain, metaclass=Meta):
        pass
    for probe in (AccessProbe(), MetaProbe()):
        handle = install(probe)
        try:
            assert cache._class_presence_plan(type(probe)) is None
            probe.eps = .5
            assert scalars(handle) == (('eps', .5),)
            require_stale(handle)
        finally:
            handle.close()


def test_registered_parameter_buffer_module_and_none_are_present():
    probe = Plain()
    probe.register_parameter('eps', nn.Parameter(torch.tensor(.125)))
    probe.register_buffer('patch_size', torch.tensor(14, dtype=torch.int64))
    probe.add_module('embed_dim', Plain())
    probe.add_module('num_heads', None)
    handle = install(probe)
    try:
        fields = dict(scalars(handle))
        assert fields['eps'] is probe.eps and fields['patch_size'] is probe.patch_size
        assert fields['embed_dim'] is probe.embed_dim and fields['num_heads'] is None
        del probe.num_heads
        assert 'num_heads' not in dict(scalars(handle))
        require_stale(handle)
        probe.add_module('num_heads', None)
        probe.register_buffer('use_reentrant', torch.tensor(1, dtype=torch.int64))
        assert 'use_reentrant' in dict(scalars(handle))
        require_stale(handle)
    finally:
        handle.close()


def test_exact_ordereddict_registries_use_fast_lookup():
    probe = Plain()
    for name in ('_parameters', '_buffers', '_modules'):
        setattr(probe, name, OrderedDict(getattr(probe, name)))
    probe.register_parameter('eps', None)
    probe.register_buffer('patch_size', torch.tensor(14, dtype=torch.int64))
    probe.add_module('num_heads', None)
    lookup = cache._instance_presence_lookup(probe, cache._class_presence_plan(type(probe)))
    assert lookup is not None
    for key in ('eps', 'patch_size', 'num_heads'):
        assert cache._has_module_field(probe, key, lookup) is True
    assert cache._has_module_field(probe, 'chunked_blocks', lookup) is False
    del probe.num_heads
    # The lookup retains live dictionaries, so an instance change is never cached.
    assert cache._has_module_field(probe, 'num_heads', lookup) is False


@pytest.mark.parametrize('base', [dict, OrderedDict])
def test_custom_registry_subclasses_keep_membership_behavior(base):
    class Registry(base):
        def __init__(self, values):
            super().__init__(values)
            self.calls = 0

        def __contains__(self, key):
            self.calls += 1
            return super().__contains__(key)

    probe = Plain()
    probe._buffers = Registry(probe._buffers)
    probe.register_buffer('eps', torch.tensor(.125))
    lookup = cache._instance_presence_lookup(probe, cache._class_presence_plan(type(probe)))
    assert lookup is None
    probe._buffers.calls = 0
    assert cache._has_module_field(probe, 'eps', lookup) is True
    assert probe._buffers.calls == 1
