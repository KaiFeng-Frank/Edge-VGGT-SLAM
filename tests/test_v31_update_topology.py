"""Integer topology only: frozen-builder oracle, bounded cache and ownership."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HELPER = HERE/'v31_update_topology.py'
if not HELPER.is_file():
    HELPER = HERE.parent/'deploy/rk3588/validation/dpvo_dynamic_update/v31_update_topology.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


module = load(HELPER, '_v31_topology_under_test')


@pytest.fixture(scope='module')
def reference():
    path = Path(os.environ.get('DPVO_V28_TOPOLOGY_REFERENCE',
        str(HERE.parent/'deploy/rk3588/validation/dpvo_update_export.py')))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == module.ORIGINAL_ADAPTER_SHA256
    # Imports definitions only. No Update/checkpoint/image/feature inference.
    return load(path, '_v31_frozen_topology_oracle')


def inputs(seed=0, edges=39):
    rng = np.random.default_rng(seed)
    return (rng.integers(-3, 4, edges, dtype=np.int64),
            rng.integers(-2, 4, edges, dtype=np.int64),
            rng.integers(-4, 4, edges, dtype=np.int64))


def assert_same(actual, expected):
    for name, value in actual.numpy_dict().items():
        oracle = getattr(expected, name).numpy()
        assert value.shape == oracle.shape and value.dtype == oracle.dtype
        assert np.array_equal(value, oracle), name
        assert getattr(actual, name).flags.c_contiguous and getattr(actual, name).flags.owndata


def test_input_names_and_ten_ordered_arrays_match_frozen_contract(reference):
    assert module.INPUT_NAMES == reference.INPUT_NAMES
    plan = module.build_topology(*inputs(edges=1))
    assert len(plan.arrays()) == 10
    for name, value in zip(module.INPUT_NAMES[3:], plan.arrays()):
        assert value is getattr(plan, name)


@pytest.mark.parametrize('seed', range(16))
def test_random_ragged_tied_graphs_exactly_match_original_integer_builder(reference, seed):
    ids = inputs(seed, edges=1+5*seed)
    actual = module.build_topology(*ids)
    assert_same(actual, reference.build_topology(*ids))
    # Original edge inputs were not sorted/renumbered/deduplicated.
    for name, value in zip(('ii', 'jj', 'kk'), ids):
        assert np.array_equal(getattr(actual, name), value)


def test_int64_wraparound_label_collisions_and_extreme_sort_order(reference):
    i = np.iinfo(np.int64)
    ids = (np.array([0, 1, i.max, i.min, i.min+1, i.max-1, -1], np.int64),
           np.array([12345, 0, i.max, i.min, 7, -9, i.max], np.int64),
           np.array([i.max, i.max, i.min, i.min, 0, 0, -1], np.int64))
    with np.errstate(all='raise'):
        actual = module.build_topology(*ids)
    assert_same(actual, reference.build_topology(*ids))
    assert actual.ij_inverse[0] == actual.ij_inverse[1]


def test_tied_destinations_keep_input_row_order_within_each_patch(reference):
    ids = (np.zeros(7, np.int64), np.array([2, 1, 2, 1, 1, 0, 2], np.int64),
           np.array([3, 3, 3, 3, 3, 3, 9], np.int64))
    p = module.build_topology(*ids)
    assert_same(p, reference.build_topology(*ids))
    assert p.previous_indices[1] == 5 and p.previous_indices[3] == 1
    assert p.previous_indices[4] == 3 and p.previous_indices[0] == 4
    assert p.previous_valid[0, 5, 0] == 0 and p.following_valid[0, 2, 0] == 0


def test_extra_empty_groups_and_slot_padding_match_original_safe_prefix_masks(reference):
    ids = inputs(9, 27)
    small = module.build_topology(*ids)
    caps = {name: [getattr(small, name+'_indices').shape[0]+2,
                   getattr(small, name+'_indices').shape[1]+3] for name in ('kk', 'ij')}
    result = module.build_topology(*ids, group_caps=caps)
    assert_same(result, reference.build_topology(*ids, group_caps=caps))
    for name in ('kk', 'ij'):
        table, mask = getattr(result, name+'_indices'), getattr(result, name+'_valid')
        assert np.all(table[mask == 0] == 0) and np.all(mask[:, :-1] >= mask[:, 1:])
        assert not np.any(mask[-2:])


def test_noncontiguous_inputs_owned_results_and_artifact_copies(reference):
    backing = [np.repeat(a, 2) for a in inputs(3, 9)]
    ids = tuple(a[::2] for a in backing)
    result = module.build_topology(*ids)
    assert_same(result, reference.build_topology(*ids))
    artifact = result.numpy_dict()
    for a in backing:
        a.fill(99)
    assert not np.any(result.ii == 99)
    artifact['ii'].fill(88)
    assert not np.any(result.ii == 88)


@pytest.mark.parametrize('bad', [np.array([], np.int64), np.array([0], np.int32),
    np.array([0.], np.float64), np.array([True]), np.array([[0]], np.int64), [0]])
def test_invalid_input_type_or_empty_shape_rejected(bad):
    with pytest.raises(ValueError):
        module.build_topology(bad, np.array([0], np.int64), np.array([0], np.int64))


@pytest.mark.parametrize('caps', [{'other': (1, 1)}, {'kk': (0, 1)}, {'kk': (True, 1)},
    {'kk': (np.int64(1), 1)}, {'kk': (1,)}, {'kk': [1, 1, 1]}, [1, 1]])
def test_invalid_capacity_contract_rejected(caps):
    with pytest.raises(ValueError):
        module.build_topology(*inputs(edges=1), group_caps=caps)


def test_group_overflow_rejects_without_dropping_edges():
    ids = (np.array([0, 0, 0], np.int64), np.array([0, 0, 0], np.int64),
           np.array([0, 0, 1], np.int64))
    for capacity in ((1, 3), (2, 1)):
        with pytest.raises(ValueError, match='capacity overflow'):
            module.build_topology(*ids, group_caps={'kk': capacity})


def test_combined_table_budget_checked_before_either_table_allocation(monkeypatch):
    def forbidden(_):
        pytest.fail('Oversized group table was allocated')
    monkeypatch.setattr(module, '_table', forbidden)
    for caps, budget in [({'kk': (10**12, 10**12)}, 32 << 20), ({}, 23)]:
        with pytest.raises(ValueError, match='before allocation'):
            module.build_topology(*inputs(edges=1), group_caps=caps, max_table_bytes=budget)


def test_exact_budget_boundary_and_edge_limit():
    p = module.build_topology(*inputs(edges=1), max_table_bytes=24)
    assert p.kk_indices.shape == p.ij_indices.shape == (1, 1)
    with pytest.raises(ValueError, match='bounded edge shape'):
        module.build_topology(*inputs(edges=2), max_edges=1)


def test_cache_uses_exact_values_and_returns_no_aliases(reference):
    ids = inputs(5, 20)
    cache = module.TopologyCache()
    first = cache.get(*ids)
    first.previous_indices.fill(99); first.ii.fill(99)
    second = cache.get(*(a.copy() for a in ids))
    assert_same(second, reference.build_topology(*ids))
    assert cache.metrics()['hits'] == 1 and cache.metrics()['builds'] == 1
    ids[1][0] += 1  # Same allocation and shape, different topology values.
    changed = cache.get(*ids)
    assert_same(changed, reference.build_topology(*ids))
    assert cache.metrics()['misses'] == 2
    assert not np.shares_memory(first.kk_indices, second.kk_indices)


def test_labels_alone_are_not_a_cache_key():
    cache = module.TopologyCache()
    kk = np.array([0, 1], np.int64)
    a = cache.get(np.array([0, 1], np.int64), np.array([12345, 0], np.int64), kk)
    b = cache.get(np.array([1, 0], np.int64), np.array([0, 12345], np.int64), kk)
    assert np.array_equal(a.ij_labels, b.ij_labels)
    assert cache.metrics()['misses'] == 2 and cache.metrics()['hits'] == 0


def test_cache_capacity_tuple_values_and_mapping_order():
    ids = inputs(edges=1); cache = module.TopologyCache()
    cache.get(*ids, group_caps={'kk': (2, 3), 'ij': (4, 2)})
    cache.get(*ids, group_caps={'ij': [4, 2], 'kk': [2, 3]})
    assert cache.metrics()['hits'] == 1
    cache.get(*ids, group_caps={'kk': (3, 3), 'ij': (4, 2)})
    assert cache.metrics()['misses'] == 2


def test_lru_entry_cap_is_bounded_and_hit_refreshes_recency():
    cache = module.TopologyCache(max_entries=2)
    a, b, c = inputs(1, 8), inputs(2, 8), inputs(3, 8)
    for ids in (a, b, a, c):
        cache.get(*ids)
    assert cache.metrics()['hits'] == 1 and cache.metrics()['evictions'] == 1
    cache.get(*a)
    assert cache.metrics()['hits'] == 2
    cache.get(*b)
    assert cache.metrics()['misses'] == 4 and cache.metrics()['current_entries'] == 2


def test_byte_budget_includes_key_arrays_and_oversize_bypass():
    a = (np.array([0], np.int64),)*3
    b = (np.array([1], np.int64),)*3
    size = module.build_topology(*a).nbytes
    cache = module.TopologyCache(max_entries=4, max_bytes=size)
    cache.get(*a); cache.get(*b)
    assert cache.metrics()['current_payload_bytes'] == size
    assert cache.metrics()['current_entries'] == 1 and cache.metrics()['evictions'] == 1
    tiny = module.TopologyCache(max_bytes=1)
    assert np.array_equal(tiny.get(*a).ii, a[0])
    tiny.get(*a)
    assert tiny.metrics()['current_payload_bytes'] == 0 and tiny.metrics()['oversized_not_cached'] == 2


def test_metrics_deep_copy_clear_and_close_keep_counts():
    cache = module.TopologyCache(); ids = inputs(edges=3)
    cache.get(*ids); report = cache.metrics()
    report['last_call']['status'] = 'modified'
    assert cache.metrics()['last_call']['status'] == 'completed'
    assert cache.metrics()['last_call']['total_seconds'] >= 0
    cache.clear(); assert cache.metrics()['current_payload_bytes'] == 0
    cache.get(*ids); assert cache.metrics()['builds'] == 2
    cache.close(); cache.close()
    assert cache.metrics()['closed'] and cache.metrics()['builds'] == 2
    assert cache.metrics()['current_payload_bytes'] == 0
    with pytest.raises(ValueError, match='closed'):
        cache.get(*ids)


def test_invalid_request_does_not_poison_cached_template():
    cache = module.TopologyCache(); ids = inputs(edges=3)
    before = cache.get(*ids)
    with pytest.raises(ValueError):
        cache.get(*ids, group_caps={'kk': (1, 1)})
    after = cache.get(*ids)
    assert np.array_equal(before.kk_indices, after.kk_indices)
    assert cache.metrics()['failed_requests'] == 1 and cache.metrics()['hits'] == 1


def test_concurrent_cache_requests_serialized_without_output_aliases():
    cache = module.TopologyCache(); ids = inputs(18, 40)
    with ThreadPoolExecutor(max_workers=4) as executor:
        plans = list(executor.map(lambda _: cache.get(*ids), range(4)))
    assert cache.metrics()['builds'] == 1 and cache.metrics()['hits'] == 3
    for a, b in zip(plans, plans[1:]):
        assert np.array_equal(a.kk_indices, b.kk_indices)
        assert not np.shares_memory(a.kk_indices, b.kk_indices)
