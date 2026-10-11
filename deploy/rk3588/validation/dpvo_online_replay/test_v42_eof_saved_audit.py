"""Tiny generated saved-audit boundaries; no actual neural arrays or model execution."""
import importlib.util
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PATH = Path(os.environ.get('V42_EOF_AUDITOR', HERE/'audit-v42-eof-export.py'))
if not PATH.is_file():
    PATH = HERE.parent/'deploy/rk3588/validation/dpvo_online_replay/audit-v42-eof-export.py'
spec = importlib.util.spec_from_file_location('_v42_saved_auditor_synthetic', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_fixed_absolute_boundary_and_json_zero_reference():
    b = np.zeros(1, np.float32)
    inside = np.asarray([np.float32(1e-5)], np.float32)
    outside = np.nextafter(inside, np.float32(np.inf))
    assert audit.compare(inside, b)['gate_passed']
    rejected = audit.compare(outside, b)
    assert rejected['failed_elements'] == 1 and rejected['relative_L2'] is None
    json.dumps(rejected, allow_nan=False)


def test_fixed_relative_boundary():
    b = np.ones(1, np.float32)
    limit = 1+1e-5+1e-4
    below = np.float32(limit)
    if float(below) > limit: below = np.nextafter(below, np.float32(-np.inf))
    assert audit.compare(np.asarray([below]), b)['gate_passed']
    assert not audit.compare(np.asarray([np.nextafter(below, np.float32(np.inf))]), b)['gate_passed']


@pytest.mark.parametrize('value', [np.asarray([np.nan],np.float32),np.asarray([np.inf],np.float32),
                                  np.ones(1,np.float64),np.ones(2,np.float32)])
def test_bad_saved_array_rejected(value):
    with pytest.raises(ValueError): audit.compare(value, np.ones(1,np.float32))


def graph():
    # A complete source and another source in deliberately reversed original order.
    ii = np.asarray([11]*48+[10], np.int64)
    jj = np.asarray([t for t in (9,10,11) for _ in range(16)]+[8], np.int64)
    kk = np.asarray([176+p for _ in range(3) for p in range(16)]+[160], np.int64)
    return {n:v[::-1].copy() for n,v in zip(('ii','jj','kk'),(ii,jj,kk))}


def test_complete_selection_preserves_original_reordered_edge_indices():
    values = graph()
    indices = audit.derive_source_indices(values)
    assert indices.shape == (1,3,16) and indices.dtype == np.int64
    assert np.array_equal(indices.ravel(), np.arange(48,0,-1))
    assert np.array_equal(values['kk'][indices], np.broadcast_to(np.arange(176,192),(1,3,16)))


@pytest.mark.parametrize('change', ['missing','duplicate','wrong_source','integer_width'])
def test_selection_rejects_missing_duplicate_or_changed_domain(change):
    values = graph()
    if change == 'missing': values = {k:v[:-1] for k,v in values.items()}
    elif change == 'duplicate':
        for key in values: values[key][1] = values[key][2]
    elif change == 'wrong_source': values['ii'][1] = 10
    else: values['kk'] = values['kk'].astype(np.int32)
    with pytest.raises(ValueError): audit.derive_source_indices(values)


def test_eight_complete_index_entries_preserved_and_tampering_rejected():
    old = dict(models={key:dict(model=dict(sha256='synthetic-'+key),lineage='original') for key in audit.OLD_KEYS},
               candidate_accepted=False,parent_manifest_sha256=audit.PARENT_SHA,additions_manifest_sha256='synthetic-addition')
    new = deepcopy(old)
    new.update(old_online_compile_index_sha256=audit.OLD_INDEX_SHA,eof_manifest_sha256=audit.EXPORT_SHA)
    new['models'][audit.MODEL_KEY] = dict(model=dict(sha256='synthetic-new'))
    audit.validate_index_retention(old, new)
    new['models'][audit.OLD_KEYS[0]]['lineage'] = 'relabelled'
    with pytest.raises(ValueError): audit.validate_index_retention(old, new)
