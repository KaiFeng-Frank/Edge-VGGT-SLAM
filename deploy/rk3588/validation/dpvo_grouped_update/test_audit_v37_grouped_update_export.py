"""Tiny saved-value/topology tests; no learned model, ORT or real arrays."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

LOCAL=Path(__file__).with_name("audit-v37-grouped-update-export.py")
PUBLIC=Path(__file__).resolve().parents[1]/"deploy/rk3588/validation/dpvo_grouped_update/audit-v37-grouped-update-export.py"
spec=importlib.util.spec_from_file_location("v37_saved_group_auditor",LOCAL if LOCAL.is_file() else PUBLIC)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def topology():
    rows=[(source,target,source*2+patch) for source,targets in ((0,(0,2)),(1,(0,1,2)),(3,(1,3)))
          for target in targets for patch in range(2)]
    values=np.array(rows,np.int64)[np.random.default_rng(37).permutation(len(rows))]
    return values[:,0].copy(),values[:,1].copy(),values[:,2].copy()


def test_equal_degree_sources_keep_true_batch_and_signed_zero_roundtrip():
    ii,jj,kk=topology();batches=audit.independent_batches(ii,jj,kk,patches=2)
    assert [b["source_slots"] for b in batches]==[[0,3],[1]]
    assert [b["degree"] for b in batches]==[2,3]
    values=np.arange(ii.size*3,dtype=np.float32).reshape(1,ii.size,3)
    values[0,0,0]=-0.
    packed=[audit.pack(values,b) for b in batches]
    assert all(v.flags.c_contiguous for v in packed)
    restored=audit.scatter(packed,batches,ii.size,3)
    assert restored.tobytes()==values.tobytes()


@pytest.mark.parametrize("fault",("duplicate","missing","wrong_owner"))
def test_incomplete_or_mixed_source_cartesian_plan_rejected(fault):
    ii,jj,kk=topology()
    if fault=="duplicate":ii[-1],jj[-1],kk[-1]=ii[0],jj[0],kk[0]
    elif fault=="missing":ii,jj,kk=ii[:-1],jj[:-1],kk[:-1]
    else:ii[0]=(ii[0]+1)%4
    with pytest.raises(ValueError):audit.independent_batches(ii,jj,kk,patches=2)


def test_partial_group_scatter_cannot_return_a_filled_full_output():
    ii,jj,kk=topology();batches=audit.independent_batches(ii,jj,kk,patches=2)
    value=np.ones((1,ii.size,2),np.float32)
    with pytest.raises(ValueError,match="All output edges"):
        audit.scatter([audit.pack(value,batches[0])],batches,ii.size,2)


def test_asymmetric_threshold_not_replaced_by_global_relative_l2():
    reference=np.zeros((1,3,2),np.float32);actual=reference.copy()
    actual.flat[0]=np.nextafter(np.float32(1e-5),np.float32(0))
    assert audit.comparison(actual,reference)["gate_passed"]
    actual.flat[0]=np.nextafter(np.float32(1e-5),np.float32(np.inf))
    stats=audit.comparison(actual,reference)
    assert stats["failed_elements"]==1 and stats["relative_L2"] is None
    assert stats["max_normalized_error"]>1 and not stats["gate_passed"]
    assert stats["failing_elements"][0]["coordinate"]==[0,0,0]
    audit.json.dumps(stats,allow_nan=False)
    recorded={k:stats[k] for k in ("finite","gate_passed","failed_elements","bitwise_equal","rtol","atol","max_abs","relative_L2")}
    assert audit.stats_match(stats,recorded)
    recorded["failed_elements"]=0
    assert not audit.stats_match(stats,recorded)


def test_original_parameter_bytes_match_only_exact_reshape_and_all_covered():
    values={"kernel":np.arange(6,dtype=np.float32).reshape(2,3),"affine":np.array([.2,.4],np.float32)}
    parameters={k:audit.meta(v) for k,v in values.items()}
    tensors=[NS(name="weight",value=values["kernel"].reshape(2,3,1,1),external_data=[],data_location=0),
             NS(name="gamma",value=values["affine"].reshape(1,2,1,1),external_data=[],data_location=0)]
    graph=NS(graph=NS(initializer=tensors));onnx=NS(TensorProto=NS(EXTERNAL=1),numpy_helper=NS(to_array=lambda t:t.value))
    info=audit.parameter_coverage(graph,parameters,onnx)
    assert info["all_original_parameters_covered"] and info["original_parameter_count"]==2
    tensors[0].value=tensors[0].value.copy();tensors[0].value.flat[0]+=.01
    with pytest.raises(ValueError,match="not original parameter"):
        audit.parameter_coverage(graph,parameters,onnx)
