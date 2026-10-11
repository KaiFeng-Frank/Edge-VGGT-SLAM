"""Synthetic metadata/array guards only; no real preprocessing or PTQ."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
BASE=Path(os.environ.get("DPVO_HALF_PTQ_HELPERS",HERE if (HERE/"compile-v39-half-int8-board.py").is_file()
                      else HERE.parent/"deploy/rk3588/validation/dpvo_encoder_halfres"))
spec=importlib.util.spec_from_file_location("_v39_PTQ_tests",BASE/"compile-v39-half-int8-board.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def meta(a):
    return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,tensor_sha256=hashlib.sha256(a.tobytes(order="C")).hexdigest())


def clean():
    return dict(source_pins={"synthetic":"a"*64},source_hashes_after={"synthetic":"a"*64},sources_and_inputs_unchanged=True,cleanup_errors=[])


def probe():
    gate=dict(rtol=1e-4,atol=1e-5,gate_passed=True,failed_elements=0)
    rows=[dict(index=i,source_index=i,image_array={"shape":[1,3,232,304]},index_array={"shape":[16]},
               original={"versus_new_size_original_CPU":{"fmap":gate.copy(),"imap":gate.copy()}},
               reciprocal={"versus_new_size_original_CPU":{"fmap":gate.copy(),"context":gate.copy()}}) for i in range(3)]
    return dict(**clean(),schema="v39_DPVO_half_encoder_board_probe",mode="probe",
        status="completed_half_M16_CPU_teacher_and_ORT_diagnostic_not_VO",script_sha256=m.PROBE_SHA,
        policy_sha256=m.POLICY_SHA,rewrite_sha256="b"*64,preprocessing={"synthetic":True},rng_policy={"synthetic":True},
        normalization_applications_per_image=1,old_resolution_outputs_used_as_new_gold=False,
        original_FP32_gate=dict(rtol=1e-4,atol=1e-5),state_before={str(i):i for i in range(94)},state_after={str(i):i for i in range(94)},
        state_unchanged=True,shape_preservation_rechecked=True,model_forward_executed=True,providers=["CPUExecutionProvider"],
        graph_optimization="ORT_DISABLE_ALL",cpu_affinity=[4,5,6,7],affinity_restored=True,session_references_cleared=True,
        historical_parent_FP32_gate_passed=False,historical_parent_rejection_retained=True,candidate_accepted=False,
        calls=rows,samples=copy.deepcopy(rows),strict_gate_each_method={"original":True,"reciprocal":True})


def test_valid_new_size_actual_schema_gate():
    p=probe();m.probe_guard(p,dict(preprocessing={"synthetic":True},rng={"synthetic":True}),"b"*64,"a"*64)


@pytest.mark.parametrize("field",["element_gate","old_gold","summary_gate","state_bytes"])
def test_probe_rejects_failed_or_mixed_new_gold(field):
    p=probe()
    if field=="element_gate":
        p["calls"][1]["reciprocal"]["versus_new_size_original_CPU"]["context"]["gate_passed"]=False
        p["samples"]=copy.deepcopy(p["calls"])
    elif field=="old_gold":p["old_resolution_outputs_used_as_new_gold"]=True
    elif field=="summary_gate":p["strict_gate_each_method"]["reciprocal"]=False
    else:p["state_after"]["0"]=-1
    with pytest.raises(ValueError):m.probe_guard(p,dict(preprocessing={"synthetic":True},rng={"synthetic":True}),"b"*64,"a"*64)


@pytest.mark.parametrize("machine,compat",[("x86_64",b"rockchip,rk3588\0"),("aarch64",b"other,soc\0")])
def test_host_PTQ_guard_precedes_any_SDK(machine,compat):
    with pytest.raises(ValueError,match="Host PTQ prohibited"):m.board_guard(machine,"Linux",compat)


def test_new_calibration_arrays_reject_old_resolution_or_float_indices():
    image=np.zeros((1,3,232,304),np.float32);indices=np.full(16,77,np.int64)
    m.validate_arrays(image,indices)
    with pytest.raises(ValueError,match="half image"):m.validate_arrays(np.zeros((1,3,464,608),np.float32),indices)
    with pytest.raises(ValueError,match="INT64"):m.validate_arrays(image,indices.astype(np.float32))


def calibration_fixture(directory):
    directory.mkdir()
    image=np.zeros((1,3,232,304),np.float32);centers=np.full((32,16,2),2,np.int64)
    indices=centers[0,:,1]*76+centers[0,:,0]
    data={"centers_xy":centers,"consumed_inverse_depth":np.full((32,16),.5,np.float32),
          "state_before":np.zeros((32,4),np.uint8),"state_after_centers":np.ones((32,4),np.uint8),
          "state_after_depth":np.zeros((32,4),np.uint8)}
    def info(path):return dict(file=path.name,bytes=path.stat().st_size,sha256=m.sha(path))
    rows=[]
    for i in range(32):
        files=[]
        for name,a in ((f"image-{i:02d}.npy",image),(f"indices-{i:02d}.npy",indices)):
            path=directory/name;np.save(path,a,allow_pickle=False);files.append(info(path))
        rows.append(dict(index=i,source_index=256+16*i,frame_id=str(i),source_image_sha256="c"*64,
                         input_files=files,image_array=meta(image),index_array=meta(indices),
                         RNG={**{k:meta(a[i]) for k,a in data.items()},"patch_indices":meta(indices)}))
    sampling=directory/"sampling.npz";np.savez(sampling,**data)
    dataset=directory/"dataset.txt";dataset.write_text("\n".join(" ".join(f["file"] for f in r["input_files"]) for r in rows)+"\n")
    receipt=dict(samples=rows,dataset=info(dataset),sampling_artifact={**info(sampling),"arrays":{k:meta(a) for k,a in data.items()}})
    return receipt,m.sha(dataset)


def test_dataset_checks_32_new_inputs_and_order(tmp_path):
    cal,digest=calibration_fixture(tmp_path/"cal")
    out=tmp_path/"out";out.mkdir();pins={}
    dataset,rows=m.prepare_dataset(tmp_path/"cal",cal,digest,out,pins,meta)
    assert len(rows)==32 and len(pins)==67 and dataset.is_file()
    assert dataset.read_text().splitlines()[0].split()==[str(tmp_path/"cal/image-00.npy"),str(tmp_path/"cal/indices-00.npy")]
    bad=copy.deepcopy(cal);bad["samples"][0]["input_files"].reverse()
    with pytest.raises(ValueError,match="order mismatch"):
        m.prepare_dataset(tmp_path/"cal",bad,digest,out,{},meta)


def test_calibration_receipt_cannot_overlap_evaluation():
    v=dict(**clean(),schema="v39_DPVO_half_encoder_board_calibration",mode="calibration",
        status="generated_half_M16_calibration_not_quantized_or_validated",script_sha256=m.PROBE_SHA,
        policy_sha256=m.POLICY_SHA,rewrite_sha256="b"*64,preprocessing={},rng_policy={},model_forward_executed=False,
        GT_read=False,GT_or_content_selection=False,normalization_applications_per_image=1,calls=[],
        calibration_source_indices=m.CAL_INDICES,evaluation_source_indices=list(range(192)),
        samples=[{"source_index":i} for i in m.CAL_INDICES],candidate_accepted=False,historical_parent_rejection_retained=True,historical_parent_FP32_gate_passed=False)
    policy=dict(preprocessing={},rng={});m.calibration_guard(v,policy,"b"*64)
    v["samples"][0]["source_index"]=0
    with pytest.raises(ValueError,match="disjoint"):m.calibration_guard(v,policy,"b"*64)
