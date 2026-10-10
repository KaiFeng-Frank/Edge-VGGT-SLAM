"""Small synthetic guards for the board-only v21 numerical regression helper."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (Path(__file__).resolve().parents[1] / "deploy/rk3588/validation"
          / "check-v21-model-execution-board.py")
spec = importlib.util.spec_from_file_location("v21_execution_regression", SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def original_arrays():
    # Synthetic constant arrays only: no image decoding or model execution.
    return {"extrinsic":np.ones((1,2,3,4),np.float32),
            "intrinsic":np.ones((1,2,3,3),np.float32),
            "depth":np.ones((1,2,84,112,1),np.float32),
            "depth_conf":np.ones((1,2,84,112),np.float32)}


def reference(kind=helper.REFERENCE_KINDS[1]):
    ids = [1305031102.175304,1305031102.443271]
    images = [dict(frame_id=fid,timestamp=fid,samefile_validated=True,
                   schedule_token=dict(frame_id=fid,timestamp=fid)) for fid in ids]
    pair = dict(revision=1,frame_ids=ids,input_tensor={"shape":[2,3,84,112]},
                arrays={key:helper.array_info(value) for key,value in original_arrays().items()})
    return dict(kind=kind,status="completed_fixed_input_diagnostic",image_size=112,
                precision="float32",device="cpu",original_weights=True,ORB_used=False,
                npu_used=False,checkpoint=dict(bytes=helper.CHECKPOINT_SIZE,sha256=helper.CHECKPOINT_SHA),
                pairs=[pair],selected_inputs=[dict(revision=1,frame_ids=ids,images=images)])


@pytest.mark.parametrize("kind",helper.REFERENCE_KINDS)
def test_both_original_capture_receipts_are_explicitly_supported(kind):
    receipt = reference(kind)
    pair, selection = helper.validate_reference(receipt)
    assert pair["frame_ids"] == selection["frame_ids"]
    assert len(pair["frame_ids"]) == 2


@pytest.mark.parametrize("key,value",[
    ("kind","unknown"),("status","started"),("image_size",518),
    ("precision","float16"),("device","npu"),("original_weights",False),
    ("ORB_used",True),("npu_used",True),
])
def test_invalid_reference_mode_is_not_an_original_oracle(key,value):
    receipt = reference()
    receipt[key] = value
    with pytest.raises(ValueError):
        helper.validate_reference(receipt)


@pytest.mark.parametrize("mutation",[
    lambda r:r["checkpoint"].update(sha256="0"*64),
    lambda r:r["checkpoint"].update(bytes=1),
    lambda r:r["pairs"][0]["frame_ids"].__setitem__(1,float("nan")),
    lambda r:r["selected_inputs"][0]["images"][1].update(frame_id=1305031102.443272),
    lambda r:r["selected_inputs"][0]["images"][1]["schedule_token"].update(timestamp=1.),
    lambda r:r["selected_inputs"][0]["images"][1].update(samefile_validated=False),
    lambda r:r["pairs"][0]["input_tensor"].update(shape=[2,3,112,112]),
    lambda r:r["pairs"][0]["arrays"]["depth"].update(shape=[1,2,112,84,1]),
    lambda r:r["pairs"][0]["arrays"]["intrinsic"].update(dtype="float64"),
])
def test_reference_source_identity_shape_and_weight_mutants_fail(mutation):
    receipt = reference()
    mutation(receipt)
    with pytest.raises(ValueError):
        helper.validate_reference(receipt)


def test_saved_original_arrays_keep_double_precision_ids(tmp_path):
    receipt = reference()
    pair, _ = helper.validate_reference(receipt)
    file = tmp_path / "reference.npz"
    np.savez(file,**original_arrays(),frame_ids=np.asarray(pair["frame_ids"],np.float64))
    got = helper.load_original_predictions(file,pair)
    assert all(got[key].shape == original_arrays()[key].shape[1:] for key in helper.KEYS)
    np.savez(file,**original_arrays(),frame_ids=np.asarray(pair["frame_ids"],np.float32))
    with pytest.raises(ValueError,match="doubles exactly"):
        helper.load_original_predictions(file,pair)


def test_saved_prediction_raw_sha_change_fails_before_oracle_use(tmp_path):
    pair, _ = helper.validate_reference(reference())
    changed = original_arrays()
    changed["depth"][0,1,0,0,0] = 2.
    file = tmp_path / "reference.npz"
    np.savez(file,**changed,frame_ids=np.asarray(pair["frame_ids"],np.float64))
    with pytest.raises(ValueError,match="array identity differs"):
        helper.load_original_predictions(file,pair)


def test_exact_gate_rejects_one_ulp_without_relaxing_tolerance():
    a = np.ones((2,3,4),np.float32)
    b = a.copy()
    assert helper.exact_comparison(a,b)["array_equal"] is True
    b[0,0,0] = np.nextafter(b[0,0,0],np.float32(2.))
    assert helper.exact_comparison(a,b)["array_equal"] is False
    actual = {key:a.copy() for key in helper.KEYS}
    expected = {key:a.copy() for key in helper.KEYS}
    expected["extrinsic"] = b
    with pytest.raises(ValueError,match="exact gate failed"):
        helper.compare_predictions(actual,expected)


def test_bitwise_gate_distinguishes_signed_zero():
    actual = {key:np.zeros((2,3,4),np.float32) for key in helper.KEYS}
    expected = {key:value.copy() for key,value in actual.items()}
    expected["extrinsic"][0,0,0] = np.float32(-0.)
    row = helper.exact_comparison(actual["extrinsic"],expected["extrinsic"])
    assert row["array_equal"] is True
    assert row["bitwise_equal"] is False
    with pytest.raises(ValueError,match="exact gate failed"):
        helper.compare_predictions(actual,expected)


@pytest.mark.parametrize("bad",[
    np.ones((1,3),np.float32),np.ones((2,3),np.float64),
    np.full((2,3),np.nan,np.float32),np.full((2,3),np.inf,np.float32),
])
def test_bad_canonical_arrays_cannot_pass_exact_gate(bad):
    with pytest.raises(ValueError):
        helper.exact_comparison(bad,np.ones((2,3),np.float32))


def test_optional_tokens_preserve_none_and_real_synthetic_values():
    assert helper.compare_opaque_tokens(None,None)["both_none"] is True
    tokens = np.ones((1,2,3),np.float32)
    assert helper.compare_opaque_tokens(tokens,tokens.copy())["array_equal"] is True
    with pytest.raises(ValueError):
        helper.compare_opaque_tokens(None,tokens)


def test_board_guard_rejects_host_and_nonexperiment_outputs():
    with pytest.raises(ValueError,match="aarch64 board only"):
        helper.board_guard("x86_64",helper.SOURCE,helper.BASE/"experiments/v21-new")
    with pytest.raises(ValueError,match="under board BASE/experiments"):
        helper.board_guard("aarch64",helper.SOURCE,helper.BASE/"results/v21-new")
    with pytest.raises(ValueError,match="checkout"):
        helper.board_guard("aarch64",Path("/tmp/host-checkout"),helper.BASE/"experiments/v21-new")


def test_board_guard_never_overwrites_existing_output(tmp_path,monkeypatch):
    monkeypatch.setattr(helper,"BASE",tmp_path)
    source = tmp_path/"src"
    output = tmp_path/"experiments/old"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        helper.board_guard("aarch64",source,output)


class TinyTensor:
    """Only the old codec boundary's tiny synthetic tensor methods."""
    def __init__(self,array):
        self.array = np.asarray(array,np.float32)
        self.shape = self.array.shape
        self.device = SimpleNamespace(type="cpu")
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


def test_direct_codec_uses_actual_hw_and_loop_ratio_is_not_probability():
    def decoder(pose,hw):
        assert hw == (84,112)
        assert pose.shape == (1,2,9)
        return TinyTensor(np.ones((1,2,3,4))),TinyTensor(np.ones((1,2,3,3)))
    raw = {key:TinyTensor(value) for key,value in original_arrays().items() if key.startswith("depth")}
    raw.update(pose_enc=TinyTensor(np.ones((1,2,9))),image_match_ratio=1.00434)
    got = helper.canonical_direct(raw,(84,112),decoder)
    assert got["image_match_ratio"] == 1.00434
    assert got["intrinsic"].shape == (2,3,3)
    assert got["depth"].shape == (2,84,112,1)


def test_direct_codec_rejects_missing_explicit_batch_dimension():
    raw = dict(pose_enc=TinyTensor(np.ones((2,9))),
               depth=TinyTensor(np.ones((2,14,14,1))),
               depth_conf=TinyTensor(np.ones((2,14,14))))
    def decoder(pose,hw):
        return TinyTensor(np.ones((2,3,4))),TinyTensor(np.ones((2,3,3)))
    with pytest.raises(ValueError,match="B=1"):
        helper.canonical_direct(raw,(14,14),decoder)


class TinySubmap:
    def __init__(self,images,paths,ids):
        self.images = TinyTensor(images)
        self.img_names = paths
        self.ids = ids
    def get_all_frames(self): return self.images
    def get_id(self): return 0
    def get_frame_ids(self): return self.ids
    def get_last_non_loop_frame_index(self): return 1


def test_actual_solver_submap_must_share_exact_prepared_pixels_and_ids():
    images = np.ones((2,3,14,14),np.float32)
    ids = [1305031102.175304,1305031102.443271]
    paths = ["first.png","second.png"]
    submap = TinySubmap(images,paths,ids)
    evidence = helper.check_prepared_submap(submap,images,paths,ids)
    assert evidence["shares_prepared_storage"] is True
    assert evidence["frame_ids"] == ids
    submap.images = TinyTensor(images.copy())
    with pytest.raises(ValueError,match="share prepared CPU pixels"):
        helper.check_prepared_submap(submap,images,paths,ids)


def test_submap_float32_timestamp_rounding_is_not_source_identity():
    images = np.ones((2,3,14,14),np.float32)
    ids = [1305031102.175304,1305031102.443271]
    paths = ["first.png","second.png"]
    submap = TinySubmap(images,paths,np.asarray(ids,np.float32).tolist())
    with pytest.raises(ValueError,match="source doubles/order"):
        helper.check_prepared_submap(submap,images,paths,ids)
