"""Small independent saved-array guards; no images, neural model or RNG replay."""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "audit-v36-dpvo-calibration.py"
if not SOURCE.is_file():
    SOURCE = HERE.parent / "deploy/rk3588/validation/board_calibration/audit-v36-dpvo-calibration.py"
spec = importlib.util.spec_from_file_location("v36_calibration_audit", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    centers = np.empty((32, 96, 2), np.int64)
    centers[..., 0] = np.arange(1, 97)
    centers[..., 1] = 2
    before = np.array([[2*i] * 4 for i in range(32)], np.uint8)
    arrays = dict(centers=centers, consumed_inverse_depth=np.full((32, 96), .25, np.float32),
                  state_before=before, state_after_centers=before+1, state_after_depth=before+2)
    records = [dict(RNG=dict(arrays={k: audit.info(v[i]) for k, v in arrays.items()}),
                    index_array=audit.info(centers[i, :, 1]*np.int64(152)+centers[i, :, 0])) for i in range(32)]
    metadata = {k: audit.info(v) for k, v in arrays.items()}
    return arrays, records, metadata


def test_complete_sampling_fixture_and_ordered_indices_pass():
    arrays, records, metadata = fixture()
    checks = audit.Checks()
    indices = audit.check_sampling(checks, arrays, records, metadata)
    assert not checks.failures
    assert len(indices) == 32
    assert indices[0].dtype == np.int64
    np.testing.assert_array_equal(indices[0], np.arange(305, 401, dtype=np.int64))


def test_rng_join_tamper_fails_even_when_all_metadata_rehashed():
    arrays, records, metadata = fixture()
    arrays['state_before'][8] = 99
    records[8]['RNG']['arrays']['state_before'] = audit.info(arrays['state_before'][8])
    metadata['state_before'] = audit.info(arrays['state_before'])
    checks = audit.Checks()
    audit.check_sampling(checks, arrays, records, metadata)
    assert checks.failures == ['RNG continuity 7->8']


def test_index_order_tamper_fails_without_reading_npy_body():
    arrays, records, metadata = fixture()
    wrong = (arrays['centers'][0, :, 1]*np.int64(152)+arrays['centers'][0, :, 0])[::-1]
    records[0]['index_array'] = audit.info(wrong)
    checks = audit.Checks()
    audit.check_sampling(checks, arrays, records, metadata)
    assert checks.failures == ['sample 0 derived indices: tensor_sha256']


def test_raw_array_metadata_rejects_float_indices():
    checks = audit.Checks()
    value = np.array([1, 2], np.int64)
    checks.array(value, audit.info(value.astype(np.float32)), 'indices')
    assert checks.failures == ['indices: dtype', 'indices: bytes', 'indices: tensor_sha256']


def test_dataset_input_swap_is_rejected():
    text = ''.join(f'image-{i:02d}.npy indices-{i:02d}.npy\n' for i in range(32))
    checks = audit.Checks()
    audit.check_dataset(checks, text)
    assert not checks.failures
    audit.check_dataset(checks, text.replace('image-00.npy indices-00.npy', 'indices-00.npy image-00.npy'))
    assert checks.failures == ['Exact dataset rows, relative filenames and input order']
