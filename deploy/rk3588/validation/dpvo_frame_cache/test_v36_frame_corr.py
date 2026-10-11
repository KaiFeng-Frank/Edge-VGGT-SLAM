"""Tiny generated arrays only: scalar host protocol/numerics, not ARM evidence."""
import ctypes as ct
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import threading
import weakref

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PARENT = Path(os.environ.get("V36_PARENT_DIR", str(HERE.parent / "dpvo_native")))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


candidate = load(HERE / "dpvo_frame_corr.py", "frame_corr_v36_tiny")
parent = load(PARENT / "dpvo_corr.py", "frame_corr_v27_tiny")
oracle = load(PARENT / "corr_numpy_oracle.py", "frame_corr_numpy_tiny")


@pytest.fixture(scope="session")
def library(tmp_path_factory):
    candidate.verify_parent_sources(PARENT)
    directory = tmp_path_factory.mktemp("v36-native-synthetic")
    output = directory / "libdpvo_frame_corr_synthetic.so"
    command = [os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-fPIC", "-shared", "-fopenmp",
               "-fno-fast-math", "-ffp-contract=off", "-I", str(PARENT),
               str(HERE / "dpvo_frame_corr.cpp"), "-o", str(output)]
    subprocess.run(command, cwd=directory, check=True, capture_output=True, text=True)
    return output


def frames(n=3, channels=8, ph=2, pw=3, m=3, height=9, width=11, seed=47):
    rng = np.random.default_rng(seed)
    result = []
    for f in range(n):
        g = rng.normal(0, .3, (m, channels, ph, pw)).astype(np.float32)
        t = rng.normal(0, .3, (channels, height, width)).astype(np.float32)
        g.flags.writeable = t.flags.writeable = False
        result.append(dict(frame_id=f"source-{seed}-{f}", gmap=g, fmap=t))
    return result


def request(records, edges=5, seed=49):
    rng = np.random.default_rng(seed)
    m, _, ph, pw = records[0]["gmap"].shape
    kk = rng.integers(0, len(records)*m, edges, dtype=np.int64)
    ii = kk // m
    jj = rng.integers(0, len(records), edges, dtype=np.int64)
    xy = rng.uniform(-3.2, 12.5, (1, edges, 2, ph, pw)).astype(np.float32)
    return xy, dict(ii=ii, jj=jj, kk=kk, frames=tuple(records))


def factory(library, records, **kwargs):
    m, c, ph, pw = records[0]["gmap"].shape
    return candidate.PerFrameCorrelation(library, library_sha256=candidate.sha256(library),
        patch_count=m, channels=c, patch_shape=(ph, pw), threads=kwargs.pop("threads", 1), **kwargs)


def old_and_oracle(library, records, xy, args, independent=True):
    g = np.concatenate([r["gmap"] for r in records])[None]
    f = np.stack([r["fmap"] for r in records])[None]
    with parent.PreparedCorrelation(library, g, f, threads=1, cache_key="tiny-original") as old:
        reference, _ = old.correlate(xy, args["kk"], args["jj"], cache_key="tiny-original")
    numeric = oracle.tiny_full_oracle(g, f, xy, args["kk"], args["jj"]) if independent else None
    return reference, numeric


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("channels,ph,pw", [(8, 2, 3), (128, 3, 3)])
def test_fixed_order_fractional_borders_and_v27_exact(library, threads, channels, ph, pw):
    records = frames(channels=channels, ph=ph, pw=pw)
    xy, args = request(records, edges=3)
    xy[:, :, 0, 0, 0] = -.125
    xy[:, :, 1, 0, 0] = 8.75
    expected, independent = old_and_oracle(library, records, xy, args)
    with factory(library, records, threads=threads) as cache:
        first = cache(xy, **args)
        second = cache(xy, **args)
        assert first.tobytes() == expected.tobytes() == second.tobytes()
        np.testing.assert_allclose(first, independent, rtol=1e-4, atol=1e-5)
        assert not np.shares_memory(first, second)
        assert cache.metrics()["prepared_frames"] <= len(records)
        assert cache.metrics()["calls"][-1]["prepare_misses"] == 0
        assert cache.metrics()["calls"][-1]["native_query"]["native"]["neon"] == int("AArch64-NEON" in cache.build_info)
    assert first.flags.owndata and cache.metrics()["current_context_bytes"] == 0


def test_original_fast_m48_default_contract_with_tiny_spatial_maps(library):
    records = frames(n=2, channels=128, ph=3, pw=3, m=48, height=4, width=4)
    xy, args = request(records, edges=2)
    args.update(ii=np.array([1, 0], np.int64), kk=np.array([95, 47], np.int64), jj=np.array([0, 1], np.int64))
    with candidate.PerFrameCorrelation(library, library_sha256=candidate.sha256(library), threads=1) as cache:
        got = cache(xy, **args)
        assert got.shape == (1, 2, 882)
        assert got.tobytes() == old_and_oracle(library, records, xy, args, False)[0].tobytes()


def test_new_frame_only_prepares_once_and_graph_changes_reuse(library):
    records = frames()
    xy, args = request(records[:2], edges=2)
    args.update(ii=np.array([0, 1], np.int64), kk=np.array([0, 3], np.int64), jj=np.array([1, 0], np.int64))
    with factory(library, records) as cache:
        cache(xy, **args)
        assert cache.metrics()["prepared_frames"] == 2
        args.update(frames=tuple(records), jj=np.array([2, 2], np.int64))
        got = cache(xy, **args)
        assert got.tobytes() == old_and_oracle(library, records, xy, args, False)[0].tobytes()
        assert cache.metrics()["prepared_frames"] == 3
        assert cache.metrics()["calls"][-1]["prepared_frame_ids"] == [records[2]["frame_id"]]
        args.update(ii=np.array([2, 0], np.int64), kk=np.array([8, 1], np.int64), jj=np.array([0, 2], np.int64))
        got = cache(xy, **args)
        assert got.tobytes() == old_and_oracle(library, records, xy, args, False)[0].tobytes()
        assert cache.metrics()["prepared_frames"] == 3


def test_deleted_keyframe_and_current_slot_reindex_preserve_values(library):
    records = frames()
    xy, args = request(records, edges=3)
    args.update(ii=np.array([2, 0, 1], np.int64), kk=np.array([7, 2, 3], np.int64), jj=np.array([0, 1, 2], np.int64))
    with factory(library, records) as cache:
        cache(xy, **args)
        reordered = [records[2], records[0], records[1]]
        remap = np.array([1, 2, 0], np.int64)
        args.update(frames=tuple(reordered), ii=remap[args["ii"]], jj=remap[args["jj"]],
                    kk=remap[args["kk"] // 3]*3 + args["kk"] % 3)
        got = cache(xy, **args)
        assert got.tobytes() == old_and_oracle(library, reordered, xy, args, False)[0].tobytes()
        assert cache.metrics()["prepared_frames"] == 3
        remaining = [reordered[0], reordered[2]]
        xy2, args2 = request(remaining, edges=2)
        args2.update(ii=np.array([0, 1], np.int64), kk=np.array([0, 4], np.int64), jj=np.array([1, 0], np.int64))
        got = cache(xy2, **args2)
        assert got.tobytes() == old_and_oracle(library, remaining, xy2, args2, False)[0].tobytes()
        assert cache.metrics()["evicted_frames"] == 1
        assert cache.metrics()["current_cached_frames"] == 2


def test_bounded_lru_only_evicts_unreferenced_handles(library):
    records = frames()
    xy, args = request(records, edges=2)
    args.update(ii=np.array([0, 0], np.int64), kk=np.array([0, 2], np.int64), jj=np.array([1, 1], np.int64))
    with factory(library, records, max_frames=2) as cache:
        cache(xy, **args)
        args["jj"] = np.array([2, 2], np.int64)
        got = cache(xy, **args)
        assert got.tobytes() == old_and_oracle(library, records, xy, args, False)[0].tobytes()
        m = cache.metrics()
        assert m["prepared_frames"] == 3 and m["current_cached_frames"] == 2
        assert m["evictions"][-1]["reason"] == "capacity_unreferenced"
        assert m["evictions"][-1]["frame_id"] == records[1]["frame_id"]


def test_budget_rejection_is_preallocation(library):
    records = frames(2)
    xy, args = request(records, edges=2)
    args.update(ii=np.array([0, 0], np.int64), kk=np.array([0, 2], np.int64), jj=np.array([1, 1], np.int64))
    size = records[0]["gmap"].nbytes + records[0]["fmap"].nbytes + 8*2*2*4
    with factory(library, records, max_bytes=size) as cache:
        with pytest.raises(ValueError, match="capacity"):
            cache(xy, **args)
        assert cache.metrics()["prepared_frames"] == 0
        assert cache.metrics()["current_context_bytes"] == 0


@pytest.mark.parametrize("bad", ["replace", "writable", "noncontiguous", "duplicate", "ownership", "negative", "dimension"])
def test_feature_identity_and_graph_contracts_rejected(library, bad):
    records = frames(2)
    xy, args = request(records, edges=2)
    args.update(ii=np.array([0, 0], np.int64), kk=np.array([0, 2], np.int64), jj=np.array([1, 1], np.int64))
    with factory(library, records) as cache:
        cache(xy, **args)
        modified = [dict(r) for r in records]
        if bad == "replace":
            modified[0]["gmap"] = modified[0]["gmap"].copy()
            modified[0]["gmap"].flags.writeable = False
        elif bad == "writable":
            modified[0]["gmap"].flags.writeable = True
        elif bad == "noncontiguous":
            modified[0]["fmap"] = modified[0]["fmap"][..., ::-1]
        elif bad == "duplicate":
            modified[1]["frame_id"] = modified[0]["frame_id"]
        elif bad == "ownership":
            args["kk"] = np.array([3, 2], np.int64)
        elif bad == "negative":
            args["jj"] = np.array([-1, 1], np.int64)
        else:
            modified[0]["fmap"] = np.zeros((8, 8, 11), np.float32)
            modified[0]["fmap"].flags.writeable = False
        with pytest.raises(ValueError):
            cache(xy, **dict(args, frames=tuple(modified)))
        assert cache.metrics()["failed_calls"] == 1
        assert cache.metrics()["prepared_frames"] == 2


@pytest.mark.parametrize("bad", ["NaN", "Inf", "large", "float64"])
def test_invalid_coordinates_fail_without_old_result(library, bad):
    records = frames(1)
    xy, args = request(records)
    if bad == "float64":
        xy = xy.astype(np.float64)
    else:
        xy.flat[0] = {"NaN": np.nan, "Inf": np.inf, "large": 2**30}[bad]
    with factory(library, records) as cache:
        with pytest.raises((ValueError, candidate.CorrelationError)):
            cache(xy, **args)
        assert cache.metrics()["completed_calls"] == 0
        assert cache.last_call is None


def test_failed_feature_prepare_releases_and_can_reset(library):
    records = frames(1)
    records[0]["fmap"].flags.writeable = True
    records[0]["fmap"].flat[0] = np.nan
    records[0]["fmap"].flags.writeable = False
    xy, args = request(records)
    with factory(library, records) as cache:
        with pytest.raises(candidate.CorrelationError, match="nonfinite"):
            cache(xy, **args)
        assert cache.metrics()["current_context_bytes"] == 0
        cache.reset()
        clean = frames(1, seed=55)
        xy, args = request(clean)
        assert np.isfinite(cache(xy, **args)).all()


def test_reset_and_close_keep_returned_arrays_owned(library):
    records = frames(1)
    xy, args = request(records)
    cache = factory(library, records)
    first = cache(xy, **args)
    before = first.tobytes()
    cache.reset()
    assert cache.metrics()["current_context_bytes"] == 0
    new = frames(1, seed=98)
    new[0]["frame_id"] = records[0]["frame_id"]
    second = cache(xy, **dict(args, frames=tuple(new)))
    assert not np.shares_memory(first, second) and before == first.tobytes()
    cache.close()
    cache.close()
    assert cache.metrics()["destroyed_frames"] == 2 and cache.metrics()["closed"]
    assert first.tobytes() == before
    with pytest.raises(candidate.CorrelationError, match="closed"):
        cache(xy, **args)


def test_original_feature_owners_stay_alive_until_reset(library):
    records = frames(1)
    weak = weakref.ref(records[0]["fmap"])
    xy, args = request(records)
    with factory(library, records) as cache:
        cache(xy, **args)
        del records, args
        assert weak() is not None
        cache.reset()
        assert weak() is None


def test_close_waits_for_inflight_native_query(library):
    records = frames(1)
    xy, args = request(records, edges=1)
    cache = factory(library, records)
    entered, resume, finished = threading.Event(), threading.Event(), threading.Event()
    original = cache._api.dpvo_corr_v36_correlate_frames
    failures = []

    def pause(*values):
        entered.set()
        assert resume.wait(3)
        assert not cache._closed and cache._bytes > 0
        return original(*values)

    def query():
        try:
            cache(xy, **args)
        except BaseException as e:
            failures.append(e)

    def close():
        try:
            cache.close()
            finished.set()
        except BaseException as e:
            failures.append(e)

    cache._api.dpvo_corr_v36_correlate_frames = pause
    first = threading.Thread(target=query)
    second = threading.Thread(target=close)
    first.start()
    assert entered.wait(3)
    second.start()
    try:
        assert not finished.wait(.05)
    finally:
        resume.set()
        first.join(3)
        second.join(3)
        cache._api.dpvo_corr_v36_correlate_frames = original
        cache.close()
    assert not failures and finished.is_set()
    assert cache.metrics()["completed_calls"] == 1
    assert cache.metrics()["current_context_bytes"] == 0


def test_parallel_callers_are_serial_and_outputs_independent(library):
    records = frames(2)
    xy, args = request(records)
    barrier = threading.Barrier(3)
    results, failures = [], []
    with factory(library, records) as cache:
        def run():
            try:
                barrier.wait()
                results.append(cache(xy, **args))
            except BaseException as e:
                failures.append(e)
        workers = [threading.Thread(target=run) for _ in range(2)]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(3)
        assert not failures and len(results) == 2
        assert results[0].tobytes() == results[1].tobytes()
        assert not np.shares_memory(*results)
        assert cache.metrics()["prepared_frames"] <= len(records)


def test_metrics_owned_and_histories_bounded(library):
    records = frames(1)
    xy, args = request(records)
    with factory(library, records, history_limit=2) as cache:
        for _ in range(4):
            cache(xy, **args)
        metric = cache.metrics()
        assert metric["completed_calls"] == 4 and metric["call_history_dropped"] == 2
        metric["calls"][0]["native_query"]["native"]["threads"] = 99
        metric["parent_source_pins"].clear()
        assert cache.metrics()["calls"][0]["native_query"]["native"]["threads"] == 1
        assert cache.metrics()["parent_source_pins"]


def test_explicit_library_sha_rejected(library):
    with pytest.raises(candidate.CorrelationError, match="SHA256"):
        candidate.load_library(library, "0"*64)
    with pytest.raises(ValueError, match="Explicit"):
        candidate.load_library(library, None)


def test_parent_sources_and_no_fast_math_are_bound():
    assert candidate.verify_parent_sources(PARENT) == candidate.PARENT_SOURCE_PINS
    assert "-ffp-contract=off" in (HERE / "dpvo_frame_corr.cpp").read_text()
    assert "#include \"dpvo_corr.cpp\"" in (HERE / "dpvo_frame_corr.cpp").read_text()
