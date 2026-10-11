"""Tiny synthetic lifecycle tests; no checkpoint, image or neural execution."""
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

BASE = Path(__file__).resolve().parent
ENTRY = BASE / "dpvo_stream.py"
if not ENTRY.exists():
    ENTRY = BASE.parent / "deploy/rk3588/validation/dpvo_stream/dpvo_stream.py"
spec = importlib.util.spec_from_file_location("dpvo_stream_under_test", ENTRY)
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def geometry_math():
    # Only tiny synthetic SE3 values use this pinned pure NumPy implementation.
    explicit = os.environ.get("DPVO_GEOMETRY_PATH")
    path = Path(explicit) if explicit else BASE.parent / "vggt-v29-dpvo-opt/dpvo_cpu_geometry_vectorized.py"
    if not path.exists():
        path = BASE.parent / "deploy/rk3588/validation/dpvo_geometry/dpvo_cpu_geometry_vectorized.py"
    spec = importlib.util.spec_from_file_location("stream_geometry_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEO = geometry_math()


class FakeGeometry:
    se3_compose = staticmethod(GEO.se3_compose)
    se3_inverse = staticmethod(GEO.se3_inverse)
    se3_exp = staticmethod(GEO.se3_exp)
    se3_log = staticmethod(GEO.se3_log)

    def __init__(self, flow=20., fail=False, nan=False):
        self.flow, self.fail, self.nan = flow, fail, nan
        self.ba_calls, self.flow_calls = [], []

    def reproject(self, poses, patches, intrinsics, ii, jj, kk):
        assert np.array_equal(ii, kk // 2)
        return patches[:, kk, :2].copy()

    def flow_mag(self, poses, patches, intrinsics, ii, jj, kk, beta):
        self.flow_calls.append((ii.copy(), jj.copy(), kk.copy(), beta))
        # Deliberately invalid mask: official keyframe criterion ignores it.
        return np.full((1, len(ii), 3, 3), self.flow, np.float32), np.zeros((1, len(ii), 3, 3), bool)

    def BA(self, poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk,
           t0, t1, M, iterations, eff_impl):
        self.ba_calls.append(dict(t0=t0, t1=t1, M=M, iterations=iterations,
            eff_impl=eff_impl, poses_before=poses.copy(), kk=kk.copy(), net_edges=len(ii), target=target.copy()))
        assert lmbda.dtype == np.float32 and lmbda[0] == np.float32(1e-4)
        poses[0, -1, 0] += np.float32(.1)
        if self.fail:
            raise RuntimeError("synthetic failed solve after temporary mutation")
        if self.nan:
            poses[0, -1, 0] = np.nan
        return dict(iterations=iterations)


class FakeUpdate:
    def __init__(self, probe_delta=3.):
        self.probe_delta = probe_delta
        self.calls = []
        self.bad = None

    def __call__(self, net, context, corr, ii, jj, kk):
        self.calls.append(dict(net=net.copy(), ii=ii.copy(), jj=jj.copy(), kk=kk.copy(), context=context.copy()))
        E = len(ii)
        assert context.shape == (1, E, 384)
        assert corr.shape == (1, E, 882)
        output = dict(net=net + 1, delta=np.zeros((1, E, 2), np.float32),
                      weight=np.full((1, E, 2), .5, np.float32))
        output["delta"][..., 0] = self.probe_delta
        if self.bad == "net":
            output["net"][0, 0, 0] = np.nan
        if self.bad == "weight":
            output["weight"][0, 0, 0] = -.001
        return output


class FakeCorrelation:
    def __init__(self):
        self.calls = []

    def __call__(self, coords, *, ii, jj, kk, frames):
        assert np.array_equal(ii, kk // 2)
        assert not ii.flags.writeable and not jj.flags.writeable and not kk.flags.writeable
        for frame in frames:
            for value in frame.values():
                if isinstance(value, np.ndarray):
                    assert not value.flags.writeable
        self.calls.append(dict(frame_ids=[r["frame_id"] for r in frames], ii=ii.copy(), jj=jj.copy(), kk=kk.copy()))
        return np.zeros((1, len(ii), 882), np.float32)


def frame(index, timestamp=None, M=2):
    patches = np.empty((M, 3, 3, 3), np.float32)
    x, y = np.meshgrid(np.arange(3, dtype=np.float32), np.arange(3, dtype=np.float32))
    patches[:, 0], patches[:, 1], patches[:, 2] = x, y, 1
    return dict(frame_id="actual-source-" + str(index), timestamp=float(index if timestamp is None else timestamp),
        captured_monotonic=1000. + index, intrinsics=np.array([40, 40, 8, 8], np.float32),
        fmap=np.zeros((128, 4, 4), np.float32), gmap=np.zeros((M, 128, 3, 3), np.float32),
        context=np.full((M, 384), index, np.float32), patches=patches)


def make(config=None, flow=20., update=None, rng=None):
    config = config or core.StreamConfig(patches_per_frame=2)
    geo, upd, corr = FakeGeometry(flow), update or FakeUpdate(), FakeCorrelation()
    rng = rng or (lambda M: np.linspace(.2, .8, M, dtype=np.float32))
    return core.DPVOStream(config, geometry=geo, update=upd, correlate=corr, rng=rng), geo, upd, corr


def initialized(**kwargs):
    stream, geo, update, corr = make(**kwargs)
    for i in range(8):
        stream.process_encoded(frame(i))
    return stream, geo, update, corr


def test_original_default_overlay_and_capacity_difference_explicit():
    cfg = core.StreamConfig()
    assert (cfg.patches_per_frame, cfg.removal_window, cfg.optimization_window, cfg.patch_lifetime,
            cfg.keyframe_index, cfg.keyframe_threshold, cfg.motion_damping) == (96, 22, 10, 13, 4, 15., .5)
    assert cfg.capacity == 64 and not cfg.loop_closure and not cfg.classic_loop_closure


@pytest.mark.parametrize("fields", [dict(loop_closure=True), dict(classic_loop_closure=True),
    dict(capacity=8), dict(observation_capacity=0), dict(patches_per_frame=True),
    dict(motion_model="invention"), dict(keyframe_threshold=float("nan")), dict(removal_window=13)])
def test_unsupported_configuration_rejected(fields):
    with pytest.raises(ValueError):
        core.StreamConfig(**fields)


def test_concrete_forward_then_backward_edge_order_and_self_edge():
    stream, _, update, _ = make()
    for i in range(3):
        result = stream.process_encoded(frame(i))
        assert result["pose_w2c"] is None and not result["fresh_pose_valid"]
    snap = stream.snapshot()
    assert snap["kk"].tolist() == [0, 1, 0, 1, 2, 2, 3, 3, 0, 1, 2, 3, 4, 4, 4, 5, 5, 5]
    assert snap["jj"].tolist() == [0, 0, 1, 1, 0, 1, 0, 1, 2, 2, 2, 2, 0, 1, 2, 0, 1, 2]
    assert np.array_equal(snap["ii"], snap["kk"] // 2)
    assert np.all(snap["net"] == 0)
    assert len(update.calls) == 2  # Motion probes only.


def test_eighth_admitted_frame_twelve_updates_then_one_per_fresh():
    stream, geo, update, _ = make()
    for i in range(7):
        assert not stream.process_encoded(frame(i))["fresh_pose_valid"]
    result = stream.process_encoded(frame(7))
    assert result["updates"] == 12 and result["fresh_pose_valid"]
    assert result["observation_id"] == 7 and result["frame_id"] == "actual-source-7"
    assert np.array_equal(result["pose_w2c"], stream.snapshot()["poses"][0, -1])
    assert len(geo.ba_calls) == 12 and all(x["t0"] == 1 and x["t1"] == 8 for x in geo.ba_calls)
    assert all(x["iterations"] == 2 and not x["eff_impl"] for x in geo.ba_calls)
    assert len(stream.snapshot()["ii"]) == 8 * 8 * 2
    assert np.all(stream.snapshot()["net"] == 12)
    stream.process_encoded(frame(8))
    assert len(geo.ba_calls) == 13
    assert stream.metrics()["fresh_poses"] == 2
    assert stream.metrics()["BA_iterations"] == 26
    assert np.all(update.calls[-1]["net"][0, :128] == 12)  # carried recurrent state
    assert np.all(update.calls[-1]["net"][0, 128:] == 0)   # new factor rows


def test_default_M96_initial_edge_count_without_neural_execution():
    # Pure graph construction demonstrates the actual E6144, not fixed E864.
    stream, _, _, _ = make(config=core.StreamConfig())
    s = stream._state.fork()
    for i in range(8):
        record = stream._record(frame(i, M=96))
        stream._append(s, record)
        s.observations.append(dict(timestamp=float(i)))
        stream._append_edges(s)
    assert len(s.ii) == 6144


def test_motion_quantile_distinct_from_depth_lower_median_and_random_draws():
    assert core.lower_median(np.array([.1, .8], np.float32)) == np.float32(.1)
    values = []
    def rng(M):
        values.append(M)
        return np.array([.1, .8], np.float32)
    update = FakeUpdate(probe_delta=0.)
    original = update.__call__
    def mixed(net, ctx, corr, ii, jj, kk):
        out = original(net, ctx, corr, ii, jj, kk)
        out["delta"][0, :, 0] = [1., 3.]
        return out
    stream, _, _, _ = make(update=mixed, rng=rng)
    stream.process_encoded(frame(0))
    result = stream.process_encoded(frame(1))
    assert result["motion_probe"] == 2. and result["admitted"]  # equal threshold admits
    for i in range(2, 9):
        # use ordinary Update after probe semantics have been demonstrated
        stream.update = FakeUpdate()
        stream.process_encoded(frame(i))
    assert len(values) == 9
    snap = stream.snapshot()
    assert np.all(snap["patches"][0, -2:, 2] == np.float32(.1))
    assert not np.all(snap["patches"][0, :2, 2] == 1)


def test_motion_reject_consumes_observation_and_identity_relation_not_fresh():
    update = FakeUpdate(probe_delta=1.99)
    stream, _, _, _ = make(update=update)
    stream.process_encoded(frame(0))
    result = stream.process_encoded(frame(1))
    assert result["admitted"] is False and result["pose_w2c"] is None
    assert (stream.n, stream.m, stream.counter) == (1, 2, 2)
    parent, relation = stream.snapshot()["relative"][1]
    assert parent == 0 and np.array_equal(relation, core._identity())
    assert stream.metrics()["motion_skipped"] == 1 and stream.metrics()["fresh_poses"] == 0
    update.probe_delta = 3.
    assert stream.process_encoded(frame(2))["admitted"]
    assert stream.snapshot()["retained_observation_ids"] == [0, 2]


def test_damped_dt_uses_all_source_observations_including_rejection():
    update = FakeUpdate()
    stream, _, _, _ = make(update=update)
    stream.process_encoded(frame(0, timestamp=0.))
    stream.process_encoded(frame(1, timestamp=1.))
    stream._state.poses[0, -1, 0] = 1.  # tiny injected synthetic pose state
    update.probe_delta = 0.
    stream.process_encoded(frame(2, timestamp=3.))
    update.probe_delta = 3.
    stream.process_encoded(frame(3, timestamp=4.))
    # Last input spacings 2 then 1; fac=.5, damping=.5 -> translation +.25.
    assert stream.snapshot()["poses"][0, -1, 0] == pytest.approx(1.25)


def test_keyframe_removal_relative_left_composition_and_edge_remap():
    stream, geo, _, corr = initialized()
    before = stream.snapshot()
    geo.flow = 0.
    result = stream.process_encoded(frame(8))
    assert result["keyframe"]["removed_observation_id"] == 5
    snap = stream.snapshot()
    assert snap["retained_observation_ids"] == [0, 1, 2, 3, 4, 6, 7, 8]
    assert np.array_equal(snap["ii"], snap["kk"] // 2)
    assert np.all(snap["jj"] < 8)
    parent, relative = snap["relative"][5]
    assert parent == 4
    expected = GEO.se3_compose(before["poses"][0, 5], GEO.se3_inverse(before["poses"][0, 4]))
    assert np.array_equal(relative, expected)
    assert not any(5 == x for x in snap["retained_observation_ids"])
    # Subsequent correlation sees current source identities after compact remap.
    geo.flow = 20.
    stream.process_encoded(frame(9))
    assert corr.calls[-1]["frame_ids"] == ["actual-source-" + str(i) for i in [0, 1, 2, 3, 4, 6, 7, 8, 9]]
    assert all(call[3] == .5 for call in geo.flow_calls)


def test_flow_criterion_ignores_valid_mask_and_uses_all_patch_pixels():
    stream, geo, _, _ = initialized(flow=20.)
    result = stream.process_encoded(frame(8))
    assert result["keyframe"]["mean_bidirectional_flow"] == 20.
    assert result["keyframe"]["removed_observation_id"] is None
    assert len(geo.flow_calls) == 2


def test_source_age_removal_stores_targets_weights_and_not_target_age():
    stream, _, _, _ = initialized()
    for i in range(8, 25):
        stream.process_encoded(frame(i))
    snap = stream.snapshot()
    assert snap["ii"].min() == 3
    assert np.any(snap["ii_inactive"] == 0) and np.any(snap["jj_inactive"] >= 2)
    assert len(snap["ii_inactive"]) == snap["target_inactive"].shape[1] == snap["weight_inactive"].shape[1]
    assert np.all(snap["weight_inactive"] == .5)
    assert np.array_equal(snap["ii_inactive"], snap["kk_inactive"] // 2)


@pytest.mark.parametrize("kind", ["weight", "net", "BA_throw", "BA_nan"])
def test_finite_domain_or_solve_failure_terminal_without_partial_frame(kind):
    stream, geo, update, _ = initialized()
    before = stream.snapshot()
    if kind.startswith("BA"):
        geo.fail, geo.nan = kind == "BA_throw", kind == "BA_nan"
    else:
        update.bad = kind
    with pytest.raises(core.StreamFailure):
        stream.process_encoded(frame(8))
    after = stream.snapshot()
    for key in ("poses", "patches", "net", "ii", "jj", "kk", "target", "weight"):
        assert before[key].tobytes() == after[key].tobytes()
    assert (stream.n, stream.counter) == (8, 8)
    assert after["status"] == "failed" and stream.metrics()["fresh_poses"] == 1
    with pytest.raises(core.StreamFailure, match="failed"):
        stream.process_encoded(frame(9))


@pytest.mark.parametrize("change", ["duplicate", "timestamp", "capture", "nonfinite", "dtype", "K"])
def test_source_identity_and_pixel_intrinsic_guard(change):
    stream, _, _, _ = make()
    stream.process_encoded(frame(0))
    record = frame(1)
    if change == "duplicate": record["frame_id"] = "actual-source-0"
    if change == "timestamp": record["timestamp"] = 0.
    if change == "capture": record["captured_monotonic"] = 1000.
    if change == "nonfinite": record["fmap"][0, 0, 0] = np.nan
    if change == "dtype": record["context"] = record["context"].astype(np.float64)
    if change == "K": record["intrinsics"][0] += 1
    with pytest.raises(core.StreamFailure):
        stream.process_encoded(record)
    assert (stream.n, stream.counter) == (1, 1)


def test_bounded_capacity_failclosed_without_eviction():
    stream, _, _, _ = initialized(config=core.StreamConfig(patches_per_frame=2, capacity=9))
    with pytest.raises(core.StreamFailure, match="Frame capacity"):
        stream.process_encoded(frame(8))
    assert (stream.n, stream.counter) == (8, 8)


def test_observation_capacity_counts_motion_rejected_inputs():
    stream, _, _, _ = make(config=core.StreamConfig(patches_per_frame=2, observation_capacity=2), update=FakeUpdate(0.))
    stream.process_encoded(frame(0)); stream.process_encoded(frame(1))
    with pytest.raises(core.StreamFailure, match="Observation history"):
        stream.process_encoded(frame(2))
    assert stream.counter == 2


def test_active_edge_budget_rejects_before_new_graph_commit():
    stream, _, _, _ = make(config=core.StreamConfig(patches_per_frame=2, max_active_edges=3))
    stream.process_encoded(frame(0))
    with pytest.raises(core.StreamFailure, match="Active edge capacity"):
        stream.process_encoded(frame(1))
    assert stream.n == 1 and stream.counter == 1


def test_explicit_unreachable_loop_off_long_edge_guard():
    stream, _, _, _ = initialized()
    s = stream._state.fork()
    # Deliberately contradictory tiny metadata to exercise the long-edge gate.
    stream.config = replace(stream.config, removal_window=17)
    s.frames = s.frames * 4
    with pytest.raises(ValueError, match="long-range/global-BA"):
        stream._step(s)


def test_snapshot_is_owned_and_default_excludes_feature_maps():
    stream, _, _, corr = make()
    original = frame(0)
    stream.process_encoded(original)
    original["context"][:] = 999
    snap = stream.snapshot()
    assert "frames" not in snap and "fmap" not in snap and "gmap" not in snap
    snap["poses"][:] = np.nan
    assert np.isfinite(stream.snapshot()["poses"]).all()
    stream.process_encoded(frame(1))
    assert len(corr.calls) == 1


def test_termination_has_twelve_offline_updates_and_no_fresh_increment():
    stream, geo, _, _ = initialized()
    old = stream.metrics()["fresh_poses"]
    result = stream.terminate()
    assert result["offline_updates"] == 12 and not result["fresh_pose_valid"] and not result["online_publication"]
    assert stream.metrics()["fresh_poses"] == old and stream.metrics()["offline_updates"] == 12
    assert len(geo.ba_calls) == 24
    assert result["pose_w2c"].shape == (8, 7)
    with pytest.raises(core.StreamFailure, match="terminated"):
        stream.process_encoded(frame(8))


def test_skip_chain_reconstruction_and_deleted_relative_pose_are_offline_only():
    update = FakeUpdate(0.)
    stream, _, _, _ = make(update=update)
    stream.process_encoded(frame(0))
    stream.process_encoded(frame(1)); stream.process_encoded(frame(2))
    result = stream.terminate()
    assert result["initialized"] is False and result["trajectory_state"] == "initialization_incomplete"
    assert result["reconstructed_observation_ids"] == [1, 2]
    assert np.array_equal(result["pose_w2c"][0], result["pose_w2c"][1])
    assert np.array_equal(result["pose_w2c"][1], result["pose_w2c"][2])
    assert stream.metrics()["fresh_poses"] == 0


def test_noncommuting_removed_pose_relation_has_correct_left_order():
    stream, _, _, _ = make()
    stream.process_encoded(frame(0))
    s = stream._state.fork()
    parent = GEO.se3_exp(np.array([1., 0, 0, 0, 0, .4], np.float32))
    relation = GEO.se3_exp(np.array([0, 1., 0, .2, 0, 0], np.float32))
    s.poses[0, 0] = parent
    s.observations.append(dict(frame_id="removed", observation_id=1, timestamp=1., captured_monotonic=1001.))
    s.relative[1] = (0, relation)
    result = stream._trajectory(s)
    assert np.allclose(result["pose_w2c"][1], GEO.se3_compose(relation, parent))
    assert not np.allclose(result["pose_w2c"][1], GEO.se3_compose(parent, relation))


def test_empty_termination_no_fake_observation_or_geometry_call():
    stream, geo, _, _ = make()
    result = stream.terminate()
    assert result["pose_w2c"].shape == (0, 7) and result["offline_updates"] == 0
    assert result["initialized"] is False and result["trajectory_state"] == "initialization_incomplete"
    assert geo.ba_calls == []


def test_injected_encoder_only_entry_calls_once_per_distinct_observation():
    stream, _, _, _ = make()
    calls = []
    def encoder(raw, *, patches_per_frame):
        calls.append((raw, patches_per_frame))
        return frame(raw)
    stream.encoder = encoder
    result = stream.process_frame(0, frame_id="source", timestamp=0., captured_monotonic=1000.,
                                  intrinsics=np.array([40, 40, 8, 8], np.float32))
    assert calls == [(0, 2)] and result["frame_id"] == "source" and not result["fresh_pose_valid"]
