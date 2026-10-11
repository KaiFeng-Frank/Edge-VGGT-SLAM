"""CPU array lifecycle for pinned original DPVO, with injected neural/geometry work.

Derived from princeton-vl/DPVO commit
0ac95b656d1fda91c271d2a106460d19ad966fc7, dpvo.py and patchgraph.py.
Copyright (c) 2022 Princeton Vision & Learning Lab. MIT; see LICENSE.

No model, CUDA, ORB, optical flow, PnP, image decoder or checkpoint is imported.
This is an inference lifecycle reference, not a performance or accuracy result.
World-to-camera poses are [tx,ty,tz,qx,qy,qz,qw], with monocular arbitrary scale.
Loop closure is explicitly unsupported, as in the pinned default configuration.
The owner calls process_encoded once per real, distinct source observation.
EOF optimization and skipped/deleted-frame reconstruction never count as fresh.

Callbacks:
  update(net, context, correlation, ii, jj, kk) -> {net, delta, weight}
  correlate(coords, *, ii, jj, kk, frames) -> float32 [1,E,882]
  encoder(frame, *, patches_per_frame=M) -> encoded array fields (optional)
The correlate frames tuple follows CURRENT retained-frame indices, followed by
the tentative candidate during a motion probe. Each read-only record contains
its stable source frame_id/observation_id; a cache must not retain old slot IDs.
Geometry supplies the frozen v29 SE3/reproject/flow_mag/BA function signatures.
Callbacks must return CPU NumPy arrays. Execution and source provenance belong
to the calling board driver; this module does not silently choose a backend.
"""
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
import hashlib
import math
import time

import numpy as np

COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
SOURCE_PINS = {
    "dpvo/dpvo.py": "7efcb45689c2da8d18e2c69774e221799c9f8472e99531955f55dcb145626fd7",
    "dpvo/patchgraph.py": "a2cef93f93c7aab83e8310ddf0207e248c62bc2eaae2a1158a305f3efccaef53",
    "dpvo/projective_ops.py": "ef20f322a118b0790bff975cce3120ad31e867c8b088a9f408c3328c0fb664c1",
    "dpvo/config.py": "f70eaf98ef32a7ea5a675de571373e32d0a6d1354bea9d13c6a779eb68f50bec",
    "config/default.yaml": "0351abfbc60d9ec93bf402857130497b2f063f8ce0e8fd6e87cc55c73273594a",
    "dpvo/utils.py": "5bc5820014045b6979d9979ae80a2b519c64f9491ae83e909a77731152751a4a",
}
GEOMETRY_FUNCTIONS = ("se3_compose", "se3_inverse", "se3_exp", "se3_log",
                      "reproject", "flow_mag", "BA")
DIM, CHANNELS, RES, P = 384, 128, 4, 3


def verify_sources(directory):
    """Optional driver-side strict check; never imports official CUDA modules."""
    result = {}
    for relative, expected in SOURCE_PINS.items():
        actual = hashlib.sha256((Path(directory) / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Pinned DPVO source mismatch: " + relative)
        result[relative] = actual
    return result


@dataclass(frozen=True)
class StreamConfig:
    patches_per_frame: int = 96
    removal_window: int = 22
    optimization_window: int = 10
    patch_lifetime: int = 13
    keyframe_index: int = 4
    keyframe_threshold: float = 15.0
    motion_model: str = "DAMPED_LINEAR"
    motion_damping: float = .5
    capacity: int = 64
    observation_capacity: int = 4096
    max_active_edges: int = 98304
    max_inactive_edges: int = 393216
    loop_closure: bool = False
    classic_loop_closure: bool = False

    def __post_init__(self):
        for name in ("patches_per_frame", "removal_window", "optimization_window",
                     "patch_lifetime", "keyframe_index", "capacity",
                     "observation_capacity", "max_active_edges", "max_inactive_edges"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if self.capacity < 9 or self.keyframe_index < 2 or self.keyframe_index > 6:
            raise ValueError("capacity >=9 and keyframe_index in [2,6] required")
        if self.removal_window < self.patch_lifetime + self.keyframe_index:
            raise ValueError("removal_window must keep inactive edges before removed keyframes")
        if self.motion_model not in ("DAMPED_LINEAR", "CONSTANT"):
            raise ValueError("Unsupported motion_model")
        for name in ("motion_damping", "keyframe_threshold"):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if type(self.loop_closure) is not bool or type(self.classic_loop_closure) is not bool:
            raise ValueError("Loop flags must be bool")
        if self.loop_closure or self.classic_loop_closure:
            raise ValueError("Loop closure is not implemented; pinned original defaults are disabled")


class StreamFailure(RuntimeError):
    """A failed observation/EOF phase is terminal; no stale pose is substituted."""


class CapacityError(StreamFailure):
    pass


def _array(value, name, shape=None, *, positive=False):
    x = np.asarray(value)
    if x.dtype != np.float32 or (shape is not None and x.shape != tuple(shape)) or not np.isfinite(x).all():
        raise ValueError(name + " requires finite float32 with the declared shape")
    if positive and np.any(x <= 0):
        raise ValueError(name + " must be positive")
    return x


def _readonly(value):
    x = value.view()
    x.flags.writeable = False
    return x


def _pose(value, shape):
    x = _array(value, "pose", shape)
    if np.any(np.abs(np.linalg.norm(x[..., 3:], axis=-1) - 1) > 1e-3):
        raise ValueError("Invalid non-unit pose quaternion")
    return x


def _identity():
    return np.array([0, 0, 0, 0, 0, 0, 1], np.float32)


def lower_median(value):
    """torch.median(flattened tensor): lower middle, unlike np.median."""
    x = _array(value, "median input").reshape(-1)
    if not x.size:
        raise ValueError("Empty depth median")
    k = (x.size - 1) // 2
    return np.partition(x, k)[k]


@dataclass
class _State:
    frames: list
    observations: list
    relative: dict
    poses: np.ndarray
    patches: np.ndarray
    intrinsics: np.ndarray
    ii: np.ndarray
    jj: np.ndarray
    kk: np.ndarray
    net: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    ii_inactive: np.ndarray
    jj_inactive: np.ndarray
    kk_inactive: np.ndarray
    target_inactive: np.ndarray
    weight_inactive: np.ndarray
    initialized: bool = False
    updates: int = 0

    def fork(self):
        # Numerical operations are copy-on-write; no full feature-map copy.
        return replace(self, frames=list(self.frames), observations=list(self.observations),
                       relative=dict(self.relative))


class DPVOStream:
    def __init__(self, config=None, *, geometry, update, correlate, encoder=None, rng=None):
        self.config = config if config is not None else StreamConfig()
        if not isinstance(self.config, StreamConfig):
            raise TypeError("config must be StreamConfig")
        for name in GEOMETRY_FUNCTIONS:
            if not callable(getattr(geometry, name, None)):
                raise TypeError("Missing injected geometry function: " + name)
        if not callable(update) or not callable(correlate) or (encoder is not None and not callable(encoder)):
            raise TypeError("Neural/correlation/encoder interfaces must be callable")
        self.geometry, self.update, self.correlate, self.encoder = geometry, update, correlate, encoder
        self.rng = rng if rng is not None else np.random.default_rng()
        if not callable(self.rng) and not callable(getattr(self.rng, "random", None)):
            raise TypeError("rng must be random_depth(M) callable or .random(size,dtype)")
        ids = np.empty(0, np.int64)
        self._state = _State([], [], {}, np.empty((1, 0, 7), np.float32),
            np.empty((1, 0, 3, P, P), np.float32), np.empty((1, 0, 4), np.float32),
            ids, ids.copy(), ids.copy(), np.empty((1, 0, DIM), np.float32),
            np.empty((1, 0, 2), np.float32), np.empty((1, 0, 2), np.float32),
            ids.copy(), ids.copy(), ids.copy(), np.empty((1, 0, 2), np.float32),
            np.empty((1, 0, 2), np.float32))
        self._status = "running"
        self._failure = None
        self._counts = dict(observations=0, admitted=0, motion_skipped=0, fresh_poses=0,
            update_calls=0, motion_probe_calls=0, BA_calls=0, BA_iterations=0,
            deleted_keyframes=0, stored_inactive_edges=0, offline_updates=0)
        self._terminated_result = None

    @property
    def n(self):
        return len(self._state.frames)

    @property
    def m(self):
        return self.n * self.config.patches_per_frame

    @property
    def counter(self):
        return len(self._state.observations)

    def _running(self):
        if self._status != "running":
            raise StreamFailure("Stream is " + self._status)

    def _fail(self, error, phase, frame_id=None):
        self._status = "failed"
        self._failure = dict(phase=phase, frame_id=frame_id,
                             type=type(error).__name__, reason=str(error))
        raise StreamFailure(phase + " failed: " + str(error)) from error

    def _record(self, record):
        if not isinstance(record, dict):
            raise ValueError("Encoded frame must be a dict")
        frame_id = record.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, (str, int, float)):
            raise ValueError("frame_id must be a scalar source identity")
        if (isinstance(frame_id, float) and not math.isfinite(frame_id)) or frame_id == "":
            raise ValueError("Invalid source frame_id")
        if any(o["frame_id"] == frame_id for o in self._state.observations):
            raise ValueError("Duplicate source observation")
        stamp, capture = record.get("timestamp"), record.get("captured_monotonic")
        for name, value in (("timestamp", stamp), ("captured_monotonic", capture)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name + " must be finite")
            if self.counter and value <= self._state.observations[-1][name]:
                raise ValueError(name + " must strictly increase; no repeated capture")
        M = self.config.patches_per_frame
        fmap = _array(record.get("fmap"), "fmap")
        if fmap.ndim != 3 or fmap.shape[0] != CHANNELS or min(fmap.shape[1:]) < 4:
            raise ValueError("fmap must be [128,h,w], h/w >=4")
        K = _array(record.get("intrinsics"), "intrinsics", (4,))
        if np.any(K[:2] <= 0):
            raise ValueError("Positive focal lengths required")
        values = dict(frame_id=frame_id, timestamp=float(stamp), captured_monotonic=float(capture),
                      observation_id=self.counter)
        for name, shape in (("fmap", fmap.shape), ("gmap", (M, CHANNELS, P, P)),
                            ("context", (M, DIM)), ("patches", (M, 3, P, P)), ("intrinsics", (4,))):
            x = _array(record.get(name), name, shape).copy(order="C")
            x.flags.writeable = False
            values[name] = x
        if self.n and values["fmap"].shape != self._state.frames[0]["fmap"].shape:
            raise ValueError("Feature-map geometry changed")
        # Original fastBA uses the common intrinsics[0] for all edges.
        if self.n and not np.array_equal(K / RES, self._state.intrinsics[0, 0]):
            raise ValueError("Varying intrinsics are unsupported by original common-K fastBA")
        return MappingProxyType(values)

    def _random_depth(self):
        M = self.config.patches_per_frame
        values = self.rng(M) if callable(self.rng) else self.rng.random(M, dtype=np.float32)
        values = _array(values, "random initial inverse depth", (M,))
        if np.any(values < 0) or np.any(values >= 1):
            raise ValueError("Random initial inverse depth must be in [0,1), like torch.rand")
        return values

    def _append(self, s, record):
        n, M = len(s.frames), self.config.patches_per_frame
        candidate_pose = _identity()
        if n > 1:
            last, previous = s.poses[0, -1], s.poses[0, -2]
            if self.config.motion_model == "DAMPED_LINEAR":
                # Upstream dt is last three source observations, including skips.
                a, b, c = [o["timestamp"] for o in (s.observations + [record])[-3:]]
                ratio = (c - b) / (b - a)
                relative = self.geometry.se3_compose(last, self.geometry.se3_inverse(previous))
                xi = self.config.motion_damping * ratio * self.geometry.se3_log(relative)
                candidate_pose = self.geometry.se3_compose(self.geometry.se3_exp(xi), last)
            else:
                candidate_pose = last.copy()
        _pose(candidate_pose, (7,))
        patches = record["patches"].copy()
        depth = self._random_depth()  # Upstream draws before overwriting initialized depth.
        patches[:, 2] = depth[:, None, None]
        if s.initialized:
            patches[:, 2] = lower_median(s.patches[0, (n - 3) * M:n * M, 2])
        s.poses = np.concatenate((s.poses, candidate_pose[None, None]), 1)
        s.patches = np.concatenate((s.patches, patches[None]), 1)
        s.intrinsics = np.concatenate((s.intrinsics, (record["intrinsics"] / RES)[None, None]), 1)
        s.frames.append(record)

    def _neural(self, s, ii, jj, kk, net, *, probe):
        E, M = len(ii), self.config.patches_per_frame
        coords = _array(self.geometry.reproject(_readonly(s.poses), _readonly(s.patches),
            _readonly(s.intrinsics), _readonly(ii), _readonly(jj), _readonly(kk)), "reprojection", (1, E, 2, P, P))
        corr = self.correlate(coords.copy(), ii=_readonly(ii), jj=_readonly(jj),
                              kk=_readonly(kk), frames=tuple(s.frames))
        corr = _array(corr, "correlation", (1, E, 2 * 49 * P * P))
        context = np.stack([s.frames[int(k // M)]["context"][int(k % M)] for k in kk])[None]
        self._counts["motion_probe_calls" if probe else "update_calls"] += 1
        output = self.update(net.copy(), context, corr.copy(), ii.copy(), jj.copy(), kk.copy())
        if not isinstance(output, dict) or not {"net", "delta", "weight"} <= set(output):
            raise ValueError("Update must return net/delta/weight mapping")
        new_net = _array(output["net"], "Update net", (1, E, DIM)).copy()
        delta = _array(output["delta"], "Update delta", (1, E, 2)).copy()
        weight = _array(output["weight"], "Update weight", (1, E, 2)).copy()
        if np.any(weight < 0) or np.any(weight > 1):
            raise ValueError("Update weight outside original sigmoid [0,1]")
        return coords, new_net, delta, weight

    def _probe(self, s):
        n, M = len(s.frames) - 1, self.config.patches_per_frame
        kk = np.arange((n - 1) * M, n * M, dtype=np.int64)
        ii, jj = kk // M, np.full(M, n, np.int64)
        _, _, delta, _ = self._neural(s, ii, jj, kk, np.zeros((1, M, DIM), np.float32), probe=True)
        return float(np.quantile(np.linalg.norm(delta[0], axis=-1).astype(np.float32), .5, method="linear"))

    def _append_edges(self, s):
        n, M, r = len(s.frames), self.config.patches_per_frame, self.config.patch_lifetime
        # flatmeshgrid(indexing='ij'): patch outer, target inner; forward first.
        old = np.arange(M * max(n - r, 0), M * (n - 1), dtype=np.int64)
        new = np.arange(M * (n - 1), M * n, dtype=np.int64)
        targets = np.arange(max(n - r, 0), n, dtype=np.int64)
        kk = np.concatenate((old, np.repeat(new, len(targets))))
        jj = np.concatenate((np.full(len(old), n - 1, np.int64), np.tile(targets, len(new))))
        if len(s.ii) + len(kk) > self.config.max_active_edges:
            raise CapacityError("Active edge capacity exhausted; no invented eviction")
        s.ii = np.concatenate((s.ii, kk // M))
        s.jj, s.kk = np.concatenate((s.jj, jj)), np.concatenate((s.kk, kk))
        s.net = np.concatenate((s.net, np.zeros((1, len(kk), DIM), np.float32)), 1)
        # These slots are filled by Update before any removal/BA consumes them.
        s.target = np.concatenate((s.target, np.zeros((1, len(kk), 2), np.float32)), 1)
        s.weight = np.concatenate((s.weight, np.zeros((1, len(kk), 2), np.float32)), 1)

    def _check_graph(self, s):
        n, M, E = len(s.frames), self.config.patches_per_frame, len(s.ii)
        _pose(s.poses, (1, n, 7))
        _array(s.patches, "graph patches", (1, n * M, 3, P, P))
        _array(s.intrinsics, "graph intrinsics", (1, n, 4))
        for ids, limit in ((s.ii, n), (s.jj, n), (s.kk, n * M)):
            if ids.dtype != np.int64 or ids.shape != (E,) or np.any(ids < 0) or np.any(ids >= limit):
                raise ValueError("Graph IDs outside current retained-frame range")
        if not np.array_equal(s.ii, s.kk // M):
            raise ValueError("Graph patch/source ownership mismatch")
        _array(s.net, "graph net", (1, E, DIM))
        _array(s.target, "graph target", (1, E, 2))
        _array(s.weight, "graph weight", (1, E, 2))

    def _step(self, s):
        n = len(s.frames)
        if not len(s.ii):
            raise ValueError("No factors for Update")
        if np.any(s.ii < n - self.config.removal_window - 1):
            raise ValueError("Unsupported long-range/global-BA edge; loop closure is disabled")
        coords, net, delta, weight = self._neural(s, s.ii, s.jj, s.kk, s.net, probe=False)
        target = (coords[..., P // 2, P // 2] + delta).astype(np.float32)
        poses, patches = s.poses.copy(), s.patches.copy()
        t0 = max(n - self.config.optimization_window if s.initialized else 1, 1)
        self._counts["BA_calls"] += 1
        result = self.geometry.BA(poses, patches, _readonly(s.intrinsics), _readonly(target),
            _readonly(weight), np.array([1e-4], np.float32), _readonly(s.ii), _readonly(s.jj),
            _readonly(s.kk), t0, n, M=self.config.patches_per_frame, iterations=2, eff_impl=False)
        _pose(poses, s.poses.shape)
        _array(patches, "BA patches", s.patches.shape, positive=False)
        if np.any(patches[0, :, 2] <= 0):
            raise ValueError("BA produced nonpositive inverse depth")
        # Commit after all numerical guards; exceptions never substitute old poses.
        s.poses, s.patches, s.net, s.target, s.weight = poses, patches, net, target, weight
        s.updates += 1
        self._counts["BA_iterations"] += 2
        return dict(optimized_pose_interval=[t0, n], edges=len(s.ii), iterations=2,
                    diagnostic=result)

    def _remove_edges(self, s, mask, *, store):
        if store and np.any(mask):
            count = int(mask.sum())
            if len(s.ii_inactive) + count > self.config.max_inactive_edges:
                raise CapacityError("Inactive edge capacity exhausted; no invented eviction")
            s.ii_inactive = np.concatenate((s.ii_inactive, s.ii[mask]))
            s.jj_inactive = np.concatenate((s.jj_inactive, s.jj[mask]))
            s.kk_inactive = np.concatenate((s.kk_inactive, s.kk[mask]))
            s.target_inactive = np.concatenate((s.target_inactive, s.target[:, mask]), 1)
            s.weight_inactive = np.concatenate((s.weight_inactive, s.weight[:, mask]), 1)
            self._counts["stored_inactive_edges"] += count
        s.ii, s.jj, s.kk = s.ii[~mask], s.jj[~mask], s.kk[~mask]
        s.net, s.target, s.weight = s.net[:, ~mask], s.target[:, ~mask], s.weight[:, ~mask]

    def _keyframe(self, s):
        n, M, r = len(s.frames), self.config.patches_per_frame, self.config.keyframe_index
        i, j = n - r - 1, n - r + 1
        values = []
        for source, target in ((i, j), (j, i)):
            mask = (s.ii == source) & (s.jj == target)
            if not np.any(mask):
                raise ValueError("Missing original keyframe-flow factors")
            flow, _ = self.geometry.flow_mag(_readonly(s.poses), _readonly(s.patches),
                _readonly(s.intrinsics), s.ii[mask], s.jj[mask], s.kk[mask], beta=.5)
            _array(flow, "keyframe flow")
            values.append(float(np.mean(flow)))  # Upstream ignores validity mask here.
        motion = (values[0] + values[1]) / 2
        removed = None
        if motion < self.config.keyframe_threshold:
            k = n - r
            previous, removed = s.frames[k - 1]["observation_id"], s.frames[k]["observation_id"]
            relative = self.geometry.se3_compose(s.poses[0, k], self.geometry.se3_inverse(s.poses[0, k - 1]))
            _pose(relative, (7,))
            s.relative[removed] = (previous, relative.copy())
            self._remove_edges(s, (s.ii == k) | (s.jj == k), store=False)
            sources_after = s.ii > k
            s.kk = s.kk - sources_after.astype(np.int64) * M
            s.ii = s.ii - sources_after.astype(np.int64)
            s.jj = s.jj - (s.jj > k).astype(np.int64)
            # Inactive sources/targets precede k by validated window constraints.
            if np.any(s.ii_inactive >= k) or np.any(s.jj_inactive >= k):
                raise ValueError("Inactive edges unexpectedly touch shifted keyframes")
            s.frames.pop(k)
            s.poses = np.delete(s.poses, k, axis=1)
            keep = np.ones(n * M, bool)
            keep[k * M:(k + 1) * M] = False
            s.patches = s.patches[:, keep]
            s.intrinsics = np.delete(s.intrinsics, k, axis=1)
            self._counts["deleted_keyframes"] += 1
        # Source-frame age, not jj age, exactly as original ix[kk] predicate.
        self._remove_edges(s, (s.kk // M) < len(s.frames) - self.config.removal_window, store=True)
        return dict(mean_bidirectional_flow=motion, removed_observation_id=removed)

    def process_frame(self, frame, *, frame_id, timestamp, captured_monotonic, intrinsics):
        self._running()
        if self.encoder is None:
            raise TypeError("No encoder injected; use process_encoded")
        try:
            encoded = self.encoder(frame, patches_per_frame=self.config.patches_per_frame)
            if not isinstance(encoded, dict):
                raise ValueError("Encoder must return encoded field mapping")
            return self.process_encoded(dict(encoded, frame_id=frame_id, timestamp=timestamp,
                captured_monotonic=captured_monotonic, intrinsics=intrinsics))
        except StreamFailure:
            raise
        except Exception as error:
            self._fail(error, "encoder", frame_id)

    def process_encoded(self, record):
        self._running()
        began = time.monotonic()
        try:
            if self.n + 1 >= self.config.capacity:
                raise CapacityError("Frame capacity exhausted (original spare-slot guard)")
            if self.counter >= self.config.observation_capacity:
                raise CapacityError("Observation history capacity exhausted")
            item = self._record(record)
            s = self._state.fork()
            self._append(s, item)
            probe = self._probe(s) if self.n > 0 and not s.initialized else None
            observation = {key: item[key] for key in ("frame_id", "timestamp", "captured_monotonic", "observation_id")}
            s.observations.append(observation)
            if probe is not None and probe < 2.0:
                s.frames.pop()
                s.poses, s.patches = s.poses[:, :-1], s.patches[:, :-self.config.patches_per_frame]
                s.intrinsics = s.intrinsics[:, :-1]
                s.relative[item["observation_id"]] = (item["observation_id"] - 1, _identity())
                self._state = s
                self._counts["observations"] += 1
                self._counts["motion_skipped"] += 1
                return dict(observation, admitted=False, initialized=False, fresh_pose_valid=False,
                    pose_w2c=None, reason="initial_motion_below_2", motion_probe=probe,
                    updates=0, process_seconds=time.monotonic() - began)
            self._append_edges(s)
            steps, keyframe = [], None
            if len(s.frames) == 8 and not s.initialized:
                s.initialized = True  # Original is_initialized before the twelve calls.
                steps = [self._step(s) for _ in range(12)]
            elif s.initialized:
                steps = [self._step(s)]
                keyframe = self._keyframe(s)
            self._check_graph(s)
            if s.frames[-1]["observation_id"] != item["observation_id"]:
                raise ValueError("Current pose no longer belongs to the admitted source observation")
            self._state = s
            self._counts["observations"] += 1
            self._counts["admitted"] += 1
            fresh = bool(s.initialized and steps)
            self._counts["fresh_poses"] += int(fresh)
            return dict(observation, admitted=True, initialized=s.initialized, fresh_pose_valid=fresh,
                pose_w2c=s.poses[0, -1].copy() if fresh else None,
                reason="updated_current_observation" if fresh else "initialization_wait",
                motion_probe=probe, updates=len(steps), update_diagnostics=steps, keyframe=keyframe,
                retained_frames=len(s.frames), active_edges=len(s.ii),
                process_seconds=time.monotonic() - began)
        except Exception as error:
            self._fail(error, "fresh_observation", record.get("frame_id") if isinstance(record, dict) else None)

    def _trajectory(self, s):
        retained = {r["observation_id"]: s.poses[0, i] for i, r in enumerate(s.frames)}
        resolved = {}
        for item in s.observations:
            key, chain, seen = item["observation_id"], [], set()
            current = key
            while current not in retained and current not in resolved:
                if current in seen or current not in s.relative:
                    raise ValueError("Invalid removed-frame relative-pose chain")
                seen.add(current)
                parent, relation = s.relative[current]
                chain.append((current, relation))
                current = parent
            value = (retained[current] if current in retained else resolved[current]).copy()
            for child, relative in reversed(chain):
                value = self.geometry.se3_compose(relative, value)
                resolved[child] = _pose(value, (7,)).copy()
            if key in retained:
                resolved[key] = retained[key].copy()
        poses = np.stack([resolved[i] for i in range(len(s.observations))]) if s.observations else np.empty((0, 7), np.float32)
        return dict(observations=[dict(x) for x in s.observations], pose_w2c=poses,
                    pose_c2w=self.geometry.se3_inverse(poses) if len(poses) else poses.copy(),
                    reconstructed_observation_ids=sorted(s.relative),
                    reconstructed_poses_counted_as_fresh=False)

    def terminate(self):
        """Original twelve final updates, then offline reconstruction, never fresh."""
        self._running()
        try:
            s = self._state.fork()
            steps = [self._step(s) for _ in range(12)] if s.frames else []
            result = self._trajectory(s)
            result.update(offline_updates=len(steps), update_diagnostics=steps,
                          fresh_pose_valid=False, online_publication=False,
                          initialized=s.initialized,
                          trajectory_state="initialized_offline_reconstruction" if s.initialized else "initialization_incomplete")
            self._state = s
            self._counts["offline_updates"] += len(steps)
            self._status = "terminated"
            self._terminated_result = result
            return result
        except Exception as error:
            self._fail(error, "offline_termination")

    def snapshot(self, *, include_features=False):
        """Owned values; diagnostic pose arrays may include uninitialized hypotheses."""
        s = self._state
        fields = ("poses", "patches", "intrinsics", "ii", "jj", "kk", "net", "target", "weight",
                  "ii_inactive", "jj_inactive", "kk_inactive", "target_inactive", "weight_inactive")
        result = {name: getattr(s, name).copy() for name in fields}
        result.update(observations=[dict(x) for x in s.observations],
            retained_observation_ids=[r["observation_id"] for r in s.frames],
            retained_frame_ids=[r["frame_id"] for r in s.frames],
            relative={k: (parent, pose.copy()) for k, (parent, pose) in s.relative.items()},
            initialized=s.initialized, status=self._status,
            poses_are_diagnostics_not_fresh_events=True)
        if include_features:
            result["frames"] = [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in r.items()} for r in s.frames]
        return result

    def metrics(self):
        s = self._state
        features = sum(value.nbytes for r in s.frames for value in r.values() if isinstance(value, np.ndarray))
        graph = sum(value.nbytes for value in vars(s).values() if isinstance(value, np.ndarray))
        return dict(self._counts, status=self._status, failure=None if self._failure is None else dict(self._failure),
            retained_frames=len(s.frames), active_edges=len(s.ii), inactive_edges=len(s.ii_inactive),
            graph_array_bytes=graph, retained_encoded_array_bytes=features,
            frame_capacity=self.config.capacity, observation_capacity=self.config.observation_capacity,
            upstream_frame_buffer_capacity=4096, buffer_capacity_is_explicit_research_limit=True,
            active_edge_capacity=self.config.max_active_edges, inactive_edge_capacity=self.config.max_inactive_edges,
            original_config_commit=COMMIT, patch_count=self.config.patches_per_frame,
            loop_closure=False, classic_loop_closure=False, depth_units="monocular arbitrary scale",
            interpolation_counted_as_fresh=False, offline_updates_counted_as_fresh=False,
            random_initial_depth="caller RNG or NumPy CPU [0,1); CUDA bitwise equivalence not claimed",
            performance_validated=False, trajectory_accuracy_validated=False)
