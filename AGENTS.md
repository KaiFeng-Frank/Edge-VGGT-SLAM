# Project constraints

These constraints record the user's requirements, confirmed on 2026-10-12.

- Use VGGT-SLAM2 with its original pretrained VGGT model only. The current tree must contain no other learned model, auxiliary checkpoint, alternative-model experiment, reproduction option or documentation entry. VGGT's own internal DINO modules remain part of the original model. Historical records remain in git history and private experiment storage.
- Add no traditional pose frontend. Preserve the original model, pose codec, submap equations and SL4 graph. Original upstream keyframe selection is distinct from adding a separate pose estimator.
- Additional-model retrieval and semantic implementations are removed. Loop closure has no replacement; support `max_loops=0` and reject unsupported loops or legacy auxiliary-model arguments before any loading or inference.
- Improve original-model operators, memory/layout, proven unchanged computation reuse and CPU/NPU partitioning. Keep original checkpoint provenance and compare original numerical outputs on identical inputs. Do not substitute a different model, omit layers or change token computation to claim equivalent execution.
- Run all actual inference, tracing/export validation and calibration on RK3588S. Host source inspection, static graph work, non-calibrating compilation, saved-data audits and small generated checks are allowed.
- Use real dataset replay for now; physical-camera tests are deferred. Do not change system-wide clocks, governors, cooling, drivers or network configuration without authorization.
- Measure at least 20 Hz fresh-observation pose rate and complete input-to-pose P95 at most 100 ms against the original model. Repeated predictions, interpolation, offline reconstruction and component-only timings do not count. Assess trajectory quality, map cadence, bounded queues, 30-minute stability and 8 GB operation separately. These goals are not yet achieved.
- Record valuable engineering evidence, failed attempts, reproduction conditions and limits. Public GitHub publication is authorized. Publish no checkpoints, captured model arrays, weight-bearing exports, SDK binaries or credentials. Do not delete private raw research records or rewrite git history during scope cleanup.
- Retain attribution for all code still used. Repository code, pinned original-model source, checkpoint and third-party dependencies have separate licenses. BSD-2-Clause does not grant commercial rights to the original weights.

See [README](README.md), [architecture](ARCHITECTURE.md), [engineering notes](docs/ENGINEERING_NOTES.md) and [model provenance](MODEL_LICENSES.md).
