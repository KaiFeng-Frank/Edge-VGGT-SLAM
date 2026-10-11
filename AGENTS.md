# Project constraints

These constraints record the user's current requirements, confirmed on 2026-10-12.

- Work on VGGT-SLAM2 with its original VGGT model. Do not replace it with DPVO, DA3, another backbone, or another learned pose/depth model. Stop alternative-model research and experiments. Existing alternative-model files and measurements are historical archives, not candidate implementation paths or VGGT acceptance evidence.
- Do not add an ORB or LK/PnP pose frontend. Original upstream keyframe selection is distinct from publishing poses estimated by a new traditional frontend.
- Improve execution of the original model: operator implementations, memory/layout, validated reuse of unchanged computations and CPU/NPU partitioning. Preserve the original checkpoint provenance and compare numerical outputs. A shared execution API permits other hardware, not a different model.
- All real model inference, tracing/export validation and calibration must run on RK3588S. Host source inspection, static graph work, non-calibrating compilation, saved-data audits and small synthetic checks are allowed.
- Use real dataset replay for now; physical camera experiments are deferred. Keep all inference on the board. Do not change system-wide clocks, governors, cooling, drivers or network configuration without authorization.
- Measure fresh-observation pose rate and complete input-to-pose latency. Do not count interpolation, repeated predictions, offline reconstruction or a different model toward VGGT's real-time target. Track map cadence, bounded queues, 30-minute stability, quality and 8 GB suitability separately.
- Record valuable engineering issues with their evidence, reproduction conditions, failed attempts and validation limits. Public GitHub publication of results is authorized. Do not publish checkpoints, captured model arrays, SDK binaries or credentials.
- Keep repository code, model weights and third-party licenses separate. The BSD-2-Clause repository license does not grant commercial rights to the original VGGT weights.

For historical evidence and current scope, see [README](README.md), [architecture](ARCHITECTURE.md) and [engineering notes](docs/ENGINEERING_NOTES.md).
