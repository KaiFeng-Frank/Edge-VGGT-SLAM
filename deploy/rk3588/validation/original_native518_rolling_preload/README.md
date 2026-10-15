# Original native518 aggregator with rolling group preload

A bounded full-model trial hides next-group setup while preserving frozen NPU
outputs. **Retain the experimental schedule; defaults are unchanged.** See
[results and evidence](../../../../results/original_native518_rolling_preload_v123.json).

Both routes compute two original `392×518` images through all 24 DINO layers
per image, all 48 aggregator blocks, four CameraHead updates, both DPT frames
and the original codec. Previous-image reuse is disabled; the same real TUM
pair is repeated. All four compiled aggregator groups and equations remain.

A loads one group at a time. B initializes only the next while the current
runs, with at most two reserved/initializing contexts. Each worker owns its
flags0/core7 init, run and close. Group runs remain serial and consume actual
previous-global output. Workers join before heads or final cleanup; no
reference intermediate supplies a prediction.

The actual **ABBA** screen records every call, without a hidden warmup:

| Call | Route | Complete model prediction | Aggregator and consumers |
| --- | --- | ---: | ---: |
| 0 | A: sequential | 33.69636 s | 18.61182 s |
| 1 | B: rolling | 27.22179 s | 16.13499 s |
| 2 | B: rolling | 27.40540 s | 16.13067 s |
| 3 | A: sequential | 29.62125 s | 18.52575 s |

Prediction starts after PNG preprocessing and includes token preparation,
checks/hash/init/IO/run/destroy, codec and output ownership. Startup, saving
and comparison are separate; PNG read/preprocess adds 34–45 ms separately.
Aggregator means are **18.56878 / 16.13283 s**, **2.436 s** less wall time.
Serial aggregator SDK sums remain **13.70681 / 13.84864 s**. Full means
31.65880/27.31359 s are descriptively 13.725% apart, with extra first-A
DINO/head overhead. Neither stable nor purely preload-caused full-model
acceleration is established. Overlapping stage walls cannot be added.

R1 aborted with SIGBUS after one successful A: low-zone NPU allocation failed
despite substantial total free RAM. PFN inspection located **512,090,112 B**
of unused group0 ONNX pages below 4 GiB; group1's compiled file was above it.
Moving the whole export directory to disk preserved bytes/hashes, old-path
symlink and all four compilation lineages. Models, drivers and memory policy
were unchanged; the move is reversible, and R2 then completed.

All 20 final arrays match the frozen native NPU pair byte for byte. The
[saved-only audit](audit_full_r2_saved.py) verifies 40 contexts, 72 runs,
92 released outputs and terminal workers. V111's FP32 trajectory failure remains.
The NPU 1 GHz controller observes 69.307°C maximum and restores 300 MHz.
Peak RSS is 1,094.13 MiB; zone/SDK fields are separate, not 8 GB acceptance.
Bundles are in [r1](r1/sources.json) and [r2](r2/sources.json); private pinned
models/data and restoring control are required. No weights, arrays or SDK
binaries are distributed.

From the repository root, with NumPy, run the portable fake-only owner checks:

```sh
python3 deploy/rk3588/validation/original_native518_rolling_preload/test_rolling_native_aggregator.py
```

Stop further preload tuning. The **26.7345 s cached-pair** result remains
separate and unchanged in [v111](../../../../results/original_native518_trajectory_quality_v111.json).
No evidence-backed 20 Hz route is demonstrated; P95, SL4, map cadence and stability remain unvalidated, without claiming physical impossibility.
