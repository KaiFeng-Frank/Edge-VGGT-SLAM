# Original native518 whole-global0 W8A8 diagnostic

The calibrated W8A8 candidate reduced complete component latency by 17.17%,
but substantially worsened numerical error. This configuration is rejected;
it is not integrated and is not extended to all 48 aggregator blocks.
[Compact recorded results](../../../../results/original_native518_global_w8a8_v115.json)
contain the exact source, artifact, receipt and launcher SHA256 pins.

The original c440 ONNX retains all 18 learned parameter tensors before conversion,
53 nodes, Erf GELU, RoPE, FC2K4, residuals, 16 heads and all 2082 queries/keys.
Six real pairs use original 518 preprocessing (`[2,3,392,518]`), original NPU
DINO24, final normalization/role tokens and standalone original frame0.
That frame0 boundary is not certified byte identical to the deployed retained
group's internal boundary. Four cases calibrate; two disjoint cases evaluate.
The unchanged Torch global0 references use each exact captured NPU-derived
input. No CPU intermediate is substituted into a complete model execution.

Capture r2 completed. Both held-out ONNX CPU FP32 checks passed the unchanged
`rtol=1e-4`, `atol=1e-5`, relative L2 `<=1e-5`, maximum error `<=1e-3` gates.
The board-only build completed in about 83 seconds using RKNN2.3.2 W8A8
`normal`/`channel`, exactly four dataset rows and no automatic hybrid scheme
or source graph rewrite. The candidate is 16,054,672 bytes; internal operator mapping
was not profiled, so all-INT8 placement is not established.

ABBA used both held-out inputs in each phase: eight calls with fresh contexts.
Each timer includes checks/hashes, init/query, SDK FP32 conversion, run,
get/copy/release, finite checks and destruction. The table uses three samples
per route after its first invocation; it is not resident-context throughput.

| Measurement | FP16 A | W8A8 B |
| --- | ---: | ---: |
| Complete lifecycle mean, seconds | 0.623813 | 0.516679 |
| SDK run mean, seconds | 0.346931 | 0.328071 |
| Normal IO boundary mean, seconds | 0.404336 | 0.357009 |
| Whole-output relative L2, two held-out cases | 0.006046 / 0.005961 | 0.239313 / 0.229614 |
| Residual-update relative L2, two held-out cases | 0.011770 / 0.011719 | 0.465912 / 0.451411 |

SDK execution decreased only 5.44%; the larger complete-boundary change also
contains lifecycle/conversion costs. Both routes fail the original strict
FP32 comparisons. All repeats within each route/input are byte identical.
Eight contexts, eight runs and eight output buffers were balanced and closed.
The controller completed in 8.647 seconds, peaked at 61.923°C and restored
exactly the original 300 MHz frequency policy.

Candidate outputs lie exactly on the decoded INT8 output grid. Even nearest
possible reference values on that fixed grid have relative L2 around 0.043.
`want_float=1` converts representation without recovering lost precision.
This bound does not identify internal error causes or predict a model with
floating boundaries. Two evaluation pairs establish no trajectory acceptance.

Prerequisites remain private: frozen S100 helpers/config, original checkpoint
and sources, c440 graph, v53 parameter receipt, v91 configuration and v98 FP16
block. Captures, compiled models and SDK binaries are not shipped or downloaded
automatically. Final capture/compiler sources are staged in S115-r2; the actual
launchers are `run_capture_1ghz_r2.sh`, `run_check_r3.sh`, `run_compile_r1.sh` and
`run_abba_r1.sh`. Checking and compilation both use board SDK Python with
process-local LD/thread settings, a 180-second timeout and fresh rootfs `/tmp`
outputs with at least 1 GiB free; compiler cwd/TMPDIR and scratch are retained.
Capture and ABBA require the existing 1 GHz / 75°C / 180-second restoring
controller. For a rerun, update worker and controller outputs to fresh paths.

Failed attempts are preserved: capture r1 failed whole-stat checkpoint
comparison; access-time sensitivity was suspected, not established. Check r1
hit missing older source metadata; check r2 used Python without ONNX. R2 still
pins every native source, verifies shared source digests and final checkpoint
SHA. No numerical gate was relaxed. No full-model, realtime, 30-minute or 8 GB
claim follows. Repository code and original weights retain separate licenses.
