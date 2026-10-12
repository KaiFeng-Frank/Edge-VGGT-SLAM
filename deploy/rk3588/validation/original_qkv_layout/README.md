# Original QKV layout diagnostic (v75)

Two original frame0 QKV layout candidates were tested entirely on RK3588S at 300 MHz. Neither is integrated into the full runtime. Both preserve all learned parameters, layers, tokens and the original frame0/global0 public outputs. [Measurements](../../../../results/original_qkv_layout_v75.json).

The first permutes the original QKV weight columns and bias from `[3,16,64]` to `[16,3,64]`, with inverse-permutation raw-byte proofs and matching reshape/transpose changes. Its normal complete-pair mean was **38.415 → 38.891 ms**. The intended approximately 1.3 ms NPU reshape remains.

The follow-up combines independent frame/head axes only inside frame attention, restores them before original head concatenation, and requires both original RoPE grids to be byte-identical before broadcasting one stored grid. Its mean was **38.994 → 46.241 ms**. Profile shows an explicit CPU FP16 five-dimensional transpose despite reducing total operator count. It is slower, so this route is closed.

Both candidates passed board CPU ORT checks: two real inputs and both public outputs must be FP32 byte-identical to the original graph before compilation. SDK A/B outputs, repeats and profile/normal outputs also match bytewise on the saved inputs. Original Torch strict comparisons remain 0/84 NPU checks passing; the second real input has no original Torch reference. The fixed tolerance was not changed.

## Reproduction

These scripts require the privately produced v55 original pair graph, v53/v56 real captures, original helper scripts, pinned external RKNN toolkit/runtime and NEON library. They intentionally reject missing or changed identities. Do not put weight-bearing artifacts in this repository.

On the board, use the compiler environment to run `build_pair_qkv_board.py --output-dir NEW_DIRECTORY`, or `build_pair_qkv_folded_board.py` for the follow-up. Each runs the exact-output gate before nonquantized FP16 compilation. The folded builder requires both rewrite helpers beside it. Then use the board runtime environment to run the matching `probe_pair_qkv*_board.py --compiled DIRECTORY RESULT_JSON_SHA256 --rounds 10 --output-dir NEW_DIRECTORY`. Use a separate new output directory and `--profile` for instrumentation; profile calls are excluded from normal timing. `run_board_stage.py` is the read-only 300 MHz monitor used for these calls. It does not change frequency.

Normal timing includes conversion, input submission, run, both output reads, owned copies, release, widening and finite checks. The ABBA schedule excludes only the first call per input/route. Hashes, file saves and profile queries are outside that interval. Source and saved-output audit receipts are included; raw captures and graph exports remain private. This is a component diagnosis, with no new trajectory-quality or real-time claim.
