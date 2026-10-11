# Original VGGT precision and compiler diagnostics

These tools retain the original pretrained VGGT and the complete original frame0/global0 pair. They introduce no other model or pose frontend. They do not implement complete NPU VGGT, real-time SLAM or loop retrieval.

The preceding [v49 capture and isolated toolchain](../original_precision/README.md) supply the original graph, eight real calibration pairs, two calibration-disjoint evaluation pairs and original Torch FP32 outputs. The two evaluation pairs have now been reused for development; they are not a fresh blind trajectory test. Actual inference, calibration and SDK execution run only on RK3588S. Host checks here read source or saved arrays, or use generated numbers/fake APIs.

- [Representation analysis](accuracy/RESOLUTION_NOTE.md) proves per-saved-tensor FP16/observed INT8 grid limits and preserves the strict component rejection. Its 16 generated math checks pass. Full-model ATE/RPE acceptance remains a separate evaluation.
- `flash/compile-v50-flash-pair-board.py` compiles the exact original ONNX with the same isolated toolchain and two fresh SDK objects, changing only `enable_flash_attention`. Both variants are nonquantizing.
- `flash/run-v50-flash-pair-board.py` reuses the frozen typed SDK call and owned-output implementation from v49. Normal and profile modes use separate processes/contexts, 16 calls each, with eight calls per role and six subsequent calls per role for the normal mean.
- `hybrid/prepare-v50-hybrid-step1-board.py` calibrates only the eight original captured inputs and preserves the generated model/data/configuration byte-for-byte. A precision policy must be derived from the actual SDK tensor catalog before step2. The policy review is an engineering configuration check, not numerical or production acceptance.

Flash off/on normal SDK IO means are 336.829 / 336.963 ms: no improvement observed in this short original-pair test. The visible compiled operator identities, shapes, dtypes and placement are the same. Equal operator tables do not prove identical low-level kernels or byte-identical binaries. The [official 2.3.2 CHANGELOG](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/CHANGELOG.md) explicitly names RK3562/RK3576 in its Flash Attention support entry; it provides no RK3588 support statement there.

All strict FP32 component failures remain recorded at `rtol=1e-4, atol=1e-5`. The closest FP16 output still cannot meet that gate for every element of these references. This does not establish trajectory failure or hardware impossibility; ATE/RPE, map quality, fresh-observation rate, latency, queues, 30-minute stability and 8 GB operation need their own complete-system measurements.

Weights, captured arrays, compiled graphs and SDK binaries remain private. Published hashes bind the actual evidence but do not provide a checkpoint license. Original source, model weights and SDK have independent licensing; see the root model provenance and third-party notices.

Independent audits are reproducible from the private saved artifacts:

- [Numeric outputs, I/O timing and generated checks](flash-audit-numeric/README.md).
- [Visible compiled and runtime operator evidence](layout/public-flash-operators-audit-r1.json). The portable auditor CLI and its frozen source hashes are in [its manifest](layout/flash-publication-r1.json).
- [Combined measurement record](../../../../results/original_mixed_precision_v50.json).

The static layout auditor retains an abstract original-MLP scope annotation (18 original activation outputs). It is graph-boundary evidence, not an SDK configuration or an executed precision policy. The first SDK policy requested INT8 for four FC1/GELU catalog selectors and FP16 for the other 91. Actual step2 rejected external `input`, before export. The revised scope is derived from the SDK-generated graph: four INT8 requests and 61 FP16 requests for actual internal node outputs, excluding external I/O, initializer selectors and absent estimator names. Excluded calibration values remain unchanged; their final arithmetic is not established by configuration. Actual compiled placement and interface descriptors determine what the SDK implements.

The revised hybrid artifact compiled successfully on RK3588S and completed 16 normal plus 16 separate-profile calls. Its queried logical I/O is floating. Normal means were 336.673 / 319.928 ms for ordinary / hybrid (4.974% lower in this short component test); hybrid relative L2 against the original FP32 references was 6.672% / 7.908%. Both candidates remain strictly rejected, and the production model path is unchanged. Use the frozen `hybrid/run-v50-hybrid-step2-r2-board.sh` with its original private step1 artifacts and reviewed r2 configuration, followed by `hybrid-runtime/run-hybrid-r2-board.sh normal` and, in a new process, `profile`. `hybrid/` retains the exact failed r1 source and receipt hashes for traceability; r1 is not a successful compilation recipe.

[Actual hybrid saved numeric audit and reproduction](hybrid-audit-numeric/README_R2.md) verifies the 32 saved outputs, descriptor/grid observations, fixed-reference failures and complete I/O timing. Its generated checks are separate from actual board evidence.

Frozen preparation notes retain their original pre-run wording. They document the tested sequence and are not current acceptance claims; the completed measurements and failed-attempt records above determine the actual status.

[Actual hybrid operator audit](layout/public-hybrid-operators-audit-r1.json) verifies both FC1/GELU INT8 placements, floating FC2 and residual inputs, 56 internal NPU operators, and four added conversions averaging 9.407 ms in the separate profile. [The frozen parser manifest](layout/hybrid-operators-publication-r1.json) records the portable CLI and one exact, SHA-bound stdout interleave repair; the original log remains intact.
