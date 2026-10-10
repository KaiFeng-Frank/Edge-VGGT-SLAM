# Model and source-code license provenance

The retained [LICENSE](LICENSE) is BSD-2-Clause for VGGT-SLAM source code, with the original MIT-SPARK copyright. It is **not a commercial-use authorization for VGGT weights or for compiled models containing them**. External components keep their own licenses.

## Original weights used in these experiments

The experiments use [Meta's original facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B), not a commercial or later-generation checkpoint:

| Field | Value |
| --- | --- |
| Inspected model revision | `860abec7937da0a4c03c41d3c269c366e82abdf9` |
| Weight file | `model.pt` |
| Bytes | `5026874952` |
| SHA256 / official LFS object ID | `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0` |
| Model-card license | **CC-BY-NC-4.0** |
| License-source check | 2026-10-11 |

The [pinned model card](https://huggingface.co/facebook/VGGT-1B/blob/860abec7937da0a4c03c41d3c269c366e82abdf9/README.md) declares `license: cc-by-nc-4.0`. The [official tree metadata](https://huggingface.co/api/models/facebook/VGGT-1B/tree/860abec7937da0a4c03c41d3c269c366e82abdf9?recursive=false&expand=false) binds that revision to the exact size and SHA above. The same weight object already exists at the [2025-03-14 upload revision](https://huggingface.co/api/models/facebook/VGGT-1B/tree/d88e637d32a505f4a64de03f8588547b7f7d3ba6?recursive=false&expand=false). That historical tree proves file identity; its older model card does not itself declare a license. At the inspected current revision, the model repository has no separate `LICENSE` file; the license declaration is in its model card.

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) limits its reproduction/sharing and adapted-material grants to non-commercial purposes (§2(a)(1)); attribution and modification notices apply when sharing (§3). The repository's BSD license does not remove these conditions.

## Code and commercial checkpoints are separate

Meta changed its code license on 2025-07-29. The [current code license](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/LICENSE.txt) is the custom VGGT License with an Acceptable Use Policy. Meta's [release notice](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/README.md#updates) expressly distinguishes the original non-commercial weights from [VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial). The latter is a separate gated release with its own terms; it is not used or validated here. Its terms, or terms for later VGGT variants, must not be applied to the original SHA above by assuming that the model names are interchangeable.

The deployed implementation is the pinned [VGGT_SPARK fork and its CC BY-NC 4.0 license](https://github.com/MIT-SPARK/VGGT_SPARK/blob/6e6e16107b88e8e76c751826af10d4295d87ecd2/LICENSE.txt). The published patch modifies that implementation. This publication does not replace that pinned license with the current Meta code license or BSD.

## Optional Co-Me confidence predictor research

The Co-Me authors release a separate confidence predictor at [commit `5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833`](https://github.com/co-me-tokens/CoMe/tree/5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833), under a repository whose [root license](https://github.com/co-me-tokens/CoMe/blob/5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833/LICENSE) is **CC BY-NC 4.0**. The inspected [predictor directory](https://github.com/co-me-tokens/CoMe/tree/5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833/output/confidence_distill/vggt/20260324_172742/step_02000) contains `checkpoint.pth` and `config.json`; no separate predictor license or model card was found there. This is not evidence of an additional commercial-use grant.

The predictor file is 831,722 bytes, Git blob `19c8a02c8700c8964064ec0b7d479ed686430dda`, SHA256 `95b0a004544c96da5c4e55748e8cf4837608627a6414563af06cb3ee25acd29f`. Its 18 BF16 tensors contain 412,417 parameters (824,834 tensor-storage bytes). Only source/weight metadata and the attributed research adaptation are published here; the checkpoint is not redistributed. Co-Me-derived helper code retains its CC BY-NC scope. The predictor and Meta's original VGGT-1B are distinct external releases, even though the research preserves VGGT base weights. See [Co-Me provenance](results/come_sources_v22.json). Token merging remains an approximation experiment, not a license change or a production model substitution.

## Alternative-model metadata research

The v23 source review also examines [DA3-SMALL at revision `e08cab65ca0ec38e7826075418411ab90cab4da3`](https://huggingface.co/depth-anything/DA3-SMALL/tree/e08cab65ca0ec38e7826075418411ab90cab4da3). Its model card declares Apache-2.0 independently of the [Depth-Anything-3 source license](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/LICENSE). This review downloaded configuration and safetensors header metadata only, not the full checkpoint; it did not replace or execute the deployed model. The declaration belongs to that distinct model release and does not change the original VGGT checkpoint's license. Exact revisions, source hashes and the distinction between expected and independently measured weight hashes appear in [the research receipt](results/acceleration_research_v23.json).

## ONNX/RKNN exports and redistribution

The export scripts and result metadata are published; pretrained checkpoints and weight-bearing ONNX/RKNN binaries are **not**. Locally compiled models encode parameters from the identified checkpoint. Compilation, FP16 conversion or use of a BSD-licensed wrapper does not supply a new commercial license for those parameters. This remains true whether a particular format conversion is a technical modification or adapted material: CC BY-NC §2(a)(4) permits technical format changes without lifting §2(a)(1)'s non-commercial condition. Rockchip SDK/runtime terms also apply independently.

Machine-readable source hashes, revision evidence and this distinction are recorded in [UPSTREAMS.json](UPSTREAMS.json) and [results/model_license_v21.json](results/model_license_v21.json).
