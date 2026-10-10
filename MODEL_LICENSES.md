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

## ONNX/RKNN exports and redistribution

The export scripts and result metadata are published; pretrained checkpoints and weight-bearing ONNX/RKNN binaries are **not**. Locally compiled models encode parameters from the identified checkpoint. Compilation, FP16 conversion or use of a BSD-licensed wrapper does not supply a new commercial license for those parameters. This remains true whether a particular format conversion is a technical modification or adapted material: CC BY-NC §2(a)(4) permits technical format changes without lifting §2(a)(1)'s non-commercial condition. Rockchip SDK/runtime terms also apply independently.

Machine-readable source hashes, revision evidence and this distinction are recorded in [UPSTREAMS.json](UPSTREAMS.json) and [results/model_license_v21.json](results/model_license_v21.json).
