# Original VGGT source and checkpoint provenance

The root [LICENSE](LICENSE) is BSD-2-Clause for VGGT-SLAM source code and retains the original MIT-SPARK copyright. It does not grant commercial-use rights to original VGGT weights or compiled models containing them. External components retain their own terms.

The sole checkpoint used by the current runtime is Meta's original [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B):

| Field | Pinned value |
| --- | --- |
| Model revision inspected | `860abec7937da0a4c03c41d3c269c366e82abdf9` |
| Weight file | `model.pt` |
| Bytes | `5026874952` |
| SHA256 and official LFS object ID | `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0` |
| Pinned model-card license | CC-BY-NC-4.0 |
| License-source check date | 2026-10-11 |

The [pinned model card](https://huggingface.co/facebook/VGGT-1B/blob/860abec7937da0a4c03c41d3c269c366e82abdf9/README.md) declares `license: cc-by-nc-4.0`. Its source SHA256 is `d666ecaba89db88cb66bf06db6daddfdc2400e13f2a39e39c84e888b27fbc615`. The [official tree metadata](https://huggingface.co/api/models/facebook/VGGT-1B/tree/860abec7937da0a4c03c41d3c269c366e82abdf9?recursive=false&expand=false) identifies file size and LFS SHA. That pinned model tree has no separate `LICENSE` file; the declaration is in the model card. [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) retains its non-commercial and attribution conditions.

The deployed source is the [VGGT_SPARK fork](https://github.com/MIT-SPARK/VGGT_SPARK/tree/6e6e16107b88e8e76c751826af10d4295d87ecd2), commit `6e6e16107b88e8e76c751826af10d4295d87ecd2`, with its [CC BY-NC 4.0 license](https://github.com/MIT-SPARK/VGGT_SPARK/blob/6e6e16107b88e8e76c751826af10d4295d87ecd2/LICENSE.txt), SHA256 `1d8ca66080a2701508a70be4ddad4be28c509bcc3397a682fb8edf767fdd7a00`. The portable patch modifies that pinned implementation; repository BSD terms do not relicense it.

Checkpoints and weight-bearing ONNX/RKNN files are not redistributed. Their encoded parameters retain the original checkpoint's terms. Exporting, compiling, converting precision or calling through a BSD-licensed wrapper grants no additional rights. Rockchip SDK/runtime terms apply independently. This document describes the fixed source and checkpoint actually used; it offers no alternative checkpoint or license comparison. [Full pins](UPSTREAMS.json), [third-party attribution](THIRD_PARTY_NOTICES.md).

The native GPU diagnostic uses an external Mali userspace library and matching GPU firmware under their vendor terms. Neither binary is redistributed here, and the repository BSD license does not cover either. The [diagnostic provenance](results/original_native_gpu_v81.json) records their fixed vendor commits, binary hashes and the associated vendor license hash separately from original VGGT source and weights.
