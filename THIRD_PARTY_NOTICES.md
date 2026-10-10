# Upstream attribution and external dependencies

This repository preserves the BSD-2-Clause license and 2025 MIT-SPARK copyright of VGGT-SLAM. The upstream tree is commit `35327ac28b7d193df9ccc39ba6346052bb6f1207` of https://github.com/MIT-SPARK/VGGT-SLAM. The upstream README is retained separately.

Third-party repositories, checkpoints, headers, compiled libraries and data are external dependencies and are not redistributed by this publication. Their licenses apply independently:

- VGGT_SPARK: https://github.com/MIT-SPARK/VGGT_SPARK, commit `6e6e16107b88e8e76c751826af10d4295d87ecd2`, `LICENSE.txt`, CC-BY-NC-4.0. The published `vggt-spark.patch` modifies this implementation and must be used with its license and attribution.
- Official VGGT-1B model: https://huggingface.co/facebook/VGGT-1B. Download from the official model page and retain its model/license notices.
- Optional SALAD: https://github.com/serizba/salad, commit `33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d`, `LICENSE`, GPL-3.0. An optional dependency is not relicensed as BSD.
- Optional DINOv2: https://github.com/facebookresearch/dinov2, commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, `LICENSE`, Apache-2.0.
- RKNN SDK: https://github.com/airockchip/rknn-toolkit2. SDK headers and binaries require their own Rockchip license; they are excluded here. Match the tested 2.3.2 API/runtime and the driver's expected header hashes.
- XNNPACK and pthreadpool: https://github.com/google/XNNPACK and https://github.com/Maratyszcza/pthreadpool, each BSD-licensed. Installed Torch 2.3.1 supplies the runtime and headers used by the adapter. External copies of reference headers are not bundled here.
- TUM RGB-D data: https://cvg.cit.tum.de/data/datasets/rgbd-dataset, CC-BY-4.0. Cite the dataset paper and retain its attribution if redistributing data. No dataset images, ground truth or sensor depth are included.

Package download manifests identify upstream distribution artifacts; a manifest does not grant a different license to those packages.
