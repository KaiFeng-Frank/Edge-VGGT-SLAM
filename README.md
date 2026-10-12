# Edge-VGGT-SLAM

将 **VGGT-SLAM2 的原始预训练 VGGT** 移植到 RK3588S，输出相机位姿和稠密地图。模型执行与原有子地图构建、SL4 图优化分离，便于实现和验证嵌入式执行后端。

当前只支持原 VGGT checkpoint；其内部 DINO、frame/global attention、CameraHead 和 DPT 均保留。没有额外模型或传统位姿前端。推理、校准与模型验证全部在 RK3588S 板端完成。

## 当前能做到什么

已完成原模型的 CPU 参考路径和实验性 RKNN 全链路，并在真实 TUM 数据集上按原时间戳回放。当前实验链路的 aggregator 与 CameraHead 采用 W8A8 编译配置，DINO 与 DPT 保留 FP16；原预处理、位姿解码、子地图和 SL4 保持原实现。

最近一次完整回放的实测如下；这是 16 GB RK3588S 上、NPU 1 GHz 的短时实验，输入为两帧 `84×112` 图像：

| 指标 | 实测 |
| --- | ---: |
| 数据集输入 / 新观测位姿更新 | 798 帧 / 31 次 |
| 新观测位姿平均更新率 | **1.1167 Hz** |
| 完整输入到位姿发布 P95 | **1.0688 s** |
| 回放时长，含 EOF 排空 | 27.7595 s |
| 启动时间，另计 | 57.3481 s |
| 峰值进程 RSS | 3380.19 MiB |

**尚未达到实时目标：新观测位姿至少 20 Hz，完整延迟 P95 不超过 100 ms。** 30 个发布间隔中仍有 1 个超过一秒，最大 1.0414 s；平均更新率不代表每秒都有新位姿。

精度仍处于实验验证阶段。17 帧固定短序列经 Sim3 对齐后的 ATE / RPE RMSE 相对原 CPU 参考的变化在 10% 以内，但旋转 RPE RMSE 上升 7.30%，最差旋转段达到 27.24°；严格 FP32 张量比较未通过。这不建立普遍轨迹或地图质量结论，也不能把固定输入的质量结果直接用于另一次不同选帧的回放。

物理相机测试暂缓；30 分钟稳定性和物理 8 GB 板运行尚未验收。当前没有已验证的回环实现，Solver 只支持 `max_loops=0`。

[完整测量、精度边界与复现步骤](deploy/rk3588/validation/original_calibrated_camera/README.md) · [机器可读结果](results/original_calibrated_camera_v82.json)

## 准备板端环境

已测试的环境是 Linux aarch64、Python 3.11.17、CPU PyTorch 2.3.1、torchvision 0.18.1，以及提供 SL4 的 GTSAM 构建。NPU 路径使用 RKNN SDK / runtime 2.3.2。

按照[部署指南](deploy/rk3588/README.md)准备原模型源码和[外部依赖](deploy/rk3588/dependencies/README.md)。以下沿用已验证的板端目录 `/mnt/vggt-data/vggt-slam2`；仓库位于其中的 `src/`。将所需的 `python/`、`wheelhouse/` 等文件准备好后，运行离线安装器：

```sh
sh /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/install-offline.sh
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
```

原权重单独下载，不随仓库分发：

```sh
python /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/download_weights.py \
  --output /mnt/vggt-data/vggt-slam2/weights/model.pt
```

唯一支持的权重为原 `facebook/VGGT-1B/model.pt`，大小 **5,026,874,952 bytes**。下载器及加载器会校验固定完整 SHA256，加载前必须通过；[固定源码、权重版本和哈希](UPSTREAMS.json)。ONNX / RKNN 文件需要在授权环境中自行生成，构建入口见对应复现目录。

## 运行时间戳回放

准备好 TUM 数据集的 `rgb.txt` 和图像后，在板端运行默认的原 CPU FP32 参考路径：

```sh
. /mnt/vggt-data/vggt-slam2/src/deploy/rk3588/env.sh
python -B -u /mnt/vggt-data/vggt-slam2/src/scripts/replay_original_vggt.py \
  --rgb-index /mnt/vggt-data/vggt-slam2/datasets/rgbd_dataset_freiburg1_xyz/rgb.txt \
  --checkpoint /mnt/vggt-data/vggt-slam2/weights/model.pt \
  --output /mnt/vggt-data/vggt-slam2/experiments/original-replay-new-run
```

使用新的输出目录。默认图像宽度为 112；该数据集的实际输入高度为 84。解码器使用 CPU0，模型默认使用 CPU4–7，不要把整个进程的亲和性限制到 CPU4–7。完整参数见 `python scripts/replay_original_vggt.py --help`。

输出包括 `poses.tum`、`map.json`、`map_chunks/` 和 `results.json`，位姿与地图在真实新观测计算及原 SL4 完成后发布，输入队列容量为 1。

上表的 NPU 实验需要原模型的固定编译产物及实验 wrapper；按[当前 NPU 全链路复现说明](deploy/rk3588/validation/original_calibrated_camera/README.md)构建和启动。普通回放命令保持 CPU 参考默认配置。频率实验使用有界温控控制器并恢复原频率；回放入口本身不修改系统时钟或 governor。

## 开发与证据

- [执行架构](ARCHITECTURE.md)：原模型输出契约、缓存和 SL4 的边界。
- [项目约束](AGENTS.md)：唯一原模型、板端执行及验收要求。
- [测量索引](MEASUREMENTS.json)与[结果目录](results/)：完整链路、组件实验及失败记录。
- [工程笔记](docs/ENGINEERING_NOTES.md)：有价值的问题、原因证据和后续验证条件。

已配置开发依赖时，可运行现有轻量回归检查：

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_runtime_loading.py tests/test_model_execution.py \
  tests/test_solver_execution.py tests/test_patch_cache.py
```

这些检查使用生成数据或 fake 服务，不代替原模型板端推理、轨迹质量或实时验收。实验的组件计时与端到端回放结果分开记录。

## 许可与致谢

基于 MIT-SPARK 的 [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) 和固定版本的 [VGGT_SPARK](https://github.com/MIT-SPARK/VGGT_SPARK)。保留 MIT-SPARK 版权及仓库的 [BSD-2-Clause 许可](LICENSE)。

**仓库代码许可不等于模型权重许可。** 本项目使用的原 VGGT 权重及固定实现有独立的非商业使用条款；编译产物中的权重仍受原条款约束。详见[模型来源与许可](MODEL_LICENSES.md)、[第三方声明](THIRD_PARTY_NOTICES.md)和[上游版本记录](UPSTREAMS.json)。仓库不分发权重、捕获数组、含权重的导出图或 SDK 二进制。
