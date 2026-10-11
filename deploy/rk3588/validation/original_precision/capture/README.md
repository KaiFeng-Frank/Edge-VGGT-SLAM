# 原 VGGT 的板端前缀数据捕获

本目录只准备原模型的校准与独立评估输入。原始 checkpoint、模型代码、像素预处理和 `frame0→global0` 运算保持原样；原模型内的全部 24 个 DINO block 均执行。生产入口未修改。实际推理只允许在 RK3588S 执行；本目录的本地检查使用生成数据和普通 Python hook 调度 fixture。

八组 `calibration` 在原 `frame_blocks[0]` 的 kwargs pre-hook 中复制真实输入和 RoPE positions，然后用私有 sentinel 截停，首个 frame block 的运算体不执行。两组 `evaluation` 则执行原 `frame0` 和原 `global0`，保存 frame 中间参考与 global 最终 FP32 参考，并在 global post-hook 中截停。这些截停只限定实验范围，不代表完整模型省略这些计算或已经加速。

| split | 起始 RGB 索引 | 每组第二张 | 捕获数组 |
|---|---|---|---|
| calibration，8 组 | 32、132、232、332、432、532、632、732 | 起始索引 + 8 | `input.npy`、`positions.npy` |
| evaluation，2 组 | 80、480 | 起始索引 + 8 | 上述两份及 `frame-reference.npy`、`output.npy` |

计划在推理前按固定索引确定，来自真实 TUM `rgbd_dataset_freiburg1_xyz`，不按预测结果挑选。20 张图片的索引、原始时间戳和 PNG SHA 均互斥，并排除旧配对评估的 RGB 0、8。完整 `[2,437,1024]` FP32 输入是唯一校准数据；位置、独立评估输入及其参考输出均不得写入校准 dataset。

预处理调用冻结的 v17 `prepare_rectified_images`：原校准参数的 OpenCV rectification、无损 PNG/BytesIO 桥接、原 VGGT PIL loader。请求尺寸 336，实际 RGB tensor 为 `[2,3,252,336]`，不改变像素处理、原 first/other camera/register token 角色或原 RoPE positions。位置数组为 `[2,437,2]` int64；global 输出为 `[1,874,1024]` FP32。捕获的是原聚合器传入 block 的实际 tensor，不手工拼接特征。

`capture.json` 保存 checkpoint/source SHA、36 个原配对参数身份、PNG 来源与原始时间戳字符串、每份 NPY 的 file/raw SHA、shape/dtype/字节数、实际 DINO/frame/global 调用计数、特殊 token 角色、原 frame/global view 连续性及 hook/forward/参数状态和进程环境恢复结果。加载器逐项核验保存的数据；离开板端后，物理 checkpoint/PNG/源码 SHA 属于板端捕获的记录，不声称本地又读取了这些板端文件。

源码冻结见 [source-freeze-r1.json](source-freeze-r1.json)。`CAPTURE_SOURCE_SHA` 在 [v49_capture_contracts.py](v49_capture_contracts.py) 中绑定主脚本，调用时再显式绑定合同文件 SHA，避免互相嵌入 SHA 的循环。不要在开始实际捕获后重写冻结源码。

```bash
cd /mnt/vggt-data/vggt-slam2
venv/bin/python experiments/vggt-v49-capture-source/capture-v49-pair-prefix-board.py \
  --plan experiments/vggt-v49-image-plan.json \
  --plan-sha256 61b14e0e0033a49059bf52c7dd9cbeaae8bd4393423b2bce827cc451f22caf37 \
  --pair-dir experiments/vggt-v48-pair \
  --contracts experiments/vggt-v49-capture-source/v49_capture_contracts.py \
  --contracts-sha256 6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8 \
  --source-pins experiments/vggt-v49-capture-source/original-vggt-source-pins.json \
  --output-dir experiments/vggt-v49-capture-r1
```

上述已部署 pair 路径需与主任务的实际冻结目录对应。输出目录必须不存在。脚本在任何真实图像解码或 NumPy/OpenCV/Torch 加载前拒绝非 RK3588 Linux aarch64 运行。进程使用允许的 A76 核 4–7、Torch 4 线程；只改当前进程的线程环境和 affinity，并在 finally 恢复。没有系统时钟、governor、驱动或散热设置变更。

可在本地运行 `python3 check_capture_contracts.py --output /tmp/v49-capture-selfchecks.json`。15 个生成数据检查涵盖两种精确截停、24 个 DINO 调用、输入/输出不替换、sentinel 身份、异常传播、部分 hook 注册清理、后续 block/head tripwire、first/other 顺序、时间戳精度、split/旧评估排除，以及保存 NPY 的 dtype/shape/finiteness/raw SHA 和路径范围。检查不运行真实模型、图像解码、校准或编译；[selfchecks-r1.json](selfchecks-r1.json) 是本次生成检查记录。

前缀捕获成功只能证明后续精度实验的数据来源与原配对范围。独立评估仅包含同一短序列的两组输入，不能证明完整 VGGT 数值精度、SLAM 轨迹质量或实时目标。所有数组、checkpoint、权重导出和 SDK 二进制保留在私有实验目录，不提交到公开仓库。代码与模型权重分别遵循自身许可，仓库的 BSD-2-Clause 不赋予原权重的商业使用权。
