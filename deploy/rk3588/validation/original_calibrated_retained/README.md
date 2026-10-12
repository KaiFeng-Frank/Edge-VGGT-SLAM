# 原 VGGT 四组聚合器 INT8 整网验证（v80）

RK3588S 的 NPU 设为 1 GHz，原 VGGT、原 CameraHead/DPT/codec/SL4 完整运行。仅将原聚合器的四组保留图改为 W8A8 编译，保留全部 48 层、token、原 ONNX 节点/参数以及 global20 出口。8 对原 FP32 实测激活用于校准；DINO 和输出头精度保持当前基线。没有其他模型或外加位姿前端，默认生产入口没有改变，闭环仍禁用。

| 同一批 16 对图像 | 当前 FP16 | 聚合器 INT8 |
|---|---:|---:|
| 完整处理均值 | 1.211219 s | 1.043062 s |
| 暖态后 15 对均值 | 1.190188 s | 1.004278 s |
| 离线完整处理 P95 | 1.299152 s | 1.191086 s |
| 首对处理 | 1.526698 s | 1.624818 s |
| 峰值进程 RSS | 3925.996 MiB | 3336.527 MiB |

均值耗时减少 **13.88%**，峰 RSS 减少 **589.47 MiB**。两路均完成原 SL4；保存数组在计时外。四个 RKNN 文件由 1203.07 降为 633.06 MiB（47.38%），文件大小不等于内存占用。FP16 新基线与先前同输入基线的 160 个预测/聚合器数组及 poses 完全一致。

独立的 798 帧原时间戳回放完成 **30 次新位姿/地图更新 / 27.912746 s = 1.074778 Hz**，包含解码、排队、原模型、SL4 和发布的 P95 为 **1.157188 s**。29 个发布间隔的 P95 为 0.946975 s、最大 1.113612 s，1 个超过一秒；从源启动到首个新位姿为 1.297600 s。启动另计 63.064901 s。不能称为每秒必有一次更新，也未达到 20 Hz / 100 ms。

源队列仅一格，797 帧实际解码、1 帧解码前跳过；31 次交付含 seed，766 次旧帧被新帧覆盖。输出 30 地图 chunks、60 SL4 节点、423339 点和 31 条 pose；峰 RSS 3340.602 MiB。12 个 SDK contexts 全部关闭、资源清理无错误。回放控制采样全为 1 GHz，最高 68.384°C，结束恢复 300 MHz/userspace 与原频率上下限。地图单位仍为 model units。物理相机、30 分钟和物理 8 GB 未验收。

在另一个与校准图像不重叠的 17 图像序列上，用正尺度 Sim3 对齐 GT；RPE 是相邻匹配位姿，平均间隔约 1.594 s，并非一秒 RPE：

| GT RMSE | 原 CPU FP32 | 当前 FP16 | 聚合器 INT8 |
|---|---:|---:|---:|
| ATE，m | 0.177191 | 0.174725 | 0.173830 |
| 平移 RPE，m | 0.319953 | 0.314895 | 0.311883 |
| 旋转 RPE，度 | 10.765080 | 10.859345 | 10.715369 |

这三项 INT8 RMSE 分别比 CPU 低 1.90%、2.52%、0.46%，只满足该短序列的 RMSE 相对门槛。最差旋转段误差达 **24.34°**，当前 FP16 为 19.30°；整体 RMSE 相近不等于每段都未退化。与保存的原 CPU 输出相比，深度/置信度全元素相对 L2 为 **5.26% / 42.91%**，不是传感器深度 GT 误差或置信度校准分数。80 个预测数组均有限，但都不满足 `rtol=1e-4, atol=1e-5` 的 FP32 等价门槛；等价检查与 ATE/RPE 门槛分别报告。稠密地图质量仍未验收。真实回放选帧与这 17 张图不同，尚无该选帧的匹配 CPU 精度对照。

编译曾两次在 CPU 75.769°C 时被原控制器停止（group1/2），均保留原记录，不声明被终止进程完成了 SDK 清理。group1 改用进程内 CPU 4/6 后完成；group2/3 则在 70°C 暂停编译进程组、降至 63°C 恢复，34/46 次暂停后成功，最高 71.153/70.230°C。该调度只用于生成图，不用于推理测速；所有编译均保持原 NPU 300 MHz，不改系统 CPU/GPU/DDR 设置。带 `_r2` 的编译控制器补上 TERM/HUP 清理和最终观察失败记录。

## 复现

这是参考板上固定原始工件的诊断入口。先准备原 checkpoint、v56 的 8 对 FP32 校准、v67 四组原 ONNX/FP16 manifest，以及既有 DINO、CameraHead、DPT、NEON 工件；代码、原权重和依赖的许可分别遵守。权重、数组、权重图、SDK/header/本机二进制均不在仓库中分发。

所有实际导出、校准、编译和推理在板端执行。`compile_retained_int8_board_r1.py` 用于 group0，`_r2.py` 用于 group1..3，仅编译 CPU 亲和不同。各组输出使用同一根目录下的 `compile0`..`compile3`；8 条校准身份、未改原图和编译配置会逐项核验。

```sh
"$VGGT_COMPILER_PY" "$VGGT_DIAG/compile_retained_int8_board_r2.py" --group-index 1 --output-dir "$VGGT_INT8/compile1"
"$VGGT_RUNTIME_PY" "$VGGT_DIAG/assemble_retained_int8_manifest.py" --output-dir "$VGGT_INT8" --compiler-source "$VGGT_DIAG/compile_retained_int8_board_r1.py" --compiler-r2-source "$VGGT_DIAG/compile_retained_int8_board_r2.py"
```

先设 `VGGT_DIAG` 为本目录，`VGGT_INT8` 为私有工件目录，并使用板端独立编译环境；首条命令只是单组示例，需要编译四组并由编译专用温控控制器包裹。映射只选择四个新 RKNN 文件，运行时仍继承原保留聚合器，核验原参数和消费者；SDK 使用 FP32 转换边界并拥有输出副本，这不表示内部 FP32。

`run_int8_1ghz.sh` 与 `run_replay_int8_1ghz.sh` 保留本次实际启动参数和固定目录/SHA；把相应源放入 launcher 指向的私有源目录，或在新实验中显式设置新源/工件身份。前者运行未改的完整比较脚本，后者运行未改的原时间戳回放脚本。通过已有授权的有界控制器运行 launcher，不能直接常驻锁频：

```sh
"$VGGT_RUNTIME_PY" "$VGGT_REPO/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py" --execute-frequency-test --frequency-hz 1000000000 --temperature-limit-c 75 --timeout-seconds 180 --launcher "$VGGT_RUN/launcher.sh" --output "$VGGT_RUN/control"
```

主机可以运行生成数组/假 SDK 的 6 项测试，以及仅保存数据的审计；不运行真实模型：

```sh
python test_rknn_calibrated_retained.py
python audit_calibrated_replay_saved.py --replay-dir "$VGGT_REPLAY_SAVED" --control "$VGGT_REPLAY_CONTROL" --wrapper "$VGGT_REPLAY_WRAPPER" --output "$VGGT_RUN/replay-audit.json"
python audit_complete_quantized_saved.py --public "$VGGT_REPO" --cpu "$VGGT_CPU_SAVED" --fp16 "$VGGT_FP16_SAVED" --int8 "$VGGT_INT8_SAVED" --groundtruth "$VGGT_GT" --output "$VGGT_RUN/quality.json"
```

[完整数值、失败与边界](../../../../results/original_calibrated_retained_v80.json)，[实际源文件 SHA](source-copy-list.json)、[回放保存结果核验](replay-saved-audit.json)。这条路线提供已测的内存和整网耗时收益，仍不是实时目标的完成版本。
