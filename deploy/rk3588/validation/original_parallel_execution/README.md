# 原 VGGT 完整模型并行执行（v79）

最新两路回放把新推理的派发间隔设为至少 0.8 s，取可派发时的最新帧，结果完成后立即发布。**32 次新位姿，平均 1.120248 Hz，完整 P95 1.976139 s**；发布间隔 P95 1.031354 s、最大 1.219162 s，超过一秒的间隔由下述未错开版本的 14/31 降至 **2/31**。它改善了时序均匀性，但仍未满足每秒必有一次新位姿。启动 93.697754 s、峰值 RSS 6217.492188 MiB。以下比较来自分别运行的短回放，选帧不同，不能代替质量或长期稳定性验证。

此前未错开启动的两路完整原模型接唯一原 SL4，在 TUM 的 798 个原时间戳输入上完成 **32 次新位姿/地图更新，平均 1.100682 Hz**，完整 input-to-pose **P95 2.364495 s**。包含 EOF 排空的回放为 29.072873 s，启动另计 99.304896 s，峰值进程 RSS 6209.210938 MiB。发布间隔 P95 1.644008 s、最大 1.683418 s，31 个间隔有 14 个超过一秒，因此平均超过 1 Hz 并不满足“每一秒至少一次”，也未达到项目的 20 Hz/100 ms 目标。旧单路回放为 0.899908 Hz / P95 1.366577 s；并发改善吞吐，延迟增加。

队列一格、每 worker 一条未提交任务，按派发顺序进入原 SL4；不存在插值、重复位姿或外加前端。787 帧解码加 11 帧解码前跳过覆盖全部 798 帧，33 帧交付包含 seed，754 帧被新帧覆盖。32 chunks/64 nodes 发布，24 SDK contexts 实际关闭、两线程 join，CPU buffer 全部释放；测试后恢复原 300 MHz。测试板为 16 GB，进程 RSS 不能单独证明物理 8 GB 可用。

同九张真实图像组成八个连续重叠 pair，在 1 GHz 上比较串行与同进程三 worker。保留原 checkpoint、全部层/token、原模型计算和 codec：包含预处理的整网批次墙钟为 10.773273→5.142682 s，离线吞吐为 0.742578→1.555609 pair/s（2.09487 倍）。16 NPZ 的 96 个数组通过 SHA/shape/dtype/finite 检查，40/40 五项预测及 8/8 准备像素与同输入串行输出逐字节一致。初始化和保存数组不计入该批次墙钟；单请求服务均值反而为 1.345907→1.764868 s，不能把吞吐称为位姿延迟。

一份原 Solver 按 pair 顺序消费保存预测，不重复推理：两路线各提交 8 次、16 图节点和 112893 点；poses 完全一致，地图 JSON 一致，32 个地图 chunk 数组字段有限且逐字节一致。这是离线原 SL4/地图重建；另一次三路真实时间戳回放在三个 worker 就绪后的首次推理发生 229376-byte SDK 分配 errno12 和 SIGSEGV；控制器恢复原频率，不宣称崩溃进程的 SDK 清理。两路回放完成，实测结果见上。

离线成功配置使用三个独立原模型/cache/session、36 SDK contexts，安装串行。master 从验证过的原 RKNN filename 初始化；其余 24 个 contexts 使用同文件的 owned CPU buffer、0x20 shared init 和 136-byte InitExtend，buffer 保留至 context 实际销毁。此前三进程第二实例和同进程 filename 子实例均返回 -6；`rknn_dup_context` 首调用另触发 SIGSEGV，其双指针/uint64 ctypes ABI 与固定 header 一致，内部原因未确定。这里不宣称物理共享字节已测明。

36 contexts 实际创建/销毁、三线程 join，cleanup 为空；301 个控制样本均为 1 GHz，最高 69.307°C，最后恢复原 300 MHz/governor/limits。进程峰值 RSS 为 8618.390625 MiB（全线程同一进程，不能累加 worker RSS 或未知重叠的 SDK DMA）；物理 8 GB 未验收。

早期 pair00 独立 context 诊断给出 1.71715 倍组件吞吐。RoPE 原 FP32/FP16 输出与基线逐字节一致，但编译仍 57 行、9 Transpose、11 Reshape，仅 2.53% 计时差，未采纳。这些历史失败和组件数字只保留精简指标/SHA，不复制全部失败源码。

执行源码及哈希见 `source-copy-list.json`，同时保留三路失败回放入口以复现该问题。`mask-shared-buffer-overlay/vggt_slam/` 保持实际执行的四 adapter 与唯一 registry 字节；调用者仍从已验证的部署源码加载其余模块。原权重、数据、图、捕获数组、SDK/header/native 二进制不随仓库分发；代码许可不等同于原权重商业许可。

板端设 `VGGT_RUNTIME_PY` 为生产 Python，`VGGT_REPO` 为已验证源码，`VGGT_DIAG` 为本目录，`VGGT_RUN` 为新目录。需自行准备源码指定 SHA 的原 RKNN 图、真实数据及 NEON 库。以下运行命令必须由已有授权的有界 1 GHz controller 包裹，不直接写系统频率：

```sh
"$VGGT_RUNTIME_PY" "$VGGT_DIAG/full_parallel_shared_board.py" --output-dir "$VGGT_RUN/full" --mask-overlay "$VGGT_DIAG/mask-shared-buffer-overlay" --timeout 165
```

在 launcher 写入上述单条命令，再调用现有控制器；75°C/180 s 有界，finally 恢复频率及原设置：

```sh
"$VGGT_RUNTIME_PY" "$VGGT_REPO/deploy/rk3588/validation/original_clock_test/npu_frequency_test.py" --execute-frequency-test --frequency-hz 1000000000 --temperature-limit-c 75 --timeout-seconds 180 --launcher "$VGGT_RUN/launcher.sh" --output "$VGGT_RUN/control"
```

独立 pair00 CPU-buffer 子实例复现（同样由 controller 包裹）：

```sh
"$VGGT_RUNTIME_PY" "$VGGT_DIAG/probe_original_parallel_lanes_board_r6.py" --output-dir "$VGGT_RUN/pair" --frequency-hz 1000000000
```

核验完整模型保存结果后，可分别重建同一原 SL4 与地图；离线提交时间不计 live latency：

```sh
"$VGGT_RUNTIME_PY" "$VGGT_DIAG/commit_parallel_predictions_board.py" --source-result "$VGGT_RUN/full/result.json" --source-result-sha256 RESULT_SHA --route serial --output-dir "$VGGT_RUN/serial-SL4"
"$VGGT_RUNTIME_PY" "$VGGT_DIAG/commit_parallel_predictions_board.py" --source-result "$VGGT_RUN/full/result.json" --source-result-sha256 RESULT_SHA --route parallel --output-dir "$VGGT_RUN/parallel-SL4"
```

20 Hz 新观测位姿、100 ms 完整 P95、30 分钟、原 Torch FP32/GT 质量和物理 8 GB 均未验收，不能由保存输出相等替代。

两路错开启动的真实时间戳回放（由同一有界 controller 包裹；新输出目录；当前仅固定的 798 帧数据集）：

```sh
"$VGGT_RUNTIME_PY" "$VGGT_DIAG/replay_parallel_staggered_board.py" --output-dir "$VGGT_RUN/replay" --mask-overlay "$VGGT_DIAG/mask-shared-buffer-overlay" --timeout 170
```

`replay_parallel_shared_board.py` 是发生分配失败的三路入口，不建议作为运行入口。完整数值与失败记录见 [v79 结果](../../../../results/original_parallel_execution_v79.json)。

SDK ABI 与共享 flag 的声明参照固定 [RKNN v2.3.2 官方头文件](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h)。运行结果由上面的板端实验验证；头文件本身不证明驱动内部的内存共享量。

未错开启动的对照入口为 `replay_parallel_two_lanes_board.py`；[派发策略差异与源哈希](staggered-source-delta.json) 单独记录。
