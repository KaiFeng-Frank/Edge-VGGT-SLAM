# M16 完整 Update 状态的 RKNN 诊断

这组独立研究入口使用原 DPVO Update 参数，测量三个冻结状态
（observation 34 / 127 / 191，E704 / E800 / E800）的**完整边集**。
同 degree 的独立 source 成批执行；每个 source 的 target×16 patch、
c1→c2 邻居顺序及 SoftAgg 依赖完整保留，再恢复原 E 顺序。
没有 padding、删边、CPU fallback、BA、循环反馈或在线 pose 提交。
v39 完整低算量 preset 的轨迹精度拒绝保留；组件结果不能反转该结论。

`v40_m16_grouped_rknn.py` 在私有模块实例中绑定新 M16 合同，继承冻结
v37 SDK 的 pack、同步执行、释放前 owned-copy、有界缓存和完整边序
scatter。新旧合同文件保持独立；旧 runtime 的原 v37 合同需保持相邻。
`run-v40-m16-grouped-board.py` 只在 RK3588 板端打开 SDK，没有 Torch/ORT
前向。正常和 profile 使用独立进程/context；SDK profile 的算子 placement
必须由实际输出判断。

## 板端入口

每个实际 ONNX 形状先由 `compile-v40-m16-grouped-update.py` 单独编译，传入
`--manifest`、固定 `--manifest-sha256`、`--model-key`、`--contracts`、
`--parent-contracts`、`--parent-compiler` 和全新的 `--output-dir`。
父合同与父 compiler 位于相邻 `../dpvo_grouped_update`；配置固定为
RK3588、float16、optimization_level=3、`build(do_quantization=False)`。
该步骤允许在主机执行，不校准、不调用 Torch/ORT/RKNN inference。
目录索引按真实编译收据绑定每张模型 SHA，完整样例保存在
[实际证据](../../../../results/dpvo_m16_grouped_v40.json) 的 `compile_index.record`。

以下路径参数均为已按 SHA 验证的板端产物，权重/RKNN 文件和 SDK 不随源码
分发。需要 Python、NumPy、原 ABI helper、Rockchip 2.3.2 runtime 与官方
header、冻结 native-log helper、v37 runtime/contract 和 source-frame adapter。

```bash
python -B run-v40-m16-grouped-board.py \
  --manifest "$M16_EXPORT/manifest.json" \
  --compile-index "$M16_COMPILED/index.json" \
  --compile-index-sha256 "$INDEX_SHA256" \
  --capture-dir "$M16_CAPTURE" \
  --contracts "$M16_SOURCES/v40_m16_grouped_contracts.py" \
  --contracts-sha256 7be6bf750dabb8a3d88464fd1502bde3bf7690596a39791710cf41a9f8cb62fe \
  --compiler "$M16_SOURCES/compile-v40-m16-grouped-update.py" \
  --compiler-sha256 3ccd9002247bf3ff15e87bf04db2e3462db782c6d5a637f50b98094bcd584294 \
  --runtime-helper "$M16_SOURCES/v40_m16_grouped_rknn.py" \
  --runtime-helper-sha256 d4cdd1f9f21b04e44b6369c8ad7aea1fe4fbd43855b13993c6fdf4550f7bf0d7 \
  --parent-runtime "$V37_SOURCES/v37_grouped_rknn.py" \
  --parent-contracts "$V37_SOURCES/v37_grouped_contracts.py" \
  --adapter "$V37_EXPORT_SOURCES/source_frame_batches.py" \
  --abi-helper "$ABI_HELPER" --header "$RKNN_HEADER" \
  --log-helper "$FROZEN_NATIVE_LOG_HELPER" --library "$RKNN_RUNTIME_LIBRARY" \
  --max-contexts 8 --diagnostic-allow-fp32-rejected \
  --output-dir "$NEW_BOARD_OUTPUT"
```

独立 profile 加 `--collect-perf-detail`，使用另一个全新输出目录。
普通每状态首调用及两次 warm，共 9 完整调用 / 36 SDK 组调用；profile
每状态一次，共 3 完整调用 / 12 SDK 组调用。状态间共用有界 cache，因此
每状态标为 cold 的首调用并不意味着其中所有模型都重新初始化。

完整调用计时包括只读 snapshot、校验、pack、缓存查找/必要初始化/驱逐、
SDK set/run/get/owned copy/release、日志处理、domain 检查和原边序 scatter。
参考误差、数组 hash 和 NPZ 保存计时另列；嵌套 group/cache/init 时间不能
相加当作总时间。常驻 model host-buffer 上限 256 MiB、DMA 上限 512 MiB；
实际初始化期间的 transient DMA 单列，该限制不等于进程 RSS 上限。
`rtol=1e-4, atol=1e-5` 对保存的原 Torch、grouped Torch、ORT 分别比较。
FP32 拒绝如实记录为组件拒绝诊断，不会提交候选；nonfinite、非法 weight、
unknown shape 或 SDK 错误 fail closed，保存可取得的 owned raw 输出。

## 已保存的实际板端结果

普通与独立 profile 已各完成；原 FP32 gate 均为 false，有限值与 weight
值域合法，cleanup 为空。普通两次 warm 完整调用均值依次为
**61.294 / 68.482 / 71.344 ms**，首调用依次为
**286.337 / 121.829 / 256.407 ms**。这是三个保存状态的完整 Update 组件
时间，未包含 CNN、相关性、BA、新观测采集或 pose 发布，不是在线 20 Hz
或完整 preset 接受证据；原门槛没有放宽。

本次 8 个常驻 context 的 model host-buffer 合计 59,357,808 B，SDK DMA
合计 78,249,984 B；正常运行峰值 RSS 221.699 MiB。关闭后常驻计数、当前
model host-buffer 与当前 DMA 均为 0。这些是本次实测分配，不是任意输入
或其他模型的内存保证；实际 profile 算子与保存数组独审另列。

八形状实际 profile 都包含 48 个内部 NPU、3 个内部 CPU（Exp×2、Min）
和 6 个 I/O 行。保存数组独审 2,836 项通过，重读十二份完整输出，核对
逆 scatter 的各组 SHA、普通重复/profile bitwise 和原 FP32 拒绝。
`audit-v40-m16-grouped.py --help` 列出全部保存目录和固定哈希参数；它只
读取保存数组/metadata，复用相邻 `dpvo_grouped_update` 的冻结 NumPy auditor。

## 8 图不等于完整在线 shape 覆盖

本次 catalogue 是 B2D4 / B2D5 / B2D6 / B2D7 / B3D6 / B1D5 / B5D6 /
B1D7（全部 M16）。同一 v39 保存轨迹的 169 个完整在线 Update 调用出现
13 种 shape，共 676 组；还缺 B3D7 / B3D4 / B3D5 / B1D4 / B4D6，
这些统计排除 EOF 和 degree-one motion probe。

冻结 `source_frame_batches.build_degree_plan(..., patches_per_frame=16,
max_sources_per_batch=2)` 已支持仅沿独立 source 轴拆 batch；D×M 不拆。
该保存轨迹将变为 870 组，所需 B1/B2 中仍缺 B1D4 / B1D6，D1 motion probe
仍需独立明确路线。这是静态复用建议，**本入口没有执行该拆分**，不减少
原参数 MAC；更多 SDK 调用、实际 shape 与数值需另测。预测变化也可能改变
图的 shape，不能凭原 CPU 轨迹目录宣称任意在线输入支持。

18 项小型 fake-SDK 测试覆盖 M16 native CF/NHWC、owned copy、失败清理、
不翻转数值拒绝、有界 catalogue、原边序恢复和迟到身份拒绝。测试按相邻
公共目录查找依赖，可用 `V40_PARENT_SDK` / `V40_M16_CONTRACTS` /
`V40_ABI_HELPER` 显式指定；不会下载或执行实际模型/SDK。

DPVO 派生几何/神经调用约定按相邻 MIT [LICENSE](LICENSE) 归属 Princeton
University；模型权重和厂商 SDK 的许可独立，源码许可证不授予权重商业权。
