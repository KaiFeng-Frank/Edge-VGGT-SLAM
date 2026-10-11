# 原 frame0/global0 的正常 SDK 对照结果

16 次正常板端调用及保存输出已独立只读核验。原 FP16 实现的完整 SDK 输入输出区间均值为 337.086 ms，PTQ 为 255.640 ms，后者少 81.445 ms（24.162%）。每种实现的均值只含两组 heldout 各 3 次后续测量，共 6 次；每组首次调用另记。两组输入的初始实现顺序反转，四轮 AB/BA 的两个 context 同时存活，无隐藏 warmup。计时包含 FP32 输入提交、同步 run、want_float 输出、独立 FP32 复制和 release，六阶段之和逐项对 monotonic 区间一致；不含加载、校验、误差统计和保存。

| 实现 | 后续完整 SDK IO 均值 | evaluation_00 相对 L2 | evaluation_01 相对 L2 | 固定原 FP32 门槛 |
|---|---:|---:|---:|---|
| 原配对 FP16 | 337.086 ms | 0.00445254 | 0.00497755 | 两组均失败 |
| 原配对 PTQ w8a8 | 255.640 ms | 0.10312154 | 0.10546455 | 两组均失败 |

原始参考始终是同一 heldout 输入的原 Torch FP32 `global0` 最终输出。固定条件为逐元素 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`，没有改门槛、参考或验收含义。16 份保存输出的 file/raw SHA、finiteness、完整误差统计、repeat equality 和调用分母均重算一致。FP16 与 PTQ 均未采用。

补充的原 FP32 CPU ORT heldout 检查整体为 `numerical_rejected`：evaluation_00 有 1 / 894,976 个元素越限，最大 tolerance ratio 为 1.03193（relative L2 为 1.19056e-6）；evaluation_01 零越限并通过。该边界失败明确保留，没有放宽条件或合并为通过；SDK 的参考仍是各组原 Torch 输出，未换成 ORT 输出。实际 ORT 收据 SHA 为 `818ca23f2b1e5986d57d2046dfa734e067faac623a0c0eaf539c21ebbf961f3c`。

FP16 逻辑边界为 type=FLOAT16、AFFINE scale=1/zero_point=0。PTQ 逻辑边界为 INT8：input scale=0.0410780087、zero_point=57；output scale=0.0222544167、zero_point=5。运行 API 始终输入 FP32、`pass_through=0`，输出 `want_float=1`，由 SDK 处理边界转换；转换误差包括在对原参考的误差内。记录边界为 INT8 不能证明全部内部算子使用 INT8 或 NPU。

两组原 FP32 输入仅有 7 / 1 个元素超出 descriptor 的 INT8 网格端点范围 `[-7.59943,2.87546]`，总数各 894,976。这个保存数据的范围观察没有模拟 SDK 转换，也未证明实际 rounding/clipping 行为；不能据此把约 10% 的输出相对 L2 归因于边界截断。边界与内部近似误差仍未分离。

SDK query 返回 FP16/PTQ 的 component DMA 分配为 141,762,560 / 74,178,560 字节，分别对应同一个原首个 frame/global 配对 context。它不代表完整模型的内存、整个应用的峰值，或 8 GB 验收。

正常收据 SHA 为 `05ac1187d9bda3893d43f2d9c834d3b4de51f97d8c455496daa0c5a6f803e591`；完整私有只读审查 SHA 为 `f6d865347e00625b644ad264bc6bb2d33607b4bd08360a456c968ff9dda9e7d6`。可公开摘要见 [public-normal-audit-r1.json](public-normal-audit-r1.json)，不包含数组、权重或 SDK 二进制。

该结果只覆盖原模型的首个 frame/global 配对和两组重复保存输入。它没有测 fresh-observation pose rate、完整 SLAM 的输入到位姿 P95、其余原层和头、轨迹质量、地图频率、30 分钟稳定性或完整 8 GB 运行；单个 component 的缩短不能记作实时目标达成。独立 profiling 在另一进程和新 context 中执行，其 IO 不能混入这里的正常均值。
