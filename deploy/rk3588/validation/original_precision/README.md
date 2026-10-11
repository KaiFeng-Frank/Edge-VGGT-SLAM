# 原 VGGT 的板端 FP16 / PTQ 对照

本次使用原 VGGT 的完整 `frame0 → global0` 图，在 RK3588S 上完成真实数据捕获、量化、保留输入 ORT 检查和两组独立 SDK 运行。**两种 NPU 数值实现都未通过原 FP32 门槛，量化候选未采用。** 没有引入模型、删层或修改正式 SLAM 执行路径。源码冻结目录中的准备阶段说明保留原文，实际完成状态以本页与[结果记录](../../../../results/original_precision_v49.json)为准。

| 对照项 | 非量化 FP16 | 原图 PTQ w8a8 |
| --- | --- | --- |
| 普通 SDK 完整 IO 均值，各 6 次复测 | 337.086 ms | 255.640 ms |
| 第一个保留输入相对 L2 误差 | 0.4453% | 10.3122% |
| 第二个保留输入相对 L2 误差 | 0.4978% | 10.5465% |
| 固定原 FP32 元素门槛 | 失败 | 失败 |
| 编译产物大小 | 55,280,948 B | 29,931,726 B |

本次局部 IO 均值下降 24.16%，产物大小下降 45.86%。这不是完整模型加速、每秒新观测位姿数、ATE/RPE、P95 时延或 8 GB 验收。两份保留输入属于同一 TUM 短序列，不代表场景泛化。

## 数据与执行顺序

1. [隔离 ARM64 工具链](toolchain/README.md)：板端新 venv 使用 Torch 2.2.0 / RKNN 2.3.2 等固定依赖，生产 Torch 2.3.1 保持原样。
2. [真实原模型捕获](capture/README.md)：推理前固定 8 组校准和 2 组保留输入，PNG 索引/时间戳/文件 SHA 互斥。原 24 层 DINO 全部执行；评测还执行原 frame0/global0，保存同次原 Torch FP32 参考。
3. [板端 PTQ](quant/README.md)：校准仅使用八份完整原输入，配置 `w8a8/normal/channel`、`auto_hybrid=False`。完成 build 耗时 71.036 s。
4. [新保留输入的板端 CPU ORT 复核](quant/HELDOUT_ORT.md)：第一组 894,976 个元素中 1 个超出固定 `rtol=1e-4, atol=1e-5`，第二组通过；整体保留 `numerical_rejected`。旧输入通过的结果不能推广为新输入全部通过。
5. [板端 SDK 对照](runtime/README.md)：两个 contexts 同进程，2 输入 × 4 轮 AB/BA，共 16 次实际调用，没有隐藏 warmup。直接比较原 Torch FP32 参考，使用显式失败诊断选项，候选始终不采用。
6. 独立新进程、新 contexts 开启 profile，另做 16 次调用。其算子时钟与普通 IO 分开记录；不混算速度收益。

完整 IO 包含 inputs_set、同步 run/get、输出检查与自有复制、release；不含初始化、哈希、数值比较、文件保存、预处理或 SLAM。六次复测均值是两个输入各自首调用之后的三次之和；第二个输入的首调用不称为 context 冷启动。FP32 API 输入和 `want_float=1` 输出由 SDK 转换，整数逻辑描述符不等于手工修改模型输入。

## 复现和审查

新 clone 需先准备各自许可下的原 checkpoint、固定原模型源码、TUM 数据、SDK、冻结 v48 原配对导出与非量化编译产物。包含权重的文件不随仓库发布。路径和 SHA 绑定本次记录；重跑时使用新输出目录，重新生成数据收据并显式绑定其 SHA，不覆盖旧结果。

[root-launchers-freeze-r1.json](root-launchers-freeze-r1.json) 保存主任务实际使用的计划和 runtime launcher 源码 SHA；各子目录的原 source freeze 保持原样。`run-precision-board.sh normal|profile` 是已完成实验的精确命令记录，固定 r1 路径会拒绝覆盖已有目录；它不是自动安装脚本。板端 ADB 返回码不能代替 JSON 状态检查。

保存数据审查仅重新计算已保存数组的哈希、FP64 误差、SDK 计时范围和描述符，不在主机执行模型。原始 NPY、权重图、SDK 和日志只留在私有实验存储；公开审查记录给出相应 SHA 和结果。原仓库代码、VGGT 源码、原权重和 SDK 各自遵循独立许可，BSD-2-Clause 不授予权重商业使用权。

目标仍是原模型的新观测实时执行。严格元素门槛失败不能被余弦相似度替代，张量相对 L2 误差也不能解释为轨迹误差。

公开目录布局的[140 项小型检查](audit/public-portability-checks-r1.json)全部通过（capture 15、PTQ 38、heldout ORT 6、SDK runtime 81），这些检查使用生成数据或 fake 服务，不执行实际模型。普通/profile 的 32 份实际保存输出经独立数值审查，16 对相应输出逐 bit 相同；精度拒绝结论保持不变。各阶段精简结果和可移植保存数据审查入口见[audit 目录](audit/README.md)。
