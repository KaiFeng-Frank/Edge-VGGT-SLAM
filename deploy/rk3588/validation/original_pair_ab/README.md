# 原 VGGT frame0 → global0 同进程 A/B 探针

本目录是新的板端测试源码，尚未执行实际 A/B。它只运行原 VGGT 的完整 frame0、global0，比较相同原捕获输入的两种执行路径；原权重、位置、参数和计算没有替换。所有实际 SDK 调用由主代理在 RK3588S 串行执行。

- `separate`：实际 NPU frame0 输出复制到自有 FP32 内存，在原 token 顺序下 view 为 `[1,874,1024]`，直接输入 NPU global0。没有 CPU 参考重置。每次保存完整 frame0 和最终 global0 输出，并绑定桥接数据 SHA。
- `fused`：相同原 frame0 输入进入完整配对图，保存完整最终输出。

三个 contexts 在同一进程一次初始化，固定 core mask 7、CPU 6/7。所有 contexts 初始化完成后，明确记录进程和系统内存、三个 context 的 SDK 内存统计及温度。四轮 `AB / BA / AB / BA`，每条路径一次首次加三次后续，共 **12 次同步 SDK 调用**，没有未报告预热。每条路径前后记录内存、温度和只读 CPU 频率；退出恢复进程 affinity 和环境。

## 来源及拒绝门槛

`v48_ab_contracts.py` 硬绑定旧 frame/global 的 export、已通过的原板端 ORT、非量化 compile 和 binary SHA。两份 export 的原 18 个参数逐项与配对 export 的原参数绑定一致；输入、参考及位置 raw SHA 保持原 frame→global 顺序。新 fused 必须绑定新配对编译收据和其实际已通过的板端 ORT 收据。

SDK 使用冻结 v12 的 C ABI 定义及 typed bindings，另按每个 context 的角色独立检验实际逻辑 shape、名称、索引、浮点类型、恒等量化描述符、format 3 和字节数。未知描述符在任何调用前拒绝。SDK 输出在 release 前复制；get 返回部分分配、frame identity 错误、输出 byte count 错误均需释放。退出销毁全部已分配 contexts，重新检查所有输入、源码、模型、SDK 的 SHA。

固定原 FP32 比较阈值 `rtol=1e-4, atol=1e-5`。本 A/B 允许保留已知 NPU 精度拒绝结果，参数 `--diagnostic-allow-FP32-rejected` 必须显式给出。`candidate_accepted` 始终 false；组件误差和时间不证明整网质量或实时。Checkpoint 来源继承 SHA 绑定的原导出收据，本程序不再次读取 5 GB checkpoint。

## 时间范围

每个 `sdk_io_sum_seconds` 是相应实际 SDK IO 之和，包含 `inputs_set → synchronous run → outputs_get → owned FP32 copy → release`。它排除 init、queries、比较、保存数组和 hash。

`route_total_seconds` 是 **instrumented route timing**：从该路径开始到完整最终输出及必要桥接检查完成，包含 frame→global view、有限性检查、桥接数据前后 SHA 和 Python 调度。桥接的诊断 SHA 开销独立列于 `bridge.pre_global_view_finite_hash_seconds` 与 `bridge.post_global_hash_verify_seconds`，没有从该时间扣除。它不是纯 SDK IO；不能把两种时间混用。所有 source hashes、数组保存、结果比较、温度及内存快照均在路径计时外。

固定输入的四次路径重复没有新的相机观测。未覆盖完整 VGGT、位姿 head、稠密 head、SL4、P95、30 分钟或 8 GB 验收。三 contexts 的额外内存及未控制的散热状态也必须随结果报告。

## 主代理实际执行

先读取 `source-freeze-r1.json`，按 SHA 将这两个源文件传板端，例如 `experiments/vggt-v48-ab-source/`；原配对 contracts 独立来自 v47 冻结源码。下面新 ORT/compile/model SHA 已由主代理实际运行取得。旧收据及二进制已经板端存在。输出目录必须全新。

```bash
cd /mnt/vggt-data/vggt-slam2
. src/deploy/rk3588/env.sh
AB=experiments/vggt-v48-ab-source
# 将 AB_CONTRACTS_SHA 设置为 source-freeze-r1.json 中 v48_ab_contracts.py 的实际 SHA。
python -B "$AB/run-v48-frame-global-ab-board.py" \
  --ab-contracts "$AB/v48_ab_contracts.py" --ab-contracts-sha256 "$AB_CONTRACTS_SHA" \
  --pair-contracts experiments/vggt-v48-pair-source/v47_pair_contracts.py \
  --pair-dir experiments/vggt-v48-pair \
  --runtime-helper src/deploy/rk3588/validation/run-v12-rknn-block-board.py \
  --sdk-header src/deploy/rk3588/reference/rknn_api_v2.3.2.h \
  --frame0-export-receipt experiments/vggt-v19-frame336-export-r3/export.json \
  --frame0-ort-receipt experiments/vggt-v19-frame336-ort-r3/check.json \
  --frame0-compile-receipt experiments/vggt-v19-frame336-compile-r1/compile.json \
  --frame0-model experiments/vggt-v19-frame336-compile-r1/block.rknn \
  --global0-export-receipt experiments/vggt-v17-global336-export-r3/export.json \
  --global0-ort-receipt experiments/vggt-v17-global336-ort-r3/check.json \
  --global0-compile-receipt experiments/vggt-v17-global336-compile-r3/compile.json \
  --global0-model experiments/vggt-v17-global336-compile-r3/block.rknn \
  --pair-compile-receipt experiments/vggt-v48-pair-compile-r1/compile.json \
  --pair-compile-sha256 5084e00143c4ab5fce94ceae81247c11939ddb7783d80d1b937bffcbdc0bcfb9 \
  --pair-ort-receipt experiments/vggt-v48-pair-ort-r2/check.json \
  --pair-ort-sha256 f93e1939be9628b7e28df8f91cf06d28002f0b3871a0d0a46adca1780ffa5fc0 \
  --pair-model experiments/vggt-v48-pair-compile-r1/compiler/block.rknn \
  --pair-model-sha256 5ba3440634a51ae56fdc46eb1c07bf5847a30640a7c2603d322b44af585d037b \
  --diagnostic-allow-FP32-rejected \
  --output-dir experiments/vggt-v48-pair-ab-r1
```

`--pair-contracts` 的 board 目录按主代理实际已部署路径调整，文件 SHA 必须是冻结值 `4ec6c0e827ab9669c942b589d2c504f0c9de306f76f4f4be0799af14ae322cd7`。读取 JSON `run.json` 的最终 status 和 cleanup，不仅看 ADB shell 返回码。任何失败保留原目录，修复后用新源码 freeze 和新输出目录。

## 主机验证范围

小型测试仅使用生成数组和 fake SDK API：实际链桥接、不使用 CPU 重置、输出释放生命周期、部分分配失败、SDK shape/typed descriptor、输入突变检测、固定平衡顺序、独占输出文件及保留原精度拒绝。主机不加载实际 SDK，不运行模型/ORT/编译。
