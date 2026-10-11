# 原 pair hybrid step2 的单次 r2 修正

实际 r1 在调用 step2 时因 `Invalid operands name 'input'` 拒绝，未到 export、没有模型输出。错误收据 SHA `7f4f1ee288e4314cd4bfea979bc8eb564e9b3b0f0d036665dd4dd55efe8c5a44`；release、源/输入和环境恢复均通过。r1 源、配置、step1 和失败证据原样保留。

纯 protobuf 读取已经保存的原 SDK `.model`：62 nodes、56 initializers、66 node outputs，其中公开输出为 `output`。与 95 项 calibration catalog 的静态交集给出 65 个内部 node outputs。另 30 项是 24 个 initializer 名、2 个公开 IO 名和 4 个未出现在保存图中的 SDPA 估计名。校准统计 key 存在，不能单独证明它可以用作 step2 selector。

root 选择一个保守修正：原四个 fc1/GELU 内部输出请求 INT8，其余 61 个实际内部输出请求 FP16。不请求 initializer、公开 IO 或 absent SDPA 名，所有原 `quantize_parameters` 条目和其 33,709 字节后缀保持不变。SDK 是否接受这 65 个 selector 要由这次实际编译判断；不逐个名字试探，不扩大 INT8 范围。外部 IO、initializer 与 attention 隐含内部算术的精度仍须实际测量，不能按请求猜测。

新增源为 `compile-v50-hybrid-step2-r2-board.py` 和 `v50_hybrid_step2_r2_contracts.py`，复用被冻结的 r1 安全解析/工件合同和 API helper。receipt schema 保持 `v50_original_VGGT_pair_board_hybrid_step2_compile`，另有 `revision=r2_actual_internal_node_output_selectors`、失败 r1 与静态 selector 证明 SHA。成功仍为 `compiled_not_runtime_validated`，输出仍 `pair.rknn`，候选保持不采用。

私有策略文件放到 board `experiments/vggt-v50-hybrid-policy-r2`；新源放到既有 source 目录。主调度核对新的 source freeze 后用全新目录运行：

```sh
bash /mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-source/run-v50-hybrid-step2-r2-board.sh \
  /mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-source \
  /mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-step2-r2 \
  /dev/shm/vggt-v50-hybrid-step2-r2
```
