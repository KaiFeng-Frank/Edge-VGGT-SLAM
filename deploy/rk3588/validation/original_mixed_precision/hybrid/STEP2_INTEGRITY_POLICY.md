# 原 pair hybrid step2 完整性策略

Step2 固定绑定已经实际运行的 step1 receipt SHA `7c2a101be488dded5086c6b418480f459bc14a62f9269991c1d88a6892ec8133`，同一原 graph/checkpoint、八份 calibration 与冻结 SDK/source 链。入口仍只在 RK3588S 的隔离 CP311/RKNN2.3.2 环境执行，不调用普通 build 或 runtime。

原 `.model/.data/.quantization.cfg` 不改动，重新核对 receipt 中大小/SHA。候选配置保存在独立的 `pair.mlp-focused.quantization.cfg`。安全 YAML 解析器拒绝重复键、别名/锚点和不安全类型；只替换顶层 `custom_quantize_layers` 的文本块，不重新序列化其余文件。因此 SDK 生成的 `quantize_parameters` ranges/scales/zero points 逐字节保留，另用冻结 layout helper 检查解析后也完全相同。收据记录保留前后缀的大小/SHA、原/编辑配置 SHA、catalog SHA 和请求列表。

95 个实际 catalog keys 是精度请求的唯一选择空间，原图 node 名、abstract scope 和 runtime operator label 不能直接作选择器。当前 fc2 在实际 compiler 中与 LayerScale/残差融合；相应 alias 必须明确记录融合范围，不能把配置请求说成 exact MLP-only。原 input/output、attention、norm/RoPE 和其余未选择 keys 请求 float16，最终实际精度另查。

本次唯一配置明确选择 `frame0/mlp.fc1.matmul_mm_tp_sw`、`frame0/gelu_tp-rs`、`global0/mlp.fc1.matmul_mm_tp_sw`、`global0/gelu_tp-rs` 请求 INT8；其他 91 keys 全部请求 FP16，包括整个 fc2/LayerScale/残差融合组。catalog 没有 fc1/fc2/qkv/proj 线性参数的独立 selectors；norm 参数及 RoPE 常量则在浮点请求中。四个输出请求不证明最终 fc1 权重或算术精度。策略与配置 SHA 是 `6eb6c909e2aeac6bdaafce680b4f39c33447f1dfbbc2cd3714be43055c9d9ad8` / `18593e0cf4d05354cbb71f7bc48075542e9083d6fb060e5041d340f8d329dab9`。

冻结的诊断策略由 root 审查具体请求列表、逐 alias 范围和配置字节身份。`precision_policy_approved=True` 只表示这份诊断配置经过工程审查；`candidate_accepted=False`、固定原 FP32 门槛和历史失败保持。操作只有 `hybrid_quantization_step2(model_input,data_input,model_quantization_cfg)`、`export_rknn`、`release`。[官方 v2.3.2 step2 API 流程](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknn-toolkit2/examples/functions/hybrid_quant/step2.py) 支持该三文件接口；示例后续运行 API 不进入本阶段。

成功 receipt schema 为 `v50_original_VGGT_pair_board_hybrid_step2_compile`，状态 `compiled_not_runtime_validated`。输出命名 `pair.rknn`，字段 `output={file,bytes,sha256}`。完整记录源/策略/step1/capture/artifacts、原参数身份、toolchain、board、API 返回值和耗时、源输入前后哈希、release、生成文件身份和环境/cwd恢复。任何错误或清理失配都拒绝成功状态。root runtime 可复用 `check_step2_compile(row,expected_script_sha,expected_contracts_sha,expected_policy_sha,expected_step1_sha,expected_model_sha)`；该纯函数不导入 SDK 或解析权重，只验证编译收据，实际模型/文件 SHA 由 runtime 外层绑定。

配置与 step1 工件含模型相关私有数据，不公开。公开记录只能给范围、源、SHA、实际边界/精度验证结果和限制。

本阶段新增 `compile-v50-hybrid-step2-board.py`；冻结的 step1 README 保留其冻结时的阶段状态。root 核对新的 step2 source freeze，原样上传新源至 `experiments/vggt-v50-hybrid-source`，私有策略文件放至 `experiments/vggt-v50-hybrid-policy-r1`，冻结 layout helper 放至 `experiments/vggt-v50-layout-source`。然后使用全新目录运行：

```sh
bash /mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-source/run-v50-hybrid-step2-board.sh \
  /mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-source \
  /mnt/vggt-data/vggt-slam2/experiments/vggt-v50-hybrid-step2-r1 \
  /dev/shm/vggt-v50-hybrid-step2-r1
```
