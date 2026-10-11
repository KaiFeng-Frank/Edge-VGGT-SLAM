# 原 VGGT 首配对的 hybrid 精度分配计划

只处理冻结原 `frame0→global0` 图，ONNX SHA `2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc`、36 个原参数不变。当前目标是把精度请求集中在原 MLP，attention/norm/RoPE、外部输入输出及非 MLP 路径请求 float16；实际融合范围须检查，不能先宣称“只有四个原线性算子采用整数”。本计划不替换模型、不删层、不采用候选；固定原 FP32 门槛和 v49 的数值拒绝保留。

## 经过核对的 API

实际板端 SDK 的纯 Python wrapper SHA 是 `63e4506fb851214e79f933f9de0d27dab04ecf93f6f46959c07c5787433fd05e`，与 v49 实际 compile receipt 的 module pin 一致。静态读取该源；没有在主机导入 SDK。wrapper 的签名为：

```python
hybrid_quantization_step1(dataset='dataset.txt', rknn_batch_size=None,
                          proposal=False, proposal_dataset_size=1, custom_hybrid=None)
hybrid_quantization_step2(model_input, data_input, model_quantization_cfg)
```

step1 接校准 dataset 并生成 `.model/.data/.quantization.cfg`；官方流程要求修改生成的精度配置后再 step2。step2 接这三项输入后用 `export_rknn` 导出。两步流程的出处是 [官方 v2.3.2 step1](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknn-toolkit2/examples/functions/hybrid_quant/step1.py)、[官方 v2.3.2 step2](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknn-toolkit2/examples/functions/hybrid_quant/step2.py)。这里只复述通用 API，不复制示例的模型、数据或推理部分。

step1 没有 `do_quantization` 或 `auto_hybrid` 参数，不能把普通 build 的参数传进去。本轮明确 `proposal=False,rknn_batch_size=None,proposal_dataset_size=1,custom_hybrid=None`。wrapper 仅说明 `custom_hybrid` 接受输入/输出名称对的列表，没有充分说明边界精度传播和融合；不凭该参数猜测原图区域。

## 配置格式与审批点

官方 step1 说明配置使用 `custom_quantize_layers`，键是层的 **output tensor name**，值支持 `int8/int16/float16`。它不是 ONNX node name/op_type，也不是运行时 profile 的 operator label。下面是语法模板，不能作为实际可运行配置：

```yaml
custom_quantize_layers:
  'input': float16
  'output': float16
  '<actual-generated-attention-output-tensor>': float16
  '<actual-approved-MLP-output-tensor>': int8
```

先由主任务在 RK3588S 运行 step1，保存生成配置的原始字节与 SHA；实际 aliases 和所有配置 sections 由该文件决定。后续只在副本修改 `custom_quantize_layers`，保留 SDK 产生的校准参数/ranges/weights 元数据，拒绝重复键、未知 tensor、未审批别名或额外 sections 修改。原配置与 `.model/.data` 都含模型相关私有信息，不公开。

拟定 step2 审批材料绑定：原 graph/checkpoint/capture SHA、step1 receipt SHA、生成的三个 artifact SHA、原配置 SHA、审批配置 SHA、完整 before/after precision map、alias 对原 ONNX 边界的来源与未知项、requested-only precision 范围。root 审完真实 catalog 后才冻结该材料与 step2 schema；wrapper 不自行选择敏感层或自动采用候选。

## step1 固定执行条件

复用 v49 的冻结 pair/capture/quant contracts、同一实板独立 CP311 工具链（Torch2.2.0、NumPy1.26.4、ONNX1.16.1、ORT1.16.0、RKNN2.3.2）。八份原 calibration `[2,437,1024]` FP32 输入是唯一 dataset；两份 evaluation 及原参考/位置不进入 dataset。mean/std None、RGB2BGR False、不转置、不加 batch。配置仍为 `rk3588 / float16 / optimization_level3 / w8a8 / normal / channel`；本阶段不是普通 build 或自动 hybrid 搜索。

实际调用只有 `config→load_onnx→hybrid_quantization_step1→release`。新输出目录位于 board experiments，新 cwd/TMPDIR 在 board `/dev/shm`，完整保留失败输出。入口与调用函数都确认板端身份；SDK lazy import、真实模块路径和 wrapper SHA 必须在同一隔离 prefix。收据记录源/graph/capture/dataset/版本、API 参数、操作返回值/耗时、生成文件大小/SHA、release、源输入前后哈希以及进程环境/cwd恢复。保存三份生成文件时再次核对字节身份。不会创建 SDK runtime 或调用 inference/accuracy_analysis。

## 已知限制

v49 的实际 compiler 把原 fc1 融合为 `ConvExGelu`，把 fc2 与 ls2/残差融合为 `ConvAdd`。因此只凭 MLP 字符串匹配不能证明 fp16 的残差或 exact GELU 留在整数区域之外。SDK 重写的名字与浮点/整数转换要逐项检查；精度配置和最终 compiler/runtime operator 表必须各自审查。

外部 IO 请求 float16 不证明内部 attention/norm 全部浮点。后续 SDK descriptor、实际 operator dtype/target、原 Torch FP32 误差和边界转换分别核验。v49 的纯 FP16 已未通过固定元素门槛，hybrid 不能预先宣称严格等价、完整模型精度、轨迹精度或实时收益。两份同序列 heldout 也不能证明泛化。

`enable_flash_attention` 在本版本 wrapper 是公开 config 参数，默认 False；[官方 v2.3.2 CHANGELOG](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/CHANGELOG.md) 的 v2.1.0 条目明确仅 RK3562/RK3576 支持 Flash Attention，后续至 v2.3.2 未声明扩展到 RK3588。因此本板启用该标志只能作为未确认支持的独立诊断，不能因参数存在而声明支持；hybrid step1 保持默认 False。
