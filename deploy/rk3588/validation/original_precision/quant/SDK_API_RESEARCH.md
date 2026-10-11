# RKNN 2.3.2 原 VGGT 配对图 PTQ API

本轮配置固定为 `quantized_dtype=w8a8`、`quantized_algorithm=normal`、`quantized_method=channel`。既有 2.3.2 SDK 包的纯 Python `rknn/api/rknn.py` 已静态核对这些公开参数及默认值，文件 SHA 是 `63e4506fb851214e79f933f9de0d27dab04ecf93f6f46959c07c5787433fd05e`；没有导入或运行该 SDK。对应 ARM64 CPython 3.11 包由 Rockchip 在 [v2.3.2 官方 wheel](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/rknn-toolkit2/packages/arm64/rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl) 提供。实际编译收据还会记录板端 SDK Python 源文件 SHA；本静态核查不声称完成了 ARM API 调用。

官方 ARM64 CPython 3.11 要求包含 Torch `>=1.13.1,<=2.2.0`、NumPy `<=1.26.4`、ONNX `1.16.1`、onnxoptimizer `0.3.8`、ORT `>=1.16.0` 及 protobuf `3.20.3`。因此校准采用独立工具链，固定 Torch 2.2.0、NumPy 1.26.4、ONNX 1.16.1、ORT 1.16.0；原实际捕获继续使用生产环境 Torch 2.3.1。要求出处为 [官方 ARM64 requirements](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknn-toolkit2/packages/arm64/arm64_requirements_cp311.txt)。

官方函数示例通过 `build(do_quantization=True,dataset=...)` 启用数据校准；其 dataset 文件把同一组输入文件写在同一行，其中包括 NPY。该例的 NPY 在调用时按浮点数组加载。[官方 build 示例](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/rknn-toolkit2/examples/functions/multi_input/test.py)、[官方 dataset 文件](https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/rknn-toolkit2/examples/functions/multi_input/dataset.txt)。

据此，当前单输入配对图使用每行一个原 `frame0` 输入 NPY，一共八行。原 tensor 是 `[2,437,1024]` FP32，首轴是原两帧排列；本工程保持原序，不添加维度、转置或输入归一化。位置 tensor 已烘焙进固定配对图，只用于来源校验。这里的 rank3 NPY 处理是工程约束；上述官方示例没有单独证明该 rank3 输入在 RKNN 校准内部的解释。实际板端 build、导出 IO 及 heldout 比较仍需后续检查。

实际 SDK 操作限定为 `config → load_onnx → build → export_rknn`，其中 build 显式 `do_quantization=True,auto_hybrid=False`。没有 init_runtime、inference 或 accuracy_analysis 调用。PTQ build 本身会执行校准计算，因此整个编译脚本受到 Linux ARM64 RK3588 device-tree、独立 CPython 3.11 prefix 及版本门槛保护，所有实际工作由主代理在板端执行。

八个 calibration pair 和两个 evaluation pair 在图像 SHA、输入 raw SHA 及 sample_id 上分离，也排除固定旧配对图的两张评估图。两份 evaluation 使用原 Torch frame0/global0 最终参考，完全不进入 dataset。八组样本只覆盖同一真实短序列；编译成功不证明泛化、全部算子 INT8、精度、完整 VGGT/SLAM 或实时目标。
