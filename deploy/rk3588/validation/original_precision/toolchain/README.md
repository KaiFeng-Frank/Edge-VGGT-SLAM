# 原 VGGT 的隔离板端量化工具链

本轮在 RK3588S 新建 `experiments/vggt-v49-toolchain-venv-r1`，保留生产环境的 Torch 2.3.1。实际安装、`pip check` 和 9 个模块的导入检查均通过；安装与导入阶段没有加载模型、创建 ORT session 或执行校准。

| 依赖 | 独立环境版本 |
| --- | --- |
| Python | 3.11.17 / aarch64 |
| RKNN Toolkit2 | 2.3.2 |
| Torch / NumPy | 2.2.0 / 1.26.4 |
| ONNX / ONNX Runtime | 1.16.1 / 1.16.0 |
| ONNX Optimizer / protobuf | 0.3.8 / 3.20.3 |

[官方 ARM64 依赖要求](https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknn-toolkit2/packages/arm64/arm64_requirements_cp311.txt) 与生产环境不同。原始模型参考仍由生产环境产生；此独立环境用于原模型的量化编译及保留输入 CPU ORT 复核。进程清除 `PYTHONPATH`、`PYTHONHOME`、`VIRTUAL_ENV`，关闭用户 site；导入检查确认实际模块来自新 prefix。共享库路径只在子进程设置，未改动系统配置。

安装复用了板上已有的纯依赖 wheel；逐个核验大小和 SHA256，并额外核验官方 RKNN wheel 的 Git blob SHA1。ONNX Optimizer 使用此前在本板由 0.3.8 源码构建的 wheel，不使用另一模型的代码或产物。它的源码与构建 wheel 身份分别记录，不能把两者的 SHA 混用。

- [wheel 清单](wheel-plan.json) 保存本次板端输入路径、版本及 SHA；这些是实验机路径，复现者须自行准备合法依赖并调整路径。
- [依赖来源](dependency-origins.json) 保存官方包与源代码下载地址，不分发二进制。
- [安装器](install-board.py) 在检查 RK3588 身份、全新目录和所有 wheel 后离线安装。[安装收据](install.json) 保留实际命令、返回值和环境版本。
- [导入检查](import-board.py) 只检查导入与路径；[实际结果](imports.json) 不代表模型、校准或 NPU 执行通过。

当前清单和安装器是本次实验的固定输入，包含历史依赖存储路径。不得直接把它们当作任意设备的一键安装器；也不得在主机上运行原模型校准。模型权重、捕获数组、含权重导出和 SDK 二进制均不在公开仓库内。
