# RK3588S 离线依赖

目标为 Ubuntu 22.04 / aarch64、独立 CPython 3.11.17、glibc 2.35。`requirements-board.txt` 固定历史验证过的 126 个环境包；PyTorch 2.3.1 / torchvision 0.18.1 使用 ARM64 CPU wheel，Viser 固定 0.2.23，GTSAM develop 包含 SL4。Open3D 0.20.0 的 wheel 要求 glibc 2.35。环境包清单不含模型权重；当前部署只允许原 VGGT，部分历史环境工具并非当前运行必需。

在仓库根目录执行，所有下载和临时文件均写入指定的、有足够空间的目录。
以下示例重新准备并验证安装包：

```bash
python3 deploy/rk3588/dependencies/artifacts.py download \
  --output "/path/to/package-cache"
python3 deploy/rk3588/dependencies/artifacts.py verify \
  --output "/path/to/package-cache" --resolve-dependencies
```

下载入口只使用 `artifacts.json` 中固定的官方地址和 SHA256，不重新解析最新版本。已验证的文件直接复用，2 MiB 下载片段保存在输出目录的 `tmp/` 中以便重试。可用 `--kind python`、`--kind wheels` 或 `--kind debs` 分批准备。DNS 临时不可用时可追加 `--resolve HOST:443:IP`，IP 应先通过当前官方域名解析确认；TLS 始终校验原域名，不修改系统 DNS。

`verify` 校验全部文件的大小、SHA256 和归档完整性；`--resolve-dependencies` 还执行 pip 的 ARM64 / Python 3.11 断网解析，屏蔽网络调用且不安装任何包。主机需要 Python 3.10+、curl；解析选项另外需要 pip 23+。运行结果 `offline-install-plan.json` 保存在输出目录中。

输出的 `wheelhouse/` 含 126 个 wheel、固定 requirements 和 `SHA256SUMS`，以及 Python 运行时压缩包；`debs/` 含 4 个 Ubuntu 库包和 `SHA256SUMS`。板端安装入口为上一级的 `install-offline.sh`，Python 运行时须先展开到部署根的 `python/`。这 4 个 deb 用于项目内的 `sysroot/`；其中 gcc-12-base 是 libgfortran5 的包依赖，实际运行仍复用板端已有的 libc、libgcc、libstdc++ 和图形库。

`artifacts.json` 记录所有官方 URL、SHA256、Python GitHub 发布来源，以及 4 个 deb 的 Ubuntu 签名索引校验记录。唯一没有上游 wheel 的 `antlr4-python3-runtime==4.9.3` 从官方 sdist 重打包；`antlr4-wheel-template.json` 固定 ZIP 和包元数据，代码文件逐个对照 SHA256，重建后的整包 SHA256 也必须与已部署 wheel 相同。此过程仅使用 Python 标准库，不执行上游 setup.py，也不改变主机环境。

2026-10-10 已验证：126 个 wheel 共 405,250,287 字节，全部 SHA256 / ZIP CRC 通过；pip 在网络调用被禁用时解析出 126 个本地文件。Python 运行时与 4 个 deb 另行通过官方哈希校验，Ubuntu 两份索引的 InRelease 签名使用系统 Ubuntu archive keyring 验证通过。
