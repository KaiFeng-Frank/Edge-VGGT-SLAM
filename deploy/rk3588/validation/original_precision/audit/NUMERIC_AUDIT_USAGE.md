# 保存输出审查的复现

审查只读私有保存数据，不加载真实模型、ORT session、SDK、编译器或解码器。目录可以通过参数传入；不要把私有数组或导出权重上传到公开仓库。

`audit_saved_pair_profile_numeric.py` 是独立脚本，只需 Python 标准库和 NumPy。它逐份读取两组运行共 32 份 SDK 输出和 4 份原输入/参考，并核验 16 个 profile 文本 SHA、实际 AB/BA 调度、两组首调用/各 3 次后续分母、完整 SDK IO 时间和固定原 FP32 数值条件。normal/profile 对应的 16 份输出逐 bit 一致；profile 的 IO 均值独立记录，未混入正常运行差值。内部 operator 表的解析属于另一份审查。

```bash
python3 audit_saved_pair_profile_numeric.py \
  --normal-dir /path/to/private/normal-r1 \
  --normal-sha256 05ac1187d9bda3893d43f2d9c834d3b4de51f97d8c455496daa0c5a6f803e591 \
  --profile-dir /path/to/private/profile-r1 \
  --profile-sha256 98b550f62eeadba951834ac71bff83d320faf1dcccd179c2320cdfa2fc4b8623 \
  --capture-dir /path/to/private/capture-r1 \
  --runtime-source-dir ../runtime \
  --output /path/to/new-profile-numeric-audit.json
```

`audit_saved_pair_runtime.py` 可单独核 normal 或 profile 的 16 份输出。它依赖 sibling `../capture/v49_capture_contracts.py` 和 `../capture/audit_saved_capture.py` 的冻结 SHA；发布到 `original_precision/audit` 时应保留这个相对布局。所有实际数据、receipt SHA 和运行源码目录都使用显式参数。

```bash
python3 audit_saved_pair_runtime.py \
  --runtime-dir /path/to/private/normal-r1 \
  --run-sha256 05ac1187d9bda3893d43f2d9c834d3b4de51f97d8c455496daa0c5a6f803e591 \
  --runtime-source-dir ../runtime \
  --capture-dir /path/to/private/capture-r1 \
  --capture-sha256 bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc \
  --output /path/to/new-normal-numeric-audit.json
```

这两份脚本为 v49 的冻结证据审查，原 checkpoint、pair graph、源身份、heldout 样本和 shape 都明确绑定。它们不能直接当成任意未来输入/模型版本的通用验收器。跨机器运行时需有完整私有目录和相同冻结源码；浮点误差统计的复核允许极小汇总舍入差异，但逐元素原 FP32 门槛保持 `rtol=1e-4,atol=1e-5`，不放宽。输出必须是保存数据目录外的新文件。

公开摘要 `public-normal-audit-r1.json` 和 `public-profile-numeric-audit-r1.json` 只记录身份、指标和边界；所有 SDK 实现仍未通过原 FP32 门槛且未采用。新 heldout CPU ORT 的 overall 也保留单元素拒绝事实，不改原 Torch 参考。capture/normal/profile 的完整私有收据不在公开树中包含数组或权重。
