# M48 完整 Update 分组研究入口

已在板端原流程中完成三个完整图的输入/gold 捕获，242 项保存数组检查通过；[实际收据](../../../../results/dpvo_grouped_update_capture_v36.json)保留文件身份与原调用来源。这组文件是独立研究候选，尚未运行真实分组模型、ORT 或 SDK。默认 VGGT 和所有 v35 冻结文件均未修改。相邻 `LICENSE` 保留 Princeton-VL/DPVO 的 MIT 许可与版权。逐源完整组的原始语义来自 DPVO commit `0ac95b656d1fda91c271d2a106460d19ad966fc7`。

`source_frame_batches.py` 将真实完整图按源帧的目标数 D 分桶；同 D 的多个源帧作为独立 batch 项，布局 `[N,384,D,48]`。每个源帧保留自己的所有目标和 48 个 patch，不补零、不截断、不跨源归约。c2 使用 c1 已更新的状态，缺邻居输入在 Linear 前归零并保留 bias；两个 SoftAgg 分别沿 D 和 M 归约。冻结 CF wrapper 的 50 个原参数、LayerNorm 和 GRU 内部 Sigmoid 保持绑定；末端采用已披露的 stable Sigmoid 表达。

现有 26 项验证仅使用小型随机 toy 模块和 fake callback。它们验证数学依赖、完整组 scatter、原边序、失败不提交、原输入不变和捕获一次调用，不证明真实完整图 FP32 或 NPU 验收通过。原 `rtol=1e-4, atol=1e-5` 不变。

## 板端真实输入捕获

Root 运行；此 launcher 不能在主机执行真实捕获。先将 `capture-v36-fast-update-board.py` 复制至板端独立实验目录，核对 SHA：

```
51b9d830762c1c1f701ca38a13378b874240a04f3d7e98181fa397ae1aa8a5d2
```

命令只在现有成功的 v35 FAST 参数前增加捕获选项。`--` 后的全部参数直接交给原脚本，包括 `--diagnostic-allow-fp32-rejected`。

```bash
python -B capture-v36-fast-update-board.py \
  --parent-runner <原板端run-v35-dpvo-fast-rknn-board.py> \
  --parent-runner-sha256 84b8f02e14c0695c79d5c47319ff47ec4c7a8dcf53a3b38d905e03bc1efde6a8 \
  --capture-output-dir <experiments下全新capture目录> \
  --capture-observations 13 127 191 \
  --capture-max-total-bytes 536870912 \
  -- <原192帧v35完整参数；--output-dir使用另一个全新同级目录>
```

捕获每个预声明 observation 的第一个完整、非 motion-probe、在线 Update，发生在该轮 BA 前。原 callback 只执行一次，其返回对象原样交回原流程；这一次调用同时提供 gold。额外 copy/hash/save 不消费随机数、不更新状态，也不重新调用编码器或 Update。EOF 的优化不用于补齐 online case。若某个指定 observation 没有完整 Update，收据明确失败并保留已有 artifacts。

每个 case 保存实际 `net/inp/corr`、原 `ii/jj/kk`、poses、patches、intrinsics、frame contexts、保留 observation IDs 和原始三个输出。`inp == contexts[kk]`、图归属、文件与原数组 SHA、原 parent input manifest、callback 计数、权重及 sources-after、cleanup 均绑定。`capture.json` 内的 graph/geometry/features revision 是捕获器基于实际数组内容生成的 serial + SHA；原 core 没有这些公开计数，不能称作完整地图 revision。recurrent revision 是实际 callback index。

所有完整调用都增加了指纹开销，选中调用另增拷贝和存盘。收据单列这些成本；parent replay 的帧耗时也因此包含捕获工作，不是 ordinary 性能基准。

## Root 的分组执行接口

1. 用保存的真实 `ii/jj/kk` 调用 `build_degree_plan(..., patches_per_frame=48)`。可设置 `max_sources_per_batch` 仅拆 N 维；尾批使用真实较小 N，仍不补零。
2. 用原数组建立不可变快照，给 `execute_grouped_update` 传入显式 `SnapshotIdentity` 和 owner 当前 identity reader。异步使用前还须由 owner 保证参数快照与 revision 对应。
3. callback `execute(batch, feeds)` 接收 CF FP32 `{net:[N,384,D,48], inp:[N,384,D,48], corr:[N,882,D,48]}`，返回 owned FP32 `{net:[N,384,D,48], delta:[N,2,D,48], weight:[N,2,D,48]}`。
4. 所有批成功且 identity 未变后，adapter 才返回按原边序还原的完整 `[1,E,C]` 结果。形状/有限性/[0,1] 只是 transport 门禁；实际完整 Update 原 FP32 门禁必须另行比较，不能直接采用结果。

`source_frame_update_channel_first.py` 与 `source_frame_plan.py` 为原冻结字节。初始 8 个 D8 源帧可形成 `[8,384,8,48]`；历史末次 pre-Update 的 17 个源帧形成 10 个 D 桶。完整静态 Linear MAC 与数组预算见 `full-M48-update-contract.json`，由已公开拓扑 metadata 推算，未加载真实 net/corr 或执行模型。约 8.36/27.49 GMAC 均未因分组减少。

**B > 1 必须建立独立的新 SDK 合同。** v35 B1 helper 不能直接复用为新形状。每个实际 `(N,D,M)` 需独立编译/查询，核实真实 native format/strides、输入 pack、输出 owned copy-before-release、完整源组 scatter、context 生命周期和实际 profile。未支持的形状应明确拒绝；不得补零、删边或无声明回退。50 个参数的静态 contexts 可能重复占内存，当前无 B > 1 编译/运行证据。

下一步按真实捕获的每个完整图测量全部桶合计的原/候选数值与总时长，计入 pack/scatter/SDK I/O，并保留每桶和全图 gate。v35 的 D10 M96 单块时长不代表完整 M48 图，更不代表 20 Hz。
