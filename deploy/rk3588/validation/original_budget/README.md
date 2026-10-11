# 原 VGGT 的计算预算与实时目标

本目录只核算 VGGT-SLAM2 原始 VGGT。脚本仅用 Python 标准库读取固定源码、AST 和保存的 JSON 元数据；没有模型、ORT、RKNN、编译器或板端执行。当前结果为 `compute-budget-r2.json`；`r1` 是保留的初稿。源码的 9 个 SHA 全部匹配原始板端 capture，配置由构造函数和实际选中的内部 DINO 工厂函数读取。保存的完整 forward 还记录了 DINO/frame/global 各 24 次、Camera trunk 16 次调用。每个已捕获 block 的四个矩阵尺寸及 Camera 的 69 个参数元数据均与算式吻合。

以一个乘加计 2 FLOP。每帧 token 数为 `N = (H/14)*(W/14)+5`，embedding `D=1024`。每个 block 的线性层为 `12*S*N*D² MAC`；DINO 和 frame 的注意力乘法为 `2*S*N²*D MAC`，global 为 `2*(S*N)²*D MAC`。保留输出层 4/11/17/23 只减少保存的中间输出，全部 72 个 transformer block 仍执行。

| 实际输入 H×W，S=2 | 完整 forward TFLOP | 复用一个旧帧 DINO 后 TFLOP | 20 个新观测位姿/秒的平均有效 TFLOP/s |
|---|---:|---:|---:|
| 84×112 | 0.219 | 0.186 | 3.728 |
| 252×336 | 1.926 | 1.643 | 32.864 |
| 392×518 | 5.082 | 4.346 | 86.911 |

这些数值是常规矩阵和稠密卷积 MAC 预算，排除 normalization、softmax、GELU、RoPE、插值、内存和布局、队列、传输、SL4 图与地图关联。它们没有使用硬件峰值，也不能证明硬件不可能达到目标。输入几何来自实际原模型运行记录；较小输入的精度损失仍须完整 ATE/RPE 和失败率验证。

S=2 的 overlap pair 每次只有 **一个新观测**。要求 20 个新位姿/秒，串行平均服务时间须不超过 50 ms；处理两张图不能把旧帧再计作一个新位姿。P95≤100 ms 还必须包括排队、预处理和数据传输，不能由平均频率推得。本目录的 S=3 行仅是明确的数学场景：一次含两个新观测，在恰好 20 Hz、均匀到达的条件下，较早的新帧已等待 50 ms 才能凑齐。吞吐的摊销收益不能当作延迟达标证据。

252×336/S=2 中，复用一个旧帧 DINO 仅节约完整常规 MAC 的 **14.70%**。它适用于相同归一化像素及不变的原模型状态。原 global attention 双向依赖所有帧的 keys/values 和 softmax 分母；加入新帧会改变旧帧输出及后续 frame/global 输入。这里没有复用 global 上下文，也没有省略层或 token 计算。

按原 backbone 和 Camera 每个新观测执行、原 dense head 改为每 2–5 秒一次的**条件场景**，该尺寸仍对应平均 **29.139–29.196 TFLOP/s**。这只是调度预算：当前尚未实现这种调度，没有验证异步输出的 SL4 对齐或质量，不能计作位姿或地图达标。即使仅数复用旧 DINO 后的 backbone 线性矩阵工作，也对应 26.394 TFLOP/s；只优化注意力不会消除主要线性计算。实际耗时还受布局、内存和后端调度影响，MAC 占比不等同于实测时间占比。

当前单个完整 frame0→global0 pair 的算术预算为 **48.683 GFLOP**。若两个 graph 的中间特征不再返回主机，FP32 特征跨边界的逻辑 payload 可从 14,319,616 B 减为 7,159,808 B；不含 positions 或内部布局副本。这是潜在 payload 差值，节省的耗时须实际连续执行测量。历史 standalone frame/global 输入分别含 CPU 参考特征，不能直接相加称为已测 NPU 连续链。

原模型当前活跃参数共 1,157,941,492 个。全数以 FP16 表示的逻辑唯一参数 payload 是 **2,315,882,984 B（2.157 GiB）**，这不证明完整 FP16 执行、精度或 8 GB 部署。另一个明确的条件假设是每次矩阵调用把权重流式读取一次，且 Camera 四次迭代之间没有权重缓存；此时 backbone+Camera 的矩阵权重逻辑读量为 **3.540 GB/forward、20 个 S=2 更新/秒对应 70.804 GB/s**。这是条件数学量，既不是实测 DRAM 带宽，也不是硬件 IO 下界。实际缓存、权重打包、持久 context 和重复读会改变流量；SDK 分配、参数 payload、主机上传也不同。现有完整 Camera SDK 的 DMA 查询值为 3.424 GB，只描述该子网络 context，不能代表全模型或全板峰值内存。

实测边界仍需保留：252×336/S=2 原 CPU 的一次无 hook 完整 forward 为 **34.069 s**；独立加 hook forward 为 50.833 s，两组不能相加。历史 frame0/global0 的短期正常 SDK IO 均值分别为 183.626/187.139 ms，完整 Camera 为 203.646 ms；都是保存输入的组件执行，严格 FP32 数值门均失败。它们说明这些现有组件实现没有达到目标，不是对所有后续实现的硬件极限证明，也没有验证新观测 20 Hz、P95、30 分钟、8 GB 或轨迹质量。

复现要求使用 SHA 匹配的原始 vggt 包目录和当前公开代码目录。所有输出须使用新路径：

```bash
python3 -B calculate_original_vggt_budget.py \
  --source-root /path/to/original/vggt \
  --public-root /path/to/Edge-VGGT-SLAM \
  --output /path/to/new-budget.json
python3 -B check_budget_formulas.py \
  --budget /path/to/new-budget.json \
  --profile /path/to/Edge-VGGT-SLAM/results/profile_336_v17.json \
  --camera-capture /path/to/Edge-VGGT-SLAM/deploy/rk3588/validation/vggt-v15-camera-capture112.json \
  --output /path/to/new-checks.json
```

`source-freeze-r1.json` 固定两个脚本、结论、当前报告、selfchecks 及所有原源码/输入 JSON 的 SHA。公开时仅包含脚本和数字元数据，原权重、模型数组与权重导出不属于本目录交付物。
